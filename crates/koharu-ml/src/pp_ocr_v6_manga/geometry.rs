//! Deterministic helpers for the PP-OCRv6 manga pipeline.
//!
//! Everything here is arithmetic on pixel buffers and strings, so it can be
//! unit-tested without ONNX Runtime. The conventions follow PaddleOCR:
//! `cv2.resize(INTER_LINEAR)`, `get_rotate_crop_image` and `CTCLabelDecode`.

use std::cmp::Ordering;

/// Long side of the detector input for a normal page, from the model card.
const DET_LONG_SIDE: f64 = 960.0;
/// Pages more elongated than this are sized by pixel budget instead of long side.
const STRIP_ASPECT: f64 = 2.0;
/// Pixel budget of a 3:4 page whose long side is `DET_LONG_SIDE`.
const STRIP_PIXEL_BUDGET: f64 = DET_LONG_SIDE * DET_LONG_SIDE * 0.75;
/// Smallest detector input side; tiny crops are lifted to keep some resolution.
const DET_MIN_SIDE: f64 = 96.0;
const DET_MAX_SIDE: f64 = 4000.0;
const MAX_PAGE_SCALE: f64 = 4.0;

pub(super) const REC_HEIGHT: usize = 48;
pub(super) const REC_MIN_WIDTH: usize = 320;
const REC_MAX_WIDTH: usize = 3200;

/// Lines taller than this multiple of their width are rotated before reading.
const VERTICAL_ASPECT: f32 = 1.5;

/// Scale the model card applies to a whole page before detection.
///
/// Koharu hands the recognizer one text-block crop at a time, but the detector
/// was trained on whole pages resized by this rule. Applying the *page* scale to
/// each crop keeps glyphs at the pixel size the detector expects.
pub fn page_scale(width: u32, height: u32) -> f64 {
    let (width, height) = (f64::from(width.max(1)), f64::from(height.max(1)));
    let long = width.max(height);
    let short = width.min(height);
    let scale = if long / short > STRIP_ASPECT {
        (STRIP_PIXEL_BUDGET / (width * height)).sqrt()
    } else {
        DET_LONG_SIDE / long
    };
    scale.clamp(0.05, MAX_PAGE_SCALE)
}

/// Detector input `(width, height)` for an image viewed at `scale`; both sides
/// are multiples of 32 as the network requires.
pub(super) fn detector_size(width: u32, height: u32, scale: f64) -> (u32, u32) {
    let mut scale = if scale.is_finite() && scale > 0.0 {
        scale
    } else {
        1.0
    };
    let (width, height) = (f64::from(width.max(1)), f64::from(height.max(1)));
    let short = (width * scale).min(height * scale);
    if short < DET_MIN_SIDE {
        scale *= DET_MIN_SIDE / short;
    }
    let long = (width * scale).max(height * scale);
    if long > DET_MAX_SIDE {
        scale *= DET_MAX_SIDE / long;
    }
    (round32(width * scale), round32(height * scale))
}

fn round32(value: f64) -> u32 {
    (((value / 32.0).round() as u32).max(1)) * 32
}

/// Source taps and blend weight for one output coordinate along one axis.
struct Axis {
    low: Vec<usize>,
    high: Vec<usize>,
    weight: Vec<f32>,
}

/// Half-pixel-center bilinear taps, clamped at the borders (`cv2.INTER_LINEAR`).
fn axis(source: usize, target: usize) -> Axis {
    let ratio = source as f32 / target as f32;
    let last = source as isize - 1;
    let mut taps = Axis {
        low: Vec::with_capacity(target),
        high: Vec::with_capacity(target),
        weight: Vec::with_capacity(target),
    };
    for index in 0..target {
        let position = (index as f32 + 0.5) * ratio - 0.5;
        let base = position.floor();
        taps.low.push((base as isize).clamp(0, last) as usize);
        taps.high.push((base as isize + 1).clamp(0, last) as usize);
        taps.weight.push(position - base);
    }
    taps
}

/// Resizes interleaved RGB8 to `[3, new_height, new_width]` floats in BGR order.
///
/// PaddleOCR feeds OpenCV's BGR arrays through the ImageNet mean and standard
/// deviation listed in RGB order, so `mean` and `std` are indexed by BGR slot.
pub(super) fn resize_normalize(
    rgb: &[u8],
    width: usize,
    height: usize,
    new_width: usize,
    new_height: usize,
    mean: [f32; 3],
    std: [f32; 3],
) -> Vec<f32> {
    debug_assert_eq!(rgb.len(), width * height * 3);
    let columns = axis(width, new_width);
    let rows = axis(height, new_height);
    let plane = new_width * new_height;
    let mut output = vec![0.0f32; plane * 3];
    for y in 0..new_height {
        let (top, bottom, weight_y) = (rows.low[y], rows.high[y], rows.weight[y]);
        for x in 0..new_width {
            let (left, right, weight_x) = (columns.low[x], columns.high[x], columns.weight[x]);
            for (slot, channel) in [2usize, 1, 0].into_iter().enumerate() {
                let at = |row: usize, column: usize| {
                    f32::from(rgb[(row * width + column) * 3 + channel])
                };
                let upper = at(top, left) * (1.0 - weight_x) + at(top, right) * weight_x;
                let lower = at(bottom, left) * (1.0 - weight_x) + at(bottom, right) * weight_x;
                let value = upper * (1.0 - weight_y) + lower * weight_y;
                output[slot * plane + y * new_width + x] = (value / 255.0 - mean[slot]) / std[slot];
            }
        }
    }
    output
}

fn distance(a: [f32; 2], b: [f32; 2]) -> f32 {
    (a[0] - b[0]).hypot(a[1] - b[1])
}

/// Upright size of the line bounded by `quad` (top-left, top-right,
/// bottom-right, bottom-left).
pub(super) fn quad_size(quad: &[[f32; 2]; 4]) -> (usize, usize) {
    let width = distance(quad[0], quad[1]).max(distance(quad[3], quad[2]));
    let height = distance(quad[0], quad[3]).max(distance(quad[1], quad[2]));
    (
        (width.round() as usize).max(1),
        (height.round() as usize).max(1),
    )
}

fn sample(rgb: &[u8], width: usize, height: usize, x: f32, y: f32) -> [f32; 3] {
    let x = x.clamp(0.0, (width - 1) as f32);
    let y = y.clamp(0.0, (height - 1) as f32);
    let (x0, y0) = (x.floor() as usize, y.floor() as usize);
    let (x1, y1) = ((x0 + 1).min(width - 1), (y0 + 1).min(height - 1));
    let (tx, ty) = (x - x0 as f32, y - y0 as f32);
    let mut pixel = [0.0f32; 3];
    for (channel, value) in pixel.iter_mut().enumerate() {
        let at = |row: usize, column: usize| f32::from(rgb[(row * width + column) * 3 + channel]);
        let upper = at(y0, x0) * (1.0 - tx) + at(y0, x1) * tx;
        let lower = at(y1, x0) * (1.0 - tx) + at(y1, x1) * tx;
        *value = upper * (1.0 - ty) + lower * ty;
    }
    pixel
}

/// Samples the upright line bounded by `quad` out of an interleaved RGB8 image.
///
/// This stands in for `cv2.warpPerspective`: the detector's boxes are rotated
/// rectangles, for which bilinear interpolation of the four corners is the same
/// mapping. Pixels outside the image replicate the border.
pub(super) fn crop_quad(
    rgb: &[u8],
    width: usize,
    height: usize,
    quad: &[[f32; 2]; 4],
) -> (Vec<u8>, usize, usize) {
    let (out_width, out_height) = quad_size(quad);
    let mut output = vec![0u8; out_width * out_height * 3];
    for y in 0..out_height {
        let v = (y as f32 + 0.5) / out_height as f32;
        for x in 0..out_width {
            let u = (x as f32 + 0.5) / out_width as f32;
            let point = |axis: usize| {
                (1.0 - u) * (1.0 - v) * quad[0][axis]
                    + u * (1.0 - v) * quad[1][axis]
                    + u * v * quad[2][axis]
                    + (1.0 - u) * v * quad[3][axis]
            };
            let pixel = sample(rgb, width, height, point(0) - 0.5, point(1) - 0.5);
            let at = (y * out_width + x) * 3;
            for (channel, value) in pixel.iter().enumerate() {
                output[at + channel] = value.round().clamp(0.0, 255.0) as u8;
            }
        }
    }
    (output, out_width, out_height)
}

/// Direction in which vertical lines are turned upright for the recognizer.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum Rotation {
    Ccw,
    Cw,
}

/// Whether a line of this upright size is read as vertical text.
pub(super) fn is_vertical(width: usize, height: usize) -> bool {
    height as f32 / width.max(1) as f32 >= VERTICAL_ASPECT
}

/// Rotates an interleaved RGB8 image by 90 degrees (`np.rot90` for `Ccw`).
pub(super) fn rotate90(
    rgb: &[u8],
    width: usize,
    height: usize,
    rotation: Rotation,
) -> (Vec<u8>, usize, usize) {
    let (out_width, out_height) = (height, width);
    let mut output = vec![0u8; rgb.len()];
    for y in 0..out_height {
        for x in 0..out_width {
            let (source_x, source_y) = match rotation {
                Rotation::Ccw => (width - 1 - y, x),
                Rotation::Cw => (y, height - 1 - x),
            };
            let from = (source_y * width + source_x) * 3;
            let to = (y * out_width + x) * 3;
            output[to..to + 3].copy_from_slice(&rgb[from..from + 3]);
        }
    }
    (output, out_width, out_height)
}

/// Recognizer input `[3, REC_HEIGHT, padded]` and its padded width.
///
/// The line is resized to a fixed height, keeping its aspect ratio, and padded on
/// the right with zeros (after normalization) up to `REC_MIN_WIDTH`.
pub(super) fn rec_input(
    rgb: &[u8],
    width: usize,
    height: usize,
    mean: [f32; 3],
    std: [f32; 3],
) -> (Vec<f32>, usize) {
    let target = ((REC_HEIGHT as f64 * width as f64 / height.max(1) as f64).ceil() as usize)
        .clamp(1, REC_MAX_WIDTH);
    let planes = resize_normalize(rgb, width, height, target, REC_HEIGHT, mean, std);
    let padded = target.max(REC_MIN_WIDTH);
    if padded == target {
        return (planes, padded);
    }
    let mut output = vec![0.0f32; 3 * REC_HEIGHT * padded];
    for row in 0..3 * REC_HEIGHT {
        output[row * padded..row * padded + target]
            .copy_from_slice(&planes[row * target..(row + 1) * target]);
    }
    (output, padded)
}

/// Reads one symbol per line; the CTC blank is added by [`ctc_decode`].
pub(super) fn parse_dictionary(text: &str) -> Vec<String> {
    let mut symbols = text
        .split('\n')
        .map(|line| line.trim_end_matches('\r').to_owned())
        .collect::<Vec<_>>();
    if symbols.last().is_some_and(String::is_empty) {
        symbols.pop();
    }
    symbols
}

/// Greedy CTC decoding of `[steps, classes]` scores.
///
/// Class 0 is the blank, classes `1..=dictionary.len()` are the dictionary and an
/// optional final class is the space PaddleOCR appends (`use_space_char`).
pub(super) fn ctc_decode(
    scores: &[f32],
    steps: usize,
    classes: usize,
    dictionary: &[String],
) -> Result<String, String> {
    if scores.len() != steps * classes {
        return Err(format!(
            "recognizer returned {} values for {steps} steps of {classes} classes",
            scores.len()
        ));
    }
    let with_space = classes == dictionary.len() + 2;
    if classes != dictionary.len() + 1 && !with_space {
        return Err(format!(
            "dictionary has {} symbols but the recognizer has {classes} classes",
            dictionary.len()
        ));
    }
    let mut text = String::new();
    let mut previous = 0usize;
    for step in 0..steps {
        let row = &scores[step * classes..(step + 1) * classes];
        let mut best = 0usize;
        for (index, value) in row.iter().enumerate() {
            if *value > row[best] {
                best = index;
            }
        }
        if best != 0 && best != previous {
            if best <= dictionary.len() {
                text.push_str(&dictionary[best - 1]);
            } else if with_space {
                text.push(' ');
            }
        }
        previous = best;
    }
    Ok(text)
}

/// One recognized line with the geometry needed to order it.
#[derive(Debug, Clone)]
pub(super) struct Line {
    pub center: [f32; 2],
    /// Short side of the line's box: glyph size across the reading direction.
    pub thickness: f32,
    pub vertical: bool,
    pub text: String,
}

fn primary(line: &Line, vertical: bool) -> f32 {
    if vertical {
        -line.center[0]
    } else {
        line.center[1]
    }
}

fn secondary(line: &Line, vertical: bool) -> f32 {
    if vertical {
        line.center[1]
    } else {
        line.center[0]
    }
}

/// Orders lines in reading order: right-to-left columns for vertical text,
/// top-to-bottom rows otherwise. Lines whose cross-axis positions are within
/// 0.6 of the median thickness share a row or column and are read along it.
pub(super) fn order_lines(mut lines: Vec<Line>) -> Vec<Line> {
    if lines.len() < 2 {
        return lines;
    }
    let vertical = lines.iter().filter(|line| line.vertical).count() * 2 > lines.len();
    let mut thickness = lines.iter().map(|line| line.thickness).collect::<Vec<_>>();
    thickness.sort_by(f32::total_cmp);
    let tolerance = thickness[thickness.len() / 2] * 0.6;

    lines.sort_by(|a, b| primary(a, vertical).total_cmp(&primary(b, vertical)));
    let mut ordered = Vec::with_capacity(lines.len());
    let mut group: Vec<Line> = Vec::new();
    let mut anchor = 0.0f32;
    for line in lines {
        if !group.is_empty() && primary(&line, vertical) - anchor > tolerance {
            group.sort_by(|a, b| {
                secondary(a, vertical)
                    .partial_cmp(&secondary(b, vertical))
                    .unwrap_or(Ordering::Equal)
            });
            ordered.append(&mut group);
        }
        if group.is_empty() {
            anchor = primary(&line, vertical);
        }
        group.push(line);
    }
    group.sort_by(|a, b| {
        secondary(a, vertical)
            .partial_cmp(&secondary(b, vertical))
            .unwrap_or(Ordering::Equal)
    });
    ordered.append(&mut group);
    ordered
}

fn is_cjk(character: char) -> bool {
    matches!(
        character as u32,
        0x2E80..=0x9FFF | 0xAC00..=0xD7AF | 0xF900..=0xFAFF | 0xFE30..=0xFE4F
            | 0xFF00..=0xFFEF | 0x20000..=0x2FA1F
    )
}

/// Joins ordered lines. CJK text is concatenated; a space separates lines only
/// when both sides of the break are non-CJK (words of a Latin-script balloon).
pub(super) fn join_lines(lines: &[Line]) -> String {
    let mut text = String::new();
    for line in lines {
        let next = line.text.trim();
        let (Some(first), last) = (next.chars().next(), text.chars().last()) else {
            continue;
        };
        if last.is_some_and(|last| !is_cjk(last) && !is_cjk(first)) {
            text.push(' ');
        }
        text.push_str(next);
    }
    text
}

#[cfg(test)]
mod tests {
    use super::*;

    const MEAN: [f32; 3] = [0.485, 0.456, 0.406];
    const STD: [f32; 3] = [0.229, 0.224, 0.225];

    #[test]
    fn page_scale_follows_the_model_card() {
        assert!((page_scale(1200, 1800) - 960.0 / 1800.0).abs() < 1e-9);
        // An 800x4000 strip is sized by pixel budget instead of long side.
        let strip = page_scale(800, 4000);
        assert!((strip - (691_200.0f64 / 3_200_000.0).sqrt()).abs() < 1e-9);
        assert!(strip > 960.0 / 4000.0);
    }

    #[test]
    fn detector_size_is_a_multiple_of_32_and_lifts_tiny_crops() {
        let (width, height) = detector_size(1200, 1800, page_scale(1200, 1800));
        assert_eq!((width % 32, height % 32), (0, 0));
        assert_eq!(height, 960);
        let (width, height) = detector_size(40, 20, 1.0);
        assert!(width >= 96 && height >= 96 - 32, "{width}x{height}");
        assert_eq!((width % 32, height % 32), (0, 0));
        let (width, height) = detector_size(100, 20_000, 1.0);
        assert!(width <= 4032 && height <= 4032);
    }

    #[test]
    fn resize_to_the_same_size_only_normalizes() {
        let rgb = [10u8, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120];
        let output = resize_normalize(&rgb, 2, 2, 2, 2, MEAN, STD);
        // Slot 0 is blue: the third byte of the first pixel.
        let expected = (30.0 / 255.0 - MEAN[0]) / STD[0];
        assert!((output[0] - expected).abs() < 1e-5);
        // Slot 2 is red: the first byte of the last pixel.
        let expected = (100.0 / 255.0 - MEAN[2]) / STD[2];
        assert!((output[2 * 4 + 3] - expected).abs() < 1e-5);
    }

    #[test]
    fn axis_aligned_quad_crops_the_matching_pixels() {
        let (width, height) = (6usize, 4usize);
        let rgb = (0..width * height * 3)
            .map(|index| (index % 251) as u8)
            .collect::<Vec<_>>();
        let quad = [[1.0, 1.0], [5.0, 1.0], [5.0, 3.0], [1.0, 3.0]];
        let (crop, crop_width, crop_height) = crop_quad(&rgb, width, height, &quad);
        assert_eq!((crop_width, crop_height), (4, 2));
        for y in 0..crop_height {
            for x in 0..crop_width {
                let from = ((y + 1) * width + x + 1) * 3;
                let to = (y * crop_width + x) * 3;
                assert_eq!(crop[to..to + 3], rgb[from..from + 3]);
            }
        }
    }

    #[test]
    fn rotation_matches_numpy_rot90() {
        // 3 wide, 2 high; each pixel stores its own index.
        let rgb = (0..6u8).flat_map(|value| [value; 3]).collect::<Vec<_>>();
        // np.rot90([[0,1,2],[3,4,5]]) == [[2,5],[1,4],[0,3]]
        let (ccw, width, height) = rotate90(&rgb, 3, 2, Rotation::Ccw);
        assert_eq!((width, height), (2, 3));
        let ccw = ccw.chunks(3).map(|pixel| pixel[0]).collect::<Vec<_>>();
        assert_eq!(ccw, [2, 5, 1, 4, 0, 3]);
        // np.rot90(..., -1) == [[3,0],[4,1],[5,2]]
        let (cw, _, _) = rotate90(&rgb, 3, 2, Rotation::Cw);
        let cw = cw.chunks(3).map(|pixel| pixel[0]).collect::<Vec<_>>();
        assert_eq!(cw, [3, 0, 4, 1, 5, 2]);
    }

    #[test]
    fn recognizer_input_is_padded_with_zeros() {
        let rgb = vec![128u8; 96 * 48 * 3];
        let (input, padded) = rec_input(&rgb, 96, 48, MEAN, STD);
        assert_eq!(padded, REC_MIN_WIDTH);
        assert_eq!(input.len(), 3 * REC_HEIGHT * padded);
        assert!(input[padded - 1].abs() < f32::EPSILON);
        assert!(input[0].abs() > 0.0);
        let (_, wide) = rec_input(&vec![0u8; 960 * 48 * 3], 960, 48, MEAN, STD);
        assert_eq!(wide, 960);
    }

    #[test]
    fn ctc_collapses_repeats_and_skips_blanks() {
        let dictionary = ["あ", "い", "う"].map(str::to_owned);
        // classes: blank, あ, い, う, space
        let step = |class: usize| {
            let mut row = [0.0f32; 5];
            row[class] = 1.0;
            row
        };
        let sequence = [1, 1, 0, 1, 2, 2, 4, 3, 0];
        let scores = sequence.iter().flat_map(|class| step(*class)).collect::<Vec<_>>();
        let text = ctc_decode(&scores, sequence.len(), 5, &dictionary).unwrap();
        assert_eq!(text, "ああい う");
        assert!(ctc_decode(&scores, sequence.len(), 7, &dictionary).is_err());
    }

    #[test]
    fn dictionary_keeps_a_blank_looking_symbol_but_drops_the_trailing_newline() {
        assert_eq!(parse_dictionary("a\r\n \nb\n"), ["a", " ", "b"]);
    }

    fn line(x: f32, y: f32, vertical: bool, text: &str) -> Line {
        Line {
            center: [x, y],
            thickness: 20.0,
            vertical,
            text: text.to_owned(),
        }
    }

    #[test]
    fn vertical_text_is_read_right_to_left() {
        let ordered = order_lines(vec![
            line(40.0, 100.0, true, "三"),
            line(200.0, 120.0, true, "一"),
            line(120.0, 90.0, true, "二"),
        ]);
        let text = ordered.iter().map(|l| l.text.as_str()).collect::<String>();
        assert_eq!(text, "一二三");
    }

    #[test]
    fn horizontal_text_is_read_top_to_bottom_and_joined_by_script() {
        let ordered = order_lines(vec![
            line(100.0, 90.0, false, "world"),
            line(100.0, 40.0, false, "hello"),
        ]);
        assert_eq!(join_lines(&ordered), "hello world");
        let ordered = order_lines(vec![
            line(100.0, 70.0, false, "第二行"),
            line(100.0, 20.0, false, "第一行，"),
        ]);
        assert_eq!(join_lines(&ordered), "第一行，第二行");
    }

    #[test]
    fn lines_on_one_row_are_read_left_to_right() {
        let ordered = order_lines(vec![
            line(300.0, 52.0, false, "right"),
            line(80.0, 48.0, false, "left"),
        ]);
        assert_eq!(join_lines(&ordered), "left right");
    }
}
