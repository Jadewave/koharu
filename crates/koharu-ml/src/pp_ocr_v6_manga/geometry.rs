//! Deterministic helpers for the PP-OCRv6 manga pipeline.
//!
//! Everything here is arithmetic on pixel buffers and strings, so it can be
//! unit-tested without ONNX Runtime. The numerical conventions follow the
//! reference demo (`Kellenok/PP-OCR_manga`, `app.py`), which uses OpenCV:
//! `cv2.resize(INTER_LINEAR)` and `cv2.warpPerspective(INTER_LINEAR)`.

/// White border added around the image before detection, as the demo does.
pub(super) const DET_MARGIN: usize = 16;
const DET_LONG_SIDE: f64 = 960.0;
const DET_SMALL_SIDE: f64 = 480.0;
/// Images more elongated than this get a pixel budget instead of a long side.
const STRIP_ASPECT: f64 = 2.0;
const DET_MIN_INPUT: usize = 64;

pub(super) const REC_HEIGHT: usize = 48;
const REC_MIN_WIDTH: usize = 16;
const REC_MAX_WIDTH: usize = 640;
/// Very long lines (credits, captions) are read wider instead of squeezed.
const REC_WIDE_FROM: usize = 2000;
const REC_MAX_WIDE_WIDTH: usize = 2400;

/// Detector input `(width, height)` for an image that already carries the margin.
///
/// Both sides are multiples of 32. Images between 480 and 960 pixels are kept at
/// their size, larger ones are shrunk to 960 on the long side and smaller ones
/// are enlarged to 480. Strips more elongated than 2:1 are sized by the pixel
/// budget of a 3:4 page at 960 instead.
pub(super) fn detector_size(width: usize, height: usize) -> (usize, usize) {
    let (width, height) = (width.max(1) as f64, height.max(1) as f64);
    let long = width.max(height);
    let short = width.min(height);
    let scale = if long / short > STRIP_ASPECT {
        (0.75 * DET_LONG_SIDE * DET_LONG_SIDE / (width * height))
            .sqrt()
            .min(1.0)
    } else if long > DET_LONG_SIDE {
        DET_LONG_SIDE / long
    } else if long < DET_SMALL_SIDE {
        (DET_SMALL_SIDE / long).max(1.0)
    } else {
        1.0
    };
    let side = |value: f64| (((value * scale / 32.0).round() as usize) * 32).max(DET_MIN_INPUT);
    (side(width), side(height))
}

/// Adds a white border of `margin` pixels to an interleaved RGB8 image.
pub(super) fn pad_white(
    rgb: &[u8],
    width: usize,
    height: usize,
    margin: usize,
) -> (Vec<u8>, usize, usize) {
    let (padded_width, padded_height) = (width + 2 * margin, height + 2 * margin);
    let mut output = vec![255u8; padded_width * padded_height * 3];
    for y in 0..height {
        let from = y * width * 3;
        let to = ((y + margin) * padded_width + margin) * 3;
        output[to..to + width * 3].copy_from_slice(&rgb[from..from + width * 3]);
    }
    (output, padded_width, padded_height)
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
/// The demo works on OpenCV's BGR arrays, so `mean` and `std` are indexed by BGR
/// slot (blue first).
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

/// Orders the corners of a box as top-left, top-right, bottom-right,
/// bottom-left using coordinate sums and differences, as the demo does.
pub(super) fn order_corners(points: &[[f32; 2]; 4]) -> [[f32; 2]; 4] {
    let (mut top_left, mut bottom_right, mut top_right, mut bottom_left) = (0, 0, 0, 0);
    for (index, point) in points.iter().enumerate() {
        let sum = point[0] + point[1];
        let diff = point[1] - point[0];
        if sum < points[top_left][0] + points[top_left][1] {
            top_left = index;
        }
        if sum > points[bottom_right][0] + points[bottom_right][1] {
            bottom_right = index;
        }
        if diff < points[top_right][1] - points[top_right][0] {
            top_right = index;
        }
        if diff > points[bottom_left][1] - points[bottom_left][0] {
            bottom_left = index;
        }
    }
    [
        points[top_left],
        points[top_right],
        points[bottom_right],
        points[bottom_left],
    ]
}

fn bilinear(rgb: &[u8], width: usize, height: usize, x: f32, y: f32) -> [f32; 3] {
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

/// Cuts the upright line bounded by `corners` (top-left, top-right,
/// bottom-right, bottom-left) out of an interleaved RGB8 image.
///
/// This stands in for `cv2.warpPerspective` with the destination corners at
/// `(0, 0)`, `(w - 1, 0)`, `(w - 1, h - 1)` and `(0, h - 1)`. The boxes are
/// rotated rectangles, for which bilinear interpolation of the four corners is
/// the same mapping. Pixels outside the image replicate the border. Returns
/// `None` for a degenerate box.
pub(super) fn crop_perspective(
    rgb: &[u8],
    width: usize,
    height: usize,
    corners: &[[f32; 2]; 4],
) -> Option<(Vec<u8>, usize, usize)> {
    let [top_left, top_right, bottom_right, bottom_left] = *corners;
    let out_width = distance(bottom_right, bottom_left).max(distance(top_right, top_left)) as usize;
    let out_height = distance(top_right, bottom_right).max(distance(top_left, bottom_left)) as usize;
    if out_width == 0 || out_height == 0 || width == 0 || height == 0 {
        return None;
    }
    let mut output = vec![0u8; out_width * out_height * 3];
    for y in 0..out_height {
        let v = if out_height > 1 {
            y as f32 / (out_height - 1) as f32
        } else {
            0.0
        };
        for x in 0..out_width {
            let u = if out_width > 1 {
                x as f32 / (out_width - 1) as f32
            } else {
                0.0
            };
            let point = |axis: usize| {
                (1.0 - u) * (1.0 - v) * corners[0][axis]
                    + u * (1.0 - v) * corners[1][axis]
                    + u * v * corners[2][axis]
                    + (1.0 - u) * v * corners[3][axis]
            };
            let pixel = bilinear(rgb, width, height, point(0), point(1));
            let at = (y * out_width + x) * 3;
            for (channel, value) in pixel.iter().enumerate() {
                output[at + channel] = value.round().clamp(0.0, 255.0) as u8;
            }
        }
    }
    Some((output, out_width, out_height))
}

/// Rotates an interleaved RGB8 image by 90 degrees counter-clockwise
/// (`np.rot90`, `cv2.ROTATE_90_COUNTERCLOCKWISE`).
pub(super) fn rotate_ccw(rgb: &[u8], width: usize, height: usize) -> (Vec<u8>, usize, usize) {
    let (out_width, out_height) = (height, width);
    let mut output = vec![0u8; rgb.len()];
    for y in 0..out_height {
        for x in 0..out_width {
            let from = (x * width + (width - 1 - y)) * 3;
            let to = (y * out_width + x) * 3;
            output[to..to + 3].copy_from_slice(&rgb[from..from + 3]);
        }
    }
    (output, out_width, out_height)
}

/// Recognizer input `[3, REC_HEIGHT, width]` and its width, or `None` for a crop
/// too small to read.
///
/// The line is resized to height 48 keeping its aspect ratio, at most 640 wide
/// (2400 for lines that would be wider than 2000), and normalized to `[-1, 1]`.
/// Unlike the stock PaddleOCR recognizer there is no padding to a minimum width.
pub(super) fn rec_input(rgb: &[u8], width: usize, height: usize) -> Option<(Vec<f32>, usize)> {
    if width < 2 || height < 2 {
        return None;
    }
    let natural = (REC_HEIGHT as f64 * width as f64 / height as f64).round() as usize;
    let cap = if natural > REC_WIDE_FROM {
        REC_MAX_WIDE_WIDTH
    } else {
        REC_MAX_WIDTH
    };
    let target = natural.min(cap).max(REC_MIN_WIDTH);
    let planes = resize_normalize(rgb, width, height, target, REC_HEIGHT, [0.5; 3], [0.5; 3]);
    Some((planes, target))
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
/// Class 0 is the blank, classes `1..=dictionary.len()` are the dictionary and
/// the demo appends a final space class.
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

#[cfg(test)]
mod tests {
    use super::*;

    const MEAN: [f32; 3] = [0.485, 0.456, 0.406];
    const STD: [f32; 3] = [0.229, 0.224, 0.225];

    #[test]
    fn detector_size_follows_the_demo() {
        // 1280x720 frame plus the margin: shrunk to a 960 long side.
        assert_eq!(detector_size(1312, 752), (960, 544));
        // Between 480 and 960 the size is kept (rounded to 32).
        assert_eq!(detector_size(700, 500), (704, 512));
        // Small crops are enlarged until the long side reaches 480.
        assert_eq!(detector_size(240, 120), (480, 256));
        // Elongated strips use the 3:4 pixel budget.
        let (width, height) = detector_size(832, 4032);
        assert_eq!((width % 32, height % 32), (0, 0));
        assert!(width * height <= 691_200 + 2 * 32 * 4032);
    }

    #[test]
    fn white_margin_surrounds_the_image() {
        let rgb = [10u8, 20, 30, 40, 50, 60];
        let (padded, width, height) = pad_white(&rgb, 2, 1, 1);
        assert_eq!((width, height), (4, 3));
        assert_eq!(&padded[..3], &[255, 255, 255]);
        let row = (width + 1) * 3;
        assert_eq!(&padded[row..row + 6], &rgb);
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
    fn corners_are_ordered_like_the_demo() {
        let ordered = order_corners(&[[40.0, 30.0], [0.0, 0.0], [0.0, 30.0], [40.0, 0.0]]);
        assert_eq!(ordered, [[0.0, 0.0], [40.0, 0.0], [40.0, 30.0], [0.0, 30.0]]);
    }

    #[test]
    fn crop_spans_the_box_between_its_end_pixels() {
        let (width, height) = (6usize, 4usize);
        let rgb = (0..width * height * 3)
            .map(|index| (index % 251) as u8)
            .collect::<Vec<_>>();
        let corners = [[1.0, 1.0], [4.0, 1.0], [4.0, 2.0], [1.0, 2.0]];
        let (crop, crop_width, crop_height) =
            crop_perspective(&rgb, width, height, &corners).unwrap();
        // The demo truncates the side lengths (3 and 1) and maps the destination
        // corners at `w - 1`, so the first and last pixels are the box's end pixels.
        assert_eq!((crop_width, crop_height), (3, 1));
        let first = (width + 1) * 3;
        let last = (width + 4) * 3;
        assert_eq!(crop[..3], rgb[first..first + 3]);
        assert_eq!(crop[6..9], rgb[last..last + 3]);
        assert!(crop_perspective(&rgb, width, height, &[[1.0, 1.0]; 4]).is_none());
    }

    #[test]
    fn rotation_matches_numpy_rot90() {
        // 3 wide, 2 high; each pixel stores its own index.
        let rgb = (0..6u8).flat_map(|value| [value; 3]).collect::<Vec<_>>();
        // np.rot90([[0,1,2],[3,4,5]]) == [[2,5],[1,4],[0,3]]
        let (rotated, width, height) = rotate_ccw(&rgb, 3, 2);
        assert_eq!((width, height), (2, 3));
        let values = rotated.chunks(3).map(|pixel| pixel[0]).collect::<Vec<_>>();
        assert_eq!(values, [2, 5, 1, 4, 0, 3]);
    }

    #[test]
    fn recognizer_input_has_no_padding_and_is_capped() {
        let rgb = vec![128u8; 96 * 48 * 3];
        let (input, width) = rec_input(&rgb, 96, 48).unwrap();
        assert_eq!(width, 96);
        assert_eq!(input.len(), 3 * REC_HEIGHT * width);
        // 128 / 255 normalized with mean = std = 0.5.
        assert!((input[0] - (128.0 / 255.0 - 0.5) / 0.5).abs() < 1e-5);
        // A very long line is squeezed to 640, an extreme one to 2400.
        assert_eq!(rec_input(&vec![0u8; 960 * 48 * 3], 960, 48).unwrap().1, 640);
        assert_eq!(rec_input(&vec![0u8; 2400 * 48 * 3], 2400, 48).unwrap().1, 2400);
        assert_eq!(rec_input(&vec![0u8; 3 * 48 * 3], 3, 48).unwrap().1, REC_MIN_WIDTH);
        assert!(rec_input(&[0u8; 3], 1, 1).is_none());
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
        let scores = sequence
            .iter()
            .flat_map(|class| step(*class))
            .collect::<Vec<_>>();
        let text = ctc_decode(&scores, sequence.len(), 5, &dictionary).unwrap();
        assert_eq!(text, "ああい う");
        assert!(ctc_decode(&scores, sequence.len(), 7, &dictionary).is_err());
    }

    #[test]
    fn dictionary_keeps_a_blank_looking_symbol_but_drops_the_trailing_newline() {
        assert_eq!(parse_dictionary("a\r\n \nb\n"), ["a", " ", "b"]);
    }
}
