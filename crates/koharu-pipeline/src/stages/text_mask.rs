//! Completes the text mask with the strokes of the text that the instance masks miss.
//!
//! The layout model draws one mask per detected text instance, and a text box can hold glyphs
//! that no kept mask reaches: a column the detector gave its own instance, a character at the end
//! of a line, a stroke that the mask threshold cut. What is left is a piece of a character in the
//! middle of an otherwise clean balloon, which is exactly what a person paints over by hand with
//! the brush of the editor.
//!
//! Inside the box of a detected text, on a background that is one flat colour, whatever differs
//! strongly from that colour and is not in the mask yet is the text. It is added to the mask,
//! with a small margin so that the soft edges of the strokes go with it. Text on a picture
//! (screentone, shading, line art) has no flat background and is left to the instance masks:
//! the strokes cannot be told from the drawing there.

use image::{GrayImage, Luma, RgbImage};
use imageproc::{distance_transform::Norm, morphology::dilate};

/// The tight box of a text is widened by this many pixels: strokes and their soft edges sit on
/// the border of the box.
const BOX_MARGIN: i64 = 3;
/// The background is sampled this far outside the box (in multiples of the mask radius), so that
/// a mask that covers the whole box still leaves pixels to sample.
const BACKGROUND_REACH: i64 = 2;
/// Fewer unmasked pixels than this make no background.
const MIN_BACKGROUND_PIXELS: usize = 64;
/// How far a pixel may be from the background colour (in luminance) and still be background.
const BACKGROUND_TOLERANCE: i32 = 30;
/// At least this share of the sampled pixels must be background: the colour is flat.
const FLAT_SHARE: f32 = 0.7;
/// How far from the background colour (in luminance) a pixel is a stroke.
const INK_CONTRAST: i32 = 40;
/// More ink than this share of the whole box is not text (the background guess was wrong).
const MAX_INK_SHARE: f32 = 0.4;

fn luminance(pixel: &image::Rgb<u8>) -> i32 {
    (299 * i32::from(pixel[0]) + 587 * i32::from(pixel[1]) + 114 * i32::from(pixel[2])) / 1000
}

fn window(
    [left, top, right, bottom]: [f32; 4],
    margin: i64,
    width: u32,
    height: u32,
) -> Option<[u32; 4]> {
    if ![left, top, right, bottom].iter().all(|value| value.is_finite()) {
        return None;
    }
    let clamp = |value: i64, limit: u32| value.clamp(0, i64::from(limit)) as u32;
    let window = [
        clamp(left.floor() as i64 - margin, width),
        clamp(top.floor() as i64 - margin, height),
        clamp(right.ceil() as i64 + margin, width),
        clamp(bottom.ceil() as i64 + margin, height),
    ];
    (window[2] > window[0] && window[3] > window[1]).then_some(window)
}

/// Adds to `mask` the strokes inside the `boxes` of the detected texts that it does not cover.
/// `grow` is the radius the instance masks were closed with; the strokes are widened by a third
/// of it. Returns how many pixels were added.
pub(super) fn complete_text_mask(
    image: &RgbImage,
    boxes: &[[f32; 4]],
    mask: &mut GrayImage,
    grow: u8,
) -> usize {
    let (width, height) = image.dimensions();
    if mask.dimensions() != (width, height) {
        return 0;
    }
    let reach = BOX_MARGIN.max(i64::from(grow) * BACKGROUND_REACH);
    let widen = (grow / 3).max(2);
    let mut added = 0;
    for &bounds in boxes {
        let (Some(sample), Some(inside)) = (
            window(bounds, reach, width, height),
            window(bounds, BOX_MARGIN, width, height),
        ) else {
            continue;
        };

        // The colour of the background: the median of what the mask does not cover around the
        // text, which has to be mostly that colour.
        let mut values = Vec::new();
        for y in sample[1]..sample[3] {
            for x in sample[0]..sample[2] {
                if mask.get_pixel(x, y)[0] == 0 {
                    values.push(luminance(image.get_pixel(x, y)));
                }
            }
        }
        if values.len() < MIN_BACKGROUND_PIXELS {
            continue;
        }
        values.sort_unstable();
        let background = values[values.len() / 2];
        let flat = values
            .iter()
            .filter(|value| (**value - background).abs() <= BACKGROUND_TOLERANCE)
            .count();
        if (flat as f32) < values.len() as f32 * FLAT_SHARE {
            continue;
        }

        // The strokes: what the mask does not cover inside the box and what stands out.
        let (left, top) = (inside[0], inside[1]);
        let mut ink = GrayImage::new(inside[2] - left, inside[3] - top);
        let mut strokes = 0usize;
        for y in inside[1]..inside[3] {
            for x in inside[0]..inside[2] {
                if mask.get_pixel(x, y)[0] != 0 {
                    continue;
                }
                if (luminance(image.get_pixel(x, y)) - background).abs() >= INK_CONTRAST {
                    strokes += 1;
                    ink.put_pixel(x - left, y - top, Luma([u8::MAX]));
                }
            }
        }
        let area = (inside[2] - inside[0]) as f32 * (inside[3] - inside[1]) as f32;
        if strokes == 0 || strokes as f32 > area * MAX_INK_SHARE {
            continue;
        }

        // A little wider than the strokes: their soft edges and the pixels between two strokes.
        let ink = dilate(&ink, Norm::L2, widen);
        for (x, y, pixel) in ink.enumerate_pixels() {
            let (x, y) = (left + x, top + y);
            if pixel[0] != 0 && mask.get_pixel(x, y)[0] == 0 {
                mask.put_pixel(x, y, Luma([u8::MAX]));
                added += 1;
            }
        }
    }
    added
}

#[cfg(test)]
mod tests {
    use image::Rgb;

    use super::*;

    const WHITE: Rgb<u8> = Rgb([250, 250, 250]);
    const BLACK: Rgb<u8> = Rgb([10, 10, 10]);

    /// A white page with a column of "glyphs": vertical bars of 5x40 pixels.
    fn page() -> RgbImage {
        let mut image = RgbImage::from_pixel(300, 300, WHITE);
        for glyph in 0..4 {
            for y in 0..40 {
                for x in 0..5 {
                    image.put_pixel(100 + x, 60 + glyph * 50 + y, BLACK);
                }
            }
        }
        image
    }

    fn covered(mask: &GrayImage, x: u32, y: u32) -> bool {
        mask.get_pixel(x, y)[0] != 0
    }

    #[test]
    fn a_glyph_that_no_instance_mask_reaches_is_added() {
        let image = page();
        let mut mask = GrayImage::new(300, 300);
        // The instance masks cover the first two glyphs, with their closing; the last two are
        // missing, as a dropped instance leaves them.
        for y in 40..160 {
            for x in 80..130 {
                mask.put_pixel(x, y, Luma([255]));
            }
        }
        let before = mask.clone();
        let boxes = [[100.0, 60.0, 105.0, 250.0]];
        let added = complete_text_mask(&image, &boxes, &mut mask, 15);
        assert!(added > 0);
        for glyph in 2..4 {
            for y in 0..40 {
                for x in 0..5 {
                    assert!(covered(&mask, 100 + x, 60 + glyph * 50 + y), "glyph {glyph} at {x},{y}");
                }
            }
        }
        // What was covered stays covered.
        assert!(before.pixels().zip(mask.pixels()).all(|(old, new)| old[0] == 0 || new[0] != 0));
        // Far from the box, nothing is added.
        assert!(!covered(&mask, 10, 10) && !covered(&mask, 290, 290));
    }

    #[test]
    fn a_fragment_beside_a_covered_area_is_added_when_the_mask_leaves_only_a_few_pixels() {
        // The mask covers nearly the whole box and one stroke sticks out of it by a few pixels.
        let image = page();
        let mut mask = GrayImage::new(300, 300);
        for y in 50..238 {
            for x in 85..122 {
                mask.put_pixel(x, y, Luma([255]));
            }
        }
        // Remove the cover of the end of the last glyph.
        for y in 225..238 {
            for x in 85..122 {
                mask.put_pixel(x, y, Luma([0]));
            }
        }
        let boxes = [[100.0, 60.0, 105.0, 250.0]];
        complete_text_mask(&image, &boxes, &mut mask, 15);
        for y in 225..240 {
            assert!(covered(&mask, 102, y), "row {y}");
        }
    }

    #[test]
    fn a_dark_object_outside_the_box_is_never_touched() {
        let mut image = page();
        for y in 100..140 {
            for x in 200..230 {
                image.put_pixel(x, y, BLACK);
            }
        }
        let mut mask = GrayImage::new(300, 300);
        let boxes = [[100.0, 60.0, 105.0, 250.0]];
        complete_text_mask(&image, &boxes, &mut mask, 15);
        assert!(!covered(&mask, 215, 120));
        // The glyphs are all in the mask now.
        assert!(covered(&mask, 102, 80) && covered(&mask, 102, 230));
    }

    #[test]
    fn text_on_a_picture_is_left_to_the_instance_masks() {
        // A checkerboard: the background is not one colour.
        let mut image = RgbImage::new(300, 300);
        for (x, y, pixel) in image.enumerate_pixels_mut() {
            *pixel = if (x / 4 + y / 4) % 2 == 0 { WHITE } else { Rgb([40, 40, 40]) };
        }
        let mut mask = GrayImage::new(300, 300);
        let boxes = [[100.0, 60.0, 105.0, 250.0]];
        assert_eq!(complete_text_mask(&image, &boxes, &mut mask, 15), 0);
        assert!(mask.pixels().all(|pixel| pixel[0] == 0));
    }

    #[test]
    fn white_text_on_a_dark_flat_background_is_found() {
        let mut image = RgbImage::from_pixel(300, 300, Rgb([20, 20, 20]));
        for y in 60..100 {
            for x in 100..105 {
                image.put_pixel(x, y, Rgb([240, 240, 240]));
            }
        }
        let mut mask = GrayImage::new(300, 300);
        let boxes = [[100.0, 60.0, 105.0, 100.0]];
        assert!(complete_text_mask(&image, &boxes, &mut mask, 15) > 0);
        assert!(covered(&mask, 102, 80));
    }

    #[test]
    fn a_page_without_text_boxes_or_with_a_bad_box_changes_nothing() {
        let image = page();
        let mut mask = GrayImage::new(300, 300);
        assert_eq!(complete_text_mask(&image, &[], &mut mask, 15), 0);
        let bad = [[f32::NAN, 0.0, 10.0, 10.0], [400.0, 400.0, 500.0, 500.0], [50.0, 50.0, 50.0, 50.0]];
        assert_eq!(complete_text_mask(&image, &bad, &mut mask, 15), 0);
        // A mask of another size is not touched.
        let mut other = GrayImage::new(10, 10);
        assert_eq!(complete_text_mask(&image, &[[1.0, 1.0, 5.0, 5.0]], &mut other, 15), 0);
    }

    #[test]
    fn a_box_with_no_stroke_adds_nothing() {
        let image = RgbImage::from_pixel(300, 300, WHITE);
        let mut mask = GrayImage::new(300, 300);
        assert_eq!(complete_text_mask(&image, &[[100.0, 60.0, 140.0, 100.0]], &mut mask, 15), 0);
    }
}
