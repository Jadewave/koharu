//! Optional diagnostics: set `KOHARU_PPOCR_DEBUG` to a directory and every call
//! writes the detector's probability map and the crop with the detected lines
//! drawn on it, and logs each line's score and text.

use std::{
    path::Path,
    sync::atomic::{AtomicUsize, Ordering},
};

use image::{GrayImage, Luma, Rgb, RgbImage};

use super::layout::LineBox;

static COUNTER: AtomicUsize = AtomicUsize::new(0);

/// The detector's probability map.
pub(super) struct DetectorMap {
    pub values: Vec<f32>,
    pub width: u32,
    pub height: u32,
}

fn draw_line(image: &mut RgbImage, from: [f32; 2], to: [f32; 2], color: Rgb<u8>) {
    let steps = (to[0] - from[0])
        .abs()
        .max((to[1] - from[1]).abs())
        .ceil()
        .max(1.0) as u32;
    for step in 0..=steps {
        let t = step as f32 / steps as f32;
        let x = from[0] + (to[0] - from[0]) * t;
        let y = from[1] + (to[1] - from[1]) * t;
        if x >= 0.0 && y >= 0.0 && (x as u32) < image.width() && (y as u32) < image.height() {
            image.put_pixel(x as u32, y as u32, color);
        }
    }
}

/// Writes `NNNN.input.png`, `NNNN.map.png` and `NNNN.boxes.png` into `directory`.
/// `texts` is aligned with `lines`; an empty text means the line was not read.
pub(super) fn dump(
    directory: &Path,
    rgb: &RgbImage,
    map: &DetectorMap,
    lines: &[LineBox],
    texts: &[String],
) {
    let index = COUNTER.fetch_add(1, Ordering::Relaxed);
    let stem = directory.join(format!("{index:04}"));

    let mut overlay = rgb.clone();
    for (number, line) in lines.iter().enumerate() {
        for edge in 0..4 {
            draw_line(
                &mut overlay,
                line.corners[edge],
                line.corners[(edge + 1) % 4],
                Rgb([255, 0, 0]),
            );
        }
        tracing::info!(
            "PP-OCRv6 manga #{index:04} line {}: score {:.2}, box ({:.0},{:.0})-({:.0},{:.0}), text {:?}",
            number + 1,
            line.score,
            line.xmin,
            line.ymin,
            line.xmax,
            line.ymax,
            texts.get(number).map(String::as_str).unwrap_or("")
        );
    }
    if lines.is_empty() {
        tracing::info!("PP-OCRv6 manga #{index:04}: the detector found no lines");
    }

    let probability = GrayImage::from_fn(map.width, map.height, |x, y| {
        let value = map.values[(y * map.width + x) as usize];
        Luma([(value.clamp(0.0, 1.0) * 255.0) as u8])
    });
    for (name, result) in [
        ("input", rgb.save(stem.with_extension("input.png"))),
        ("boxes", overlay.save(stem.with_extension("boxes.png"))),
        ("map", probability.save(stem.with_extension("map.png"))),
    ] {
        if let Err(error) = result {
            tracing::warn!("PP-OCRv6 manga debug {name} image was not saved: {error}");
        }
    }
}
