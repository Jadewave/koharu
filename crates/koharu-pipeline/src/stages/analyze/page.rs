//! A page as the analyzer sees it: the scan with each block's clean-up laid
//! over it, so the lettering is gone, each text's box outlined in red and
//! tagged with its number, scaled down, and encoded as lossy WebP in base 64.
//!
//! The clean-ups are laid on by the CPU, from the tiles the project holds, so
//! the analysis needs no GPU.

use std::{collections::HashMap, sync::Arc};

use anyhow::{Context as _, Result};
use base64::Engine as _;
use image::{DynamicImage, Rgba, RgbaImage};
use imageproc::{drawing::draw_filled_rect_mut, rect::Rect};
use koharu_scene::{PageId, PixelFormat, Session, Tile, Tiles};
use parking_lot::Mutex;

use super::AnalyzerConfig;

/// What the analyzer is shown of a page.
pub(crate) struct Prepared {
    /// The texts of the page's blocks, in reading order, blank ones left out,
    /// as the translation takes them.
    pub(crate) texts: Vec<String>,
    /// The page as base 64 WebP; `None` when it has no scan.
    pub(crate) image: Option<String>,
}

/// A block with text, and what its marks and clean-up need.
struct Line {
    text: String,
    bounds: Option<[f64; 4]>,
    cleanup: Tiles,
    mask: Tiles,
}

pub(crate) fn prepare(
    session: &Arc<Mutex<Session>>,
    page: PageId,
    config: &AnalyzerConfig,
) -> Result<Prepared> {
    // Everything that needs the session is read first, so its lock isn't held
    // while the scan is decoded and drawn on.
    let (bytes, lines) = {
        let session = session.lock();
        let project = session.project();
        let bytes = match project.page(page)?.background {
            Some(background) => Some(session.file(&background.file)?),
            None => None,
        };
        let mut lines = Vec::new();
        for (_, block) in project.blocks(page)? {
            let text = block.source.content.text();
            if text.trim().is_empty() {
                continue;
            }
            let bounds = block
                .source
                .bounds()
                .map(|rect| [rect.x0, rect.y0, rect.x1, rect.y1]);
            // Only a block that reads and is cleaned has anything to lay over.
            let cleanup = session.tiles(block.cleanup.id, None)?;
            let mask = if cleanup.tiles.is_empty() {
                Tiles {
                    offset: cleanup.offset,
                    format: PixelFormat::Gray8,
                    tiles: Vec::new(),
                }
            } else {
                session.tiles(block.mask.id, None)?
            };
            lines.push(Line {
                text,
                bounds,
                cleanup,
                mask,
            });
        }
        (bytes, lines)
    };
    let image = match bytes {
        Some(bytes) => Some(picture(&bytes, &lines, config)?),
        None => None,
    };
    Ok(Prepared {
        texts: lines.into_iter().map(|line| line.text).collect(),
        image,
    })
}

fn picture(bytes: &[u8], lines: &[Line], config: &AnalyzerConfig) -> Result<String> {
    let mut page = image::load_from_memory(bytes)
        .context("couldn't decode the page's scan")?
        .to_rgba8();
    let (width, height) = page.dimensions();
    for line in lines {
        lay(&mut page, &line.cleanup, &line.mask);
    }

    let side = config.size.max(64);
    let scale = (f64::from(side) / f64::from(width.max(height))).min(1.0);
    let mut page = if scale < 1.0 {
        DynamicImage::ImageRgba8(page).thumbnail(side, side).to_rgba8()
    } else {
        page
    };
    let boxes: Vec<Option<[f64; 4]>> = lines
        .iter()
        .map(|line| line.bounds.map(|bounds| bounds.map(|edge| edge * scale)))
        .collect();
    mark(&mut page, &boxes);

    let rgb = DynamicImage::ImageRgba8(page).to_rgb8();
    let quality = f32::from(config.quality.clamp(1, 100));
    let encoded = webp::Encoder::from_rgb(rgb.as_raw(), rgb.width(), rgb.height()).encode(quality);
    Ok(base64::engine::general_purpose::STANDARD.encode(&*encoded))
}

/// The mask's value at a page pixel: how much of the clean-up shows there.
struct Coverage<'a> {
    tiles: HashMap<(i32, i32), &'a Tile>,
    offset: koharu_scene::Offset,
}

impl<'a> Coverage<'a> {
    fn new(mask: &'a Tiles) -> Self {
        Self {
            tiles: mask.tiles.iter().map(|tile| ((tile.x, tile.y), tile)).collect(),
            offset: mask.offset,
        }
    }

    fn at(&self, x: i64, y: i64) -> u8 {
        let size = i64::from(Tile::SIZE);
        let (x, y) = (x - i64::from(self.offset.x), y - i64::from(self.offset.y));
        let (column, row) = (x.div_euclid(size), y.div_euclid(size));
        let (Ok(column), Ok(row)) = (i32::try_from(column), i32::try_from(row)) else {
            return 0;
        };
        self.tiles.get(&(column, row)).map_or(0, |tile| {
            let at = (y.rem_euclid(size) * size + x.rem_euclid(size)) as usize;
            tile.pixels.get(at).copied().unwrap_or(0)
        })
    }
}

/// Lays `cleanup` over `page` where `mask` is set, as a block's clean-up
/// shows only inside its mask.
fn lay(page: &mut RgbaImage, cleanup: &Tiles, mask: &Tiles) {
    if cleanup.format != PixelFormat::Rgba8 || cleanup.tiles.is_empty() {
        return;
    }
    let coverage = Coverage::new(mask);
    let size = i64::from(Tile::SIZE);
    let (width, height) = (i64::from(page.width()), i64::from(page.height()));
    for tile in &cleanup.tiles {
        let left = i64::from(tile.x) * size + i64::from(cleanup.offset.x);
        let top = i64::from(tile.y) * size + i64::from(cleanup.offset.y);
        for (at, pixel) in tile.pixels.chunks_exact(4).enumerate() {
            let (x, y) = (left + at as i64 % size, top + at as i64 / size);
            if !(0..width).contains(&x) || !(0..height).contains(&y) {
                continue;
            }
            let shown = u32::from(coverage.at(x, y)) * u32::from(pixel[3]);
            if shown == 0 {
                continue;
            }
            let under = page.get_pixel_mut(x as u32, y as u32);
            for channel in 0..3 {
                let mixed = u32::from(pixel[channel]) * shown
                    + u32::from(under[channel]) * (255 * 255 - shown);
                under[channel] = (mixed / (255 * 255)) as u8;
            }
        }
    }
}

/// 5×7 dot patterns of the digits, to tag the boxes without a font.
const DIGITS: [[&str; 7]; 10] = [
    ["01110", "10001", "10011", "10101", "11001", "10001", "01110"],
    ["00100", "01100", "00100", "00100", "00100", "00100", "01110"],
    ["01110", "10001", "00001", "00010", "00100", "01000", "11111"],
    ["11110", "00001", "00001", "01110", "00001", "00001", "11110"],
    ["00010", "00110", "01010", "10010", "11111", "00010", "00010"],
    ["11111", "10000", "11110", "00001", "00001", "10001", "01110"],
    ["00110", "01000", "10000", "11110", "10001", "10001", "01110"],
    ["11111", "00001", "00010", "00100", "01000", "01000", "01000"],
    ["01110", "10001", "10001", "01110", "10001", "10001", "01110"],
    ["01110", "10001", "10001", "01111", "00001", "00010", "01100"],
];

/// Outlines each box in red, outside it so the text stays readable, with a
/// tag above it holding its number, from 1, which the prompt's lines carry.
fn mark(canvas: &mut RgbaImage, boxes: &[Option<[f64; 4]>]) {
    let (width, height) = canvas.dimensions();
    let thickness = (height / 500).clamp(2, 6) as i32;
    let dot = (height / 220).clamp(3, 10) as i32;
    let red = Rgba([220, 20, 20, 255]);
    let white = Rgba([255, 255, 255, 255]);
    let clamp_x = |value: f64| value.clamp(0.0, f64::from(width.saturating_sub(1))) as i32;
    let clamp_y = |value: f64| value.clamp(0.0, f64::from(height.saturating_sub(1))) as i32;
    for (offset, bounds) in boxes.iter().enumerate() {
        let Some([left, top, right, bottom]) = *bounds else {
            continue;
        };
        let (left, top) = (clamp_x(left), clamp_y(top));
        let (right, bottom) = (clamp_x(right).max(left + 1), clamp_y(bottom).max(top + 1));
        let (box_width, box_height) = ((right - left) as u32, (bottom - top) as u32);
        let outer_width = box_width + 2 * thickness as u32;
        for (x, y, w, h) in [
            (left - thickness, top - thickness, outer_width, thickness as u32),
            (left - thickness, bottom, outer_width, thickness as u32),
            (left - thickness, top, thickness as u32, box_height),
            (right, top, thickness as u32, box_height),
        ] {
            draw_filled_rect_mut(canvas, Rect::at(x, y).of_size(w, h), red);
        }
        let digits: Vec<usize> = (offset + 1)
            .to_string()
            .bytes()
            .map(|byte| usize::from(byte - b'0'))
            .collect();
        let padding = dot;
        let tag_height = 7 * dot + 2 * padding;
        let tag_width = digits.len() as i32 * 6 * dot - dot + 2 * padding;
        let tag_x = left - thickness;
        // Above the box, or inside its top edge when there's no room above.
        let tag_y = if top - thickness - tag_height >= 0 {
            top - thickness - tag_height
        } else {
            top
        };
        draw_filled_rect_mut(
            canvas,
            Rect::at(tag_x, tag_y).of_size(tag_width as u32, tag_height as u32),
            red,
        );
        for (index, digit) in digits.into_iter().enumerate() {
            let origin_x = tag_x + padding + index as i32 * 6 * dot;
            for (row, pattern) in DIGITS[digit].iter().enumerate() {
                for (column, cell) in pattern.bytes().enumerate() {
                    if cell == b'1' {
                        draw_filled_rect_mut(
                            canvas,
                            Rect::at(
                                origin_x + column as i32 * dot,
                                tag_y + padding + row as i32 * dot,
                            )
                            .of_size(dot as u32, dot as u32),
                            white,
                        );
                    }
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use koharu_scene::Offset;

    use super::*;

    fn tiles(format: PixelFormat, value: &[u8]) -> Tiles {
        Tiles {
            offset: Offset { x: 0, y: 0 },
            format,
            tiles: vec![Tile {
                x: 0,
                y: 0,
                pixels: value.repeat(64 * 64),
            }],
        }
    }

    #[test]
    fn a_clean_up_shows_only_where_the_mask_is_set() {
        let mut page = RgbaImage::from_pixel(64, 64, Rgba([0, 0, 0, 255]));
        let cleanup = tiles(PixelFormat::Rgba8, &[255, 255, 255, 255]);
        let mut mask = tiles(PixelFormat::Gray8, &[0]);
        mask.tiles[0].pixels[0] = 255;
        lay(&mut page, &cleanup, &mask);
        assert_eq!(page.get_pixel(0, 0), &Rgba([255, 255, 255, 255]));
        assert_eq!(page.get_pixel(1, 0), &Rgba([0, 0, 0, 255]));
    }

    #[test]
    fn a_tag_is_drawn_inside_the_page_when_the_box_touches_its_top() {
        let mut canvas = RgbaImage::from_pixel(200, 400, Rgba([0, 0, 0, 255]));
        mark(&mut canvas, &[Some([20.0, 0.0, 80.0, 60.0])]);
        assert!(canvas.pixels().any(|pixel| pixel == &Rgba([220, 20, 20, 255])));
    }
}
