//! DB post-processing of the text detector, as in the reference demo.
//!
//! The demo binarizes the probability map, takes every contour, scores it, grows
//! it by `area * ratio / perimeter` with `pyclipper`, fits a minimum-area rectangle
//! to the grown polygon and adds 4 pixels to both sides of that rectangle.
//!
//! The minimum-area rectangle of a polygon grown by a round offset depends only on
//! the polygon's convex hull: its extent along any direction grows by exactly twice
//! the offset, and some side lies on a hull edge. So the rectangle is computed from
//! the hull directly instead of materializing the grown polygon.

use image::{GrayImage, Luma};
use imageproc::contours::find_contours_with_threshold;

use super::layout::LineBox;
use crate::pp_ocr_v6::det::polygon_score;

/// Boxes with a side shorter than this (in pixels of the original image) are dropped.
const MIN_BOX_SIDE: f32 = 6.0;
/// Pixels added to each side of the fitted rectangle (the demo adds 4 to its size).
const BOX_GROWTH: f32 = 2.0;
const MAX_CONTOURS: usize = 5000;

#[derive(Debug, Clone, Copy, PartialEq)]
pub(super) struct DbParams {
    pub threshold: f32,
    pub box_threshold: f32,
    pub unclip_ratio: f32,
}

/// How the detector's probability map relates to the original image.
#[derive(Debug, Clone, Copy)]
pub(super) struct MapLayout {
    pub map_width: u32,
    pub map_height: u32,
    /// Size of the image fed to the detector before resizing (original + margin).
    pub padded_width: usize,
    pub padded_height: usize,
    pub margin: usize,
    /// Size of the original image.
    pub width: usize,
    pub height: usize,
}

/// Finds text lines in the detector's probability map.
pub(super) fn line_boxes(prediction: &[f32], layout: &MapLayout, params: DbParams) -> Vec<LineBox> {
    let (map_width, map_height) = (layout.map_width, layout.map_height);
    // A background border reproduces OpenCV's treatment of pixels outside the image.
    let bitmap = GrayImage::from_fn(map_width + 2, map_height + 2, |x, y| {
        if x == 0 || y == 0 || x > map_width || y > map_height {
            return Luma([0]);
        }
        let value = prediction[(y as usize - 1) * map_width as usize + x as usize - 1];
        Luma([if value > params.threshold { 255 } else { 0 }])
    });
    let scale_x = layout.padded_width as f32 / map_width as f32;
    let scale_y = layout.padded_height as f32 / map_height as f32;
    let margin = layout.margin as f32;
    let (width, height) = (layout.width as f32, layout.height as f32);

    let mut boxes = Vec::new();
    // OpenCV's RETR_LIST exposes contours in the reverse of imageproc's discovery order.
    for contour in find_contours_with_threshold::<i32>(&bitmap, 0)
        .into_iter()
        .rev()
        .take(MAX_CONTOURS)
    {
        let points = contour
            .points
            .iter()
            .map(|point| [(point.x - 1) as f32, (point.y - 1) as f32])
            .collect::<Vec<_>>();
        if points.len() < 4 {
            continue;
        }
        let score = polygon_score(prediction, map_width, map_height, &points);
        if score < params.box_threshold {
            continue;
        }
        let Some((area, perimeter)) = area_and_perimeter(&points) else {
            continue;
        };
        let offset = area * params.unclip_ratio / perimeter;

        // The offset is applied in map space; the hull is then moved to image space.
        let hull = convex_hull(&points)
            .into_iter()
            .map(|point| [point[0] * scale_x - margin, point[1] * scale_y - margin])
            .collect::<Vec<_>>();
        let pad = offset * (scale_x + scale_y) / 2.0;

        let (mut min_x, mut max_x) = (f32::INFINITY, f32::NEG_INFINITY);
        let (mut min_y, mut max_y) = (f32::INFINITY, f32::NEG_INFINITY);
        for point in &hull {
            min_x = min_x.min(point[0]);
            max_x = max_x.max(point[0]);
            min_y = min_y.min(point[1]);
            max_y = max_y.max(point[1]);
        }
        let (min_x, max_x) = ((min_x - pad).max(0.0), (max_x + pad).min(width));
        let (min_y, max_y) = ((min_y - pad).max(0.0), (max_y + pad).min(height));
        if max_x - min_x < MIN_BOX_SIDE || max_y - min_y < MIN_BOX_SIDE {
            continue;
        }

        let Some(corners) = grown_min_area_rect(&hull, pad, BOX_GROWTH) else {
            continue;
        };
        let corners = corners.map(|corner| [corner[0].clamp(0.0, width), corner[1].clamp(0.0, height)]);
        boxes.push(LineBox::from_corners(corners, score));
    }
    boxes
}

/// Area (shoelace) and perimeter of the closed polygon through `points`; `None`
/// when either is not positive, which the demo also discards.
fn area_and_perimeter(points: &[[f32; 2]]) -> Option<(f32, f32)> {
    let mut twice_area = 0.0f32;
    let mut perimeter = 0.0f32;
    for index in 0..points.len() {
        let current = points[index];
        let next = points[(index + 1) % points.len()];
        twice_area += current[0] * next[1] - current[1] * next[0];
        perimeter += (next[0] - current[0]).hypot(next[1] - current[1]);
    }
    let area = twice_area.abs() / 2.0;
    (area > 0.0 && perimeter > 0.0).then_some((area, perimeter))
}

fn cross(origin: [f32; 2], a: [f32; 2], b: [f32; 2]) -> f32 {
    (a[0] - origin[0]) * (b[1] - origin[1]) - (a[1] - origin[1]) * (b[0] - origin[0])
}

fn convex_hull(points: &[[f32; 2]]) -> Vec<[f32; 2]> {
    let mut points = points.to_vec();
    points.sort_by(|a, b| a[0].total_cmp(&b[0]).then_with(|| a[1].total_cmp(&b[1])));
    points.dedup();
    if points.len() <= 2 {
        return points;
    }
    let mut lower: Vec<[f32; 2]> = Vec::new();
    for &point in &points {
        while lower.len() >= 2
            && cross(lower[lower.len() - 2], lower[lower.len() - 1], point) <= 0.0
        {
            lower.pop();
        }
        lower.push(point);
    }
    let mut upper: Vec<[f32; 2]> = Vec::new();
    for &point in points.iter().rev() {
        while upper.len() >= 2
            && cross(upper[upper.len() - 2], upper[upper.len() - 1], point) <= 0.0
        {
            upper.pop();
        }
        upper.push(point);
    }
    lower.pop();
    upper.pop();
    lower.extend(upper);
    lower
}

/// The minimum-area rectangle around `hull` grown by `pad` on every side (the area
/// is judged after growing), then pushed out by `growth` more pixels per side.
/// Corners are returned around the rectangle's centre.
fn grown_min_area_rect(hull: &[[f32; 2]], pad: f32, growth: f32) -> Option<[[f32; 2]; 4]> {
    if hull.is_empty() {
        return None;
    }
    // One frame per hull edge; the axes cover hulls with fewer than two points.
    let mut frames = vec![[1.0f32, 0.0]];
    for index in 0..hull.len() {
        let (from, to) = (hull[index], hull[(index + 1) % hull.len()]);
        let length = (to[0] - from[0]).hypot(to[1] - from[1]);
        if length > 1e-6 {
            frames.push([(to[0] - from[0]) / length, (to[1] - from[1]) / length]);
        }
    }

    // area, direction along the first side, direction along the second side, extents
    let mut best: Option<(f32, [f32; 2], [f32; 2], [f32; 4])> = None;
    for u in frames {
        let v = [-u[1], u[0]];
        let (mut min_a, mut max_a) = (f32::INFINITY, f32::NEG_INFINITY);
        let (mut min_b, mut max_b) = (f32::INFINITY, f32::NEG_INFINITY);
        for point in hull {
            let a = point[0] * u[0] + point[1] * u[1];
            let b = point[0] * v[0] + point[1] * v[1];
            min_a = min_a.min(a);
            max_a = max_a.max(a);
            min_b = min_b.min(b);
            max_b = max_b.max(b);
        }
        let area = (max_a - min_a + 2.0 * pad) * (max_b - min_b + 2.0 * pad);
        let better = match &best {
            Some((smallest, ..)) => area < *smallest,
            None => true,
        };
        if better {
            best = Some((area, u, v, [min_a, max_a, min_b, max_b]));
        }
    }

    let (_, u, v, [min_a, max_a, min_b, max_b]) = best?;
    let centre = [
        u[0] * (min_a + max_a) / 2.0 + v[0] * (min_b + max_b) / 2.0,
        u[1] * (min_a + max_a) / 2.0 + v[1] * (min_b + max_b) / 2.0,
    ];
    let half_a = (max_a - min_a) / 2.0 + pad + growth;
    let half_b = (max_b - min_b) / 2.0 + pad + growth;
    let corner = |sign_a: f32, sign_b: f32| {
        [
            centre[0] + sign_a * half_a * u[0] + sign_b * half_b * v[0],
            centre[1] + sign_a * half_a * u[1] + sign_b * half_b * v[1],
        ]
    };
    Some([
        corner(-1.0, -1.0),
        corner(1.0, -1.0),
        corner(1.0, 1.0),
        corner(-1.0, 1.0),
    ])
}

#[cfg(test)]
mod tests {
    use super::*;

    const PARAMS: DbParams = DbParams {
        threshold: 0.2,
        box_threshold: 0.45,
        unclip_ratio: 1.6,
    };

    fn layout() -> MapLayout {
        // The detector input is the padded image at its own size.
        MapLayout {
            map_width: 160,
            map_height: 96,
            padded_width: 160,
            padded_height: 96,
            margin: 16,
            width: 128,
            height: 64,
        }
    }

    fn map_with(blob: impl Fn(usize, usize) -> bool) -> Vec<f32> {
        (0..96)
            .flat_map(|y| (0..160).map(move |x| (x, y)))
            .map(|(x, y)| if blob(x, y) { 0.9 } else { 0.0 })
            .collect()
    }

    #[test]
    fn a_rectangle_grows_by_the_offset_and_four_pixels() {
        let hull = [[10.0, 10.0], [110.0, 10.0], [110.0, 40.0], [10.0, 40.0]];
        let corners = grown_min_area_rect(&hull, 0.0, 2.0).unwrap();
        let expected = [[8.0, 8.0], [112.0, 8.0], [112.0, 42.0], [8.0, 42.0]];
        for (corner, expected) in corners.iter().zip(expected) {
            assert!((corner[0] - expected[0]).abs() < 1e-3, "{corners:?}");
            assert!((corner[1] - expected[1]).abs() < 1e-3, "{corners:?}");
        }
    }

    #[test]
    fn a_bright_rectangle_becomes_one_line_box() {
        let prediction = map_with(|x, y| (40..120).contains(&x) && (36..52).contains(&y));
        let boxes = line_boxes(&prediction, &layout(), PARAMS);
        assert_eq!(boxes.len(), 1);
        let line = &boxes[0];
        // Contour points run over pixel centres: x 40..=119, y 36..=51, minus the
        // 16 pixel margin. The offset is area * 1.6 / perimeter = 10.09 on each side.
        assert!((line.cx - 63.5).abs() < 1.0 && (line.cy - 27.5).abs() < 1.0, "{line:?}");
        assert!(((line.xmax - line.xmin) - 103.2).abs() < 1.5, "{line:?}");
        assert!(((line.ymax - line.ymin) - 39.2).abs() < 1.5, "{line:?}");
        assert!(line.score > 0.8);
    }

    #[test]
    fn a_speck_and_a_dim_region_are_ignored() {
        let speck = map_with(|x, y| (60..63).contains(&x) && (40..43).contains(&y));
        assert!(line_boxes(&speck, &layout(), PARAMS).is_empty());
        let dim = vec![0.1f32; 160 * 96];
        assert!(line_boxes(&dim, &layout(), PARAMS).is_empty());
    }
}
