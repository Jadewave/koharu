//! DB contour scoring: the mean of the probability map inside a polygon, rasterised the way
//! OpenCV's integer `fillPoly` does it. Ported from the PP-OCRv6 medium detector.

pub(super) fn polygon_score(
    map: &[f32],
    width: u32,
    height: u32,
    polygon: &[[f32; 2]],
) -> f32 {
    // Transformers passes an int32 copy of the rotated box to cv2.fillPoly.
    let polygon = polygon
        .iter()
        .map(|point| [point[0] as i32, point[1] as i32])
        .collect::<Vec<_>>();
    let min_x = polygon
        .iter()
        .map(|point| point[0])
        .min()
        .unwrap_or_default()
        .clamp(0, width.saturating_sub(1) as i32);
    let max_x = polygon
        .iter()
        .map(|point| point[0])
        .max()
        .unwrap_or_default()
        .clamp(0, width.saturating_sub(1) as i32);
    let min_y = polygon
        .iter()
        .map(|point| point[1])
        .min()
        .unwrap_or_default()
        .clamp(0, height.saturating_sub(1) as i32);
    let max_y = polygon
        .iter()
        .map(|point| point[1])
        .max()
        .unwrap_or_default()
        .clamp(0, height.saturating_sub(1) as i32);
    if min_x > max_x || min_y > max_y {
        return 0.0;
    }

    let mask_width = (max_x - min_x + 1) as usize;
    let mask_height = (max_y - min_y + 1) as usize;
    let local_polygon = polygon
        .iter()
        .map(|point| [point[0] - min_x, point[1] - min_y])
        .collect::<Vec<_>>();
    let mask = opencv_fill_polygon(mask_width, mask_height, &local_polygon);
    let mut sum = 0.0f64;
    let mut count = 0usize;
    for y in 0..mask_height {
        for x in 0..mask_width {
            if mask[y * mask_width + x] {
                let map_x = min_x as usize + x;
                let map_y = min_y as usize + y;
                sum += map[map_y * width as usize + map_x] as f64;
                count += 1;
            }
        }
    }
    if count == 0 {
        0.0
    } else {
        (sum / count as f64) as f32
    }
}

#[derive(Clone, Copy)]
struct PolygonEdge {
    y0: i32,
    y1: i32,
    x: i64,
    dx: i64,
}

// Port of OpenCV's integer fillPoly scan conversion. OpenCV first draws every
// polygon edge and then fills inclusive spans between fixed-point edge pairs.
// https://github.com/opencv/opencv/blob/4.13.0/modules/imgproc/src/drawing.cpp#L1262-L1493
fn opencv_fill_polygon(width: usize, height: usize, polygon: &[[i32; 2]]) -> Vec<bool> {
    const XY_SHIFT: u32 = 16;
    const XY_ONE: i64 = 1 << XY_SHIFT;

    let mut mask = vec![false; width * height];
    if polygon.is_empty() {
        return mask;
    }

    let mut edges = Vec::with_capacity(polygon.len());
    let mut previous = polygon[polygon.len() - 1];
    for &current in polygon {
        draw_line_8_connected(&mut mask, width, height, previous, current);
        if previous[1] != current[1] {
            let dx =
                ((current[0] - previous[0]) as i64 * XY_ONE) / (current[1] - previous[1]) as i64;
            let (y0, y1, x) = if previous[1] < current[1] {
                (previous[1], current[1], previous[0] as i64 * XY_ONE)
            } else {
                (current[1], previous[1], current[0] as i64 * XY_ONE)
            };
            edges.push(PolygonEdge { y0, y1, x, dx });
        }
        previous = current;
    }
    if edges.len() < 2 {
        return mask;
    }

    edges.sort_by_key(|edge| (edge.y0, edge.x, edge.dx));
    let y_min = edges.iter().map(|edge| edge.y0).min().unwrap_or_default();
    let y_max = edges.iter().map(|edge| edge.y1).max().unwrap_or_default();
    let mut active = Vec::<PolygonEdge>::new();
    for y in y_min..y_max.min(height as i32) {
        active.retain(|edge| edge.y1 != y);
        active.extend(edges.iter().filter(|edge| edge.y0 == y).copied());
        active.sort_by_key(|edge| edge.x);

        for pair in active.as_chunks::<2>().0 {
            let left = pair[0].x.min(pair[1].x);
            let right = pair[0].x.max(pair[1].x);
            let x1 = (left + XY_ONE - 1) >> XY_SHIFT;
            let x2 = right >> XY_SHIFT;
            if y >= 0 && x1 < width as i64 && x2 >= 0 {
                let x1 = x1.max(0) as usize;
                let x2 = x2.min(width as i64 - 1) as usize;
                for x in x1..=x2 {
                    mask[y as usize * width + x] = true;
                }
            }
        }
        for edge in &mut active {
            edge.x += edge.dx;
        }
    }
    mask
}

fn draw_line_8_connected(
    mask: &mut [bool],
    width: usize,
    height: usize,
    [mut x0, mut y0]: [i32; 2],
    [x1, y1]: [i32; 2],
) {
    // cv::Line uses LineIterator with 8-connectivity and leftToRight=true.
    // https://github.com/opencv/opencv/blob/4.13.0/modules/imgproc/src/drawing.cpp#L159-L260
    let mut delta_x = 1;
    let mut delta_y = 1;
    let mut dx = x1 - x0;
    let mut dy = y1 - y0;
    if dx < 0 {
        dx = -dx;
        dy = -dy;
        x0 = x1;
        y0 = y1;
    }
    if dy < 0 {
        dy = -dy;
        delta_y = -1;
    }
    let vertical = dy > dx;
    if vertical {
        std::mem::swap(&mut dx, &mut dy);
        std::mem::swap(&mut delta_x, &mut delta_y);
    }

    let mut error = dx - 2 * dy;
    let plus_delta = 2 * dx;
    let minus_delta = -2 * dy;
    let mut minus_x = delta_x;
    let mut plus_x = 0;
    let mut minus_y = 0;
    let mut plus_y = delta_y;
    if vertical {
        std::mem::swap(&mut plus_y, &mut plus_x);
        std::mem::swap(&mut minus_y, &mut minus_x);
    }

    for _ in 0..=dx {
        if x0 >= 0 && y0 >= 0 && x0 < width as i32 && y0 < height as i32 {
            mask[y0 as usize * width + x0 as usize] = true;
        }
        let take_plus = error < 0;
        error += minus_delta + if take_plus { plus_delta } else { 0 };
        x0 += minus_x + if take_plus { plus_x } else { 0 };
        y0 += minus_y + if take_plus { plus_y } else { 0 };
    }
}

