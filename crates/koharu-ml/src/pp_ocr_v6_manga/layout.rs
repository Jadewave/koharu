//! Line boxes and reading order.
//!
//! A port of the reference demo's block-aware ordering (`sort_reading_order_v2`):
//! lines that touch are grouped into blocks (balloons), the lines of each block are
//! read in columns (vertical text, right to left) or rows (horizontal text), and
//! blocks are ordered on the page.

/// A detected text line: its four corners and the axis-aligned extent.
#[derive(Debug, Clone)]
pub(super) struct LineBox {
    pub xmin: f32,
    pub xmax: f32,
    pub ymin: f32,
    pub ymax: f32,
    pub cx: f32,
    pub cy: f32,
    pub score: f32,
    pub corners: [[f32; 2]; 4],
}

impl LineBox {
    pub(super) fn from_corners(corners: [[f32; 2]; 4], score: f32) -> Self {
        let xs = corners.map(|corner| corner[0]);
        let ys = corners.map(|corner| corner[1]);
        let xmin = xs.into_iter().fold(f32::INFINITY, f32::min);
        let xmax = xs.into_iter().fold(f32::NEG_INFINITY, f32::max);
        let ymin = ys.into_iter().fold(f32::INFINITY, f32::min);
        let ymax = ys.into_iter().fold(f32::NEG_INFINITY, f32::max);
        Self {
            xmin,
            xmax,
            ymin,
            ymax,
            cx: (xmin + xmax) / 2.0,
            cy: (ymin + ymax) / 2.0,
            score,
            corners,
        }
    }

    fn width(&self) -> f32 {
        self.xmax - self.xmin
    }

    fn height(&self) -> f32 {
        self.ymax - self.ymin
    }

    fn vertical(&self) -> bool {
        self.height() > self.width()
    }

    /// Size across the reading direction.
    fn thickness(&self) -> f32 {
        if self.vertical() {
            self.width()
        } else {
            self.height()
        }
    }
}

/// Whether the page is read as vertical text, by area-weighted vote of its lines.
/// Nearly square boxes count for half their area on the side the page leans to.
pub(super) fn page_is_vertical(boxes: &[LineBox], page_width: f32, page_height: f32) -> bool {
    let (mut vertical, mut horizontal) = (0.0f32, 0.0f32);
    for line in boxes {
        let (width, height) = (line.width(), line.height());
        let area = width * height;
        if height > width * 1.1 {
            vertical += area;
        } else if width > height * 1.1 {
            horizontal += area;
        } else if page_height >= page_width {
            vertical += area * 0.5;
        } else {
            horizontal += area * 0.5;
        }
    }
    vertical >= horizontal
}

/// Lines closer than 0.6 of the thinner one's thickness belong to the same block.
const BLOCK_GROW: f32 = 0.6;

fn touches(a: &LineBox, b: &LineBox) -> bool {
    let gap = BLOCK_GROW * a.thickness().min(b.thickness());
    !(a.xmax + gap < b.xmin
        || b.xmax + gap < a.xmin
        || a.ymax + gap < b.ymin
        || b.ymax + gap < a.ymin)
}

fn root(parent: &mut [usize], mut index: usize) -> usize {
    while parent[index] != index {
        parent[index] = parent[parent[index]];
        index = parent[index];
    }
    index
}

/// A column (vertical text) or row (horizontal text) of lines.
struct Strip {
    lines: Vec<LineBox>,
    low: f32,
    high: f32,
}

/// Orders the lines of one block: right-to-left columns of top-to-bottom lines for
/// vertical text, top-to-bottom rows of left-to-right lines otherwise.
fn order_block(mut lines: Vec<LineBox>, vertical: bool) -> Vec<LineBox> {
    if vertical {
        lines.sort_by(|a, b| b.cx.total_cmp(&a.cx));
    } else {
        lines.sort_by(|a, b| a.cy.total_cmp(&b.cy));
    }
    let mut strips: Vec<Strip> = Vec::new();
    for line in lines {
        let (low, high) = if vertical {
            (line.xmin, line.xmax)
        } else {
            (line.ymin, line.ymax)
        };
        let found = strips.iter().position(|strip| {
            high.min(strip.high) - low.max(strip.low)
                > 0.5 * (high - low).min(strip.high - strip.low)
        });
        match found {
            Some(index) => {
                let strip = &mut strips[index];
                strip.low = strip.low.min(low);
                strip.high = strip.high.max(high);
                strip.lines.push(line);
            }
            None => strips.push(Strip {
                lines: vec![line],
                low,
                high,
            }),
        }
    }
    if vertical {
        strips.sort_by(|a, b| (b.low + b.high).total_cmp(&(a.low + a.high)));
    } else {
        strips.sort_by(|a, b| (a.low + a.high).total_cmp(&(b.low + b.high)));
    }
    let mut ordered = Vec::new();
    for mut strip in strips {
        if vertical {
            strip.lines.sort_by(|a, b| a.ymin.total_cmp(&b.ymin));
        } else {
            strip.lines.sort_by(|a, b| a.xmin.total_cmp(&b.xmin));
        }
        ordered.append(&mut strip.lines);
    }
    ordered
}

struct Block {
    lines: Vec<LineBox>,
    xmin: f32,
    xmax: f32,
    ymin: f32,
    ymax: f32,
}

struct Band {
    blocks: Vec<Block>,
    ymin: f32,
    ymax: f32,
}

/// Puts the detected lines in reading order.
pub(super) fn reading_order(boxes: Vec<LineBox>, page_vertical: bool) -> Vec<LineBox> {
    if boxes.is_empty() {
        return boxes;
    }
    let count = boxes.len();
    let mut parent = (0..count).collect::<Vec<_>>();
    for i in 0..count {
        for j in i + 1..count {
            if boxes[i].vertical() == boxes[j].vertical() && touches(&boxes[i], &boxes[j]) {
                let (root_i, root_j) = (root(&mut parent, i), root(&mut parent, j));
                parent[root_i] = root_j;
            }
        }
    }

    // Blocks keep the order in which their first line appears.
    let mut groups: Vec<(usize, Vec<LineBox>)> = Vec::new();
    for (index, line) in boxes.into_iter().enumerate() {
        let key = root(&mut parent, index);
        match groups.iter_mut().find(|(existing, _)| *existing == key) {
            Some((_, lines)) => lines.push(line),
            None => groups.push((key, vec![line])),
        }
    }

    let mut blocks = Vec::with_capacity(groups.len());
    for (_, lines) in groups {
        let xmin = lines.iter().map(|line| line.xmin).fold(f32::INFINITY, f32::min);
        let xmax = lines.iter().map(|line| line.xmax).fold(f32::NEG_INFINITY, f32::max);
        let ymin = lines.iter().map(|line| line.ymin).fold(f32::INFINITY, f32::min);
        let ymax = lines.iter().map(|line| line.ymax).fold(f32::NEG_INFINITY, f32::max);
        let vertical = if lines.len() > 1 {
            lines.iter().filter(|line| line.vertical()).count() * 2 >= lines.len()
        } else if (lines[0].height() - lines[0].width()).abs() > 4.0 {
            lines[0].vertical()
        } else {
            page_vertical
        };
        blocks.push(Block {
            lines: order_block(lines, vertical),
            xmin,
            xmax,
            ymin,
            ymax,
        });
    }

    // Page order: bands of blocks that overlap vertically, top to bottom; inside a
    // band right to left for vertical pages and left to right otherwise.
    blocks.sort_by(|a, b| a.ymin.total_cmp(&b.ymin));
    let mut bands: Vec<Band> = Vec::new();
    for block in blocks {
        let found = bands.iter().position(|band| {
            block.ymax.min(band.ymax) - block.ymin.max(band.ymin)
                > 0.3 * (block.ymax - block.ymin).min(band.ymax - band.ymin)
        });
        match found {
            Some(index) => {
                let band = &mut bands[index];
                band.ymin = band.ymin.min(block.ymin);
                band.ymax = band.ymax.max(block.ymax);
                band.blocks.push(block);
            }
            None => bands.push(Band {
                ymin: block.ymin,
                ymax: block.ymax,
                blocks: vec![block],
            }),
        }
    }
    bands.sort_by(|a, b| a.ymin.total_cmp(&b.ymin));

    let mut ordered = Vec::new();
    for mut band in bands {
        band.blocks.sort_by(|a, b| {
            let (a, b) = (a.xmin + a.xmax, b.xmin + b.xmax);
            if page_vertical {
                b.total_cmp(&a)
            } else {
                a.total_cmp(&b)
            }
        });
        for mut block in band.blocks {
            ordered.append(&mut block.lines);
        }
    }
    ordered
}

fn is_cjk(character: char) -> bool {
    matches!(
        character as u32,
        0x2E80..=0x9FFF | 0xAC00..=0xD7AF | 0xF900..=0xFAFF | 0xFE30..=0xFE4F
            | 0xFF00..=0xFFEF | 0x20000..=0x2FA1F
    )
}

/// Joins recognized lines. CJK text is concatenated; a space separates two lines
/// only when both sides of the break are non-CJK (words of a Latin-script balloon).
pub(super) fn join_lines(lines: &[String]) -> String {
    let mut text = String::new();
    for line in lines {
        let next = line.trim();
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

    fn line(xmin: f32, ymin: f32, xmax: f32, ymax: f32) -> LineBox {
        LineBox::from_corners([[xmin, ymin], [xmax, ymin], [xmax, ymax], [xmin, ymax]], 1.0)
    }

    fn centers(lines: &[LineBox]) -> Vec<(i32, i32)> {
        lines.iter().map(|l| (l.cx as i32, l.cy as i32)).collect()
    }

    #[test]
    fn vertical_columns_are_read_right_to_left() {
        let boxes = vec![
            line(30.0, 90.0, 50.0, 190.0),
            line(190.0, 100.0, 210.0, 200.0),
            line(110.0, 80.0, 130.0, 180.0),
        ];
        assert!(page_is_vertical(&boxes, 300.0, 400.0));
        let ordered = reading_order(boxes, true);
        assert_eq!(centers(&ordered), [(200, 150), (120, 130), (40, 140)]);
    }

    #[test]
    fn horizontal_rows_are_read_top_to_bottom_and_left_to_right() {
        let boxes = vec![
            line(10.0, 100.0, 210.0, 130.0),
            line(10.0, 20.0, 110.0, 50.0),
            line(130.0, 22.0, 230.0, 52.0),
        ];
        assert!(!page_is_vertical(&boxes, 400.0, 200.0));
        let ordered = reading_order(boxes, false);
        assert_eq!(centers(&ordered), [(60, 35), (180, 37), (110, 115)]);
    }

    #[test]
    fn distant_balloons_are_ordered_on_the_page() {
        // Two horizontal balloons far apart: the upper one is read first.
        let boxes = vec![
            line(10.0, 400.0, 210.0, 430.0),
            line(10.0, 20.0, 210.0, 50.0),
        ];
        let ordered = reading_order(boxes, false);
        assert_eq!(centers(&ordered), [(110, 35), (110, 415)]);
    }

    #[test]
    fn lines_are_joined_by_script() {
        let join = |lines: &[&str]| {
            join_lines(&lines.iter().map(|line| (*line).to_owned()).collect::<Vec<_>>())
        };
        assert_eq!(join(&["hello", "world"]), "hello world");
        assert_eq!(join(&["第一行，", "第二行"]), "第一行，第二行");
        assert_eq!(join(&["", " ", "x"]), "x");
    }
}
