use std::{
    collections::BTreeMap,
    sync::{Arc, Mutex},
};

use anyhow::{Result, bail};
use async_trait::async_trait;
use image::{DynamicImage, Rgba, RgbaImage};
use imageproc::{drawing::draw_filled_rect_mut, rect::Rect};
use koharu_scene::{
    Authored, EntityId, Geometry, LanguageTag, Origin, SourceText, Translation,
};
use koharu_translator::{Provider, TranslationRequest, Translator, needs_translation};

use crate::TranslationConfig;

use super::{StageInput, StageProcessor, finish, generation};

const PRODUCER: &str = "dev.koharu.pipeline.translation";

/// With a local vision model, up to this many consecutive pages are translated in one request,
/// so the model sees who spoke on the page before and keeps the pronouns it chose.
const CLUSTER_PAGES: usize = 3;
/// A request never carries more segments than this, which keeps its output within the limit.
const CLUSTER_MAX_SEGMENTS: usize = 40;

pub(super) struct Processor {
    config: TranslationConfig,
    translator: Translator,
    /// Translations that a multi-page request produced for the pages after its first one,
    /// by text content: the source they were made from, and the translation. Each page takes
    /// its own when its turn comes.
    pending: Mutex<BTreeMap<EntityId, (String, String)>>,
}

impl Processor {
    pub(super) fn new(config: TranslationConfig, translator: Translator) -> Self {
        Self {
            config,
            translator,
            pending: Mutex::new(BTreeMap::new()),
        }
    }

    fn pending(&self) -> std::sync::MutexGuard<'_, BTreeMap<EntityId, (String, String)>> {
        self.pending
            .lock()
            .unwrap_or_else(|error| error.into_inner())
    }

    /// True when a glossary is written for this model selection (local models only).
    pub(super) fn builds_glossary(&self) -> bool {
        self.config.model.provider == koharu_translator::Provider::Local
    }

    /// Lets the glossary model read the OCR text of the whole work before any page is
    /// translated.
    pub(super) async fn build_glossary(
        &self,
        project: &str,
        lines: Vec<String>,
        stop: impl Fn() -> bool + Send + Sync,
    ) -> Result<()> {
        // A new run: whatever an earlier run left over is stale.
        self.pending().clear();
        self.translator
            .build_glossary(
                &self.config.model,
                self.config.generation,
                self.config.target_language,
                project,
                lines,
                stop,
            )
            .await
    }
}

#[async_trait]
impl StageProcessor for Processor {
    fn model(&self) -> &'static str {
        Translator::model(&self.config.model)
    }

    fn unload(&self) -> bool {
        self.translator.unload()
    }

    async fn load(&self) -> Result<()> {
        self.translator.load_model(&self.config.model).await
    }

    async fn process(&self, input: StageInput) -> Result<koharu_scene::Patch> {
        let current = page_texts(&input, input.page, |entity| input.contains_entity(entity))?;

        // A page that came after the first of a multi-page request was translated with it.
        let mut translations = BTreeMap::new();
        let mut missing = false;
        {
            let mut pending = self.pending();
            for (content, source) in &current.targets {
                if !needs_translation(source) {
                    continue;
                }
                match pending.remove(content) {
                    Some((from, text)) if from == *source => {
                        translations.insert(*content, text);
                    }
                    _ => missing = true,
                }
            }
        }
        let mut provider: &'static str = self.config.model.provider.into();
        if missing {
            (provider, translations) = self.translate_pages(&input, &current).await?;
        }

        let language = LanguageTag::new(self.config.target_language.tag())?;
        let generated = generation(PRODUCER, provider)?;
        let mut edit = input.scene.edit_as(generated.clone());
        for (entity, _) in &current.targets {
            edit.observe::<SourceText>(*entity)?;
            edit.observe::<Translation>(*entity)?;
        }
        for (entity, source) in current.targets {
            if input
                .scene
                .component::<Translation>(entity)?
                .is_some_and(|value| matches!(value.text.origin, Origin::User))
            {
                continue;
            }
            let text = match translations.remove(&entity) {
                Some(text) => clean_translation(&source, &text),
                None => silent_text(&source)
                    .unwrap_or_else(|| clean_translation(&source, &source)),
            };
            edit.set(
                entity,
                &Translation {
                    text: Authored::generated(text, generated.clone()),
                    language: Some(language.clone()),
                },
            )?;
        }
        finish(edit)
    }
}

impl Processor {
    /// Translates the current page, together with the next ones when the local vision model
    /// can take them: the pages go to the model as one scene, each with its image. The
    /// translations of the later pages are kept in `pending`; those of the current page are
    /// returned.
    async fn translate_pages(
        &self,
        input: &StageInput,
        current: &PageTexts,
    ) -> Result<(&'static str, BTreeMap<EntityId, String>)> {
        let local = self.config.model.provider == Provider::Local;
        let mut pages = vec![current.clone()];
        let mut images: Vec<Arc<DynamicImage>> = Vec::new();
        if Translator::supports_vision(&self.config.model, &self.config.generation)
            && let Some(image) = input.images.get(&input.scene, input.page, "source").await?
        {
            images.push(image);
        }
        let with_images = !images.is_empty();
        if with_images
            && local
            && input.entities.is_none()
            && input.region.is_none()
            && let Some(order) = input.pages.as_deref()
            && let Some(position) = order.iter().position(|page| *page == input.page)
        {
            let mut total = current.translatable().len();
            for &page in &order[position + 1..] {
                if pages.len() >= CLUSTER_PAGES || total >= CLUSTER_MAX_SEGMENTS {
                    break;
                }
                let texts = page_texts(input, page, |entity| {
                    crate::scope::contains_entity(&input.scene, page, None, None, entity)
                })?;
                let count = texts.translatable().len();
                if count == 0 {
                    continue;
                }
                if total + count > CLUSTER_MAX_SEGMENTS {
                    break;
                }
                let Some(image) = input.images.get(&input.scene, page, "source").await? else {
                    break;
                };
                total += count;
                images.push(image);
                pages.push(texts);
            }
        }

        // Segments are numbered across the pages, and the outlines drawn on the images carry
        // the same numbers as the ids in the request.
        let mut segments = Vec::new();
        let mut segment_pages = Vec::new();
        let mut marked = Vec::new();
        for (index, texts) in pages.iter().enumerate() {
            let translatable = texts.translatable();
            if with_images {
                let boxes: Vec<_> = translatable.iter().map(|&target| texts.boxes[target]).collect();
                marked.push(Arc::new(mark_segments(&images[index], &boxes, segments.len())));
            }
            for &target in &translatable {
                segments.push(texts.targets[target].1.clone());
                segment_pages.push(index);
            }
        }
        let expected = segments.len();
        tracing::info!(
            pages = pages.len(),
            segments = expected,
            images = marked.len(),
            "translating pages together"
        );
        let mut request = TranslationRequest::new(segments, self.config.target_language);
        if let Some(instructions) = self.config.instructions.as_deref() {
            request = request.with_instructions(instructions);
        }
        if with_images {
            request = if local {
                request.with_pages(marked, segment_pages)
            } else if let Some(image) = marked.into_iter().next() {
                request.with_image(image)
            } else {
                request
            };
        }
        let (provider, translated) = self
            .translator
            .translate(&self.config.model, self.config.generation, request)
            .await?;
        if translated.len() != expected {
            bail!(
                "translator returned {} translations for {expected} segments",
                translated.len()
            );
        }

        let mut translated = translated.into_iter();
        let mut own = BTreeMap::new();
        let mut pending = self.pending();
        for (index, texts) in pages.iter().enumerate() {
            for target in texts.translatable() {
                let Some(text) = translated.next() else {
                    break;
                };
                let (content, source) = &texts.targets[target];
                if index == 0 {
                    own.insert(*content, text);
                } else {
                    pending.insert(*content, (source.clone(), text));
                }
            }
        }
        Ok((provider, own))
    }
}

/// The text layers of a page: their content, source text, and where each lies on the page.
#[derive(Clone)]
struct PageTexts {
    targets: Vec<(EntityId, String)>,
    boxes: Vec<Option<[f64; 4]>>,
}

impl PageTexts {
    /// Indices of the targets that have something to translate; the others (sound effects of
    /// dots, placeholders) are not sent to the model.
    fn translatable(&self) -> Vec<usize> {
        (0..self.targets.len())
            .filter(|&index| needs_translation(&self.targets[index].1))
            .collect()
    }
}

fn page_texts(
    input: &StageInput,
    page: EntityId,
    included: impl Fn(EntityId) -> Result<bool>,
) -> Result<PageTexts> {
    let mut texts = PageTexts {
        targets: Vec::new(),
        boxes: Vec::new(),
    };
    if let Some(group) = input.scene.page(page)?.text_group()? {
        for layer in group.text_layers()? {
            if !included(layer.id())? {
                continue;
            }
            let content = layer.content()?;
            let Some(source) = content.source()? else {
                continue;
            };
            if !source.text.value.trim().is_empty() {
                texts.targets.push((content.id(), source.text.value));
                texts.boxes.push(layer.frame()?.as_ref().and_then(bounds));
            }
        }
    }
    Ok(texts)
}

/// Characters that stand for a silence or a trailing off when a bubble holds nothing else:
/// dots of every kind and the placeholder glyphs OCR emits for them.
const SILENCE_CHARACTERS: [char; 17] = [
    '\u{2026}', '\u{22ef}', '\u{2025}', '.', '\u{30fb}', '\u{ff65}', '\u{b7}', '\u{3002}',
    '\u{ff0e}', '\u{3001}', '\u{ff64}', '\u{2610}', '\u{25a1}', '\u{25a2}', '\u{25a3}',
    '\u{25af}', '\u{fffd}',
];

/// A bubble that holds only dots or placeholders becomes a plain ellipsis, which every font
/// can draw, instead of being kept as the odd glyphs the OCR produced.
fn silent_text(source: &str) -> Option<String> {
    if needs_translation(source) {
        return None;
    }
    let marks: Vec<char> = source
        .chars()
        .filter(|character| !character.is_whitespace())
        .collect();
    let single_ellipsis = matches!(marks.as_slice(), ['\u{2026}' | '\u{22ef}' | '\u{2025}']);
    let all_silence = marks
        .iter()
        .all(|character| SILENCE_CHARACTERS.contains(character));
    (all_silence && (marks.len() >= 2 || single_ellipsis)).then(|| "\u{2026}".to_owned())
}

/// The left, top, right and bottom of a geometry, in page pixels.
fn bounds(geometry: &Geometry) -> Option<[f64; 4]> {
    let first = geometry.points.first()?;
    let mut bounds = [first.x, first.y, first.x, first.y];
    for point in &geometry.points {
        bounds[0] = bounds[0].min(point.x);
        bounds[1] = bounds[1].min(point.y);
        bounds[2] = bounds[2].max(point.x);
        bounds[3] = bounds[3].max(point.y);
    }
    Some(bounds)
}

/// 5x7 dot patterns of the digits, used to tag the boxes without needing a font.
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

/// A copy of the page with a red outline around every segment's box and a tag with the
/// segment id (`first_id` plus its position) above it, so a vision model can tell which bubble each line belongs to. The
/// outline is drawn outside the box, so the text itself stays readable.
fn mark_segments(
    image: &DynamicImage,
    boxes: &[Option<[f64; 4]>],
    first_id: usize,
) -> DynamicImage {
    let mut canvas: RgbaImage = image.to_rgba8();
    let (width, height) = canvas.dimensions();
    let thickness = (height / 500).clamp(2, 6) as i32;
    let dot = (height / 220).clamp(3, 10) as i32;
    let red = Rgba([220, 20, 20, 255]);
    let white = Rgba([255, 255, 255, 255]);
    let clamp_x = |value: f64| value.clamp(0.0, f64::from(width.saturating_sub(1))) as i32;
    let clamp_y = |value: f64| value.clamp(0.0, f64::from(height.saturating_sub(1))) as i32;
    for (offset, bounds) in boxes.iter().enumerate() {
        let id = first_id + offset;
        let Some([left, top, right, bottom]) = *bounds else {
            continue;
        };
        let (left, top) = (clamp_x(left), clamp_y(top));
        let (right, bottom) = (clamp_x(right).max(left + 1), clamp_y(bottom).max(top + 1));
        // Four bars around the box.
        let (box_width, box_height) = ((right - left) as u32, (bottom - top) as u32);
        let outer_width = box_width + 2 * thickness as u32;
        for (x, y, w, h) in [
            (left - thickness, top - thickness, outer_width, thickness as u32),
            (left - thickness, bottom, outer_width, thickness as u32),
            (left - thickness, top, thickness as u32, box_height),
            (right, top, thickness as u32, box_height),
        ] {
            draw_filled_rect_mut(&mut canvas, Rect::at(x, y).of_size(w, h), red);
        }
        // The tag sits above the box, or inside its top edge when there is no room above.
        let digits: Vec<usize> = id
            .to_string()
            .bytes()
            .map(|byte| usize::from(byte - b'0'))
            .collect();
        let padding = dot;
        let tag_height = 7 * dot + 2 * padding;
        let tag_width = digits.len() as i32 * 6 * dot - dot + 2 * padding;
        let tag_x = left - thickness;
        let tag_y = if top - thickness - tag_height >= 0 {
            top - thickness - tag_height
        } else {
            top
        };
        draw_filled_rect_mut(
            &mut canvas,
            Rect::at(tag_x, tag_y).of_size(tag_width as u32, tag_height as u32),
            red,
        );
        for (index, digit) in digits.into_iter().enumerate() {
            let origin_x = tag_x + padding + index as i32 * 6 * dot;
            for (row, pattern) in DIGITS[digit].iter().enumerate() {
                for (column, cell) in pattern.bytes().enumerate() {
                    if cell == b'1' {
                        draw_filled_rect_mut(
                            &mut canvas,
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
    DynamicImage::ImageRgba8(canvas)
}

const PARENTHESES: [char; 4] = ['(', ')', '\u{ff08}', '\u{ff09}'];

/// Drops `(...)` notes such as "(SFX name/action)" that models append. Only applies when the
/// source has no parentheses of its own, so genuine parenthesized dialogue is kept. Unbalanced
/// parentheses leave the text untouched.
fn strip_parenthetical(source: &str, text: &str) -> String {
    if source.contains(PARENTHESES) {
        return text.to_owned();
    }
    let mut result = String::with_capacity(text.len());
    let mut depth = 0usize;
    for character in text.chars() {
        match character {
            '(' | '\u{ff08}' => depth += 1,
            ')' | '\u{ff09}' if depth > 0 => depth -= 1,
            ')' | '\u{ff09}' => return text.to_owned(),
            _ if depth == 0 => result.push(character),
            _ => {}
        }
    }
    if depth == 0 { result } else { text.to_owned() }
}

/// Vertical-bar look-alikes that models emit as line separators.
const SEPARATOR_CHARACTERS: [char; 5] = ['|', '\u{ff5c}', '\u{2502}', '\u{00a6}', '\u{2016}'];

/// Removes artifacts local models leave in translations: separator bars used in place of
/// line breaks, control and zero-width characters, and replacement glyphs. Line breaks are
/// kept. A separator that already appears in the source text is left alone.
fn clean_translation(source: &str, text: &str) -> String {
    let stripped = strip_parenthetical(source, text);
    let text = if stripped.trim().is_empty() {
        text
    } else {
        stripped.as_str()
    };
    let strip_separators = !source.contains(SEPARATOR_CHARACTERS);
    let cleaned = text
        .chars()
        .filter_map(|character| match character {
            '\n' => Some('\n'),
            '\r' => None,
            character if strip_separators && SEPARATOR_CHARACTERS.contains(&character) => {
                Some(' ')
            }
            '\u{200b}' | '\u{200c}' | '\u{200d}' | '\u{feff}' | '\u{fffd}' => None,
            character if character.is_control() => Some(' '),
            character => Some(character),
        })
        .collect::<String>();
    let lines = cleaned
        .lines()
        .map(|line| line.split_whitespace().collect::<Vec<_>>().join(" "))
        .filter(|line| !line.is_empty())
        .collect::<Vec<_>>();
    if lines.is_empty() {
        text.to_owned()
    } else {
        lines.join("\n")
    }
}

#[cfg(test)]
mod tests {
    use image::{DynamicImage, GenericImageView, Rgba, RgbaImage};

    use super::{bounds, clean_translation, mark_segments, silent_text};

    #[test]
    fn segments_are_outlined_and_tagged_on_a_copy_of_the_page() {
        let white = Rgba([255, 255, 255, 255]);
        let page = DynamicImage::ImageRgba8(RgbaImage::from_pixel(400, 400, white));
        let marked = mark_segments(
            &page,
            &[Some([100.0, 100.0, 200.0, 200.0]), None, Some([150.0, 0.0, 250.0, 60.0])],
            0,
        );
        assert_eq!(marked.dimensions(), (400, 400));
        // The outline sits just outside the box and the box itself is untouched.
        assert_ne!(marked.get_pixel(100, 99), white);
        assert_ne!(marked.get_pixel(99, 150), white);
        assert_eq!(marked.get_pixel(150, 150), white);
        // The original page is not modified.
        assert_eq!(page.get_pixel(100, 99), white);
        // A box at the top edge is still tagged, and unknown boxes are skipped.
        assert_ne!(marked.get_pixel(150, 1), white);
        assert_eq!(marked.get_pixel(350, 350), white);
    }

    #[test]
    fn a_box_outside_the_page_does_not_panic() {
        let page = DynamicImage::ImageRgba8(RgbaImage::new(50, 50));
        let marked = mark_segments(
            &page,
            &[Some([-30.0, -30.0, 500.0, 500.0]), Some([10.0, 10.0, 10.0, 10.0])],
            0,
        );
        assert_eq!(marked.dimensions(), (50, 50));
    }

    #[test]
    fn tags_continue_the_numbering_of_the_pages_before() {
        let white = Rgba([255, 255, 255, 255]);
        let page = DynamicImage::ImageRgba8(RgbaImage::from_pixel(400, 400, white));
        let boxes = [Some([100.0, 100.0, 200.0, 200.0])];
        let first = mark_segments(&page, &boxes, 0);
        let later = mark_segments(&page, &boxes, 12);
        // The outline is the same, the tag differs: 12 has two digits and is wider than 0.
        assert_ne!(first, later);
        assert_ne!(later.get_pixel(100, 99), white);
    }

    #[test]
    fn dots_and_placeholders_become_an_ellipsis() {
        let ellipsis = Some("\u{2026}".to_owned());
        assert_eq!(silent_text("\u{2026}"), ellipsis);
        assert_eq!(silent_text("\u{2026}\u{2026}"), ellipsis);
        assert_eq!(silent_text("\u{30fb}\u{30fb}\u{30fb}\u{30fb}\u{30fb}\u{30fb}"), ellipsis);
        assert_eq!(silent_text("\u{25af}\u{25af}\u{25af}"), ellipsis);
        assert_eq!(silent_text("...."), ellipsis);
        // A single dot, real text and other punctuation stay as they are.
        assert_eq!(silent_text("."), None);
        assert_eq!(silent_text("\u{3002}"), None);
        assert_eq!(silent_text("Xin chào"), None);
        assert_eq!(silent_text("?!"), None);
    }

    #[test]
    fn bounds_cover_every_point() {
        let geometry = koharu_scene::Geometry::rectangle(10.0, 20.0, 30.0, 40.0);
        assert_eq!(bounds(&geometry), Some([10.0, 20.0, 40.0, 60.0]));
    }

    #[test]
    fn separator_bars_become_spaces_and_line_breaks_survive() {
        assert_eq!(
            clean_translation("a\nb", "Tối từng\n| cô ấy | lạ"),
            "Tối từng\ncô ấy lạ"
        );
    }

    #[test]
    fn parenthetical_notes_are_removed_unless_the_source_has_parentheses() {
        assert_eq!(clean_translation("ポイ", "Vứt\n(tên SFX/hành động-thả)"), "Vứt");
        assert_eq!(clean_translation("（笑）", "(cười)"), "(cười)");
        assert_eq!(clean_translation("ポイ", "(ghi chú)"), "(ghi chú)");
        assert_eq!(clean_translation("ポイ", "Vứt (chưa đóng"), "Vứt (chưa đóng");
    }

    #[test]
    fn separators_in_the_source_are_preserved() {
        assert_eq!(clean_translation("a | b", "x | y"), "x | y");
    }

    #[test]
    fn empty_result_falls_back_to_the_original_text() {
        assert_eq!(clean_translation("a", "|"), "|");
    }
        }
