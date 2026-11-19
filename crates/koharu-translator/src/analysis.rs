//! Reading a work with a vision model before it is translated.
//!
//! A vision model on an OpenAI-compatible server (the one of `providers.openai_compatible`)
//! looks at every page together with the text OCR read from it. It tells the translation model,
//! which never sees the pictures, everything that depends on them: who speaks to whom, how
//! old they are and how they talk, which words they use for themselves and for each other,
//! and where OCR misread the text. The result is a glossary of the whole work (merged into the
//! glossary every model follows) and, for every page, the notes the translation model reads
//! with that page.
//!
//! The models are tried in the order of `analysis.models`: when one fails or refuses a page,
//! the next one reads it. A model that the server does not know or does not let the user call
//! is skipped for the rest of the run. A page that no model could read is translated the way
//! it was before, without notes.
//!
//! The pages are sent in requests of `analysis.pages_per_request` pictures (twelve by default).
//! A picture is the page with its lettering inpainted away and the OCR lines marked by numbered
//! red boxes, compressed without loss (lossless WebP). The first request carries twelve new
//! pages. Every later one carries the last `analysis.overlap_pages` pages of the request before
//! (two by default: read already, only there to show where the story left off, with what was
//! decided for their lines), the story so far, what is known of the characters, and new pages
//! for the rest. The model thinks before it answers and decides, for every line, who speaks,
//! who listens and which words they use for "I" and "you". Pages that a model leaves out of its
//! answer go to the next model of the chain, and a single page may be read with its panels (see
//! `pages_per_request`).
//!
//! The model that translates afterwards (see `polish`) is not told the story: it follows the
//! decisions for every line and the draft.

use std::{collections::HashSet, fmt::Write as _, sync::Arc};

use anyhow::{Context, anyhow, ensure};
use image::DynamicImage;
use serde::{Deserialize, Deserializer, Serialize};
use serde_json::{Value, json};

mod settle;

use crate::{
    Error, GenerationConfig, Language, ProvidersConfig, Translator,
    backend::{ExpectedPair, GlossaryEntry, encode_lossless_webp, shrink_to},
    remote,
};

/// The configuration section: `[analysis]`.
const CONFIG_SECTION: &str = "analysis";

const MAX_CHARACTERS: usize = 40;
const MAX_TERMS: usize = 200;
const FIELD_CHARS: usize = 160;
/// The profile of a character in the glossary: its age, its manner, its place in the story.
const PROFILE_CHARS: usize = 240;
/// What the vision model is told it knows already is cut to this many characters.
const BIBLE_PROMPT_CHARS: usize = 6000;
/// The notes of one page are cut to this many characters. A line the notes do not reach is
/// translated by the model's own guess, which is where the pronouns drift, so the limit is
/// generous: a crowded page still has a note for every line.
const INSTRUCTIONS_CHARS: usize = 9000;

/// `[analysis]` of the configuration file.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct AnalysisConfig {
    /// Reads the work with the vision models below when the OpenAI-compatible provider has
    /// a server of its own (not the local default). Set to false to turn it off.
    pub enabled: bool,
    /// The models that read the pages, in the order they are tried for every page.
    pub models: Vec<AnalysisModel>,
    /// Translates every page again after the analysis (see `polish`): the model chosen for the
    /// translation (the local one) writes the final translation from the notes, the draft of
    /// the vision model and the story of the work. Off: the draft of the vision model is the
    /// translation.
    pub translate: bool,
    /// The size of the story so far sent with every request after the first, in characters.
    pub story_chars: usize,
    /// How many pictures one request carries, one for each page. One page per request is the
    /// only way a page is sent with its panels.
    pub pages_per_request: usize,
    /// How many of those pictures, from the second request on, are the last pages of the
    /// request before: they were read already and are only there for the model to see where
    /// the story left off. The rest of the request is new pages.
    pub overlap_pages: usize,
    /// Lets the models think while they read, when the model does not say it itself. Slower
    /// and it needs `max_tokens` to be large.
    pub reasoning: bool,
    /// The most tokens the answer to one request may take: it holds the notes and the draft of
    /// every line of every page of the batch.
    pub max_tokens: u32,
    /// The longest side of the page picture that is sent, in pixels. The picture is shrunk to
    /// this and then compressed without loss (lossless WebP).
    pub image_max_dimension: u32,
    /// The longest side of a panel picture that is sent, in pixels.
    pub panel_max_dimension: u32,
    /// The order panels are read in: from the right of a row to its left (Japanese manga), or
    /// from the left to the right.
    pub right_to_left: bool,
}

/// The vision model of the analysis.
const NEMOTRON_VL: &str = "nvidia/nemotron-3.5-super-vl-120b-a12b";
/// The model that reads a batch when the vision model fails or leaves pages out of its answer.
const NEMOTRON_BOLT: &str = "nvidia/nemotron-3.5-bolt-e4b";

/// A model of the analysis chain.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct AnalysisModel {
    pub model: String,
    /// Whether the model is sent the page and its panels. Set it to false for a model without
    /// vision, which then only reads the text.
    #[serde(default = "yes")]
    pub vision: bool,
    /// Lets this model think while it reads a page. Unset: follows `[analysis] reasoning`.
    #[serde(default)]
    pub reasoning: Option<bool>,
}

const fn yes() -> bool {
    true
}

impl Default for AnalysisConfig {
    fn default() -> Self {
        // Both models think: who speaks to whom, and so which words they use for each other,
        // is decided by reasoning over the pictures.
        let model = |name: &str| AnalysisModel {
            model: name.to_owned(),
            vision: true,
            reasoning: Some(true),
        };
        Self {
            enabled: true,
            models: vec![model(NEMOTRON_VL), model(NEMOTRON_BOLT)],
            // The model the user chose for the translation (the local one) writes the final
            // translation of every page.
            translate: true,
            story_chars: 4000,
            pages_per_request: 12,
            overlap_pages: 2,
            reasoning: true,
            // A page is answered with the notes and the draft of every line in about five
            // thousand tokens, twelve pages share one answer and the thinking comes on top of
            // it: a model that loops (the log of one page wrote the same phrase until the
            // limit) is cut off at the limit.
            max_tokens: 49152,
            image_max_dimension: 1600,
            panel_max_dimension: 1024,
            right_to_left: true,
        }
    }
}

/// A lossless WebP file smaller than this share of the raw pixels' size is reported.
const MIN_WEBP_SAVING: f64 = 95.0;

/// The picture as the data URL of a lossless WebP. The size it saves is logged: how much
/// smaller than the raw pixels it is depends on the page (screentone compresses worse than
/// flat white), so a page that saves less than `MIN_WEBP_SAVING` percent is a warning.
fn picture_url(image: &DynamicImage, what: &str) -> anyhow::Result<String> {
    let encoded = encode_lossless_webp(image)?;
    let saved = format!("{:.1}%", encoded.saved_percent());
    if encoded.saved_percent() < MIN_WEBP_SAVING {
        tracing::warn!(what, %saved, bytes = encoded.encoded_bytes(), "the lossless WebP of a picture saves less than 95% of its raw pixels");
    } else {
        tracing::debug!(what, %saved, bytes = encoded.encoded_bytes(), "picture encoded as lossless WebP");
    }
    Ok(encoded.data_url())
}

/// A rectangle: left, top, right and bottom, in page pixels.
pub type Rect = [f64; 4];

/// The panels of a page and the boxes of its OCR lines, in page pixels.
#[derive(Clone, Debug, Default)]
pub struct PanelLayout {
    /// The panels, in any order.
    pub panels: Vec<Rect>,
    /// The box of every OCR line, in the order of the lines.
    pub boxes: Vec<Option<Rect>>,
}

/// The picture of one panel and the OCR lines in it.
#[derive(Clone)]
struct PanelPicture {
    picture: String,
    lines: Vec<usize>,
}

/// Panels beyond this many on one page are not sent one by one; only the whole page is sent.
const MAX_PANELS: usize = 10;
/// A panel smaller than this on a side is not sent.
const MIN_PANEL_SIDE: f64 = 32.0;

/// One page to read: its picture (with the boxes and numbers of the OCR lines drawn on it,
/// scaled down and encoded once), its panels when it has at least two that hold text, and the
/// OCR lines, in the order of the numbers.
#[derive(Clone)]
pub struct AnalysisPage {
    picture: String,
    panels: Vec<PanelPicture>,
    pub segments: Vec<String>,
}

impl AnalysisPage {
    /// Prepares a page for the analysis. `image` carries the numbers of `segments`; `layout`
    /// holds the panels when the page has any.
    pub fn new(
        image: &Arc<DynamicImage>,
        segments: Vec<String>,
        layout: Option<&PanelLayout>,
    ) -> anyhow::Result<Self> {
        let config = analysis_config().unwrap_or_default();
        let scaled = shrink_to(image, config.image_max_dimension.max(256))?;
        // The panels are only sent when a request carries one page; a batch has a picture for
        // every page and no room for the panels of each.
        let wants_panels =
            config.pages_per_request <= 1 && config.models.iter().any(|model| model.vision);
        let panels = match layout {
            Some(layout) if wants_panels => {
                panel_pictures(image, layout, segments.len(), &config).unwrap_or_else(|error| {
                    tracing::warn!(%error, "could not cut the panels of a page; only the whole page is sent");
                    Vec::new()
                })
            }
            _ => Vec::new(),
        };
        Ok(Self {
            picture: picture_url(&scaled, "page")?,
            panels,
            segments,
        })
    }
}

/// The order panels are read in: rows from the top to the bottom (panels that share more than
/// half of the lower one's height are one row), and inside a row from the right to the left
/// when `right_to_left`, else from the left to the right. Returns indexes of `panels`.
#[must_use]
pub fn reading_order(panels: &[Rect], right_to_left: bool) -> Vec<usize> {
    let mut by_top: Vec<usize> = (0..panels.len()).collect();
    by_top.sort_by(|&left, &right| panels[left][1].total_cmp(&panels[right][1]));
    // A row: its vertical extent and the panels in it.
    let mut rows: Vec<([f64; 2], Vec<usize>)> = Vec::new();
    for index in by_top {
        let [_, top, _, bottom] = panels[index];
        let height = (bottom - top).max(1.0);
        let same_row = rows.last_mut().filter(|(extent, _)| {
            let overlap = bottom.min(extent[1]) - top.max(extent[0]);
            overlap > 0.5 * height.min((extent[1] - extent[0]).max(1.0))
        });
        match same_row {
            Some((extent, members)) => {
                extent[0] = extent[0].min(top);
                extent[1] = extent[1].max(bottom);
                members.push(index);
            }
            None => rows.push(([top, bottom], vec![index])),
        }
    }
    let mut order = Vec::with_capacity(panels.len());
    for (_, mut members) in rows {
        members.sort_by(|&left, &right| {
            let (left, right) = (panels[left][0] + panels[left][2], panels[right][0] + panels[right][2]);
            if right_to_left {
                right.total_cmp(&left)
            } else {
                left.total_cmp(&right)
            }
        });
        order.extend(members);
    }
    order
}

fn area(rect: Rect) -> f64 {
    (rect[2] - rect[0]).max(0.0) * (rect[3] - rect[1]).max(0.0)
}

/// The panel (an index of `panels`) every OCR line belongs to: the smallest panel that holds
/// the center of the line's box, else the nearest panel. A line without a box goes with the line
/// before it (the first one with the first panel). `panels` must not be empty.
#[must_use]
pub fn assign_segments(panels: &[Rect], boxes: &[Option<Rect>]) -> Vec<usize> {
    let mut owners: Vec<usize> = Vec::with_capacity(boxes.len());
    for rect in boxes {
        let owner = match rect {
            Some(rect) => {
                let (x, y) = ((rect[0] + rect[2]) / 2.0, (rect[1] + rect[3]) / 2.0);
                let holding = (0..panels.len())
                    .filter(|&index| {
                        let panel = panels[index];
                        x >= panel[0] && x <= panel[2] && y >= panel[1] && y <= panel[3]
                    })
                    .min_by(|&left, &right| area(panels[left]).total_cmp(&area(panels[right])));
                holding.unwrap_or_else(|| {
                    let distance = |index: usize| {
                        let panel = panels[index];
                        let dx = (panel[0] - x).max(0.0).max(x - panel[2]);
                        let dy = (panel[1] - y).max(0.0).max(y - panel[3]);
                        dx * dx + dy * dy
                    };
                    (0..panels.len())
                        .min_by(|&left, &right| distance(left).total_cmp(&distance(right)))
                        .unwrap_or(0)
                })
            }
            None => owners.last().copied().unwrap_or(0),
        };
        owners.push(owner);
    }
    owners
}

/// The crop of a panel: the panel with a margin, inside the page; `None` when it is too small.
fn crop_rect(panel: Rect, width: u32, height: u32) -> Option<(u32, u32, u32, u32)> {
    let margin = 8.0 + 0.01 * (panel[2] - panel[0]).max(panel[3] - panel[1]);
    let left = (panel[0] - margin).max(0.0).floor();
    let top = (panel[1] - margin).max(0.0).floor();
    let right = (panel[2] + margin).min(f64::from(width)).ceil();
    let bottom = (panel[3] + margin).min(f64::from(height)).ceil();
    if right - left < MIN_PANEL_SIDE || bottom - top < MIN_PANEL_SIDE {
        return None;
    }
    Some((left as u32, top as u32, (right - left) as u32, (bottom - top) as u32))
}

/// The pictures of the panels that hold OCR lines, in reading order. Empty when the panels are
/// not worth sending: fewer than two such panels, more than `MAX_PANELS`, or an OCR line that no
/// picture holds (the whole page is sent anyway).
fn panel_pictures(
    image: &Arc<DynamicImage>,
    layout: &PanelLayout,
    lines: usize,
    config: &AnalysisConfig,
) -> anyhow::Result<Vec<PanelPicture>> {
    if layout.panels.len() < 2 || layout.boxes.len() != lines {
        return Ok(Vec::new());
    }
    let owners = assign_segments(&layout.panels, &layout.boxes);
    let mut pictures = Vec::new();
    for panel in reading_order(&layout.panels, config.right_to_left) {
        let held: Vec<usize> = owners
            .iter()
            .enumerate()
            .filter_map(|(line, &owner)| (owner == panel).then_some(line))
            .collect();
        if held.is_empty() {
            continue;
        }
        let Some((x, y, width, height)) = crop_rect(layout.panels[panel], image.width(), image.height())
        else {
            continue;
        };
        let crop = Arc::new(image.crop_imm(x, y, width, height));
        let scaled = shrink_to(&crop, config.panel_max_dimension.max(256))?;
        pictures.push(PanelPicture {
            picture: picture_url(&scaled, "panel")?,
            lines: held,
        });
    }
    let covered: usize = pictures.iter().map(|picture| picture.lines.len()).sum();
    if pictures.len() < 2 || pictures.len() > MAX_PANELS || covered != lines {
        return Ok(Vec::new());
    }
    Ok(pictures)
}

pub(crate) fn analysis_config() -> anyhow::Result<AnalysisConfig> {
    let config = koharu_config::load::<AnalysisConfig>(CONFIG_SECTION)?;
    let config = config.read()?.clone();
    Ok(config)
}

// ---------------------------------------------------------------------------------------
// What the vision model answers.

fn lenient_string<'de, D: Deserializer<'de>>(deserializer: D) -> Result<String, D::Error> {
    Ok(match Value::deserialize(deserializer)? {
        Value::String(text) => text,
        Value::Null => String::new(),
        other => other.to_string(),
    })
}

fn lenient_strings<'de, D: Deserializer<'de>>(deserializer: D) -> Result<Vec<String>, D::Error> {
    Ok(match Value::deserialize(deserializer)? {
        Value::Array(items) => items
            .into_iter()
            .filter_map(|item| match item {
                Value::String(text) => Some(text),
                Value::Null => None,
                other => Some(other.to_string()),
            })
            .collect(),
        Value::String(text) if !text.trim().is_empty() => vec![text],
        _ => Vec::new(),
    })
}

fn lenient_id<'de, D: Deserializer<'de>>(deserializer: D) -> Result<usize, D::Error> {
    Ok(match Value::deserialize(deserializer)? {
        Value::Number(number) => number.as_u64().map_or(usize::MAX, |id| id as usize),
        Value::String(text) => text.trim().parse().unwrap_or(usize::MAX),
        _ => usize::MAX,
    })
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(default)]
struct Relationship {
    #[serde(deserialize_with = "lenient_string")]
    with: String,
    #[serde(deserialize_with = "lenient_string")]
    relation: String,
    /// What the character calls themselves when they talk to `with`.
    #[serde(rename = "self", deserialize_with = "lenient_string")]
    self_word: String,
    /// What the character calls `with`.
    #[serde(deserialize_with = "lenient_string")]
    you: String,
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(default)]
struct Character {
    #[serde(deserialize_with = "lenient_string")]
    name: String,
    #[serde(deserialize_with = "lenient_strings")]
    aliases: Vec<String>,
    #[serde(deserialize_with = "lenient_string")]
    gender: String,
    #[serde(deserialize_with = "lenient_string")]
    age: String,
    #[serde(deserialize_with = "lenient_string")]
    role: String,
    /// What identifies the character in the picture: hair, clothes, build, marks.
    #[serde(deserialize_with = "lenient_string")]
    look: String,
    #[serde(deserialize_with = "lenient_string")]
    speech_style: String,
    relationships: Vec<Relationship>,
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(default)]
struct Term {
    /// `name` or `term`.
    #[serde(deserialize_with = "lenient_string")]
    kind: String,
    #[serde(deserialize_with = "lenient_string")]
    source: String,
    #[serde(deserialize_with = "lenient_string")]
    target: String,
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(default)]
struct Line {
    #[serde(deserialize_with = "lenient_id")]
    id: usize,
    #[serde(deserialize_with = "lenient_string")]
    kind: String,
    #[serde(deserialize_with = "lenient_string")]
    text: String,
    #[serde(deserialize_with = "lenient_string")]
    speaker: String,
    #[serde(deserialize_with = "lenient_string")]
    listener: String,
    #[serde(rename = "self", deserialize_with = "lenient_string")]
    self_word: String,
    #[serde(deserialize_with = "lenient_string")]
    you: String,
    #[serde(deserialize_with = "lenient_string")]
    tone: String,
    /// What identifies the speaker on this line: a look, or the voice of one who is not seen.
    #[serde(deserialize_with = "lenient_string")]
    speaker_desc: String,
    /// `seen`, `inferred` or `unknown`: how the speaker was found out.
    #[serde(deserialize_with = "lenient_string")]
    speaker_basis: String,
    /// What identifies the listener, or who the group is.
    #[serde(deserialize_with = "lenient_string")]
    listener_desc: String,
    #[serde(deserialize_with = "lenient_string")]
    listener_basis: String,
    /// `one`, `group`, `self` or `unknown`: who is spoken to.
    #[serde(deserialize_with = "lenient_string")]
    audience: String,
    /// What in the text shows the register: 一人称 and 二人称 words, sentence endings, honorifics.
    #[serde(deserialize_with = "lenient_string")]
    cues: String,
    /// Why the line uses other words than the usual pair of its direction (a quarrel, a taunt,
    /// tenderness, a pretence); empty for a line that follows its direction.
    #[serde(deserialize_with = "lenient_string")]
    exception: String,
    /// The first translation of `text` into the target language, by the model that saw the page.
    #[serde(deserialize_with = "lenient_string")]
    draft: String,
}

impl Line {
    /// Makes the keys agree with what the line says about its audience and its certainty, so
    /// that a line to a group, to nobody, or from a voice that is not seen is never counted as a
    /// direction between two characters.
    fn normalize(&mut self) {
        let audience = self.audience.trim().to_lowercase();
        self.audience = match audience.as_str() {
            "group" | "crowd" | "several" | "many" | "all" | "everyone" | "plural" | "audience" => "group",
            "self" | "thought" | "soliloquy" | "monologue" => "self",
            "unknown" | "unclear" | "none" => "unknown",
            _ => "one",
        }
        .to_owned();
        let basis = |text: &str| match text.trim().to_lowercase().as_str() {
            "seen" | "visible" | "shown" => "seen",
            "inferred" | "guessed" | "context" => "inferred",
            "unknown" | "unclear" | "unseen" | "none" => "unknown",
            _ => "",
        }
        .to_owned();
        self.speaker_basis = basis(&self.speaker_basis);
        self.listener_basis = basis(&self.listener_basis);
        // A person who cannot be told is nobody in particular: the key is `unknown` and what
        // was said about them stays as a description.
        if self.speaker_basis == "unknown" && !is_generic(&self.speaker) {
            if self.speaker_desc.trim().is_empty() {
                self.speaker_desc = std::mem::take(&mut self.speaker);
            }
            self.speaker = "unknown".to_owned();
        }
        match self.audience.as_str() {
            "group" => {
                if !is_generic(&self.listener) && self.listener_desc.trim().is_empty() {
                    self.listener_desc = std::mem::take(&mut self.listener);
                }
                self.listener = "everyone".to_owned();
            }
            "self" => self.listener = "self".to_owned(),
            "unknown" => {
                if !is_generic(&self.listener) && self.listener_desc.trim().is_empty() {
                    self.listener_desc = std::mem::take(&mut self.listener);
                }
                self.listener = "unknown".to_owned();
            }
            _ if self.listener_basis == "unknown" && !is_generic(&self.listener) => {
                if self.listener_desc.trim().is_empty() {
                    self.listener_desc = std::mem::take(&mut self.listener);
                }
                self.listener = "unknown".to_owned();
            }
            _ => {}
        }
    }
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(default)]
struct PageOutput {
    #[serde(deserialize_with = "lenient_string")]
    scene: String,
    characters: Vec<Character>,
    terms: Vec<Term>,
    lines: Vec<Line>,
    /// What happens on the page, for whoever reads only the summaries of the story.
    #[serde(deserialize_with = "lenient_string")]
    summary: String,
}

/// The JSON schema of the answer for a page of `expected` lines.
fn schema(expected: usize) -> Value {
    let text = |description: &str| json!({ "type": "string", "description": description });
    json!({
        "type": "object",
        "properties": {
            "scene": text("One or two sentences: where, when and what happens on this page."),
            "characters": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "name": text("The label of the character (sex, age, look) until the page gives a real name; then that name."),
                        "aliases": { "type": "array", "items": { "type": "string" } },
                        "gender": text("man, woman, boy, girl, or unknown."),
                        "age": text("A number or a stage: child, teenager, young adult, adult, middle-aged, elderly."),
                        "role": text("Who the character is in the story."),
                        "look": text("What identifies the character in the picture: hair, clothes, build, marks."),
                        "speech_style": text("How the character talks: polite, rough, childish, archaic, ..."),
                        "relationships": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "with": text("The other character."),
                                    "relation": text("Their relation: colleague, mother and son, ..."),
                                    "self": text("The word this character uses for themselves when talking to the other."),
                                    "you": text("The word this character uses for the other.")
                                },
                                "required": ["with"]
                            }
                        }
                    },
                    "required": ["name"]
                }
            },
            "terms": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "kind": { "type": "string", "enum": ["name", "term"] },
                        "source": text("As written in the lines."),
                        "target": text("The rendering to use in the translation.")
                    },
                    "required": ["source", "target"]
                }
            },
            "lines": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "id": {
                            "type": "integer",
                            "minimum": 0,
                            "maximum": expected.saturating_sub(1),
                            "description": "The number of the OCR line."
                        },
                        "kind": { "type": "string", "enum": ["speech", "thought", "narration", "sfx", "sign", "other"] },
                        "text": text("The line, corrected when OCR misread it; otherwise copied unchanged."),
                        "speaker": text("The label of the character who says or thinks it; 'narrator', or 'unknown' when nobody can be told."),
                        "speaker_desc": text("What identifies the speaker here: look, or the voice of one who is not seen."),
                        "speaker_basis": { "type": "string", "enum": ["seen", "inferred", "unknown"] },
                        "listener": text("The label of the character spoken to; 'self' for a thought, 'everyone' for a group, 'unknown'."),
                        "listener_desc": text("What identifies the listener, or who the group is (ages, sexes, relation to the speaker)."),
                        "listener_basis": { "type": "string", "enum": ["seen", "inferred", "unknown"] },
                        "audience": { "type": "string", "enum": ["one", "group", "self", "unknown"] },
                        "self": text("The word the speaker uses for themselves."),
                        "you": text("The word the speaker uses for the listener (the plural for a group)."),
                        "tone": text("Two or three words."),
                        "cues": text("What in the text shows the register: I/you words, endings, honorifics."),
                        "exception": text("Why this line needs other words than the usual pair of its direction; empty otherwise."),
                        "draft": text("The translation of `text` into the target language, written with the words chosen in self and you; empty for a line with nothing to translate.")
                    },
                    // Who speaks, to whom and with which words is decided for every line.
                    "required": ["id", "kind", "speaker", "listener", "audience", "self", "you", "draft"]
                }
            },
            "summary": text("One to three sentences: what happens on this page, who does what to whom and why, so that the story can be followed from the summaries alone.")
        },
        "required": ["lines", "summary"]
    })
}

/// The JSON schema of the answer for a batch of pages with `counts` OCR lines each: one entry of
/// the page schema for every page, numbered by the position of its picture in the request.
fn batch_schema(counts: &[usize]) -> Value {
    let widest = counts.iter().copied().max().unwrap_or(1);
    let mut page = schema(widest);
    if let Some(properties) = page["properties"].as_object_mut() {
        properties.insert(
            "page".to_owned(),
            json!({
                "type": "integer",
                "minimum": 1,
                "maximum": counts.len(),
                "description": "The number of the picture of this page in the request, from 1."
            }),
        );
    }
    page["required"] = json!(["page", "lines", "summary"]);
    json!({
        "type": "object",
        "properties": {
            "pages": {
                "type": "array",
                "minItems": counts.len(),
                "maxItems": counts.len(),
                "items": page
            }
        },
        "required": ["pages"]
    })
}

fn system_prompt(target: Language) -> String {
    let mut prompt = format!(
        "You read a manga page that is going to be translated into {target}. You see the page \
picture (when one is attached) and the text OCR read from it, as numbered lines. The pictures \
are cleaned: the lettering was erased from the balloons, so the words are only in the OCR lines; \
the red boxes on the picture mark where each line stood and carry the same numbers as the lines. \
Several pages may come in one request, one picture each; the numbers of the lines start again at \
0 on every page. You are the first translator: you translate every line \
as you read it. Your answer then goes to a second translator, who does not see the \
pictures, reads your notes and your draft and writes the final translation, so everything that \
depends on the picture must be in it: who speaks, to whom, how old they are, how they talk.

For every OCR line:
- id: the number of the line.
- kind: speech, thought (an inner voice, often in a cloud or a box without a tail), narration, \
sfx, sign or other.
- text: the line, corrected when OCR misread it (wrong, missing or extra characters, kana read \
as kanji, noise from the drawing). Check it against the story and the scene (the picture has no lettering left). Copy the line \
unchanged when it is right. This is the source: do not translate or reword it here (that is \
draft) and never add words that are not on the page.
- speaker: who says or thinks it ('narrator' for narration, 'unknown' when nobody can be told). \
Follow the tail of the bubble, the positions and the story. A person is identified by what they \
are, not by a name: give the label of 'Known so far' when the person is there, else a short \
label made of what shows them (sex, age, the most telling look or clothes): 'man about 30 with \
glasses and a suit', 'girl with twin tails in a school uniform', 'old woman in a kimono'. Use \
exactly that label on every page. A real name counts only when the page says or writes it (a \
line that calls someone, a name tag, a caption): then it is the name of that person, and the \
label goes in aliases. Never make up a name from a look, never use a role such as nurse, patient \
or boss as the label, and never take the person who is called in a line for the speaker. Two \
people who answer each other are two different characters, and nobody speaks to themself \
except in a thought.
- speaker_desc: what identifies the speaker on this line, in a few words: the look, or the \
voice (deep, high, old, childish) of a speaker who is not in the picture.
- speaker_basis: seen (the tail or the mouth shows them), inferred (the chain of balloons, the \
person who answers, who is in the scene, the way of speaking) or unknown.
- listener: the person spoken to, labelled the same way; 'self' for a thought, 'everyone' for a \
group, 'unknown'. listener_desc and listener_basis work like speaker_desc and speaker_basis \
(the gaze, the body, who answers, who is named in the line).
- audience: one (a single person), group (several people, a crowd, a class, an army, a \
broadcast: say who they are in listener_desc: how old, which sex, how they stand to the speaker), \
self (a thought or a talk to oneself) or unknown (it cannot be told, even from the context).
- self and you: the words the speaker uses for themselves and for the listener in {target}, \
chosen from age, sex, rank, closeness and manner of speaking. They are decided for every line, \
speech and thought alike, and never left empty (a sound effect or a sign may have none). Decide \
in this order: who speaks, who listens, how they stand to each other in this scene, then the \
words. Name the speaker and the listener on every line, also when they are the same as on the \
line before: the translator who comes after you has no picture and uses exactly your words.
- tone: two or three words (polite, rough, teasing, tender, scared, ...), read from the face, the \
lettering, the punctuation and the story.
- cues: what in the text of the line shows the register: the words for I and you (俺, 僕, 私, \
あたし, お前, 君, あなた), the ending (です/ます, だ, ぜ, わ, ～なさい), the honorific, the title.
- exception: empty for a line that follows the usual words of its direction. Otherwise the \
reason in a few words (quarrel, taunt, threat, tenderness, pretence, a talk to a child, quoting \
someone), and then self and you are the words that this line needs.
- draft: the translation of `text` into {target}, written last, with the words you chose in \
self and you, as a good translator of comics would write it: all of the meaning and nothing \
else, the tone of the speaker, the humour, the hesitation and the emphasis, in words a native \
reader would say aloud, short enough for the balloon. The same rendering for the same name or \
term on every page. A sound effect gets the word a {target} reader expects, a sign or a title is \
translated too. No notes, no explanations, no brackets, no quotation marks around the line. \
Empty for a line with nothing to translate.

Also answer: scene, summary (one to \
three sentences: what happens on this page, who does what to whom and why; someone who reads \
only the summaries has to be able to follow the story), characters \
(new ones, or what you learned about known ones: name, aliases, gender, age, look, role, \
speech_style, relationships with the pronouns they use) and terms (names of people, places, organizations, \
titles and recurring words, with the {target} rendering to use).

Rules:
- Recognize a person by the look; decide their gender, age and manner from how they speak first \
(pronouns such as 俺 or 僕, sentence endings such as ぞ or わ, honorifics) and from the picture, \
never from hair, glasses or build alone. Write down the look of every character (look) so that \
the translator can tell the people of a page apart without a name.
- A speaker who is not in the picture (the tail leaves the panel, a voice from a phone or from \
behind a door): find out who from the chain of balloons, who answers, who is in the scene and \
how it is said, and say it is inferred; when nothing tells, speaker is 'unknown' and \
speaker_desc says what the voice is like. The same for a listener. Never give an unknown person \
a name or the pair of someone else; choose words that fit what is known (see the pronoun \
rules).
- Words to a group are not the pair of a person: audience is group and you is the plural form \
that fits the group.
- The tone and the situation decide the words as much as the relationship: the same two people \
speak differently in a quarrel, in front of others, when one mocks the other or when they are \
tender. Take the register from the tone, the cues and the scene, not from the names.
- Keep the names, genders, ages and pronoun pairs of 'Known so far' unless the page clearly \
shows otherwise. Report only what is new or changed.
- A character has one name everywhere; other ways of calling them go in aliases. Honorifics \
(さん, くん, ちゃん) show the relationship and do not belong to the name.
- Write the lines in the order of their numbers and answer for every number you were given.
- Answer with the JSON object only."
    );
    prompt.push_str(pronoun_rules(target));
    prompt
}

/// What the target language adds to the rules for the words of "I" and "you".
fn pronoun_rules(target: Language) -> &'static str {
    if target == Language::Vietnamese {
        "

Vietnamese pronouns for self and you:
- Pick the pair from the relationship, the ages and the rank, for example 'tôi - cô', \
'anh - em', 'em - chị', 'tớ - cậu', 'tao - mày'. People who speak politely to each other get \
a formal pair, close people a familiar one. Do not give everyone the same pair.
- A pair belongs to one direction: what A calls B is not what B calls A. Keep the pair of a \
direction the same on every page.
- An older speaker calls himself 'tôi' (or 'bác', 'chú', 'ông', 'lão' by age and manner), never \
'con' or 'cháu', unless the speaker is a child or a younger relative of the listener.
- A word that names a sex belongs to that sex, as 'I' and as 'you' alike: 'anh', 'chú', 'ông' \
are for men and boys only; 'chị', 'cô', 'dì', 'bà' for women and girls only. A woman never calls \
herself 'anh' and nobody calls a man 'chị' or 'cô'. 'tôi', 'tớ', 'mình', 'em', 'cháu', 'con' fit \
either sex. Check the gender of the person the word is about before you write it.
- The two directions of a pair fit each other. When A calls himself 'anh' and calls B 'em', \
B calls A 'anh' and calls themselves 'em' (not 'tôi - cô'); when A calls B 'chị' and calls \
themselves 'em', B calls A 'em' and themselves 'chị'. A direction may differ only for a real \
reason: a quarrel, a stranger who speaks formally, a talk to a child.
- The first-person word of a thought is the one the character uses aloud, or 'mình'.
- Tone and situation decide the pair as much as the relationship: the same two people say \
'anh - em' tenderly, 'tôi - anh' coldly in anger or in front of others, 'tao - mày' in a fight, \
'ta - ngươi' when one looks down on the other in a period or fantasy story, 'em - anh' or 'em - \
chị' with fear or respect. Choose from the tone, the cues and the scene of the line. A line that \
clearly leaves the usual pair of its direction gets an exception with the reason and the words \
it needs.
- To several people (audience group) the word for 'you' is a plural that fits who they are: \
'các em' (children, students, juniors), 'các anh', 'các chị', 'các cô', 'các ông', 'các bà' \
(a group of one sex, older or respected), 'các bạn' (peers), 'mọi người' (mixed or not known, \
polite and neutral), 'các vị' (formal, respectful), 'bọn mày', 'chúng mày' (rude or contemptuous), \
'các ngươi' (period story). The speaker's own word follows the same level ('tôi', 'anh', 'chị', \
'cô', 'thầy', 'tao', 'ta').
- An unknown speaker gets words that name no sex unless the voice, the manner or the words of \
the source show one: 'tôi' for an adult, 'tớ' or 'mình' for a casual peer. An unknown listener \
gets no word for 'you' when the sentence allows it (an imperative, no pronoun); when one is \
needed, 'bạn' (neutral or polite), 'cậu' (casual, peers) and 'mày' only for a rude tone."
    } else {
        ""
    }
}

/// What the vision model is shown of a page.
enum View<'a> {
    /// No picture: the text only.
    Text,
    /// The whole page in one picture.
    Page,
    /// The whole page, then one picture for every panel, in reading order.
    PageAndPanels(&'a [PanelPicture]),
}

fn user_prompt(
    number: usize,
    total: usize,
    segments: &[String],
    bible: &Bible,
    previous: Option<&str>,
    view: &View<'_>,
) -> String {
    let mut prompt = format!("Page {} of {total}. ", number + 1);
    match view {
        View::Page => prompt.push_str("The picture of the page is attached.\n"),
        View::Text => prompt.push_str(
            "No picture is attached: decide from the text only, and answer 'unknown' where a \
speaker cannot be told.\n",
        ),
        View::PageAndPanels(panels) => {
            let _ = writeln!(
                prompt,
                "{} pictures are attached. Picture 1 is the whole page: use it for the layout, for \
who stands where and for which balloon points at whom. The next {} pictures are the panels of the \
page in the order they are read, each one enlarged: panel 1 is the first one and the story goes on \
in the next panel. A panel is part of the whole page, never a page of its own: read it in the \
light of the pictures before it. The same character in different panels is the same person. The \
red boxes carry the same numbers of the OCR lines in every picture.",
                panels.len() + 1,
                panels.len()
            );
            for (position, panel) in panels.iter().enumerate() {
                let lines: Vec<String> = panel.lines.iter().map(ToString::to_string).collect();
                let _ = writeln!(prompt, "Panel {}: lines {}", position + 1, lines.join(", "));
            }
        }
    }
    match previous.map(str::trim).filter(|scene| !scene.is_empty()) {
        Some(scene) => {
            let _ = writeln!(
                prompt,
                "The previous page ended like this (the story goes on from it): {}",
                clip(scene, 400)
            );
        }
        None => {}
    }
    let known = bible.prompt_json();
    if known.is_empty() {
        prompt.push_str("Known so far: nothing yet.\n");
    } else {
        let _ = writeln!(prompt, "Known so far (keep it consistent): {known}");
    }
    prompt.push_str("OCR lines:\n");
    for (id, segment) in segments.iter().enumerate() {
        let quoted = serde_json::to_string(&clip(segment, 400)).unwrap_or_default();
        let _ = writeln!(prompt, "{id}: {quoted}");
    }
    let _ = write!(
        prompt,
        "Answer for these {} lines (ids 0 to {}).",
        segments.len(),
        segments.len().saturating_sub(1)
    );
    prompt
}

/// A page of the request before, sent again so that the model sees where the story left off.
/// The model does not answer for it.
struct ContextPage<'a> {
    /// The number of the page, from 0.
    number: usize,
    page: &'a AnalysisPage,
    /// What was decided for its lines the first time (see `PageAnalysis::decisions`); empty
    /// when no model could read the page.
    decided: String,
}

/// What happened on the pages read so far, in reading order: the nearest pages when there is
/// no room for all. `summaries` holds the number of a page (from 0) and what happens on it.
fn story_so_far(summaries: &[(usize, String)], limit: usize) -> String {
    let mut kept: Vec<String> = Vec::new();
    let mut used = 0;
    for (number, summary) in summaries.iter().rev() {
        let summary = summary.trim();
        if summary.is_empty() {
            continue;
        }
        let line = format!("page {}: {summary}", number + 1);
        let size = line.chars().count() + 1;
        if used + size > limit {
            break;
        }
        used += size;
        kept.push(line);
    }
    kept.reverse();
    kept.join("\n")
}

/// The prompt of a request: the new pages `numbers` (from 0, in reading order), each with its
/// picture (when `pictures`) and its OCR lines, which are numbered from 0 on every page. From
/// the second request on `context` holds the last pages of the request before, whose pictures
/// come first, and `story` what happened so far.
fn batch_prompt(
    numbers: &[usize],
    total: usize,
    pages: &[&AnalysisPage],
    context: &[ContextPage<'_>],
    story: &str,
    bible: &Bible,
    pictures: bool,
) -> String {
    let consecutive = numbers.windows(2).all(|pair| pair[1] == pair[0] + 1);
    let new = if consecutive {
        format!(
            "pages {} to {}",
            numbers[0] + 1,
            numbers[numbers.len() - 1] + 1
        )
    } else {
        let named: Vec<String> = numbers.iter().map(|number| (number + 1).to_string()).collect();
        format!("pages {} (the pages between them are not part of this request)", named.join(", "))
    };
    let mut prompt = format!("New {new} of {total}, in reading order.\n");
    let read = context.len();
    if pictures {
        if let (Some(first), Some(last)) = (context.first(), context.last()) {
            let _ = writeln!(
                prompt,
                "{} pictures are attached. Pictures 1 to {read} are pages {} to {}, which you read \
already: they are only here so that you see where the story left off, so do not answer for them. \
Pictures {} to {} are the new pages: answer for these.",
                read + pages.len(),
                first.number + 1,
                last.number + 1,
                read + 1,
                read + pages.len()
            );
        } else {
            let _ = writeln!(
                prompt,
                "{} pictures are attached, one for every page, in order.",
                pages.len()
            );
        }
        prompt.push_str(
            "Use each picture for the layout, for who stands where and for which balloon points at \
whom. The red boxes carry the numbers of the OCR lines of their own page. The pages follow one \
another: the same character on different pages is the same person, and a line can answer someone \
who spoke on an earlier page, so read the pictures as one scene.\n",
        );
    } else {
        prompt.push_str(
            "No picture is attached: decide from the text only, and answer 'unknown' where a \
speaker cannot be told.\n",
        );
    }
    if !story.trim().is_empty() {
        let _ = writeln!(prompt, "Story so far, page by page:\n{}", story.trim());
    }
    let known = bible.prompt_json();
    if known.is_empty() {
        prompt.push_str("Known so far: nothing yet.\n");
    } else {
        let _ = writeln!(prompt, "Known so far (keep it consistent): {known}");
    }
    for page in context {
        if page.decided.trim().is_empty() {
            continue;
        }
        let _ = writeln!(
            prompt,
            "Page {} (read already; keep the same words for the same people):{}",
            page.number + 1,
            page.decided
        );
    }
    for (position, page) in pages.iter().enumerate() {
        let _ = writeln!(
            prompt,
            "New page {} (picture {}, page {}), OCR lines:",
            position + 1,
            read + position + 1,
            numbers[position] + 1
        );
        for (id, segment) in page.segments.iter().enumerate() {
            let quoted = serde_json::to_string(&clip(segment, 400)).unwrap_or_default();
            let _ = writeln!(prompt, "{id}: {quoted}");
        }
    }
    let _ = write!(
        prompt,
        "Answer with one entry in `pages` for each of the {} new pages, in order, with `page` set \
to the number of the new page (1 for the first). Each entry holds the lines of its own page (ids \
from 0, as listed above) and its own scene and summary. For every line say who speaks, who \
listens and the words for 'I' and 'you'. Keep the optional fields of a line short.",
        pages.len()
    );
    prompt
}

/// The JSON object in an answer, also when a model wrote something around it.
fn json_object(text: &str) -> Option<&str> {
    let text = text.trim();
    if text.starts_with('{') && serde_json::from_str::<Value>(text).is_ok() {
        return Some(text);
    }
    let (start, end) = (text.find('{')?, text.rfind('}')?);
    let object = text.get(start..=end)?;
    serde_json::from_str::<Value>(object).is_ok().then_some(object)
}

/// Keeps the lines whose id is one of the `count` OCR lines of the page, each id once, and makes
/// their keys agree with what they say.
fn clean_lines(output: &mut PageOutput, count: usize) {
    let mut seen = HashSet::new();
    output
        .lines
        .retain(|line| line.id < count && seen.insert(line.id));
    for line in &mut output.lines {
        line.normalize();
    }
}

fn parse_page(text: &str, count: usize) -> anyhow::Result<PageOutput> {
    let json = json_object(text).ok_or_else(|| anyhow!("the answer holds no JSON object"))?;
    let mut output: PageOutput = serde_json::from_str(json).context("invalid analysis JSON")?;
    clean_lines(&mut output, count);
    ensure!(
        !output.lines.is_empty(),
        "the answer has no line for any of the {count} OCR lines"
    );
    Ok(output)
}

/// The answer to a batch: what was read of every page, `None` for a page the answer leaves out
/// or holds no usable line of. An entry is placed by its `page` number, else by its position.
fn parse_batch(text: &str, counts: &[usize]) -> anyhow::Result<Vec<Option<PageOutput>>> {
    let json = json_object(text).ok_or_else(|| anyhow!("the answer holds no JSON object"))?;
    let root: Value = serde_json::from_str(json).context("invalid analysis JSON")?;
    let entries = root
        .get("pages")
        .and_then(Value::as_array)
        .ok_or_else(|| anyhow!("the answer has no `pages`"))?;
    let mut outputs: Vec<Option<PageOutput>> = vec![None; counts.len()];
    for (position, entry) in entries.iter().enumerate() {
        let stated = match entry.get("page") {
            Some(Value::Number(number)) => number.as_u64().map(|number| number as usize),
            Some(Value::String(text)) => text.trim().parse().ok(),
            _ => None,
        };
        let number = stated
            .filter(|number| (1..=counts.len()).contains(number))
            .unwrap_or(position + 1);
        let Some(index) = number.checked_sub(1).filter(|&index| index < counts.len()) else {
            continue;
        };
        if outputs[index].is_some() {
            continue;
        }
        let Ok(mut output) = serde_json::from_value::<PageOutput>(entry.clone()) else {
            continue;
        };
        clean_lines(&mut output, counts[index]);
        if !output.lines.is_empty() {
            outputs[index] = Some(output);
        }
    }
    ensure!(
        outputs.iter().any(Option::is_some),
        "the answer has no line for any of the {} pages",
        counts.len()
    );
    Ok(outputs)
}

// ---------------------------------------------------------------------------------------
// What the vision models have learned about the work so far.

fn clip(text: &str, max: usize) -> String {
    text.trim().chars().take(max).collect()
}

fn same_name(left: &str, right: &str) -> bool {
    let (left, right) = (left.trim().to_lowercase(), right.trim().to_lowercase());
    !left.is_empty() && left == right
}

fn is_generic(name: &str) -> bool {
    matches!(
        name.trim().to_lowercase().as_str(),
        "" | "unknown" | "narrator" | "narration" | "everyone" | "self" | "none" | "n/a" | "?"
    )
}

/// The characters and terms of the work, in the order they were learned. What is known stays:
/// a later page fills the blanks of an entry and adds new ones, and never flips a choice.
#[derive(Clone, Debug, Default, PartialEq)]
struct Bible {
    characters: Vec<Character>,
    terms: Vec<Term>,
    /// How many lines used each pair of words, by direction (speaker, listener; lowercase).
    /// What the first line of a direction used is not what the direction is: a line can be
    /// noise. `settle_pairs` gives every direction the pair most of its lines used.
    tally: Tally,
}

/// What a speaker calls themselves and what they call the listener.
type WordPair = (String, String);
/// Pairs of words with the number of lines that used them, for every direction.
type Tally = Vec<((String, String), Vec<(WordPair, u32)>)>;

fn add_votes(tally: &mut Tally, key: (String, String), pair: WordPair, votes: u32) {
    match tally.iter_mut().find(|(known, _)| *known == key) {
        Some((_, pairs)) => match pairs.iter_mut().find(|(known, _)| *known == pair) {
            Some((_, count)) => *count += votes,
            None => pairs.push((pair, votes)),
        },
        None => tally.push((key, vec![(pair, votes)])),
    }
}

fn fill(known: &mut String, new: &str) {
    if known.trim().is_empty() {
        *known = clip(new, FIELD_CHARS);
    }
}

impl Bible {
    fn character_index(&self, name: &str) -> Option<usize> {
        self.characters.iter().position(|character| {
            same_name(&character.name, name)
                || character
                    .aliases
                    .iter()
                    .any(|alias| same_name(alias, name))
        })
    }

    fn merge_relationship(known: &mut Vec<Relationship>, new: Relationship) {
        if new.with.trim().is_empty() {
            return;
        }
        match known
            .iter_mut()
            .find(|known| same_name(&known.with, &new.with))
        {
            Some(known) => {
                fill(&mut known.relation, &new.relation);
                fill(&mut known.self_word, &new.self_word);
                fill(&mut known.you, &new.you);
            }
            None => known.push(Relationship {
                with: clip(&new.with, FIELD_CHARS),
                relation: clip(&new.relation, FIELD_CHARS),
                self_word: clip(&new.self_word, FIELD_CHARS),
                you: clip(&new.you, FIELD_CHARS),
            }),
        }
    }

    fn merge_character(&mut self, new: Character) {
        if is_generic(&new.name) {
            return;
        }
        // A character that was met under a label ('man with glasses') and is now named comes
        // with that label in `aliases`: it is the same person, not a second one, or the words
        // the two entries hold would never meet.
        let by_name = self.character_index(&new.name);
        let index = by_name.or_else(|| {
            new.aliases
                .iter()
                .find_map(|alias| self.character_index(alias))
        });
        let Some(index) = index else {
            if self.characters.len() < MAX_CHARACTERS {
                let mut character = Character {
                    name: clip(&new.name, FIELD_CHARS),
                    aliases: Vec::new(),
                    gender: clip(&new.gender, FIELD_CHARS),
                    age: clip(&new.age, FIELD_CHARS),
                    role: clip(&new.role, FIELD_CHARS),
                    look: clip(&new.look, 80),
                    speech_style: clip(&new.speech_style, FIELD_CHARS),
                    relationships: Vec::new(),
                };
                character.aliases = new.aliases.iter().map(|alias| clip(alias, FIELD_CHARS)).collect();
                for relationship in new.relationships {
                    Self::merge_relationship(&mut character.relationships, relationship);
                }
                self.characters.push(character);
            }
            return;
        };
        let known = &mut self.characters[index];
        // Found only through the label: the name that came with it is the one to use from now
        // on (an honorific makes no new name: 'Sasaki-kun' is still Sasaki).
        if by_name.is_none() && crate::address::norm_name(&known.name) != crate::address::norm_name(&new.name) {
            let label = std::mem::replace(&mut known.name, clip(&new.name, FIELD_CHARS));
            known.aliases.push(label);
        }
        fill(&mut known.gender, &new.gender);
        fill(&mut known.age, &new.age);
        fill(&mut known.role, &new.role);
        fill(&mut known.look, &new.look);
        fill(&mut known.speech_style, &new.speech_style);
        for alias in new.aliases.into_iter().chain([new.name]) {
            if !same_name(&alias, &known.name)
                && !alias.trim().is_empty()
                && !known.aliases.iter().any(|known| same_name(known, &alias))
            {
                known.aliases.push(clip(&alias, FIELD_CHARS));
            }
        }
        for relationship in new.relationships {
            Self::merge_relationship(&mut known.relationships, relationship);
        }
    }

    fn merge_term(&mut self, new: Term) {
        let (source, target) = (clip(&new.source, FIELD_CHARS), clip(&new.target, FIELD_CHARS));
        if source.is_empty()
            || target.is_empty()
            || self.terms.len() >= MAX_TERMS
            || self.terms.iter().any(|known| same_name(&known.source, &source))
        {
            return;
        }
        let kind = if new.kind.trim().eq_ignore_ascii_case("name") {
            "name"
        } else {
            "term"
        };
        self.terms.push(Term {
            kind: kind.to_owned(),
            source,
            target,
        });
    }

    /// What one line shows about how a speaker talks to a listener.
    fn note_pair(&mut self, speaker: &str, listener: &str, self_word: &str, you: &str) {
        if is_generic(speaker)
            || is_generic(listener)
            || (self_word.trim().is_empty() && you.trim().is_empty())
        {
            return;
        }
        // A listener known under an alias is written under the name of the character.
        let listener = match self.character_index(listener) {
            Some(index) => self.characters[index].name.clone(),
            None => listener.to_owned(),
        };
        let relationship = Relationship {
            with: listener.clone(),
            relation: String::new(),
            self_word: self_word.to_owned(),
            you: you.to_owned(),
        };
        match self.character_index(speaker) {
            Some(index) => Self::merge_relationship(&mut self.characters[index].relationships, relationship),
            None => self.merge_character(Character {
                name: speaker.to_owned(),
                relationships: vec![relationship],
                ..Character::default()
            }),
        }
        let (mine, yours) = (self_word.trim().to_lowercase(), you.trim().to_lowercase());
        if !mine.is_empty() && !yours.is_empty() {
            let key = (self.direction_name(speaker), self.direction_name(&listener));
            add_votes(&mut self.tally, key, (mine, yours), 1);
        }
    }

    /// The name a direction is counted under: the character's name when the name or alias is
    /// known, else the name as written; always lowercase.
    fn direction_name(&self, name: &str) -> String {
        self.character_index(name).map_or_else(
            || name.trim().to_lowercase(),
            |index| self.characters[index].name.trim().to_lowercase(),
        )
    }

    fn merge_output(&mut self, output: &PageOutput) {
        for character in &output.characters {
            self.merge_character(character.clone());
        }
        for term in &output.terms {
            self.merge_term(term.clone());
        }
        for line in &output.lines {
            // A line that leaves the usual words of its direction says nothing about them.
            if line.exception.trim().is_empty() {
                self.note_pair(&line.speaker, &line.listener, &line.self_word, &line.you);
            }
            // What shows a person on a line is kept for the people the work knows by a label.
            if line.speaker_basis != "unknown" {
                self.learn_look(&line.speaker, &line.speaker_desc);
            }
            if line.audience == "one" && line.listener_basis != "unknown" {
                self.learn_look(&line.listener, &line.listener_desc);
            }
        }
    }

    /// Remembers what identifies a known character when nothing is known of how they look.
    fn learn_look(&mut self, name: &str, desc: &str) {
        if is_generic(name) || desc.trim().is_empty() {
            return;
        }
        if let Some(index) = self.character_index(name) {
            fill(&mut self.characters[index].look, &clip(desc, 80));
        }
    }

    /// What the vision model is told it knows already, as compact JSON within a size limit
    /// (the oldest entries go first when there is too much).
    fn prompt_json(&self) -> String {
        let mut characters = self.characters.clone();
        let mut terms = self.terms.clone();
        loop {
            if characters.is_empty() && terms.is_empty() {
                return String::new();
            }
            let json = json!({ "characters": characters, "terms": terms }).to_string();
            if json.chars().count() <= BIBLE_PROMPT_CHARS {
                return json;
            }
            if terms.len() > characters.len() {
                terms.remove(0);
            } else if characters.len() > 1 {
                characters.remove(0);
            } else {
                return json.chars().take(BIBLE_PROMPT_CHARS).collect();
            }
        }
    }

    fn profile(character: &Character) -> String {
        let mut parts = Vec::new();
        for part in [&character.gender, &character.age, &character.look, &character.role] {
            if !part.trim().is_empty() {
                parts.push(part.trim().to_owned());
            }
        }
        let mut profile = parts.join(", ");
        if !character.speech_style.trim().is_empty() {
            let _ = write!(profile, "; speaks: {}", character.speech_style.trim());
        }
        profile
    }

    /// A character as a reader sees them, without the name: sex, age, look and role
    /// ('young adult man, black hair and glasses, colleague'). Empty when nothing is known.
    fn describe(character: &Character) -> String {
        fn known(text: &str) -> Option<&str> {
            let text = text.trim();
            (!text.is_empty() && !text.eq_ignore_ascii_case("unknown")).then_some(text)
        }
        let person = match (known(&character.age), known(&character.gender)) {
            (Some(age), Some(gender)) if age.chars().all(|c| c.is_ascii_digit()) => {
                format!("{gender} about {age}")
            }
            (Some(age), Some(gender)) => format!("{age} {gender}"),
            (None, Some(gender)) => gender.to_owned(),
            (Some(age), None) => format!("{age} person"),
            (None, None) => String::new(),
        };
        let mut parts = Vec::new();
        if !person.is_empty() {
            parts.push(person);
        }
        for part in [&character.look, &character.role] {
            if !part.trim().is_empty() {
                parts.push(clip(part, 40));
            }
        }
        clip(&parts.join(", "), 90)
    }

    /// Who a speaker or a listener is, for the notes of the translator: what identifies them,
    /// never their name. `key` is what the line holds, `desc` what the line says about them.
    fn who(&self, key: &str, desc: &str, speaker: bool) -> String {
        let desc = desc.trim();
        match key.trim().to_lowercase().as_str() {
            "narrator" | "narration" => return "the narrator".to_owned(),
            "self" => return "themself (a thought)".to_owned(),
            "everyone" => {
                return if desc.is_empty() {
                    "a group".to_owned()
                } else {
                    format!("a group ({})", clip(desc, 80))
                };
            }
            "" | "unknown" | "none" | "n/a" | "?" => {
                let who = if speaker { "an unknown speaker" } else { "an unknown listener" };
                return if desc.is_empty() { who.to_owned() } else { format!("{who} ({})", clip(desc, 80)) };
            }
            _ => {}
        }
        if let Some(index) = self.character_index(key) {
            let described = Self::describe(&self.characters[index]);
            if !described.is_empty() {
                return described;
            }
        }
        // A person the work holds nothing on: what the line says, else the label it gave.
        if desc.is_empty() { clip(key, 60) } else { clip(desc, 90) }
    }

    /// The glossary entries of the work: a profile for every character, the pair of words of
    /// every direction, and the names and terms.
    fn glossary_entries(&self) -> Vec<GlossaryEntry> {
        let mut entries = Vec::new();
        for character in &self.characters {
            let mut source = character.name.clone();
            for alias in &character.aliases {
                let _ = write!(source, " / {alias}");
            }
            entries.push(GlossaryEntry {
                kind: "character".to_owned(),
                source: clip(&source, PROFILE_CHARS),
                target: clip(&Self::profile(character), PROFILE_CHARS),
            });
        }
        for character in &self.characters {
            for relationship in &character.relationships {
                let (mine, yours) = (relationship.self_word.trim(), relationship.you.trim());
                if mine.is_empty() || yours.is_empty() {
                    continue;
                }
                // The listener under the name the character has now, not the label it was met by.
                let listener = self.character_index(&relationship.with).map_or_else(
                    || relationship.with.trim().to_owned(),
                    |index| self.characters[index].name.clone(),
                );
                entries.push(GlossaryEntry {
                    kind: "address".to_owned(),
                    source: format!("{} -> {listener}", character.name),
                    target: format!("{mine} - {yours}"),
                });
            }
        }
        for term in &self.terms {
            entries.push(GlossaryEntry {
                kind: term.kind.clone(),
                source: term.source.clone(),
                target: term.target.clone(),
            });
        }
        entries
    }
}

// ---------------------------------------------------------------------------------------
// The notes of one page.

/// Whether a correction of an OCR line is plausible: a few characters changed, not a new text.
fn accept_correction(source: &str, corrected: &str) -> Option<String> {
    let corrected = corrected.trim();
    if corrected.is_empty() || corrected == source.trim() {
        return None;
    }
    let (source_chars, corrected_chars): (Vec<char>, Vec<char>) =
        (source.trim().chars().collect(), corrected.chars().collect());
    let longest = source_chars.len().max(corrected_chars.len());
    if longest > 400 {
        let ratio = corrected_chars.len() as f64 / source_chars.len().max(1) as f64;
        return (0.5..=2.0).contains(&ratio).then(|| corrected.to_owned());
    }
    let allowed = (source_chars.len() / 2).max(3);
    (edit_distance(&source_chars, &corrected_chars) <= allowed).then(|| corrected.to_owned())
}

fn edit_distance(left: &[char], right: &[char]) -> usize {
    let mut previous: Vec<usize> = (0..=right.len()).collect();
    for (row, left_char) in left.iter().enumerate() {
        let mut current = vec![row + 1];
        for (column, right_char) in right.iter().enumerate() {
            let substitution = previous[column] + usize::from(left_char != right_char);
            current.push(substitution.min(previous[column + 1] + 1).min(current[column] + 1));
        }
        previous = current;
    }
    previous[right.len()]
}

/// What one line of a page is, according to the vision model.
#[derive(Clone, Debug, Default, PartialEq)]
struct AnalyzedLine {
    kind: String,
    speaker: String,
    listener: String,
    self_word: String,
    you: String,
    tone: String,
    /// How the speaker is related to the listener, when the work has said so.
    relation: String,
    corrected: Option<String>,
    speaker_desc: String,
    listener_desc: String,
    speaker_basis: String,
    listener_basis: String,
    audience: String,
    cues: String,
    exception: String,
    /// The first translation of the line, by the model that saw the page.
    draft: String,
    /// Who speaks and who listens as the notes describe them: sex, age, look; no names.
    speaker_who: String,
    listener_who: String,
}

/// What the vision model learned about one page: the notes for the translation model and the
/// corrections of the OCR text.
#[derive(Clone, Debug, PartialEq)]
pub struct PageAnalysis {
    /// The OCR lines the page was read with; the notes only apply while the page still has them.
    sources: Vec<String>,
    lines: Vec<AnalyzedLine>,
    /// What happens on the page (the scene when the model wrote no summary).
    summary: String,
}

impl PageAnalysis {
    fn new(output: &PageOutput, sources: &[String], bible: &Bible) -> Self {
        let mut lines = vec![AnalyzedLine::default(); sources.len()];
        for line in &output.lines {
            let Some(slot) = lines.get_mut(line.id) else {
                continue;
            };
            *slot = AnalyzedLine {
                kind: clip(&line.kind, 20).to_lowercase(),
                speaker: clip(&line.speaker, FIELD_CHARS),
                listener: clip(&line.listener, FIELD_CHARS),
                self_word: clip(&line.self_word, 40),
                you: clip(&line.you, 40),
                tone: clip(&line.tone, 60),
                relation: bible
                    .character_index(&line.speaker)
                    .and_then(|index| {
                        bible.characters[index]
                            .relationships
                            .iter()
                            .find(|relationship| same_name(&relationship.with, &line.listener))
                    })
                    .map(|relationship| clip(&relationship.relation, 60))
                    .unwrap_or_default(),
                corrected: accept_correction(&sources[line.id], &line.text),
                speaker_desc: clip(&line.speaker_desc, 90),
                listener_desc: clip(&line.listener_desc, 90),
                speaker_basis: line.speaker_basis.clone(),
                listener_basis: line.listener_basis.clone(),
                audience: line.audience.clone(),
                cues: clip(&line.cues, 80),
                exception: clip(&line.exception, 60),
                draft: clip(&line.draft, 800),
                speaker_who: String::new(),
                listener_who: String::new(),
            };
        }
        let mut page = Self {
            sources: sources.to_vec(),
            lines,
            summary: if output.summary.trim().is_empty() {
                clip(&output.scene, 400)
            } else {
                clip(&output.summary, 400)
            },
        };
        page.refresh_who(bible);
        page
    }

    /// Writes who speaks and who listens on every line the way the notes describe people (sex,
    /// age, look; never a name). Two people of the page that the
    /// description does not tell apart get a letter, so that a description is one person.
    fn refresh_who(&mut self, bible: &Bible) {
        for line in &mut self.lines {
            line.speaker_who = bible.who(&line.speaker, &line.speaker_desc, true);
            line.listener_who = bible.who(&line.listener, &line.listener_desc, false);
        }
        // The characters of the page by first appearance, with the description they share.
        let mut people: Vec<usize> = Vec::new();
        for line in &self.lines {
            for key in [&line.speaker, &line.listener] {
                if is_generic(key) {
                    continue;
                }
                if let Some(index) = bible.character_index(key) {
                    if !people.contains(&index) {
                        people.push(index);
                    }
                }
            }
        }
        let described = |index: usize| {
            let described = Bible::describe(&bible.characters[index]);
            if described.is_empty() { clip(&bible.characters[index].name, 60) } else { described }
        };
        let mut labels: Vec<(usize, String)> = Vec::new();
        for &index in &people {
            let base = described(index);
            let same: Vec<usize> = people.iter().copied().filter(|&other| described(other) == base).collect();
            let label = if same.len() > 1 {
                let letter = char::from(b'A' + u8::try_from(same.iter().position(|&other| other == index).unwrap_or(0)).unwrap_or(0));
                format!("{base} ({letter})")
            } else {
                base
            };
            labels.push((index, label));
        }
        let label_of = |key: &str| {
            bible
                .character_index(key)
                .and_then(|index| labels.iter().find(|(known, _)| *known == index))
                .map(|(_, label)| label.clone())
        };
        for line in &mut self.lines {
            if !is_generic(&line.speaker) {
                if let Some(label) = label_of(&line.speaker) {
                    line.speaker_who = label;
                }
            }
            if !is_generic(&line.listener) {
                if let Some(label) = label_of(&line.listener) {
                    line.listener_who = label;
                }
            }
        }
    }

    fn same_page(&self, current: &[String]) -> bool {
        self.sources == current
    }

    /// The text of the page's lines for translation: the OCR text with the corrections of the
    /// vision model. A page whose lines changed since it was read is left as it is.
    #[must_use]
    pub fn texts(&self, current: &[String]) -> Vec<String> {
        if !self.same_page(current) {
            return current.to_vec();
        }
        current
            .iter()
            .zip(&self.lines)
            .map(|(text, line)| line.corrected.clone().unwrap_or_else(|| text.clone()))
            .collect()
    }

    /// For every line of the page the words the notes decided for it, or `None` for a line the
    /// notes give no words for (narration, a thought, a sign). Lines to a group and lines of a
    /// speaker nobody can tell have their words too. Empty when the page changed since it was
    /// read. The translation is checked against these.
    #[must_use]
    pub fn expected_pairs(&self, current: &[String]) -> Vec<Option<ExpectedPair>> {
        if !self.same_page(current) {
            return Vec::new();
        }
        self.lines
            .iter()
            .map(|line| {
                let speech = line.kind.is_empty() || line.kind == "speech";
                let (mine, yours) = (line.self_word.trim(), line.you.trim());
                (speech && !mine.is_empty() && !yours.is_empty() && line.audience != "self").then(|| {
                    ExpectedPair {
                        speaker: line.speaker_who.clone(),
                        listener: line.listener_who.clone(),
                        mine: mine.to_owned(),
                        yours: yours.to_owned(),
                    }
                })
            })
            .collect()
    }

    /// The first translation of every line of the page, in the order of `current`; an empty text
    /// where the vision model wrote none. Empty when the page changed since it was read.
    #[must_use]
    pub fn drafts(&self, current: &[String]) -> Vec<String> {
        if !self.same_page(current) {
            return Vec::new();
        }
        self.lines.iter().map(|line| line.draft.trim().to_owned()).collect()
    }

    /// What was decided for every line of the page, one line each, for the pages that go with
    /// the next request: who speaks to whom and the words for "I" and "you".
    fn decisions(&self) -> String {
        let mut text = String::new();
        for (id, line) in self.lines.iter().enumerate() {
            if line.speaker.trim().is_empty() {
                continue;
            }
            let source = self.sources.get(id).map_or("", String::as_str);
            let quoted = serde_json::to_string(&clip(line.corrected.as_deref().unwrap_or(source), 60))
                .unwrap_or_default();
            let speaker = if line.speaker_who.is_empty() { "unknown" } else { &line.speaker_who };
            let listener = if line.listener_who.is_empty() { "unknown" } else { &line.listener_who };
            let _ = write!(text, "\n  {id}: {quoted} {speaker} -> {listener}");
            if !line.self_word.is_empty() || !line.you.is_empty() {
                let _ = write!(text, "; I \"{}\", you \"{}\"", line.self_word, line.you);
            }
        }
        text
    }

    /// What happens on the page.
    #[must_use]
    pub fn summary(&self) -> &str {
        &self.summary
    }


    /// How many lines the vision model corrected.
    #[must_use]
    pub fn corrections(&self) -> usize {
        self.lines.iter().filter(|line| line.corrected.is_some()).count()
    }

    /// The notes for the translation model, which does not see the picture: the scene, who is
    /// on the page, and for every line who says it to whom and with which words. Empty when the
    /// page changed since it was read.
    #[must_use]
    pub fn instructions(&self, current: &[String]) -> String {
        if !self.same_page(current) {
            return String::new();
        }
        let mut notes = String::from(
            "Decisions of a vision model that read the pictures of this page, one for every \
segment (the numbers are the ids of the segments): who speaks, to whom, the words the speaker \
uses for themselves and for the listener, and the tone. People are described by their sex, age \
and look, never by name: match every description to the person who speaks or listens. Use \
exactly these words for \"I\" and \"you\" in every line.",
        );
        notes.push_str("\nSegments:");
        for (id, line) in self.lines.iter().enumerate() {
            // A line the vision model said nothing about gets no note, instead of a made-up one.
            if line.kind.is_empty()
                && line.speaker.trim().is_empty()
                && line.self_word.is_empty()
                && line.you.is_empty()
            {
                continue;
            }
            let kind = if line.kind.is_empty() { "speech" } else { &line.kind };
            let speaker = if line.speaker_who.is_empty() { "an unknown speaker" } else { &line.speaker_who };
            let mut entry = format!("\n- {id} {kind}: {speaker}");
            // Only a spoken line has someone it is said to: a group and an unknown listener
            // included.
            if matches!(kind, "speech") && !line.listener.trim().is_empty() {
                let _ = write!(entry, " -> {}", line.listener_who);
                if !line.relation.is_empty() {
                    let _ = write!(entry, " ({})", line.relation);
                }
            }
            let mut words = Vec::new();
            if !line.self_word.is_empty() {
                words.push(format!("calls self \"{}\"", line.self_word));
            }
            if !line.you.is_empty() {
                let who = if line.audience == "group" { "calls the group" } else { "calls the listener" };
                words.push(format!("{who} \"{}\"", line.you));
            }
            if !words.is_empty() {
                let _ = write!(entry, "; {}", words.join(", "));
            }
            if !line.tone.is_empty() {
                let _ = write!(entry, "; tone: {}", line.tone);
            }
            if !line.exception.is_empty() {
                let _ = write!(entry, "; differs from their usual words: {}", line.exception);
            }
            // How sure the vision model is, said only when it is not sure.
            let mut doubts = Vec::new();
            for (side, basis) in [("speaker", &line.speaker_basis), ("listener", &line.listener_basis)] {
                if basis == "inferred" || basis == "unknown" {
                    doubts.push(format!("{side} {basis}"));
                }
            }
            if !doubts.is_empty() {
                let _ = write!(entry, "; ({})", doubts.join(", "));
            }
            if !line.cues.is_empty() {
                let _ = write!(entry, "; source cues: {}", line.cues);
            }
            if notes.chars().count() + entry.chars().count() > INSTRUCTIONS_CHARS {
                notes.push_str("\n- ...");
                break;
            }
            notes.push_str(&entry);
        }
        notes
    }
}

// ---------------------------------------------------------------------------------------
// Reading the work.

impl Translator {
    /// Whether the work is read by the vision models before it is translated: it is turned on
    /// in `[analysis]` and the OpenAI-compatible provider has a server of its own.
    #[must_use]
    pub fn analysis_enabled(&self) -> bool {
        let Ok(config) = analysis_config() else {
            return false;
        };
        if !config.enabled || config.models.is_empty() {
            return false;
        }
        let Ok(providers) = self.providers.read() else {
            return false;
        };
        !remote::is_loopback(providers.openai_compatible.base_url.as_ref())
    }

    /// Reads every page with the vision models (see the module). Returns the analysis of every
    /// page, `None` for a page no model could read, and merges what was learned about the whole
    /// work into the glossary.
    pub async fn analyze_work(
        &self,
        project: &str,
        language: Language,
        pages: &[AnalysisPage],
        stop: impl Fn() -> bool + Send + Sync,
    ) -> anyhow::Result<Vec<Option<PageAnalysis>>> {
        let config = analysis_config()?;
        let providers = self
            .providers
            .read()
            .map_err(|_| anyhow!("provider configuration lock is poisoned"))?
            .clone();
        let generation = GenerationConfig {
            temperature: Some(0.3),
            top_p: Some(0.95),
            max_tokens: Some(config.max_tokens),
            reasoning: Some(config.reasoning),
            ..GenerationConfig::default()
        };
        let system = system_prompt(language);
        let mut bible = Bible::default();
        let mut dead: HashSet<String> = HashSet::new();
        let mut results: Vec<Option<PageAnalysis>> = Vec::with_capacity(pages.len());
        // The scene of the page read last: the next page goes on from it.
        let mut previous = String::new();
        let usage_before = remote::token_usage();

        // The first request carries `size` new pages. Every later one starts with the last
        // `overlap` pages of the request before (already read, shown again for the story) and
        // fills the rest with new pages.
        let size = config.pages_per_request.max(1);
        let overlap = config.overlap_pages.min(size - 1);
        let mut summaries: Vec<(usize, String)> = Vec::new();
        let mut next = 0;
        while next < pages.len() {
            if stop() {
                results.resize(pages.len(), None);
                break;
            }
            let from = next.saturating_sub(overlap);
            let end = (next + size - (next - from)).min(pages.len());
            let context: Vec<ContextPage<'_>> = (from..next)
                .map(|number| ContextPage {
                    number,
                    page: &pages[number],
                    decided: results[number]
                        .as_ref()
                        .map(PageAnalysis::decisions)
                        .unwrap_or_default(),
                })
                .collect();
            let story = story_so_far(&summaries, config.story_chars);
            let outputs = self
                .read_batch(
                    &config,
                    &providers,
                    &generation,
                    &system,
                    (next, pages.len()),
                    &pages[next..end],
                    &context,
                    &story,
                    &bible,
                    &previous,
                    &mut dead,
                )
                .await;
            for (offset, (page, output)) in pages[next..end].iter().zip(outputs).enumerate() {
                let number = next + offset;
                match output {
                    Some((model, output)) => {
                        previous = clip(&output.scene, 400);
                        bible.merge_output(&output);
                        let analysis = PageAnalysis::new(&output, &page.segments, &bible);
                        tracing::info!(
                            page = number + 1,
                            %model,
                            corrections = analysis.corrections(),
                            "page analysed"
                        );
                        summaries.push((number, analysis.summary().to_owned()));
                        results.push(Some(analysis));
                    }
                    None => {
                        tracing::warn!(page = number + 1, "no analysis model could read the page");
                        results.push(None);
                    }
                }
            }
            next = end;
        }

        let settled = bible.settle_pairs();
        tracing::info!(settled, "directions settled by the number of lines that used a pair");
        // What most lines of a direction used is its pair, so that a line that strayed does not
        // make the notes of the pages disagree.
        for page in results.iter_mut().flatten() {
            page.apply_pairs(&bible);
        }
        for problem in bible.address_conflicts() {
            tracing::warn!(%problem, "a settled pair cannot be right");
        }

        // What the whole work decided replaces what an earlier run of it wrote.
        let restarted = match self.glossary.lock() {
            Ok(mut glossary) => {
                let restarted = glossary.begin(project, language);
                glossary.replace(bible.glossary_entries());
                restarted
            }
            Err(_) => false,
        };
        if restarted && let Ok(mut memory) = self.memory.lock() {
            // Another work (or language): its first page must not see the lines of the last.
            memory.recent.clear();
            memory.recent_pages.clear();
        }
        // The pairs the translations locked in an earlier run are stale; the pairs of this
        // reading are what the translation is held to.
        if let Ok(mut book) = self.pairs.lock() {
            book.clear();
            let seeded = settle::seed_pairs(&mut book, &bible, &results);
            tracing::info!(seeded, "pronoun pairs of the analysis given to the translation");
        }
        tracing::info!(
            characters = bible.characters.len(),
            terms = bible.terms.len(),
            pages = results.iter().filter(|page| page.is_some()).count(),
            tokens = %remote::token_usage().since(&usage_before),
            "analysis finished"
        );
        Ok(results)
    }

    /// Reads the pages of one batch. Every model of the chain is asked for the pages that are
    /// still unread (all of them for the first model); a model that fails, or leaves a page out
    /// of its answer, leaves that page to the next one. A batch of one page is read the way a
    /// single page is, with its panels when it has them. The result has one entry for every
    /// page of `batch`: the model that read it and what it read, `None` when none could.
    /// `place` is the number of the first page of the batch (from 0) and the number of pages;
    /// `context` are the pages of the request before that go along with it and `story` what
    /// happened on the pages read so far.
    #[allow(clippy::too_many_arguments)]
    async fn read_batch(
        &self,
        config: &AnalysisConfig,
        providers: &ProvidersConfig,
        generation: &GenerationConfig,
        system: &str,
        place: (usize, usize),
        batch: &[AnalysisPage],
        context: &[ContextPage<'_>],
        story: &str,
        bible: &Bible,
        previous: &str,
        dead: &mut HashSet<String>,
    ) -> Vec<Option<(String, PageOutput)>> {
        let (first, total) = place;
        let mut outputs: Vec<Option<(String, PageOutput)>> = vec![None; batch.len()];
        for entry in &config.models {
            let unread: Vec<usize> = (0..batch.len())
                .filter(|&index| outputs[index].is_none())
                .collect();
            if unread.is_empty() {
                break;
            }
            if dead.contains(&entry.model) {
                continue;
            }
            let counts: Vec<usize> = unread.iter().map(|&index| batch[index].segments.len()).collect();
            let (chat, single) = if let ([index], true) = (unread.as_slice(), context.is_empty()) {
                let page = &batch[*index];
                let number = first + index;
                // The whole page comes first, so that the panels that follow are read in its context.
                let (images, view) = if !entry.vision {
                    (Vec::new(), View::Text)
                } else if page.panels.is_empty() {
                    (vec![page.picture.clone()], View::Page)
                } else {
                    let mut images = vec![page.picture.clone()];
                    images.extend(page.panels.iter().map(|panel| panel.picture.clone()));
                    (images, View::PageAndPanels(&page.panels))
                };
                tracing::info!(
                    page = number + 1,
                    model = %entry.model,
                    images = images.len(),
                    panels = page.panels.len(),
                    view = match &view {
                        View::Text => "text only",
                        View::Page => "whole page",
                        View::PageAndPanels(_) => "whole page and panels",
                    },
                    "sending the page to the analysis model"
                );
                let chat = remote::Chat {
                    system: system.to_owned(),
                    // The scene before the batch only precedes the first page of it.
                    user: user_prompt(
                        number,
                        total,
                        &page.segments,
                        bible,
                        (*index == 0).then_some(previous),
                        &view,
                    ),
                    images,
                    schema_name: "manga_page_analysis",
                    schema: schema(counts[0]),
                };
                (chat, true)
            } else {
                let pages: Vec<&AnalysisPage> = unread.iter().map(|&index| &batch[index]).collect();
                // The pages already read come first, then the new ones.
                let images: Vec<String> = if entry.vision {
                    context
                        .iter()
                        .map(|page| page.page)
                        .chain(pages.iter().copied())
                        .map(|page| page.picture.clone())
                        .collect()
                } else {
                    Vec::new()
                };
                tracing::info!(
                    pages = pages.len(),
                    context = context.len(),
                    from = first + unread[0] + 1,
                    model = %entry.model,
                    images = images.len(),
                    "sending a batch of pages to the analysis model"
                );
                // After a failure the unread pages may not be next to each other: each one is
                // named by its own number.
                let numbers: Vec<usize> = unread.iter().map(|&index| first + index).collect();
                let chat = remote::Chat {
                    system: system.to_owned(),
                    user: batch_prompt(
                        &numbers,
                        total,
                        &pages,
                        context,
                        story,
                        bible,
                        entry.vision,
                    ),
                    images,
                    schema_name: "manga_pages_analysis",
                    schema: batch_schema(&counts),
                };
                (chat, false)
            };
            let generation = GenerationConfig {
                reasoning: Some(entry.reasoning.unwrap_or(config.reasoning)),
                ..*generation
            };
            let attempt = remote::chat_json(&self.client, providers, &entry.model, &generation, chat)
                .await
                .map_err(|error| {
                    if matches!(&error, Error::Api { status, .. } if matches!(*status, 401 | 403 | 404)) {
                        dead.insert(entry.model.clone());
                    }
                    anyhow::Error::from(error)
                })
                .and_then(|text| {
                    if single {
                        parse_page(&text, counts[0]).map(|output| vec![Some(output)])
                    } else {
                        parse_batch(&text, &counts)
                    }
                });
            match attempt {
                Ok(read) => {
                    for (&index, output) in unread.iter().zip(read) {
                        if let Some(output) = output {
                            outputs[index] = Some((entry.model.clone(), output));
                        }
                    }
                    let missing = outputs.iter().filter(|output| output.is_none()).count();
                    if missing > 0 {
                        tracing::warn!(
                            model = %entry.model,
                            missing,
                            "the analysis model left pages out; the next one reads them"
                        );
                    }
                }
                Err(error) => tracing::warn!(
                    from = first + unread[0] + 1,
                    pages = unread.len(),
                    model = %entry.model,
                    error = %format!("{error:#}"),
                    "analysis model could not read the pages; trying the next one"
                ),
            }
        }
        outputs
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sources(lines: &[&str]) -> Vec<String> {
        lines.iter().map(|line| (*line).to_owned()).collect()
    }

    const ANSWER: &str = r#"{
        "scene": "A hospital room.",
        "page_note": "The young man is a colleague of the woman.",
        "characters": [
            {"name": "Sasaki", "gender": "man", "age": "young adult", "role": "Rio's colleague",
             "speech_style": "rough but kind",
             "relationships": [{"with": "Rio", "relation": "colleague", "self": "tôi", "you": "chị"}]},
            {"name": null}
        ],
        "terms": [{"kind": "name", "source": "りお", "target": "Rio"}],
        "lines": [
            {"id": 0, "kind": "speech", "text": "おい ゆう", "speaker": "Sasaki", "listener": "Yuu",
             "self": "tôi", "you": "cậu", "tone": "teasing"},
            {"id": "1", "kind": "speech", "text": "りおさん、さみしがってるぞ", "speaker": "Sasaki",
             "listener": "Yuu", "self": "tôi", "you": "cậu"},
            {"id": 7, "text": "outside", "speaker": "x", "listener": "y"},
            {"id": 1, "text": "duplicate", "speaker": "x", "listener": "y"}
        ]
    }"#;

    #[test]
    fn an_answer_with_nulls_string_ids_and_stray_lines_is_read() {
        let output = parse_page(ANSWER, 3).unwrap();
        assert_eq!(output.lines.len(), 2);
        assert_eq!(output.lines[1].id, 1);
        assert_eq!(output.lines[1].text, "りおさん、さみしがってるぞ");
        assert_eq!(output.characters[1].name, "");
        assert_eq!(output.lines[0].self_word, "tôi");
    }

    #[test]
    fn text_around_the_json_is_ignored_and_an_empty_answer_is_an_error() {
        let wrapped = format!("Here you go:\n```json\n{ANSWER}\n```");
        assert!(parse_page(&wrapped, 3).is_ok());
        assert!(parse_page("I cannot help with that.", 3).is_err());
        assert!(parse_page(r#"{"lines": []}"#, 3).is_err());
        assert!(parse_page(r#"{"lines": [{"id": 9, "speaker": "a", "listener": "b"}]}"#, 3).is_err());
    }

    #[test]
    fn the_work_is_remembered_without_flipping_what_is_known() {
        let mut bible = Bible::default();
        bible.merge_output(&parse_page(ANSWER, 3).unwrap());
        assert_eq!(bible.characters.len(), 1, "an unnamed character is not kept");
        assert_eq!(bible.terms.len(), 1);
        let later = PageOutput {
            characters: vec![Character {
                name: "sasaki".into(),
                aliases: vec!["Sasaki-kun".into()],
                gender: "woman".into(),
                age: "30".into(),
                relationships: vec![Relationship {
                    with: "Rio".into(),
                    self_word: "anh".into(),
                    you: "em".into(),
                    ..Relationship::default()
                }],
                ..Character::default()
            }],
            ..PageOutput::default()
        };
        bible.merge_output(&later);
        let sasaki = &bible.characters[0];
        assert_eq!(sasaki.gender, "man", "a known gender stays");
        assert_eq!(sasaki.age, "young adult");
        assert!(sasaki.aliases.iter().any(|alias| alias == "Sasaki-kun"));
        let rio = sasaki.relationships.iter().find(|r| r.with == "Rio").unwrap();
        assert_eq!((rio.self_word.as_str(), rio.you.as_str()), ("tôi", "chị"));
        // The pairs of the lines count too: Sasaki -> Yuu came from line 0 only.
        assert!(sasaki.relationships.iter().any(|r| r.with == "Yuu" && r.you == "cậu"));
    }

    #[test]
    fn the_glossary_gets_profiles_pairs_and_names() {
        let mut bible = Bible::default();
        bible.merge_output(&parse_page(ANSWER, 3).unwrap());
        let entries = bible.glossary_entries();
        let find = |kind: &str, source: &str| {
            entries.iter().find(|entry| entry.kind == kind && entry.source == source)
        };
        assert!(find("character", "Sasaki").unwrap().target.contains("young adult"));
        assert_eq!(find("address", "Sasaki -> Rio").unwrap().target, "tôi - chị");
        assert_eq!(find("address", "Sasaki -> Yuu").unwrap().target, "tôi - cậu");
        assert_eq!(find("name", "りお").unwrap().target, "Rio");
    }

    #[test]
    fn only_small_corrections_of_the_ocr_text_are_accepted() {
        assert_eq!(accept_correction("さみしがつてるぞ", "さみしがってるぞ").as_deref(), Some("さみしがってるぞ"));
        assert_eq!(accept_correction("同じ", "同じ"), None);
        assert_eq!(accept_correction("同じ", "  "), None);
        // A different text is not a correction.
        assert_eq!(accept_correction("おい ゆう", "全然違う長いテキストです"), None);
        assert_eq!(edit_distance(&['a', 'b'], &['a', 'c']), 1);
    }

    #[test]
    fn a_page_gets_its_corrected_text_and_its_notes() {
        let lines = sources(&["おい ゆう", "りおさん さみしがつてるぞ", "…"]);
        let output = parse_page(ANSWER, 3).unwrap();
        let mut bible = Bible::default();
        bible.merge_output(&output);
        let analysis = PageAnalysis::new(&output, &lines, &bible);
        assert_eq!(analysis.corrections(), 1, "line 1 was misread, line 0 only differs in spacing");
        let texts = analysis.texts(&lines);
        assert_eq!(texts[0], "おい ゆう");
        assert_eq!(texts[1], "りおさん、さみしがってるぞ");
        assert_eq!(texts[2], "…");
        let notes = analysis.instructions(&lines);
        assert!(notes.contains("never by name"));
        // The decisions for the lines go to the model of the translation, the story does not.
        assert!(!notes.contains("Scene:") && !notes.contains("Note:"), "{notes}");
        assert!(!notes.contains("People on this page"), "{notes}");
        assert!(!notes.contains("A hospital room"), "{notes}");
        // The speaker is described, not named; Yuu is nobody the work knows, so the label stays.
        assert!(notes.contains(
            "- 0 speech: young adult man, Rio's colleague -> Yuu; calls self \"tôi\", calls the listener \"cậu\"; tone: teasing"
        ));
        assert!(!notes.contains("Sasaki"), "the notes never name a person");
        assert!(notes.chars().count() <= INSTRUCTIONS_CHARS + 20);
        assert!(!notes.contains("- 2 "), "a line nobody described has no note");
        // A page whose lines changed since it was read gets no notes and no corrections.
        let edited = sources(&["おい ゆう!", "x", "…"]);
        assert!(analysis.instructions(&edited).is_empty());
        assert_eq!(analysis.texts(&edited), edited);
    }

    #[test]
    fn the_prompts_carry_the_rules_and_the_lines() {
        let system = system_prompt(Language::Vietnamese);
        assert!(system.contains("translated into Vietnamese"));
        assert!(system.contains("never from hair, glasses or build alone"));
        assert!(system.contains("A person is identified by what they are, not by a name"));
        assert!(system.contains("audience: one"));
        assert!(system.contains("Words to a group are not the pair of a person"));
        assert!(system.contains("To several people (audience group)"));
        assert!(system.contains("An unknown speaker gets words that name no sex"));
        assert!(system.contains("never 'con' or 'cháu'"));
        assert!(system.contains("for men and boys only"), "a word of one sex is not for the other");
        assert!(system.contains("The two directions of a pair fit each other"));
        assert!(!system_prompt(Language::English).contains("Vietnamese pronouns"));
        let mut bible = Bible::default();
        assert!(user_prompt(0, 2, &sources(&["a"]), &bible, None, &View::Page).contains("Known so far: nothing yet."));
        bible.merge_output(&parse_page(ANSWER, 3).unwrap());
        let user = user_prompt(1, 2, &sources(&["おい", "ゆう"]), &bible, Some("A hospital room."), &View::Text);
        assert!(user.starts_with("Page 2 of 2. No picture is attached"));
        assert!(user.contains("\"Sasaki\""));
        assert!(user.contains("The previous page ended like this (the story goes on from it): A hospital room."));
        assert!(user.contains("1: \"ゆう\""));
        assert!(user.ends_with("(ids 0 to 1)."));
    }

    #[test]
    fn panels_are_read_in_rows_from_top_to_bottom() {
        // Two panels side by side (the left one starts a little higher), one below them.
        let panels = [[0.0, 0.0, 95.0, 100.0], [100.0, 4.0, 200.0, 100.0], [0.0, 110.0, 200.0, 200.0]];
        assert_eq!(reading_order(&panels, true), [1, 0, 2], "right to left");
        assert_eq!(reading_order(&panels, false), [0, 1, 2], "left to right");
        // Three tiers of different heights: the tall one on the right is read first.
        let tiers = [
            [100.0, 0.0, 200.0, 200.0],
            [0.0, 0.0, 95.0, 90.0],
            [0.0, 100.0, 95.0, 200.0],
        ];
        assert_eq!(reading_order(&tiers, true), [0, 1, 2]);
        assert!(reading_order(&[], true).is_empty());
    }

    #[test]
    fn a_line_goes_with_the_smallest_panel_that_holds_it_or_the_nearest_one() {
        let panels = [[0.0, 0.0, 200.0, 200.0], [10.0, 10.0, 60.0, 60.0], [300.0, 0.0, 400.0, 100.0]];
        let boxes = [
            Some([20.0, 20.0, 40.0, 40.0]),   // inside the small panel too: the small one
            Some([100.0, 100.0, 120.0, 120.0]), // only in the large one
            Some([250.0, 40.0, 270.0, 60.0]), // in none: the nearest, the third
            None,                             // no box: with the line before it
        ];
        assert_eq!(assign_segments(&panels, &boxes), [1, 0, 2, 2]);
        assert_eq!(assign_segments(&panels, &[None]), [0]);
    }

    #[test]
    fn a_panel_is_cut_with_a_margin_inside_the_page() {
        assert_eq!(crop_rect([50.0, 50.0, 150.0, 150.0], 1000, 1000), Some((41, 41, 118, 118)));
        assert_eq!(crop_rect([0.0, 0.0, 100.0, 100.0], 105, 1000), Some((0, 0, 105, 109)));
        assert_eq!(crop_rect([10.0, 10.0, 20.0, 20.0], 1000, 1000), None, "too small");
    }

    #[test]
    fn the_panels_are_listed_in_the_prompt() {
        let panel = |lines: Vec<usize>| PanelPicture {
            picture: String::new(),
            lines,
        };
        let panels = [panel(vec![0, 1]), panel(vec![2])];
        let user = user_prompt(0, 3, &sources(&["a", "b", "c"]), &Bible::default(), None, &View::PageAndPanels(&panels));
        assert!(user.contains("3 pictures are attached. Picture 1 is the whole page"));
        assert!(user.contains("The next 2 pictures are the panels"));
        assert!(user.contains("Panel 1: lines 0, 1\nPanel 2: lines 2\n"));
        assert!(!user.contains("The previous page"), "the first page has none");
        assert!(user.contains("2: \"c\""));
    }

    #[test]
    fn the_schema_bounds_the_ids() {
        let value = schema(5);
        assert_eq!(value["properties"]["lines"]["items"]["properties"]["id"]["maximum"], 4);
        assert_eq!(value["required"][0], "lines");
    }

    #[test]
    fn the_default_chain_is_the_vision_model_then_the_small_one() {
        let config = AnalysisConfig::default();
        assert!(config.models.iter().all(|model| model.vision));
        assert_eq!(config.models.len(), 2);
        assert_eq!(config.models[0].model, NEMOTRON_VL);
        assert_eq!(config.models[1].model, NEMOTRON_BOLT);
        assert!(config.right_to_left);
        // Twelve pictures go in one request, two of them the last pages of the request before,
        // and the model of the translation writes the final text.
        assert_eq!(config.pages_per_request, 12);
        assert_eq!(config.overlap_pages, 2);
        assert!(config.translate);
        // Both models think.
        assert!(config.reasoning);
        assert!(config.models.iter().all(|model| model.reasoning == Some(true)));
        // An old file that names the panels of a model or the settings of the second reading
        // is still read; those settings are gone.
        let parsed: AnalysisConfig = serde_json::from_str(
            r#"{"models": [{"model": "a/b", "panels": true}], "review": true, "review_models": []}"#,
        )
        .unwrap();
        assert!(parsed.enabled && parsed.models.len() == 1 && parsed.models[0].vision);
    }

    #[test]
    fn a_page_tells_the_translation_the_words_of_every_speech_line() {
        let lines = sources(&["おい ゆう", "りおさん さみしがつてるぞ", "…"]);
        let output = parse_page(ANSWER, 3).unwrap();
        let mut bible = Bible::default();
        bible.merge_output(&output);
        let analysis = PageAnalysis::new(&output, &lines, &bible);
        let pairs = analysis.expected_pairs(&lines);
        assert_eq!(pairs.len(), 3);
        let first = pairs[0].as_ref().unwrap();
        assert_eq!((first.speaker.as_str(), first.listener.as_str()), ("young adult man, Rio's colleague", "Yuu"));
        assert_eq!((first.mine.as_str(), first.yours.as_str()), ("tôi", "cậu"));
        assert!(pairs[2].is_none(), "a line nobody described has no pair");
        // A page that changed since it was read has none.
        assert!(analysis.expected_pairs(&sources(&["x", "y", "z"])).is_empty());
    }

    #[test]
    fn the_pair_most_lines_used_wins_over_the_first_line() {
        let page = |lines: &str| parse_page(lines, 4).unwrap();
        let mut bible = Bible::default();
        // The first line of the direction is the stray one.
        bible.merge_output(&page(
            r#"{"lines": [
                {"id": 0, "speaker": "Sasaki", "listener": "Rio", "self": "tôi", "you": "cô"},
                {"id": 1, "speaker": "Sasaki", "listener": "Rio", "self": "anh", "you": "em"},
                {"id": 2, "speaker": "Sasaki", "listener": "Rio", "self": "anh", "you": "em"}]}"#,
        ));
        assert_eq!(bible.characters[0].relationships[0].you, "cô", "the first line came first");
        assert_eq!(bible.settle_pairs(), 1);
        let rio = &bible.characters[0].relationships[0];
        assert_eq!((rio.self_word.as_str(), rio.you.as_str()), ("anh", "em"));
        // Nothing left to settle.
        assert_eq!(bible.settle_pairs(), 0);
    }

    #[test]
    fn a_tie_keeps_the_pair_that_came_first_and_a_label_counts_under_its_name() {
        let mut bible = Bible::default();
        bible.merge_output(&parse_page(
            r#"{"lines": [
                {"id": 0, "speaker": "man with glasses", "listener": "Rio", "self": "tôi", "you": "cô"},
                {"id": 1, "speaker": "man with glasses", "listener": "Rio", "self": "anh", "you": "em"}]}"#,
            3,
        ).unwrap());
        // The name is learned later: the label becomes an alias.
        bible.merge_output(&PageOutput {
            characters: vec![Character {
                name: "Sasaki".into(),
                aliases: vec!["man with glasses".into()],
                ..Character::default()
            }],
            lines: vec![Line {
                id: 0,
                speaker: "Sasaki".into(),
                listener: "Rio".into(),
                self_word: "anh".into(),
                you: "em".into(),
                ..Line::default()
            }],
            ..PageOutput::default()
        });
        bible.settle_pairs();
        let known = bible.character_index("Sasaki").unwrap();
        let rio = &bible.characters[known].relationships[0];
        assert_eq!((rio.self_word.as_str(), rio.you.as_str()), ("anh", "em"), "two votes for anh - em");
    }

    fn notes_of(json: &str, lines: usize, characters: &[&str]) -> (Bible, String) {
        let mut bible = Bible::default();
        for character in characters {
            bible.merge_character(serde_json::from_str(character).unwrap());
        }
        let output = parse_page(json, lines).unwrap();
        bible.merge_output(&output);
        let sources: Vec<String> = (0..lines).map(|id| format!("line {id}")).collect();
        let analysis = PageAnalysis::new(&output, &sources, &bible);
        let notes = analysis.instructions(&sources);
        (bible, notes)
    }

    #[test]
    fn a_line_to_a_group_or_from_a_voice_nobody_sees_is_not_a_direction() {
        let (bible, notes) = notes_of(
            r#"{"lines": [
                {"id": 0, "kind": "speech", "speaker": "Sasaki", "listener": "the three girls",
                 "audience": "group", "listener_desc": "three teenage girls", "self": "cô", "you": "các em",
                 "tone": "stern"},
                {"id": 1, "kind": "speech", "speaker": "man at the door", "speaker_basis": "unknown",
                 "speaker_desc": "deep male voice", "listener": "Sasaki", "self": "tôi", "you": "cô"},
                {"id": 2, "kind": "speech", "speaker": "Sasaki", "listener": "someone", "listener_basis": "unknown",
                 "self": "tôi", "you": "bạn", "tone": "calm"}
            ]}"#,
            3,
            &[r#"{"name": "Sasaki", "gender": "woman", "age": "adult", "look": "black bun, white coat", "role": "teacher"}"#],
        );
        // Nobody is counted as someone's listener or as a speaker for these lines.
        assert!(bible.tally.is_empty(), "{:?}", bible.tally);
        assert!(bible.characters[0].relationships.is_empty());
        assert!(notes.contains("a group (three teenage girls)"), "{notes}");
        assert!(notes.contains("calls the group \"các em\""), "{notes}");
        assert!(notes.contains("an unknown speaker (deep male voice)"), "{notes}");
        assert!(notes.contains("(speaker unknown)"), "{notes}");
        assert!(notes.contains("an unknown listener (someone)"), "{notes}");
        assert!(notes.contains("(listener unknown)"), "{notes}");
        assert!(notes.contains("adult woman, black bun, white coat, teacher"), "{notes}");
    }

    #[test]
    fn a_line_that_says_who_it_is_said_to_by_the_context_is_marked_inferred() {
        let (_, notes) = notes_of(
            r#"{"lines": [{"id": 0, "kind": "speech", "speaker": "Sasaki", "speaker_basis": "inferred",
                "listener": "Rio", "listener_basis": "seen", "self": "anh", "you": "em",
                "cues": "俺/お前, だぜ"}]}"#,
            1,
            &[
                r#"{"name": "Sasaki", "gender": "man", "age": "young adult"}"#,
                r#"{"name": "Rio", "gender": "girl", "age": "teenager"}"#,
            ],
        );
        assert!(notes.contains("young adult man -> teenager girl; calls self \"anh\", calls the listener \"em\""), "{notes}");
        assert!(notes.contains("(speaker inferred)") && !notes.contains("listener seen"), "{notes}");
        assert!(notes.contains("source cues: 俺/お前, だぜ"), "{notes}");
    }

    #[test]
    fn two_people_a_description_does_not_tell_apart_get_a_letter_and_the_others_do_not() {
        let (_, notes) = notes_of(
            r#"{"lines": [
                {"id": 0, "kind": "speech", "speaker": "Mika", "listener": "Rina", "self": "tớ", "you": "cậu"},
                {"id": 1, "kind": "speech", "speaker": "Rina", "listener": "Mika", "self": "tớ", "you": "cậu"},
                {"id": 2, "kind": "speech", "speaker": "Mika", "listener": "Ken", "self": "em", "you": "anh"}
            ]}"#,
            3,
            &[
                r#"{"name": "Mika", "gender": "girl", "age": "teenager"}"#,
                r#"{"name": "Rina", "gender": "girl", "age": "teenager"}"#,
                r#"{"name": "Ken", "gender": "man", "age": "young adult", "look": "glasses"}"#,
            ],
        );
        assert!(notes.contains("- 0 speech: teenager girl (A) -> teenager girl (B)"), "{notes}");
        assert!(notes.contains("- 1 speech: teenager girl (B) -> teenager girl (A)"), "{notes}");
        assert!(notes.contains("-> young adult man, glasses;"), "{notes}");
        for name in ["Mika", "Rina", "Ken"] {
            assert!(!notes.contains(name), "{name} in {notes}");
        }
    }

    #[test]
    fn the_look_of_a_person_is_kept_from_the_lines_and_an_exception_is_no_evidence() {
        let mut bible = Bible::default();
        bible.merge_output(
            &parse_page(
                r#"{"lines": [
                    {"id": 0, "speaker": "man with glasses", "speaker_desc": "tall, glasses, grey suit",
                     "listener": "Rio", "listener_desc": "girl in a uniform", "self": "tôi", "you": "cô"},
                    {"id": 1, "speaker": "man with glasses", "listener": "Rio", "self": "tao", "you": "mày",
                     "exception": "quarrel"}
                ]}"#,
                2,
            )
            .unwrap(),
        );
        let known = bible.character_index("man with glasses").unwrap();
        assert_eq!(bible.characters[known].look, "tall, glasses, grey suit");
        // Only the first line is a vote for the direction; the quarrel is not.
        assert_eq!(bible.tally.len(), 1);
        assert_eq!(bible.tally[0].1, vec![(("tôi".to_owned(), "cô".to_owned()), 1)]);
        assert_eq!(bible.characters[known].relationships[0].you, "cô");
    }

    #[test]
    fn the_audience_and_the_basis_are_read_the_way_models_write_them() {
        let output = parse_page(
            r#"{"lines": [
                {"id": 0, "speaker": "A", "listener": "the class", "audience": "Crowd"},
                {"id": 1, "speaker": "A", "listener": "A", "audience": "monologue"},
                {"id": 2, "speaker": "A", "listener": "B", "audience": "someone else", "speaker_basis": "Seen"},
                {"id": 3, "speaker": "A", "listener": "B", "audience": "unclear"}
            ]}"#,
            4,
        )
        .unwrap();
        let by = |id: usize| &output.lines[id];
        assert_eq!((by(0).audience.as_str(), by(0).listener.as_str(), by(0).listener_desc.as_str()), ("group", "everyone", "the class"));
        assert_eq!((by(1).audience.as_str(), by(1).listener.as_str()), ("self", "self"));
        assert_eq!((by(2).audience.as_str(), by(2).listener.as_str(), by(2).speaker_basis.as_str()), ("one", "B", "seen"));
        assert_eq!((by(3).audience.as_str(), by(3).listener.as_str(), by(3).listener_desc.as_str()), ("unknown", "unknown", "B"));
    }

    #[test]
    fn a_group_and_an_unknown_speaker_have_the_words_the_translation_is_held_to() {
        let lines = sources(&["a", "b", "c"]);
        let (bible, _) = notes_of(
            r#"{"lines": [
                {"id": 0, "kind": "speech", "speaker": "Sasaki", "listener": "x", "audience": "group",
                 "listener_desc": "the students", "self": "cô", "you": "các em"},
                {"id": 1, "kind": "speech", "speaker": "x", "speaker_basis": "unknown", "listener": "Sasaki",
                 "speaker_desc": "a man's voice", "self": "tôi", "you": "cô"},
                {"id": 2, "kind": "thought", "speaker": "Sasaki", "listener": "self", "self": "mình"}
            ]}"#,
            3,
            &[r#"{"name": "Sasaki", "gender": "woman", "age": "adult", "role": "teacher"}"#],
        );
        let output = parse_page(
            r#"{"lines": [
                {"id": 0, "kind": "speech", "speaker": "Sasaki", "listener": "x", "audience": "group",
                 "listener_desc": "the students", "self": "cô", "you": "các em"},
                {"id": 1, "kind": "speech", "speaker": "x", "speaker_basis": "unknown", "listener": "Sasaki",
                 "speaker_desc": "a man's voice", "self": "tôi", "you": "cô"},
                {"id": 2, "kind": "thought", "speaker": "Sasaki", "listener": "self", "self": "mình"}
            ]}"#,
            3,
        )
        .unwrap();
        let analysis = PageAnalysis::new(&output, &lines, &bible);
        let pairs = analysis.expected_pairs(&lines);
        let group = pairs[0].as_ref().unwrap();
        assert_eq!((group.mine.as_str(), group.yours.as_str()), ("cô", "các em"));
        assert_eq!(group.listener, "a group (the students)");
        let unseen = pairs[1].as_ref().unwrap();
        assert_eq!(unseen.speaker, "an unknown speaker (a man's voice)");
        assert!(pairs[2].is_none(), "a thought has no pair to hold the translation to");
    }

    #[test]
    fn the_vision_model_is_asked_for_a_draft_of_every_line_and_a_summary_of_the_page() {
        let system = system_prompt(Language::Vietnamese);
        assert!(system.contains("You are the first translator"));
        assert!(system.contains("- draft: the translation of `text` into Vietnamese"));
        assert!(system.contains("summary (one to three sentences"));
        let schema = schema(2);
        let line = &schema["properties"]["lines"]["items"];
        assert!(line["required"].as_array().unwrap().iter().any(|name| name == "draft"));
        assert!(line["properties"]["draft"].is_object());
        assert!(schema["properties"]["summary"].is_object());
        assert!(schema["required"].as_array().unwrap().iter().any(|name| name == "summary"));
    }

    #[test]
    fn a_page_keeps_the_drafts_the_summary_and_hands_the_final_translator_everything() {
        let lines = sources(&["おい、ゆう", "…", "りおさん"]);
        let output = parse_page(
            r#"{"summary": "Sasaki teases Yuu about Rio.", "lines": [
                {"id": 0, "kind": "speech", "speaker": "Sasaki", "listener": "Yuu", "self": "tôi", "you": "cậu",
                 "draft": "  Này, Yuu.  "},
                {"id": 1, "kind": "sfx"},
                {"id": 2, "kind": "speech", "speaker": "Sasaki", "listener": "Yuu", "self": "tôi", "you": "cậu",
                 "draft": "Rio-san kìa."}]}"#,
            3,
        )
        .unwrap();
        let mut bible = Bible::default();
        bible.merge_output(&output);
        let analysis = PageAnalysis::new(&output, &lines, &bible);
        assert_eq!(analysis.drafts(&lines), ["Này, Yuu.", "", "Rio-san kìa."]);
        assert_eq!(analysis.summary(), "Sasaki teases Yuu about Rio.");
        assert!(analysis.drafts(&sources(&["x", "y", "z"])).is_empty(), "the page changed");

        // Without a summary the scene stands in for it.
        let scene = parse_page(r#"{"scene": "A hospital room.", "lines": [{"id": 0, "speaker": "A", "listener": "B"}]}"#, 1).unwrap();
        let analysis_scene = PageAnalysis::new(&scene, &sources(&["a"]), &Bible::default());
        assert_eq!(analysis_scene.summary(), "A hospital room.");

        let request = crate::polish::polish_request(
            &analysis,
            &lines,
            Language::Vietnamese,
            Some("Keep it light."),
        )
        .unwrap();
        let instructions = request.instructions.clone().unwrap();
        // The user's words come first, then the decisions for the lines. No story.
        let user = instructions.find("Keep it light.").unwrap();
        let notes = instructions.find("Decisions of a vision model").unwrap();
        assert!(user < notes, "{instructions}");
        assert!(!instructions.contains("Story of the work"), "{instructions}");
        assert!(request.standalone, "no remembered lines and no profiles are added");
                assert_eq!(request.drafts, ["Này, Yuu.", "", "Rio-san kìa."]);
        assert!(request.has_notes() && request.has_drafts() && request.image.is_none());
        assert_eq!(request.pairs.len(), 3);
        assert!(request.pairs[1].is_none(), "a sound effect has no pair");
        // Notes of another page are no notes for this one.
        assert!(
            crate::polish::polish_request(&analysis, &sources(&["x"]), Language::Vietnamese, None).is_none()
        );
    }

    fn page_of(lines: &[&str]) -> AnalysisPage {
        AnalysisPage {
            picture: "data:image/png;base64,AAAA".to_owned(),
            panels: Vec::new(),
            segments: sources(lines),
        }
    }

    #[test]
    fn a_batch_answer_is_placed_by_the_number_of_the_picture() {
        let answer = r#"{"pages": [
            {"page": 2, "summary": "two", "lines": [{"id": 0, "speaker": "a", "listener": "b", "draft": "x"}]},
            {"page": "1", "summary": "one", "lines": [{"id": 1, "speaker": "a", "listener": "b", "draft": "y"}, {"id": 9, "draft": "out of range"}]}
        ]}"#;
        let read = parse_batch(answer, &[2, 1]).unwrap();
        assert_eq!(read[0].as_ref().unwrap().summary, "one");
        assert_eq!(read[0].as_ref().unwrap().lines.len(), 1, "an id beyond the page is dropped");
        assert_eq!(read[1].as_ref().unwrap().summary, "two");
    }

    #[test]
    fn a_page_the_batch_answer_leaves_out_is_none_and_an_answer_without_lines_is_an_error() {
        let answer = r#"{"pages": [{"summary": "first", "lines": [{"id": 0, "speaker": "a", "listener": "b", "draft": "x"}]}]}"#;
        let read = parse_batch(answer, &[1, 1, 1]).unwrap();
        assert!(read[0].is_some() && read[1].is_none() && read[2].is_none());
        assert!(parse_batch(r#"{"pages": [{"page": 1, "lines": []}]}"#, &[1]).is_err());
        assert!(parse_batch(r#"{"lines": []}"#, &[1]).is_err());
        assert!(parse_batch("no json here", &[1]).is_err());
    }

    #[test]
    fn the_batch_schema_asks_for_one_entry_per_picture() {
        let schema = batch_schema(&[3, 5, 2]);
        let pages = &schema["properties"]["pages"];
        assert_eq!(pages["minItems"], 3);
        assert_eq!(pages["maxItems"], 3);
        assert_eq!(pages["items"]["properties"]["page"]["maximum"], 3);
        assert_eq!(pages["items"]["properties"]["lines"]["items"]["properties"]["id"]["maximum"], 4);
    }

    #[test]
    fn the_batch_prompt_numbers_the_pictures_and_restarts_the_lines_on_every_page() {
        let (first, second) = (page_of(&["one", "two"]), page_of(&["three"]));
        let prompt = batch_prompt(&[4, 5], 30, &[&first, &second], &[], "", &Bible::default(), true);
        assert!(prompt.contains("New pages 5 to 6 of 30"), "{prompt}");
        assert!(prompt.contains("2 pictures are attached"), "{prompt}");
        assert!(prompt.contains("New page 1 (picture 1, page 5), OCR lines:\n0: \"one\"\n1: \"two\""), "{prompt}");
        assert!(prompt.contains("New page 2 (picture 2, page 6), OCR lines:\n0: \"three\""), "{prompt}");
        assert!(!prompt.contains("Story so far"), "the first request has no story: {prompt}");
        // Pages that are not next to each other are named one by one.
        let gap = batch_prompt(&[0, 7], 30, &[&first, &second], &[], "", &Bible::default(), false);
        assert!(gap.contains("pages 1, 8"), "{gap}");
        assert!(gap.contains("No picture is attached"), "{gap}");
    }

    #[test]
    fn a_later_request_starts_with_the_last_pages_read_and_the_story_so_far() {
        let (old, new) = (page_of(&["old line"]), page_of(&["new line"]));
        let context = [ContextPage {
            number: 9,
            page: &old,
            decided: "\n  0: \"old line\" girl -> boy; I \"em\", you \"anh\"".to_owned(),
        }];
        let story = story_so_far(&[(8, "A fight.".to_owned()), (9, "A truce.".to_owned())], 4000);
        let prompt = batch_prompt(&[10], 30, &[&new], &context, &story, &Bible::default(), true);
        assert!(prompt.contains("2 pictures are attached. Pictures 1 to 1 are pages 10 to 10"), "{prompt}");
        assert!(prompt.contains("do not answer for them"), "{prompt}");
        assert!(prompt.contains("Story so far, page by page:\npage 9: A fight.\npage 10: A truce."), "{prompt}");
        assert!(prompt.contains("Page 10 (read already; keep the same words for the same people):\n  0: \"old line\" girl -> boy; I \"em\", you \"anh\""), "{prompt}");
        // The lines to answer for are the new page's, numbered after the picture of the old one.
        assert!(prompt.contains("New page 1 (picture 2, page 11), OCR lines:\n0: \"new line\""), "{prompt}");
        assert!(!prompt.contains("\n0: \"old line\"\nAnswer"), "{prompt}");
    }

    #[test]
    fn the_story_keeps_the_nearest_pages_when_there_is_no_room() {
        let summaries: Vec<(usize, String)> = (0..40)
            .map(|page| (page, format!("Page number {page} has a long summary of what happens in it.")))
            .collect();
        let story = story_so_far(&summaries, 300);
        assert!(story.chars().count() <= 300, "{}", story.chars().count());
        assert!(story.contains("page 40:"), "the page just read stays");
        assert!(!story.contains("page 1:"), "the far past goes first");
        assert!(story.find("page 39:").unwrap() < story.find("page 40:").unwrap(), "reading order");
        assert!(story_so_far(&[(0, "  ".to_owned())], 100).is_empty());
    }

    #[test]
    fn the_decisions_of_a_page_are_shown_again_with_the_next_request() {
        let lines = sources(&["おい ゆう", "りおさん さみしがつてるぞ", "…"]);
        let output = parse_page(ANSWER, 3).unwrap();
        let mut bible = Bible::default();
        bible.merge_output(&output);
        let analysis = PageAnalysis::new(&output, &lines, &bible);
        let decided = analysis.decisions();
        assert!(decided.contains("\n  0: "), "{decided}");
        assert!(decided.contains(" -> "), "who speaks to whom: {decided}");
        assert!(decided.contains("I \"tôi\", you \"cậu\""), "{decided}");
        assert!(!decided.contains("\n  2: "), "a line nobody spoke has no decision: {decided}");
    }
}
