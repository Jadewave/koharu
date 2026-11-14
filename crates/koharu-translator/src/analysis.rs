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
//! Every model is shown the page the way a reader sees it: the whole page first, for the layout
//! and for where everyone stands, then, when the page has panels, every panel on its own in the
//! order it is read, large enough to see faces and balloons. The page the story left off on and
//! what is known of the characters go with it, so that a panel is never read alone.
//!
//! Once every page is read, the pairs of words of every direction are settled by the number of
//! lines that used them (see `settle`), and the whole work is read again as text (see `review`)
//! by the models of `analysis.review_models`, DeepSeek first, before the local model translates
//! anything.

use std::{collections::HashSet, fmt::Write as _, sync::Arc};

use anyhow::{Context, anyhow, ensure};
use image::DynamicImage;
use serde::{Deserialize, Deserializer, Serialize};
use serde_json::{Value, json};

mod review;
mod settle;

use crate::{
    Error, GenerationConfig, Language, ProvidersConfig, Translator,
    backend::{ExpectedPair, GlossaryEntry, encode_image, shrink_to},
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
const INSTRUCTIONS_CHARS: usize = 6000;

/// `[analysis]` of the configuration file.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct AnalysisConfig {
    /// Reads the work with the vision models below when the OpenAI-compatible provider has
    /// a server of its own (not the local default). Set to false to turn it off.
    pub enabled: bool,
    /// The models that read the pages, in the order they are tried for every page.
    pub models: Vec<AnalysisModel>,
    /// The models of the second reading, in the order they are tried for every part of the
    /// work. It is text only, so these are names, not `AnalysisModel`s. DeepSeek comes first:
    /// it settles the pronouns of the whole work before the local model translates a page.
    pub review_models: Vec<ReviewModel>,
    /// Lets the models think while they read a page. Slower and it needs `max_tokens` to be
    /// large. Off by default: the second reading (`review`) is where thinking pays.
    pub reasoning: bool,
    pub max_tokens: u32,
    /// After every page is read, the `review_models` read the whole work again as text, settle
    /// the words for "I" and "you" of every pair of characters and correct the notes of the
    /// pages that disagree with it.
    pub review: bool,
    /// Lets the models of the second reading think. Unlike a page, the whole work is one
    /// decision, and thinking pays there.
    pub review_reasoning: bool,
    /// The most tokens a model of the second reading may write, thinking included.
    pub review_max_tokens: u32,
    /// The size of one request of the second reading, in characters of the pages' text; a
    /// longer work is read in several parts.
    pub review_chars: usize,
    /// The longest side of the page picture that is sent, in pixels.
    pub image_max_dimension: u32,
    /// The longest side of a panel picture that is sent, in pixels.
    pub panel_max_dimension: u32,
    /// The order panels are read in: from the right of a row to its left (Japanese manga), or
    /// from the left to the right.
    pub right_to_left: bool,
}

/// The DeepSeek model of the analysis.
const DEEPSEEK: &str = "deepseek-ai/deepseek-v4.1-flash";
/// The vision model of the analysis.
const NEMOTRON_VL: &str = "nvidia/nemotron-3.5-super-vl-120b-a12b";

/// A model of the second reading.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ReviewModel {
    pub model: String,
    /// Lets this model think during the review. Unset: follows `review_reasoning`.
    #[serde(default)]
    pub reasoning: Option<bool>,
}

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
        let model = |name: &str, reasoning: Option<bool>| AnalysisModel {
            model: name.to_owned(),
            vision: true,
            reasoning,
        };
        Self {
            enabled: true,
            models: vec![model(NEMOTRON_VL, None), model(DEEPSEEK, None)],
            // DeepSeek answers the review without thinking; the vision model, its fallback,
            // follows `review_reasoning`.
            review_models: vec![
                ReviewModel { model: DEEPSEEK.to_owned(), reasoning: Some(false) },
                ReviewModel { model: NEMOTRON_VL.to_owned(), reasoning: None },
            ],
            reasoning: false,
            // A page is answered in about two thousand tokens: a model that loops (the log of one
            // page wrote the same phrase until the limit) is cut off early.
            max_tokens: 6144,
            review: true,
            review_reasoning: true,
            review_max_tokens: 32768,
            review_chars: 60_000,
            image_max_dimension: 1600,
            panel_max_dimension: 1024,
            right_to_left: true,
        }
    }
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
        let wants_panels = config.models.iter().any(|model| model.vision);
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
            picture: encode_image(&scaled)?.data_url(),
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
            picture: encode_image(&scaled)?.data_url(),
            lines: held,
        });
    }
    let covered: usize = pictures.iter().map(|picture| picture.lines.len()).sum();
    if pictures.len() < 2 || pictures.len() > MAX_PANELS || covered != lines {
        return Ok(Vec::new());
    }
    Ok(pictures)
}

fn analysis_config() -> anyhow::Result<AnalysisConfig> {
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
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(default)]
struct PageOutput {
    #[serde(deserialize_with = "lenient_string")]
    scene: String,
    #[serde(deserialize_with = "lenient_string")]
    page_note: String,
    characters: Vec<Character>,
    terms: Vec<Term>,
    lines: Vec<Line>,
}

/// The JSON schema of the answer for a page of `expected` lines.
fn schema(expected: usize) -> Value {
    let text = |description: &str| json!({ "type": "string", "description": description });
    json!({
        "type": "object",
        "properties": {
            "scene": text("One or two sentences: where, when and what happens on this page."),
            "page_note": text("What the translator must remember on this page: who is who, what to avoid."),
            "characters": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "name": text("The one name this character is called."),
                        "aliases": { "type": "array", "items": { "type": "string" } },
                        "gender": text("man, woman, boy, girl, or unknown."),
                        "age": text("A number or a stage: child, teenager, young adult, adult, middle-aged, elderly."),
                        "role": text("Who the character is in the story."),
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
                        "speaker": text("The character who says or thinks it."),
                        "listener": text("The character spoken to; 'self' for a thought."),
                        "self": text("The word the speaker uses for themselves."),
                        "you": text("The word the speaker uses for the listener."),
                        "tone": text("Two or three words.")
                    },
                    "required": ["id", "speaker", "listener"]
                }
            }
        },
        "required": ["lines"]
    })
}

fn system_prompt(target: Language) -> String {
    let mut prompt = format!(
        "You read a manga page that is going to be translated into {target}. You see the page \
picture (when one is attached) and the text OCR read from it, as numbered lines; the red boxes \
on the picture carry the same numbers. Your answer goes to a translator who never sees the \
picture, so everything that depends on the picture must be in it: who speaks, to whom, how old \
they are, how they talk.

For every OCR line:
- id: the number of the line.
- kind: speech, thought (an inner voice, often in a cloud or a box without a tail), narration, \
sfx, sign or other.
- text: the line, corrected when OCR misread it (wrong, missing or extra characters, kana read \
as kanji, noise from the drawing). Check it against the picture and the story. Copy the line \
unchanged when it is right. Never translate it, never reword it, never add words that are not \
on the page.
- speaker: the name of the character who says or thinks it ('narrator' for narration, \
'unknown' when nobody can be told). Follow the tail of the bubble, the positions and the story.
- listener: the character spoken to ('self' for a thought, 'everyone', 'unknown').
- Speaker and listener are names, never roles: not nurse, patient, visitor, customer, boss or \
friend. The text says who is who: a person who is called 'Sasaki-kun' is Sasaki, 'Rio-san' is \
Rio, and a name that is said in a line belongs to someone other than the speaker. Use the names \
of 'Known so far' and the names that are said or written in the OCR lines. When a character has \
no name yet, give a short label by look ('woman with long hair', 'man with glasses') and use \
exactly that label on every page until the name is learned, then put the label in aliases. Two \
people who answer each other are two different characters, and nobody speaks to themself \
except in a thought.
- self and you: the words the speaker uses for themselves and for the listener in {target}, \
chosen from age, sex, rank, closeness and manner of speaking.
- tone: two or three words (polite, rough, teasing, tender, scared, ...).

Also answer: scene, page_note (what the translator must remember on this page), characters \
(new ones, or what you learned about known ones: name, aliases, gender, age, role, speech_style, \
relationships with the pronouns they use) and terms (names of people, places, organizations, \
titles and recurring words, with the {target} rendering to use).

Rules:
- Decide gender, age and manner from how a person speaks first (pronouns such as 俺 or 僕, \
sentence endings such as ぞ or わ, honorifics) and from the picture, never from hair, glasses or \
build alone.
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
- The first-person word of a thought is the one the character uses aloud, or 'mình'."
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

fn parse_page(text: &str, count: usize) -> anyhow::Result<PageOutput> {
    let json = json_object(text).ok_or_else(|| anyhow!("the answer holds no JSON object"))?;
    let mut output: PageOutput = serde_json::from_str(json).context("invalid analysis JSON")?;
    let mut seen = HashSet::new();
    output
        .lines
        .retain(|line| line.id < count && seen.insert(line.id));
    ensure!(
        !output.lines.is_empty(),
        "the answer has no line for any of the {count} OCR lines"
    );
    Ok(output)
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
            self.note_pair(&line.speaker, &line.listener, &line.self_word, &line.you);
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
        for part in [&character.gender, &character.age, &character.role] {
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
}

/// What the vision model learned about one page: the notes for the translation model and the
/// corrections of the OCR text.
#[derive(Clone, Debug, PartialEq)]
pub struct PageAnalysis {
    /// The OCR lines the page was read with; the notes only apply while the page still has them.
    sources: Vec<String>,
    lines: Vec<AnalyzedLine>,
    scene: String,
    note: String,
    /// The profiles of the characters that speak or are spoken to on the page.
    cast: Vec<String>,
}

impl PageAnalysis {
    fn new(output: &PageOutput, sources: &[String], bible: &Bible) -> Self {
        let mut lines = vec![AnalyzedLine::default(); sources.len()];
        let mut cast: Vec<String> = Vec::new();
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
            };
            for name in [&line.speaker, &line.listener] {
                if is_generic(name) {
                    continue;
                }
                if let Some(index) = bible.character_index(name) {
                    let character = &bible.characters[index];
                    let entry = format!("{}: {}", character.name, Bible::profile(character));
                    if !cast.contains(&entry) {
                        cast.push(entry);
                    }
                }
            }
        }
        Self {
            sources: sources.to_vec(),
            lines,
            scene: clip(&output.scene, 400),
            note: clip(&output.page_note, 600),
            cast,
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
    /// notes give no pair for (narration, a thought, an unknown speaker). Empty when the page
    /// changed since it was read. The translation is checked against these.
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
                (speech
                    && !is_generic(&line.speaker)
                    && !is_generic(&line.listener)
                    && !mine.is_empty()
                    && !yours.is_empty())
                .then(|| ExpectedPair {
                    speaker: line.speaker.trim().to_owned(),
                    listener: line.listener.trim().to_owned(),
                    mine: mine.to_owned(),
                    yours: yours.to_owned(),
                })
            })
            .collect()
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
            "Notes on this page from a vision model that saw the picture (you do not see it). \
Follow them for who speaks to whom and for the words each speaker uses for themselves and for \
the listener; the numbers are the ids of the segments.",
        );
        if !self.scene.is_empty() {
            let _ = write!(notes, "\nScene: {}", self.scene);
        }
        if !self.note.is_empty() {
            let _ = write!(notes, "\nNote: {}", self.note);
        }
        if !self.cast.is_empty() {
            notes.push_str("\nCharacters on this page:");
            for entry in &self.cast {
                let _ = write!(notes, "\n- {entry}");
            }
        }
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
            let speaker = if line.speaker.trim().is_empty() { "unknown" } else { line.speaker.trim() };
            let mut entry = format!("\n- {id} {kind}: {speaker}");
            if !is_generic(&line.listener) || line.listener.trim().eq_ignore_ascii_case("self") {
                let _ = write!(entry, " -> {}", line.listener.trim());
                if !line.relation.is_empty() {
                    let _ = write!(entry, " ({})", line.relation);
                }
            }
            let mut words = Vec::new();
            if !line.self_word.is_empty() {
                words.push(format!("calls self \"{}\"", line.self_word));
            }
            if !line.you.is_empty() {
                words.push(format!("calls the listener \"{}\"", line.you));
            }
            if !words.is_empty() {
                let _ = write!(entry, "; {}", words.join(", "));
            }
            if !line.tone.is_empty() {
                let _ = write!(entry, "; {}", line.tone);
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

        for (number, page) in pages.iter().enumerate() {
            if stop() {
                results.resize(pages.len(), None);
                break;
            }
            let count = page.segments.len();
            let mut result = None;
            for entry in &config.models {
                if dead.contains(&entry.model) {
                    continue;
                }
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
                // Shows in the log what the model is really sent: with `vision` off it is no
                // picture at all, with panels it is the page and one picture for every panel.
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
                    system: system.clone(),
                    user: user_prompt(
                        number,
                        pages.len(),
                        &page.segments,
                        &bible,
                        Some(&previous),
                        &view,
                    ),
                    images,
                    schema_name: "manga_page_analysis",
                    schema: schema(count),
                };
                let generation = GenerationConfig {
                    reasoning: Some(entry.reasoning.unwrap_or(config.reasoning)),
                    ..generation
                };
                let attempt = remote::chat_json(&self.client, &providers, &entry.model, &generation, chat)
                    .await
                    .map_err(|error| {
                        if matches!(&error, Error::Api { status, .. } if matches!(*status, 401 | 403 | 404)) {
                            dead.insert(entry.model.clone());
                        }
                        anyhow::Error::from(error)
                    })
                    .and_then(|text| parse_page(&text, count));
                match attempt {
                    Ok(output) => {
                        result = Some((entry.model.clone(), output));
                        break;
                    }
                    Err(error) => tracing::warn!(
                        page = number + 1,
                        model = %entry.model,
                        error = %format!("{error:#}"),
                        "analysis model could not read the page; trying the next one"
                    ),
                }
            }
            match result {
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
                    results.push(Some(analysis));
                }
                None => {
                    tracing::warn!(page = number + 1, "no analysis model could read the page");
                    results.push(None);
                }
            }
        }

        let settled = bible.settle_pairs();
        tracing::info!(settled, "directions settled by the number of lines that used a pair");
        let reviewed = config.review
            && self
                .review_work(&config, &providers, language, &mut bible, &mut results, &mut dead, &stop)
                .await;
        if !reviewed {
            // Nobody read the work again: what most lines of a direction used is its pair, so
            // that a line that strayed does not make the notes of the pages disagree.
            for page in results.iter_mut().flatten() {
                page.apply_review(&bible, None);
            }
            for problem in bible.address_conflicts() {
                tracing::warn!(%problem, "a settled pair cannot be right");
            }
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
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sources(lines: &[&str]) -> Vec<String> {
        lines.iter().map(|line| (*line).to_owned()).collect()
    }

    const ANSWER: &str = r#"{
        "scene": "A hospital room.",
        "page_note": "Sasaki is a man.",
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
        assert!(notes.contains("Scene: A hospital room."));
        assert!(notes.contains("Sasaki: man, young adult, Rio's colleague; speaks: rough but kind"));
        assert!(notes.contains("- 0 speech: Sasaki -> Yuu; calls self \"tôi\", calls the listener \"cậu\"; teasing"));
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
    fn the_default_chain_is_the_vision_model_then_deepseek() {
        let config = AnalysisConfig::default();
        assert!(config.models.iter().all(|model| model.vision));
        assert_eq!(config.models.len(), 2, "the nano omni model is gone");
        assert_eq!(config.models[0].model, NEMOTRON_VL);
        assert_eq!(config.models[1].model, DEEPSEEK);
        assert!(config.right_to_left);
        assert!(config.models.iter().all(|model| !model.model.contains("nano")));
        // The second reading starts with DeepSeek, which does not think, then the vision model.
        assert_eq!(config.review_models[0].model, DEEPSEEK);
        assert_eq!(config.review_models[0].reasoning, Some(false));
        assert_eq!(config.review_models[1].model, NEMOTRON_VL);
        // An old file that names the panels of a model is still read; that setting is gone.
        let parsed: AnalysisConfig =
            serde_json::from_str(r#"{"models": [{"model": "a/b", "panels": true}]}"#).unwrap();
        assert!(parsed.enabled && parsed.models.len() == 1 && parsed.models[0].vision);
        assert_eq!(parsed.review_models, config.review_models, "an unset chain is the default");
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
        assert_eq!((first.speaker.as_str(), first.listener.as_str()), ("Sasaki", "Yuu"));
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
}
