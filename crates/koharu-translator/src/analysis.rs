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

use std::{collections::HashSet, fmt::Write as _, sync::Arc};

use anyhow::{Context, anyhow, ensure};
use image::DynamicImage;
use serde::{Deserialize, Deserializer, Serialize};
use serde_json::{Value, json};

use crate::{
    Error, GenerationConfig, Language, Translator,
    backend::{GlossaryEntry, encode_image, shrink_to},
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
/// The notes of one page are cut to this many characters.
const INSTRUCTIONS_CHARS: usize = 3500;

/// `[analysis]` of the configuration file.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct AnalysisConfig {
    /// Reads the work with the vision models below when the OpenAI-compatible provider has
    /// a server of its own (not the local default). Set to false to turn it off.
    pub enabled: bool,
    /// The models, in the order they are tried for every page.
    pub models: Vec<AnalysisModel>,
    /// Lets the models think before they answer. Slower and it needs `max_tokens` to be large.
    pub reasoning: bool,
    pub max_tokens: u32,
    /// The longest side of the page picture that is sent, in pixels.
    pub image_max_dimension: u32,
}

/// A model of the analysis chain.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct AnalysisModel {
    pub model: String,
    /// Whether the model is sent the page picture. A model without vision only reads the text.
    #[serde(default = "yes")]
    pub vision: bool,
}

const fn yes() -> bool {
    true
}

impl Default for AnalysisConfig {
    fn default() -> Self {
        let model = |name: &str, vision: bool| AnalysisModel {
            model: name.to_owned(),
            vision,
        };
        Self {
            enabled: true,
            models: vec![
                model("nvidia/nemotron-3.5-super-vl-120b-a12b", true),
                model("nvidia/nemotron-3-nano-omni-30b-a3b-reasoning", true),
                model("deepseek-ai/deepseek-v4.1-flash", false),
            ],
            reasoning: false,
            max_tokens: 8192,
            image_max_dimension: 1600,
        }
    }
}

/// One page to read: its picture (with the boxes and numbers of the OCR lines drawn on it,
/// scaled down and encoded once) and the OCR lines, in the order of the numbers.
#[derive(Clone)]
pub struct AnalysisPage {
    picture: String,
    pub segments: Vec<String>,
}

impl AnalysisPage {
    /// Prepares a page for the analysis. `image` carries the numbers of `segments`.
    pub fn new(image: &Arc<DynamicImage>, segments: Vec<String>) -> anyhow::Result<Self> {
        let longest = analysis_config()
            .map_or(AnalysisConfig::default().image_max_dimension, |config| {
                config.image_max_dimension
            })
            .max(256);
        let scaled = shrink_to(image, longest)?;
        Ok(Self {
            picture: encode_image(&scaled)?.data_url(),
            segments,
        })
    }
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
    if target == Language::Vietnamese {
        prompt.push_str(
            "

Vietnamese pronouns for self and you:
- Pick the pair from the relationship, the ages and the rank, for example 'tôi - cô', \
'anh - em', 'em - chị', 'tớ - cậu', 'tao - mày'. People who speak politely to each other get \
a formal pair, close people a familiar one. Do not give everyone the same pair.
- A pair belongs to one direction: what A calls B is not what B calls A. Keep the pair of a \
direction the same on every page.
- An older speaker calls himself 'tôi' (or 'bác', 'chú', 'ông', 'lão' by age and manner), never \
'con' or 'cháu', unless the speaker is a child or a younger relative of the listener.
- The first-person word of a thought is the one the character uses aloud, or 'mình'.",
        );
    }
    prompt
}

fn user_prompt(
    number: usize,
    total: usize,
    segments: &[String],
    bible: &Bible,
    has_image: bool,
) -> String {
    let mut prompt = format!("Page {} of {total}. ", number + 1);
    if has_image {
        prompt.push_str("The picture of the page is attached.\n");
    } else {
        prompt.push_str(
            "No picture is attached: decide from the text only, and answer 'unknown' where a \
speaker cannot be told.\n",
        );
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
        let Some(index) = self.character_index(&new.name) else {
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
        let relationship = Relationship {
            with: listener.to_owned(),
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
                entries.push(GlossaryEntry {
                    kind: "address".to_owned(),
                    source: format!("{} -> {}", character.name, relationship.with.trim()),
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
Follow them for who speaks to whom and for the words people use for themselves and each other; \
the numbers are the ids of the segments.",
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
            let kind = if line.kind.is_empty() { "speech" } else { &line.kind };
            let speaker = if line.speaker.trim().is_empty() { "unknown" } else { line.speaker.trim() };
            let mut entry = format!("\n- {id} {kind}: {speaker}");
            if !is_generic(&line.listener) || line.listener.trim().eq_ignore_ascii_case("self") {
                let _ = write!(entry, " -> {}", line.listener.trim());
            }
            if !line.self_word.is_empty() || !line.you.is_empty() {
                let _ = write!(
                    entry,
                    "; words: {} / {}",
                    if line.self_word.is_empty() { "-" } else { &line.self_word },
                    if line.you.is_empty() { "-" } else { &line.you },
                );
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
        let mut dead: HashSet<usize> = HashSet::new();
        let mut results: Vec<Option<PageAnalysis>> = Vec::with_capacity(pages.len());

        for (number, page) in pages.iter().enumerate() {
            if stop() {
                results.resize(pages.len(), None);
                break;
            }
            let count = page.segments.len();
            let mut result = None;
            for (position, entry) in config.models.iter().enumerate() {
                if dead.contains(&position) {
                    continue;
                }
                let images = if entry.vision {
                    vec![page.picture.clone()]
                } else {
                    Vec::new()
                };
                let chat = remote::Chat {
                    system: system.clone(),
                    user: user_prompt(number, pages.len(), &page.segments, &bible, entry.vision),
                    images,
                    schema_name: "manga_page_analysis",
                    schema: schema(count),
                };
                let attempt = remote::chat_json(&self.client, &providers, &entry.model, &generation, chat)
                    .await
                    .map_err(|error| {
                        if matches!(&error, Error::Api { status, .. } if matches!(*status, 401 | 403 | 404)) {
                            dead.insert(position);
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

        if let Ok(mut glossary) = self.glossary.lock() {
            glossary.begin(project, language);
            glossary.merge(bible.glossary_entries());
        }
        tracing::info!(
            characters = bible.characters.len(),
            terms = bible.terms.len(),
            pages = results.iter().filter(|page| page.is_some()).count(),
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
        assert!(notes.contains("- 0 speech: Sasaki -> Yuu; words: tôi / cậu; teasing"));
        assert!(notes.chars().count() <= INSTRUCTIONS_CHARS + 20);
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
        assert!(!system_prompt(Language::English).contains("Vietnamese pronouns"));
        let mut bible = Bible::default();
        assert!(user_prompt(0, 2, &sources(&["a"]), &bible, true).contains("Known so far: nothing yet."));
        bible.merge_output(&parse_page(ANSWER, 3).unwrap());
        let user = user_prompt(1, 2, &sources(&["おい", "ゆう"]), &bible, false);
        assert!(user.starts_with("Page 2 of 2. No picture is attached"));
        assert!(user.contains("\"Sasaki\""));
        assert!(user.contains("1: \"ゆう\""));
        assert!(user.ends_with("(ids 0 to 1)."));
    }

    #[test]
    fn the_schema_bounds_the_ids() {
        let value = schema(5);
        assert_eq!(value["properties"]["lines"]["items"]["properties"]["id"]["maximum"], 4);
        assert_eq!(value["required"][0], "lines");
    }

    #[test]
    fn the_default_chain_starts_with_the_vision_model() {
        let config = AnalysisConfig::default();
        assert!(config.models[0].vision && config.models[1].vision && !config.models[2].vision);
        assert_eq!(config.models[2].model, "deepseek-ai/deepseek-v4.1-flash");
        let parsed: AnalysisConfig =
            serde_json::from_str(r#"{"models": [{"model": "a/b"}]}"#).unwrap();
        assert!(parsed.enabled && parsed.models.len() == 1 && parsed.models[0].vision);
    }
}
