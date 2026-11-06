use std::fmt::Write;

use anyhow::Context;
use indoc::indoc;
use serde::{Deserialize, Deserializer, Serialize, de};
use serde_json::{Value, json};

use crate::{GlossaryEntry, Language, TranslationContext, TranslationRequest};

pub(crate) fn prompts(request: &TranslationRequest) -> anyhow::Result<(String, String)> {
    let input = TranslationInput {
        source_language: request.source_language,
        target_language: request.target_language,
        context: &request.context,
        segments: request
            .segments
            .iter()
            .enumerate()
            .map(|(id, text)| TranslationInputSegment {
                id,
                text,
                page: if request.page_images.len() > 1 {
                    request.segment_pages.get(id).map(|page| page + 1)
                } else {
                    None
                },
            })
            .collect(),
    };
    let user = serde_json::to_string(&input).context("failed to serialize translation input")?;
    Ok((translation_system_prompt(request), user))
}

pub(crate) fn translations(
    provider: &str,
    text: &str,
    source_segments: &[String],
) -> anyhow::Result<Vec<String>> {
    parse_translations(provider, text, source_segments).map(|(translations, _)| translations)
}

/// Parses a model response and also reports which segment indices the model never
/// returned. Those entries still hold their source text in the first element.
pub(crate) fn parse_translations(
    provider: &str,
    text: &str,
    source_segments: &[String],
) -> anyhow::Result<(Vec<String>, Vec<usize>)> {
    let output = serde_json::from_str::<TranslationOutput>(text).with_context(|| {
        InvalidTranslationOutput {
            message: format!(
                "{provider} returned invalid translation JSON for {} segments; response was: {}",
                source_segments.len(),
                snippet(text),
            ),
            response: text.to_owned(),
        }
    })?;
    let mut translations = source_segments.to_vec();
    let mut translated = vec![false; source_segments.len()];

    for translation in output.translations {
        if translation.id < translations.len() && !translated[translation.id] {
            translations[translation.id] = translation.text;
            translated[translation.id] = true;
        }
    }

    let missing = translated
        .iter()
        .enumerate()
        .filter_map(|(index, done)| (!done).then_some(index))
        .collect();
    Ok((translations, missing))
}

/// Context attached to a response that is not valid translation JSON. It keeps the raw
/// response so callers can still recover the segments written before the model was cut
/// off (typically by a repetition loop that ran into the token limit).
#[derive(Debug)]
pub(crate) struct InvalidTranslationOutput {
    message: String,
    pub(crate) response: String,
}

impl std::fmt::Display for InvalidTranslationOutput {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str(&self.message)
    }
}

/// Reads whatever complete `{"id", "text"}` objects a truncated or malformed response
/// holds. Returns the translations (source text for anything not recovered) and the
/// indices that were not recovered. Never fails: an unusable response recovers nothing.
pub(crate) fn salvage_translations(
    text: &str,
    source_segments: &[String],
) -> (Vec<String>, Vec<usize>) {
    let mut translations = source_segments.to_vec();
    let mut translated = vec![false; source_segments.len()];

    let array = text
        .find("\"translations\"")
        .and_then(|key| text[key..].find('[').map(|offset| key + offset + 1));
    if let Some(start) = array {
        let mut rest = &text[start..];
        loop {
            rest = rest.trim_start_matches(|c: char| c.is_whitespace() || c == ',');
            if !rest.starts_with('{') {
                break;
            }
            let mut stream =
                serde_json::Deserializer::from_str(rest).into_iter::<TranslationOutputSegment>();
            let Some(Ok(segment)) = stream.next() else {
                break;
            };
            let consumed = stream.byte_offset();
            if segment.id < translations.len() && !translated[segment.id] {
                translations[segment.id] = segment.text;
                translated[segment.id] = true;
            }
            rest = &rest[consumed..];
        }
    }

    let missing = translated
        .iter()
        .enumerate()
        .filter_map(|(index, done)| (!done).then_some(index))
        .collect();
    (translations, missing)
}

#[derive(Serialize)]
struct GlossaryInput<'a> {
    glossary: &'a [GlossaryEntry],
    lines: &'a [String],
}

#[derive(Deserialize)]
struct GlossaryOutput {
    entries: Vec<GlossaryEntry>,
}

/// Whether the glossary model also writes character profiles and address pairs. Both are
/// off: they were written from OCR text alone (no speakers, no images) and were often wrong
/// or ignored, so the translation model decides pronouns itself while it looks at the pages.
/// The code stays so that they can be switched on again.
pub(crate) const PROFILES: bool = false;

const PROFILE_KINDS: &str = indoc! {"
    - character: `source` is the bare name as it appears in the lines, without honorifics and never a pronoun, with one entry per character even when the lines write or call the name in several ways (for a recurring character without a name, a short description such as the regular customer); `target` a short profile: gender, age (the exact number when the work states it, otherwise rough), role, relationship to the other main characters, and own manner of speaking (formal, casual, rough, childish, archaic, ...) including whether they speak politely or plainly to strangers and to the other main characters, and the first-person pronoun they use. Put the manner of speaking and the pronoun first.
    - address: how one character addresses another. `source` is \"Speaker -> Listener\" with the names used in the lines (or a short description for an unnamed recurring character), `target` is the pair of words the speaker uses for themselves and for the listener, chosen from their relationship, age gap, rank, and tone (for Vietnamese, for example \"tôi - cô\", \"anh - em\", \"tớ - cậu\", or \"tao - mày\"; do not give everyone the same pair). Formality decides first: people who speak politely to each other get a formal pair, close people a familiar one."};

/// System and user prompts that ask a model for the new glossary entries of a stretch of the
/// work, from its source lines alone.
pub(crate) fn glossary_prompts(
    target: Language,
    glossary: &[GlossaryEntry],
    lines: &[String],
) -> anyhow::Result<(String, String)> {
    let input = GlossaryInput {
        glossary,
        lines,
    };
    let user = serde_json::to_string(&input).context("failed to serialize glossary input")?;
    let system = format!(
        indoc! {"
            You maintain the glossary of a manga that is being translated into {target}.
            The translator will follow your glossary exactly on every page, so you decide the renderings.

            You receive the current glossary and the source lines of a stretch of the work (OCR text in reading order, page after page).
            Return only NEW entries:
            - name: a character, place, organization, or title seen in the lines and the {target} rendering to use for it. Every named character gets a name entry, written without honorifics, so that the name is spelled the same way on every page.
            - term: a recurring term, skill, technique, or item and the rendering to use for it. Leave sound effects out, because the translator renders them on each page, and never romanize anything.{profile_kinds}
            - style: `source` is exactly work, `target` one short note on genre, setting, and overall register. At most one, and only if the glossary has none yet.

            Rules:
            - Choose natural renderings that will still fit the rest of the work.
            - Be exhaustive: list every named character, place, organization, title, and recurring term in the lines.{exhaustive_tail}
            - Skip entries already in the glossary you were given and anything that does not appear in these lines. Never list the same character again under another spelling, honorific, or pronoun.
            - Write `kind` as exactly {kind_list}.
            - At most 20 entries, every string short.
            - Return an empty `entries` array when there is nothing new.
        "},
        target = target,
        profile_kinds = if PROFILES {
            format!("\n{PROFILE_KINDS}")
        } else {
            String::new()
        },
        exhaustive_tail = if PROFILES {
            " Also list every character who speaks, and every pair of speaking characters whose way of addressing each other you can tell. Write the character entries before the address entries."
        } else {
            ""
        },
        kind_list = if PROFILES {
            "style, address, character, name, or term"
        } else {
            "style, name, or term"
        },
    )
    .trim_end()
    .to_owned();
    Ok((system, user))
}

/// Only Vietnamese has the many ways of saying "I" and "you" that must be fixed per pair of
/// characters before the translation starts.
pub(crate) fn decides_address(target: Language) -> bool {
    target == Language::Vietnamese
}

/// The prompts of the pass that decides how the characters address each other. It reads the
/// character profiles the first pass wrote (ages, genders, relationships), not the raw lines,
/// because a line of OCR text does not say who speaks it.
pub(crate) fn address_prompts(
    target: Language,
    characters: &[GlossaryEntry],
) -> anyhow::Result<(String, String)> {
    let input = json!({
        "characters": characters
            .iter()
            .map(|entry| json!({ "name": entry.source, "profile": entry.target }))
            .collect::<Vec<_>>(),
    });
    let user = serde_json::to_string(&input).context("failed to serialize address input")?;
    let system = format!(
        indoc! {"
            You decide how the characters of a manga address each other when it is translated into {target}.
            The translator will follow your choices exactly on every page.

            You receive the characters of the work with their profiles (gender, age, role, relationships).
            Return `address` entries, one for each direction of every pair of characters who talk to each other:
            - `kind` is exactly address.
            - `source` is \"Speaker -> Listener\" with the character names exactly as given.
            - `target` is the pair of words the speaker uses for themselves and for the listener in {target} (for Vietnamese, for example \"tôi - cô\", \"anh - em\", \"tớ - cậu\", \"tao - mày\", \"em - chị\").

            Rules:
            - Formality decides first, then age gap, gender, rank, relationship, and tone. People who speak politely to each other (strangers, colleagues, a customer and a worker, a junior and a senior) get a formal pair (for Vietnamese 'tôi - cô', 'tôi - anh', 'em - anh'); close people (family, couples, old friends) get a familiar pair. Do not give everyone the same pair, and do not use 'anh - em' for people who are not close.
            - The two directions of a pair need not mirror each other.
            - Use only characters from the list. Skip a pair when the profiles say nothing about how the two relate, or give neither an age nor how formally they speak: leave it out rather than guess.
            - At most 20 entries, every string short.
            - Return an empty `entries` array when no pair can be decided.
        "},
        target = target,
    )
    .trim_end()
    .to_owned();
    Ok((system, user))
}

pub(crate) fn glossary_schema() -> Value {
    json!({
        "type": "object",
        "properties": {
            "entries": {
                "type": "array",
                "maxItems": 20,
                "items": {
                    "type": "object",
                    "properties": {
                        "kind": { "type": "string" },
                        "source": { "type": "string" },
                        "target": { "type": "string" }
                    },
                    "required": ["kind", "source", "target"],
                    "additionalProperties": false
                }
            }
        },
        "required": ["entries"],
        "additionalProperties": false
    })
}

/// Reads the glossary entries of a response, keeping the complete ones when it was cut off.
pub(crate) fn parse_glossary(text: &str) -> Vec<GlossaryEntry> {
    if let Ok(output) = serde_json::from_str::<GlossaryOutput>(text) {
        return output.entries;
    }
    let mut entries = Vec::new();
    let array = text
        .find("\"entries\"")
        .and_then(|key| text[key..].find('[').map(|offset| key + offset + 1));
    if let Some(start) = array {
        let mut rest = &text[start..];
        loop {
            rest = rest.trim_start_matches(|c: char| c.is_whitespace() || c == ',');
            if !rest.starts_with('{') {
                break;
            }
            let mut stream =
                serde_json::Deserializer::from_str(rest).into_iter::<GlossaryEntry>();
            let Some(Ok(entry)) = stream.next() else {
                break;
            };
            let consumed = stream.byte_offset();
            entries.push(entry);
            rest = &rest[consumed..];
        }
    }
    entries
}

/// The glossary as a block for the translation prompt.
pub(crate) fn render_glossary(entries: &[GlossaryEntry]) -> String {
    let mut block = String::from(
        "Glossary (established earlier in this work; always use exactly these renderings):",
    );
    for entry in entries {
        let _ = write!(
            block,
            "\n- {}: {} = {}",
            entry.kind, entry.source, entry.target
        );
    }
    block
}

/// Keeps a model response readable in a log line without truncating so hard
/// that the shape of the failure is lost.
///
/// Responses are usually pretty-printed, so line breaks and other control
/// characters are escaped rather than passed through: the snippet is formatted
/// into an error context, and one failure should stay one line.
pub(crate) fn snippet(text: &str) -> String {
    const LIMIT: usize = 2000;
    let trimmed = text.trim();
    let (visible, elided) = match trimmed.char_indices().nth(LIMIT) {
        Some((end, _)) => (&trimmed[..end], true),
        None => (trimmed, false),
    };

    let mut output = String::with_capacity(visible.len());
    for character in visible.chars() {
        match character {
            '\n' => output.push_str("\\n"),
            '\r' => output.push_str("\\r"),
            '\t' => output.push_str("\\t"),
            control if control.is_control() => {
                let _ = write!(output, "\\u{{{:04x}}}", control as u32);
            }
            character => output.push(character),
        }
    }
    if elided {
        let _ = write!(output, "… ({} bytes total)", trimmed.len());
    }
    output
}

pub(crate) fn output_schema(expected: usize) -> Value {
    output_schema_with(expected, false)
}

/// The schema for the translations of `request`: with a `speaker` string before `text` when
/// the request asks the model to decide who speaks to whom first.
pub(crate) fn request_schema(request: &TranslationRequest) -> Value {
    output_schema_with(request.segments.len(), request.wants_speaker())
}

fn output_schema_with(expected: usize, speaker: bool) -> Value {
    // Built in this order on purpose: the model writes the properties in the order they are
    // listed, so the speaker is decided before the text.
    let mut properties = serde_json::Map::new();
    properties.insert(
        "id".to_owned(),
        json!({
            "type": "integer",
            "minimum": 0,
            "maximum": expected.saturating_sub(1),
            "description": "The ID copied from the corresponding input segment."
        }),
    );
    let mut required = vec![json!("id")];
    if speaker {
        properties.insert(
            "speaker".to_owned(),
            json!({
                "type": "string",
                "description": "Who speaks to whom and the pronoun pair, e.g. 'Rio -> Yuu: em - anh', or 'narration' or 'thought'."
            }),
        );
        required.push(json!("speaker"));
    }
    properties.insert(
        "text".to_owned(),
        json!({
            "type": "string",
            "description": "The translation of the input segment with this ID."
        }),
    );
    required.push(json!("text"));
    json!({
        "type": "object",
        "properties": {
            "translations": {
                "type": "array",
                "minItems": expected,
                "maxItems": expected,
                "items": {
                    "type": "object",
                    "properties": properties,
                    "required": required,
                    "additionalProperties": false
                }
            }
        },
        "required": ["translations"],
        "additionalProperties": false
    })
}

fn translation_system_prompt(request: &TranslationRequest) -> String {
    let source = request
        .source_language
        .map(|language| language.to_string())
        .unwrap_or_else(|| "the detected source language".to_owned());
    let mut prompt = format!(
        indoc! {"
            You are a professional manga translator.

            Translation requirements:
            - Translate every input segment from {source} into natural {target}.
            - Preserve meaning, character voice, emotional tone, relationship nuance, emphasis, and sound effects; never invent or drop content.
            - Sound like a native {target} speaker in that situation, not like a word-for-word translation, and keep the emotion (anger, shyness, sarcasm, teasing, ...).
            - Localize idioms, slang, wordplay, and sound effects with an equivalent {target} expression or the same effect in other words, never literally, while keeping wording concise enough for speech bubbles.
            - Read the segments in order as one scene and use the surrounding segments and any supplied context for disambiguation and continuity; never merge or split segments.
            - A sentence can run across consecutive segments (for example one line split between two speech bubbles); translate each segment so that, read in order, they form one natural sentence. The subject appears only in the segment where the source states it; the continuing segment does not repeat it or add another.
            - Write every translated `text` value only in {target}; do not include source text, notes, explanations, or alternatives.
            - Never preserve or repeat original-language text; translate names, terms, and sound effects using natural {target} conventions.

            Speakers and consistency:
            - For every segment work out who is speaking to whom, and whether it is spoken dialogue, narration, a monologue, or a thought. Give each its fitting voice: spoken language for dialogue, plainer and more inward for monologue and thoughts. Let age, rank, closeness, and mood shape the wording.
            - Do not assume the speakers alternate: a segment can continue the same speaker, answer the previous one, or come from a third character.
            - The same character keeps the same name, pronouns, and manner of speaking while the situation stays the same; change them only when the relationship or mood clearly changes.
            - Render names, places, skills, and terms the same way every time, following the glossary and context when they are supplied.
            - Keep each character's own way of speaking (polite, rough, childish, archaic, ...).
            Output requirements:
            - Each input segment has a numeric `id`.
            - Return only a JSON object whose `translations` array contains one object with `id` and translated `text` for every input segment.
            - Copy every input ID exactly once; order does not matter.
            - Never merge, split, omit, duplicate, or add segments.
        "},
        source = source,
        target = request.target_language,
    )
    .trim_end()
    .to_owned();

    if request.target_language == Language::Vietnamese {
        prompt.push_str("\n\n");
        prompt.push_str(indoc! {"
            Vietnamese address and subject consistency:
            - Do not add a subject that the source leaves out. Japanese often drops it, and natural Vietnamese can too (行くよ becomes 'Đi nhé', not 'Anh đi nhé'). Add one only if the sentence would be wrong without it and you are sure who is speaking. 私, 僕, 俺 and あたし always mean the speaker, never the listener, and a sentence split over bubbles keeps its subject in the earlier bubble.
            - For every segment decide who speaks and who listens from the bubble tail in the images, the story, and the speaker's way of talking. The first-person word names the speaker and the second-person word the listener, each matching that person's gender. Never fall back on one pair for everybody ('tôi - cậu' and 'anh - em' included).
            - A glossary address entry 'Speaker -> Listener = pair' is the chosen pair: use it. Otherwise reuse the pronouns that the context entries and the earlier pages already show for the same two people, and decide a new pair only from the evidence below. A character keeps one pair with each listener until the relationship or tone clearly changes, and within a sentence refers to the speaker in one way only.
            - Evidence, strongest first. (1) Formality of the source: polite speech (です/ます, さん, 様, 先生) between adults, strangers, colleagues, and customers and workers takes a formal pair ('tôi - cô', 'tôi - anh', 'tôi - chị', or 'em - anh', 'em - chị' for a clearly younger speaker); plain familiar speech (だ, よ, ちゃん, くん, first names) takes a close pair ('anh - em', 'chị - em', 'tớ - cậu', 'tao - mày'). A speaker who talks politely keeps polite endings (ạ, vâng, xin, làm ơn) and does not turn requests into 'đi', 'nhé', 'đấy'; teasing said in polite Japanese stays polite. (2) Age and look: a younger person calls an older man 'anh' ('chú', 'bác' when much older) and an older woman 'chị' ('cô', 'dì' when much older), an older person calls a clearly younger one 'em' or 'cháu', and people of the same age use 'cậu', 'bạn', 'tớ', 'mình'. 'Em' belongs to a clearly younger person or a close bond; do not use 'anh - em' between adults who speak politely or are not close. (3) The source's own words: 俺 casual or rough male (tao, tớ, mình, anh), 僕 gentler male (tớ, mình, em to elders, tôi when polite), 私 neutral or polite (tôi, em), あたし casual female (mình, tớ, em), お前 and 貴様 rude (mày, cậu), 君 (cậu, bạn, em), あなた (anh, chị, bạn). A speaker who adds くん to a man's name sees him as an equal or younger: do not make that speaker call him 'anh'.
            - When neither formality nor age is known, do not guess: use the name, 'bạn', or no pronoun.
            - Vietnamese separates the inclusive 'chúng ta' (speaker and listener) from the exclusive 'chúng tôi', 'bọn anh', 'bọn em' (the speaker's group without the listener). 僕たち or 私たち said to someone outside the group takes the exclusive form.
            - Giving and receiving verbs show direction: ～てくれる means the other person does it for the speaker's side, ～てあげる means the speaker does it for the other, ～てもらう means the speaker receives it. Keep that direction.
            - Render each character name, title, and the way others address them the same way every time.

            Vietnamese fidelity and naturalness:
            - First get the exact meaning of each sentence right; never invent content or guess a different meaning. Check polite forms, homographs, and the sense in context (for example Japanese 要る 'need' versus 居る 'be', 御着用ください 'please wear', だめ! as an exclamation is 'Không được!' and never 'hỏng rồi', さみしがってる is 'đang buồn' or 'đang cô đơn' without adding who misses whom).
            - Then say it the way a Vietnamese speaker would in that situation: short spoken sentences, fitting sentence-final particles (nhé, nhỉ, mà, đấy, thôi, ạ) for the tone and politeness, no word-for-word calques.
            - Translate compound words by their real meaning, never by sounding out their parts, and keep every phrase of the source instead of replacing it with a different description. On profile or label pages use short standard labels such as Tên thật, Tuổi, Nghề nghiệp.
            - Sound effects become a short Vietnamese onomatopoeia or stay as they are; never romanize them.
            - Keep each line about as long as the source allows, because Vietnamese text must fit the same speech bubble.
        "}.trim_end());
    }

    if request.wants_speaker() {
        prompt.push_str("\n\n");
        prompt.push_str(indoc! {"
            Speaker field:
            - Each object has `id`, `speaker` and `text`, written in that order. Fill `speaker` first, in a few words: 'Speaker -> Listener: first-person word - second-person word' (for example 'Rio -> Yuu: em - anh'), or 'narration', or 'thought'. Use the name from the glossary or the context, or a short description such as 'khách' when there is no name.
            - Then write `text` with exactly the pair you wrote, and use the same pair for the same Speaker -> Listener in every segment and on every page until the relationship clearly changes.
        "}.trim_end());
    }

    if !request.context.is_empty() {
        prompt.push_str("\n\n");
        prompt.push_str(indoc! {"
            Context requirements:
            Use the supplied context to follow the story and to preserve terminology, character voice, pronouns, and dialogue continuity.
            Do not translate or return the context entries.
        "}.trim_end());
    }

    if request.page_images.len() > 1 {
        prompt.push_str("\n\n");
        prompt.push_str(&format!(
            indoc! {"
                Image requirements:
                {count} consecutive original page images are attached in reading order: the first image is page 1. Every input segment has a `page` number, and its text is on that page's image. The pages follow each other in the story, so a line can answer someone who spoke on an earlier page: read all the pages as one scene.
                The red outlines on the images are tagged with numbers: each number is the id of the segment whose text lies inside that outline (ids run on from page to page). Use the outline to find the segment's bubble and follow its tail to the speaker, and read the text itself from the input segments, not from the outlines.
                Use the images as visual context for who speaks and who listens, whether a bubble is speech, thought, or narration, the characters' gender, age, and mood, and for tone, layout, and ambiguous OCR.
                Translate only the supplied segments; do not add text seen in the images that is absent from the input segments.
            "},
            count = request.page_images.len(),
        ).trim_end());
    } else if request.image.is_some() || request.page_images.len() == 1 {
        prompt.push_str("\n\n");
        prompt.push_str(indoc! {"
            Image requirements:
            When the attached original page image shows red outlines tagged with numbers, each number is the id of the segment whose text lies inside that outline: use the outline to find the segment's bubble and follow its tail to the speaker, and read the text itself from the input segments, not from the outlines.
            Use the attached original page image as visual context for who speaks and who listens, whether a bubble is speech, thought, or narration, the characters' gender, age, and mood, and for tone, layout, and ambiguous OCR.
            Translate only the supplied segments; do not add text seen in the image that is absent from the input segments.
        "}.trim_end());
    }

    if let Some(instructions) = request
        .instructions
        .as_deref()
        .map(str::trim)
        .filter(|instructions| !instructions.is_empty())
    {
        prompt.push_str("\n\nAdditional instructions:\n");
        prompt.push_str(instructions);
    }
    prompt
}

#[derive(Serialize)]
struct TranslationInput<'a> {
    source_language: Option<Language>,
    target_language: Language,
    context: &'a [TranslationContext],
    segments: Vec<TranslationInputSegment<'a>>,
}

#[derive(Serialize)]
struct TranslationInputSegment<'a> {
    id: usize,
    text: &'a str,
    /// 1-based page of the segment when several pages are translated together.
    #[serde(skip_serializing_if = "Option::is_none")]
    page: Option<usize>,
}

#[derive(Debug, Deserialize)]
struct TranslationOutput {
    translations: Vec<TranslationOutputSegment>,
}

#[derive(Debug, Deserialize)]
struct TranslationOutputSegment {
    #[serde(deserialize_with = "deserialize_segment_id")]
    id: usize,
    /// "Speaker -> Listener: pronoun pair", written before the text so the pronouns are
    /// decided first. Only read back for the logs.
    #[serde(default)]
    #[allow(dead_code)]
    speaker: Option<String>,
    text: String,
}

fn deserialize_segment_id<'de, D>(deserializer: D) -> Result<usize, D::Error>
where
    D: Deserializer<'de>,
{
    #[derive(Deserialize)]
    #[serde(untagged)]
    enum SegmentId {
        Number(usize),
        String(String),
    }

    match SegmentId::deserialize(deserializer)? {
        SegmentId::Number(id) => Ok(id),
        SegmentId::String(id) => id.trim().parse().map_err(de::Error::custom),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn snippet_keeps_a_response_on_one_line() {
        let flattened = snippet("{\n  \"translations\": [\n\t\"hello\"\n  ]\n}");

        assert!(!flattened.contains('\n'), "{flattened}");
        assert!(!flattened.contains('\t'), "{flattened}");
        assert_eq!(flattened, r#"{\n  "translations": [\n\t"hello"\n  ]\n}"#);
    }

    #[test]
    fn snippet_escapes_other_control_characters() {
        assert_eq!(snippet("before\u{7}after"), r"before\u{0007}after");
    }

    #[test]
    fn snippet_reports_the_length_it_elided() {
        let flattened = snippet(&"x".repeat(2_500));

        assert!(flattened.starts_with(&"x".repeat(2_000)));
        assert!(
            flattened.ends_with("\u{2026} (2500 bytes total)"),
            "{flattened}"
        );
    }

    #[test]
    fn parses_plain_json() {
        let source = ["one".to_owned(), "two".to_owned()];
        let response = r#"{"translations":[{"id":0,"text":"hello"},{"id":1,"text":"world"}]}"#;

        assert_eq!(
            translations("test", response, &source).unwrap(),
            ["hello", "world"]
        );
    }

    #[test]
    fn rejects_wrapped_and_malformed_json() {
        let source = ["one".to_owned(), "two".to_owned()];
        for response in [
            "```json\n{\"translations\":[{\"id\":0,\"text\":\"hello\"},{\"id\":1,\"text\":\"world\"}]}\n```",
            r#"{translations: [{id: 0, text: 'hello'}, {id: 1, text: 'world'},],}"#,
            r#"Here is the result: {"translations": [{"id": 0, "text": "hello"}, {"id": 1, "text": "world"},]}"#,
            "{\"translations\":[{\"id\":0,\"text\":\"hello\"},{\"id\":1,\"text\":\"world\"",
        ] {
            assert!(
                translations("test", response, &source).is_err(),
                "{response}"
            );
        }
    }

    #[test]
    fn parse_error_preserves_the_root_failure_and_response() {
        let source = ["one".to_owned()];
        let response = "{\n  \"translations\": [{\"id\": 0}]\n}";
        let error = format!(
            "{:#}",
            translations("test", response, &source).expect_err("missing text should fail")
        );

        assert!(error.contains("missing field `text`"), "{error}");
        assert!(
            error.contains(r#"response was: {\n  "translations": [{"id": 0}]\n}"#),
            "{error}"
        );
    }

    #[test]
    fn restores_input_order_from_ids() {
        let source = ["one".to_owned(), "two".to_owned()];
        let response = r#"{"translations":[{"id":1,"text":"world"},{"id":0,"text":"hello"}]}"#;
        assert_eq!(
            translations("test", response, &source).unwrap(),
            ["hello", "world"]
        );
    }

    #[test]
    fn tolerates_duplicate_missing_and_out_of_range_ids() {
        let source = ["one".to_owned(), "two".to_owned()];
        let short = r#"{"translations":[{"id":1,"text":"world"}]}"#;
        assert_eq!(
            translations("test", short, &source).unwrap(),
            ["one", "world"]
        );

        let response = concat!(
            r#"{"translations":["#,
            r#"{"id":0,"text":"hello"},"#,
            r#"{"id":0,"text":"duplicate"},"#,
            r#"{"id":9,"text":"extra"}"#,
            "]}"
        );
        assert_eq!(
            translations("test", response, &source).unwrap(),
            ["hello", "two"]
        );
    }

    #[test]
    fn prompt_payload_contains_ordered_context() {
        let request = TranslationRequest::new(["new"], Language::English)
            .with_context([TranslationContext::new("old", "previous")]);
        let (_, user) = prompts(&request).unwrap();
        let input: serde_json::Value = serde_json::from_str(&user).unwrap();
        assert_eq!(input["context"][0]["source"], "old");
        assert_eq!(input["context"][0]["translation"], "previous");
        assert_eq!(input["segments"][0]["id"], 0);
        assert_eq!(input["segments"][0]["text"], "new");
    }

    #[test]
    fn system_prompt_encodes_invariants_and_custom_instructions() {
        let request = TranslationRequest::new(["hello"], Language::Korean)
            .with_source_language(Language::Japanese)
            .with_instructions("Use informal speech.");
        let prompt = translation_system_prompt(&request);
        assert!(prompt.contains("from Japanese into natural Korean"));
        assert!(prompt.contains("Copy every input ID exactly once"));
        assert!(prompt.contains("Use informal speech."));
    }

    #[test]
    fn schema_requires_the_expected_number_of_id_text_pairs() {
        let schema = output_schema(3);
        let translations = &schema["properties"]["translations"];
        assert_eq!(translations["minItems"], 3);
        assert_eq!(translations["maxItems"], 3);
        assert_eq!(translations["items"]["properties"]["id"]["minimum"], 0);
        assert_eq!(translations["items"]["properties"]["id"]["maximum"], 2);
        assert_eq!(translations["items"]["additionalProperties"], false);
        assert_eq!(schema["additionalProperties"], false);
    }

    #[test]
    fn empty_custom_instructions_are_ignored() {
        let request = TranslationRequest::new(["hello"], Language::English).with_instructions("  ");
        assert!(!translation_system_prompt(&request).contains("Additional instructions"));
    }

    #[test]
    fn context_is_reference_only() {
        let request = TranslationRequest::new(["Where is she?"], Language::Japanese)
            .with_context([TranslationContext::new("I saw Alice.", "アリスを見た。")]);
        let prompt = translation_system_prompt(&request);
        assert!(prompt.contains("dialogue continuity"));
        assert!(prompt.contains("Do not translate or return the context"));
    }

    #[test]
    fn image_context_does_not_expand_the_translation_scope() {
        let request = TranslationRequest::new(["text"], Language::English)
            .with_image(std::sync::Arc::new(image::DynamicImage::new_rgb8(1, 1)));
        let prompt = translation_system_prompt(&request);
        assert!(prompt.contains("attached original page image"));
        assert!(prompt.contains("Translate only the supplied segments"));
    }

    #[test]
    fn salvage_keeps_complete_segments_from_a_truncated_response() {
        let sources = vec!["a".to_owned(), "b".to_owned(), "c".to_owned()];
        let text = r#"{"translations":[{"id":0,"text":"Xin chào"},{"id":1,"text":"lặp lặp lặp lặp"#;

        let (translations, missing) = salvage_translations(text, &sources);

        assert_eq!(translations, ["Xin chào", "b", "c"]);
        assert_eq!(missing, [1, 2]);
    }

    #[test]
    fn salvage_recovers_nothing_from_unusable_text() {
        let sources = vec!["a".to_owned(), "b".to_owned()];
        for text in ["", "not json", r#"{"translations":"#, r#"{"translations":[{"id":0"#] {
            let (translations, missing) = salvage_translations(text, &sources);
            assert_eq!(translations, ["a", "b"], "{text}");
            assert_eq!(missing, [0, 1], "{text}");
        }
    }

    #[test]
    fn invalid_output_keeps_the_raw_response() {
        let sources = vec!["a".to_owned()];
        let response = r#"{"translations":[{"id":0,"text":"cut"#;
        let error = translations("test", response, &sources).unwrap_err();
        let invalid = error
            .downcast_ref::<InvalidTranslationOutput>()
            .expect("typed context");
        assert_eq!(invalid.response, response);
    }

    #[test]
    fn glossary_output_is_parsed_even_when_truncated() {
        let entry = |kind: &str, source: &str, target: &str| GlossaryEntry {
            kind: kind.to_owned(),
            source: source.to_owned(),
            target: target.to_owned(),
        };
        let full = r#"{"entries":[{"kind":"name","source":"田中","target":"Tanaka"}]}"#;
        assert_eq!(parse_glossary(full), [entry("name", "田中", "Tanaka")]);
        let cut = r#"{"entries":[{"kind":"name","source":"田中","target":"Tanaka"},{"kind":"term","source":"霊力","target":"linh"#;
        assert_eq!(parse_glossary(cut), [entry("name", "田中", "Tanaka")]);
        assert!(parse_glossary("garbage").is_empty());
        assert!(parse_glossary(r#"{"entries":[]}"#).is_empty());
    }

    #[test]
    fn glossary_is_rendered_one_entry_per_line() {
        let entries = vec![
            GlossaryEntry {
                kind: "name".to_owned(),
                source: "田中".to_owned(),
                target: "Tanaka".to_owned(),
            },
            GlossaryEntry {
                kind: "term".to_owned(),
                source: "霊力".to_owned(),
                target: "linh lực".to_owned(),
            },
            GlossaryEntry {
                kind: "address".to_owned(),
                source: "田中 -> 鈴木".to_owned(),
                target: "tớ - cậu".to_owned(),
            },
        ];
        let block = render_glossary(&entries);
        assert!(block.starts_with("Glossary"));
        assert!(block.contains("\n- name: 田中 = Tanaka"));
        assert!(block.contains("\n- term: 霊力 = linh lực"));
        assert!(block.contains("\n- address: 田中 -> 鈴木 = tớ - cậu"));
    }

    #[test]
    fn vietnamese_prompts_carry_address_consistency_rules() {
        let vietnamese =
            translation_system_prompt(&TranslationRequest::new(["a"], Language::Vietnamese));
        assert!(vietnamese.contains("Vietnamese address and subject consistency"));
        assert!(vietnamese.contains("Vietnamese fidelity and naturalness"));
        let english = translation_system_prompt(&TranslationRequest::new(["a"], Language::English));
        assert!(!english.contains("Vietnamese address and subject consistency"));
    }

    #[test]
    fn parse_translations_reports_missing_and_duplicate_ids() {
        let sources = vec!["a".to_owned(), "b".to_owned(), "c".to_owned()];
        let text = r#"{"translations":[{"id":0,"text":"A"},{"id":0,"text":"dup"},{"id":2,"text":"C"}]}"#;

        let (translations, missing) = parse_translations("test", text, &sources).unwrap();

        assert_eq!(translations, ["A", "b", "C"]);
        assert_eq!(missing, [1]);
    }

    #[test]
    fn the_address_pass_reads_character_profiles() {
        let characters = vec![GlossaryEntry {
            kind: "character".to_owned(),
            source: "田中".to_owned(),
            target: "Nam, 56 tuổi, khách quen".to_owned(),
        }];
        let (system, user) = address_prompts(Language::Vietnamese, &characters).unwrap();
        assert!(system.contains("Speaker -> Listener"));
        assert!(system.contains("into Vietnamese"));
        let input: serde_json::Value = serde_json::from_str(&user).unwrap();
        assert_eq!(input["characters"][0]["name"], "田中");
        assert_eq!(input["characters"][0]["profile"], "Nam, 56 tuổi, khách quen");
        assert!(decides_address(Language::Vietnamese));
        assert!(!decides_address(Language::English));
    }

    fn two_pages(language: Language) -> TranslationRequest {
        let blank = || std::sync::Arc::new(image::DynamicImage::new_rgb8(1, 1));
        TranslationRequest::new(["a", "b", "c"], language)
            .with_pages(vec![blank(), blank()], vec![0, 0, 1])
    }

    #[test]
    fn pages_translated_together_carry_their_page_and_images() {
        let request = two_pages(Language::Vietnamese);
        let (system, user) = prompts(&request).unwrap();
        assert!(system.contains("2 consecutive original page images"));
        assert!(system.contains("Speaker field:"));
        let input: serde_json::Value = serde_json::from_str(&user).unwrap();
        assert_eq!(input["segments"][0]["page"], 1);
        assert_eq!(input["segments"][2]["page"], 2);

        let single = TranslationRequest::new(["a"], Language::Vietnamese)
            .with_pages(vec![std::sync::Arc::new(image::DynamicImage::new_rgb8(1, 1))], vec![0]);
        let (_, user) = prompts(&single).unwrap();
        let input: serde_json::Value = serde_json::from_str(&user).unwrap();
        assert!(input["segments"][0].get("page").is_none());
    }

    #[test]
    fn the_speaker_is_written_between_the_id_and_the_text() {
        let schema = request_schema(&two_pages(Language::Vietnamese));
        let item = &schema["properties"]["translations"]["items"];
        assert_eq!(item["required"], serde_json::json!(["id", "speaker", "text"]));
        let keys: Vec<&String> = item["properties"].as_object().unwrap().keys().collect();
        assert_eq!(keys, ["id", "speaker", "text"]);

        let plain = request_schema(&two_pages(Language::English));
        let item = &plain["properties"]["translations"]["items"];
        assert_eq!(item["required"], serde_json::json!(["id", "text"]));
        let single = request_schema(&TranslationRequest::new(["a"], Language::Vietnamese));
        assert_eq!(single, output_schema(1));
    }

    #[test]
    fn the_speaker_in_the_output_is_accepted_and_ignored() {
        let sources = vec!["a".to_owned(), "b".to_owned()];
        let text = r#"{"translations":[{"id":0,"speaker":"Rio -> Yuu: em - anh","text":"A"},{"id":1,"text":"B"}]}"#;
        let (translations, missing) = parse_translations("test", text, &sources).unwrap();
        assert_eq!(translations, ["A", "B"]);
        assert!(missing.is_empty());
    }

    #[test]
    fn selecting_segments_keeps_their_pages() {
        let subset = two_pages(Language::Vietnamese).select(&[1, 2]);
        assert_eq!(subset.segments, ["b", "c"]);
        assert_eq!(subset.segment_pages, [0, 1]);
        assert_eq!(subset.page_images.len(), 2);
    }

    #[test]
    fn the_glossary_asks_for_names_and_terms_without_profiles() {
        let (system, _) =
            glossary_prompts(Language::Vietnamese, &[], &["a".to_owned()]).unwrap();
        assert!(!PROFILES);
        assert!(system.contains("- name:"));
        assert!(!system.contains("- character:"));
        assert!(!system.contains("- address:"));
        assert!(system.contains("style, name, or term"));
    }
            }
