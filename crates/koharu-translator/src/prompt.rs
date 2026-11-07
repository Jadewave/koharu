use std::fmt::Write;

use anyhow::Context;
use indoc::indoc;
use serde::{Deserialize, Deserializer, Serialize, de};
use serde_json::{Value, json};

use crate::{GlossaryEntry, Language, TranslationContext, TranslationRequest};

pub(crate) fn prompts(request: &TranslationRequest) -> anyhow::Result<(String, String)> {
    prompts_with_address_observations(request, false)
}

/// Local Vision translation can learn directional address pairs from the same response.
/// Remote providers keep the original prompt and JSON schema unless explicitly opted in.
pub(crate) fn prompts_with_address_observations(
    request: &TranslationRequest,
    include_address_observations: bool,
) -> anyhow::Result<(String, String)> {
    let input = TranslationInput {
        source_language: request.source_language,
        target_language: request.target_language,
        context: &request.context,
        segments: request
            .segments
            .iter()
            .enumerate()
            .map(|(id, text)| TranslationInputSegment { id, text })
            .collect(),
    };
    let user = serde_json::to_string(&input).context("failed to serialize translation input")?;
    Ok((
        translation_system_prompt_with_address_observations(request, include_address_observations),
        user,
    ))
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
    let (translations, missing, _) =
        parse_translations_with_addresses(provider, text, source_segments)?;
    Ok((translations, missing))
}

/// Parses translations and any address observations emitted by the local image-grounded pass.
/// Observations are kept only when they point to an existing input segment and contain complete,
/// finite confidence/evidence fields. Character-name validation happens in the glossary owner.
pub(crate) fn parse_translations_with_addresses(
    provider: &str,
    text: &str,
    source_segments: &[String],
) -> anyhow::Result<(Vec<String>, Vec<usize>, Vec<AddressObservation>)> {
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
    let observations = output
        .address_observations
        .into_iter()
        .filter(|item| {
            item.segment_id < source_segments.len()
                && translated.get(item.segment_id).copied().unwrap_or(false)
                && item.confidence.is_finite()
                && (0.85..=1.0).contains(&item.confidence)
                && !item.speaker.trim().is_empty()
                && !item.listener.trim().is_empty()
                && !item.first_person.trim().is_empty()
                && !item.second_person.trim().is_empty()
                && item.evidence.trim().chars().count() >= 8
        })
        .collect();
    Ok((translations, missing, observations))
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
            - term: a recurring term, skill, technique, or item and the rendering to use for it. Leave sound effects out, because the translator renders them on each page, and never romanize anything.
            - character: `source` is the bare name as it appears in the lines, without honorifics and never a pronoun, with one entry per character even when the lines write or call the name in several ways (for a recurring character without a name, a short description such as the regular customer); `target` is a short profile in {target}: gender and age only when supported by the text, role, explicitly established relationships, and manner of speaking (formal, casual, rough, childish, archaic, ...). Record a first-person pronoun only when the source text makes it clear. Mark uncertain facts as unknown instead of guessing.
            - style: `source` is exactly work, `target` one short note on genre, setting, and overall register. At most one, and only if the glossary has none yet.

            Rules:
            - Choose natural renderings that will still fit the rest of the work.
            - Be exhaustive about named characters, places, organizations, titles, and recurring terms in the lines, but do not invent a speaker/listener relationship from OCR text alone.
            - Do not infer a character's gender from their name, age from their apparent role, or closeness from casual wording alone. Prefer unknown to an unsupported guess.
            - Do not create `address` entries here. This OCR-only pass has no reliable speaker-to-bubble mapping; address pairs are learned only during an image-grounded translation when both people and the relationship are supported.
            - Skip entries already in the glossary you were given and anything that does not appear in these lines. Never list the same character again under another spelling, honorific, or pronoun.
            - Write `kind` as exactly style, character, name, or term.
            - At most 20 entries, every string short.
            - Return an empty `entries` array when there is nothing new.
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
    output_schema_with_addresses(expected, false)
}

/// Local image translation may additionally return a small set of strongly grounded,
/// directional address observations. Remote providers retain the legacy schema.
pub(crate) fn output_schema_with_addresses(expected: usize, include_addresses: bool) -> Value {
    let mut properties = json!({
        "translations": {
            "type": "array",
            "minItems": expected,
            "maxItems": expected,
            "items": {
                "type": "object",
                "properties": {
                    "id": {
                        "type": "integer",
                        "minimum": 0,
                        "maximum": expected.saturating_sub(1),
                        "description": "The ID copied from the corresponding input segment."
                    },
                    "text": {
                        "type": "string",
                        "description": "The translation of the input segment with this ID."
                    }
                },
                "required": ["id", "text"],
                "additionalProperties": false
            }
        }
    });
    let mut required = vec!["translations"];

    if include_addresses {
        properties["address_observations"] = json!({
            "type": "array",
            "maxItems": 8,
            "items": {
                "type": "object",
                "properties": {
                    "segment_id": {
                        "type": "integer",
                        "minimum": 0,
                        "maximum": expected.saturating_sub(1),
                        "description": "The input dialogue segment that grounds this observation."
                    },
                    "speaker": { "type": "string" },
                    "listener": { "type": "string" },
                    "first_person": { "type": "string" },
                    "second_person": { "type": "string" },
                    "confidence": { "type": "number", "minimum": 0.85, "maximum": 1.0 },
                    "evidence": { "type": "string" }
                },
                "required": ["segment_id", "speaker", "listener", "first_person", "second_person", "confidence", "evidence"],
                "additionalProperties": false
            }
        });
        required.push("address_observations");
    }

    json!({
        "type": "object",
        "properties": properties,
        "required": required,
        "additionalProperties": false
    })
}

fn translation_system_prompt(request: &TranslationRequest) -> String {
    translation_system_prompt_with_address_observations(request, false)
}

fn translation_system_prompt_with_address_observations(
    request: &TranslationRequest,
    include_address_observations: bool,
) -> String {
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
            - Do not add a subject that the source leaves out. Japanese often drops it, and natural Vietnamese can too (行くよ becomes 'Đi nhé', not 'Anh đi nhé'). Add a subject only if the sentence would be wrong without it and you are sure who is speaking, using that character's pronoun. When a sentence is split over bubbles, the subject stays in the earlier bubble.
            - For every dialogue segment, identify the speaker and listener from the tagged bubble, tail, image, and scene before choosing pronouns. Match both people to the character profiles; do not identify them from gender or appearance alone. The first word in an address pair belongs to the speaker and the second to the listener.
            - An existing `address` glossary entry is directional: use it only when both speaker and listener match, in that order. Never reverse it, apply it to a different pair, or use it just because one name appears in OCR. Keep a supported pair stable across pages.
            - If either person cannot be identified confidently, or no reliable pair exists, do not guess a gendered relationship. Prefer omitting the subject when Vietnamese allows it; otherwise use a name or a neutral form such as 'bạn'. Never fall back on one pair for everybody ('tôi - cậu' and 'anh - em' included).
            - When the source is Japanese, its own words are evidence: 俺 is a casual or rough male voice (tao, tớ, mình, anh), 僕 a gentler or younger male one (tớ, mình, em to elders), 私 neutral or polite (tôi), あたし a casual female one (mình, tớ, em), お前 / 貴様 rude (mày, cậu), 君 (cậu, bạn, em), あなた (anh, chị, bạn). Honorifics (-kun, -san, -chan, -senpai, -sama, 先生) show the relationship.
            - When the source is Chinese, 您 is a respect/politeness cue while 你 alone does not prove intimacy; terms such as 哥、姐、老师、前辈、师父 and family titles can establish role or relationship when the scene confirms who they refer to. Distinguish 咱们 (usually inclusive) from 我们 (inclusive or exclusive by context).
            - When the source is Korean, 저 and 나 help indicate speech level, and -요 / -습니다 usually signal polite speech. Terms such as 형、오빠、누나、언니、선배、선생님、씨 and 님 are relationship or status cues only when the speaker and listener are identified. Do not translate 당신 as a universal 'you'; Korean often omits the direct pronoun.
            - A character keeps one pronoun pair with each listener while the relationship stays the same. A matching directional `address` glossary entry is a HARD RULE: if it says `tôi - cậu`, use `tôi` for that speaker and `cậu` for that listener consistently; do not replace either word with `anh`, `chị`, `em`, `mày`, or another alternative just because a source pronoun could be translated that way. Change the pair only when a clearly established relationship/status change is present. Within a sentence refer to the speaker in one way only ('anh đây' and 'tôi' never together). Use recent dialogue context only when there is no matching pair; never treat an entry for another pair as a general rule.
            - Address follows established formality and relationship first, then age. Polite speech (です/ます forms, さん, 様, 先生) often calls for a formal pair; plain familiar speech (だ, よ, ちゃん, くん, first names) can support a closer pair, but these cues alone do not prove intimacy. Do not use 'anh - em' between adults who are polite or not close. Use ages only when the glossary establishes them; do not infer age from appearance alone. When relationship, age, and formality are unknown, do not guess: use a name, 'bạn', or no pronoun.
            - Vietnamese separates the inclusive 'chúng ta' (speaker and listener) from the exclusive 'chúng tôi', 'bọn anh', 'bọn em' (the speaker's group without the listener). 僕たち or 私たち said to someone outside the group takes the exclusive form.
            - Giving and receiving verbs show direction: ～てくれる means the other person does it for the speaker's side, ～てあげる means the speaker does it for the other, ～てもらう means the speaker receives it. Keep that direction.
            - Render each character name, title, and the way others address them the same way every time.

            Vietnamese fidelity and naturalness:
            - First get the exact meaning of each sentence right; never invent content or guess a different meaning. Check polite forms and homographs (for example Japanese 要る 'need' versus 居る 'be', 御着用ください 'please wear').
            - Then say it the way a Vietnamese speaker would in that situation: short spoken sentences, fitting sentence-final particles (nhé, nhỉ, mà, đấy, thôi, ạ) for the tone and politeness, no word-for-word calques.
            - Translate compound words by their real meaning, never by sounding out their parts, and keep every phrase of the source instead of replacing it with a different description. On profile or label pages use short standard labels such as Tên thật, Tuổi, Nghề nghiệp.
            - Sound effects become a short Vietnamese onomatopoeia or stay as they are; never romanize them.
            - Keep each line about as long as the source allows, because Vietnamese text must fit the same speech bubble.
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

    if include_address_observations
        && request.image.is_some()
        && request.target_language == Language::Vietnamese
    {
        prompt.push_str("\n\n");
        prompt.push_str(indoc! {"
            Address learning in this same image-grounded pass:
            - Before writing translations, decide whether this page contains any supported directional address pairs. Apply each chosen pair consistently to all matching dialogue segments on this page.
            - Return an `address_observations` array in the JSON. It may be empty; do not invent observations to fill it.
            - Each observation must refer to one supplied dialogue segment using `segment_id`, and identify `speaker` and `listener` by the exact character names in the supplied character profiles. Do not use visual descriptions as names.
            - The image/bubble tail may establish who is speaking, but image, age, gender, or apparent seniority alone cannot establish how they address each other.
            - Record a directional pair only when BOTH speaker and listener are identified with high confidence AND the Vietnamese first-/second-person choice is supported by explicit address language, an already established relationship/context, or a clear recurring speech pattern. Japanese omission of pronouns is not evidence for a new pair.
            - `first_person` is how this speaker refers to themselves; `second_person` is how this same speaker addresses this listener. Never reverse the pair or infer the reverse direction. When the same pair speaks again on this page, reuse the same choice instead of switching pronouns mid-page.
            - `evidence` must briefly name the concrete source cue and/or visual linkage for this exact segment. Do not use vague evidence such as 'context' or 'they seem close'.
            - Use confidence only from 0.85 to 1.0. If any part is uncertain, omit the observation. Never turn a guess into a permanent glossary rule.
            - Do not emit observations for narration, internal thoughts, sound effects, or speech with an unknown addressee.
            - Before returning, perform a consistency check across this page: whenever a known or confidently inferred directional address pair applies, make every translated bubble between that speaker and listener follow the exact same pair. Correct inconsistent pronouns before emitting the final translations, but do not insert pronouns that the source omits.
            - Include `address_observations` in the JSON response, even when there are no supported observations; use an empty array in that case.
        "}.trim_end());
    }

    if request.image.is_some() {
        prompt.push_str("\n\n");
        prompt.push_str(indoc! {"
            Image requirements:
            In this same translation pass, visually ground each numbered segment: when the attached original page image shows red outlines tagged with numbers, each number is the id of the segment whose text lies inside that outline. Use the outline to locate its bubble and follow the bubble tail to the speaker; identify the listener only when the scene supports it. Read the dialogue text from the supplied input segments, not from the red outlines.
            Use the attached original page image as visual context for speaker/listener identity, speech versus thought versus narration, character appearance, emotion, tone, layout, and ambiguous OCR. Apply the established character profiles and directional address pairs. If speaker or listener identity is uncertain, do not guess a pronoun pair; use a name, a neutral form, or omit the pronoun.
            Do not produce a separate visual report or add text seen in the image that is absent from the input segments; return only translations for the supplied segments.
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
}

#[derive(Debug, Deserialize)]
struct TranslationOutput {
    translations: Vec<TranslationOutputSegment>,
    #[serde(default)]
    address_observations: Vec<AddressObservation>,
}

/// A directional address choice observed during the same image-grounded translation pass.
/// It is only persisted after the caller checks both names against known character profiles.
#[derive(Debug, Clone, Deserialize)]
pub(crate) struct AddressObservation {
    #[serde(deserialize_with = "deserialize_segment_id")]
    pub(crate) segment_id: usize,
    pub(crate) speaker: String,
    pub(crate) listener: String,
    pub(crate) first_person: String,
    pub(crate) second_person: String,
    pub(crate) confidence: f32,
    pub(crate) evidence: String,
}

#[derive(Debug, Deserialize)]
struct TranslationOutputSegment {
    #[serde(deserialize_with = "deserialize_segment_id")]
    id: usize,
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
        assert!(vietnamese.contains("HARD RULE"));
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
    fn glossary_profile_pass_does_not_invent_address_pairs_from_ocr() {
        let (system, _) = glossary_prompts(
            Language::Vietnamese,
            &[],
            &["おはよう".to_owned()],
        )
        .unwrap();
        assert!(system.contains("Do not create `address` entries here"));
        assert!(system.contains("Prefer unknown to an unsupported guess"));
    }


    #[test]
    fn local_vision_schema_includes_optional_directional_address_observations() {
        let schema = output_schema_with_addresses(2, true);
        assert!(schema["required"].as_array().unwrap().iter().any(|v| v == "address_observations"));
        assert_eq!(schema["properties"]["address_observations"]["items"]["properties"]["confidence"]["minimum"], 0.85);
        assert!(output_schema_with_addresses(2, false)["properties"].get("address_observations").is_none());
    }

    #[test]
    fn parser_keeps_only_high_confidence_address_observations_for_valid_segments() {
        let sources = vec!["お前がやれ".to_owned(), "ありがとう".to_owned()];
        let response = r#"{
            "translations":[{"id":0,"text":"Cậu làm đi"},{"id":1,"text":"Cảm ơn"}],
            "address_observations":[
                {"segment_id":0,"speaker":"Tanaka","listener":"Suzuki","first_person":"tôi","second_person":"cậu","confidence":0.93,"evidence":"đuôi bong bóng trỏ về Tanaka; dùng お前"},
                {"segment_id":1,"speaker":"Tanaka","listener":"Suzuki","first_person":"tôi","second_person":"anh","confidence":0.70,"evidence":"không rõ"},
                {"segment_id":9,"speaker":"Tanaka","listener":"Suzuki","first_person":"tôi","second_person":"cậu","confidence":0.98,"evidence":"bằng chứng cụ thể"}
            ]
        }"#;

        let (translations, missing, observations) =
            parse_translations_with_addresses("test", response, &sources).unwrap();
        assert_eq!(translations, ["Cậu làm đi", "Cảm ơn"]);
        assert!(missing.is_empty());
        assert_eq!(observations.len(), 1);
        assert_eq!(observations[0].speaker, "Tanaka");
        assert_eq!(observations[0].second_person, "cậu");
    }

    #[test]
    fn ordinary_prompt_does_not_request_address_observations_even_with_an_image() {
        let request = TranslationRequest::new(["お前がやれ"], Language::Vietnamese)
            .with_image(std::sync::Arc::new(image::DynamicImage::new_rgb8(1, 1)));
        let prompt = translation_system_prompt(&request);
        assert!(!prompt.contains("Address learning in this same image-grounded pass"));
        assert!(prompt.contains("Image requirements"));
    }

    #[test]
    fn image_vietnamese_prompt_explains_conservative_address_learning() {
        let request = TranslationRequest::new(["お前がやれ"], Language::Vietnamese)
            .with_image(std::sync::Arc::new(image::DynamicImage::new_rgb8(1, 1)));
        let prompt = translation_system_prompt_with_address_observations(&request, true);
        assert!(prompt.contains("Address learning in this same image-grounded pass"));
        assert!(prompt.contains("image, age, gender, or apparent seniority alone"));
        assert!(prompt.contains("address_observations"));
    }

}
