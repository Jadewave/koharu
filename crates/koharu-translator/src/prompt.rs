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

/// Builds a reusable glossary from OCR text without inferring character biographies or
/// relationships that OCR alone cannot establish.
pub(crate) fn glossary_prompts(
    target: Language,
    glossary: &[GlossaryEntry],
    lines: &[String],
) -> anyhow::Result<(String, String)> {
    let input = GlossaryInput { glossary, lines };
    let user = serde_json::to_string(&input).context("failed to serialize glossary input")?;
    let system = format!(
        indoc! {"
            You maintain a reusable glossary for an illustrated story translated into {target}.
            It may be a comic, manga, manhwa, manhua, webtoon, graphic novel, or another visual narrative.
            Do not assume any particular country, source language, culture, genre, audience, or reading direction.

            You receive the current glossary and OCR source lines in the order supplied. Return only useful NEW entries:
            - name: a named character, place, organization, group, title, rank, or recurring named entity, with a stable {target} rendering. Keep a core name separate from honorifics, titles, ranks, kinship terms, and temporary forms of address. Do not invent a full name from a partial one.
            - term: a recurring story-specific term, ability, technique, object, species, faction, place type, or world-building concept and its stable {target} rendering. Do not add ordinary vocabulary or one-off sound effects.
            - style: `source` is exactly `work`; `target` is a concise note about register, genre, or setting only when the supplied lines provide clear evidence. Do not guess from one isolated line.

            Rules:
            - Identify the actual source language or writing system from the supplied text; a work may contain more than one language or code-switching. Preserve source distinctions rather than applying conventions from another language.
            - Use natural, stable {target} renderings. Transliterate or retain names according to established {target} conventions. Do not translate a person's name by its dictionary meaning unless the story clearly uses wordplay or a meaningful alias.
            - Distinguish names, aliases, titles, ranks, kinship terms, pronouns, nicknames, and honorifics. Do not store a temporary form of address as a character's name.
            - Do not infer gender, exact age, family ties, romance, hierarchy, personality, or relationships from OCR text alone. These require direct story evidence and are considered during page translation.
            - Leave sound effects out of the glossary; translate them in their local visual and narrative context.
            - Skip entries already present and anything not supported by the supplied lines. Merge spelling variants only when they clearly refer to the same entity.
            - Write `kind` as exactly `style`, `name`, or `term`. Return at most 20 concise entries; return an empty `entries` array when there is nothing new.
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
        "Work glossary (use stable names and recurring terms when the context confirms they refer to the same entity):",
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
    json!({
        "type": "object",
        "properties": {
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
        },
        "required": ["translations"],
        "additionalProperties": false
    })
}

/// The instructions that stay the same from page to page of a run. They come first in the
/// system prompt, so the model reads them once and reuses them for the next pages.
fn translation_rules(request: &TranslationRequest) -> String {
    let source = request
        .source_language
        .map(|language| language.to_string())
        .unwrap_or_else(|| "the source language detected from each segment".to_owned());
    let mut prompt = format!(
        indoc! {"
            You are a professional translator of comics and illustrated stories.
            Work only from the supplied material. Do not assume a particular country, language family, culture, genre, character archetype, or reading direction.

            Translation priorities, in order:
            1. Preserve the exact meaning, facts, participants, intent, and logical relationships of the source.
            2. Preserve the speaker's tone, emotion, register, voice, humor, and subtext when supported by context.
            3. Express the same meaning naturally, concisely, and fluently in {target}.

            Core translation rules:
            - Translate primarily from {source} into {target}. Treat a declared source language as the default, not as proof that every segment uses only that language. Recognize quotations, mixed languages, dialects, names, loanwords, and code-switching within a work. If no source language is specified, identify the language of each segment from its actual text and script; never infer it from the work's presumed country or genre.
            - Do not summarize, censor, embellish, explain, or add information not supported by the source and context. Do not omit meaningful details, negation, modality, tense/aspect, quantities, names, or who performs an action and who receives it.
            - Avoid unnatural word-for-word calques, but do not paraphrase away distinctions or details. Localize idioms, humor, slang, and wordplay only when the intended meaning is clear; preserve the same implication and emotional effect.
            - Treat the supplied OCR segments as the text to translate. Use images to understand layout, scene, speaker, listener, expression, action, and genuinely ambiguous OCR. Never invent unreadable text or translate other text visible in an image when it is not included in the input segments.
            - Keep each input segment separate and in supplied reading order. Do not assume left-to-right or right-to-left order, and do not merge or split segments. If a sentence continues across bubbles, make each part read naturally while preserving its contribution to the full sentence.
            - Distinguish dialogue, inner thought, narration, captions, signs, letters, interface text, labels, and sound effects. Do not turn narration into dialogue or sound effects into spoken sentences.
            - Render names, aliases, titles, ranks, organizations, recurring terms, and world-specific concepts consistently with the glossary. Keep the distinction between a name, title, rank, kinship term, nickname, and honorific. Do not translate a proper name by dictionary meaning unless the story clearly makes it wordplay or a meaningful alias.
            - Translate sound effects according to the depicted action, texture, intensity, and tone. Do not force every source sound into one fixed equivalent; retain the original only when that is the most natural choice or the effect is intentionally stylized.

            Character and visual continuity:
            - Use bubble tails, panel composition, visual action, expressions, dialogue content, and prior context together to identify the speaker and listener. Characters do not necessarily alternate.
            - A character's apparent gender, body shape, clothing, screen position, or proximity alone is not sufficient proof of exact age, identity, relationship, social rank, or who is speaking.
            - Maintain a consistent voice and way of addressing others while the same relationship and situation continue. Change them only when the story provides evidence of a change in role, relationship, distance, formality, conflict, intimacy, or disguise.
            - Do not invent character biographies or relationships. Do not assume characters are relatives, romantic partners, friends, or enemies based only on appearance or one isolated exchange.
            - When identity or intent is genuinely uncertain, preserve the source ambiguity and choose a neutral natural construction instead of fabricating a fact.

            Output requirements:
            - Every input segment has a numeric `id`.
            - Return only a JSON object whose `translations` array contains exactly one object for every input segment.
            - Copy each input ID exactly once. Do not duplicate, omit, merge, split, or add segments.
            - Every `text` value must contain only the {target} translation, without source text, notes, explanations, or alternatives.
        "},
        source = source,
        target = request.target_language,
    )
    .trim_end()
    .to_owned();

    if request.target_language == Language::Vietnamese {
        prompt.push_str("\n\n");
        prompt.push_str(indoc! {"
            Vietnamese dialogue and address consistency:
            - Vietnamese pronouns and forms of address encode the relationship between a specific speaker and listener. Treat the choice as context-dependent; do not force one fixed pair on all characters or works.
            - Base a choice on evidence in this work: prior dialogue between the same people, explicit self-reference or direct forms of address in the source, relationships or roles established by the story, and conversational register. Use visual cues as supporting evidence for identity and scene interpretation, not as the sole basis for age, gender, rank, or relationship.
            - Do not mechanically map a pronoun, suffix, honorific, grammatical marker, or politeness form from the source language to a Vietnamese pronoun. First determine its function in that specific language and context. Never apply a language-specific rule to a different source language.
            - Distinguish the speaker's self-reference from the way the speaker addresses the listener. The two directions between two characters need not be symmetrical.
            - Use a name, an address term, or omit a pronoun only when it preserves the source meaning and sounds natural. Do not add a subject, listener, or relationship that changes who acts, feels, requests, gives, receives, benefits, or is affected.
            - Keep address choices consistent for the same people under the same circumstances, but change them when the story establishes a real change in relationship, role, distance, formality, or emotion.
            - If the speaker, listener, or relationship is uncertain, do not fabricate an address pair. Prefer a natural neutral construction that preserves the source ambiguity.
            - Preserve distinctions such as whether a group includes the listener whenever the source makes them clear.
            - Source fidelity is more important than forcing a preferred Vietnamese pronoun pair. Rephrase naturally without changing the participants or intent.
        "}.trim_end());
    }

    if request.page_images.len() > 1 {
        prompt.push_str("\n\n");
        prompt.push_str(&format!(
            indoc! {"
                Image requirements:
                {count} consecutive page images are attached in the same order as the supplied page indices. Each input segment has a `page` number. Read the pages together as one continuous scene only when their sequence supports that interpretation.
                Red outlines and numeric tags, when present, associate a segment ID with its text region. Use them only to locate the matching supplied text region, not as story text.
                Use images as visual context for dialogue ownership, panel sequence, expression, action, mood, and OCR ambiguity. Visual interpretation must not override clear source wording without strong evidence of OCR error.
                Translate only the supplied segments. Do not add or omit text merely because other text appears in the images.
            "},
            count = request.page_images.len(),
        ).trim_end());
    } else if request.image.is_some() || request.page_images.len() == 1 {
        prompt.push_str("\n\n");
        prompt.push_str(indoc! {"
            Image requirements:
            Use the attached original page as context for panel sequence, dialogue ownership, tails, expressions, actions, tone, and genuinely ambiguous OCR. If red outlines and numeric segment tags are present, use them only to locate the matching supplied segment.
            Do not assume a fixed reading direction, and do not add text visible in the image that is absent from the supplied segments. The supplied OCR segments remain the scope of translation.
        "}.trim_end());
    }
    prompt
}

/// The part of the system prompt that does not change from page to page. The model reads it
/// once and reuses what it computed for the next pages (`Input::with_shared_prefix`).
pub(crate) fn stable_system(request: &TranslationRequest) -> String {
    translation_rules(request)
}

fn translation_system_prompt(request: &TranslationRequest) -> String {
    let mut prompt = translation_rules(request);

    if !request.context.is_empty() {
        prompt.push_str("\n\n");
        prompt.push_str(indoc! {"
            Context requirements:
            Use the supplied context to follow the story and to preserve terminology, character voice, pronouns, and dialogue continuity.
            Do not translate or return the context entries.
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
    }

    #[test]
    fn snippet_escapes_other_control_characters() {
        assert_eq!(snippet("before\u{7}after"), r"before\u{0007}after");
    }

    #[test]
    fn snippet_reports_the_length_it_elided() {
        let flattened = snippet(&"x".repeat(2_500));
        assert!(flattened.starts_with(&"x".repeat(2_000)));
        assert!(flattened.ends_with("… (2500 bytes total)"));
    }

    #[test]
    fn parses_plain_translation_json() {
        let source = ["one".to_owned(), "two".to_owned()];
        let response = r#"{"translations":[{"id":0,"text":"hello"},{"id":1,"text":"world"}]}"#;
        assert_eq!(translations("test", response, &source).unwrap(), ["hello", "world"]);
    }

    #[test]
    fn rejects_wrapped_and_malformed_json() {
        let source = ["one".to_owned(), "two".to_owned()];
        for response in [
            r#"```json
{"translations":[{"id":0,"text":"hello"},{"id":1,"text":"world"}]}
```"#,
            r#"{translations: [{id: 0, text: 'hello'}, {id: 1, text: 'world'},],}"#,
            r#"Here is the result: {"translations": [{"id": 0, "text": "hello"}, {"id": 1, "text": "world"},]}"#,
            r#"{"translations":[{"id":0,"text":"hello"},{"id":1,"text":"world"#,
        ] {
            assert!(translations("test", response, &source).is_err(), "{response}");
        }
    }

    #[test]
    fn parse_error_preserves_response() {
        let source = ["one".to_owned()];
        let response = r#"{
  "translations": [{"id": 0}]
}"#;
        let error = translations("test", response, &source).expect_err("missing text should fail");
        let invalid = error.downcast_ref::<InvalidTranslationOutput>().expect("typed context");
        assert_eq!(invalid.response, response);
    }

    #[test]
    fn parses_glossary_even_when_response_is_truncated() {
        let entry = |kind: &str, source: &str, target: &str| GlossaryEntry {
            kind: kind.to_owned(),
            source: source.to_owned(),
            target: target.to_owned(),
        };
        let full = r#"{"entries":[{"kind":"name","source":"CHAR_A","target":"Character A"}]}"#;
        assert_eq!(parse_glossary(full), [entry("name", "CHAR_A", "Character A")]);
        let cut = r#"{"entries":[{"kind":"name","source":"CHAR_A","target":"Character A"},{"kind":"term","source":"TERM_X","target":"term"#;
        assert_eq!(parse_glossary(cut), [entry("name", "CHAR_A", "Character A")]);
        assert!(parse_glossary("garbage").is_empty());
    }

    #[test]
    fn glossary_is_rendered_one_entry_per_line() {
        let entries = vec![
            GlossaryEntry {
                kind: "name".to_owned(),
                source: "CHAR_A".to_owned(),
                target: "Character A".to_owned(),
            },
            GlossaryEntry {
                kind: "term".to_owned(),
                source: "TERM_X".to_owned(),
                target: "term X".to_owned(),
            },
        ];
        let block = render_glossary(&entries);
        assert!(block.starts_with("Work glossary"));
        assert!(block.contains("\n- name: CHAR_A = Character A"));
        assert!(block.contains("\n- term: TERM_X = term X"));
    }

    #[test]
    fn vietnamese_prompt_has_contextual_address_guidance_only_for_vietnamese() {
        let vi = translation_system_prompt(&TranslationRequest::new(["hello"], Language::Vietnamese));
        assert!(vi.contains("Vietnamese dialogue and address consistency"));
        assert!(vi.contains("Do not mechanically map a pronoun"));
        let en = translation_system_prompt(&TranslationRequest::new(["hello"], Language::English));
        assert!(!en.contains("Vietnamese dialogue and address consistency"));
    }

    #[test]
    fn output_schema_requires_only_id_and_translation_text() {
        let schema = output_schema(2);
        let item = &schema["properties"]["translations"]["items"];
        assert_eq!(item["required"], serde_json::json!(["id", "text"]));
        let properties = item["properties"].as_object().unwrap();
        assert_eq!(properties.len(), 2);
        assert!(properties.contains_key("id"));
        assert!(properties.contains_key("text"));
    }

    #[test]
    fn parse_translations_reports_missing_and_duplicate_ids() {
        let sources = vec!["a".to_owned(), "b".to_owned(), "c".to_owned()];
        let text = r#"{"translations":[{"id":0,"text":"A"},{"id":0,"text":"duplicate"},{"id":2,"text":"C"}]}"#;
        let (translations, missing) = parse_translations("test", text, &sources).unwrap();
        assert_eq!(translations, ["A", "b", "C"]);
        assert_eq!(missing, [1]);
    }

    fn two_pages(language: Language) -> TranslationRequest {
        let blank = || std::sync::Arc::new(image::DynamicImage::new_rgb8(1, 1));
        TranslationRequest::new(["a", "b", "c"], language)
            .with_pages(vec![blank(), blank()], vec![0, 0, 1])
    }

    #[test]
    fn pages_translated_together_carry_page_numbers_and_visual_instructions() {
        let request = two_pages(Language::Vietnamese);
        let (system, user) = prompts(&request).unwrap();
        assert!(system.contains("2 consecutive page images"));
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
    fn glossary_requests_only_general_name_term_and_evidenced_style_entries() {
        let (system, _) = glossary_prompts(Language::Vietnamese, &[], &["line".to_owned()]).unwrap();
        assert!(system.contains("- name:"));
        assert!(system.contains("- term:"));
        assert!(system.contains("- style:"));
        assert!(system.contains("Do not assume any particular country"));
        assert!(!system.contains("Speaker -> Listener"));
    }

    #[test]
    fn the_stable_instructions_come_first_in_the_system_prompt() {
        let mut request = two_pages(Language::Vietnamese);
        request.context = vec![TranslationContext {
            source: "a".to_owned(),
            translation: "b".to_owned(),
        }];
        request.instructions = Some("Work glossary (x)".to_owned());
        let stable = stable_system(&request);
        let system = translation_system_prompt(&request);
        assert!(system.starts_with(&stable));
        assert!(system.len() > stable.len());
        assert!(!stable.contains("Context requirements"));
        assert!(!stable.contains("Additional instructions"));
    }

    #[test]
    fn salvage_keeps_complete_segments_from_a_truncated_response() {
        let sources = vec!["a".to_owned(), "b".to_owned(), "c".to_owned()];
        let text = r#"{"translations":[{"id":0,"text":"Xin chào"},{"id":1,"text":"cut"#;
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
}
