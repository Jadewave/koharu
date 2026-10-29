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
            .map(|(id, text)| TranslationInputSegment { id, text })
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
            Other, smaller models will translate every page and must follow your glossary exactly, so you decide the renderings.

            You receive the current glossary and the source lines of a stretch of the work (OCR text in reading order, page after page).
            Return only NEW entries:
            - name: a character, place, organization, or title seen in the lines and the {target} rendering to use for it.
            - term: a recurring term, skill, item, or sound-effect convention and the rendering to use for it.
            - character: `source` is the character's name as it appears in the lines, `target` a short profile in {target}: gender, rough age, role, how they speak (formal, casual, rough), and the first-person pronoun they use.
            - style: `source` is exactly work, `target` one short note on genre, setting, and overall register. At most one, and only if the glossary has none yet.
            - address: how one character addresses another. `source` is \"Speaker -> Listener\" with the names used in the lines, `target` is the pronoun pair to use for the speaker and for the listener (for Vietnamese, for example \"tôi - cậu\"), chosen from their relationship, age, rank, and tone.

            Rules:
            - Choose natural renderings that will still fit the rest of the work.
            - Be exhaustive: list every named character, place, organization, title, and recurring term in the lines, and every pair of speaking characters whose way of addressing each other you can tell.
            - Skip entries already in the glossary you were given and anything that does not appear in these lines.
            - Write `kind` as exactly style, address, character, name, or term.
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
            - Preserve meaning, character voice, emotional tone, relationship nuance, emphasis, and sound effects.
            - Localize idioms and sound effects naturally while keeping wording concise enough for speech bubbles.
            - Use surrounding segments only for disambiguation and continuity; never merge or split segments.
            - A sentence can run across consecutive segments (for example one line split between two speech bubbles); translate each segment so that, read in order, they form one natural sentence.
            - Write every translated `text` value only in {target}; do not include source text, notes, explanations, or alternatives.
            - Never preserve or repeat original-language text; translate names, terms, and sound effects using natural {target} conventions.

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
            - Choose the pronouns for each speaker and listener (tôi, tao, mình, tớ, cậu, anh, chị, em, ông, bà, ...) from their relationship, age, rank, and tone, and keep the same pair for the whole page.
            - Change a pair only when the source clearly shifts the relationship or tone (anger, intimacy, respect).
            - Never alternate between different first-person or second-person pronouns for the same speaker or the same listener, inside one segment or across segments.
            - Source text often drops the subject. Work out who speaks and who is addressed from the context and the image, and write the subject explicitly in Vietnamese when the sentence needs it, always with the pronoun already chosen for that character.
            - Render each character name, title, and the way others address them the same way every time.
            - When context entries are supplied, reuse the pronouns and names already used there.

            Vietnamese fidelity and naturalness:
            - First get the exact meaning of each sentence right; never invent content or guess a different meaning. Check polite forms and homographs (for example Japanese 要る 'need' versus 居る 'be', 御着用ください 'please wear').
            - Then say it the way a Vietnamese speaker would in that situation: short spoken sentences, fitting sentence-final particles (nhé, nhỉ, mà, đấy, thôi, ạ) for the tone and politeness, no word-for-word calques.
            - Sound effects become a short Vietnamese onomatopoeia or stay as they are; never romanize them.
            - Keep each line about as long as the source allows, because Vietnamese text must fit the same speech bubble.
        "}.trim_end());
    }

    if !request.context.is_empty() {
        prompt.push_str("\n\n");
        prompt.push_str(indoc! {"
            Context requirements:
            Use the supplied context only to preserve terminology, character voice, and dialogue continuity.
            Do not translate or return the context entries.
        "}.trim_end());
    }

    if request.image.is_some() {
        prompt.push_str("\n\n");
        prompt.push_str(indoc! {"
            Image requirements:
            Use the attached original page image as visual context for speaker identity, tone, layout, and ambiguous OCR.
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
                kind: "address".to_owned(),
                source: "田中 -> 鈴木".to_owned(),
                target: "tôi - cậu".to_owned(),
            },
        ];
        let block = render_glossary(&entries);
        assert!(block.starts_with("Glossary"));
        assert!(block.contains("\n- name: 田中 = Tanaka"));
        assert!(block.contains("\n- address: 田中 -> 鈴木 = tôi - cậu"));
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
}
