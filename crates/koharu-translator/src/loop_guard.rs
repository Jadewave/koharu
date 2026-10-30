//! Cheap detection of runaway repetition in a translated segment.
//!
//! Models with reasoning enabled sometimes fall into a loop and keep repeating the same
//! phrase until the token limit. A real translation of a speech bubble is short, so two
//! signals are enough: the output is far longer than the source could justify, or a block
//! of text repeats back to back many times.

/// A block must repeat back to back at least this many times to count as a loop.
const MIN_REPEATS: usize = 4;
/// ...and the repeating region must cover at least this many characters.
const MIN_SPAN: usize = 32;
/// Longest repeating block (in characters) that is searched for.
const MAX_PERIOD: usize = 80;
/// A translation longer than `LENGTH_SLACK + LENGTH_RATIO * source` characters is a loop.
const LENGTH_SLACK: usize = 80;
const LENGTH_RATIO: usize = 10;

/// How much of the end of a partial generation is inspected while it streams.
const TAIL_CHARS: usize = 400;
/// A single character repeated this many times in a row at the end of a partial generation
/// is a loop (`”””””…`); shorter runs (ellipses, screams) are legitimate.
const MAX_SINGLE_RUN: usize = 48;

/// True when a segment contains something worth translating. Segments made only of
/// punctuation, quotation marks or ellipses are returned unchanged: models tend to fall
/// into a repetition loop on them.
pub(crate) fn needs_translation(source: &str) -> bool {
    source.chars().any(char::is_alphanumeric)
}

/// True when the end of a partial generation has fallen into a repetition loop, so the
/// generation can be stopped instead of running to the token limit.
pub(crate) fn tail_is_looping(text: &str) -> bool {
    let skip = text.chars().count().saturating_sub(TAIL_CHARS);
    let tail: String = text.chars().skip(skip).collect();
    longest_run(&tail) >= MAX_SINGLE_RUN || has_runaway_repeat(&collapse_runs(&tail))
}

/// Length of the longest run of one identical character.
fn longest_run(text: &str) -> usize {
    let mut longest = 0usize;
    let mut run = 0usize;
    let mut previous = None;
    for character in text.chars() {
        if previous == Some(character) {
            run += 1;
        } else {
            run = 1;
            previous = Some(character);
        }
        longest = longest.max(run);
    }
    longest
}

/// Indices of segments that came back unchanged although their source contains non-ASCII
/// letters (kana, kanji, hangul...): the model skipped them and the source text was kept.
pub(crate) fn untranslated_segments(sources: &[String], translations: &[String]) -> Vec<usize> {
    sources
        .iter()
        .zip(translations)
        .enumerate()
        .filter_map(|(index, (source, translation))| {
            let has_foreign_letters = source
                .chars()
                .any(|character| character.is_alphabetic() && !character.is_ascii());
            (has_foreign_letters && translation.trim() == source.trim()).then_some(index)
        })
        .collect()
}

/// Indices of segments that look like romaji: the source is short kana (a stammer, a voiced
/// sound), yet the translation is plain ASCII with a doubled or tripled consonant such as
/// `Nan...nnn`, which Vietnamese words never have. Only meaningful for Vietnamese output.
pub(crate) fn romanized_segments(sources: &[String], translations: &[String]) -> Vec<usize> {
    sources
        .iter()
        .zip(translations)
        .enumerate()
        .filter_map(|(index, (source, translation))| {
            is_romanized(source, translation).then_some(index)
        })
        .collect()
}

fn is_kana(character: char) -> bool {
    matches!(character, '\u{3041}'..='\u{3096}' | '\u{30A1}'..='\u{30FA}')
}

fn is_ideograph(character: char) -> bool {
    matches!(character, '\u{4E00}'..='\u{9FFF}' | '\u{3400}'..='\u{4DBF}')
}

fn is_romanized(source: &str, translation: &str) -> bool {
    let has_kana = source.chars().any(is_kana);
    let has_ideograph = source.chars().any(is_ideograph);
    if !has_kana || has_ideograph {
        return false;
    }
    let letters: Vec<char> = translation
        .chars()
        .filter(|character| character.is_alphabetic())
        .collect();
    if letters.is_empty() || !letters.iter().all(char::is_ascii_alphabetic) {
        return false;
    }
    let lowered: Vec<char> = letters.iter().map(char::to_ascii_lowercase).collect();
    lowered
        .windows(2)
        .any(|pair| pair[0] == pair[1] && !"aeiouy".contains(pair[0]))
}

/// Indices of segments whose translation looks like a repetition loop.
pub(crate) fn looping_segments(sources: &[String], translations: &[String]) -> Vec<usize> {
    sources
        .iter()
        .zip(translations)
        .enumerate()
        .filter_map(|(index, (source, translation))| {
            is_looping(source, translation).then_some(index)
        })
        .collect()
}

pub(crate) fn is_looping(source: &str, translation: &str) -> bool {
    let source = collapse_runs(source);
    let translation_raw = translation.chars().count();
    if translation_raw > LENGTH_SLACK + LENGTH_RATIO * source.len() {
        return true;
    }
    // A source that already repeats itself (laughter, stuttering) may legitimately
    // translate into something repetitive.
    !has_runaway_repeat(&source) && has_runaway_repeat(&collapse_runs(translation))
}

/// Characters with any run of an identical character shortened to two, so screams and
/// ellipses ("Aaaaaaa…", "......") never count as a repeating block.
fn collapse_runs(text: &str) -> Vec<char> {
    let mut collapsed = Vec::with_capacity(text.len());
    let mut run = 0usize;
    let mut previous = None;
    for character in text.chars() {
        if previous == Some(character) {
            run += 1;
        } else {
            run = 1;
            previous = Some(character);
        }
        if run <= 2 {
            collapsed.push(character);
        }
    }
    collapsed
}

/// True when some block of `2..=MAX_PERIOD` characters repeats back to back at least
/// `MIN_REPEATS` times and the repeating region spans at least `MIN_SPAN` characters.
fn has_runaway_repeat(text: &[char]) -> bool {
    let length = text.len();
    if length < MIN_SPAN {
        return false;
    }
    for period in 2..=MAX_PERIOD.min(length / MIN_REPEATS) {
        let mut run = 0usize;
        for index in 0..length - period {
            if text[index] == text[index + period] {
                run += 1;
                if run >= period * (MIN_REPEATS - 1) && run + period >= MIN_SPAN {
                    return true;
                }
            } else {
                run = 0;
            }
        }
    }
    false
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ordinary_translations_are_not_loops() {
        assert!(!is_looping("おはよう", "Chào buổi sáng!"));
        let source = "あなたは誰ですか？どうしてここにいるの？";
        let translation = "Bạn là ai vậy? Tại sao bạn lại ở đây?";
        assert!(!is_looping(source, translation));
        assert!(!is_looping("…", "…"));
        assert!(!is_looping("", ""));
    }

    #[test]
    fn short_emphasis_and_laughter_are_not_loops() {
        let scream = format!("A{}", "a".repeat(40));
        assert!(!is_looping("ああああああ", &scream));
        assert!(!is_looping("はははは", "Ha ha ha ha ha ha"));
        assert!(!is_looping("……", &".".repeat(43)));
    }

    #[test]
    fn a_phrase_repeated_back_to_back_is_a_loop() {
        let looping = "Tôi không biết. ".repeat(6);
        assert!(is_looping("わからない", &looping));
        let cycle = "Anh ấy nói rằng anh ấy sẽ đến, nhưng anh ấy không đến. ".repeat(4);
        assert!(is_looping("彼は来ると言った", &cycle));
    }

    #[test]
    fn a_translation_far_longer_than_its_source_is_a_loop() {
        let rambling: String = (0..60).map(|i| format!("từ{i} ")).collect();
        assert!(is_looping("はい", &rambling));
    }

    #[test]
    fn a_repeating_source_may_translate_into_a_repeating_text() {
        let source = "ハハハハ、ごめん。".repeat(5);
        let translation = "Ha ha, xin lỗi. ".repeat(5);
        assert!(!is_looping(&source, &translation));
    }

    #[test]
    fn a_looping_tail_is_detected_while_streaming() {
        let prefix = r#"{"translations": [{"id": 0, "text": ""#;
        let looping = format!("{prefix}{}", r#"\"\" "#.repeat(30));
        assert!(tail_is_looping(&looping));
        let phrase = format!("{prefix}{}", "Tôi không biết. ".repeat(8));
        assert!(tail_is_looping(&phrase));
    }

    #[test]
    fn a_single_character_flood_is_a_looping_tail() {
        let prefix = r#"{"translations": [{"id": 1, "text": ""#;
        assert!(tail_is_looping(&format!("{prefix}{}", "”".repeat(60))));
        assert!(!tail_is_looping(&format!("{prefix}{}", "…".repeat(12))));
    }

    #[test]
    fn only_segments_with_letters_or_digits_need_translation() {
        assert!(needs_translation("おはよう"));
        assert!(needs_translation("「やめろ！」"));
        assert!(needs_translation("3"));
        assert!(!needs_translation("…"));
        assert!(!needs_translation("“"));
        assert!(!needs_translation("「」"));
        assert!(!needs_translation("！？"));
        assert!(!needs_translation(" "));
    }

    #[test]
    fn ordinary_json_output_is_not_a_looping_tail() {
        let mut text = String::from(r#"{"translations": ["#);
        for id in 0..12 {
            text.push_str(&format!(r#"{{"id": {id}, "text": "Ừ."}}, "#));
        }
        assert!(!tail_is_looping(&text));
        assert!(!tail_is_looping(""));
    }

    #[test]
    fn romaji_for_a_kana_sound_is_flagged() {
        let sources = [
            "なんっんん".to_owned(),
            "なんっんん".to_owned(),
            "ゆうくん".to_owned(),
            "彼女".to_owned(),
        ];
        let translations = [
            "Nan...nnn".to_owned(),
            "Ưm... hự...".to_owned(),
            "Yuu-kun".to_owned(),
            "Kanna".to_owned(),
        ];
        assert_eq!(romanized_segments(&sources, &translations), [0]);
    }

    #[test]
    fn unchanged_foreign_text_counts_as_untranslated() {
        let sources = vec![
            "おはよう".to_owned(),
            "OK".to_owned(),
            "…".to_owned(),
            "ありがとう".to_owned(),
        ];
        let translations = vec![
            "おはよう".to_owned(),
            "OK".to_owned(),
            "…".to_owned(),
            "Cảm ơn".to_owned(),
        ];
        assert_eq!(untranslated_segments(&sources, &translations), [0]);
    }

    #[test]
    fn reports_only_the_looping_indices() {
        let sources = vec!["a".to_owned(), "b".to_owned(), "c".to_owned()];
        let translations = vec![
            "Xin chào".to_owned(),
            "Tôi không biết. ".repeat(6),
            "Tạm biệt".to_owned(),
        ];
        assert_eq!(looping_segments(&sources, &translations), [1]);
    }
}
