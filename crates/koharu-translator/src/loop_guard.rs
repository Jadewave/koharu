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
