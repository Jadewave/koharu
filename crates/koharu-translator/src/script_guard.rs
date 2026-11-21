//! Keeps Japanese, Chinese and Korean characters out of text written for a language that does
//! not use them.
//!
//! The vision model that reads a Japanese page tends to copy what it sees into the words it
//! chooses for "I" and "you" (`yuu君`, `rioちゃん`) and into its drafts. The translation model
//! is told to use those words exactly, so they end up in the final text, and a balloon whose
//! text holds kana is laid out by the renderer as vertical Japanese.
//!
//! An honorific written right after a name in Latin letters becomes the form a reader of the
//! target language knows (`-kun`, `-chan`). Anything else of those scripts is not a word of the
//! target language: it is dropped from the words of address, and reported for the text of a
//! translation so that the line can be translated again.

/// Japanese honorifics and what they become after a name in Latin letters. A longer spelling
/// comes before the shorter one it contains.
const HONORIFICS: &[(&str, &str)] = &[
    ("ちゃん", "-chan"),
    ("チャン", "-chan"),
    ("くん", "-kun"),
    ("クン", "-kun"),
    ("君", "-kun"),
    ("さん", "-san"),
    ("サン", "-san"),
    ("さま", "-sama"),
    ("サマ", "-sama"),
    ("様", "-sama"),
    ("せんぱい", "-senpai"),
    ("センパイ", "-senpai"),
    ("先輩", "-senpai"),
    ("せんせい", "-sensei"),
    ("センセイ", "-sensei"),
    ("先生", "-sensei"),
    ("殿", "-dono"),
    ("たん", "-tan"),
];

/// Whether the character belongs to Han, kana, hangul or bopomofo. The punctuation of those
/// scripts (U+3000 to U+303F) is not counted: it does not make a text vertical or foreign.
pub(crate) fn is_cjk_char(character: char) -> bool {
    matches!(
        u32::from(character),
        0x1100..=0x11FF       // hangul jamo
        | 0x2E80..=0x2FDF     // CJK radicals
        | 0x3005 | 0x3007     // 々 〇
        | 0x3040..=0x30FF     // hiragana, katakana
        | 0x3100..=0x312F     // bopomofo
        | 0x3130..=0x318F     // hangul compatibility jamo
        | 0x31A0..=0x31BF     // bopomofo extended
        | 0x31F0..=0x31FF     // katakana phonetic extensions
        | 0x3400..=0x4DBF     // CJK extension A
        | 0x4E00..=0x9FFF     // CJK unified ideographs
        | 0xA960..=0xA97F     // hangul jamo extended A
        | 0xAC00..=0xD7FF     // hangul syllables, jamo extended B
        | 0xF900..=0xFAFF     // CJK compatibility ideographs
        | 0xFF66..=0xFF9F     // halfwidth katakana
        | 0x20000..=0x2FA1F   // CJK extensions B to F, compatibility supplement
    )
}

/// Whether the text holds a character of Han, kana, hangul or bopomofo.
pub(crate) fn has_cjk(text: &str) -> bool {
    text.chars().any(is_cjk_char)
}

fn is_latin_letter(character: char) -> bool {
    character.is_alphabetic() && !is_cjk_char(character)
}

/// Whether the text written so far ends with a name in Latin letters (a hyphen between the name
/// and the honorific, as in `Yuu-君`, is let through).
fn ends_with_name(written: &str) -> bool {
    let mut characters = written.chars().rev();
    match characters.next() {
        Some('-') => characters.next().is_some_and(is_latin_letter),
        Some(last) => is_latin_letter(last),
        None => false,
    }
}

/// Writes the Japanese honorifics that follow a name in Latin letters the way the target
/// language does: `Yuu君` becomes `Yuu-kun`, `rioちゃん` becomes `rio-chan`. A honorific after
/// anything else is part of a Japanese word and stays.
pub(crate) fn latinize_honorifics(text: &str) -> String {
    if !has_cjk(text) {
        return text.to_owned();
    }
    let mut written = String::with_capacity(text.len() + 8);
    let mut rest = text;
    while let Some(character) = rest.chars().next() {
        if ends_with_name(&written) {
            if let Some(&(spelling, latin)) =
                HONORIFICS.iter().find(|(spelling, _)| rest.starts_with(spelling))
            {
                // `Yuu-君` already has its hyphen.
                let latin = if written.ends_with('-') { &latin[1..] } else { latin };
                written.push_str(latin);
                rest = &rest[spelling.len()..];
                continue;
            }
        }
        written.push(character);
        rest = &rest[character.len_utf8()..];
    }
    written
}

/// `latinize_honorifics` for every line of a translation.
pub(crate) fn latinize_all(lines: Vec<String>) -> Vec<String> {
    lines
        .into_iter()
        .map(|line| if has_cjk(&line) { latinize_honorifics(&line) } else { line })
        .collect()
}

/// A word the speaker uses for "I" or for "you", made fit for a language that does not use
/// Han, kana or hangul: its honorifics are latinized, and a word that still holds such a
/// character is not a word of the language, so nothing is returned for it (the line then has
/// no pair of its own and the translation chooses the words).
pub(crate) fn address_word(word: &str) -> String {
    let word = latinize_honorifics(word.trim());
    if has_cjk(&word) { String::new() } else { word }
}

/// Whether a translation has left a little of Han, kana or hangul in a line that is written in
/// another script: the text of a name or an honorific that was copied, not a line that was not
/// translated at all. A line that is the source itself, or mostly of those scripts, is a sound
/// effect or a sign the model chose to leave, and translating it again would not change it.
pub(crate) fn leaks_foreign_script(source: &str, translated: &str) -> bool {
    let (foreign, letters) = translated
        .chars()
        .filter(|character| character.is_alphanumeric())
        .fold((0usize, 0usize), |(foreign, letters), character| {
            (foreign + usize::from(is_cjk_char(character)), letters + 1)
        });
    foreign > 0 && foreign * 2 <= letters && translated.trim() != source.trim()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn honorifics_after_a_latin_name_are_latinized() {
        assert_eq!(latinize_honorifics("riocちゃん"), "rioc-chan");
        assert_eq!(latinize_honorifics("yuu君"), "yuu-kun");
        assert_eq!(latinize_honorifics("sasakiくん"), "sasaki-kun");
        assert_eq!(latinize_honorifics("rioさん"), "rio-san");
        assert_eq!(latinize_honorifics("Sai rồi!! riocちゃん chỉ cần nghe lời khách thôi!!"),
            "Sai rồi!! rioc-chan chỉ cần nghe lời khách thôi!!");
        assert_eq!(latinize_honorifics("Yuu君…!! Rioちゃん, Việt君"), "Yuu-kun…!! Rio-chan, Việt-kun");
        assert_eq!(latinize_honorifics("Yuu-君"), "Yuu-kun");
    }

    #[test]
    fn plain_text_and_japanese_words_are_left_alone() {
        assert_eq!(latinize_honorifics("Rio-chan à, em hiểu rồi"), "Rio-chan à, em hiểu rồi");
        // An honorific that is part of a Japanese word, or after a digit, is not a name's.
        assert_eq!(latinize_honorifics("旦那さん"), "旦那さん");
        assert_eq!(latinize_honorifics("2さん"), "2さん");
        assert_eq!(latinize_honorifics(""), "");
    }

    #[test]
    fn a_word_of_address_keeps_no_foreign_script() {
        assert_eq!(address_word(" yuu君 "), "yuu-kun");
        assert_eq!(address_word("tôi"), "tôi");
        assert_eq!(address_word("お前"), "");
        assert_eq!(address_word("anh"), "anh");
    }

    #[test]
    fn only_a_leak_is_reported() {
        assert!(leaks_foreign_script("りおちゃん", "Em là Rioちゃん của anh"));
        assert!(!leaks_foreign_script("りおちゃん", "Em là Rio-chan của anh"));
        // The line is the source, or mostly of the foreign script: nothing to translate again.
        assert!(!leaks_foreign_script("ビク", "ビク"));
        assert!(!leaks_foreign_script("ビクビク", "ビクビクッ"));
        assert!(!leaks_foreign_script("ゆう", "♡"));
    }

    #[test]
    fn punctuation_of_the_scripts_is_not_foreign() {
        assert!(!has_cjk("「Xin chào」。、"));
        assert!(has_cjk("a々"));
        assert!(has_cjk("안녕"));
    }
}
