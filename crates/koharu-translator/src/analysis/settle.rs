//! The pairs of words a work settles on, without a model.
//!
//! The pages are read one by one, so what the model chose on page 3 can clash with what it
//! chose on page 40: a pair flips between "anh - em" and "tôi - cô". What this module does:
//!
//! - every direction (speaker to listener) gets the pair most of its lines used, over what its
//!   first line said (`settle_pairs`), so a stray first line does not decide the direction;
//! - the pairs all lines of a direction agree on are locked for the translation (`seed_pairs`);
//! - the pairs that cannot be right (a word of one sex for a person of the other, two
//!   directions of one pair that do not fit) are named (`address_conflicts`): the review asks
//!   the editor about them, and without a review they are logged.
//!
//! Writing the settled words into the notes of the pages is the job of `review`.

use super::*;
use crate::address::PairBook;

/// Locks, in the pair book the translation is held to, the pair of every direction whose
/// lines all use the same words. A direction with exceptions is left out: the lines that differ
/// would be translated again with the pair of the others. Returns how many were locked.
pub(super) fn seed_pairs(book: &mut PairBook, bible: &Bible, results: &[Option<PageAnalysis>]) -> usize {
    let name = |name: &str| {
        bible
            .character_index(name)
            .map_or_else(|| name.trim().to_owned(), |index| bible.characters[index].name.clone())
    };
    let mut directions: Vec<((String, String), Vec<(String, String)>)> = Vec::new();
    for page in results.iter().flatten() {
        for line in &page.lines {
            if is_generic(&line.speaker)
                || is_generic(&line.listener)
                || line.self_word.trim().is_empty()
                || line.you.trim().is_empty()
            {
                continue;
            }
            let key = (name(&line.speaker), name(&line.listener));
            let pair = (line.self_word.trim().to_lowercase(), line.you.trim().to_lowercase());
            match directions.iter_mut().find(|(known, _)| *known == key) {
                Some((_, pairs)) => {
                    if !pairs.contains(&pair) {
                        pairs.push(pair);
                    }
                }
                None => directions.push((key, vec![pair])),
            }
        }
    }
    let mut seeded = 0;
    for ((from, to), pairs) in directions {
        if pairs.len() == 1 && book.seed(&from, &to, &pairs[0].0, &pairs[0].1) {
            seeded += 1;
        }
    }
    seeded
}

impl Bible {
    /// The words of one direction: what `speaker` calls themselves and `listener`, when the
    /// work has settled it. Names are matched through the aliases.
    pub(super) fn pair(&self, speaker: &str, listener: &str) -> Option<&Relationship> {
        let character = &self.characters[self.character_index(speaker)?];
        let target = self.character_index(listener);
        character.relationships.iter().find(|relationship| {
            same_name(&relationship.with, listener)
                || (target.is_some() && self.character_index(&relationship.with) == target)
        })
    }

    /// The name a character goes by in the notes: the name of the character the given name
    /// or alias belongs to; other names (narrator, self, ...) stay as they are.
    pub(super) fn canonical(&self, name: &str) -> String {
        if is_generic(name) {
            return name.trim().to_owned();
        }
        self.character_index(name)
            .map_or_else(|| name.trim().to_owned(), |index| self.characters[index].name.clone())
    }
}

/// Words that name a man or a boy, as "I" or as "you".
const MALE_ONLY: [&str; 8] = ["anh", "chú", "ông", "lão", "bác trai", "bố", "cha", "ba"];
/// Words that name a woman or a girl, as "I" or as "you".
const FEMALE_ONLY: [&str; 9] = ["chị", "cô", "dì", "bà", "thím", "mợ", "bác gái", "mẹ", "má"];

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Sex {
    Male,
    Female,
}

/// The sex a gender note says ("man", "young woman", "boy", ...); `None` when it says none
/// or both.
fn sex_of(gender: &str) -> Option<Sex> {
    let lower = gender.to_lowercase();
    let mut male = false;
    let mut female = false;
    for word in lower.split(|c: char| !c.is_alphabetic()) {
        male |= matches!(word, "man" | "male" | "boy");
        female |= matches!(word, "woman" | "female" | "girl" | "lady");
    }
    match (male, female) {
        (true, false) => Some(Sex::Male),
        (false, true) => Some(Sex::Female),
        _ => None,
    }
}

/// Whether `word` (lowercase) can only be said about the other sex than `sex`.
fn names_other_sex(word: &str, sex: Sex) -> bool {
    let word = word.trim().to_lowercase();
    match sex {
        Sex::Male => FEMALE_ONLY.contains(&word.as_str()),
        Sex::Female => MALE_ONLY.contains(&word.as_str()),
    }
}

impl Bible {
    /// Gives every direction the pair most of its lines used (the first one on a tie), over
    /// what the first line of the direction said. Returns how many directions changed.
    pub(super) fn settle_pairs(&mut self) -> usize {
        // Names learned later turn labels into names: count under the name they have now.
        let mut merged = Tally::new();
        for ((from, to), pairs) in std::mem::take(&mut self.tally) {
            let key = (self.direction_name(&from), self.direction_name(&to));
            for (pair, votes) in pairs {
                add_votes(&mut merged, key.clone(), pair, votes);
            }
        }
        let mut settled = 0;
        for ((from, to), pairs) in &merged {
            if pairs.len() < 2 {
                continue;
            }
            let mut best = &pairs[0];
            for candidate in &pairs[1..] {
                if candidate.1 > best.1 {
                    best = candidate;
                }
            }
            let Some(index) = self.character_index(from) else {
                continue;
            };
            let target = self.character_index(to);
            let position = self.characters[index].relationships.iter().position(|relationship| {
                same_name(&relationship.with, to)
                    || (target.is_some() && self.character_index(&relationship.with) == target)
            });
            let Some(at) = position else {
                continue;
            };
            let slot = &mut self.characters[index].relationships[at];
            let (mine, yours) = (clip(&best.0.0, 40), clip(&best.0.1, 40));
            if slot.self_word.trim().to_lowercase() != mine || slot.you.trim().to_lowercase() != yours {
                slot.self_word = mine;
                slot.you = yours;
                settled += 1;
            }
        }
        self.tally = merged;
        settled
    }

    /// The settled pairs that cannot be right, each described in a sentence: a word of one sex
    /// for a person of the other, and the two directions of a pair of people that do not fit
    /// each other ("anh - em" one way, "tôi - cô" the other).
    pub(super) fn address_conflicts(&self) -> Vec<String> {
        let mut found: Vec<String> = Vec::new();
        let mut seen: Vec<(String, String)> = Vec::new();
        for character in &self.characters {
            let sex = sex_of(&character.gender);
            for relationship in &character.relationships {
                let (mine, yours) = (
                    relationship.self_word.trim().to_lowercase(),
                    relationship.you.trim().to_lowercase(),
                );
                if mine.is_empty() || yours.is_empty() {
                    continue;
                }
                let listener = self.character_index(&relationship.with).map(|at| &self.characters[at]);
                let pair = format!("{} -> {}: {mine} - {yours}", character.name, relationship.with);
                if let Some(sex) = sex
                    && names_other_sex(&mine, sex)
                {
                    found.push(format!(
                        "{pair}: {} ({}) calls themselves \"{mine}\", a word of the other sex",
                        character.name,
                        character.gender.trim()
                    ));
                }
                let Some(listener) = listener else {
                    continue;
                };
                if let Some(sex) = sex_of(&listener.gender)
                    && names_other_sex(&yours, sex)
                {
                    found.push(format!(
                        "{pair}: {} ({}) is called \"{yours}\", a word of the other sex",
                        listener.name,
                        listener.gender.trim()
                    ));
                }
                // Siblings and the like: the elder says "anh/chị - em", the younger "em - anh/chị".
                let elder = matches!(mine.as_str(), "anh" | "chị") && yours == "em";
                let younger = mine == "em" && matches!(yours.as_str(), "anh" | "chị");
                if !(elder || younger) {
                    continue;
                }
                let mut names = [character.name.to_lowercase(), listener.name.to_lowercase()];
                names.sort();
                let key = (names[0].clone(), names[1].clone());
                if seen.contains(&key) {
                    continue;
                }
                let Some(back) = self.pair(&listener.name, &character.name) else {
                    continue;
                };
                let (back_mine, back_yours) =
                    (back.self_word.trim().to_lowercase(), back.you.trim().to_lowercase());
                if back_mine.is_empty() || back_yours.is_empty() {
                    continue;
                }
                let (want_mine, want_yours) = if elder {
                    ("em".to_owned(), mine.clone())
                } else {
                    (yours.clone(), "em".to_owned())
                };
                seen.push(key);
                if (back_mine.as_str(), back_yours.as_str()) != (want_mine.as_str(), want_yours.as_str()) {
                    found.push(format!(
                        "{pair} but {} -> {}: {back_mine} - {back_yours}; the two directions do not fit \
each other (expected {want_mine} - {want_yours})",
                        listener.name, character.name
                    ));
                }
            }
        }
        found
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn page(json: &str, lines: usize, bible: &mut Bible) -> PageAnalysis {
        let output = parse_page(json, lines).unwrap();
        bible.merge_output(&output);
        let sources: Vec<String> = (0..lines).map(|id| format!("line {id}")).collect();
        PageAnalysis::new(&output, &sources, bible)
    }

    const FIRST: &str = r#"{
        "scene": "A hospital room.",
        "characters": [
            {"name": "Sasaki", "gender": "man", "age": "young adult",
             "relationships": [{"with": "Rio", "relation": "colleague", "self": "tôi", "you": "cô"}]},
            {"name": "Rio", "aliases": ["Rio-san"], "gender": "woman", "age": "adult"}
        ],
        "lines": [
            {"id": 0, "kind": "speech", "speaker": "Sasaki", "listener": "Rio", "self": "tôi", "you": "cô"},
            {"id": 1, "kind": "speech", "speaker": "Sasaki", "listener": "Rio-san", "self": "tôi", "you": "cô"},
            {"id": 2, "kind": "thought", "speaker": "Rio", "listener": "self", "self": "mình"},
            {"id": 3, "kind": "narration", "speaker": "narrator", "listener": "everyone"}
        ]
    }"#;

    #[test]
    fn the_translation_is_held_to_the_pairs_all_lines_agree_on() {
        let mut bible = Bible::default();
        let mut analysis = page(FIRST, 4, &mut bible);
        analysis.apply_review(&bible, None);
        let mut book = PairBook::default();
        assert_eq!(seed_pairs(&mut book, &bible, &[Some(analysis.clone()), None]), 1);
        assert_eq!(book.entries(), [("Sasaki -> Rio".to_owned(), "tôi - cô".to_owned())]);
        // A direction whose lines disagree is not locked.
        analysis.lines[1].you = "em".to_owned();
        let mut other = PairBook::default();
        assert_eq!(seed_pairs(&mut other, &bible, &[Some(analysis)]), 0);
    }

    fn cast(json: &str) -> Bible {
        let mut bible = Bible::default();
        bible.merge_output(&serde_json::from_str::<PageOutput>(json).unwrap());
        bible
    }

    #[test]
    fn a_gender_is_read_from_its_words_not_from_their_letters() {
        assert_eq!(sex_of("man"), Some(Sex::Male));
        assert_eq!(sex_of("young man"), Some(Sex::Male));
        assert_eq!(sex_of("Boy"), Some(Sex::Male));
        // "woman" holds the letters "man"; it is a woman.
        assert_eq!(sex_of("woman"), Some(Sex::Female));
        assert_eq!(sex_of("girl, teenager"), Some(Sex::Female));
        assert_eq!(sex_of("unknown"), None);
        assert_eq!(sex_of(""), None);
        assert!(names_other_sex("chị", Sex::Male) && names_other_sex("Anh", Sex::Female));
        assert!(!names_other_sex("em", Sex::Male) && !names_other_sex("tôi", Sex::Female));
    }

    #[test]
    fn a_word_of_the_other_sex_is_a_problem_for_the_speaker_and_for_the_listener() {
        let bible = cast(
            r#"{"characters": [
                {"name": "Sasaki", "gender": "man", "relationships": [
                    {"with": "Rio", "self": "tôi", "you": "cô"}]},
                {"name": "Rio", "gender": "woman", "relationships": [
                    {"with": "Sasaki", "self": "anh", "you": "chị"}]}]}"#,
        );
        let problems = bible.address_conflicts();
        // Sasaki -> Rio is fine; Rio calls herself "anh" and Sasaki "chị".
        assert_eq!(problems.len(), 2, "{problems:?}");
        assert!(problems.iter().any(|p| p.contains("Rio -> Sasaki") && p.contains("calls themselves \"anh\"")));
        assert!(problems.iter().any(|p| p.contains("Sasaki (man) is called \"chị\"")));
    }

    #[test]
    fn the_two_directions_of_older_and_younger_have_to_fit() {
        let bible = cast(
            r#"{"characters": [
                {"name": "Yuu", "gender": "man", "relationships": [
                    {"with": "Rio", "self": "anh", "you": "em"}]},
                {"name": "Rio", "gender": "woman", "relationships": [
                    {"with": "Yuu", "self": "tôi", "you": "cô"}]}]}"#,
        );
        let problems = bible.address_conflicts();
        // "cô" for Yuu is also a word of the other sex; the pair as a whole does not fit either.
        assert!(problems.iter().any(|p| p.contains("do not fit each other") && p.contains("expected em - anh")), "{problems:?}");
        // The same pair seen from the other side is reported once.
        assert_eq!(problems.iter().filter(|p| p.contains("do not fit")).count(), 1);
    }

    #[test]
    fn fitting_pairs_and_neutral_words_raise_no_problem() {
        let bible = cast(
            r#"{"characters": [
                {"name": "Yuu", "gender": "man", "relationships": [
                    {"with": "Rio", "self": "anh", "you": "em"}]},
                {"name": "Rio", "gender": "woman", "relationships": [
                    {"with": "Yuu", "self": "em", "you": "anh"}]},
                {"name": "Mika", "gender": "girl", "relationships": [
                    {"with": "Rio", "self": "em", "you": "chị"},
                    {"with": "Yuu", "self": "tớ", "you": "cậu"}]},
                {"name": "Ken", "gender": "unknown", "relationships": [
                    {"with": "Rio", "self": "anh", "you": "cô"}]}]}"#,
        );
        assert_eq!(bible.address_conflicts(), Vec::<String>::new());
    }
}
