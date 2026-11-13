//! The pronoun pair of every direction (who speaks to whom) of a work.
//!
//! While a page is translated the model writes, before every line, `Speaker>Listener: I - you`
//! (for example `Sasaki>Rio: anh - em`). Those decisions used to be thrown away, so every page
//! chose again. Here they are kept, but never on one say-so: a pair is only locked after the
//! translations themselves have used it on several lines, and a locked pair goes to every later
//! page as a glossary `address` entry. Within a page, a line whose pair or self pronoun differs
//! from the one decided for the same direction is flagged so that it can be translated again.

use std::collections::{HashMap, HashSet};

use crate::ExpectedPair;

/// First person words that cannot be anything else: a line that uses one of them speaks about
/// its speaker, so a different pair for that speaker is a mistake.
const FIRST_PERSON: [&str; 4] = ["tôi", "tao", "tớ", "tui"];
/// Likewise for the second person.
const SECOND_PERSON: [&str; 1] = ["mày"];

/// Names that say nobody in particular.
const GENERIC: [&str; 12] = [
    "", "?", "unknown", "none", "n/a", "narration", "narrator", "thought", "lời kể",
    "suy nghĩ", "không rõ", "someone",
];
const HONORIFICS: [&str; 14] = [
    "-kun", "-san", "-chan", "-sama", "-sensei", "-senpai", "-dono", "くん", "さん", "ちゃん", "様",
    "先生", "君", "-tan",
];

/// A locked pair needs this many lines that show it, on this many pages at least, unless it
/// has the votes of `LOCK_VOTES_ANY_PAGES`.
const LOCK_VOTES: u32 = 2;
const LOCK_PAGES: usize = 2;
const LOCK_VOTES_ANY_PAGES: u32 = 3;

/// What the model wrote in `speaker` for one line.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct Declared {
    pub from: String,
    pub to: String,
    pub from_name: String,
    pub to_name: String,
    /// What the speaker calls himself.
    pub mine: String,
    /// What the speaker calls the listener.
    pub yours: String,
}

impl Declared {
    /// What the notes of a page decided for a line, written the way a translation declares it.
    /// `None` for names or words that make no pair.
    pub(crate) fn expected(expected: &ExpectedPair) -> Option<Self> {
        let (from, to) = (norm_name(&expected.speaker), norm_name(&expected.listener));
        if GENERIC.contains(&from.as_str()) || GENERIC.contains(&to.as_str()) || from == to {
            return None;
        }
        Some(Self {
            from,
            to,
            from_name: expected.speaker.trim().to_owned(),
            to_name: expected.listener.trim().to_owned(),
            mine: clean_word(&expected.mine)?,
            yours: clean_word(&expected.yours)?,
        })
    }

    fn direction(&self) -> (String, String) {
        (self.from.clone(), self.to.clone())
    }
    fn pair(&self) -> (String, String) {
        (self.mine.clone(), self.yours.clone())
    }
}

#[derive(Default)]
struct Vote {
    count: u32,
    pages: HashSet<u32>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct Locked {
    declared: Declared,
}

#[derive(Default)]
pub(crate) struct PairBook {
    page: u32,
    votes: HashMap<(String, String), HashMap<(String, String), Vote>>,
    names: HashMap<(String, String), (String, String)>,
    locked: Vec<Locked>,
}

pub(crate) fn norm_name(name: &str) -> String {
    let mut name = name.trim().to_lowercase();
    for cut in ['(', '（', ',', '，'] {
        if let Some(at) = name.find(cut) {
            name.truncate(at);
        }
    }
    let mut name = name
        .trim_matches(|c: char| c.is_whitespace() || "\"'“”‘’[]「」".contains(c))
        .to_owned();
    loop {
        let before = name.len();
        for honorific in HONORIFICS {
            if name.len() > honorific.len() && name.ends_with(honorific) {
                name.truncate(name.len() - honorific.len());
                name = name.trim().to_owned();
            }
        }
        if name.len() == before {
            break;
        }
    }
    name
}

fn clean_word(word: &str) -> Option<String> {
    let word = word
        .trim()
        .trim_matches(|c: char| !c.is_alphabetic())
        .to_lowercase();
    let words = word.split_whitespace().count();
    (!word.is_empty() && words <= 3 && !word.chars().any(|c| c.is_ascii_digit())).then_some(word)
}

/// Reads `Speaker>Listener: I - you`. Anything else (narration, thought, a name that is not a
/// name, a pair that is not two words) is not a declaration of a pair.
pub(crate) fn parse_declared(speaker: &str) -> Option<Declared> {
    let speaker = speaker.replace("->", ">").replace('→', ">").replace('＞', ">");
    let (people, words) = speaker.split_once([':', '：'])?;
    let (from_name, to_name) = people.split_once('>')?;
    let (from, to) = (norm_name(from_name), norm_name(to_name));
    if GENERIC.contains(&from.as_str()) || GENERIC.contains(&to.as_str()) || from == to {
        return None;
    }
    let words = words.replace(['–', '—', '‒'], "-");
    let (mine, yours) = match words.split_once(" - ") {
        Some(parts) => parts,
        None => {
            let mut parts = words.split('-');
            match (parts.next(), parts.next(), parts.next()) {
                (Some(a), Some(b), None) => (a, b),
                _ => return None,
            }
        }
    };
    let (mine, yours) = (clean_word(mine)?, clean_word(yours)?);
    if yours.contains(" - ") {
        return None;
    }
    Some(Declared {
        from,
        to,
        from_name: from_name.trim().to_owned(),
        to_name: to_name.trim().to_owned(),
        mine,
        yours,
    })
}

/// The kinds a speaker may give for a thought: `Rio>thought: mình`.
const THOUGHT_KINDS: [&str; 6] = [
    "thought", "monologue", "self", "suy nghĩ", "độc thoại", "nội tâm",
];

/// The plain first-person word of an inner voice. A thought may use it whatever the character
/// says aloud.
const INNER_SELF: &str = "mình";

/// What the model wrote in `speaker` for a thought or a monologue: whose it is and the
/// first-person word it uses.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct Thought {
    pub name: String,
    pub display: String,
    pub mine: String,
}

/// Reads `Name>thought: I-word`. Anything else is not a thought of a known character.
pub(crate) fn parse_thought(speaker: &str) -> Option<Thought> {
    let speaker = speaker.replace("->", ">").replace('→', ">").replace('＞', ">");
    let (people, words) = speaker.split_once([':', '：'])?;
    let (name, kind) = people.split_once('>')?;
    if !THOUGHT_KINDS.contains(&kind.trim().to_lowercase().as_str()) {
        return None;
    }
    let normalized = norm_name(name);
    if GENERIC.contains(&normalized.as_str()) {
        return None;
    }
    let words = words.replace(['–', '—', '‒'], "-");
    let mine = clean_word(words.split('-').next()?)?;
    Some(Thought {
        name: normalized,
        display: name.trim().to_owned(),
        mine,
    })
}

/// Whether `word` stands in `text` (already lowercase) as a word of its own.
fn has_word(text: &str, word: &str) -> bool {
    let mut from = 0;
    while let Some(found) = text[from..].find(word) {
        let start = from + found;
        let end = start + word.len();
        let before = text[..start].chars().next_back();
        let after = text[end..].chars().next();
        if !before.is_some_and(char::is_alphabetic) && !after.is_some_and(char::is_alphabetic) {
            return true;
        }
        from = start + text[start..].chars().next().map_or(1, char::len_utf8);
    }
    false
}

/// The pair a line has to follow: the locked one, otherwise the one most lines of the page
/// declare for the direction (the first of them when they tie).
fn page_pairs(
    book: &PairBook,
    declared: &[Option<Declared>],
) -> HashMap<(String, String), (String, String)> {
    let mut reference: HashMap<(String, String), (String, String)> = book
        .locked
        .iter()
        .map(|locked| (locked.declared.direction(), locked.declared.pair()))
        .collect();
    let mut counts: HashMap<(String, String), Vec<((String, String), usize)>> = HashMap::new();
    for declared in declared.iter().flatten() {
        let direction = declared.direction();
        if reference.contains_key(&direction) {
            continue;
        }
        let pairs = counts.entry(direction).or_default();
        match pairs.iter_mut().find(|(pair, _)| *pair == declared.pair()) {
            Some((_, count)) => *count += 1,
            None => pairs.push((declared.pair(), 1)),
        }
    }
    for (direction, pairs) in counts {
        let mut best = &pairs[0];
        for candidate in &pairs[1..] {
            if candidate.1 > best.1 {
                best = candidate;
            }
        }
        reference.insert(direction, best.0.clone());
    }
    reference
}

/// The lines (by index) whose declared words or text contradict what the notes of the page
/// decided for them. The notes name the words of every line, so a line is held to its own and not
/// to what the rest of its direction uses: a line the notes give other words than the rest is
/// right to have them. The names the model wrote are not compared, only the words.
pub(crate) fn conflicts_with_notes(
    expected: &[Option<Declared>],
    declared: &[Option<Declared>],
    translations: &[String],
) -> Vec<usize> {
    let mut flagged = Vec::new();
    for (index, translation) in translations.iter().enumerate() {
        let Some(expect) = expected.get(index).and_then(Option::as_ref) else {
            continue;
        };
        let text = translation.to_lowercase();
        let wrong_declared = declared
            .get(index)
            .and_then(Option::as_ref)
            .is_some_and(|declared| declared.pair() != expect.pair());
        let wrong_self = FIRST_PERSON
            .iter()
            .any(|word| *word != expect.mine && has_word(&text, word));
        let wrong_you = SECOND_PERSON
            .iter()
            .any(|word| *word != expect.yours && has_word(&text, word));
        if wrong_declared || wrong_self || wrong_you {
            flagged.push(index);
        }
    }
    flagged
}

/// The instruction for translating the lines that contradict the notes again. `retried` are the
/// lines the retry carries, in order (its ids are their positions); `wrong` are those among them
/// that contradict the notes.
pub(crate) fn notes_note(
    expected: &[Option<Declared>],
    retried: &[usize],
    wrong: &[usize],
) -> String {
    let mut listed: Vec<String> = Vec::new();
    for (position, index) in retried.iter().enumerate() {
        if !wrong.contains(index) {
            continue;
        }
        let Some(expect) = expected.get(*index).and_then(Option::as_ref) else {
            continue;
        };
        listed.push(format!(
            "{position}: {}>{}: {} - {}",
            expect.from_name, expect.to_name, expect.mine, expect.yours
        ));
    }
    format!(
        "Strict retry: each of these lines was written with other words for 'I' and 'you' than \
         the notes of the page decided for it. Write each line again with exactly the words of its \
         note, and keep the `speaker` of the line in the form `Speaker>Listener: I-word - you-word` \
         with the same pair. The words by `id`: {}.",
        listed.join(" | ")
    )
}

impl PairBook {
    pub(crate) fn clear(&mut self) {
        *self = Self::default();
    }

    /// The lines (by index) whose pair or self pronoun differs from the pair of their
    /// direction, which is the locked pair or else what the rest of the page uses.
    pub(crate) fn conflicts(
        &self,
        declared: &[Option<Declared>],
        translations: &[String],
    ) -> Vec<usize> {
        let reference = page_pairs(self, declared);
        let mut flagged = Vec::new();
        for (index, (declared, translation)) in declared.iter().zip(translations).enumerate() {
            let Some(declared) = declared else { continue };
            let Some((mine, yours)) = reference.get(&declared.direction()) else {
                continue;
            };
            let text = translation.to_lowercase();
            let wrong_pair = declared.pair() != (mine.clone(), yours.clone());
            let wrong_self = FIRST_PERSON
                .iter()
                .any(|word| word != mine && has_word(&text, word));
            let wrong_you = SECOND_PERSON
                .iter()
                .any(|word| word != yours && has_word(&text, word));
            if wrong_pair || wrong_self || wrong_you {
                flagged.push(index);
            }
        }
        flagged
    }

    /// The first-person word each speaker uses aloud: the one of the locked pairs, otherwise
    /// the one the page declares. A speaker who uses more than one (different words for
    /// different listeners) is left out, because then no single word is right for a thought.
    fn spoken_self_words(&self, declared: &[Option<Declared>]) -> HashMap<String, String> {
        let reference = page_pairs(self, declared);
        let mut words: HashMap<String, HashSet<String>> = HashMap::new();
        for ((from, _), (mine, _)) in &reference {
            words.entry(from.clone()).or_default().insert(mine.clone());
        }
        words
            .into_iter()
            .filter_map(|(name, set)| {
                if set.len() == 1 {
                    set.into_iter().next().map(|word| (name, word))
                } else {
                    None
                }
            })
            .collect()
    }

    /// The thoughts (by index) of a character whose first-person word, declared or used in the
    /// text, is neither the word the character uses aloud nor the plain inner 'mình'.
    pub(crate) fn thought_conflicts(
        &self,
        thoughts: &[Option<Thought>],
        declared: &[Option<Declared>],
        translations: &[String],
    ) -> Vec<usize> {
        let spoken = self.spoken_self_words(declared);
        let mut flagged = Vec::new();
        for (index, (thought, translation)) in thoughts.iter().zip(translations).enumerate() {
            let Some(thought) = thought else { continue };
            let Some(word) = spoken.get(&thought.name) else {
                continue;
            };
            let allowed = |candidate: &str| candidate == word || candidate == INNER_SELF;
            let text = translation.to_lowercase();
            let wrong_word = !allowed(&thought.mine);
            let wrong_text = FIRST_PERSON
                .iter()
                .any(|candidate| !allowed(candidate) && has_word(&text, candidate));
            if wrong_word || wrong_text {
                flagged.push(index);
            }
        }
        flagged
    }

    /// The instruction for translating the flagged thoughts again.
    pub(crate) fn thought_note(
        &self,
        thoughts: &[Option<Thought>],
        declared: &[Option<Declared>],
        flagged: &[usize],
    ) -> String {
        let spoken = self.spoken_self_words(declared);
        let mut listed: Vec<String> = Vec::new();
        for &index in flagged {
            let Some(thought) = thoughts.get(index).and_then(Option::as_ref) else {
                continue;
            };
            if let Some(word) = spoken.get(&thought.name) {
                let item = format!("{}: {word}", thought.display);
                if !listed.contains(&item) {
                    listed.push(item);
                }
            }
        }
        format!(
            "Strict retry: each of these thoughts or monologues used a first-person word that \
             its character does not use. A thought keeps the first-person word its character uses \
             aloud (or the plain '{INNER_SELF}' for an inner voice). Write each one again with it, \
             and keep its `speaker` in the form `Name>thought: I-word`. The words: {}.",
            listed.join(" | ")
        )
    }

    /// The instruction for translating the flagged lines again: the pair of every direction
    /// they belong to.
    pub(crate) fn note(&self, declared: &[Option<Declared>], flagged: &[usize]) -> String {
        let reference = page_pairs(self, declared);
        let mut listed: Vec<String> = Vec::new();
        for &index in flagged {
            let Some(declared) = declared.get(index).and_then(Option::as_ref) else {
                continue;
            };
            if let Some((mine, yours)) = reference.get(&declared.direction()) {
                let item = format!(
                    "{}>{}: {mine} - {yours}",
                    declared.from_name, declared.to_name
                );
                if !listed.contains(&item) {
                    listed.push(item);
                }
            }
        }
        format!(
            "Strict retry: each of these lines was written with a different pair of pronouns than \
             the one already decided for the same speaker and listener. Write each line again with \
             exactly that pair, and keep the `speaker` of the line in the form `Speaker>Listener: \
             I-word - you-word` with the same pair. The decided pairs: {}.",
            listed.join(" | ")
        )
    }

    /// Counts the lines that show their pair (the translation uses one of its words) as votes,
    /// and locks a pair that has enough of them. Returns how many pairs were locked now.
    pub(crate) fn learn(&mut self, declared: &[Option<Declared>], translations: &[String]) -> usize {
        self.page += 1;
        let page = self.page;
        let mut touched: Vec<(String, String)> = Vec::new();
        for (declared, translation) in declared.iter().zip(translations) {
            let Some(declared) = declared else { continue };
            let direction = declared.direction();
            if self.locked.iter().any(|locked| locked.declared.direction() == direction) {
                continue;
            }
            let text = translation.to_lowercase();
            if !(has_word(&text, &declared.mine) || has_word(&text, &declared.yours)) {
                continue;
            }
            self.names
                .entry(direction.clone())
                .or_insert_with(|| (declared.from_name.clone(), declared.to_name.clone()));
            let vote = self
                .votes
                .entry(direction.clone())
                .or_default()
                .entry(declared.pair())
                .or_default();
            vote.count += 1;
            vote.pages.insert(page);
            if !touched.contains(&direction) {
                touched.push(direction);
            }
        }
        let mut newly = 0;
        for direction in touched {
            let pairs = &self.votes[&direction];
            let mut ranked: Vec<(&(String, String), &Vote)> = pairs.iter().collect();
            ranked.sort_by(|a, b| b.1.count.cmp(&a.1.count).then_with(|| a.0.cmp(b.0)));
            let (pair, top) = ranked[0];
            let runner_up = ranked.get(1).map_or(0, |(_, vote)| vote.count);
            let enough = top.count >= LOCK_VOTES_ANY_PAGES
                || (top.count >= LOCK_VOTES && top.pages.len() >= LOCK_PAGES);
            if enough && top.count >= 2 * runner_up && top.count >= runner_up + LOCK_VOTES {
                let (from_name, to_name) = self.names[&direction].clone();
                let locked = Declared {
                    from: direction.0.clone(),
                    to: direction.1.clone(),
                    from_name,
                    to_name,
                    mine: pair.0.clone(),
                    yours: pair.1.clone(),
                };
                self.locked.push(Locked { declared: locked });
                newly += 1;
            }
        }
        newly
    }

    /// Locks a pair that was decided outside the translations (by the analysis of the whole
    /// work) and so replaces what the translations had locked for the direction. Returns
    /// false for names or words that make no pair.
    pub(crate) fn seed(&mut self, from_name: &str, to_name: &str, mine: &str, yours: &str) -> bool {
        let (from, to) = (norm_name(from_name), norm_name(to_name));
        let (Some(mine), Some(yours)) = (clean_word(mine), clean_word(yours)) else {
            return false;
        };
        if from == to || GENERIC.contains(&from.as_str()) || GENERIC.contains(&to.as_str()) {
            return false;
        }
        self.locked
            .retain(|locked| (locked.declared.from.as_str(), locked.declared.to.as_str()) != (from.as_str(), to.as_str()));
        self.locked.push(Locked {
            declared: Declared {
                from,
                to,
                from_name: from_name.trim().to_owned(),
                to_name: to_name.trim().to_owned(),
                mine,
                yours,
            },
        });
        true
    }

    /// The locked pairs as glossary `address` entries: (`Speaker -> Listener`, `I - you`).
    pub(crate) fn entries(&self) -> Vec<(String, String)> {
        self.locked
            .iter()
            .map(|locked| {
                (
                    format!("{} -> {}", locked.declared.from_name, locked.declared.to_name),
                    format!("{} - {}", locked.declared.mine, locked.declared.yours),
                )
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn declared(text: &str) -> Option<Declared> {
        parse_declared(text)
    }

    fn texts(items: &[&str]) -> Vec<String> {
        items.iter().map(|item| (*item).to_owned()).collect()
    }

    #[test]
    fn a_pair_is_read_with_its_two_people() {
        let d = declared("Sasaki-kun>Rio-san: anh - em").unwrap();
        assert_eq!((d.from.as_str(), d.to.as_str()), ("sasaki", "rio"));
        assert_eq!((d.mine.as_str(), d.yours.as_str()), ("anh", "em"));
        assert_eq!(d.from_name, "Sasaki-kun");
        assert!(declared("A→B: tôi – chị").is_some());
        assert!(declared("A > B：tớ - cậu").is_some());
    }

    #[test]
    fn narration_thoughts_and_loose_text_are_not_pairs() {
        for text in [
            "narration",
            "thought",
            "Rio>Rio: tôi - tôi",
            "Rio>narration: tôi - anh",
            "unknown>Rio: tôi - cô",
            "A>B: anh",
            "A>B: anh - em - chị",
            "A>B",
            "",
        ] {
            assert!(declared(text).is_none(), "{text}");
        }
    }

    #[test]
    fn words_are_matched_whole() {
        assert!(has_word("anh biết rồi", "anh"));
        assert!(has_word("chào anh.", "anh"));
        assert!(!has_word("tranh vẽ", "anh"));
        assert!(!has_word("anh ấy", "anh ấy ơi"));
    }

    fn line(book: &mut PairBook, speaker: &str, text: &str) {
        book.learn(&[declared(speaker)], &texts(&[text]));
    }

    #[test]
    fn a_pair_locks_after_votes_from_two_pages_and_not_before() {
        let mut book = PairBook::default();
        line(&mut book, "Sasaki>Rio: anh - em", "Anh sẽ giúp em.");
        assert!(book.entries().is_empty());
        line(&mut book, "Sasaki>Rio: anh - em", "Em cứ nói với anh nhé.");
        assert_eq!(
            book.entries(),
            [("Sasaki -> Rio".to_owned(), "anh - em".to_owned())]
        );
    }

    #[test]
    fn two_votes_from_one_page_are_not_enough_but_three_are() {
        let mut book = PairBook::default();
        let both = [
            declared("A>B: tôi - cô"),
            declared("A>B: tôi - cô"),
        ];
        book.learn(&both, &texts(&["Tôi biết.", "Cô nghĩ sao?"]));
        assert!(book.entries().is_empty());
        let three = [declared("A>B: tôi - cô")];
        book.learn(&three, &texts(&["Tôi đi đây."]));
        assert_eq!(book.entries().len(), 1);
    }

    #[test]
    fn a_line_that_does_not_use_its_pair_is_no_vote() {
        let mut book = PairBook::default();
        for _ in 0..4 {
            line(&mut book, "A>B: anh - em", "Cảm ơn nhé.");
        }
        assert!(book.entries().is_empty());
    }

    #[test]
    fn competing_pairs_do_not_lock() {
        let mut book = PairBook::default();
        line(&mut book, "A>B: anh - em", "Anh đây.");
        line(&mut book, "A>B: tôi - cô", "Tôi đây.");
        line(&mut book, "A>B: anh - em", "Em ơi, anh đây.");
        line(&mut book, "A>B: tôi - cô", "Cô à, tôi đây.");
        assert!(book.entries().is_empty());
    }

    #[test]
    fn a_clear_majority_locks_even_when_the_model_wavered_once() {
        let mut book = PairBook::default();
        line(&mut book, "A>B: tôi - cô", "Tôi đây.");
        for _ in 0..3 {
            line(&mut book, "A>B: anh - em", "Em ơi, anh đây.");
        }
        assert_eq!(book.entries(), [("A -> B".to_owned(), "anh - em".to_owned())]);
    }

    #[test]
    fn a_page_that_disagrees_with_itself_flags_the_minority() {
        let book = PairBook::default();
        let declared = [
            declared("Sasaki>Rio: anh - em"),
            declared("Sasaki>Rio: anh - em"),
            declared("Sasaki>Rio: tôi - cô"),
            declared("Rio>Sasaki: em - anh"),
        ];
        let translations = texts(&["Anh đây.", "Em đừng lo.", "Tôi sẽ giúp cô.", "Em cảm ơn anh."]);
        assert_eq!(book.conflicts(&declared, &translations), [2]);
        let note = book.note(&declared, &[2]);
        assert!(note.contains("Sasaki>Rio: anh - em"), "{note}");
    }

    #[test]
    fn a_seeded_pair_replaces_a_locked_one_and_is_enforced() {
        let mut book = PairBook::default();
        line(&mut book, "Sasaki>Rio: anh - em", "Anh đây.");
        line(&mut book, "Sasaki>Rio: anh - em", "Em ơi.");
        assert!(book.seed("Sasaki-kun", "Rio", "Tôi", "chị"));
        assert!(!book.seed("narrator", "Rio", "tôi", "chị"));
        assert!(!book.seed("Sasaki", "Rio", "", "chị"));
        assert_eq!(book.entries(), [("Sasaki-kun -> Rio".to_owned(), "tôi - chị".to_owned())]);
        let declared = [declared("Sasaki>Rio: anh - em"), declared("Sasaki>Rio: tôi - chị")];
        let translations = texts(&["Anh cảm ơn em.", "Tôi muốn giúp chị."]);
        assert_eq!(book.conflicts(&declared, &translations), [0]);
    }

    #[test]
    fn a_locked_pair_beats_the_page_majority() {
        let mut book = PairBook::default();
        line(&mut book, "A>B: anh - em", "Anh đây.");
        line(&mut book, "A>B: anh - em", "Em ơi.");
        let declared = [declared("A>B: tôi - cô"), declared("A>B: tôi - cô")];
        let translations = texts(&["Tôi biết.", "Cô đi đi."]);
        assert_eq!(book.conflicts(&declared, &translations), [0, 1]);
    }

    #[test]
    fn a_self_pronoun_that_changes_inside_the_text_is_flagged() {
        let mut book = PairBook::default();
        line(&mut book, "A>B: anh - em", "Anh đây.");
        line(&mut book, "A>B: anh - em", "Em ơi.");
        let declared = [declared("A>B: anh - em")];
        assert_eq!(book.conflicts(&declared, &texts(&["Tôi muốn giúp em."])), [0]);
        assert!(book.conflicts(&declared, &texts(&["Anh muốn giúp em."])).is_empty());
        // a line without a declared pair is never flagged
        assert!(book.conflicts(&[None], &texts(&["Tôi đây."])).is_empty());
    }

    #[test]
    fn the_names_are_matched_without_honorifics() {
        let mut book = PairBook::default();
        line(&mut book, "Sasaki-kun>Rio-san: anh - em", "Anh đây.");
        line(&mut book, "Sasaki>Rio: anh - em", "Anh đi đây, em nhé.");
        assert_eq!(book.entries().len(), 1);
    }

    #[test]
    fn clearing_forgets_everything() {
        let mut book = PairBook::default();
        line(&mut book, "A>B: anh - em", "Anh đây.");
        line(&mut book, "A>B: anh - em", "Em ơi.");
        book.clear();
        assert!(book.entries().is_empty());
    }

    #[test]
    fn a_thought_is_read_with_its_character() {
        let t = parse_thought("Rio-san>thought: mình").unwrap();
        assert_eq!((t.name.as_str(), t.display.as_str(), t.mine.as_str()), ("rio", "Rio-san", "mình"));
        assert!(parse_thought("Rio>suy nghĩ: chị - ").is_some());
        for text in ["thought", "narration", "Rio>Sasaki: tôi - anh", "unknown>thought: tôi", "Rio>thought", ""] {
            assert!(parse_thought(text).is_none(), "{text}");
        }
        // A thought is never a pair.
        assert!(declared("Rio>thought: mình").is_none());
    }

    #[test]
    fn a_thought_keeps_the_word_its_character_uses_aloud() {
        let mut book = PairBook::default();
        let lines = ["Rio>Sasaki: chị - cậu", "Rio>Sasaki: chị - cậu", "Rio>thought: tôi", "Rio>thought: mình", "Rio>thought: chị"];
        let declared: Vec<_> = lines.iter().map(|l| parse_declared(l)).collect();
        let thoughts: Vec<_> = lines.iter().map(|l| parse_thought(l)).collect();
        let translations = texts(&[
            "Cậu giúp chị nhiều quá",
            "Chị cảm ơn cậu",
            "Tôi phải cố gắng mới được…",
            "Mình phải cố gắng mới được…",
            "Chị phải cố gắng mới được…",
        ]);
        book.learn(&declared, &translations);
        // 'tôi' is not what Rio says aloud; 'mình' and her own 'chị' are fine.
        assert_eq!(book.thought_conflicts(&thoughts, &declared, &translations), [2]);
        let note = book.thought_note(&thoughts, &declared, &[2]);
        assert!(note.contains("Rio>thought: chị") || note.contains("Rio: chị"), "{note}");
    }

    #[test]
    fn a_speaker_with_two_spoken_words_has_no_thought_to_check() {
        let book = PairBook::default();
        let lines = ["Rio>Sasaki: chị - cậu", "Rio>Boss: tôi - ông", "Rio>thought: tớ"];
        let declared: Vec<_> = lines.iter().map(|l| parse_declared(l)).collect();
        let thoughts: Vec<_> = lines.iter().map(|l| parse_thought(l)).collect();
        let translations = texts(&["a", "b", "Tớ phải cố"]);
        assert!(book.thought_conflicts(&thoughts, &declared, &translations).is_empty());
    }

    fn expect(speaker: &str, listener: &str, mine: &str, yours: &str) -> Option<Declared> {
        Declared::expected(&ExpectedPair {
            speaker: speaker.to_owned(),
            listener: listener.to_owned(),
            mine: mine.to_owned(),
            yours: yours.to_owned(),
        })
    }

    #[test]
    fn the_notes_make_a_pair_only_from_two_names_and_two_words() {
        let pair = expect("Sasaki-kun", "Rio", "Tôi", "chị").unwrap();
        assert_eq!((pair.from.as_str(), pair.to.as_str()), ("sasaki", "rio"));
        assert_eq!(pair.pair(), ("tôi".to_owned(), "chị".to_owned()));
        assert!(expect("unknown", "Rio", "tôi", "chị").is_none());
        assert!(expect("Sasaki", "Rio", "", "chị").is_none());
    }

    #[test]
    fn a_line_is_held_to_its_own_note_not_to_the_rest_of_its_direction() {
        // Two lines of one direction: the notes give the second one other words (an exception).
        let expected = vec![expect("A", "B", "anh", "em"), expect("A", "B", "tao", "mày")];
        let declared = vec![parse_declared("A>B: anh - em"), parse_declared("A>B: tao - mày")];
        let translations = texts(&["Anh đây.", "Tao đây, mày đừng lo."]);
        assert!(conflicts_with_notes(&expected, &declared, &translations).is_empty());
    }

    #[test]
    fn a_declared_pair_or_a_word_in_the_text_that_the_note_did_not_give_is_flagged() {
        let expected = vec![
            expect("A", "B", "anh", "em"),
            expect("A", "B", "anh", "em"),
            expect("A", "B", "anh", "em"),
            None,
        ];
        let declared = vec![
            parse_declared("A>B: tôi - cô"),
            parse_declared("A>B: anh - em"),
            None,
            parse_declared("A>B: tôi - cô"),
        ];
        let translations = texts(&["Cô đi đi.", "Tôi nói rồi mà.", "Tao biết rồi.", "Tôi đây."]);
        // 0: the declared pair differs; 1: declared right but "tôi" is in the text; 2: nothing
        // declared but "tao" is in the text; 3: no note, no judgement.
        assert_eq!(conflicts_with_notes(&expected, &declared, &translations), [0, 1, 2]);
    }

    #[test]
    fn the_retry_note_numbers_the_lines_the_way_the_retry_does() {
        let expected = vec![
            expect("A", "B", "anh", "em"),
            None,
            expect("Sasaki", "Rio", "tôi", "chị"),
            expect("A", "B", "anh", "em"),
        ];
        // The retry carries lines 0, 2 and 3 (positions 0, 1, 2); 0 and 2 are the wrong ones.
        let note = notes_note(&expected, &[0, 2, 3], &[0, 2]);
        assert!(note.contains("0: A>B: anh - em | 1: Sasaki>Rio: tôi - chị."), "{note}");
        assert!(!note.contains("2:"), "{note}");
    }
}
