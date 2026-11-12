//! The second reading of a work.
//!
//! The first reading goes page by page, so what a model chose on page 3 can clash with what
//! it chose on page 40: one person calls a man "anh" and another "cô", a pair flips between
//! "anh - em" and "tôi - cô", a thought gets the wrong word for "I". Once every page is read,
//! the same models see the whole work as text (the picture is not needed: the first reading
//! already turned it into who, to whom and how) and decide the words for good:
//!
//! - for every direction (speaker to listener) one pair of words that fits the relationship,
//!   the ages, the rank and the story, written over what the first reading learned;
//! - the lines that must use other words than their direction's pair, because the relationship
//!   changes in the story or because the line has no pair (a thought);
//! - the gender and age of a character, when the lines show that the first reading was wrong.
//!
//! The pairs and the exceptions are applied to the notes of every page and to the glossary. A
//! part that no model can read leaves the notes of the first reading as they were.
//!
//! This is where thinking pays, so the models think here (`review_reasoning`) and not while
//! they read a page.

use std::{collections::HashMap, fmt::Write as _, ops::Range};

use super::*;
use crate::address::PairBook;

const SCHEMA_NAME: &str = "manga_address_review";

/// A line that must use words other than the pair of its direction.
#[derive(Clone, Debug, Default, PartialEq, Deserialize)]
#[serde(default)]
struct Change {
    /// The number of the page, counted from 1.
    #[serde(deserialize_with = "lenient_id")]
    page: usize,
    /// The number of the line on the page.
    #[serde(deserialize_with = "lenient_id")]
    id: usize,
    #[serde(rename = "self", deserialize_with = "lenient_string")]
    self_word: String,
    #[serde(deserialize_with = "lenient_string")]
    you: String,
}

#[derive(Clone, Debug, Default, PartialEq, Deserialize)]
#[serde(default)]
struct ReviewOutput {
    characters: Vec<Character>,
    changes: Vec<Change>,
}

fn schema() -> Value {
    let text = |description: &str| json!({ "type": "string", "description": description });
    json!({
        "type": "object",
        "properties": {
            "characters": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "name": text("The name of the character, exactly as in the list you were given."),
                        "gender": text("man, woman, boy, girl: only to correct the list."),
                        "age": text("A number or a stage: only to correct the list."),
                        "relationships": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "with": text("The listener: another character, exactly as in the list."),
                                    "relation": text("Their relation: colleague, mother and son, ..."),
                                    "self": text("The word this character uses for themselves when talking to the listener."),
                                    "you": text("The word this character uses for the listener.")
                                },
                                "required": ["with", "self", "you"]
                            }
                        }
                    },
                    "required": ["name", "relationships"]
                }
            },
            "changes": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "page": { "type": "integer", "minimum": 1, "description": "The number of the page." },
                        "id": { "type": "integer", "minimum": 0, "description": "The number of the line on that page." },
                        "self": text("The word the speaker uses for themselves in this line."),
                        "you": text("The word the speaker uses for the listener in this line.")
                    },
                    "required": ["page", "id"]
                }
            }
        },
        "required": ["characters", "changes"]
    })
}

fn system_prompt(target: Language) -> String {
    let mut prompt = format!(
        "You are the editor of a manga translated into {target}. Readers looked at the pages one \
by one and noted for every line who says it to whom and which words the speaker uses for \
themselves ('self') and for the listener ('you') in {target}. They worked page by page, so \
their choices can clash. You now see the whole work as text and you settle the words for good, \
so that they fit the story and are the same wherever the same people talk to each other.

What you do:
- Check the gender and age of every character against all their lines: how they speak (俺, 僕, \
あたし, sentence endings, honorifics) and how the others talk to them. Report a character's \
gender or age only to correct it.
- For every direction (a speaker to a listener) that appears in the lines, decide the one pair of \
words that fits their relationship, their ages, their rank and their closeness, and give it in \
characters[].relationships (with, relation, self, you). A pair belongs to one direction: what A \
calls B is not what B calls A.
- The same person is called the same way by everyone who stands in the same relation to them. \
Do not let one speaker call a man 'anh' and another speaker, in the same relation, call him 'cô'.
- Honorifics (さん, くん, ちゃん, 様) show closeness and rank and never the gender of the \
person who has them.
- A pair can change in the story: strangers become friends, a quarrel, a talk to a child. Give \
in characters the pair that holds for most of the work, and list in changes each line that \
must use other words: page (counted from 1), id (the number of the line), self and/or you. \
Give the line a word only when it differs from the pair of its direction, or when the line has \
no pair (a thought uses the word the character uses aloud, or 'mình').
- Do not list a line that is already right. Never rename or invent a character: use the names \
of the list. Do not touch the lines of narration, signs or sound effects.
- Take your time to think about the whole work before you answer.
- Answer with the JSON object only."
    );
    prompt.push_str(pronoun_rules(target));
    prompt
}

/// How a page looks to the editor: its scene and its lines with who says them to whom and with
/// which words. `None` for a page without such lines.
fn page_digest(number: usize, page: &PageAnalysis) -> Option<String> {
    let texts = page.texts(&page.sources);
    let mut digest = format!("Page {}", number + 1);
    if !page.scene.is_empty() {
        let _ = write!(digest, ": {}", clip(&page.scene, 160));
    }
    digest.push('\n');
    let mut any = false;
    for (id, line) in page.lines.iter().enumerate() {
        if line.kind.is_empty()
            && line.speaker.trim().is_empty()
            && line.self_word.is_empty()
            && line.you.is_empty()
        {
            continue;
        }
        any = true;
        let kind = if line.kind.is_empty() { "speech" } else { &line.kind };
        let speaker = if line.speaker.trim().is_empty() { "unknown" } else { line.speaker.trim() };
        let listener = if line.listener.trim().is_empty() { "unknown" } else { line.listener.trim() };
        let _ = write!(digest, "{id} {kind} {speaker} -> {listener}");
        if !line.relation.is_empty() {
            let _ = write!(digest, " ({})", line.relation);
        }
        if !line.self_word.is_empty() || !line.you.is_empty() {
            let quote = |text: &str| serde_json::to_string(text).unwrap_or_default();
            let _ = write!(digest, " | self {} you {}", quote(&line.self_word), quote(&line.you));
        }
        if !line.tone.is_empty() {
            let _ = write!(digest, " | {}", line.tone);
        }
        if let Some(text) = texts.get(id) {
            let _ = write!(
                digest,
                " | {}",
                serde_json::to_string(&clip(text, 80)).unwrap_or_default()
            );
        }
        digest.push('\n');
    }
    any.then_some(digest)
}

fn user_prompt(bible: &Bible, pages: &[(usize, String)], total: usize) -> String {
    let characters = json!({ "characters": bible.characters }).to_string();
    let first = pages.first().map_or(0, |(number, _)| number + 1);
    let last = pages.last().map_or(0, |(number, _)| number + 1);
    let mut prompt = format!(
        "The work has {total} pages. The characters as the readers left them (JSON): \
{characters}\n\nPages {first} to {last}, one line per OCR line: id, kind, speaker -> listener \
(relation), the words used, the tone and the text.\n\n"
    );
    for (_, digest) in pages {
        prompt.push_str(digest);
        prompt.push('\n');
    }
    prompt.push_str("Settle the words now and answer with the JSON object.");
    prompt
}

fn parse(text: &str) -> anyhow::Result<ReviewOutput> {
    let json = json_object(text).ok_or_else(|| anyhow!("the answer holds no JSON object"))?;
    serde_json::from_str(json).context("invalid review JSON")
}

/// Splits pages of the given sizes into consecutive parts of at most `limit` characters (a
/// page larger than the limit is a part of its own).
fn parts(sizes: &[usize], limit: usize) -> Vec<Range<usize>> {
    let mut parts = Vec::new();
    let mut start = 0;
    while start < sizes.len() {
        let mut end = start + 1;
        let mut total = sizes[start];
        while end < sizes.len() && total + sizes[end] <= limit {
            total += sizes[end];
            end += 1;
        }
        parts.push(start..end);
        start = end;
    }
    parts
}

/// Replaces `field` with `value` when `value` says something. Returns whether it changed.
fn overwrite(field: &mut String, value: &str) -> bool {
    let value = clip(value, FIELD_CHARS);
    if value.is_empty() || *field == value {
        return false;
    }
    *field = value;
    true
}

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
    fn pair(&self, speaker: &str, listener: &str) -> Option<&Relationship> {
        let character = &self.characters[self.character_index(speaker)?];
        let target = self.character_index(listener);
        character.relationships.iter().find(|relationship| {
            same_name(&relationship.with, listener)
                || (target.is_some() && self.character_index(&relationship.with) == target)
        })
    }

    /// Writes what the editor decided over what the first reading learned (unlike `fill`,
    /// which keeps the first answer). A character the work does not know is ignored. Returns
    /// whether anything changed.
    fn review_character(&mut self, new: &Character) -> bool {
        let Some(index) = self.character_index(&new.name) else {
            return false;
        };
        // The character every known and every new relationship points at, aliases resolved.
        let existing: Vec<Option<usize>> = self.characters[index]
            .relationships
            .iter()
            .map(|relationship| self.character_index(&relationship.with))
            .collect();
        let wanted: Vec<Option<usize>> = new
            .relationships
            .iter()
            .map(|relationship| self.character_index(&relationship.with))
            .collect();
        let known = &mut self.characters[index];
        let mut changed = overwrite(&mut known.gender, &new.gender);
        changed |= overwrite(&mut known.age, &new.age);
        for (relationship, target) in new.relationships.iter().zip(wanted) {
            if is_generic(&relationship.with)
                || (relationship.self_word.trim().is_empty()
                    && relationship.you.trim().is_empty()
                    && relationship.relation.trim().is_empty())
            {
                continue;
            }
            let position = (0..known.relationships.len()).find(|&at| {
                same_name(&known.relationships[at].with, &relationship.with)
                    || (target.is_some() && existing.get(at).copied().flatten() == target)
            });
            match position {
                Some(at) => {
                    let slot = &mut known.relationships[at];
                    changed |= overwrite(&mut slot.relation, &relationship.relation);
                    changed |= overwrite(&mut slot.self_word, &relationship.self_word);
                    changed |= overwrite(&mut slot.you, &relationship.you);
                }
                None => {
                    known.relationships.push(Relationship {
                        with: clip(&relationship.with, FIELD_CHARS),
                        relation: clip(&relationship.relation, FIELD_CHARS),
                        self_word: clip(&relationship.self_word, FIELD_CHARS),
                        you: clip(&relationship.you, FIELD_CHARS),
                    });
                    changed = true;
                }
            }
        }
        changed
    }
}

impl PageAnalysis {
    /// Puts the settled words into the notes of the page: a line gets the pair of its
    /// direction, unless `exceptions` (the words by line number) say otherwise, and the cast of
    /// the page is written again from the characters as they are now. Returns how many lines
    /// ended up with other words than before.
    fn apply_review(&mut self, bible: &Bible, exceptions: Option<&HashMap<usize, (String, String)>>) -> usize {
        let mut changed = 0;
        for (id, line) in self.lines.iter_mut().enumerate() {
            if line.kind.is_empty() && line.speaker.trim().is_empty() {
                continue;
            }
            let before = (line.self_word.clone(), line.you.clone());
            let pair = if is_generic(&line.speaker) || is_generic(&line.listener) {
                None
            } else {
                bible.pair(&line.speaker, &line.listener)
            };
            if let Some(pair) = pair {
                if !pair.relation.trim().is_empty() {
                    line.relation = clip(&pair.relation, 60);
                }
            }
            match exceptions.and_then(|lines| lines.get(&id)) {
                Some((mine, yours)) => {
                    if !mine.is_empty() {
                        line.self_word = mine.clone();
                    }
                    if !yours.is_empty() {
                        line.you = yours.clone();
                    }
                }
                None => {
                    if let Some(pair) = pair {
                        if !pair.self_word.trim().is_empty() {
                            line.self_word = clip(&pair.self_word, 40);
                        }
                        if !pair.you.trim().is_empty() {
                            line.you = clip(&pair.you, 40);
                        }
                    }
                }
            }
            if (line.self_word.clone(), line.you.clone()) != before {
                changed += 1;
            }
        }
        let mut cast: Vec<String> = Vec::new();
        for line in &self.lines {
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
        self.cast = cast;
        changed
    }
}

impl Translator {
    /// Reads the whole work again as text and settles the words for "I" and "you" (see the
    /// module). Changes `bible` and the notes in `results`; whatever fails leaves them as the
    /// first reading made them.
    #[allow(clippy::too_many_arguments)]
    pub(super) async fn review_work(
        &self,
        config: &AnalysisConfig,
        providers: &ProvidersConfig,
        language: Language,
        bible: &mut Bible,
        results: &mut [Option<PageAnalysis>],
        dead: &mut HashSet<usize>,
        stop: &(impl Fn() -> bool + Send + Sync),
    ) {
        let digests: Vec<(usize, String)> = results
            .iter()
            .enumerate()
            .filter_map(|(number, page)| {
                page.as_ref()
                    .and_then(|page| page_digest(number, page))
                    .map(|digest| (number, digest))
            })
            .collect();
        if digests.is_empty() || bible.characters.is_empty() {
            return;
        }
        let total = results.len();
        let system = system_prompt(language);
        let generation = GenerationConfig {
            temperature: Some(0.2),
            top_p: Some(0.95),
            max_tokens: Some(config.max_tokens),
            reasoning: Some(config.review_reasoning),
            ..GenerationConfig::default()
        };
        let sizes: Vec<usize> = digests.iter().map(|(_, digest)| digest.chars().count()).collect();
        let parts = parts(&sizes, config.review_chars.max(2000));
        let mut exceptions: HashMap<usize, HashMap<usize, (String, String)>> = HashMap::new();
        let mut reviewed = 0;

        for (part, range) in parts.iter().enumerate() {
            if stop() {
                break;
            }
            let pages = &digests[range.clone()];
            let user = user_prompt(bible, pages, total);
            let mut output = None;
            for (position, entry) in config.models.iter().enumerate() {
                if dead.contains(&position) {
                    continue;
                }
                let chat = remote::Chat {
                    system: system.clone(),
                    user: user.clone(),
                    images: Vec::new(),
                    schema_name: SCHEMA_NAME,
                    schema: schema(),
                };
                let attempt = remote::chat_json(&self.client, providers, &entry.model, &generation, chat)
                    .await
                    .map_err(|error| {
                        if matches!(&error, Error::Api { status, .. } if matches!(*status, 401 | 403 | 404)) {
                            dead.insert(position);
                        }
                        anyhow::Error::from(error)
                    })
                    .and_then(|text| parse(&text));
                match attempt {
                    Ok(answer) => {
                        output = Some((entry.model.clone(), answer));
                        break;
                    }
                    Err(error) => tracing::warn!(
                        part = part + 1,
                        model = %entry.model,
                        error = %format!("{error:#}"),
                        "review model could not read the work; trying the next one"
                    ),
                }
            }
            let Some((model, answer)) = output else {
                tracing::warn!(part = part + 1, "no model could review this part of the work");
                continue;
            };
            let characters = answer
                .characters
                .iter()
                .filter(|character| bible.review_character(character))
                .count();
            let (first, last) = (pages[0].0, pages[pages.len() - 1].0);
            let mut kept = 0;
            for change in answer.changes {
                if change.page == 0 || change.page == usize::MAX || change.id == usize::MAX {
                    continue;
                }
                let number = change.page - 1;
                if number < first || number > last {
                    continue;
                }
                exceptions.entry(number).or_default().insert(
                    change.id,
                    (clip(&change.self_word, 40), clip(&change.you, 40)),
                );
                kept += 1;
            }
            reviewed += 1;
            tracing::info!(
                part = part + 1,
                of = parts.len(),
                %model,
                characters,
                exceptions = kept,
                "part of the work reviewed"
            );
        }

        if reviewed == 0 {
            return;
        }
        // Once, at the end: a later part can change a pair an earlier one decided.
        let mut lines = 0;
        for (number, page) in results.iter_mut().enumerate() {
            if let Some(page) = page {
                lines += page.apply_review(bible, exceptions.get(&number));
            }
        }
        tracing::info!(lines, "notes of the pages settled by the review");
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
            {"id": 1, "kind": "speech", "speaker": "Sasaki", "listener": "Rio-san", "self": "anh", "you": "em"},
            {"id": 2, "kind": "thought", "speaker": "Rio", "listener": "self", "self": "mình"},
            {"id": 3, "kind": "narration", "speaker": "narrator", "listener": "everyone"}
        ]
    }"#;

    fn editor_answer() -> ReviewOutput {
        parse(
            r#"Here you go: {"characters": [
                {"name": "Sasaki", "gender": "", "relationships": [
                    {"with": "Rio-san", "relation": "colleague", "self": "tôi", "you": "chị"},
                    {"with": "narrator", "self": "x", "you": "y"}]},
                {"name": "Nobody", "relationships": [{"with": "Rio", "self": "a", "you": "b"}]}
            ], "changes": [{"page": "1", "id": 1, "you": "em"}, {"page": 0, "id": 0, "you": "z"}]}"#,
        )
        .unwrap()
    }

    #[test]
    fn the_editor_overwrites_a_pair_through_an_alias_and_ignores_strangers() {
        let mut bible = Bible::default();
        page(FIRST, 4, &mut bible);
        let answer = editor_answer();
        assert!(bible.review_character(&answer.characters[0]));
        assert!(!bible.review_character(&answer.characters[1]));
        let pair = bible.pair("Sasaki", "Rio").unwrap();
        assert_eq!((pair.self_word.as_str(), pair.you.as_str()), ("tôi", "chị"));
        // The alias found the known relationship instead of adding a second one.
        assert_eq!(bible.characters[0].relationships.len(), 1);
        assert_eq!(bible.characters[0].gender, "man");
        assert!(!bible.review_character(&answer.characters[0]), "nothing left to change");
    }

    #[test]
    fn a_page_gets_the_pair_of_its_direction_and_the_exceptions() {
        let mut bible = Bible::default();
        let mut analysis = page(FIRST, 4, &mut bible);
        let answer = editor_answer();
        bible.review_character(&answer.characters[0]);
        let mut exceptions = HashMap::new();
        exceptions.insert(1, (String::new(), "em".to_owned()));
        let changed = analysis.apply_review(&bible, Some(&exceptions));
        // Line 0 takes the pair ("tôi", "chị"); line 1 keeps its own self word and takes the
        // exception's "you" (so it does not change); the thought and the narration are not
        // touched.
        assert_eq!(changed, 1);
        assert_eq!((analysis.lines[0].self_word.as_str(), analysis.lines[0].you.as_str()), ("tôi", "chị"));
        assert_eq!((analysis.lines[1].self_word.as_str(), analysis.lines[1].you.as_str()), ("anh", "em"));
        assert_eq!(analysis.lines[2].self_word, "mình");
        assert!(analysis.lines[3].self_word.is_empty());
        let notes = analysis.instructions(&analysis.sources.clone());
        assert!(notes.contains("calls the listener \"chị\""));
        assert!(analysis.cast.iter().any(|entry| entry.starts_with("Sasaki: man")));
    }

    #[test]
    fn a_line_with_blank_words_gets_the_pair_and_an_unknown_speaker_gets_nothing() {
        let mut bible = Bible::default();
        let mut analysis = page(
            r#"{"lines": [
                {"id": 0, "kind": "speech", "speaker": "Sasaki", "listener": "Rio"},
                {"id": 1, "kind": "speech", "speaker": "unknown", "listener": "Rio"}]}"#,
            2,
            &mut bible,
        );
        bible.merge_output(&parse_page(FIRST, 4).unwrap());
        assert_eq!(analysis.apply_review(&bible, None), 1);
        assert_eq!(analysis.lines[0].you, "cô");
        assert!(analysis.lines[1].you.is_empty());
    }

    #[test]
    fn the_answer_is_read_with_text_around_it_and_string_numbers() {
        let answer = editor_answer();
        assert_eq!(answer.characters.len(), 2);
        assert_eq!(answer.changes[0], Change { page: 1, id: 1, self_word: String::new(), you: "em".to_owned() });
        assert!(parse("no json here").is_err());
        assert_eq!(parse("{}").unwrap(), ReviewOutput::default());
    }

    #[test]
    fn the_translation_is_held_to_the_pairs_all_lines_agree_on() {
        let mut bible = Bible::default();
        let mut analysis = page(FIRST, 4, &mut bible);
        let answer = editor_answer();
        bible.review_character(&answer.characters[0]);
        let mut exceptions = HashMap::new();
        exceptions.insert(1, (String::new(), "em".to_owned()));
        analysis.apply_review(&bible, Some(&exceptions));
        // Sasaki -> Rio has two pairs now (line 0 and the exception of line 1): not locked.
        let mut book = PairBook::default();
        assert_eq!(seed_pairs(&mut book, &bible, &[Some(analysis.clone())]), 0);
        // Without the exception the direction has one pair, which is locked.
        analysis.apply_review(&bible, None);
        assert_eq!(seed_pairs(&mut book, &bible, &[Some(analysis), None]), 1);
        assert_eq!(book.entries(), [("Sasaki -> Rio".to_owned(), "tôi - chị".to_owned())]);
    }

    #[test]
    fn the_work_is_cut_into_parts_of_whole_pages() {
        assert_eq!(parts(&[10, 10, 10], 25), vec![0..2, 2..3]);
        assert_eq!(parts(&[50, 5], 20), vec![0..1, 1..2]);
        assert_eq!(parts(&[5, 5, 5], 100), vec![0..3]);
        assert!(parts(&[], 10).is_empty());
    }

    #[test]
    fn the_editor_sees_the_pages_the_characters_and_the_rules() {
        let mut bible = Bible::default();
        let analysis = page(FIRST, 4, &mut bible);
        let digest = page_digest(0, &analysis).unwrap();
        assert!(digest.starts_with("Page 1: A hospital room.\n"));
        assert!(digest.contains("0 speech Sasaki -> Rio (colleague) | self \"tôi\" you \"cô\""));
        assert!(digest.contains("2 thought Rio -> self | self \"mình\" you \"\""));
        let prompt = user_prompt(&bible, &[(0, digest)], 12);
        assert!(prompt.starts_with("The work has 12 pages."));
        assert!(prompt.contains("\"name\":\"Sasaki\""));
        let system = system_prompt(Language::Vietnamese);
        assert!(system.contains("never the gender of the person who has them"));
        assert!(system.contains("Vietnamese pronouns"));
        assert!(!system_prompt(Language::English).contains("Vietnamese pronouns"));
        assert!(page_digest(0, &page(r#"{"lines": [{"id": 0}]}"#, 1, &mut Bible::default())).is_none());
    }

    #[test]
    fn the_defaults_read_twice_and_think_only_in_the_second_reading() {
        let config = AnalysisConfig::default();
        assert!(config.review && config.review_reasoning && !config.reasoning);
        assert!(config.models.iter().all(|model| model.reasoning.is_none()));
    }
}
