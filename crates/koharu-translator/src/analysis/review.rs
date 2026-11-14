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
//! The models are those of `analysis.review_models`, tried in that order (DeepSeek first, without
//! thinking; each model may say whether it thinks, else `review_reasoning` decides). When the
//! settled pairs still give a word of one sex to a person of the other, or the two directions
//! of a pair do not fit each other, the editor is asked once more about exactly those (the
//! checks are in `settle`).

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
    /// Who really says the line, when the readers got it wrong.
    #[serde(deserialize_with = "lenient_string")]
    speaker: String,
    /// Who the line is really said to.
    #[serde(deserialize_with = "lenient_string")]
    listener: String,
    #[serde(rename = "self", deserialize_with = "lenient_string")]
    self_word: String,
    #[serde(deserialize_with = "lenient_string")]
    you: String,
}

/// What the editor corrected in one line.
#[derive(Clone, Debug, Default, PartialEq)]
pub(super) struct LineFix {
    speaker: String,
    listener: String,
    self_word: String,
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
                        "speaker": text("The character who really says the line: only to correct the readers."),
                        "listener": text("The character the line is really said to: only to correct the readers."),
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
- Find out who the characters really are. The readers sometimes named a speaker by a role \
(nurse, patient, visitor, customer) or took two people for one, because the names were not known \
on the first pages. The text tells: a name said in a line belongs to someone other than the \
speaker, people who answer each other are different people, and nobody speaks to themself \
except in a thought. When the text shows who really says or hears a line, list that line in \
changes with speaker and/or listener set to the right name. Use the names of the list; add a \
character to characters only when a real person is missing from it.
- Check the gender and age of every character against all their lines: how they speak (俺, 僕, \
あたし, sentence endings, honorifics) and how the others talk to them. Report a character's \
gender or age only to correct it.
- For every direction (a speaker to a listener) decide the one pair of words that fits their \
relationship, their ages, their rank and their closeness, and give it in \
characters[].relationships (with, relation, self, you). A pair belongs to one direction: what A \
calls B is not what B calls A. Lines of a direction take its pair by themselves, so do not go \
through the lines one by one: think about the people and their pairs, then about the few lines \
that need something else.
- The same person is called the same way by everyone who stands in the same relation to them. \
Do not let one speaker call a man 'anh' and another speaker, in the same relation, call him 'cô'.
- Honorifics (さん, くん, ちゃん, 様) show closeness and rank and never the gender of the \
person who has them.
- A pair can change in the story: strangers become friends, a quarrel, a talk to a child. Give \
in characters the pair that holds for most of the work, and list in changes each line that \
must use other words: page (counted from 1), id (the number of the line), self and/or you. \
Give the line a word only when it differs from the pair of its direction, or when the line has \
no pair (a thought uses the word the character uses aloud, or 'mình').
- Do not list a line that is already right. Do not touch the lines of narration, signs or \
sound effects.
- Keep your thinking short: the answer is a table of pairs and a short list of lines.
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

impl Bible {
    /// Writes what the editor decided over what the first reading learned (unlike `fill`,
    /// which keeps the first answer). A character the work does not know is added. Returns
    /// whether anything changed.
    fn review_character(&mut self, new: &Character) -> bool {
        let Some(index) = self.character_index(&new.name) else {
            // A real person the readers missed (they may have called them by a role).
            let known = self.characters.len();
            self.merge_character(new.clone());
            return self.characters.len() > known;
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
    /// Puts the settled words into the notes of the page: a line gets the speaker and listener
    /// the editor corrected, then the pair of its direction, unless `fixes` (by line number)
    /// carry words of their own for it, and the cast of the page is written again from the
    /// characters as they are now. Returns how many lines ended up different than before.
    pub(super) fn apply_review(&mut self, bible: &Bible, fixes: Option<&HashMap<usize, LineFix>>) -> usize {
        let mut changed = 0;
        for (id, line) in self.lines.iter_mut().enumerate() {
            if line.kind.is_empty() && line.speaker.trim().is_empty() {
                continue;
            }
            let before = (
                line.speaker.clone(),
                line.listener.clone(),
                line.self_word.clone(),
                line.you.clone(),
            );
            let fix = fixes.and_then(|lines| lines.get(&id));
            if let Some(fix) = fix {
                if !fix.speaker.is_empty() {
                    line.speaker = fix.speaker.clone();
                }
                if !fix.listener.is_empty() {
                    line.listener = fix.listener.clone();
                }
            }
            line.speaker = bible.canonical(&line.speaker);
            line.listener = bible.canonical(&line.listener);
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
            match fix {
                Some(fix) if !fix.self_word.is_empty() || !fix.you.is_empty() => {
                    if !fix.self_word.is_empty() {
                        line.self_word = fix.self_word.clone();
                    }
                    if !fix.you.is_empty() {
                        line.you = fix.you.clone();
                    }
                }
                _ => {
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
            if (
                line.speaker.clone(),
                line.listener.clone(),
                line.self_word.clone(),
                line.you.clone(),
            ) != before
            {
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

/// What the editor is asked when the settled pairs contradict genders or each other: the
/// problems, and the pages where the characters of the problems talk.
fn repair_prompt(bible: &Bible, problems: &[String], pages: &[(usize, String)], total: usize) -> String {
    let characters = json!({ "characters": bible.characters }).to_string();
    let mut prompt = format!(
        "The work has {total} pages. You settled the words of its characters, and some of them \
cannot be right. The characters now (JSON): {characters}\n\nThe problems:\n"
    );
    for problem in problems {
        let _ = writeln!(prompt, "- {problem}");
    }
    prompt.push_str(
        "\nThe pages where these characters talk, one line per OCR line: id, kind, speaker -> \
listener (relation), the words used, the tone and the text.\n\n",
    );
    for (_, digest) in pages {
        prompt.push_str(digest);
        prompt.push('\n');
    }
    prompt.push_str(
        "For every problem decide from the lines what is true. Answer in characters[].relationships \
with the pair that is right for the direction (the same pair again when it is right after all), \
and give a gender only when the gender of the character is what was wrong. Leave changes empty. \
Answer with the JSON object.",
    );
    prompt
}

impl Translator {
    /// Asks the review models, in the order of `review_models`, until one answers. A model that
    /// thought until its tokens ran out is asked again without thinking; a model the server
    /// does not know or does not let the user call is skipped for the rest of the run.
    async fn ask_review(
        &self,
        config: &AnalysisConfig,
        providers: &ProvidersConfig,
        system: &str,
        user: &str,
        what: &str,
        dead: &mut HashSet<String>,
    ) -> Option<(String, ReviewOutput)> {
        let generation = GenerationConfig {
            // The setting reasoning models are made for: a low temperature makes them loop.
            temperature: Some(0.6),
            top_p: Some(0.95),
            max_tokens: Some(config.review_max_tokens),
            reasoning: Some(config.review_reasoning),
            ..GenerationConfig::default()
        };
        for entry in &config.review_models {
            let model = &entry.model;
            if dead.contains(model) {
                continue;
            }
            let mut reasoning = entry.reasoning.unwrap_or(config.review_reasoning);
            loop {
                let generation = GenerationConfig {
                    reasoning: Some(reasoning),
                    ..generation
                };
                let chat = remote::Chat {
                    system: system.to_owned(),
                    user: user.to_owned(),
                    images: Vec::new(),
                    schema_name: SCHEMA_NAME,
                    schema: schema(),
                };
                let attempt = remote::chat_json(&self.client, providers, model, &generation, chat)
                    .await
                    .map_err(|error| {
                        if matches!(&error, Error::Api { status, .. } if matches!(*status, 401 | 403 | 404)) {
                            dead.insert(model.clone());
                        }
                        anyhow::Error::from(error)
                    })
                    .and_then(|text| parse(&text));
                match attempt {
                    Ok(answer) => return Some((model.clone(), answer)),
                    Err(error) => {
                        let message = format!("{error:#}");
                        // It thought until its tokens ran out: ask the same model again,
                        // this time to answer straight away.
                        let thought_too_long = reasoning && message.contains("token limit");
                        tracing::warn!(
                            what,
                            %model,
                            reasoning,
                            error = %message,
                            "review model could not read the work"
                        );
                        if thought_too_long {
                            reasoning = false;
                            continue;
                        }
                        break;
                    }
                }
            }
        }
        None
    }

    /// One more question to the editor when the settled pairs contradict the genders of the
    /// characters or each other. Only what it says about the characters is taken.
    #[allow(clippy::too_many_arguments)]
    async fn repair_addresses(
        &self,
        config: &AnalysisConfig,
        providers: &ProvidersConfig,
        system: &str,
        bible: &mut Bible,
        digests: &[(usize, String)],
        total: usize,
        dead: &mut HashSet<String>,
    ) {
        let problems = bible.address_conflicts();
        if problems.is_empty() {
            return;
        }
        tracing::warn!(count = problems.len(), ?problems, "settled pairs contradict genders or each other");
        // The pages that name a character of a problem, up to the size of one review request.
        let names: Vec<String> = bible
            .characters
            .iter()
            .map(|character| character.name.to_lowercase())
            .filter(|name| {
                problems
                    .iter()
                    .any(|problem| problem.to_lowercase().contains(name.as_str()))
            })
            .collect();
        let budget = config.review_chars.max(2000);
        let mut used = 0;
        let mut pages: Vec<(usize, String)> = Vec::new();
        for (number, digest) in digests {
            let lower = digest.to_lowercase();
            if !names.iter().any(|name| lower.contains(name.as_str())) {
                continue;
            }
            let size = digest.chars().count();
            if used + size > budget && !pages.is_empty() {
                break;
            }
            used += size;
            pages.push((*number, digest.clone()));
        }
        if pages.is_empty() {
            return;
        }
        let user = repair_prompt(bible, &problems, &pages, total);
        let Some((model, answer)) = self
            .ask_review(config, providers, system, &user, "repair", dead)
            .await
        else {
            tracing::warn!("no model could answer for the pairs that cannot be right");
            return;
        };
        let characters = answer
            .characters
            .iter()
            .filter(|character| bible.review_character(character))
            .count();
        tracing::info!(
            %model,
            characters,
            remaining = bible.address_conflicts().len(),
            "pairs that cannot be right asked again"
        );
    }

    /// Reads the whole work again as text and settles the words for "I" and "you" (see the
    /// module). Changes `bible` and the notes in `results`; whatever fails leaves them as the
    /// first reading made them. Returns whether the notes were settled by the review.
    #[allow(clippy::too_many_arguments)]
    pub(super) async fn review_work(
        &self,
        config: &AnalysisConfig,
        providers: &ProvidersConfig,
        language: Language,
        bible: &mut Bible,
        results: &mut [Option<PageAnalysis>],
        dead: &mut HashSet<String>,
        stop: &(impl Fn() -> bool + Send + Sync),
    ) -> bool {
        let digests: Vec<(usize, String)> = results
            .iter()
            .enumerate()
            .filter_map(|(number, page)| {
                page.as_ref()
                    .and_then(|page| page_digest(number, page))
                    .map(|digest| (number, digest))
            })
            .collect();
        if digests.is_empty() || bible.characters.is_empty() || config.review_models.is_empty() {
            return false;
        }
        let total = results.len();
        let system = system_prompt(language);
        let sizes: Vec<usize> = digests.iter().map(|(_, digest)| digest.chars().count()).collect();
        let parts = parts(&sizes, config.review_chars.max(2000));
        let mut exceptions: HashMap<usize, HashMap<usize, LineFix>> = HashMap::new();
        let mut reviewed = 0;

        for (part, range) in parts.iter().enumerate() {
            if stop() {
                break;
            }
            let pages = &digests[range.clone()];
            let user = user_prompt(bible, pages, total);
            let what = format!("part {} of {}", part + 1, parts.len());
            let Some((model, answer)) = self
                .ask_review(config, providers, &system, &user, &what, dead)
                .await
            else {
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
                let fix = LineFix {
                    speaker: clip(&change.speaker, FIELD_CHARS),
                    listener: clip(&change.listener, FIELD_CHARS),
                    self_word: clip(&change.self_word, 40),
                    you: clip(&change.you, 40),
                };
                if fix == LineFix::default() {
                    continue;
                }
                exceptions.entry(number).or_default().insert(change.id, fix);
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
            return false;
        }
        // A later part can change a pair an earlier one decided, and a word of the wrong sex
        // or a pair that does not fit its other direction is a mistake whoever made it: ask
        // once more about those before the notes are written.
        if language == Language::Vietnamese && !stop() {
            self.repair_addresses(config, providers, &system, bible, &digests, total, dead)
                .await;
        }
        // Once, at the end.
        let mut lines = 0;
        for (number, page) in results.iter_mut().enumerate() {
            if let Some(page) = page {
                lines += page.apply_review(bible, exceptions.get(&number));
            }
        }
        tracing::info!(lines, "notes of the pages settled by the review");
        true
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
                {"name": "narrator", "relationships": [{"with": "Rio", "self": "a", "you": "b"}]}
            ], "changes": [{"page": "1", "id": 1, "you": "em"}, {"page": 0, "id": 0, "you": "z"}]}"#,
        )
        .unwrap()
    }

    #[test]
    fn the_editor_overwrites_a_pair_through_an_alias_and_ignores_the_narrator() {
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
        exceptions.insert(1, LineFix { you: "em".to_owned(), ..LineFix::default() });
        let changed = analysis.apply_review(&bible, Some(&exceptions));
        // Line 0 takes the pair ("tôi", "chị"); line 1 keeps its own self word and takes the
        // exception's "you" (its words do not change, but its listener "Rio-san" is written as
        // "Rio"); the thought and the narration are not touched.
        assert_eq!(changed, 2);
        assert_eq!(analysis.lines[1].listener, "Rio");
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
        assert_eq!(
            answer.changes[0],
            Change { page: 1, id: 1, you: "em".to_owned(), ..Change::default() }
        );
        assert!(parse("no json here").is_err());
        assert_eq!(parse("{}").unwrap(), ReviewOutput::default());
    }

    #[test]
    fn a_line_the_readers_gave_to_a_role_gets_the_right_speaker_and_the_pair_of_its_direction() {
        let mut bible = Bible::default();
        let mut analysis = page(FIRST, 4, &mut bible);
        analysis.lines[1].speaker = "visitor".to_owned();
        analysis.lines[1].listener = "nurse".to_owned();
        analysis.lines[1].self_word = "xxx".to_owned();
        analysis.lines[1].you = "yyy".to_owned();
        let mut fixes = HashMap::new();
        fixes.insert(
            1,
            LineFix {
                speaker: "Sasaki".to_owned(),
                listener: "Rio-san".to_owned(),
                ..LineFix::default()
            },
        );
        analysis.apply_review(&bible, Some(&fixes));
        let line = &analysis.lines[1];
        assert_eq!((line.speaker.as_str(), line.listener.as_str()), ("Sasaki", "Rio"));
        assert_eq!((line.self_word.as_str(), line.you.as_str()), ("tôi", "cô"));
        assert_eq!(line.relation, "colleague");
    }

    #[test]
    fn a_person_the_readers_missed_is_added_with_their_pairs() {
        let mut bible = Bible::default();
        page(FIRST, 4, &mut bible);
        let answer = parse(
            r#"{"characters": [{"name": "Yuu", "gender": "man", "relationships":
                [{"with": "Rio", "self": "anh", "you": "em"}]}], "changes": []}"#,
        )
        .unwrap();
        assert!(bible.review_character(&answer.characters[0]));
        let pair = bible.pair("Yuu", "Rio-san").unwrap();
        assert_eq!((pair.self_word.as_str(), pair.you.as_str()), ("anh", "em"));
        assert!(!bible.review_character(&answer.characters[0]), "known now, nothing to change");
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
    fn the_repair_question_carries_the_problems_and_the_pages() {
        let mut bible = Bible::default();
        let analysis = page(FIRST, 4, &mut bible);
        let digest = page_digest(0, &analysis).unwrap();
        let prompt = repair_prompt(&bible, &["Rio -> Sasaki: anh - chị".to_owned()], &[(0, digest)], 9);
        assert!(prompt.starts_with("The work has 9 pages."));
        assert!(prompt.contains("- Rio -> Sasaki: anh - chị\n"));
        assert!(prompt.contains("0 speech Sasaki -> Rio"));
        assert!(prompt.ends_with("Answer with the JSON object."));
    }

    #[test]
    fn the_defaults_read_twice_ask_deepseek_first_and_think_only_in_the_second_reading() {
        let config = AnalysisConfig::default();
        assert!(config.review && config.review_reasoning && !config.reasoning);
        assert!(config.review_max_tokens > config.max_tokens);
        assert!(config.models.iter().all(|model| model.reasoning.is_none()));
        assert_eq!(config.review_models.len(), 2);
        assert_eq!(config.review_models[0].model, "deepseek-ai/deepseek-v4.1-flash");
        assert_eq!(config.review_models[0].reasoning, Some(false), "DeepSeek does not think in the review");
        assert_eq!(config.review_models[1].reasoning, None, "the fallback follows review_reasoning");
    }
}
