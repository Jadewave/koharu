//! Translation through local and hosted providers.

mod address;
mod backend;
mod error;
mod language;
mod local;
mod loop_guard;
mod model;
mod prompt;
mod provider;
mod remote;

use std::{
    collections::VecDeque,
    sync::Arc,
    time::{Duration, Instant},
};

use koharu_ml::Device;

use error::{Error, Result};
use local::LocalTranslator;

pub use backend::{GlossaryEntry, TranslationContext, TranslationRequest};

/// Whether a segment has anything to translate: sound effects made only of punctuation or
/// placeholder glyphs do not, and are not sent to the model.
#[must_use]
pub fn needs_translation(text: &str) -> bool {
    loop_guard::needs_translation(text)
}
pub use language::Language;
pub use model::{GenerationConfig, Model, ModelSelection, Quantization};
pub(crate) use model::{ModelGeneration, QuantizationDefinition, display_name};
pub use provider::{Provider, ProviderConfig, ProvidersConfig};

/// The model that writes the glossary (names, terms, character profiles) every other model follows.
const GLOSSARY_MODEL: &str = "gemma4-26b-a4b-uncensored";

/// How many recent translated lines are replayed as context for the next page, so
/// pronouns, names and tone stay the same from page to page.
const MEMORY_PAIRS: usize = 10;
/// Each remembered line is cut to this many characters to keep the prompt short.
const MEMORY_SEGMENT_CHARS: usize = 80;

/// The glossary is written once per run, after OCR has read every page and before the first
/// page is translated. The glossary model reads this many source lines at a time...
const GLOSSARY_CHUNK_LINES: usize = 40;
/// ...in at most this many reads. This only guards against a runaway work: the whole work is
/// read unless it has more lines than `GLOSSARY_CHUNK_LINES * GLOSSARY_MAX_CHUNKS`.
const GLOSSARY_MAX_CHUNKS: usize = 200;
/// Entries kept in the glossary; the oldest are dropped first.
const GLOSSARY_MAX_ENTRIES: usize = 600;
/// Each glossary field and each shown line is cut to this many characters.
const GLOSSARY_FIELD_CHARS: usize = 240;
const GLOSSARY_LINE_CHARS: usize = 120;
/// The glossary is complete, but a page is translated with only the entries that matter to
/// it: at most this many, most useful kinds first.
const GLOSSARY_BLOCK_MAX_ENTRIES: usize = 50;
/// Character profiles carry the ages and relationships that decide how people address each
/// other, so every page gets them, not only the pages that happen to repeat a name. At most
/// this many.
const GLOSSARY_CHARACTER_MAX_ENTRIES: usize = 16;
/// Address pairs that were given (not guessed) go to every page. At most this many.
const GLOSSARY_ADDRESS_MAX_ENTRIES: usize = 24;
/// Asked of the model when a translation opened with a subject its source line does not have.
const NO_SUBJECT_NOTE: &str = "Strict retry: each of these lines was translated before with a wrong subject: either one (a pronoun or a name) that the source line does not contain, or 'con' / 'cháu' for a speaker who is not shown to be a child or a younger relative of the listener. Translate them again and start no line with a pronoun or a name unless the source line itself contains that subject or form of address. When the source states the speaker (I, 私, 我, 나), use a neutral 'tôi' (or the pronoun that fits the speaker's age and relationship), never 'con' or 'cháu' unless the line shows a family tie. Keep the speaker and the pronoun pair of every line as decided before.";
/// Source text of this many previous pages also counts when picking a page's entries, so
/// characters who are only implied on the page keep their pronouns.
const MEMORY_PAGES: usize = 3;
const MEMORY_PAGE_CHARS: usize = 4500;

/// Names, terms and address pairs decided by the glossary model for one work.
#[derive(Default)]
struct Glossary {
    language: Option<Language>,
    project: Option<String>,
    entries: Vec<GlossaryEntry>,
}

impl Glossary {
    /// Starts over for another work or another target language; the same work keeps what it
    /// has, so a later run extends the glossary instead of rebuilding it.
    /// Returns whether it started over.
    fn begin(&mut self, project: &str, language: Language) -> bool {
        let restart = self.language != Some(language) || self.project.as_deref() != Some(project);
        if restart {
            self.entries.clear();
        }
        self.language = Some(language);
        self.project = Some(project.to_owned());
        restart
    }

    /// Adds entries that are not known yet. Existing entries win, so a rendering never
    /// flips once it has been chosen.
    fn merge(&mut self, entries: Vec<GlossaryEntry>) {
        for entry in entries.into_iter().filter_map(clean_entry) {
            let source = entry.source.to_lowercase();
            let known = self
                .entries
                .iter()
                .any(|known| known.kind == entry.kind && known.source.to_lowercase() == source);
            if !known {
                self.entries.push(entry);
            }
        }
        let excess = self.entries.len().saturating_sub(GLOSSARY_MAX_ENTRIES);
        self.entries.drain(..excess);
    }
}

/// `style` entries always apply; the others are ordered by how much they help a page.
const GLOSSARY_KINDS: [&str; 5] = ["style", "address", "character", "name", "term"];

/// The entries that matter to a page, most useful kinds first: those whose source text
/// appears in `corpus`, plus every `style`, `character` and `address` entry (a pair the model
/// chose earlier applies to the next pages even when neither name is spoken on them).
fn relevant_entries(entries: &[GlossaryEntry], corpus: &str) -> Vec<GlossaryEntry> {
    let corpus = corpus.to_lowercase();
    let mentioned = |text: &str| {
        let text = text.trim().to_lowercase();
        !text.is_empty() && corpus.contains(&text)
    };
    let mut selected: Vec<&GlossaryEntry> = entries
        .iter()
        .filter(|entry| match entry.kind.as_str() {
            "style" | "character" | "address" => true,
            _ => mentioned(&entry.source),
        })
        .collect();
    let (mut characters, mut addresses) = (0, 0);
    selected.retain(|entry| match entry.kind.as_str() {
        "character" => {
            characters += 1;
            characters <= GLOSSARY_CHARACTER_MAX_ENTRIES
        }
        "address" => {
            addresses += 1;
            addresses <= GLOSSARY_ADDRESS_MAX_ENTRIES
        }
        _ => true,
    });
    selected.sort_by_key(|entry| {
        GLOSSARY_KINDS
            .iter()
            .position(|kind| *kind == entry.kind)
            .unwrap_or(GLOSSARY_KINDS.len())
    });
    selected.truncate(GLOSSARY_BLOCK_MAX_ENTRIES);
    selected.into_iter().cloned().collect()
}

fn clean_entry(entry: GlossaryEntry) -> Option<GlossaryEntry> {
    let kind = entry.kind.trim().to_lowercase();
    let kind = if GLOSSARY_KINDS.contains(&kind.as_str()) {
        kind
    } else {
        "term".to_owned()
    };
    let source: String = entry.source.trim().chars().take(GLOSSARY_FIELD_CHARS).collect();
    let target: String = entry.target.trim().chars().take(GLOSSARY_FIELD_CHARS).collect();
    (!source.is_empty() && !target.is_empty()).then_some(GlossaryEntry {
        kind,
        source,
        target,
    })
}

/// Source lines worth showing to the glossary model: with text, trimmed, cut short, each
/// distinct line once, in reading order.
fn prepare_glossary_lines(lines: Vec<String>) -> Vec<String> {
    let mut seen = std::collections::HashSet::new();
    lines
        .into_iter()
        .filter(|line| loop_guard::needs_translation(line))
        .map(|line| {
            line.trim()
                .chars()
                .take(GLOSSARY_LINE_CHARS)
                .collect::<String>()
        })
        .filter(|line| seen.insert(line.clone()))
        .collect()
}

/// The reads of the glossary model: consecutive chunks, thinned out evenly when the work is
/// long so the phase stays bounded.
fn glossary_chunks(lines: &[String]) -> Vec<&[String]> {
    let chunks: Vec<&[String]> = lines.chunks(GLOSSARY_CHUNK_LINES).collect();
    if chunks.len() <= GLOSSARY_MAX_CHUNKS {
        return chunks;
    }
    (0..GLOSSARY_MAX_CHUNKS)
        .map(|index| chunks[index * chunks.len() / GLOSSARY_MAX_CHUNKS])
        .collect()
}

/// The character profiles for the address pass: one per distinct profile, so a character the
/// lines name in several ways is not counted twice, and no more than a page carries.
fn distinct_characters(entries: &[GlossaryEntry]) -> Vec<GlossaryEntry> {
    let mut seen = std::collections::HashSet::new();
    entries
        .iter()
        .filter(|entry| entry.kind == "character" && seen.insert(entry.target.clone()))
        .take(GLOSSARY_CHARACTER_MAX_ENTRIES)
        .cloned()
        .collect()
}

fn glossary_selection() -> ModelSelection {
    ModelSelection {
        provider: Provider::Local,
        model: Some(GLOSSARY_MODEL.to_owned()),
        quantization: None,
        vision: false,
        reasoning: false,
    }
}

struct TranslationMemory {
    language: Option<Language>,
    recent: VecDeque<TranslationContext>,
    /// Source text of the last few pages, to pick the glossary entries of implied characters.
    recent_pages: VecDeque<String>,
}

impl Default for TranslationMemory {
    fn default() -> Self {
        Self {
            language: None,
            recent: VecDeque::new(),
            recent_pages: VecDeque::new(),
        }
    }
}

#[derive(Clone)]
pub struct Translator {
    providers: koharu_config::Config<ProvidersConfig>,
    local: Arc<tokio::sync::Mutex<Option<LoadedLocal>>>,
    /// Holds the glossary model while another model translates, so neither evicts the other.
    glossary_model: Arc<tokio::sync::Mutex<Option<LoadedLocal>>>,
    memory: Arc<std::sync::Mutex<TranslationMemory>>,
    glossary: Arc<std::sync::Mutex<Glossary>>,
    /// The pronoun pair of every direction (who speaks to whom) that the translations confirmed.
    pairs: Arc<std::sync::Mutex<address::PairBook>>,
    client: reqwest::Client,
    device: Device,
}

struct LoadedLocal {
    model: Option<String>,
    quantization: Option<String>,
    translator: Arc<LocalTranslator>,
}

impl LoadedLocal {
    fn matches(&self, selection: &ModelSelection) -> bool {
        self.model == selection.model && self.quantization == selection.quantization
    }
}

impl Translator {
    pub fn from_config(
        device: Device,
        providers: koharu_config::Config<ProvidersConfig>,
    ) -> anyhow::Result<Self> {
        Ok(Self {
            providers,
            local: Arc::new(tokio::sync::Mutex::new(None)),
            glossary_model: Arc::new(tokio::sync::Mutex::new(None)),
            memory: Arc::new(std::sync::Mutex::new(TranslationMemory::default())),
            glossary: Arc::new(std::sync::Mutex::new(Glossary::default())),
            pairs: Arc::new(std::sync::Mutex::new(address::PairBook::default())),
            client: koharu_runtime::http_client()?,
            device,
        })
    }

    #[must_use]
    pub fn model(selection: &ModelSelection) -> &'static str {
        selection.provider.into()
    }

    #[must_use]
    pub fn supports_vision(selection: &ModelSelection, generation: &GenerationConfig) -> bool {
        generation.vision.unwrap_or(false)
            && (selection.provider != Provider::Local || local::supports_vision(selection))
    }

    #[must_use]
    pub fn loaded(&self, selection: &ModelSelection) -> bool {
        if selection.provider != Provider::Local {
            return true;
        }
        self.local
            .try_lock()
            .map(|loaded| {
                loaded
                    .as_ref()
                    .is_some_and(|loaded| loaded.matches(selection))
            })
            .unwrap_or(true)
    }

    pub fn unload(&self) -> bool {
        let fallback = self
            .glossary_model
            .try_lock()
            .map(|mut loaded| loaded.take().is_some())
            .unwrap_or(false);
        let main = self
            .local
            .try_lock()
            .map(|mut loaded| loaded.take().is_some())
            .unwrap_or(false);
        main || fallback
    }

    #[tracing::instrument(skip_all)]
    pub async fn load_model(&self, selection: &ModelSelection) -> anyhow::Result<()> {
        if selection.provider == Provider::Local {
            self.local(selection).await?;
        }
        Ok(())
    }

    #[tracing::instrument(
        target = "koharu_metrics",
        name = "model_run",
        skip_all,
        fields(
            stage = "translation",
            provider = %selection.provider,
            model = selection.model.as_deref().unwrap_or("provider_default"),
            target_language = request.target_language.tag(),
            outcome = tracing::field::Empty,
        ),
    )]
    pub async fn translate(
        &self,
        selection: &ModelSelection,
        generation: GenerationConfig,
        mut request: TranslationRequest,
    ) -> anyhow::Result<(&'static str, Vec<String>)> {
        let _metric = tracing::info_span!(
            target: "koharu_metrics",
            "translation_request",
            provider = %selection.provider,
            model = selection.model.as_deref().unwrap_or("provider_default"),
            target_language = request.target_language.tag(),
        );
        let provider = selection.provider;
        let provider_id: &'static str = provider.into();
        if request.segments.is_empty() {
            tracing::Span::current().record("outcome", "skipped");
            return Ok((provider_id, request.segments));
        }

        let generation = generation.for_model(selection);

        if request.context.is_empty() {
            request.context = self.recall(request.target_language);
        }
        if let Some(block) = self.glossary_block(request.target_language, &request.segments) {
            request.instructions = Some(
                match request.instructions.as_deref().map(str::trim) {
                    Some(instructions) if !instructions.is_empty() => {
                        format!("{instructions}\n\n{block}")
                    }
                    _ => block,
                },
            );
        }

        if Self::supports_vision(selection, &generation) {
            request.prepare_image()?;
        } else {
            request.remove_image();
        }

        let expected = request.segments.len();
        let translatable: Vec<usize> = request
            .segments
            .iter()
            .enumerate()
            .filter(|(_, segment)| loop_guard::needs_translation(segment))
            .map(|(index, _)| index)
            .collect();
        let translated = if translatable.len() == expected {
            self.translate_segments(selection, generation, &request)
                .await?
        } else if translatable.is_empty() {
            request.segments.clone()
        } else {
            let subset = request.select(&translatable);
            let partial = self
                .translate_segments(selection, generation, &subset)
                .await?;
            if partial.len() != translatable.len() {
                return Err(Error::SegmentCount {
                    provider: provider_id,
                    expected: translatable.len(),
                    actual: partial.len(),
                }
                .into());
            }
            let mut merged = request.segments.clone();
            for (&index, text) in translatable.iter().zip(partial) {
                merged[index] = text;
            }
            merged
        };
        if translated.len() != expected {
            return Err(Error::SegmentCount {
                provider: provider_id,
                expected,
                actual: translated.len(),
            }
            .into());
        }
        let translated = if request.target_language == Language::Vietnamese
            && selection.provider == Provider::Local
        {
            self.with_consistent_pairs(selection, generation, &request, translated)
                .await
        } else {
            translated
        };
        let translated = if request.target_language == Language::Vietnamese {
            self.without_invented_subjects(selection, generation, &request, translated)
                .await
        } else {
            translated
        };
        self.remember(request.target_language, &request.segments, &translated);
        tracing::Span::current().record("outcome", "completed");
        Ok((provider_id, translated))
    }

    /// Keeps the pronoun pairs of the work consistent. For every line the model wrote who
    /// speaks to whom and the pair of pronouns. A line whose pair (or self pronoun) differs from
    /// the pair of its direction is translated again with that pair spelled out. The pairs the
    /// final translations show are remembered, and a pair with enough evidence is locked: it
    /// goes to every later page as a glossary `address` entry.
    async fn with_consistent_pairs(
        &self,
        selection: &ModelSelection,
        generation: GenerationConfig,
        request: &TranslationRequest,
        mut translated: Vec<String>,
    ) -> Vec<String> {
        let Ok(local) = self.local(selection).await else {
            return translated;
        };
        let speakers = local.take_speakers();
        let declare = |speakers: &std::collections::HashMap<String, String>, index: usize| {
            speakers
                .get(&request.segments[index])
                .and_then(|speaker| address::parse_declared(speaker))
        };
        let think = |speakers: &std::collections::HashMap<String, String>, index: usize| {
            speakers
                .get(&request.segments[index])
                .and_then(|speaker| address::parse_thought(speaker))
        };
        let mut declared: Vec<Option<address::Declared>> = (0..translated.len())
            .map(|index| declare(&speakers, index))
            .collect();
        let mut thoughts: Vec<Option<address::Thought>> = (0..translated.len())
            .map(|index| think(&speakers, index))
            .collect();
        if declared.iter().all(Option::is_none) {
            return translated;
        }
        let (pair_flagged, thought_flagged) = self
            .pairs
            .lock()
            .map(|book| {
                (
                    book.conflicts(&declared, &translated),
                    book.thought_conflicts(&thoughts, &declared, &translated),
                )
            })
            .unwrap_or_default();
        let mut flagged: Vec<usize> = pair_flagged
            .iter()
            .chain(&thought_flagged)
            .copied()
            .collect();
        flagged.sort_unstable();
        flagged.dedup();
        if !flagged.is_empty() {
            let note = self
                .pairs
                .lock()
                .map(|book| {
                    let mut notes: Vec<String> = Vec::new();
                    if !pair_flagged.is_empty() {
                        notes.push(book.note(&declared, &pair_flagged));
                    }
                    if !thought_flagged.is_empty() {
                        notes.push(book.thought_note(&thoughts, &declared, &thought_flagged));
                    }
                    notes.join("\n")
                })
                .unwrap_or_default();
            tracing::warn!(
                count = flagged.len(),
                sources = ?preview(&request.segments, &flagged),
                "lines use another pair of pronouns than the rest of their direction; translating them again"
            );
            let mut retry = request.select(&flagged);
            retry.drop_images();
            retry.instructions = Some(match retry.instructions.take() {
                Some(instructions) if !instructions.trim().is_empty() => {
                    format!("{instructions}\n\n{note}")
                }
                _ => note,
            });
            match self.translate_segments(selection, generation, &retry).await {
                Ok(retried) if retried.len() == flagged.len() => {
                    let fresh = local.take_speakers();
                    for (&index, text) in flagged.iter().zip(retried) {
                        translated[index] = text;
                        declared[index] = declare(&fresh, index);
                        thoughts[index] = think(&fresh, index);
                    }
                }
                Ok(_) => tracing::warn!(
                    "the pronoun retry returned the wrong number of lines; keeping the first"
                ),
                Err(error) => {
                    tracing::warn!(%error, "the pronoun retry failed; keeping the first translation");
                }
            }
            // A line that still disagrees is no evidence for any pair.
            let still = self
                .pairs
                .lock()
                .map(|book| book.conflicts(&declared, &translated))
                .unwrap_or_default();
            for index in still {
                declared[index] = None;
            }
        }
        if let Ok(mut book) = self.pairs.lock() {
            let locked = book.learn(&declared, &translated);
            if locked > 0 {
                tracing::info!(locked, pairs = ?book.entries(), "pronoun pairs locked for this work");
            }
        }
        translated
    }

    /// The names of the glossary: a line that holds one of them may rightly start with it.
    fn glossary_names(&self) -> Vec<String> {
        self.glossary
            .lock()
            .map(|glossary| {
                glossary
                    .entries
                    .iter()
                    .filter(|entry| entry.kind == "name" || entry.kind == "character")
                    .map(|entry| entry.source.clone())
                    .collect()
            })
            .unwrap_or_default()
    }

    /// Translates again the lines that open with a subject their source does not have, with
    /// the rule spelled out. A line that still has one afterwards keeps the new translation
    /// and is reported.
    async fn without_invented_subjects(
        &self,
        selection: &ModelSelection,
        generation: GenerationConfig,
        request: &TranslationRequest,
        mut translated: Vec<String>,
    ) -> Vec<String> {
        let names = self.glossary_names();
        let flagged: Vec<usize> = (0..translated.len())
            .filter(|&index| {
                loop_guard::needs_translation(&request.segments[index])
                    && prompt::invented_subject(
                        &request.segments[index],
                        &translated[index],
                        &names,
                    )
            })
            .collect();
        if flagged.is_empty() {
            return translated;
        }
        tracing::warn!(
            count = flagged.len(),
            sources = ?preview(&request.segments, &flagged),
            "translations open with a subject the source does not have; translating them again"
        );
        let mut retry = request.select(&flagged);
        if selection.provider == Provider::Local {
            retry.drop_images();
        }
        retry.instructions = Some(match retry.instructions.take() {
            Some(instructions) if !instructions.trim().is_empty() => {
                format!("{instructions}\n\n{NO_SUBJECT_NOTE}")
            }
            _ => NO_SUBJECT_NOTE.to_owned(),
        });
        match self.translate_segments(selection, generation, &retry).await {
            Ok(retried) if retried.len() == flagged.len() => {
                for (&index, text) in flagged.iter().zip(retried) {
                    translated[index] = text;
                }
                let still: Vec<usize> = flagged
                    .iter()
                    .copied()
                    .filter(|&index| {
                        prompt::invented_subject(
                            &request.segments[index],
                            &translated[index],
                            &names,
                        )
                    })
                    .collect();
                if !still.is_empty() {
                    tracing::warn!(
                        count = still.len(),
                        sources = ?preview(&request.segments, &still),
                        "lines still open with a subject the source does not have"
                    );
                }
            }
            Ok(_) => tracing::warn!("the second pass returned the wrong number of lines; keeping the first"),
            Err(error) => tracing::warn!(%error, "the second pass failed; keeping the first translation"),
        }
        translated
    }

    /// The part of the glossary that matters to this page, as a prompt block. The glossary
    /// holds everything decided for the work; a page only carries the entries about the
    /// names, terms and characters on it (or on the pages just before it).
    fn glossary_block(&self, language: Language, segments: &[String]) -> Option<String> {
        let mut selected = Vec::new();
        if let Ok(glossary) = self.glossary.lock()
            && glossary.language == Some(language)
            && !glossary.entries.is_empty()
        {
            let mut corpus = segments.join("\n");
            if let Ok(memory) = self.memory.lock() {
                for page in &memory.recent_pages {
                    corpus.push('\n');
                    corpus.push_str(page);
                }
            }
            selected = relevant_entries(&glossary.entries, &corpus);
        }
        // The pairs the translations confirmed go to every page, whatever the glossary says.
        if language == Language::Vietnamese
            && let Ok(pairs) = self.pairs.lock()
        {
            for (source, target) in pairs.entries().into_iter().take(GLOSSARY_ADDRESS_MAX_ENTRIES) {
                let known = selected.iter().any(|entry| {
                    entry.kind == "address" && entry.source.eq_ignore_ascii_case(&source)
                });
                if !known {
                    selected.push(GlossaryEntry {
                        kind: "address".to_owned(),
                        source,
                        target,
                    });
                }
            }
        }
        if selected.is_empty() {
            return None;
        }
        Some(prompt::render_glossary(&selected))
    }

    /// Writes the glossary of a work from the OCR text of all its pages: the glossary model
    /// reads the lines in chunks and decides the renderings of names and terms and the address pairs, then it is released so the translation model can take its place. Only local
    /// models write glossaries; any other provider translates without one.
    ///
    /// A failure leaves the glossary as it was and is reported to the caller, which is
    /// expected to translate anyway.
    pub async fn build_glossary(
        &self,
        selection: &ModelSelection,
        generation: GenerationConfig,
        language: Language,
        project: &str,
        lines: Vec<String>,
        stop: impl Fn() -> bool + Send + Sync,
    ) -> anyhow::Result<()> {
        let restarted = self
            .glossary
            .lock()
            .map(|mut glossary| glossary.begin(project, language))
            .unwrap_or(false);
        if restarted && let Ok(mut memory) = self.memory.lock() {
            // Another work (or language): its first page must not see the lines of the last.
            memory.recent.clear();
            memory.recent_pages.clear();
        }
        if restarted && let Ok(mut pairs) = self.pairs.lock() {
            pairs.clear();
        }
        if selection.provider != Provider::Local {
            return Ok(());
        }
        let lines = prepare_glossary_lines(lines);
        if lines.is_empty() {
            return Ok(());
        }
        let generation = generation.for_model(selection);
        let shared = selection.model.as_deref() == Some(GLOSSARY_MODEL);
        if !shared {
            // Make room: the translation model is loaded again once the glossary is done.
            *self.local.lock().await = None;
        }
        let started = Instant::now();
        let result = self
            .write_glossary(selection, shared, generation, language, &lines, &stop)
            .await;
        if !shared {
            *self.glossary_model.lock().await = None;
        }
        let entries = self
            .glossary
            .lock()
            .map(|glossary| glossary.entries.len())
            .unwrap_or(0);
        if let Ok(glossary) = self.glossary.lock() {
            for entry in &glossary.entries {
                // No parent span, so the line is printed at once instead of when the whole
                // translation job (which can take hours on a CPU) finishes.
                tracing::debug!(
                    parent: None,
                    kind = %entry.kind,
                    source = %entry.source,
                    target = %entry.target,
                    "glossary entry"
                );
            }
        }
        tracing::info!(
            parent: None,
            entries,
            lines = lines.len(),
            secs = started.elapsed().as_secs_f32(),
            "glossary phase finished"
        );
        result
    }

    async fn write_glossary(
        &self,
        selection: &ModelSelection,
        shared: bool,
        generation: GenerationConfig,
        language: Language,
        lines: &[String],
        stop: &(impl Fn() -> bool + Send + Sync),
    ) -> anyhow::Result<()> {
        let loaded = if shared {
            self.local(selection).await
        } else {
            self.load_into(&self.glossary_model, &glossary_selection())
                .await
        };
        let translator = loaded?;
        for chunk in glossary_chunks(lines) {
            if stop() {
                break;
            }
            let known = self
                .glossary
                .lock()
                .map(|glossary| relevant_entries(&glossary.entries, &chunk.join("\n")))
                .unwrap_or_default();
            match translator
                .extract_glossary(language, &known, chunk, generation)
                .await
            {
                Ok(entries) => {
                    if let Ok(mut glossary) = self.glossary.lock() {
                        glossary.merge(entries);
                    }
                }
                Err(error) => tracing::warn!(%error, "glossary chunk failed; continuing"),
            }
        }
        // A line of OCR text does not say who speaks it, so the pairs are decided from the
        // character profiles the chunks wrote, once the whole work has been read.
        if prompt::PROFILES && prompt::decides_address(language) && !stop() {
            let characters = self
                .glossary
                .lock()
                .map(|glossary| distinct_characters(&glossary.entries))
                .unwrap_or_default();
            if characters.len() >= 2 {
                match translator
                    .extract_addresses(language, &characters, generation)
                    .await
                {
                    Ok(entries) => {
                        if let Ok(mut glossary) = self.glossary.lock() {
                            glossary.merge(entries);
                        }
                    }
                    Err(error) => tracing::warn!(%error, "address pass failed; continuing"),
                }
            }
        }
        Ok(())
    }

    /// Lines remembered from the previous pages of the same work and target language. They do
    /// not expire: a work can take hours, and another work clears them when it begins.
    fn recall(&self, language: Language) -> Vec<TranslationContext> {
        let Ok(mut memory) = self.memory.lock() else {
            return Vec::new();
        };
        if memory.language != Some(language) {
            memory.recent.clear();
            memory.recent_pages.clear();
            return Vec::new();
        }
        memory.recent.iter().cloned().collect()
    }

    /// Remembers the lines that translated cleanly, newest last.
    fn remember(&self, language: Language, sources: &[String], translations: &[String]) {
        let Ok(mut memory) = self.memory.lock() else {
            return;
        };
        if memory.language != Some(language) {
            memory.recent.clear();
            memory.recent_pages.clear();
        }
        memory.language = Some(language);
        for (source, translation) in sources.iter().zip(translations) {
            if !loop_guard::needs_translation(source)
                || translation.trim() == source.trim()
                || loop_guard::is_looping(source, translation)
            {
                continue;
            }
            memory.recent.push_back(TranslationContext {
                source: source.chars().take(MEMORY_SEGMENT_CHARS).collect(),
                translation: translation.chars().take(MEMORY_SEGMENT_CHARS).collect(),
            });
        }
        while memory.recent.len() > MEMORY_PAIRS {
            memory.recent.pop_front();
        }
        let page: String = sources
            .iter()
            .filter(|source| loop_guard::needs_translation(source))
            .map(String::as_str)
            .collect::<Vec<_>>()
            .join("\n")
            .chars()
            .take(MEMORY_PAGE_CHARS)
            .collect();
        memory.recent_pages.push_back(page);
        while memory.recent_pages.len() > MEMORY_PAGES {
            memory.recent_pages.pop_front();
        }
    }

    /// Forgets the remembered lines and the glossary, e.g. when another work begins.
    pub fn forget(&self) {
        if let Ok(mut memory) = self.memory.lock() {
            memory.recent.clear();
            memory.recent_pages.clear();
        }
        if let Ok(mut glossary) = self.glossary.lock() {
            glossary.entries.clear();
        }
        if let Ok(mut pairs) = self.pairs.lock() {
            pairs.clear();
        }
    }

    /// Translates every segment; what cannot be translated keeps its source text.
    async fn translate_segments(
        &self,
        selection: &ModelSelection,
        generation: GenerationConfig,
        request: &TranslationRequest,
    ) -> anyhow::Result<Vec<String>> {
        self.translate_guarded(selection, generation, request)
            .await
            .map(|(translated, _kept)| translated)
    }

    /// Translates every segment with the main model. With reasoning enabled the first pass
    /// runs with reasoning;
    /// segments that come back looping (or never come back because the response was cut
    /// off) are translated again with reasoning disabled, and only those segments.
    /// Whatever still fails keeps its source text instead of failing the page; the indices of
    /// those segments are returned too.
    async fn translate_guarded(
        &self,
        selection: &ModelSelection,
        generation: GenerationConfig,
        request: &TranslationRequest,
    ) -> anyhow::Result<(Vec<String>, Vec<usize>)> {
        let reasoning = generation.reasoning == Some(true);
        let total = request.segments.len();
        let (mut translated, redo) = match self.dispatch(selection, generation, request).await {
            Ok(translated) => {
                if translated.len() != total {
                    return Ok((translated, Vec::new()));
                }
                let mut redo = if reasoning {
                    loop_guard::looping_segments(&request.segments, &translated)
                } else {
                    Vec::new()
                };
                for index in loop_guard::untranslated_segments(&request.segments, &translated) {
                    if !redo.contains(&index) {
                        redo.push(index);
                    }
                }
                redo.sort_unstable();
                (translated, redo)
            }
            Err(error) => {
                let recovered = invalid_output(&error).map(|output| {
                    prompt::salvage_translations(&output.response, &request.segments)
                });
                if !retryable(&error) || !(reasoning || recovered.is_some()) {
                    return Err(error);
                }
                let (translated, mut redo) =
                    recovered.unwrap_or_else(|| (request.segments.clone(), (0..total).collect()));
                for index in loop_guard::looping_segments(&request.segments, &translated) {
                    if !redo.contains(&index) {
                        redo.push(index);
                    }
                }
                redo.sort_unstable();
                tracing::warn!(
                    %error,
                    redo = redo.len(),
                    total,
                    "translation failed; unrecovered segments need another pass"
                );
                (translated, redo)
            }
        };
        if redo.is_empty() {
            return Ok((translated, redo));
        }
        if !reasoning {
            tracing::warn!(
                kept = redo.len(),
                total,
                sources = ?preview(&request.segments, &redo),
                "segments could not be translated; keeping source text"
            );
            return Ok((translated, redo));
        }

        tracing::warn!(
            looping = redo.len(),
            total,
            sources = ?preview(&request.segments, &redo),
            "reasoning pass produced looping segments; translating them again without reasoning"
        );
        let mut retry = request.select(&redo);
        if selection.provider == Provider::Local {
            // Encoding the page images on CPU takes minutes per call, so the local retry is
            // text only.
            retry.drop_images();
        }
        let mut failed: Vec<usize> = Vec::new();
        let without_reasoning = GenerationConfig {
            reasoning: Some(false),
            ..generation
        };
        let retried = match self.dispatch(selection, without_reasoning, &retry).await {
            Ok(retried) => {
                if retried.len() != redo.len() {
                    let provider: &'static str = selection.provider.into();
                    return Err(Error::SegmentCount {
                        provider,
                        expected: redo.len(),
                        actual: retried.len(),
                    }
                    .into());
                }
                retried
            }
            Err(error) => match invalid_output(&error).filter(|_| retryable(&error)) {
                Some(output) => {
                    let (partial, missing) =
                        prompt::salvage_translations(&output.response, &retry.segments);
                    failed.extend(missing.iter().map(|&position| redo[position]));
                    tracing::warn!(
                        %error,
                        kept = missing.len(),
                        "translation without reasoning failed; keeping source text for the rest"
                    );
                    partial
                }
                None => return Err(error),
            },
        };
        for (&index, text) in redo.iter().zip(retried) {
            translated[index] = text;
        }
        for index in loop_guard::untranslated_segments(&request.segments, &translated) {
            if !failed.contains(&index) {
                failed.push(index);
            }
        }
        let remaining = loop_guard::looping_segments(&request.segments, &translated);
        if !remaining.is_empty() {
            tracing::warn!(
                kept = remaining.len(),
                sources = ?preview(&request.segments, &remaining),
                "segments still loop without reasoning; keeping source text"
            );
            for &index in &remaining {
                translated[index] = request.segments[index].clone();
                if !failed.contains(&index) {
                    failed.push(index);
                }
            }
        }
        failed.sort_unstable();
        Ok((translated, failed))
    }

    async fn dispatch(
        &self,
        selection: &ModelSelection,
        generation: GenerationConfig,
        request: &TranslationRequest,
    ) -> anyhow::Result<Vec<String>> {
        if selection.provider == Provider::Local {
            Ok(self
                .local(selection)
                .await?
                .translate(request.clone(), generation)
                .await?)
        } else {
            let providers = self.providers.read()?.clone();
            Ok(remote::translate(&self.client, &providers, selection, &generation, request).await?)
        }
    }

    #[tracing::instrument(skip_all)]
    pub async fn models() -> anyhow::Result<Vec<Model>> {
        static CLIENT: tokio::sync::OnceCell<reqwest::Client> = tokio::sync::OnceCell::const_new();

        let providers = ProvidersConfig::load()?;
        let providers = providers.read()?.clone();
        let client = CLIENT
            .get_or_try_init(|| async {
                reqwest::Client::builder()
                    .timeout(Duration::from_secs(5))
                    .build()
            })
            .await?;
        let mut models = local::models();
        models.extend(remote::models(client, &providers).await);
        Ok(models)
    }

    async fn local(&self, selection: &ModelSelection) -> Result<Arc<LocalTranslator>> {
        self.load_into(&self.local, selection).await
    }

    async fn load_into(
        &self,
        slot: &tokio::sync::Mutex<Option<LoadedLocal>>,
        selection: &ModelSelection,
    ) -> Result<Arc<LocalTranslator>> {
        let mut loaded = slot.lock().await;
        if loaded
            .as_ref()
            .is_none_or(|loaded| !loaded.matches(selection))
        {
            *loaded = Some(LoadedLocal {
                model: selection.model.clone(),
                quantization: selection.quantization.clone(),
                translator: Arc::new(LocalTranslator::load(self.device.clone(), selection).await?),
            });
        }
        Ok(Arc::clone(
            &loaded
                .as_ref()
                .expect("local translator was loaded")
                .translator,
        ))
    }
}

/// The start of each listed segment, for log lines.
fn preview(segments: &[String], indices: &[usize]) -> Vec<String> {
    indices
        .iter()
        .map(|&index| segments[index].chars().take(24).collect())
        .collect()
}

/// Finds the raw response of an unparsable translation, however the error was wrapped.
fn invalid_output(error: &anyhow::Error) -> Option<&prompt::InvalidTranslationOutput> {
    error
        .downcast_ref::<prompt::InvalidTranslationOutput>()
        .or_else(|| match error.downcast_ref::<Error>() {
            Some(Error::Other(inner)) => inner.downcast_ref::<prompt::InvalidTranslationOutput>(),
            _ => None,
        })
}

/// Errors that retrying without reasoning cannot fix (quota, rejected request, language).
fn retryable(error: &anyhow::Error) -> bool {
    !matches!(
        error.downcast_ref::<Error>(),
        Some(
            Error::QuotaExceeded { .. }
                | Error::Api { .. }
                | Error::UnsupportedLanguage { .. }
                | Error::UnsupportedSourceLanguage { .. }
        )
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn glossary_keeps_the_first_rendering_and_caps_its_size() {
        let entry = |kind: &str, source: &str, target: &str| GlossaryEntry {
            kind: kind.to_owned(),
            source: source.to_owned(),
            target: target.to_owned(),
        };
        let mut glossary = Glossary::default();
        glossary.merge(vec![
            entry("name", "田中", "Tanaka"),
            entry("bogus", "霊力", "linh lực"),
            entry("character", "田中", "nam, 17 tuổi, hay nói thân mật"),
        ]);
        glossary.merge(vec![entry("name", "田中", "Tanaca"), entry("name", "", "x")]);
        assert_eq!(glossary.entries.len(), 3);
        assert_eq!(glossary.entries[0].target, "Tanaka");
        assert_eq!(glossary.entries[1].kind, "term");

        let extra = GLOSSARY_MAX_ENTRIES + 20;
        let many: Vec<GlossaryEntry> = (0..extra)
            .map(|index| entry("name", &format!("n{index}"), "x"))
            .collect();
        glossary.merge(many);
        assert_eq!(glossary.entries.len(), GLOSSARY_MAX_ENTRIES);
        let last = format!("n{}", extra - 1);
        assert_eq!(glossary.entries.last().map(|e| e.source.as_str()), Some(last.as_str()));
    }

    #[test]
    fn a_page_only_carries_the_entries_that_matter_to_it() {
        let entry = |kind: &str, source: &str, target: &str| GlossaryEntry {
            kind: kind.to_owned(),
            source: source.to_owned(),
            target: target.to_owned(),
        };
        let entries = vec![
            entry("term", "霊力", "linh lực"),
            entry("name", "田中", "Tanaka"),
            entry("name", "鈴木", "Suzuki"),
            entry("character", "鈴木", "nữ, 16 tuổi, lịch sự"),
            entry("style", "work", "học đường, thân mật"),
            entry("address", "田中 -> 鈴木", "tớ - cậu"),
            entry("address", "山田 → 佐藤", "anh - em"),
        ];
        let selected = relevant_entries(&entries, "田中: おはよう\n霊力が足りない");
        let kinds: Vec<&str> = selected.iter().map(|e| e.kind.as_str()).collect();
        // Character profiles and address pairs apply to every page, even when the names are
        // not on it.
        assert_eq!(
            kinds,
            ["style", "address", "address", "character", "name", "term"]
        );
        assert!(relevant_entries(&entries, "").iter().all(|entry| {
            matches!(entry.kind.as_str(), "style" | "character" | "address")
        }));
    }

    #[test]
    fn a_character_named_in_several_ways_is_counted_once_for_the_address_pass() {
        let entry = |kind: &str, source: &str, target: &str| GlossaryEntry {
            kind: kind.to_owned(),
            source: source.to_owned(),
            target: target.to_owned(),
        };
        let entries = vec![
            entry("character", "田中", "Nữ, 27 tuổi"),
            entry("character", "田中さん", "Nữ, 27 tuổi"),
            entry("name", "田中", "Tanaka"),
            entry("character", "店主", "Nam, 56 tuổi"),
        ];
        let characters = distinct_characters(&entries);
        let names: Vec<&str> = characters.iter().map(|entry| entry.source.as_str()).collect();
        assert_eq!(names, ["田中", "店主"]);
    }

    #[test]
    fn an_address_pair_keeps_its_first_wording() {
        let entry = |kind: &str, source: &str, target: &str| GlossaryEntry {
            kind: kind.to_owned(),
            source: source.to_owned(),
            target: target.to_owned(),
        };
        let mut glossary = Glossary::default();
        glossary.merge(vec![
            entry("address", "田中 -> 鈴木", "tớ - cậu"),
            entry("address", "田中 -> 鈴木", "anh - em"),
        ]);
        assert_eq!(glossary.entries, [entry("address", "田中 -> 鈴木", "tớ - cậu")]);
    }

    #[test]
    fn glossary_lines_are_cleaned_and_deduplicated() {
        let lines = vec![
            "おはよう".to_owned(),
            "…".to_owned(),
            " おはよう ".to_owned(),
            "やあ".to_owned(),
        ];
        assert_eq!(prepare_glossary_lines(lines), ["おはよう", "やあ"]);
    }

    #[test]
    fn long_works_are_sampled_in_a_bounded_number_of_chunks() {
        let short: Vec<String> = (0..100).map(|index| format!("line {index}")).collect();
        assert_eq!(glossary_chunks(&short).len(), 3);

        let work: Vec<String> = (0..2000).map(|index| format!("line {index}")).collect();
        assert_eq!(glossary_chunks(&work).len(), 50);

        let long: Vec<String> = (0..GLOSSARY_CHUNK_LINES * GLOSSARY_MAX_CHUNKS * 2)
            .map(|index| format!("line {index}"))
            .collect();
        let chunks = glossary_chunks(&long);
        assert_eq!(chunks.len(), GLOSSARY_MAX_CHUNKS);
        assert_eq!(chunks[0][0], "line 0");
        assert!(glossary_chunks(&[]).is_empty());
    }

    #[test]
    fn a_glossary_belongs_to_one_work_and_language() {
        let entry = GlossaryEntry {
            kind: "name".to_owned(),
            source: "田中".to_owned(),
            target: "Tanaka".to_owned(),
        };
        let mut glossary = Glossary::default();
        assert!(glossary.begin("project-a", Language::Vietnamese));
        glossary.merge(vec![entry]);
        assert!(!glossary.begin("project-a", Language::Vietnamese));
        assert_eq!(glossary.entries.len(), 1);
        assert!(glossary.begin("project-b", Language::Vietnamese));
        assert!(glossary.entries.is_empty());
    }

    #[test]
    fn invalid_output_is_found_through_the_error_wrapper() {
        let source = ["a".to_owned()];
        let response = r#"{"translations":[{"id":0,"text":"x"#;
        let failure = prompt::translations("test", response, &source).unwrap_err();
        let error = anyhow::Error::from(Error::Other(failure));
        assert!(invalid_output(&error).is_some());
        assert!(retryable(&error));

        let quota = anyhow::Error::from(Error::QuotaExceeded { provider: "test" });
        assert!(invalid_output(&quota).is_none());
        assert!(!retryable(&quota));
    }

    fn local_selection(model: &str) -> ModelSelection {
        ModelSelection {
            provider: Provider::Local,
            model: Some(model.to_owned()),
            quantization: None,
            vision: true,
            reasoning: true,
        }
    }

    #[test]
    fn local_vision_requires_capability_and_generation_setting() {
        assert!(Translator::supports_vision(
            &local_selection("gemma4-e2b-it"),
            &GenerationConfig {
                vision: Some(true),
                ..GenerationConfig::default()
            }
        ));
        assert!(!Translator::supports_vision(
            &local_selection("gemma4-e2b-it"),
            &GenerationConfig {
                vision: Some(false),
                ..GenerationConfig::default()
            }
        ));
        assert!(!Translator::supports_vision(
            &local_selection("lfm2.5-1.2b-instruct"),
            &GenerationConfig {
                vision: Some(true),
                ..GenerationConfig::default()
            }
        ));
    }
}
