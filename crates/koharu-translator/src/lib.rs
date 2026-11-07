//! Translation through local and hosted providers.

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
pub use language::Language;
pub use model::{GenerationConfig, Model, ModelSelection, Quantization};
pub(crate) use model::{ModelGeneration, QuantizationDefinition, display_name};
pub use provider::{Provider, ProviderConfig, ProvidersConfig};

/// How many recent translated lines are replayed as context for the next page, so
/// pronouns, names and tone stay the same from page to page.
const MEMORY_PAIRS: usize = 10;
/// Each remembered line is cut to this many characters to keep the prompt short.
const MEMORY_SEGMENT_CHARS: usize = 80;
/// Remembered lines are dropped after this long without a translation (a new chapter).
const MEMORY_TTL: Duration = Duration::from_secs(30 * 60);

/// The glossary is written once per run, after OCR has read every page and before the first
/// page is translated. The glossary model reads this many source lines at a time...
const GLOSSARY_CHUNK_LINES: usize = 40;
/// ...in at most this many reads. This only guards against a runaway work: the whole work is
/// read unless it has more lines than `GLOSSARY_CHUNK_LINES * GLOSSARY_MAX_CHUNKS`.
const GLOSSARY_MAX_CHUNKS: usize = 200;
/// Entries kept in the glossary; the oldest are dropped first.
const GLOSSARY_MAX_ENTRIES: usize = 600;
/// Each glossary field and each shown line is cut to this many characters.
const GLOSSARY_FIELD_CHARS: usize = 120;
const GLOSSARY_LINE_CHARS: usize = 120;
/// The glossary is complete, but a page is translated with only the entries that matter to
/// it: at most this many, most useful kinds first.
const GLOSSARY_BLOCK_MAX_ENTRIES: usize = 50;
/// Character profiles carry the visual identities and relationships used to resolve the
/// speakers on a page. Prefer profiles whose names occur in the current/recent dialogue.
const GLOSSARY_CHARACTER_MAX_ENTRIES: usize = 16;
/// Address pairs are useful even when a page never spells either character's name: the page
/// image lets the translator match the visible speakers to the character profiles.
const GLOSSARY_ADDRESS_MAX_ENTRIES: usize = 16;
/// Source text of this many previous pages also counts when picking a page's entries, so
/// characters who are only implied on the page keep their pronouns.
const MEMORY_PAGES: usize = 3;
const MEMORY_PAGE_CHARS: usize = 1500;

/// Names, terms and image-grounded directional address pairs remembered for one work.
#[derive(Default)]
struct Glossary {
    language: Option<Language>,
    project: Option<String>,
    entries: Vec<GlossaryEntry>,
}

impl Glossary {
    /// Starts over for another work or another target language; the same work keeps what it
    /// has, so a later run extends the glossary instead of rebuilding it.
    fn begin(&mut self, project: &str, language: Language) {
        if self.language != Some(language) || self.project.as_deref() != Some(project) {
            self.entries.clear();
        }
        self.language = Some(language);
        self.project = Some(project.to_owned());
    }

    /// Adds entries that are not known yet. Existing entries win, so a rendering never
    /// flips once a high-confidence directional choice has been stored. Address identity is
    /// compared by normalized speaker/listener names, not by the arrow glyph or honorific.
    fn merge(&mut self, entries: Vec<GlossaryEntry>) {
        for entry in entries.into_iter().filter_map(clean_entry) {
            let known = self.entries.iter().any(|known| {
                if known.kind != entry.kind {
                    return false;
                }
                if entry.kind == "address" {
                    match (address_pair_key(&known.source), address_pair_key(&entry.source)) {
                        (Some(known_key), Some(incoming_key)) => known_key == incoming_key,
                        _ => known.source.eq_ignore_ascii_case(&entry.source),
                    }
                } else {
                    known.source.eq_ignore_ascii_case(&entry.source)
                }
            });
            if !known {
                self.entries.push(entry);
            }
        }
        let excess = self.entries.len().saturating_sub(GLOSSARY_MAX_ENTRIES);
        self.entries.drain(..excess);
    }
}

/// `style` entries always apply; character profiles and address pairs are selected together
/// so the translator can match visible speakers to the right directional pronoun pair.
const GLOSSARY_KINDS: [&str; 5] = ["style", "character", "address", "name", "term"];

fn address_names(source: &str) -> Option<(&str, &str)> {
    source
        .split_once("->")
        .or_else(|| source.split_once('→'))
        .map(|(speaker, listener)| (speaker.trim(), listener.trim()))
}

/// A canonical key for a directional address pair. Arrow style and honorific variants do
/// not create a second rule for the same speaker/listener direction.
fn address_pair_key(source: &str) -> Option<String> {
    let (speaker, listener) = address_names(source)?;
    let speaker = character_key(speaker);
    let listener = character_key(listener);
    (!speaker.is_empty() && !listener.is_empty())
        .then(|| format!("{speaker} -> {listener}"))
}

/// Treat common Japanese/Korean honorific suffixes as part of a name variant, not as a
/// different character. Do not deduplicate by profile text: unrelated characters often share
/// the same age, gender, and role.
fn character_key(source: &str) -> String {
    let source = source.trim().to_lowercase();
    for suffix in [
        "さん", "さま", "様", "ちゃん", "くん", "君", "先生", "先輩", "씨", "님",
    ] {
        if let Some(name) = source.strip_suffix(suffix) {
            let name = name.trim();
            if !name.is_empty() {
                return name.to_owned();
            }
        }
    }
    source
}

/// Selects a compact page-specific glossary. Address pairs are included not only when a
/// character name appears in OCR, but also when both endpoints have selected character
/// profiles: speech bubbles normally contain dialogue, not the speakers' names.
fn relevant_entries(entries: &[GlossaryEntry], corpus: &str) -> Vec<GlossaryEntry> {
    let corpus = corpus.to_lowercase();
    let mentioned = |text: &str| {
        let text = text.trim().to_lowercase();
        !text.is_empty() && corpus.contains(&text)
    };

    // Put explicitly mentioned characters first, then fill the remaining slots with the
    // work's other known profiles. The image can identify them even when their names are
    // absent from OCR.
    let mut seen_characters = std::collections::HashSet::new();
    let mut characters: Vec<&GlossaryEntry> = entries
        .iter()
        .filter(|entry| {
            entry.kind == "character" && seen_characters.insert(character_key(&entry.source))
        })
        .collect();
    characters.sort_by_key(|entry| !mentioned(&entry.source));
    characters.truncate(GLOSSARY_CHARACTER_MAX_ENTRIES);
    let character_names: std::collections::HashSet<String> = characters
        .iter()
        .map(|entry| character_key(&entry.source))
        .collect();

    // Prefer pairs grounded by a name on this page/recent context. Also include pairs for
    // two profiled characters, because the page image—not OCR text—identifies the speakers.
    let mut addresses: Vec<(&GlossaryEntry, u8)> = entries
        .iter()
        .filter(|entry| entry.kind == "address")
        .filter_map(|entry| {
            let (speaker, listener) = address_names(&entry.source)?;
            let name_mentioned = mentioned(speaker) || mentioned(listener);
            let both_profiled = character_names.contains(&character_key(speaker))
                && character_names.contains(&character_key(listener));
            (name_mentioned || both_profiled)
                .then_some((entry, if name_mentioned { 0 } else { 1 }))
        })
        .collect();
    addresses.sort_by_key(|(_, priority)| *priority);
    addresses.truncate(GLOSSARY_ADDRESS_MAX_ENTRIES);

    let mut selected: Vec<&GlossaryEntry> = entries
        .iter()
        .filter(|entry| entry.kind == "style")
        .collect();
    selected.extend(characters);
    selected.extend(addresses.into_iter().map(|(entry, _)| entry));
    selected.extend(entries.iter().filter(|entry| {
        matches!(entry.kind.as_str(), "name" | "term") && mentioned(&entry.source)
    }));

    selected.sort_by_key(|entry| {
        GLOSSARY_KINDS
            .iter()
            .position(|kind| *kind == entry.kind)
            .unwrap_or(GLOSSARY_KINDS.len())
    });
    selected.truncate(GLOSSARY_BLOCK_MAX_ENTRIES);
    selected.into_iter().cloned().collect()
}

fn address_entries_from_observations(
    observations: Vec<prompt::AddressObservation>,
    entries: &[GlossaryEntry],
    segment_count: usize,
) -> Vec<GlossaryEntry> {
    let characters: Vec<&GlossaryEntry> = entries
        .iter()
        .filter(|entry| entry.kind == "character")
        .collect();
    if characters.len() < 2 {
        return Vec::new();
    }

    let mut accepted = std::collections::HashMap::<String, GlossaryEntry>::new();
    let mut conflicted = std::collections::HashSet::<String>::new();
    for observation in observations {
        let speaker = observation.speaker.trim();
        let listener = observation.listener.trim();
        let first_person = observation.first_person.trim();
        let second_person = observation.second_person.trim();
        let evidence = observation.evidence.trim();

        if observation.segment_id >= segment_count
            || !observation.confidence.is_finite()
            || !(0.85..=1.0).contains(&observation.confidence)
            || evidence.chars().count() < 8
            || evidence.chars().count() > 240
            || first_person.is_empty()
            || second_person.is_empty()
            || first_person.chars().count() > 24
            || second_person.chars().count() > 24
            || first_person.contains('\n')
            || first_person.contains('\r')
            || second_person.contains('\n')
            || second_person.contains('\r')
            || first_person.contains("->")
            || second_person.contains("->")
            || first_person.contains('|')
            || second_person.contains('|')
            || first_person.contains('/')
            || second_person.contains('/')
            || first_person.contains('?')
            || second_person.contains('?')
        {
            continue;
        }

        let speaker_key = character_key(speaker);
        let listener_key = character_key(listener);
        if speaker_key.is_empty() || listener_key.is_empty() || speaker_key == listener_key {
            continue;
        }

        // The model must refer to the exact character roster from the glossary. Names that
        // merely look plausible, visual descriptions, and unknown addressees are not saved.
        let Some(speaker_entry) = characters
            .iter()
            .find(|entry| character_key(&entry.source) == speaker_key)
        else {
            continue;
        };
        let Some(listener_entry) = characters
            .iter()
            .find(|entry| character_key(&entry.source) == listener_key)
        else {
            continue;
        };

        let source = format!("{} -> {}", speaker_entry.source.trim(), listener_entry.source.trim());
        let target = format!("{first_person} - {second_person}");
        let Some(key) = address_pair_key(&source) else {
            continue;
        };
        if conflicted.contains(&key) {
            continue;
        }
        let entry = GlossaryEntry {
            kind: "address".to_owned(),
            source,
            target,
        };

        match accepted.get(&key) {
            Some(existing) if existing.target == entry.target => {}
            Some(_) => {
                // Conflicting choices for the same direction on one page are ambiguous;
                // drop that pair rather than letting the last segment win.
                accepted.remove(&key);
                conflicted.insert(key);
            }
            None => {
                accepted.insert(key, entry);
            }
        }
    }

    let mut result: Vec<GlossaryEntry> = accepted.into_values().collect();
    result.sort_by(|left, right| left.source.cmp(&right.source));
    result
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

struct TranslationMemory {
    language: Option<Language>,
    recent: VecDeque<TranslationContext>,
    /// Source text of the last few pages, to pick the glossary entries of implied characters.
    recent_pages: VecDeque<String>,
    updated: Instant,
}

impl Default for TranslationMemory {
    fn default() -> Self {
        Self {
            language: None,
            recent: VecDeque::new(),
            recent_pages: VecDeque::new(),
            updated: Instant::now(),
        }
    }
}

#[derive(Clone)]
pub struct Translator {
    providers: koharu_config::Config<ProvidersConfig>,
    local: Arc<tokio::sync::Mutex<Option<LoadedLocal>>>,
    memory: Arc<std::sync::Mutex<TranslationMemory>>,
    glossary: Arc<std::sync::Mutex<Glossary>>,
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
            memory: Arc::new(std::sync::Mutex::new(TranslationMemory::default())),
            glossary: Arc::new(std::sync::Mutex::new(Glossary::default())),
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
        self.local
            .try_lock()
            .map(|mut loaded| loaded.take().is_some())
            .unwrap_or(false)
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

        // Fast single-pass Vision: the marked page image is sent once with the translation
        // request. The translation prompt already instructs the model to map each numbered
        // bubble to its speaker/listener before choosing Vietnamese pronouns. Avoid a second
        // full multimodal inference here; on CPU it repeats expensive image encoding and
        // generates a separate, long visual JSON report for every page.

        let expected = request.segments.len();
        let mut address_observations = Vec::new();
        let translatable: Vec<usize> = request
            .segments
            .iter()
            .enumerate()
            .filter(|(_, segment)| loop_guard::needs_translation(segment))
            .map(|(index, _)| index)
            .collect();
        let translated = if translatable.len() == expected {
            let (translated, observations) =
                self.translate_segments(selection, generation, &request).await?;
            address_observations = observations;
            translated
        } else if translatable.is_empty() {
            request.segments.clone()
        } else {
            let mut subset = request.clone();
            // Removing non-translatable segments changes the IDs, while the marked source image
            // still contains the original IDs. Avoid misattributing speakers in this rare path.
            subset.image = None;
            subset.segments = translatable
                .iter()
                .map(|&index| request.segments[index].clone())
                .collect();
            let (partial, observations) = self
                .translate_segments(selection, generation, &subset)
                .await?;
            address_observations = observations;
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
        self.remember_addresses(address_observations, expected);
        self.remember(request.target_language, &request.segments, &translated);
        tracing::Span::current().record("outcome", "completed");
        Ok((provider_id, translated))
    }

    /// The part of the glossary that matters to this page, as a prompt block. The glossary
    /// holds everything decided for the work; a page only carries the entries about the
    /// names, terms and characters on it (or on the pages just before it).
    fn glossary_block(&self, language: Language, segments: &[String]) -> Option<String> {
        let glossary = self.glossary.lock().ok()?;
        if glossary.language != Some(language) || glossary.entries.is_empty() {
            return None;
        }
        let mut corpus = segments.join("\n");
        if let Ok(memory) = self.memory.lock() {
            for page in &memory.recent_pages {
                corpus.push('\n');
                corpus.push_str(page);
            }
        }
        let selected = relevant_entries(&glossary.entries, &corpus);
        if selected.is_empty() {
            return None;
        }
        Some(prompt::render_glossary(&selected))
    }

    /// Writes the glossary of a work from OCR text using the same selected local model that
    /// performs translation. This prevents a hidden model from influencing benchmark results.
    /// Only local models write glossaries; address pairs are learned later from page images.
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
        let new_work = if let Ok(mut glossary) = self.glossary.lock() {
            let changed = glossary.language != Some(language)
                || glossary.project.as_deref() != Some(project);
            glossary.begin(project, language);
            changed
        } else {
            false
        };
        if new_work {
            if let Ok(mut memory) = self.memory.lock() {
                memory.language = Some(language);
                memory.recent.clear();
                memory.recent_pages.clear();
                memory.updated = Instant::now();
            }
        }
        if selection.provider != Provider::Local {
            return Ok(());
        }
        let lines = prepare_glossary_lines(lines);
        if lines.is_empty() {
            return Ok(());
        }
        let generation = generation.for_model(selection);
        // Use the same selected model for glossary and translation.
        // This makes model comparisons honest and avoids silently running Gemma in the
        // background when the user selected another model.
        let started = Instant::now();
        let result = self
            .write_glossary(selection, generation, language, &lines, &stop)
            .await;
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
        generation: GenerationConfig,
        language: Language,
        lines: &[String],
        stop: &(impl Fn() -> bool + Send + Sync),
    ) -> anyhow::Result<()> {
        let translator = self.local(selection).await?;
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
        Ok(())
    }

    /// Persists only high-confidence address observations that name two characters already
    /// present in this work's glossary. No additional model call is made for this update.
    fn remember_addresses(
        &self,
        observations: Vec<prompt::AddressObservation>,
        segment_count: usize,
    ) {
        if observations.is_empty() {
            return;
        }
        let Ok(mut glossary) = self.glossary.lock() else {
            return;
        };
        let accepted =
            address_entries_from_observations(observations, &glossary.entries, segment_count);
        if accepted.is_empty() {
            return;
        }
        for entry in &accepted {
            tracing::debug!(
                source = %entry.source,
                target = %entry.target,
                "learned image-grounded directional address rule"
            );
        }
        glossary.merge(accepted);
    }

    /// Lines remembered from the previous pages, if they are still fresh.
    fn recall(&self, language: Language) -> Vec<TranslationContext> {
        let Ok(mut memory) = self.memory.lock() else {
            return Vec::new();
        };
        if memory.language != Some(language) || memory.updated.elapsed() > MEMORY_TTL {
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
        if memory.language != Some(language) || memory.updated.elapsed() > MEMORY_TTL {
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
        memory.updated = Instant::now();
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
    }

    /// Translates every segment; what cannot be translated keeps its source text.
    async fn translate_segments(
        &self,
        selection: &ModelSelection,
        generation: GenerationConfig,
        request: &TranslationRequest,
    ) -> anyhow::Result<(Vec<String>, Vec<prompt::AddressObservation>)> {
        self.translate_guarded(selection, generation, request)
            .await
            .map(|(translated, _kept, observations)| (translated, observations))
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
    ) -> anyhow::Result<(Vec<String>, Vec<usize>, Vec<prompt::AddressObservation>)> {
        let reasoning = generation.reasoning == Some(true);
        let total = request.segments.len();
        let (mut translated, redo, address_observations) =
            match self.dispatch(selection, generation, request).await {
            Ok((translated, observations)) => {
                if translated.len() != total {
                    return Ok((translated, Vec::new(), observations));
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
                let observations = observations
                    .into_iter()
                    .filter(|item| !redo.contains(&item.segment_id))
                    .collect();
                (translated, redo, observations)
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
                (translated, redo, Vec::new())
            }
        };
        if redo.is_empty() {
            return Ok((translated, redo, address_observations));
        }
        if !reasoning {
            tracing::warn!(
                kept = redo.len(),
                total,
                sources = ?preview(&request.segments, &redo),
                "segments could not be translated; keeping source text"
            );
            return Ok((translated, redo, address_observations));
        }

        tracing::warn!(
            looping = redo.len(),
            total,
            sources = ?preview(&request.segments, &redo),
            "reasoning pass produced looping segments; translating them again without reasoning"
        );
        let mut retry = request.clone();
        if selection.provider == Provider::Local {
            // Encoding the page image on CPU takes minutes per call, so the local retry is
            // text only.
            retry.image = None;
        }
        retry.segments = redo
            .iter()
            .map(|&index| request.segments[index].clone())
            .collect();
        let mut failed: Vec<usize> = Vec::new();
        let without_reasoning = GenerationConfig {
            reasoning: Some(false),
            ..generation
        };
        let retried = match self.dispatch(selection, without_reasoning, &retry).await {
            Ok((retried, _retry_observations)) => {
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
        Ok((translated, failed, address_observations))
    }

    async fn dispatch(
        &self,
        selection: &ModelSelection,
        generation: GenerationConfig,
        request: &TranslationRequest,
    ) -> anyhow::Result<(Vec<String>, Vec<prompt::AddressObservation>)> {
        if selection.provider == Provider::Local {
            Ok(self
                .local(selection)
                .await?
                .translate(request.clone(), generation)
                .await?)
        } else {
            let providers = self.providers.read()?.clone();
            let translated =
                remote::translate(&self.client, &providers, selection, &generation, request).await?;
            Ok((translated, Vec::new()))
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
        // Character profiles precede their directional address pairs. A pair is included
        // when one endpoint is mentioned in the recent dialogue, even if the other is not.
        assert_eq!(kinds, ["style", "character", "address", "name", "term"]);
        assert!(selected.iter().all(|entry| entry.source != "山田 → 佐藤"));
        assert!(relevant_entries(&entries, "")
            .iter()
            .all(|entry| entry.kind == "style" || entry.kind == "character"));
    }

    #[test]
    fn address_pairs_are_available_when_names_are_absent_from_page_ocr() {
        let entry = |kind: &str, source: &str, target: &str| GlossaryEntry {
            kind: kind.to_owned(),
            source: source.to_owned(),
            target: target.to_owned(),
        };
        let entries = vec![
            entry("character", "田中", "nam, 17 tuổi, học sinh"),
            entry("character", "鈴木", "nữ, 17 tuổi, học sinh"),
            entry("character", "山田", "nam, 30 tuổi"),
            entry("character", "佐藤", "nữ, 30 tuổi"),
            entry("address", "田中さん -> 鈴木くん", "tớ - cậu"),
            entry("address", "山田 -> 佐藤", "tôi - cô"),
        ];

        let selected = relevant_entries(&entries, "本当に？");
        let address_sources: Vec<&str> = selected
            .iter()
            .filter(|entry| entry.kind == "address")
            .map(|entry| entry.source.as_str())
            .collect();

        assert_eq!(address_sources, ["田中さん -> 鈴木くん", "山田 -> 佐藤"]);
    }

    #[test]
    fn address_pairs_outside_the_selected_cast_are_not_added_without_a_name_hit() {
        let entry = |kind: &str, source: &str, target: &str| GlossaryEntry {
            kind: kind.to_owned(),
            source: source.to_owned(),
            target: target.to_owned(),
        };
        let entries = vec![
            entry("character", "田中", "nam, 17 tuổi"),
            entry("character", "鈴木", "nữ, 17 tuổi"),
            entry("character", "山田", "nam, 30 tuổi"),
            entry("address", "田中 -> 鈴木", "tớ - cậu"),
            entry("address", "山田 -> 佐藤", "tôi - cô"),
        ];

        let selected = relevant_entries(&entries, "本当に？");
        assert!(selected.iter().any(|entry| entry.source == "田中 -> 鈴木"));
        assert!(!selected.iter().any(|entry| entry.source == "山田 -> 佐藤"));
    }

    #[test]
    fn address_merge_normalizes_arrow_and_honorific_variants() {
        let entry = |source: &str, target: &str| GlossaryEntry {
            kind: "address".to_owned(),
            source: source.to_owned(),
            target: target.to_owned(),
        };
        let mut glossary = Glossary::default();
        glossary.merge(vec![entry("田中さん -> 鈴木くん", "tôi - cậu")]);
        glossary.merge(vec![entry("田中 → 鈴木", "anh - em")]);

        assert_eq!(glossary.entries.len(), 1);
        assert_eq!(glossary.entries[0].target, "tôi - cậu");
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
        glossary.begin("project-a", Language::Vietnamese);
        glossary.merge(vec![entry]);
        glossary.begin("project-a", Language::Vietnamese);
        assert_eq!(glossary.entries.len(), 1);
        glossary.begin("project-b", Language::Vietnamese);
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
    #[test]
    fn address_observations_require_known_distinct_characters_and_high_confidence() {
        let entry = |kind: &str, source: &str, target: &str| GlossaryEntry {
            kind: kind.to_owned(),
            source: source.to_owned(),
            target: target.to_owned(),
        };
        let characters = vec![
            entry("character", "田中", "nam, học sinh"),
            entry("character", "鈴木", "nữ, học sinh"),
        ];
        let observation = |speaker: &str, listener: &str, confidence: f32, second: &str| {
            prompt::AddressObservation {
                segment_id: 0,
                speaker: speaker.to_owned(),
                listener: listener.to_owned(),
                first_person: "tôi".to_owned(),
                second_person: second.to_owned(),
                confidence,
                evidence: "bong bóng có đuôi hướng về Tanaka và có từ お前".to_owned(),
            }
        };
        let accepted = address_entries_from_observations(
            vec![
                observation("田中", "鈴木", 0.93, "cậu"),
                observation("田中", "người lạ", 0.99, "anh"),
                observation("鈴木", "田中", 0.70, "anh"),
                observation("田中", "田中", 0.99, "mày"),
            ],
            &characters,
            1,
        );
        assert_eq!(accepted.len(), 1);
        assert_eq!(accepted[0].source, "田中 -> 鈴木");
        assert_eq!(accepted[0].target, "tôi - cậu");
    }

    #[test]
    fn conflicting_address_observations_on_one_page_are_discarded() {
        let characters = vec![
            GlossaryEntry { kind: "character".into(), source: "A".into(), target: "profile".into() },
            GlossaryEntry { kind: "character".into(), source: "B".into(), target: "profile".into() },
        ];
        let observation = |second: &str| prompt::AddressObservation {
            segment_id: 0,
            speaker: "A".into(),
            listener: "B".into(),
            first_person: "tôi".into(),
            second_person: second.into(),
            confidence: 0.95,
            evidence: "có dấu hiệu xưng hô rõ ràng trong câu".into(),
        };
        let accepted = address_entries_from_observations(
            vec![observation("cậu"), observation("anh")],
            &characters,
            1,
        );
        assert!(accepted.is_empty());
    }

}
