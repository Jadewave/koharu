//! Analysis: before a run's first translation, a hosted model that reads
//! images reads the whole work in windows of pages, each page as an image with
//! its lettering cleaned away and its texts' boxes numbered, beside the OCR
//! lines. It answers with the characters, and for each page a summary and,
//! for each line, who speaks it and in what tone, and a corrected reading
//! where the OCR was wrong. The translation then holds the characters and
//! every page's summary in its system prompt, and each page's own notes in its
//! turn.
//!
//! It runs once, in the run's first translation, which waits for every page's
//! recognition and inpainting; a window that fails twice leaves its pages
//! without notes, and two failed windows in a row stop the analysis, so a bad
//! key doesn't cost every window.

mod page;
mod prompt;

use std::{
    collections::HashMap,
    future::IntoFuture,
    sync::{
        Arc,
        atomic::{AtomicU64, Ordering},
    },
};

use anyhow::{Context as _, Result, bail, ensure};
use koharu_llm::{
    Size,
    rig::{
        completion::{
            GenerationOptions, Message, Reasoning,
            message::{DocumentSourceKind, Image, ImageMediaType, UserContent},
        },
        driver::DynModel,
        operation::Completion,
    },
};
use koharu_scene::{PageId, Session};
use parking_lot::Mutex;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use tokio::sync::OnceCell;
use tokio_util::sync::CancellationToken;

use self::{page::Prepared, prompt::Reply};

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default)]
#[derive(specta::Type)]
pub struct AnalyzerConfig {
    /// A `vendor/model` reference to a hosted model that reads images, such
    /// as an OpenAI-compatible server's; empty turns the analysis off.
    pub model: String,
    /// The pages of one request, the context pages among them.
    pub pages: u32,
    /// The pages a window repeats from the one before, to follow the story.
    pub overlap: u32,
    /// The longest side of a page as the model sees it, in pixels.
    pub size: u32,
    /// The WebP quality of a page, from 1 to 100.
    pub quality: u8,
    /// How much the model reasons before it answers; its own default when
    /// unset.
    #[serde(with = "koharu_llm::reasoning")]
    #[specta(type = Option<String>)]
    pub reasoning: Option<Reasoning>,
}

impl Default for AnalyzerConfig {
    fn default() -> Self {
        Self {
            model: String::new(),
            pages: 12,
            overlap: 2,
            size: 1600,
            quality: 90,
            reasoning: None,
        }
    }
}

impl AnalyzerConfig {
    /// Whether a run analyzes the work before it translates.
    #[must_use]
    pub fn enabled(&self) -> bool {
        !self.model.trim().is_empty()
    }
}

/// What the analyzer made of a work.
pub(crate) struct Analysis {
    id: u64,
    cast: Vec<Character>,
    /// Each page of the run, in order; `None` where the analyzer gave none.
    pages: Vec<Option<PageNotes>>,
}

#[derive(Clone, Debug, PartialEq)]
pub(crate) struct Character {
    pub(crate) name: String,
    pub(crate) description: String,
}

/// What the analyzer said of one page.
#[derive(Clone, Debug, Default, PartialEq)]
pub(crate) struct PageNotes {
    pub(crate) summary: String,
    /// One for each text of the page, in reading order.
    pub(crate) lines: Vec<LineNote>,
}

#[derive(Clone, Debug, Default, PartialEq)]
pub(crate) struct LineNote {
    pub(crate) speaker: String,
    pub(crate) tone: String,
    /// The line corrected, where the OCR misread it.
    pub(crate) text: Option<String>,
}

/// Tokens a text takes, by its characters: Japanese and Chinese take about
/// one each and English a quarter, and the notes are mostly English.
const CHARS_PER_TOKEN: u64 = 3;

impl Analysis {
    fn new(pages: usize) -> Self {
        static NEXT: AtomicU64 = AtomicU64::new(1);
        Self {
            id: NEXT.fetch_add(1, Ordering::Relaxed),
            cast: Vec::new(),
            pages: vec![None; pages],
        }
    }

    /// Tells one analysis from another, so a conversation written under one
    /// starts again under the next.
    pub(crate) fn id(&self) -> u64 {
        self.id
    }

    pub(crate) fn page(&self, index: usize) -> Option<&PageNotes> {
        self.pages.get(index)?.as_ref()
    }

    /// The characters and every page's summary, for the translation's system
    /// prompt, or `None` when there are none. A model whose window is known
    /// keeps a fifth of it, and at least 8,192 tokens, for the turns, so
    /// summaries that don't fit the rest are left off the end.
    pub(crate) fn world(&self, window: Option<u64>) -> Option<String> {
        let summaries: Vec<(usize, &str)> = self
            .pages
            .iter()
            .enumerate()
            .filter_map(|(index, notes)| Some((index, notes.as_ref()?.summary.trim())))
            .filter(|(_, summary)| !summary.is_empty())
            .collect();
        if self.cast.is_empty() && summaries.is_empty() {
            return None;
        }
        let mut text = "Notes from reading the whole work, to keep names, pronouns and tone the same from page to page.".to_owned();
        if !self.cast.is_empty() {
            text.push_str("\n\nCharacters:");
            for character in &self.cast {
                match character.description.trim() {
                    "" => text.push_str(&format!("\n- {}", character.name)),
                    description => {
                        text.push_str(&format!("\n- {}: {description}", character.name));
                    }
                }
            }
        }
        if summaries.is_empty() {
            return Some(text);
        }
        text.push_str("\n\nStory, page by page:");
        let budget = window.map(|window| {
            let reserve = (window / 5).max(8192);
            window.saturating_sub(reserve) * CHARS_PER_TOKEN
        });
        let mut used = text.chars().count() as u64;
        for (index, summary) in summaries {
            let line = format!("\nPage {}: {summary}", index + 1);
            used += line.chars().count() as u64;
            if budget.is_some_and(|budget| used > budget) {
                text.push_str(
                    "\n(The summaries of the pages after this are left out to fit the context window.)",
                );
                break;
            }
            text.push_str(&line);
        }
        Some(text)
    }

    /// Adds `characters`, a name once: a later description replaces an
    /// earlier one that's shorter.
    fn remember(&mut self, characters: Vec<prompt::CharacterReply>) {
        for character in characters {
            let name = character.name.trim();
            if name.is_empty() {
                continue;
            }
            let description = character.description.trim();
            match self
                .cast
                .iter_mut()
                .find(|known| known.name.eq_ignore_ascii_case(name))
            {
                Some(known) => {
                    if description.len() > known.description.len() {
                        known.description = description.to_owned();
                    }
                }
                None => self.cast.push(Character {
                    name: name.to_owned(),
                    description: description.to_owned(),
                }),
            }
        }
    }

    /// Takes the notes of the pages in `new` from `reply`, each page's
    /// lines by their ids.
    fn absorb(&mut self, reply: Reply, texts: &HashMap<usize, Arc<Prepared>>, new: std::ops::Range<usize>) {
        for page in reply.pages {
            let Some(index) = page.page.checked_sub(1) else {
                continue;
            };
            let Some(prepared) = texts.get(&index).filter(|_| new.contains(&index)) else {
                continue;
            };
            let count = prepared.texts.len();
            let mut lines = vec![LineNote::default(); count];
            for line in page.lines {
                if (1..=count).contains(&line.id) {
                    lines[line.id - 1] = LineNote {
                        speaker: line.speaker.trim().to_owned(),
                        tone: line.tone.trim().to_owned(),
                        text: line
                            .text
                            .map(|text| text.trim().to_owned())
                            .filter(|text| !text.is_empty()),
                    };
                }
            }
            self.pages[index] = Some(PageNotes {
                summary: page.summary.trim().to_owned(),
                lines,
            });
        }
        self.remember(reply.characters);
    }
}

impl PageNotes {
    /// Puts the corrected readings in `texts`, the page's own, when the notes
    /// are for as many texts as it has; how many it changed.
    pub(crate) fn correct(&self, texts: &mut [String]) -> usize {
        if self.lines.len() != texts.len() {
            return 0;
        }
        let mut changed = 0;
        for (text, line) in texts.iter_mut().zip(&self.lines) {
            if let Some(corrected) = &line.text
                && corrected != text.trim()
            {
                *text = corrected.clone();
                changed += 1;
            }
        }
        changed
    }
}

/// A run's analysis, made by whichever translation asks first.
pub(crate) struct Analyzing {
    pages: Vec<PageId>,
    result: OnceCell<Option<Arc<Analysis>>>,
}

impl Analyzing {
    /// For a run over `pages`, in the run's order.
    pub(crate) fn new(pages: Vec<PageId>) -> Self {
        Self {
            pages,
            result: OnceCell::new(),
        }
    }

    /// The analysis, made now if no translation has asked yet; `None` when it
    /// failed, which the translation goes on without.
    pub(crate) async fn get(
        &self,
        session: &Arc<Mutex<Session>>,
        config: &AnalyzerConfig,
        cancel: &CancellationToken,
    ) -> Option<Arc<Analysis>> {
        self.result
            .get_or_init(|| async {
                match analyze(session, &self.pages, config, cancel).await {
                    Ok(analysis) => Some(Arc::new(analysis)),
                    Err(error) => {
                        tracing::warn!("the analysis failed, so the translation goes without it: {error:#}");
                        None
                    }
                }
            })
            .await
            .clone()
    }
}

async fn analyze(
    session: &Arc<Mutex<Session>>,
    pages: &[PageId],
    config: &AnalyzerConfig,
    cancel: &CancellationToken,
) -> Result<Analysis> {
    let (vendor, _) = config
        .model
        .split_once('/')
        .with_context(|| format!("`{}` names no model: expected `vendor/model`", config.model))?;
    ensure!(
        vendor != koharu_llm::KOHARU,
        "the analyzer must be a hosted model that reads images, such as an OpenAI-compatible server's"
    );
    let model = koharu_llm::connect(
        &config.model,
        Size {
            sequences: 1,
            context: 32_768,
            images: true,
        },
    )
    .await?;
    ensure!(
        model.spec().is_some_and(|spec| spec.input.image),
        "the analysis model reads no images"
    );

    let size = (config.pages.max(1)) as usize;
    let overlap = (config.overlap as usize).min(size - 1);
    let mut analysis = Analysis::new(pages.len());
    let mut prepared: HashMap<usize, Arc<Prepared>> = HashMap::new();
    let (mut next, mut failed) = (0, 0);
    while next < pages.len() {
        ensure!(!cancel.is_cancelled(), "the run stopped");
        let from = next.saturating_sub(overlap);
        let to = (from + size).min(pages.len());

        let wanted: Vec<usize> = (from..to)
            .filter(|index| !prepared.contains_key(index))
            .collect();
        let made = {
            let ids: Vec<PageId> = wanted.iter().map(|&index| pages[index]).collect();
            let (session, config) = (Arc::clone(session), config.clone());
            tokio::task::spawn_blocking(move || {
                ids.par_iter()
                    .map(|&id| page::prepare(&session, id, &config))
                    .collect::<Vec<_>>()
            })
            .await?
        };
        for (index, made) in wanted.into_iter().zip(made) {
            let made = made.unwrap_or_else(|error| {
                tracing::warn!(page = index + 1, "the page can't be shown to the analyzer: {error:#}");
                Prepared {
                    texts: Vec::new(),
                    image: None,
                }
            });
            prepared.insert(index, Arc::new(made));
        }
        prepared.retain(|&index, _| index >= from);

        tracing::info!(from = from + 1, to, context = next - from, "analyzing pages");
        let message = message(from, to, next, &prepared, &analysis);
        match ask(&model, config, message, cancel).await {
            Ok(reply) => {
                failed = 0;
                analysis.absorb(reply, &prepared, next..to);
            }
            Err(error) => {
                if cancel.is_cancelled() {
                    return Err(error);
                }
                failed += 1;
                tracing::warn!(from = from + 1, to, "a window of pages failed: {error:#}");
                if failed >= 2 {
                    tracing::warn!("two windows failed in a row; the analysis stops here");
                    break;
                }
            }
        }
        next = to;
    }
    Ok(analysis)
}

/// The window `from..to`, whose pages before `next` are context only: the
/// characters so far, then each page's lines and image.
fn message(
    from: usize,
    to: usize,
    next: usize,
    prepared: &HashMap<usize, Arc<Prepared>>,
    analysis: &Analysis,
) -> Message {
    let mut intro = format!("Pages {} to {} of the work follow, in reading order.", from + 1, to);
    if next > from {
        intro.push_str(&format!(" Pages {} to {} are context only.", from + 1, next));
    }
    if !analysis.cast.is_empty() {
        let names: Vec<&str> = analysis
            .cast
            .iter()
            .map(|character| character.name.as_str())
            .collect();
        intro.push_str(&format!(" The characters so far: {}.", names.join(", ")));
    }
    let mut content = vec![UserContent::text(intro)];
    for index in from..to {
        let Some(page) = prepared.get(&index) else {
            continue;
        };
        let marker = if index < next { " (context only)" } else { "" };
        content.push(UserContent::text(format!(
            "Page {}{marker}\nOCR lines: {}",
            index + 1,
            prompt::lines(&page.texts)
        )));
        if let Some(image) = &page.image {
            content.push(UserContent::Image(Image {
                data: DocumentSourceKind::Base64(image.clone()),
                media_type: Some(ImageMediaType::WEBP),
                detail: None,
                native: None,
            }));
        }
    }
    Message::from(content)
}

/// `message` asked of `model`, twice at most.
async fn ask(
    model: &DynModel<Completion>,
    config: &AnalyzerConfig,
    message: Message,
    cancel: &CancellationToken,
) -> Result<Reply> {
    let mut failure = None;
    for attempt in 1..=2 {
        let mut options = GenerationOptions::default();
        if let Some(reasoning) = config.reasoning {
            options = options.reasoning(reasoning);
        }
        let request = koharu_llm::answers(model, prompt::schema()?)
            .preamble(prompt::SYSTEM)
            .options(options)
            .build()
            .prompt(message.clone())
            .history(Vec::<Message>::new());
        let response = tokio::select! {
            response = request.into_future() => response.map_err(|error| anyhow::anyhow!("{error}")),
            () = cancel.cancelled() => bail!("the run stopped"),
        };
        let reply = response.and_then(|response| prompt::parse(&response.output()));
        match reply {
            Ok(reply) => return Ok(reply),
            Err(error) => {
                if cancel.is_cancelled() {
                    return Err(error);
                }
                tracing::warn!(attempt, "the analyzer's answer failed: {error:#}");
                failure = Some(error);
            }
        }
    }
    Err(failure.expect("two attempts, each a failure"))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn analysis() -> Analysis {
        let mut analysis = Analysis::new(3);
        analysis.cast.push(Character {
            name: "Ann".to_owned(),
            description: "a student".to_owned(),
        });
        for (index, summary) in ["One.", "Two.", "Three."].into_iter().enumerate() {
            analysis.pages[index] = Some(PageNotes {
                summary: summary.to_owned(),
                lines: Vec::new(),
            });
        }
        analysis
    }

    #[test]
    fn the_world_holds_the_cast_and_every_summary() {
        let world = analysis().world(None).expect("a world");
        assert!(world.contains("- Ann: a student"));
        assert!(world.contains("Page 1: One.") && world.contains("Page 3: Three."));
    }

    #[test]
    fn a_small_window_leaves_off_the_last_summaries() {
        // 8,192 tokens are reserved, so a window of 8,200 has room for 8.
        let world = analysis().world(Some(8_200)).expect("a world");
        assert!(world.contains("left out to fit the context window"));
        assert!(!world.contains("Page 3: Three."));
    }

    #[test]
    fn corrections_apply_only_to_notes_for_as_many_texts() {
        let notes = PageNotes {
            summary: String::new(),
            lines: vec![
                LineNote {
                    text: Some("やあ".to_owned()),
                    ..LineNote::default()
                },
                LineNote::default(),
            ],
        };
        let mut texts = vec!["やぁ".to_owned(), "うん".to_owned()];
        assert_eq!(notes.correct(&mut texts), 1);
        assert_eq!(texts, ["やあ", "うん"]);
        let mut fewer = vec!["やぁ".to_owned()];
        assert_eq!(notes.correct(&mut fewer), 0);
    }

    #[test]
    fn the_characters_keep_the_longer_description() {
        let mut analysis = Analysis::new(0);
        let reply = |description: &str| {
            vec![prompt::CharacterReply {
                name: "Ann".to_owned(),
                description: description.to_owned(),
            }]
        };
        analysis.remember(reply("a student"));
        analysis.remember(reply("a student who speaks formally"));
        analysis.remember(reply("shy"));
        assert_eq!(analysis.cast.len(), 1);
        assert_eq!(analysis.cast[0].description, "a student who speaks formally");
    }
}
