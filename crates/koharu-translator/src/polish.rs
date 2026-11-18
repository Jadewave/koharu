//! The final translation of a page.
//!
//! The vision model reads every page with its panels (see `analysis`): it decides who speaks to
//! whom and with which words, writes a first translation of every line, and tells the story of
//! the work page by page. Then a second model, DeepSeek by default, translates the page again
//! for the reader: it sees the whole page, and is sent
//!
//! - the notes of the page (who speaks to whom, the words for "I" and "you", the tone),
//! - the draft of the vision model for every line, to improve and not to copy,
//! - the story of the work: the cast, what happened before the page and what comes next,
//! - what the glossary and the memory of the translator add to every translation.
//!
//! What it writes is held to the notes: a line that uses other words for "I" and "you" than the
//! note of the line decided is translated again with those words spelled out. When no model
//! answers, the draft of the vision model is the translation, if it has one for every line; when
//! it has not, the page is translated the way it was before, by the model the user chose.

use std::{collections::HashSet, fmt::Write as _, sync::Arc};

use image::DynamicImage;

use crate::{
    Error, GenerationConfig, Language, ModelSelection, PageAnalysis, Provider, TranslationRequest,
    Translator,
    address::{self, Declared},
    analysis::analysis_config,
};

/// The story of the work as the analysis told it.
#[derive(Default)]
pub(crate) struct Story {
    /// What happens on every page, in reading order (empty for a page that was not read).
    pub summaries: Vec<String>,
    /// One line for every character: names, who they are, what they call the others.
    pub cast: Vec<String>,
    /// The models of the final translation that the server refused for the rest of the run.
    pub dead: HashSet<String>,
}

/// The pages after the one translated whose summary is sent.
const NEXT_PAGES: usize = 3;
/// How long the summary of one of the next pages may be.
const NEXT_CHARS: usize = 220;

impl Story {
    /// The story told to the translator of the page at `position`, at most `limit` characters:
    /// the cast, the pages before it (the nearest first when there is no room for all), and a
    /// few of the pages after it.
    pub(crate) fn context(&self, position: usize, limit: usize) -> String {
        let mut text = String::from(
            "Story of the work, told by the vision model that read every page. Tell it the same \
way on every page: the same names, the same words for the same people.",
        );
        let mut room = limit.saturating_sub(text.chars().count());

        // The cast takes up to a third of the room.
        let mut cast = String::new();
        for entry in &self.cast {
            let line = format!("\n- {entry}");
            if cast.chars().count() + line.chars().count() > room / 3 {
                break;
            }
            cast.push_str(&line);
        }
        if !cast.is_empty() {
            room = room.saturating_sub(cast.chars().count() + 12);
            text.push_str("\nCast:");
            text.push_str(&cast);
        }

        // What follows is kept for the end, so the pages before get what is left of the room.
        let mut next = String::new();
        for (page, summary) in self.summaries.iter().enumerate().skip(position + 1).take(NEXT_PAGES) {
            let summary = summary.trim();
            if summary.is_empty() {
                continue;
            }
            let summary: String = summary.chars().take(NEXT_CHARS).collect();
            let _ = write!(next, "\npage {}: {summary}", page + 1);
        }
        room = room.saturating_sub(next.chars().count() + 24);

        let mut before: Vec<String> = Vec::new();
        let mut used = 0;
        for (page, summary) in self.summaries.iter().enumerate().take(position).rev() {
            let summary = summary.trim();
            if summary.is_empty() {
                continue;
            }
            let line = format!("\npage {}: {summary}", page + 1);
            let size = line.chars().count();
            if used + size > room {
                break;
            }
            used += size;
            before.push(line);
        }
        if !before.is_empty() {
            text.push_str("\nBefore this page:");
            for line in before.iter().rev() {
                text.push_str(line);
            }
        }
        if !next.is_empty() {
            text.push_str("\nAfter this page:");
            text.push_str(&next);
        }
        text
    }
}

/// The translations of a page and who wrote them.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Polished {
    pub translations: Vec<String>,
    /// The model that wrote them, or `draft` for the first translation of the vision model.
    pub by: String,
}

/// The request for the final translation of a page: its source lines as the vision model
/// corrected them, the page picture, and everything the model is told about the page. `None`
/// when the notes are not those of this page.
pub(crate) fn polish_request(
    analysis: &PageAnalysis,
    sources: &[String],
    language: Language,
    instructions: Option<&str>,
    story: &str,
    image: Option<Arc<DynamicImage>>,
) -> Option<TranslationRequest> {
    let notes = analysis.instructions(sources);
    if notes.is_empty() {
        return None;
    }
    let texts = analysis.texts(sources);
    let mut parts: Vec<String> = Vec::new();
    if let Some(instructions) = instructions.map(str::trim).filter(|text| !text.is_empty()) {
        parts.push(instructions.to_owned());
    }
    if !story.trim().is_empty() {
        parts.push(story.trim().to_owned());
    }
    parts.push(notes);
    let mut request = TranslationRequest::new(texts, language)
        .with_instructions(parts.join("\n\n"))
        .with_pairs(analysis.expected_pairs(sources))
        .with_drafts(analysis.drafts(sources));
    if let Some(image) = image {
        request = request.with_image(image);
    }
    Some(request)
}

impl Translator {
    /// Whether the pages the analysis read are translated again by the models of
    /// `analysis.translate_models`.
    #[must_use]
    pub fn polish_enabled(&self) -> bool {
        analysis_config().is_ok_and(|config| {
            config.enabled && config.translate && !config.translate_models.is_empty()
        })
    }

    /// Translates a page the analysis read. `sources` are the lines of the page as OCR read them,
    /// `image` the whole page with the numbered boxes of the lines. Returns `None` when the page
    /// has to be translated the old way: the notes are not those of this page, no model
    /// answered, and the vision model left a line without a draft.
    pub async fn polish_page(
        &self,
        language: Language,
        analysis: &PageAnalysis,
        sources: &[String],
        image: Option<Arc<DynamicImage>>,
        instructions: Option<&str>,
    ) -> anyhow::Result<Option<Polished>> {
        let config = analysis_config()?;
        if !config.enabled || !config.translate {
            return Ok(None);
        }
        let story = self
            .story
            .lock()
            .map(|story| story.context(analysis.position(), config.story_chars))
            .unwrap_or_default();
        let Some(request) = polish_request(analysis, sources, language, instructions, &story, image) else {
            return Ok(None);
        };
        let expected: Vec<Option<Declared>> = request
            .pairs
            .iter()
            .map(|pair| pair.as_ref().and_then(Declared::expected))
            .collect();
        let wanted = request.segments.len();

        for entry in &config.translate_models {
            if self
                .story
                .lock()
                .is_ok_and(|story| story.dead.contains(&entry.model))
            {
                continue;
            }
            let reasoning = entry.reasoning.unwrap_or(config.translate_reasoning);
            let selection = ModelSelection {
                provider: Provider::OpenAiCompatible,
                model: Some(entry.model.clone()),
                quantization: None,
                vision: entry.vision,
                reasoning,
            };
            let generation = GenerationConfig {
                temperature: Some(config.translate_temperature),
                top_p: Some(0.95),
                max_tokens: Some(config.translate_max_tokens),
                reasoning: Some(reasoning),
                vision: Some(entry.vision),
                ..GenerationConfig::default()
            };
            match self.translate(&selection, generation, request.clone()).await {
                Ok((_, mut translated)) if translated.len() == wanted => {
                    self.hold_to_the_notes(&selection, generation, &request, &expected, &mut translated)
                        .await;
                    tracing::info!(
                        model = %entry.model,
                        lines = wanted,
                        drafts = request.drafts.iter().filter(|draft| !draft.is_empty()).count(),
                        "page translated by the final model"
                    );
                    return Ok(Some(Polished {
                        translations: translated,
                        by: entry.model.clone(),
                    }));
                }
                Ok((_, translated)) => {
                    tracing::warn!(
                        model = %entry.model,
                        got = translated.len(),
                        wanted,
                        "final model returned the wrong number of translations"
                    );
                }
                Err(error) => {
                    let refused = error.chain().any(|cause| {
                        matches!(
                            cause.downcast_ref::<Error>(),
                            Some(Error::Api { status, .. }) if matches!(*status, 401 | 403 | 404)
                        )
                    });
                    if refused && let Ok(mut story) = self.story.lock() {
                        story.dead.insert(entry.model.clone());
                    }
                    tracing::warn!(model = %entry.model, error = %format!("{error:#}"), "final model could not translate the page");
                }
            }
        }

        // Nobody translated the page again: the first translation of the vision model stands in
        // when it has one for every line.
        let drafts = analysis.drafts(sources);
        if drafts.len() == wanted && drafts.iter().all(|draft| !draft.is_empty()) {
            tracing::warn!("the draft of the vision model is the translation of this page");
            return Ok(Some(Polished {
                translations: drafts,
                by: "draft".to_owned(),
            }));
        }
        Ok(None)
    }

    /// Translates again the lines that use other words for "I" and "you" than their notes
    /// decided, with those words spelled out. What is still wrong afterwards stays.
    async fn hold_to_the_notes(
        &self,
        selection: &ModelSelection,
        generation: GenerationConfig,
        request: &TranslationRequest,
        expected: &[Option<Declared>],
        translated: &mut [String],
    ) {
        if !expected.iter().any(Option::is_some) {
            return;
        }
        let flagged = address::conflicts_with_notes(expected, &[], translated);
        if flagged.is_empty() {
            return;
        }
        tracing::info!(lines = flagged.len(), "lines that contradict their notes are translated again");
        let mut retry = request.select(&flagged);
        let note = address::notes_note(expected, &flagged, &flagged);
        retry.instructions = Some(match request.instructions.as_deref() {
            Some(instructions) => format!("{instructions}\n\n{note}"),
            None => note,
        });
        match self.translate(selection, generation, retry).await {
            Ok((_, fresh)) if fresh.len() == flagged.len() => {
                for (&index, text) in flagged.iter().zip(fresh) {
                    translated[index] = text;
                }
            }
            Ok(_) => tracing::warn!("the retry returned the wrong number of translations"),
            Err(error) => tracing::warn!(error = %format!("{error:#}"), "the retry of the lines failed"),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn story() -> Story {
        Story {
            summaries: (1..=10).map(|page| format!("Summary of page {page}.")).collect(),
            cast: vec![
                "Sasaki / Sasaki-san: adult woman, teacher; speaks politely".to_owned(),
                "Rio: teenage girl, twin tails; says to Sasaki \"em\" / \"cô\"".to_owned(),
            ],
            dead: HashSet::new(),
        }
    }

    #[test]
    fn the_story_tells_the_cast_what_came_before_and_what_comes_next() {
        let text = story().context(4, 4000);
        assert!(text.contains("Cast:\n- Sasaki / Sasaki-san: adult woman, teacher"), "{text}");
        assert!(text.contains("Before this page:\npage 1: Summary of page 1.\npage 2:"), "{text}");
        assert!(text.contains("page 4: Summary of page 4."), "{text}");
        // The page itself is told by its notes, not here.
        assert!(!text.contains("Summary of page 5."), "{text}");
        assert!(text.contains("After this page:\npage 6: Summary of page 6.\npage 7:"), "{text}");
        assert!(text.contains("page 8: Summary of page 8."), "{text}");
        assert!(!text.contains("page 9"), "only a few pages ahead");
    }

    #[test]
    fn the_first_page_has_nothing_before_it_and_the_last_nothing_after() {
        let first = story().context(0, 4000);
        assert!(!first.contains("Before this page"), "{first}");
        assert!(first.contains("After this page:\npage 2:"), "{first}");
        let last = story().context(9, 4000);
        assert!(last.contains("Before this page:"), "{last}");
        assert!(!last.contains("After this page"), "{last}");
    }

    #[test]
    fn when_there_is_no_room_the_pages_nearest_to_the_page_stay() {
        let mut long = story();
        long.summaries = (1..=40).map(|page| format!("Page number {page} has a long summary of what happens in it.")).collect();
        let text = long.context(30, 1200);
        assert!(text.chars().count() <= 1300, "{}", text.chars().count());
        assert!(text.contains("page 30:"), "the page before is the one to keep");
        assert!(!text.contains("page 1:"), "the far past goes first");
        // The order is still the order of the story.
        let first = text.find("page 29").unwrap();
        let second = text.find("page 30").unwrap();
        assert!(first < second);
    }

    #[test]
    fn pages_that_were_not_read_are_skipped_and_an_empty_story_says_only_how_to_tell_it() {
        let mut gaps = story();
        gaps.summaries[2].clear();
        let text = gaps.context(4, 4000);
        assert!(!text.contains("page 3:"));
        assert!(text.contains("page 2:") && text.contains("page 4:"));
        let empty = Story::default().context(0, 4000);
        assert!(empty.starts_with("Story of the work"));
        assert!(!empty.contains("Cast:") && !empty.contains("Before") && !empty.contains("After"));
    }
}
