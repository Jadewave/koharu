//! The final translation of a page.
//!
//! The vision model reads every page with its cleaned picture (see `analysis`): it decides who
//! speaks to whom and with which words, writes a first translation of every line, and tells the
//! story of the work page by page. Then the model the user chose for the translation (the local
//! one) translates the page again for the reader. It is sent
//!
//! - the notes of the page (who speaks to whom, the words for "I" and "you", the tone),
//! - the draft of the vision model for every line, to improve and not to copy,
//! - the story of the work: the cast, what happened before the page and what comes next,
//! - what the glossary and the memory of the translator add to every translation.
//!
//! The translation of a model is already held to the notes (see `with_consistent_pairs`): a line
//! that uses other words for "I" and "you" than the note of the line decided is translated again
//! with those words spelled out. When the model fails, the draft of the vision model is the
//! translation, if it has one for every line; when it has not, the page is translated the way it
//! was before.

use std::{fmt::Write as _, sync::Arc};

use image::DynamicImage;

use crate::{
    GenerationConfig, Language, ModelSelection, PageAnalysis, TranslationRequest, Translator,
    analysis::analysis_config,
};

/// The story of the work as the analysis told it.
#[derive(Default)]
pub(crate) struct Story {
    /// What happens on every page, in reading order (empty for a page that was not read).
    pub summaries: Vec<String>,
    /// One line for every character: names, who they are, what they call the others.
    pub cast: Vec<String>,
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
    /// Whether the pages the analysis read are translated again by the model of the translation.
    #[must_use]
    pub fn polish_enabled(&self) -> bool {
        analysis_config().is_ok_and(|config| config.enabled && config.translate)
    }

    /// Translates a page the analysis read with `selection` (the model of the translation).
    /// `sources` are the lines of the page as OCR read them, `image` the whole page (cleaned,
    /// with the numbered boxes of the lines), which a model without vision does not get.
    /// Returns `None` when the page has to be translated the old way: the notes are not those of
    /// this page, the model did not answer, and the vision model left a line without a draft.
    #[allow(clippy::too_many_arguments)]
    pub async fn polish_page(
        &self,
        selection: &ModelSelection,
        generation: GenerationConfig,
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
        let wanted = request.segments.len();
        let drafts = request.drafts.iter().filter(|draft| !draft.is_empty()).count();
        let by = selection
            .model
            .clone()
            .unwrap_or_else(|| Translator::model(selection).to_owned());
        match self.translate(selection, generation, request).await {
            Ok((_, translated)) if translated.len() == wanted => {
                tracing::info!(model = %by, lines = wanted, drafts, "page translated by the final model");
                return Ok(Some(Polished {
                    translations: translated,
                    by,
                }));
            }
            Ok((_, translated)) => tracing::warn!(
                model = %by,
                got = translated.len(),
                wanted,
                "final model returned the wrong number of translations"
            ),
            Err(error) => {
                tracing::warn!(model = %by, error = %format!("{error:#}"), "final model could not translate the page");
            }
        }

        // The model did not translate the page: the first translation of the vision model stands
        // in when it has one for every line.
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
