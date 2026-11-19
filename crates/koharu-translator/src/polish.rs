//! The final translation of a page.
//!
//! The vision model reads the cleaned pictures of the pages (see `analysis`): it decides, for
//! every line, who speaks, who listens and which words they use for "I" and "you", and writes a
//! first translation. Then the model the user chose for the translation (the local one)
//! translates the page again for the reader. It is sent only what it needs to write the lines:
//!
//! - the decisions for every line (who speaks to whom, the words for "I" and "you", the tone),
//! - the draft of the vision model for every line, to improve and not to copy,
//! - the names and terms of the glossary.
//!
//! It is not told the story: no scene, no summaries, no cast, no lines of the pages before. The
//! words for "I" and "you" were decided where the pictures were seen, and the model follows them.
//!
//! The translation of a model is held to the decisions (see `with_consistent_pairs`): a line that
//! uses other words for "I" and "you" than the line decided is translated again with those words
//! spelled out. When the model fails, the draft of the vision model is the translation, if it has
//! one for every line; when it has not, the page is translated the way it was before.

use crate::{
    GenerationConfig, Language, ModelSelection, PageAnalysis, TranslationRequest, Translator,
    analysis::analysis_config,
};

/// The translations of a page and who wrote them.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Polished {
    pub translations: Vec<String>,
    /// The model that wrote them, or `draft` for the first translation of the vision model.
    pub by: String,
}

/// The request for the final translation of a page: its source lines as the vision model
/// corrected them and the decisions and the draft for every line. No picture goes with it. It
/// carries everything the model is given (`standalone`). `None` when the notes are not those of
/// this page.
pub(crate) fn polish_request(
    analysis: &PageAnalysis,
    sources: &[String],
    language: Language,
    instructions: Option<&str>,
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
    parts.push(notes);
    Some(
        TranslationRequest::new(texts, language)
            .with_instructions(parts.join("\n\n"))
            .with_pairs(analysis.expected_pairs(sources))
            .with_drafts(analysis.drafts(sources))
            .standalone(),
    )
}

impl Translator {
    /// Whether the pages the analysis read are translated again by the model of the translation.
    #[must_use]
    pub fn polish_enabled(&self) -> bool {
        analysis_config().is_ok_and(|config| config.enabled && config.translate)
    }

    /// Translates a page the analysis read with `selection` (the model of the translation).
    /// `sources` are the lines of the page as OCR read them. Returns `None` when the page has to be translated the old way: the notes are not those of
    /// this page, the model did not answer, and the vision model left a line without a draft.
    pub async fn polish_page(
        &self,
        selection: &ModelSelection,
        generation: GenerationConfig,
        language: Language,
        analysis: &PageAnalysis,
        sources: &[String],
        instructions: Option<&str>,
    ) -> anyhow::Result<Option<Polished>> {
        let config = analysis_config()?;
        if !config.enabled || !config.translate {
            return Ok(None);
        }
        let Some(request) = polish_request(analysis, sources, language, instructions) else {
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
