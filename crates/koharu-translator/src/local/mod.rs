use std::sync::Arc;

use anyhow::Context;
use koharu_ml::llm::{
    ChatMessage, ChatTemplateOptions, GenerationControl, GenerationOptions, Input, Llm, LoadOptions,
    MtmdOptions, TokenChunk, media_marker,
};

mod catalog;

pub use catalog::LocalConfig;
use catalog::LocalModelDescriptor;
pub(crate) use catalog::{DEFAULT_MODEL, DEFAULT_QUANTIZATION};

use crate::{
    Device, Error, GenerationConfig, GlossaryEntry, Language, Model, ModelSelection, Provider,
    Quantization, Result, TranslationRequest, loop_guard, prompt,
};

/// How many times segments the model skipped (missing or duplicated IDs) are translated again.
const MAX_MISSING_RETRIES: u32 = 2;

/// Upper bound for the tokens of one glossary update.
const GLOSSARY_MAX_TOKENS: usize = 900;

/// The partial output is checked for a repetition loop every 16 generated tokens.
const LOOP_CHECK_MASK: usize = 15;

/// Stops a generation that has started repeating itself instead of letting it run to the
/// token limit (a looping generation on CPU wastes minutes).
fn stop_on_loop(chunk: TokenChunk<'_>) -> anyhow::Result<GenerationControl> {
    if chunk.generated_tokens & LOOP_CHECK_MASK == 0 && loop_guard::tail_is_looping(chunk.text) {
        tracing::warn!(
            tokens = chunk.generated_tokens,
            "generation is repeating itself; stopping early"
        );
        return Ok(GenerationControl::Stop);
    }
    Ok(GenerationControl::Continue)
}

#[derive(Debug)]
pub struct LocalTranslator {
    descriptor: LocalModelDescriptor,
    llm: Arc<Llm>,
}

impl LocalTranslator {
    pub async fn load(device: Device, selection: &ModelSelection) -> Result<Self> {
        let model = selection
            .model
            .as_deref()
            .context("local translation requires a selected model")?;
        let descriptor = catalog::MODELS
            .iter()
            .copied()
            .find(|descriptor| descriptor.id == model)
            .with_context(|| format!("unknown local translator '{model}'"))?;
        let resolved = descriptor.resolve(selection).await?;
        let options = LoadOptions {
            mtmd: resolved.projector.map(MtmdOptions::new),
            ..LoadOptions::default()
        };
        let llm = Llm::load_with_options(device, resolved.model, options)
            .await
            .context("failed to load local translation model")?;
        if llm.capabilities().vision != descriptor.projector.is_some() {
            return Err(anyhow::anyhow!(
                "local translator vision capability does not match its catalog"
            )
            .into());
        }
        Ok(Self {
            descriptor,
            llm: Arc::new(llm),
        })
    }

    pub(crate) async fn translate(
        &self,
        request: TranslationRequest,
        generation: GenerationConfig,
    ) -> Result<Vec<String>> {
        let expected = request.segments.len();
        if expected == 0 {
            return Ok(Vec::new());
        }
        if !self
            .descriptor
            .target_languages
            .contains(request.target_language)
        {
            return Err(Error::UnsupportedLanguage {
                provider: "local",
                language: request.target_language,
            });
        }

        let reasoning = generation.reasoning.unwrap_or(false) && self.descriptor.reasoning;
        let options = self.descriptor.generation.options(generation);

        let (mut translations, mut missing) = self.generate(&request, reasoning, &options).await?;
        for attempt in 1..=MAX_MISSING_RETRIES {
            if missing.is_empty() {
                break;
            }
            tracing::warn!(
                missing = missing.len(),
                total = expected,
                attempt,
                "local translation skipped segments; translating them again"
            );
            let mut retry = request.clone();
            // Encoding the page image on CPU takes minutes, so retries are text only.
            retry.image = None;
            retry.segments = missing
                .iter()
                .map(|&index| request.segments[index].clone())
                .collect();
            let mut retry_options = options.clone();
            retry_options.seed = retry_options.seed.wrapping_add(attempt);
            let (retried, still_missing) = self.generate(&retry, reasoning, &retry_options).await?;
            for (position, &index) in missing.iter().enumerate() {
                if !still_missing.contains(&position) {
                    translations[index] = retried[position].clone();
                }
            }
            let remaining: Vec<usize> = still_missing
                .into_iter()
                .map(|position| missing[position])
                .collect();
            missing = remaining;
        }
        if !missing.is_empty() {
            tracing::warn!(
                missing = missing.len(),
                total = expected,
                "local translation still skipped segments after retries; keeping source text"
            );
        }
        Ok(translations)
    }

    /// Analyzes the marked manga page before translation. The selected translation model
    /// performs this pass too, so changing the model in the UI changes the visual analyst;
    /// no hidden Gemma model is used for page-level speaker identification.
    pub(crate) async fn analyze_page(
        &self,
        request: &TranslationRequest,
        previous_context: &str,
        generation: GenerationConfig,
    ) -> Result<Option<String>> {
        let Some(image) = request.image.clone() else {
            return Ok(None);
        };
        if !self.llm.capabilities().vision {
            return Ok(None);
        }
        let (system, user, schema) = prompt::visual_analysis_prompt(request, previous_context)?;
        let prompt = self
            .llm
            .render_chat_prompt_with_options(
                &[ChatMessage::system(system), ChatMessage::user(user)],
                ChatTemplateOptions {
                    add_generation_prompt: true,
                    enable_thinking: false,
                },
            )
            .context("failed to render visual-analysis prompt")?;
        let mut options = self.descriptor.generation.options(GenerationConfig {
            reasoning: Some(false),
            ..generation
        });
        options.max_tokens = (request.segments.len() * 80 + 500).clamp(1400, 3000);
        let llm = Arc::clone(&self.llm);
        let output = tokio_rayon::spawn(move || {
            let input = Input::new(&prompt).with_image(image.as_ref());
            llm.inference_with_json_schema_callback(&input, &options, &schema, stop_on_loop)
        })
        .await?;
        tracing::info!(
            prompt_tokens = output.prompt_tokens,
            generated_tokens = output.generated_tokens,
            generation_secs = output.generation_duration.as_secs_f32(),
            finish = ?output.finish_reason,
            "local visual analysis finished"
        );
        let value: serde_json::Value = match serde_json::from_str::<serde_json::Value>(&output.text) {
            Ok(value) if value.is_object() => value,
            Ok(_) => {
                tracing::warn!("visual analysis returned JSON that was not an object; continuing without it");
                return Ok(None);
            }
            Err(error) => {
                tracing::warn!(%error, "visual analysis returned invalid JSON; continuing without it");
                return Ok(None);
            }
        };
        let Some(dialogues) = value.get("dialogues").and_then(serde_json::Value::as_array) else {
            tracing::warn!("visual analysis omitted the dialogue mapping; continuing without it");
            return Ok(None);
        };
        let ids: std::collections::HashSet<u64> = dialogues
            .iter()
            .filter_map(|dialogue| dialogue.get("segment_id").and_then(serde_json::Value::as_u64))
            .collect();
        if dialogues.len() != request.segments.len()
            || ids.len() != request.segments.len()
            || (0..request.segments.len()).any(|id| !ids.contains(&(id as u64)))
        {
            tracing::warn!(
                expected = request.segments.len(),
                actual = dialogues.len(),
                "visual analysis did not map every OCR segment exactly once; continuing without it"
            );
            return Ok(None);
        }
        Ok(Some(value.to_string()))
    }

    /// Runs one constrained generation and returns the parsed translations plus the
    /// indices of segments the model did not return.
    async fn generate(
        &self,
        request: &TranslationRequest,
        reasoning: bool,
        options: &GenerationOptions,
    ) -> Result<(Vec<String>, Vec<usize>)> {
        let image = request.image.clone();
        let prompt = self.render_prompt(request, reasoning)?;
        let schema = prompt::output_schema(request.segments.len());
        let llm = Arc::clone(&self.llm);
        let options = options.clone();
        let output = tokio_rayon::spawn(move || {
            let input = image.as_deref().map_or_else(
                || Input::new(&prompt),
                |image| Input::new(&prompt).with_image(image),
            );
            llm.inference_with_json_schema_callback(&input, &options, &schema, stop_on_loop)
        })
        .await?;
        tracing::info!(
            prompt_tokens = output.prompt_tokens,
            prompt_secs = output.prompt_duration.as_secs_f32(),
            generated_tokens = output.generated_tokens,
            generation_secs = output.generation_duration.as_secs_f32(),
            finish = ?output.finish_reason,
            reasoning,
            "local generation finished"
        );
        Ok(prompt::parse_translations(
            "local",
            &output.text,
            &request.segments,
        )?)
    }

    /// Asks the selected model for glossary entries from OCR source lines. Text only,
    /// without reasoning, with a short output.
    pub(crate) async fn extract_glossary(
        &self,
        target: Language,
        glossary: &[GlossaryEntry],
        lines: &[String],
        generation: GenerationConfig,
    ) -> Result<Vec<GlossaryEntry>> {
        let (system, user) = prompt::glossary_prompts(target, glossary, lines)?;
        let prompt = self
            .llm
            .render_chat_prompt_with_options(
                &[ChatMessage::system(system), ChatMessage::user(user)],
                ChatTemplateOptions {
                    add_generation_prompt: true,
                    enable_thinking: false,
                },
            )
            .context("failed to render local glossary prompt")?;
        let mut options = self.descriptor.generation.options(GenerationConfig {
            reasoning: Some(false),
            ..generation
        });
        options.max_tokens = GLOSSARY_MAX_TOKENS;
        let schema = prompt::glossary_schema();
        let llm = Arc::clone(&self.llm);
        let output = tokio_rayon::spawn(move || {
            let input = Input::new(&prompt);
            llm.inference_with_json_schema_callback(&input, &options, &schema, stop_on_loop)
        })
        .await?;
        tracing::info!(
            prompt_tokens = output.prompt_tokens,
            prompt_secs = output.prompt_duration.as_secs_f32(),
            generated_tokens = output.generated_tokens,
            generation_secs = output.generation_duration.as_secs_f32(),
            finish = ?output.finish_reason,
            "local glossary generation finished"
        );
        // This pass sees OCR lines without page geometry or bubble tails. Never persist
        // address pairs from it; only the dedicated profile-based address pass may create
        // directional pairs.
        Ok(prompt::parse_glossary(&output.text)
            .into_iter()
            .filter(|entry| !entry.kind.trim().eq_ignore_ascii_case("address"))
            .collect())
    }

    /// Asks the selected model how characters address each other from supported profiles.
    /// Text only, without reasoning, with a short output.
    pub(crate) async fn extract_addresses(
        &self,
        target: Language,
        characters: &[GlossaryEntry],
        generation: GenerationConfig,
    ) -> Result<Vec<GlossaryEntry>> {
        let (system, user) = prompt::address_prompts(target, characters)?;
        let prompt = self
            .llm
            .render_chat_prompt_with_options(
                &[ChatMessage::system(system), ChatMessage::user(user)],
                ChatTemplateOptions {
                    add_generation_prompt: true,
                    enable_thinking: false,
                },
            )
            .context("failed to render local address prompt")?;
        let mut options = self.descriptor.generation.options(GenerationConfig {
            reasoning: Some(false),
            ..generation
        });
        options.max_tokens = GLOSSARY_MAX_TOKENS;
        let schema = prompt::glossary_schema();
        let llm = Arc::clone(&self.llm);
        let output = tokio_rayon::spawn(move || {
            let input = Input::new(&prompt);
            llm.inference_with_json_schema_callback(&input, &options, &schema, stop_on_loop)
        })
        .await?;
        tracing::info!(
            prompt_tokens = output.prompt_tokens,
            prompt_secs = output.prompt_duration.as_secs_f32(),
            generated_tokens = output.generated_tokens,
            generation_secs = output.generation_duration.as_secs_f32(),
            finish = ?output.finish_reason,
            "local address generation finished"
        );
        Ok(prompt::parse_glossary(&output.text)
            .into_iter()
            .map(|mut entry| {
                entry.kind = "address".to_owned();
                entry
            })
            .collect())
    }

    fn render_prompt(&self, request: &TranslationRequest, reasoning: bool) -> Result<String> {
        let (system, payload) = prompt::prompts(request)?;
        let payload = if request.image.is_some() {
            format!("{}\n{payload}", media_marker())
        } else {
            payload
        };
        Ok(self
            .llm
            .render_chat_prompt_with_options(
                &[ChatMessage::system(system), ChatMessage::user(payload)],
                ChatTemplateOptions {
                    add_generation_prompt: true,
                    enable_thinking: reasoning,
                },
            )
            .context("failed to render local translation prompt")?)
    }
}

pub(crate) fn models() -> Vec<Model> {
    catalog::MODELS
        .iter()
        .map(|descriptor| Model {
            provider: Provider::Local,
            model: Some(descriptor.id.to_owned()),
            name: descriptor.name.to_owned(),
            quantizations: descriptor
                .quantizations
                .iter()
                .map(|quantization| Quantization {
                    id: quantization.id.to_owned(),
                    name: quantization.name.to_owned(),
                    downloaded: std::iter::once(quantization.filename)
                        .chain(descriptor.projector)
                        .all(|filename| descriptor.file(filename).exists()),
                })
                .collect(),
            vision: descriptor.projector.is_some(),
            reasoning: descriptor.reasoning,
        })
        .collect()
}

pub(crate) fn supports_vision(selection: &ModelSelection) -> bool {
    selection.model.as_deref().is_some_and(|model| {
        catalog::MODELS
            .iter()
            .find(|descriptor| descriptor.id == model)
            .is_some_and(|descriptor| descriptor.projector.is_some())
    })
}
