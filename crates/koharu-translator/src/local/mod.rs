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
/// Output tokens kept per segment: its id, a speaker such as `A>B: em - anh`, and the
/// translated text of a speech bubble.
const TOKENS_PER_SEGMENT: usize = 60;

/// A page image costs about this many tokens at most. Encoding the image is the slowest part
/// of a page on a CPU and the cost grows with the token count; the image only has to show who
/// speaks, because the text comes from the OCR.
const IMAGE_MAX_TOKENS: i32 = 700;

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
            mtmd: resolved
                .projector
                .map(|projector| MtmdOptions::new(projector).with_image_token_range(-1, IMAGE_MAX_TOKENS)),
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
            // Encoding the page images on CPU takes minutes, so retries are text only.
            let mut retry = request.select(&missing);
            retry.drop_images();
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

    /// Runs one constrained generation and returns the parsed translations plus the
    /// indices of segments the model did not return.
    async fn generate(
        &self,
        request: &TranslationRequest,
        reasoning: bool,
        options: &GenerationOptions,
    ) -> Result<(Vec<String>, Vec<usize>)> {
        let images: Vec<_> = if request.page_images.is_empty() {
            request.image.iter().cloned().collect()
        } else {
            request.page_images.clone()
        };
        let (prompt, shared_prefix) = self.render_prompt(request, reasoning)?;
        let schema = prompt::request_schema(request);
        let llm = Arc::clone(&self.llm);
        let mut options = options.clone();
        // Each segment needs room for its id, its speaker and its text, so a request with
        // many segments (several pages) must not be cut off by the single page limit.
        options.max_tokens = options
            .max_tokens
            .max(request.segments.len() * TOKENS_PER_SEGMENT + 64);
        let output = tokio_rayon::spawn(move || {
            // The instructions at the start of the prompt are the same on every page: the model
            // reads them once and keeps what it computed.
            let mut input = Input::new(&prompt).with_shared_prefix(shared_prefix);
            for image in &images {
                input = input.with_image(&**image);
            }
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
        // Shows the speakers the model decided on; `parent: None` prints it right away.
        tracing::debug!(parent: None, response = %output.text, "local translation response");
        if request.wants_speaker() {
            // Only for the log: what the model decided about who speaks to whom. It is not
            // kept or sent to the next pages, so a wrong guess cannot become a rule.
            let mut decided: Vec<String> = Vec::new();
            for speaker in prompt::speakers(&output.text) {
                if !decided.contains(&speaker) {
                    decided.push(speaker);
                }
            }
            if !decided.is_empty() {
                tracing::info!(parent: None, speakers = %decided.join(" | "), "speakers decided for this page");
            }
        }
        Ok(prompt::parse_translations(
            "local",
            &output.text,
            &request.segments,
        )?)
    }

    /// Asks the model for the glossary entries of one page, from its source lines. Text only,
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
        Ok(prompt::parse_glossary(&output.text))
    }

    /// Asks the model how the characters address each other, from their profiles. Text only,
    /// without reasoning, with a short output. Every entry that comes back is an address pair.
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

    /// The chat prompt, and how many of its leading bytes are the instructions that do not
    /// change from page to page (0 when they cannot be told apart).
    fn render_prompt(
        &self,
        request: &TranslationRequest,
        reasoning: bool,
    ) -> Result<(String, usize)> {
        let (system, payload) = prompt::prompts(request)?;
        let images = request
            .page_images
            .len()
            .max(usize::from(request.image.is_some()));
        let payload = if images > 0 {
            let markers = vec![media_marker(); images].join("\n");
            format!("{markers}\n{payload}")
        } else {
            payload
        };
        let prompt = self
            .llm
            .render_chat_prompt_with_options(
                &[ChatMessage::system(system), ChatMessage::user(payload)],
                ChatTemplateOptions {
                    add_generation_prompt: true,
                    enable_thinking: reasoning,
                },
            )
            .context("failed to render local translation prompt")?;
        let stable = prompt::stable_system(request);
        let shared_prefix = prompt
            .find(&stable)
            .map_or(0, |start| start + stable.len());
        Ok((prompt, shared_prefix))
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
