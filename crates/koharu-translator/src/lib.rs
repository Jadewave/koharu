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

use std::{sync::Arc, time::Duration};

use koharu_ml::Device;

use error::{Error, Result};
use local::LocalTranslator;

pub use backend::{TranslationContext, TranslationRequest};
pub use language::Language;
pub use model::{GenerationConfig, Model, ModelSelection, Quantization};
pub(crate) use model::{ModelGeneration, QuantizationDefinition, display_name};
pub use provider::{Provider, ProviderConfig, ProvidersConfig};

#[derive(Clone)]
pub struct Translator {
    providers: koharu_config::Config<ProvidersConfig>,
    local: Arc<tokio::sync::Mutex<Option<LoadedLocal>>>,
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

        if Self::supports_vision(selection, &generation) {
            request.prepare_image()?;
        } else {
            request.remove_image();
        }

        let expected = request.segments.len();
        let translated = self
            .translate_segments(selection, generation, &request)
            .await?;
        if translated.len() != expected {
            return Err(Error::SegmentCount {
                provider: provider_id,
                expected,
                actual: translated.len(),
            }
            .into());
        }
        tracing::Span::current().record("outcome", "completed");
        Ok((provider_id, translated))
    }

    /// Translates every segment. With reasoning enabled the first pass runs with reasoning;
    /// segments that come back looping (or never come back because the response was cut
    /// off) are translated again with reasoning disabled, and only those segments.
    async fn translate_segments(
        &self,
        selection: &ModelSelection,
        generation: GenerationConfig,
        request: &TranslationRequest,
    ) -> anyhow::Result<Vec<String>> {
        if generation.reasoning != Some(true) {
            return self.dispatch(selection, generation, request).await;
        }

        let total = request.segments.len();
        let (mut translated, redo) = match self.dispatch(selection, generation, request).await {
            Ok(translated) => {
                if translated.len() != total {
                    return Ok(translated);
                }
                let redo = loop_guard::looping_segments(&request.segments, &translated);
                (translated, redo)
            }
            Err(error) if retryable(&error) => {
                let (translated, mut redo) = match invalid_output(&error) {
                    Some(output) => {
                        prompt::salvage_translations(&output.response, &request.segments)
                    }
                    None => (request.segments.clone(), (0..total).collect()),
                };
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
                    "reasoning pass failed; translating unrecovered segments without reasoning"
                );
                (translated, redo)
            }
            Err(error) => return Err(error),
        };
        if redo.is_empty() {
            return Ok(translated);
        }

        tracing::warn!(
            looping = redo.len(),
            total,
            "reasoning pass produced looping segments; translating them again without reasoning"
        );
        let mut retry = request.clone();
        retry.segments = redo
            .iter()
            .map(|&index| request.segments[index].clone())
            .collect();
        let without_reasoning = GenerationConfig {
            reasoning: Some(false),
            ..generation
        };
        let retried = self.dispatch(selection, without_reasoning, &retry).await?;
        if retried.len() != redo.len() {
            let provider: &'static str = selection.provider.into();
            return Err(Error::SegmentCount {
                provider,
                expected: redo.len(),
                actual: retried.len(),
            }
            .into());
        }
        for (&index, text) in redo.iter().zip(retried) {
            translated[index] = text;
        }
        let remaining = loop_guard::looping_segments(&request.segments, &translated);
        if !remaining.is_empty() {
            tracing::warn!(
                remaining = remaining.len(),
                "segments still look like loops after retrying without reasoning"
            );
        }
        Ok(translated)
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
        let mut loaded = self.local.lock().await;
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
