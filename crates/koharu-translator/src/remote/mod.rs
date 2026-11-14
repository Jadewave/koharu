mod caiyun;
mod claude;
mod deepl;
mod deepseek;
mod gemini;
mod google_cloud;
mod grok;
mod lm_studio;
mod minimax;
mod openai;
mod openai_compatible;
mod openrouter;
mod usage;

use anyhow::Context;
use futures::{FutureExt, future::BoxFuture, future::join_all};
use reqwest::{Client, RequestBuilder, StatusCode};
use serde::de::DeserializeOwned;

pub use caiyun::CaiyunConfig;
pub use claude::ClaudeConfig;
pub use deepl::DeepLConfig;
pub use deepseek::DeepSeekConfig;
pub use gemini::GeminiConfig;
pub use google_cloud::GoogleCloudConfig;
pub use grok::GrokConfig;
pub use lm_studio::LmStudioConfig;
pub use minimax::MiniMaxConfig;
pub use openai::OpenAiConfig;
pub use openai_compatible::OpenAiCompatibleConfig;
pub(crate) use openai_compatible::Chat;
pub use usage::{TokenUsage, Tokens, reset_token_usage, token_usage};
pub use openrouter::OpenRouterConfig;

use crate::{
    Error, GenerationConfig, Model, ModelSelection, Provider, ProvidersConfig, Result,
    TranslationRequest,
};

pub(crate) async fn translate(
    client: &Client,
    providers: &ProvidersConfig,
    selection: &ModelSelection,
    generation: &GenerationConfig,
    request: &TranslationRequest,
) -> Result<Vec<String>> {
    let model = || {
        selection
            .model
            .as_deref()
            .with_context(|| format!("{} requires a selected model", selection.provider))
    };

    match selection.provider {
        Provider::OpenAi => {
            openai::translate(client, &providers.openai, model()?, generation, request).await
        }
        Provider::Gemini => {
            gemini::translate(client, &providers.gemini, model()?, generation, request).await
        }
        Provider::Claude => {
            claude::translate(client, &providers.claude, model()?, generation, request).await
        }
        Provider::Grok => {
            grok::translate(client, &providers.grok, model()?, generation, request).await
        }
        Provider::MiniMax => {
            minimax::translate(client, &providers.minimax, model()?, generation, request).await
        }
        Provider::DeepSeek => {
            deepseek::translate(client, &providers.deepseek, model()?, generation, request).await
        }
        Provider::OpenAiCompatible => {
            openai_compatible::translate(
                client,
                &providers.openai_compatible,
                model()?,
                generation,
                request,
            )
            .await
        }
        Provider::OpenRouter => {
            openrouter::translate(client, &providers.openrouter, model()?, generation, request)
                .await
        }
        Provider::LmStudio => {
            lm_studio::translate(client, &providers.lm_studio, model()?, generation, request)
                .await
        }
        Provider::DeepL => deepl::translate(client, &providers.deepl, request).await,
        Provider::GoogleCloudTranslation => {
            google_cloud::translate(client, &providers.google_cloud_translation, request).await
        }
        Provider::Caiyun => caiyun::translate(client, &providers.caiyun, request).await,
        Provider::Local => unreachable!("local translation has its own backend"),
    }
}

/// Asks an OpenAI-compatible server (the one of `providers.openai_compatible`) for a JSON
/// answer and returns its text.
pub(crate) async fn chat_json(
    client: &Client,
    providers: &ProvidersConfig,
    model: &str,
    generation: &GenerationConfig,
    chat: Chat,
) -> Result<String> {
    openai_compatible::chat_json(client, &providers.openai_compatible, model, generation, chat)
        .await
}

pub(crate) async fn models(client: &Client, providers: &ProvidersConfig) -> Vec<Model> {
    let pending: Vec<(&'static str, BoxFuture<'_, Result<Vec<Model>>>)> = vec![
        ("openai", openai::models(client).boxed()),
        ("gemini", gemini::models(client).boxed()),
        ("claude", claude::models(client).boxed()),
        ("grok", grok::models(client).boxed()),
        ("minimax", minimax::models(client).boxed()),
        ("deepseek", deepseek::models(client).boxed()),
        (
            "openai-compatible",
            openai_compatible::models(client, &providers.openai_compatible).boxed(),
        ),
        ("openrouter", openrouter::models(client).boxed()),
        (
            "lm-studio",
            lm_studio::models(client, &providers.lm_studio).boxed(),
        ),
        ("deepl", deepl::models().boxed()),
        (
            "google-cloud-translation",
            google_cloud::models().boxed(),
        ),
        ("caiyun", caiyun::models().boxed()),
    ];

    let results = join_all(
        pending
            .into_iter()
            .map(|(provider, future)| async move { (provider, future.await) }),
    )
    .await;

    let mut models = Vec::new();

    for (provider, result) in results {
        match result {
            Ok(mut provider_models) => {
                tracing::debug!(
                    provider,
                    count = provider_models.len(),
                    "translation models discovered"
                );
                models.append(&mut provider_models);
            }
            Err(error) => {
                // The whole chain, not only "request failed": it says whether the server
                // refused the connection, timed out, or could not be resolved.
                let detail = format!("{error:#}");
                // A server on this machine that nobody started (the defaults of
                // OpenAI-compatible and LM Studio) is the normal state of a machine
                // without one, for instance a hosted space: not worth a warning.
                let probing_local = match provider {
                    "openai-compatible" => {
                        is_loopback(providers.openai_compatible.base_url.as_ref())
                    }
                    "lm-studio" => is_loopback(providers.lm_studio.base_url.as_ref()),
                    _ => false,
                };
                if probing_local {
                    tracing::trace!(
                        provider,
                        error = %detail,
                        "no local server to list models from"
                    );
                } else {
                    tracing::warn!(
                        provider,
                        error = %detail,
                        "failed to list translation models"
                    );
                }
            }
        }
    }

    models.sort_by(|left, right| {
        left.provider
            .to_string()
            .cmp(&right.provider.to_string())
            .then_with(|| left.model.cmp(&right.model))
    });

    models.dedup_by(|left, right| {
        left.provider == right.provider && left.model == right.model
    });

    models
}

/// Whether a server URL is on this machine (an unset URL stands for the local default).
pub(crate) fn is_loopback(url: Option<&url::Url>) -> bool {
    url.is_none_or(|url| {
        matches!(
            url.host_str(),
            Some("localhost" | "127.0.0.1" | "[::1]" | "::1" | "0.0.0.0")
        )
    })
}

pub(super) async fn send_json<T: DeserializeOwned>(
    provider: &'static str,
    request: RequestBuilder,
) -> Result<T> {
    let response = request
        .send()
        .await
        .with_context(|| format!("{provider} request failed"))?;

    let status = response.status();
    let url = response.url().to_string();

    let text = response
        .text()
        .await
        .with_context(|| format!("failed to read {provider} response"))?;

    // The whole answer of the server, before anything reads it. Shown with
    // `RUST_LOG=info,koharu_translator::http=debug` (or `--debug`).
    tracing::debug!(
        target: "koharu_translator::http",
        provider,
        %url,
        status = status.as_u16(),
        body = %text,
        "response"
    );

    if !status.is_success() {
        let lower = text.to_ascii_lowercase();

        if status == StatusCode::TOO_MANY_REQUESTS
            || lower.contains("insufficient_quota")
            || lower.contains("resource_exhausted")
            || lower.contains("rate limit exceeded")
            || lower.contains("credit balance is too low")
        {
            return Err(Error::QuotaExceeded { provider });
        }

        return Err(Error::Api {
            provider,
            status: status.as_u16(),
            message: text.chars().take(4096).collect(),
        });
    }

    usage::record(provider, &text);

    serde_json::from_str(&text)
        .with_context(|| format!("failed to decode {provider} response"))
        .map_err(Into::into)
}
