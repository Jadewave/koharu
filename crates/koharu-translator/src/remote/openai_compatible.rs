// https://developers.openai.com/api/reference/resources/chat/subresources/completions/methods/create

use std::{
    collections::HashMap,
    sync::{Mutex, OnceLock},
    time::Duration,
};

use anyhow::Context;
use koharu_secrets::ExposeSecret;
use reqwest::Client;
use serde::{Deserialize, Serialize};
use url::Url;

use super::send_json;
use crate::{
    Error, GenerationConfig, Model, Provider, Result, TranslationRequest, backend::encode_image,
    display_name, prompt,
};

const DEFAULT_BASE_URL: &str = "http://localhost:11434/v1";

/// Sent when the generation settings give no limit. Many servers (NVIDIA NIM among them) fall
/// back to a small default, which a reasoning model spends entirely on thinking before it
/// writes a single character of the answer.
const DEFAULT_MAX_TOKENS: u32 = 16_384;

/// Request fields that some OpenAI-compatible servers refuse, in the order they are dropped
/// when the server rejects the request without naming the one it dislikes.
const OPTIONAL_FIELDS: [&str; 5] = [
    "response_format",
    "chat_template_kwargs",
    "reasoning_effort",
    "frequency_penalty",
    "presence_penalty",
];

/// Words in an error message that point at an optional field.
const FIELD_HINTS: [(&str, &[&str]); 5] = [
    (
        "response_format",
        &["response_format", "json_schema", "strict", "guided", "structured"],
    ),
    (
        "chat_template_kwargs",
        &["chat_template_kwargs", "enable_thinking", "template_kwargs"],
    ),
    ("reasoning_effort", &["reasoning_effort", "reasoning"]),
    ("frequency_penalty", &["frequency_penalty"]),
    ("presence_penalty", &["presence_penalty"]),
];

#[derive(Clone, Debug, PartialEq, serde::Serialize, serde::Deserialize, specta::Type)]
#[serde(default)]
pub struct OpenAiCompatibleConfig {
    pub base_url: Option<Url>,
}

impl Default for OpenAiCompatibleConfig {
    fn default() -> Self {
        Self {
            base_url: Some(
                Url::parse(DEFAULT_BASE_URL).expect("default OpenAI-compatible URL is valid"),
            ),
        }
    }
}

/// One request for a JSON answer: what to ask, the images that go with it (as data URLs), and
/// the JSON schema the answer follows.
pub(crate) struct Chat {
    pub(crate) system: String,
    pub(crate) user: String,
    pub(crate) images: Vec<String>,
    pub(crate) schema_name: &'static str,
    pub(crate) schema: serde_json::Value,
}

pub(super) async fn translate(
    client: &Client,
    config: &OpenAiCompatibleConfig,
    model: &str,
    generation: &GenerationConfig,
    request: &TranslationRequest,
) -> Result<Vec<String>> {
    let (system, user) = prompt::prompts(request)?;
    let images = match request.image.as_deref() {
        Some(image) => vec![encode_image(image)?.data_url()],
        None => Vec::new(),
    };
    let text = chat_json(
        client,
        config,
        model,
        generation,
        Chat {
            system,
            user,
            images,
            schema_name: "manga_translation",
            schema: prompt::output_schema(request.segments.len()),
        },
    )
    .await?;
    Ok(prompt::translations(
        "openai-compatible",
        &text,
        &request.segments,
    )?)
}

/// Sends `chat` to the server and returns the text of the answer, cleaned of the thinking and
/// the markdown fence that reasoning models wrap around a JSON answer.
pub(super) async fn chat_json(
    client: &Client,
    config: &OpenAiCompatibleConfig,
    model: &str,
    generation: &GenerationConfig,
    chat: Chat,
) -> Result<String> {
    let api_key = koharu_secrets::get("openai-compatible")?;
    let user_content = if chat.images.is_empty() {
        MessageContent::Text(chat.user)
    } else {
        let mut parts = vec![ContentPart::Text { text: chat.user }];
        parts.extend(chat.images.into_iter().map(|url| ContentPart::ImageUrl {
            image_url: ImageUrl { url },
        }));
        MessageContent::Parts(parts)
    };
    let body = ChatRequest {
        model,
        messages: [
            Message {
                role: "system",
                content: MessageContent::Text(chat.system),
            },
            Message {
                role: "user",
                content: user_content,
            },
        ],
        temperature: generation.temperature,
        top_p: generation.top_p,
        max_tokens: Some(generation.max_tokens.unwrap_or(DEFAULT_MAX_TOKENS)),
        frequency_penalty: generation.frequency_penalty,
        presence_penalty: generation.presence_penalty,
        reasoning_effort: generation
            .reasoning
            .map(|enabled| if enabled { "medium" } else { "none" }),
        // Reasoning models of vLLM and NVIDIA NIM switch thinking through the chat template.
        chat_template_kwargs: generation.reasoning.map(|enabled| ChatTemplateKwargs {
            enable_thinking: enabled,
        }),
        response_format: ResponseFormat {
            kind: "json_schema",
            json_schema: JsonSchema {
                name: chat.schema_name,
                strict: true,
                schema: chat.schema,
            },
        },
    };
    let body = serde_json::to_value(&body).context("failed to encode the request")?;
    let url = endpoint(config.base_url.as_ref(), "chat/completions");
    let key = format!("{url}|{model}");
    let mut dropped = remembered_drops(&key);
    let response: ChatResponse = loop {
        let http = client.post(&url).json(&without_fields(&body, &dropped));
        let http = match &api_key {
            Some(api_key) => http.bearer_auth(api_key.expose_secret()),
            None => http,
        };
        match send_json("openai-compatible", http).await {
            Ok(response) => break response,
            Err(error) => {
                let more = match &error {
                    Error::Api {
                        status, message, ..
                    } if matches!(*status, 400 | 422) => rejected_fields(message, &dropped),
                    _ => Vec::new(),
                };
                if more.is_empty() {
                    return Err(error);
                }
                tracing::warn!(
                    fields = ?more,
                    "openai-compatible server rejected optional request fields; sending again without them"
                );
                dropped.extend(more);
            }
        }
    };
    remember_drops(&key, &dropped);
    let choice = response
        .choices
        .into_iter()
        .next()
        .context("OpenAI-compatible backend returned no choices")?;
    Ok(answer_text(choice)?)
}

/// What the server refused before for this endpoint and model, so that a later request does
/// not fail the same way again.
fn drops() -> &'static Mutex<HashMap<String, Vec<&'static str>>> {
    static DROPS: OnceLock<Mutex<HashMap<String, Vec<&'static str>>>> = OnceLock::new();
    DROPS.get_or_init(|| Mutex::new(HashMap::new()))
}

fn remembered_drops(key: &str) -> Vec<&'static str> {
    drops()
        .lock()
        .ok()
        .and_then(|drops| drops.get(key).cloned())
        .unwrap_or_default()
}

fn remember_drops(key: &str, dropped: &[&'static str]) {
    if dropped.is_empty() {
        return;
    }
    if let Ok(mut drops) = drops().lock() {
        drops.insert(key.to_owned(), dropped.to_vec());
    }
}

/// The body without the fields in `dropped`.
fn without_fields(body: &serde_json::Value, dropped: &[&str]) -> serde_json::Value {
    let mut body = body.clone();
    if let Some(object) = body.as_object_mut() {
        for field in dropped {
            object.remove(*field);
        }
    }
    body
}

/// The optional fields a 400 or 422 answer is about. When it names none of them, the next
/// field of `OPTIONAL_FIELDS` is tried (unless the message is about something else a retry
/// cannot fix, such as the key, the image or the model), so a server that refuses a field
/// without saying which one still ends up with a request it accepts.
fn rejected_fields(message: &str, dropped: &[&'static str]) -> Vec<&'static str> {
    let message = message.to_lowercase();
    let named: Vec<&'static str> = FIELD_HINTS
        .iter()
        .filter(|(field, hints)| {
            !dropped.contains(field) && hints.iter().any(|hint| message.contains(hint))
        })
        .map(|(field, _)| *field)
        .collect();
    if !named.is_empty() {
        return named;
    }
    let unrelated = ["api key", "api_key", "authorization", "image", "model", "token", "context"];
    if unrelated.iter().any(|word| message.contains(word)) {
        return Vec::new();
    }
    OPTIONAL_FIELDS
        .iter()
        .copied()
        .find(|field| !dropped.contains(field))
        .into_iter()
        .collect()
}

/// The text of the answer. Reasoning models may leave `content` empty: when the token limit
/// ended the thinking, there is no answer at all (said so, instead of a bare "no content"); and
/// when the server did not split the thinking from the answer, the answer is in the reasoning
/// text.
fn answer_text(choice: Choice) -> anyhow::Result<String> {
    let message = choice.message;
    let content = message.content.map(Content::text).unwrap_or_default();
    let cleaned = clean_output(&content);
    if !cleaned.trim().is_empty() {
        return Ok(cleaned);
    }
    for reasoning in [message.reasoning_content, message.reasoning]
        .into_iter()
        .flatten()
    {
        if let Some(object) = last_translations_object(&reasoning) {
            return Ok(object.to_owned());
        }
    }
    if choice.finish_reason.as_deref() == Some("length") {
        anyhow::bail!(
            "OpenAI-compatible backend reached the token limit before it wrote an answer; the \
             model spent its tokens on thinking. Turn reasoning off or raise the maximum tokens."
        );
    }
    anyhow::bail!("OpenAI-compatible backend returned no message content")
}

/// Removes what reasoning models wrap around the JSON: a thinking block that was left in the
/// text, and one surrounding markdown fence. A response that is still not JSON after this is
/// reported as invalid, like any other.
fn clean_output(text: &str) -> String {
    let mut text = text.trim();
    if let Some(end) = text.rfind("</think>") {
        text = text[end + "</think>".len()..].trim();
    } else if text.starts_with("<think>") {
        // The thinking never ended: there is no answer in it.
        return String::new();
    }
    if let Some(rest) = text.strip_prefix("```") {
        let rest = rest.trim_start_matches(|c: char| c.is_ascii_alphabetic());
        if let Some(body) = rest.trim().strip_suffix("```") {
            text = body.trim();
        }
    }
    text.to_owned()
}

/// The last `{ "translations": ... }` object in a text that mixes thinking and an answer.
fn last_translations_object(text: &str) -> Option<&str> {
    let key = text.rfind("\"translations\"")?;
    let start = text[..key].rfind('{')?;
    let mut depth = 0usize;
    let mut in_string = false;
    let mut escaped = false;
    for (offset, c) in text[start..].char_indices() {
        if in_string {
            match c {
                _ if escaped => escaped = false,
                '\\' => escaped = true,
                '"' => in_string = false,
                _ => {}
            }
            continue;
        }
        match c {
            '"' => in_string = true,
            '{' => depth += 1,
            '}' => {
                depth -= 1;
                if depth == 0 {
                    let object = &text[start..=start + offset];
                    return serde_json::from_str::<serde_json::Value>(object)
                        .is_ok()
                        .then_some(object);
                }
            }
            _ => {}
        }
    }
    None
}

pub(super) async fn models(client: &Client, config: &OpenAiCompatibleConfig) -> Result<Vec<Model>> {
    let api_key = koharu_secrets::get("openai-compatible")?;
    // A hosted server (NVIDIA lists over a hundred models) takes longer than the 5 seconds of
    // the client that discovers local servers.
    let request = client
        .get(endpoint(config.base_url.as_ref(), "models"))
        .timeout(Duration::from_secs(30));
    let request = match api_key {
        Some(api_key) => request.bearer_auth(api_key.expose_secret()),
        None => request,
    };
    let response: ModelsResponse = send_json("openai-compatible", request).await?;
    Ok(response
        .data
        .into_iter()
        .map(|model| Model {
            provider: Provider::OpenAiCompatible,
            name: display_name(&model.id),
            model: Some(model.id),
            quantizations: Vec::new(),
            vision: true,
            reasoning: true,
        })
        .collect())
}

fn endpoint(base_url: Option<&Url>, suffix: &str) -> String {
    let base_url = base_url.map_or(DEFAULT_BASE_URL, Url::as_str);
    format!(
        "{}/{}",
        base_url.trim_end_matches('/'),
        suffix.trim_start_matches('/')
    )
}

#[derive(Serialize)]
struct ChatRequest<'a> {
    model: &'a str,
    messages: [Message; 2],
    #[serde(skip_serializing_if = "Option::is_none")]
    temperature: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    top_p: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    max_tokens: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    frequency_penalty: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    presence_penalty: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    reasoning_effort: Option<&'static str>,
    #[serde(skip_serializing_if = "Option::is_none")]
    chat_template_kwargs: Option<ChatTemplateKwargs>,
    response_format: ResponseFormat,
}

#[derive(Serialize)]
struct ChatTemplateKwargs {
    enable_thinking: bool,
}

#[derive(Serialize)]
struct ResponseFormat {
    #[serde(rename = "type")]
    kind: &'static str,
    json_schema: JsonSchema,
}

#[derive(Serialize)]
struct JsonSchema {
    name: &'static str,
    strict: bool,
    schema: serde_json::Value,
}

#[derive(Serialize)]
struct Message {
    role: &'static str,
    content: MessageContent,
}

#[derive(Serialize)]
#[serde(untagged)]
enum MessageContent {
    Text(String),
    Parts(Vec<ContentPart>),
}

#[derive(Serialize)]
#[serde(tag = "type", rename_all = "snake_case")]
enum ContentPart {
    Text { text: String },
    ImageUrl { image_url: ImageUrl },
}

#[derive(Serialize)]
struct ImageUrl {
    url: String,
}

#[derive(Deserialize)]
struct ChatResponse {
    choices: Vec<Choice>,
}

#[derive(Deserialize)]
struct Choice {
    message: ResponseMessage,
    #[serde(default)]
    finish_reason: Option<String>,
}

#[derive(Deserialize)]
struct ResponseMessage {
    #[serde(default)]
    content: Option<Content>,
    /// The thinking of a reasoning model, under the name vLLM and NVIDIA NIM use.
    #[serde(default)]
    reasoning_content: Option<String>,
    /// The same, under the name other servers use.
    #[serde(default)]
    reasoning: Option<String>,
}

/// A message content: a string, or (on some servers) a list of text parts.
#[derive(Deserialize)]
#[serde(untagged)]
enum Content {
    Text(String),
    Parts(Vec<ContentText>),
}

#[derive(Deserialize)]
struct ContentText {
    #[serde(default)]
    text: Option<String>,
}

impl Content {
    fn text(self) -> String {
        match self {
            Self::Text(text) => text,
            Self::Parts(parts) => parts.into_iter().filter_map(|part| part.text).collect(),
        }
    }
}

#[derive(Deserialize)]
struct ModelsResponse {
    data: Vec<ListedModel>,
}

#[derive(Deserialize)]
struct ListedModel {
    id: String,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn endpoint_preserves_base_path() {
        let url = Url::parse("http://localhost:1234/v1").unwrap();
        assert_eq!(
            endpoint(Some(&url), "models"),
            "http://localhost:1234/v1/models"
        );
    }

    #[test]
    fn serializes_compatible_request_contract() {
        let body = ChatRequest {
            model: "model",
            messages: [
                Message {
                    role: "system",
                    content: MessageContent::Text("system".to_owned()),
                },
                Message {
                    role: "user",
                    content: MessageContent::Text("user".to_owned()),
                },
            ],
            temperature: None,
            top_p: None,
            max_tokens: Some(1024),
            frequency_penalty: None,
            presence_penalty: None,
            reasoning_effort: Some("none"),
            chat_template_kwargs: Some(ChatTemplateKwargs {
                enable_thinking: false,
            }),
            response_format: ResponseFormat {
                kind: "json_schema",
                json_schema: JsonSchema {
                    name: "manga_translation",
                    strict: true,
                    schema: prompt::output_schema(2),
                },
            },
        };
        let value = serde_json::to_value(body).unwrap();

        assert_eq!(value["max_tokens"], 1024);
        assert_eq!(value["reasoning_effort"], "none");
        assert_eq!(value["chat_template_kwargs"]["enable_thinking"], false);
        assert_eq!(value["response_format"]["type"], "json_schema");
        assert_eq!(value["response_format"]["json_schema"]["strict"], true);
        assert_eq!(
            value["response_format"]["json_schema"]["schema"]["properties"]["translations"]["maxItems"],
            2
        );
    }

    #[test]
    fn serializes_text_before_an_attached_image() {
        let content = MessageContent::Parts(vec![
            ContentPart::Text {
                text: "translate".to_owned(),
            },
            ContentPart::ImageUrl {
                image_url: ImageUrl {
                    url: "data:image/jpeg;base64,image".to_owned(),
                },
            },
        ]);
        let value = serde_json::to_value(content).unwrap();

        assert_eq!(
            value[0],
            serde_json::json!({ "type": "text", "text": "translate" })
        );
        assert_eq!(value[1]["type"], "image_url");
        assert_eq!(value[1]["image_url"]["url"], "data:image/jpeg;base64,image");
    }

    fn choice(content: Option<&str>, reasoning: Option<&str>, finish: Option<&str>) -> Choice {
        Choice {
            message: ResponseMessage {
                content: content.map(|text| Content::Text(text.to_owned())),
                reasoning_content: reasoning.map(str::to_owned),
                reasoning: None,
            },
            finish_reason: finish.map(str::to_owned),
        }
    }

    const ANSWER: &str = r#"{"translations":[{"id":0,"text":"Chào"}]}"#;

    #[test]
    fn the_answer_is_taken_from_content_without_thinking_or_fences() {
        let fenced = format!("```json\n{ANSWER}\n```");
        let thinking = format!("<think>hmm {{ not json }}</think>\n{ANSWER}");
        for content in [ANSWER, fenced.as_str(), thinking.as_str()] {
            let text = answer_text(choice(Some(content), None, Some("stop"))).unwrap();
            assert_eq!(text, ANSWER, "{content}");
        }
    }

    #[test]
    fn an_answer_that_a_server_left_in_the_reasoning_text_is_found() {
        let reasoning = format!("Draft: {{\"translations\":[{{\"id\":0,\"text\":\"a\"}}]}} final: {ANSWER}");
        for content in [None, Some(""), Some("  ")] {
            let text = answer_text(choice(content, Some(&reasoning), Some("stop"))).unwrap();
            assert_eq!(text, ANSWER);
        }
    }

    #[test]
    fn a_response_cut_off_while_thinking_says_so() {
        let error = answer_text(choice(None, Some("let me think"), Some("length"))).unwrap_err();
        assert!(error.to_string().contains("token limit"), "{error}");
        let unfinished = answer_text(choice(Some("<think>still"), None, Some("length"))).unwrap_err();
        assert!(unfinished.to_string().contains("token limit"));
        let empty = answer_text(choice(None, None, Some("stop"))).unwrap_err();
        assert!(empty.to_string().contains("no message content"));
    }

    #[test]
    fn content_may_be_a_list_of_text_parts() {
        let value = serde_json::json!({
            "choices": [{"message": {"content": [{"type": "text", "text": ANSWER}]}, "finish_reason": "stop"}]
        });
        let response: ChatResponse = serde_json::from_value(value).unwrap();
        let text = answer_text(response.choices.into_iter().next().unwrap()).unwrap();
        assert_eq!(text, ANSWER);
    }

    #[test]
    fn a_rejected_field_is_found_in_the_message_or_by_elimination() {
        let none: [&'static str; 0] = [];
        assert_eq!(
            rejected_fields("Unsupported value: 'response_format.json_schema.strict'", &none),
            ["response_format"]
        );
        assert_eq!(
            rejected_fields("extra inputs are not permitted: chat_template_kwargs", &none),
            ["chat_template_kwargs"]
        );
        // No field named: the next one is tried; nothing is left to try at the end.
        assert_eq!(rejected_fields("Bad request", &none), ["response_format"]);
        assert_eq!(
            rejected_fields("Bad request", &["response_format"]),
            ["chat_template_kwargs"]
        );
        assert!(rejected_fields("Bad request", &OPTIONAL_FIELDS).is_empty());
        // Problems that a retry without a field cannot fix.
        assert!(rejected_fields("Invalid API key provided", &none).is_empty());
        assert!(rejected_fields("image is too large", &none).is_empty());
    }

    #[test]
    fn dropped_fields_leave_the_rest_of_the_body() {
        let body = serde_json::json!({"model": "m", "response_format": {}, "max_tokens": 5});
        let value = without_fields(&body, &["response_format", "reasoning_effort"]);
        assert_eq!(value, serde_json::json!({"model": "m", "max_tokens": 5}));
    }
}
