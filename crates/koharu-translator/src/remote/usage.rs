//! The tokens the models of the providers have used.
//!
//! Every provider answers through `send_json`, which hands the body of a successful answer to
//! `record`. It reads the usage the way each API reports it, so no provider needs its own code:
//!
//! - `usage.prompt_tokens`, `completion_tokens`, `total_tokens` (OpenAI and everything that
//!   speaks like it: DeepSeek, Grok, OpenRouter, MiniMax, LM Studio, OpenAI-compatible servers);
//! - `usage.input_tokens`, `output_tokens` (Claude, the OpenAI responses API);
//! - `usageMetadata.promptTokenCount`, `candidatesTokenCount`, `totalTokenCount` (Gemini).
//!
//! The counts are by provider and model, for the whole run of the program. The local model is
//! not counted: it uses no provider. Translation services that bill by characters (DeepL, Google
//! Cloud, Caiyun) report no tokens.

use std::{collections::BTreeMap, fmt, sync::Mutex};

use serde::Serialize;
use serde_json::Value;

/// What a set of requests used.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize)]
pub struct Tokens {
    pub requests: u64,
    /// Tokens sent to the model (prompt, pictures included).
    pub input: u64,
    /// Tokens the model wrote (thinking included).
    pub output: u64,
    /// What the provider counted in all; at least `input + output`.
    pub total: u64,
}

impl Tokens {
    fn add(&mut self, other: Tokens) {
        self.requests += other.requests;
        self.input += other.input;
        self.output += other.output;
        self.total += other.total;
    }

    fn minus(self, before: Tokens) -> Tokens {
        Tokens {
            requests: self.requests.saturating_sub(before.requests),
            input: self.input.saturating_sub(before.input),
            output: self.output.saturating_sub(before.output),
            total: self.total.saturating_sub(before.total),
        }
    }
}

impl fmt::Display for Tokens {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "{} tokens ({} in, {} out) in {} requests",
            self.total, self.input, self.output, self.requests
        )
    }
}

/// The tokens used so far, by `provider/model`.
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize)]
pub struct TokenUsage {
    pub models: BTreeMap<String, Tokens>,
}

impl TokenUsage {
    /// The sum over every model.
    #[must_use]
    pub fn total(&self) -> Tokens {
        let mut sum = Tokens::default();
        for tokens in self.models.values() {
            sum.add(*tokens);
        }
        sum
    }

    /// What was used since `before` (an earlier snapshot).
    #[must_use]
    pub fn since(&self, before: &TokenUsage) -> TokenUsage {
        let models = self
            .models
            .iter()
            .filter_map(|(name, now)| {
                let used = now.minus(before.models.get(name).copied().unwrap_or_default());
                (used.requests > 0).then(|| (name.clone(), used))
            })
            .collect();
        TokenUsage { models }
    }
}

impl fmt::Display for TokenUsage {
    /// The total, then every model: `1200 tokens (...) in 3 requests [a/b: ..., c/d: ...]`.
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.total())?;
        if !self.models.is_empty() {
            let parts: Vec<String> = self
                .models
                .iter()
                .map(|(name, tokens)| format!("{name}: {} (in {}, out {})", tokens.total, tokens.input, tokens.output))
                .collect();
            write!(f, " [{}]", parts.join("; "))?;
        }
        Ok(())
    }
}

static USAGE: Mutex<BTreeMap<String, Tokens>> = Mutex::new(BTreeMap::new());

/// The tokens every model of every provider has used since the program started or since
/// `reset_token_usage`.
#[must_use]
pub fn token_usage() -> TokenUsage {
    TokenUsage {
        models: USAGE.lock().map(|usage| usage.clone()).unwrap_or_default(),
    }
}

/// Starts counting from zero.
pub fn reset_token_usage() {
    if let Ok(mut usage) = USAGE.lock() {
        usage.clear();
    }
}

fn number(value: &Value, key: &str) -> Option<u64> {
    value.get(key).and_then(Value::as_u64)
}

/// The usage an answer reports, and the model it names (empty when it names none).
fn read(body: &Value) -> Option<(String, Tokens)> {
    let (input, output, total) = if let Some(usage) = body.get("usage").filter(|usage| usage.is_object()) {
        let input = number(usage, "prompt_tokens").or_else(|| number(usage, "input_tokens"));
        let output = number(usage, "completion_tokens").or_else(|| number(usage, "output_tokens"));
        (input, output, number(usage, "total_tokens"))
    } else if let Some(usage) = body.get("usageMetadata").filter(|usage| usage.is_object()) {
        let input = number(usage, "promptTokenCount");
        let total = number(usage, "totalTokenCount");
        // Thinking tokens are not in `candidatesTokenCount`; the total holds them.
        let output = match (total, input) {
            (Some(total), Some(input)) => Some(total.saturating_sub(input)),
            _ => number(usage, "candidatesTokenCount"),
        };
        (input, output, total)
    } else {
        return None;
    };
    if input.is_none() && output.is_none() && total.is_none() {
        return None;
    }
    let (input, output) = (input.unwrap_or(0), output.unwrap_or(0));
    let model = body
        .get("model")
        .or_else(|| body.get("modelVersion"))
        .and_then(Value::as_str)
        .unwrap_or_default()
        .trim_start_matches("models/")
        .to_owned();
    Some((
        model,
        Tokens {
            requests: 1,
            input,
            output,
            total: total.unwrap_or(0).max(input + output),
        },
    ))
}

/// Counts the usage in the body of a successful answer of `provider`. An answer without usage
/// (a list of models, a translation service) counts for nothing.
pub(crate) fn record(provider: &str, body: &str) {
    // Most answers that carry no usage are cheap to skip.
    if !body.contains("usage") {
        return;
    }
    let Ok(value) = serde_json::from_str::<Value>(body) else {
        return;
    };
    let Some((model, tokens)) = read(&value) else {
        return;
    };
    let name = if model.is_empty() { provider.to_owned() } else { format!("{provider}/{model}") };
    tracing::debug!(%name, %tokens, "tokens used by a request");
    if let Ok(mut usage) = USAGE.lock() {
        usage.entry(name).or_default().add(tokens);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tokens(body: &str) -> (String, Tokens) {
        read(&serde_json::from_str(body).unwrap()).unwrap()
    }

    #[test]
    fn an_openai_style_answer_is_read() {
        let (model, used) = tokens(
            r#"{"model": "deepseek-ai/deepseek-v4.1-flash", "choices": [],
                "usage": {"prompt_tokens": 1200, "completion_tokens": 300, "total_tokens": 1500}}"#,
        );
        assert_eq!(model, "deepseek-ai/deepseek-v4.1-flash");
        assert_eq!(used, Tokens { requests: 1, input: 1200, output: 300, total: 1500 });
    }

    #[test]
    fn a_claude_answer_has_no_total_and_it_is_the_sum() {
        let (_, used) = tokens(r#"{"model": "claude-x", "usage": {"input_tokens": 10, "output_tokens": 5}}"#);
        assert_eq!(used, Tokens { requests: 1, input: 10, output: 5, total: 15 });
    }

    #[test]
    fn a_gemini_answer_counts_the_thinking_in_the_output() {
        let (model, used) = tokens(
            r#"{"modelVersion": "gemini-x", "usageMetadata": {"promptTokenCount": 100,
                "candidatesTokenCount": 20, "thoughtsTokenCount": 80, "totalTokenCount": 200}}"#,
        );
        assert_eq!(model, "gemini-x");
        assert_eq!(used, Tokens { requests: 1, input: 100, output: 100, total: 200 });
    }

    #[test]
    fn an_answer_without_usage_counts_for_nothing() {
        assert!(read(&serde_json::from_str(r#"{"data": []}"#).unwrap()).is_none());
        assert!(read(&serde_json::from_str(r#"{"usage": null}"#).unwrap()).is_none());
        assert!(read(&serde_json::from_str(r#"{"usage": {}}"#).unwrap()).is_none());
    }

    #[test]
    fn the_counts_add_up_by_model_and_a_snapshot_tells_what_came_after() {
        // A provider name of its own: other tests share the counter.
        record("test-provider", r#"{"model": "m", "usage": {"prompt_tokens": 10, "completion_tokens": 2}}"#);
        let before = token_usage();
        record("test-provider", r#"{"model": "m", "usage": {"prompt_tokens": 5, "completion_tokens": 1, "total_tokens": 7}}"#);
        record("test-provider", r#"{"model": "n", "usage": {"prompt_tokens": 1, "completion_tokens": 1}}"#);
        record("test-provider", "not json with usage");
        let used = token_usage().since(&before);
        assert_eq!(used.models["test-provider/m"], Tokens { requests: 1, input: 5, output: 1, total: 7 });
        assert_eq!(used.models["test-provider/n"].total, 2);
        assert_eq!(used.total().requests, 2);
        let line = used.to_string();
        assert!(line.starts_with("9 tokens (6 in, 2 out) in 2 requests ["), "{line}");
        assert!(line.contains("test-provider/m: 7"));
        assert!(token_usage().models["test-provider/m"].input >= 15);
    }
}
