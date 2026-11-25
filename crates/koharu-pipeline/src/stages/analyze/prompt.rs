//! What the analyzer is asked, and how its answer is read.

use anyhow::{Context as _, Result, bail};
use serde::Deserialize;
use serde_json::{Value, json};

/// The analyzer's system prompt.
pub(super) const SYSTEM: &str = "\
You analyse manga pages so that a translator who can't see them translates every line correctly. For each page you get its OCR lines as JSON, and an image of the page with its lettering erased and each line's place outlined in red and tagged with the line's id.

Answer with one JSON object:
- \"characters\": each character who speaks or is spoken to in the new pages: \"name\" (the name the work gives them, or a short description such as \"girl with twin tails\") and \"description\" (age, role, relation to the others, how they speak).
- \"pages\": one entry for each new page, with its \"page\" number, a \"summary\" of one to three sentences on what happens, and \"lines\": an entry for every OCR line of the page, with its \"id\", its \"speaker\" (a character's name, \"narrator\", \"sfx\" or \"unknown\"), its \"tone\" (a few words, such as \"teasing, casual\" or \"shouting, angry\"), and \"text\": the line corrected if the OCR misread it, otherwise null.

Pages marked \"context only\" were analysed already. Use them to follow the story and to keep names the same, but leave them out of \"pages\". Output only the JSON.";

/// The shape of an answer, for a server that can be told one.
pub(super) fn schema() -> Result<schemars::Schema> {
    schemars::Schema::try_from(json!({
        "type": "object",
        "properties": {
            "characters": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string"},
                        "description": {"type": "string"},
                    },
                    "required": ["name", "description"],
                },
            },
            "pages": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "page": {"type": "integer"},
                        "summary": {"type": "string"},
                        "lines": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "id": {"type": "integer"},
                                    "speaker": {"type": "string"},
                                    "tone": {"type": "string"},
                                    "text": {"type": ["string", "null"]},
                                },
                                "required": ["id", "speaker", "tone", "text"],
                            },
                        },
                    },
                    "required": ["page", "summary", "lines"],
                },
            },
        },
        "required": ["characters", "pages"],
    }))
    .context("the analyzer's schema isn't valid")
}

/// One OCR line as the prompt gives it.
pub(super) fn lines(texts: &[String]) -> String {
    let lines: Vec<Value> = texts
        .iter()
        .enumerate()
        .map(|(at, text)| json!({"id": at + 1, "text": text}))
        .collect();
    Value::Array(lines).to_string()
}

#[derive(Debug, Default, Deserialize)]
pub(super) struct Reply {
    #[serde(default)]
    pub(super) characters: Vec<CharacterReply>,
    #[serde(default)]
    pub(super) pages: Vec<PageReply>,
}

#[derive(Debug, Deserialize)]
pub(super) struct CharacterReply {
    #[serde(default)]
    pub(super) name: String,
    #[serde(default)]
    pub(super) description: String,
}

#[derive(Debug, Deserialize)]
pub(super) struct PageReply {
    pub(super) page: usize,
    #[serde(default)]
    pub(super) summary: String,
    #[serde(default)]
    pub(super) lines: Vec<LineReply>,
}

#[derive(Debug, Deserialize)]
pub(super) struct LineReply {
    pub(super) id: usize,
    #[serde(default)]
    pub(super) speaker: String,
    #[serde(default)]
    pub(super) tone: String,
    #[serde(default)]
    pub(super) text: Option<String>,
}

/// The answer's object, found whether the model fenced it in Markdown or
/// wrote words around it. An answer with no page in it is no answer.
pub(super) fn parse(answer: &str) -> Result<Reply> {
    let answer = answer.trim();
    let from = answer.find('{');
    let to = answer.rfind('}');
    let (Some(from), Some(to)) = (from, to) else {
        bail!("the answer holds no JSON object");
    };
    let reply: Reply = serde_json::from_str(&answer[from..=to])
        .context("the answer isn't the JSON object asked for")?;
    if reply.pages.is_empty() {
        bail!("the answer has no pages");
    }
    Ok(reply)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn an_answer_is_found_inside_a_fence() {
        let reply = parse(
            "```json\n{\"characters\": [], \"pages\": [{\"page\": 3, \"summary\": \"Hi\", \"lines\": [{\"id\": 1, \"speaker\": \"Ann\", \"tone\": \"calm\", \"text\": null}]}]}\n```",
        )
        .expect("an answer");
        assert_eq!(reply.pages[0].page, 3);
        assert_eq!(reply.pages[0].lines[0].speaker, "Ann");
    }

    #[test]
    fn an_answer_without_pages_fails() {
        assert!(parse("{\"characters\": []}").is_err());
        assert!(parse("no JSON here").is_err());
    }

    #[test]
    fn lines_are_numbered_from_one() {
        assert_eq!(
            lines(&["やあ".to_owned(), "うん".to_owned()]),
            r#"[{"id":1,"text":"やあ"},{"id":2,"text":"うん"}]"#
        );
    }
}
