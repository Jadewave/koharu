use anyhow::Result;
use async_trait::async_trait;
use koharu_scene::{Authored, LanguageTag, Origin, SourceText, Translation};
use koharu_translator::{TranslationRequest, Translator};

use crate::TranslationConfig;

use super::{StageInput, StageProcessor, finish, generation};

const PRODUCER: &str = "dev.koharu.pipeline.translation";

pub(super) struct Processor {
    config: TranslationConfig,
    translator: Translator,
}

impl Processor {
    pub(super) fn new(config: TranslationConfig, translator: Translator) -> Self {
        Self { config, translator }
    }
}

#[async_trait]
impl StageProcessor for Processor {
    fn model(&self) -> &'static str {
        Translator::model(&self.config.model)
    }

    fn unload(&self) -> bool {
        self.translator.unload()
    }

    async fn load(&self) -> Result<()> {
        self.translator.load_model(&self.config.model).await
    }

    async fn process(&self, input: StageInput) -> Result<koharu_scene::Patch> {
        let mut targets = Vec::new();
        if let Some(group) = input.scene.page(input.page)?.text_group()? {
            for layer in group.text_layers()? {
                if !input.contains_entity(layer.id())? {
                    continue;
                }
                let content = layer.content()?;
                let Some(source) = content.source()? else {
                    continue;
                };
                if !source.text.value.trim().is_empty() {
                    targets.push((content.id(), source.text.value));
                }
            }
        }
        let mut request = TranslationRequest::new(
            targets.iter().map(|(_, source)| source.clone()),
            self.config.target_language,
        );
        if let Some(instructions) = self.config.instructions.as_deref() {
            request = request.with_instructions(instructions);
        }
        if Translator::supports_vision(&self.config.model, &self.config.generation)
            && let Some(image) = input.images.get(&input.scene, input.page, "source").await?
        {
            request = request.with_image(image);
        }
        let (provider, translated) = self
            .translator
            .translate(&self.config.model, self.config.generation, request)
            .await?;
        let language = LanguageTag::new(self.config.target_language.tag())?;
        let generated = generation(PRODUCER, provider)?;
        let mut edit = input.scene.edit_as(generated.clone());
        for (entity, _) in &targets {
            edit.observe::<SourceText>(*entity)?;
            edit.observe::<Translation>(*entity)?;
        }
        for ((entity, source), text) in targets.into_iter().zip(translated) {
            if input
                .scene
                .component::<Translation>(entity)?
                .is_some_and(|value| matches!(value.text.origin, Origin::User))
            {
                continue;
            }
            let text = if source.trim() == "\u{2026}" {
                "\u{2026}".to_owned()
            } else {
                clean_translation(&source, &text)
            };
            edit.set(
                entity,
                &Translation {
                    text: Authored::generated(text, generated.clone()),
                    language: Some(language.clone()),
                },
            )?;
        }
        finish(edit)
    }
}

/// Vertical-bar look-alikes that models emit as line separators.
const SEPARATOR_CHARACTERS: [char; 5] = ['|', '\u{ff5c}', '\u{2502}', '\u{00a6}', '\u{2016}'];

/// Removes artifacts local models leave in translations: separator bars used in place of
/// line breaks, control and zero-width characters, and replacement glyphs. Line breaks are
/// kept. A separator that already appears in the source text is left alone.
fn clean_translation(source: &str, text: &str) -> String {
    let strip_separators = !source.contains(SEPARATOR_CHARACTERS);
    let cleaned = text
        .chars()
        .filter_map(|character| match character {
            '\n' => Some('\n'),
            '\r' => None,
            character if strip_separators && SEPARATOR_CHARACTERS.contains(&character) => {
                Some(' ')
            }
            '\u{200b}' | '\u{200c}' | '\u{200d}' | '\u{feff}' | '\u{fffd}' => None,
            character if character.is_control() => Some(' '),
            character => Some(character),
        })
        .collect::<String>();
    let lines = cleaned
        .lines()
        .map(|line| line.split_whitespace().collect::<Vec<_>>().join(" "))
        .filter(|line| !line.is_empty())
        .collect::<Vec<_>>();
    if lines.is_empty() {
        text.to_owned()
    } else {
        lines.join("\n")
    }
}

#[cfg(test)]
mod tests {
    use super::clean_translation;

    #[test]
    fn separator_bars_become_spaces_and_line_breaks_survive() {
        assert_eq!(
            clean_translation("a\nb", "Tối từng\n| cô ấy | lạ"),
            "Tối từng\ncô ấy lạ"
        );
    }

    #[test]
    fn separators_in_the_source_are_preserved() {
        assert_eq!(clean_translation("a | b", "x | y"), "x | y");
    }

    #[test]
    fn empty_result_falls_back_to_the_original_text() {
        assert_eq!(clean_translation("a", "|"), "|");
    }
}
