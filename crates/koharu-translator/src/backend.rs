use std::{io::Cursor, sync::Arc};

use anyhow::Context;
use base64::{Engine as _, engine::general_purpose::STANDARD};
use fast_image_resize::{FilterType, ResizeAlg, ResizeOptions, Resizer};
use image::{DynamicImage, ImageEncoder, RgbImage, codecs::jpeg::JpegEncoder};

use crate::Language;

const MAX_IMAGE_DIMENSION: u32 = 2048;
const JPEG_QUALITY: u8 = 88;

/// The words a line must use for "I" and "you", as the notes of the analysis decided them for
/// its speaker and listener.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ExpectedPair {
    pub speaker: String,
    pub listener: String,
    /// What the speaker calls themselves.
    pub mine: String,
    /// What the speaker calls the listener.
    pub yours: String,
}

#[derive(Debug, Clone, PartialEq)]
pub struct TranslationRequest {
    pub segments: Vec<String>,
    pub source_language: Option<Language>,
    pub target_language: Language,
    pub instructions: Option<String>,
    pub context: Vec<TranslationContext>,
    pub image: Option<Arc<DynamicImage>>,
    /// One image per page when several pages are translated in one request, in page order.
    /// Used by the local model instead of `image`.
    pub page_images: Vec<Arc<DynamicImage>>,
    /// For each segment, the index of its page among `page_images`. Empty for one page.
    pub segment_pages: Vec<usize>,
    /// For each segment, the words its notes decided for it (`None` where the notes give none).
    /// Empty when the page has no notes. The translation is held to them.
    pub pairs: Vec<Option<ExpectedPair>>,
    /// Asks the model to write `speaker` before every `text` although no page image is attached,
    /// so that what it wrote can be checked against `pairs`.
    pub speakers: bool,
}

impl TranslationRequest {
    #[must_use]
    pub fn new(
        segments: impl IntoIterator<Item = impl Into<String>>,
        target_language: Language,
    ) -> Self {
        Self {
            segments: segments.into_iter().map(Into::into).collect(),
            source_language: None,
            target_language,
            instructions: None,
            context: Vec::new(),
            image: None,
            page_images: Vec::new(),
            segment_pages: Vec::new(),
            pairs: Vec::new(),
            speakers: false,
        }
    }

    #[cfg(test)]
    #[must_use]
    pub fn with_source_language(mut self, language: Language) -> Self {
        self.source_language = Some(language);
        self
    }

    #[must_use]
    pub fn with_instructions(mut self, instructions: impl Into<String>) -> Self {
        self.instructions = Some(instructions.into());
        self
    }

    #[must_use]
    pub fn with_image(mut self, image: Arc<DynamicImage>) -> Self {
        self.image = Some(image);
        self
    }

    /// Translates the segments of several consecutive pages in one request: `images` holds one
    /// image per page and `segment_pages` the page of every segment.
    #[must_use]
    pub fn with_pages(
        mut self,
        images: Vec<Arc<DynamicImage>>,
        segment_pages: Vec<usize>,
    ) -> Self {
        self.page_images = images;
        self.segment_pages = segment_pages;
        self
    }

    /// The words the notes of the page decided for every segment (see `pairs`). `pairs` must
    /// have one entry for each segment, else it is ignored.
    #[must_use]
    pub fn with_pairs(mut self, pairs: Vec<Option<ExpectedPair>>) -> Self {
        if pairs.len() == self.segments.len() {
            self.pairs = pairs;
        }
        self
    }

    /// Makes the model write who speaks to whom before each translation (see `speakers`).
    #[must_use]
    pub fn with_speakers(mut self) -> Self {
        self.speakers = true;
        self
    }

    /// Whether the page was read by the analysis: its notes are in the instructions.
    pub(crate) fn has_notes(&self) -> bool {
        !self.pairs.is_empty()
    }

    /// The same request restricted to the segments at `indices`, keeping their pages.
    pub(crate) fn select(&self, indices: &[usize]) -> Self {
        let mut subset = self.clone();
        subset.segments = indices
            .iter()
            .map(|&index| self.segments[index].clone())
            .collect();
        if self.segment_pages.len() == self.segments.len() {
            subset.segment_pages = indices
                .iter()
                .map(|&index| self.segment_pages[index])
                .collect();
        }
        if self.pairs.len() == self.segments.len() {
            subset.pairs = indices.iter().map(|&index| self.pairs[index].clone()).collect();
        }
        subset
    }

    /// Whether the model writes a `speaker` (who speaks to whom, and the pronoun pair) before
    /// each translation. Only requests built for the local model ask for it: those with page
    /// images, and those that were told to (`speakers`, for a page with notes, which has no
    /// image). Only when the target is Vietnamese, the language that needs a pronoun pair.
    pub(crate) fn wants_speaker(&self) -> bool {
        self.target_language == Language::Vietnamese
            && (!self.segment_pages.is_empty() || self.speakers)
    }

    /// Drops every image: retries are text only because encoding images on a CPU is slow.
    pub(crate) fn drop_images(&mut self) {
        self.image = None;
        self.page_images.clear();
    }

    #[cfg(test)]
    #[must_use]
    pub fn with_context(mut self, context: impl IntoIterator<Item = TranslationContext>) -> Self {
        self.context = context.into_iter().collect();
        self
    }

    pub(crate) fn prepare_image(&mut self) -> anyhow::Result<()> {
        if let Some(image) = self.image.as_ref() {
            self.image = Some(shrink(image)?);
        }
        for image in &mut self.page_images {
            *image = shrink(image)?;
        }
        Ok(())
    }

    pub(crate) fn remove_image(&mut self) {
        self.drop_images();
        self.segment_pages.clear();
    }
}

/// The image scaled down so that its longest side is at most `MAX_IMAGE_DIMENSION`.
fn shrink(image: &Arc<DynamicImage>) -> anyhow::Result<Arc<DynamicImage>> {
    shrink_to(image, MAX_IMAGE_DIMENSION)
}

/// The image scaled down so that its longest side is at most `maximum` pixels.
pub(crate) fn shrink_to(
    image: &Arc<DynamicImage>,
    maximum: u32,
) -> anyhow::Result<Arc<DynamicImage>> {
    let (width, height) = (image.width(), image.height());
    let longest = width.max(height);
    if longest <= maximum {
        return Ok(Arc::clone(image));
    }
    let resized_width = (u64::from(width) * u64::from(maximum) / u64::from(longest)).max(1) as u32;
    let resized_height =
        (u64::from(height) * u64::from(maximum) / u64::from(longest)).max(1) as u32;
    let source = image.to_rgb8();
    let mut resized = RgbImage::new(resized_width, resized_height);
    Resizer::new()
        .resize(
            &source,
            &mut resized,
            &ResizeOptions::new()
                .resize_alg(ResizeAlg::Convolution(FilterType::Lanczos3))
                .use_alpha(false),
        )
        .context("failed to resize translation image")?;
    Ok(Arc::new(DynamicImage::ImageRgb8(resized)))
}

pub(crate) struct EncodedImage {
    pub(crate) data: String,
}

impl EncodedImage {
    pub(crate) fn data_url(&self) -> String {
        format!("data:image/jpeg;base64,{}", self.data)
    }
}

pub(crate) fn encode_image(image: &DynamicImage) -> anyhow::Result<EncodedImage> {
    let rgb = image.to_rgb8();
    let mut bytes = Cursor::new(Vec::new());
    JpegEncoder::new_with_quality(&mut bytes, JPEG_QUALITY)
        .write_image(
            rgb.as_raw(),
            rgb.width(),
            rgb.height(),
            image::ExtendedColorType::Rgb8,
        )
        .context("failed to encode translation image")?;
    Ok(EncodedImage {
        data: STANDARD.encode(bytes.into_inner()),
    })
}

/// One entry of the glossary built while translating: a name, a recurring term, a
/// character profile, or how one character addresses another. It is replayed to every model that translates later pages.
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct GlossaryEntry {
    /// `style`, `address`, `character`, `name` or `term`.
    pub kind: String,
    pub source: String,
    pub target: String,
}

#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct TranslationContext {
    pub source: String,
    pub translation: String,
}

impl TranslationContext {
    #[cfg(test)]
    #[must_use]
    pub fn new(source: impl Into<String>, translation: impl Into<String>) -> Self {
        Self {
            source: source.into(),
            translation: translation.into(),
        }
    }
            }
