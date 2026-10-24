//! PP-OCRv6 manga: a manga fine-tune of the PP-OCRv6 tiny detector and small
//! recognizer, run on ONNX Runtime.
//!
//! Upstream: <https://huggingface.co/Kellenok/PP-OCRv6_manga>. The checkpoint is
//! pinned below; its README describes the preprocessing and the DB thresholds.
//!
//! The recognizer reads one text line, so a text block is processed in three
//! steps: the detector finds the lines inside the crop, each line is cut out
//! upright (vertical lines are rotated), and the recognized lines are put back in
//! reading order.
//!
//! Intentional divergences from the upstream example, kept here so they can be
//! revisited:
//! - Koharu supplies one block crop at a time, so the detector input is sized
//!   from the *page* (see [`page_scale`]) rather than from the crop.
//! - The upstream example's learned furigana filter is not ported. Furigana is
//!   rare in the Chinese and Korean text this engine is mostly used for.
//! - When the detector finds no line in a crop, the whole crop is read as one
//!   line instead of returning nothing, because Koharu's layout detector has
//!   already decided that the crop contains text.

mod geometry;
mod session;

use anyhow::{Context, Result, anyhow, ensure};
use image::{DynamicImage, RgbImage};

use self::{
    geometry::{Line, Rotation},
    session::OnnxSession,
};
use crate::pp_ocr_v6::det::boxes_from_bitmap;

pub use self::geometry::page_scale;

crate::model_repository!("Kellenok/PP-OCRv6_manga" @ "72e97f5f9938b06bcf1dda07b4d49cd0632091ca" {
    DETECTOR = "det/manga_det_v0.2.onnx",
    RECOGNIZER = "rec/manga_rec_v0.2.onnx",
    DICTIONARY = "ppocrv6_dict.txt",
});

// PaddleOCR normalizes OpenCV's BGR arrays with the ImageNet statistics listed
// in RGB order, so these are indexed by BGR slot.
const MEAN: [f32; 3] = [0.485, 0.456, 0.406];
const STD: [f32; 3] = [0.229, 0.224, 0.225];

// DB post-processing values recommended by the model card. The shared
// implementation is the one used by the PP-OCRv6 medium detector.
const DB_THRESHOLD: f32 = 0.15;
const DB_BOX_THRESHOLD: f32 = 0.25;
const DB_UNCLIP_RATIO: f32 = 1.4;
const DB_MIN_SIZE: i32 = 3;
const DB_MAX_CANDIDATES: usize = 1000;

/// Optional overrides for the DB post-processing values above, so the detector
/// can be tuned on a deployed Space without a rebuild.
const DB_THRESHOLD_ENV: &str = "KOHARU_PPOCR_DB_THRESHOLD";
const DB_BOX_THRESHOLD_ENV: &str = "KOHARU_PPOCR_DB_BOX_THRESHOLD";
const DB_UNCLIP_RATIO_ENV: &str = "KOHARU_PPOCR_DB_UNCLIP_RATIO";

/// Second attempt when the detector finds no line at the normal thresholds:
/// multiplied into the pixel and box thresholds before falling back to reading
/// the whole crop as a single line.
const RELAXED_FACTOR: f32 = 0.5;

#[derive(Debug, Clone, Copy)]
struct DbParams {
    threshold: f32,
    box_threshold: f32,
    unclip_ratio: f32,
}

impl DbParams {
    fn from_env() -> Self {
        let read = |name: &str, default: f32| {
            std::env::var(name)
                .ok()
                .and_then(|value| value.trim().parse::<f32>().ok())
                .filter(|value| value.is_finite() && *value > 0.0)
                .unwrap_or(default)
        };
        Self {
            threshold: read(DB_THRESHOLD_ENV, DB_THRESHOLD),
            box_threshold: read(DB_BOX_THRESHOLD_ENV, DB_BOX_THRESHOLD),
            unclip_ratio: read(DB_UNCLIP_RATIO_ENV, DB_UNCLIP_RATIO),
        }
    }

    fn relaxed(self) -> Self {
        Self {
            threshold: self.threshold * RELAXED_FACTOR,
            box_threshold: self.box_threshold * RELAXED_FACTOR,
            ..self
        }
    }
}

/// Selects how vertical lines are turned upright: `cw` or the default `ccw`
/// (`np.rot90`, as PaddleOCR does). Only a debugging aid for checking which
/// direction a checkpoint was trained with.
const VERTICAL_ROTATION_ENV: &str = "KOHARU_PPOCR_VERTICAL_ROTATION";

#[derive(Debug)]
pub struct PpOcrV6Manga {
    detector: OnnxSession,
    recognizer: OnnxSession,
    dictionary: Vec<String>,
    rotation: Rotation,
    db: DbParams,
}

impl PpOcrV6Manga {
    /// Runs on the CPU through ONNX Runtime; `device` is accepted to keep the
    /// lifecycle of every model the same.
    pub async fn load(_device: crate::Device) -> Result<Self> {
        let (detector_path, recognizer_path, dictionary_path) = tokio::try_join!(
            async {
                DETECTOR
                    .resolve()
                    .await
                    .context("failed to resolve the PP-OCRv6 manga detector")
            },
            async {
                RECOGNIZER
                    .resolve()
                    .await
                    .context("failed to resolve the PP-OCRv6 manga recognizer")
            },
            async {
                DICTIONARY
                    .resolve()
                    .await
                    .context("failed to resolve the PP-OCRv6 manga dictionary")
            },
        )?;

        let dictionary = geometry::parse_dictionary(
            &std::fs::read_to_string(&dictionary_path)
                .with_context(|| format!("failed to read {}", dictionary_path.display()))?,
        );
        let detector = OnnxSession::load(&detector_path)
            .with_context(|| format!("failed to load {}", detector_path.display()))?;
        let recognizer = OnnxSession::load(&recognizer_path)
            .with_context(|| format!("failed to load {}", recognizer_path.display()))?;
        let rotation = match std::env::var(VERTICAL_ROTATION_ENV).as_deref() {
            Ok("cw") => Rotation::Cw,
            _ => Rotation::Ccw,
        };

        Ok(Self {
            detector,
            recognizer,
            dictionary,
            rotation,
            db: DbParams::from_env(),
        })
    }

    /// Reads a text block without knowing its page: the crop is viewed at its own
    /// resolution, shrunk if it exceeds the detector's long side.
    pub fn inference(&self, image: &DynamicImage) -> Result<String> {
        let scale = page_scale(image.width(), image.height()).min(1.0);
        self.inference_with_scale(image, scale)
    }

    /// Reads a text block cropped from a page whose detector scale is `scale`
    /// (see [`page_scale`]).
    pub fn inference_with_scale(&self, image: &DynamicImage, scale: f64) -> Result<String> {
        let rgb = image.to_rgb8();
        let (width, height) = rgb.dimensions();
        if width < 2 || height < 2 {
            return Ok(String::new());
        }

        let mut quads = self.detect(&rgb, scale, self.db)?;
        let mut detected = quads.len();
        if quads.is_empty() {
            quads = self.detect(&rgb, scale, self.db.relaxed())?;
            detected = quads.len();
        }
        let whole_crop_fallback = quads.is_empty();
        if whole_crop_fallback {
            let (right, bottom) = (width as f32, height as f32);
            quads.push([[0.0, 0.0], [right, 0.0], [right, bottom], [0.0, bottom]]);
        }
        tracing::debug!(
            crop_width = width,
            crop_height = height,
            scale,
            lines = detected,
            whole_crop_fallback,
            "pp-ocrv6-manga detection"
        );

        let mut lines = Vec::with_capacity(quads.len());
        for quad in &quads {
            let (crop, crop_width, crop_height) =
                geometry::crop_quad(rgb.as_raw(), width as usize, height as usize, quad);
            let vertical = geometry::is_vertical(crop_width, crop_height);
            let (upright, upright_width, upright_height) = if vertical {
                geometry::rotate90(&crop, crop_width, crop_height, self.rotation)
            } else {
                (crop, crop_width, crop_height)
            };
            let text = self.recognize(&upright, upright_width, upright_height)?;
            tracing::debug!(
                line_width = upright_width,
                line_height = upright_height,
                vertical,
                text = text.as_str(),
                "pp-ocrv6-manga line"
            );
            if text.trim().is_empty() {
                continue;
            }
            lines.push(Line {
                center: [
                    quad.iter().map(|point| point[0]).sum::<f32>() / 4.0,
                    quad.iter().map(|point| point[1]).sum::<f32>() / 4.0,
                ],
                thickness: crop_width.min(crop_height) as f32,
                vertical,
                text,
            });
        }
        Ok(geometry::join_lines(&geometry::order_lines(lines)))
    }

    /// Finds text lines as quadrilaterals (top-left, top-right, bottom-right,
    /// bottom-left) in the coordinates of `rgb`.
    fn detect(&self, rgb: &RgbImage, scale: f64, db: DbParams) -> Result<Vec<[[f32; 2]; 4]>> {
        let (width, height) = rgb.dimensions();
        let (input_width, input_height) = geometry::detector_size(width, height, scale);
        let input = geometry::resize_normalize(
            rgb.as_raw(),
            width as usize,
            height as usize,
            input_width as usize,
            input_height as usize,
            MEAN,
            STD,
        );
        let (shape, probabilities) = self
            .detector
            .run_f32([1, 3, i64::from(input_height), i64::from(input_width)], input)?;
        ensure!(
            probabilities.len() == input_width as usize * input_height as usize,
            "PP-OCRv6 manga detector returned {shape:?} for a {input_width}x{input_height} input"
        );
        Ok(boxes_from_bitmap(
            &probabilities,
            input_width,
            input_height,
            width,
            height,
            db.threshold,
            db.box_threshold,
            db.unclip_ratio,
            DB_MIN_SIZE,
            DB_MAX_CANDIDATES,
        )
        .into_iter()
        .map(|detection| detection.polygon)
        .collect())
    }

    fn recognize(&self, rgb: &[u8], width: usize, height: usize) -> Result<String> {
        let (input, padded_width) = geometry::rec_input(rgb, width, height, MEAN, STD);
        let (shape, scores) = self.recognizer.run_f32(
            [1, 3, geometry::REC_HEIGHT as i64, padded_width as i64],
            input,
        )?;
        ensure!(
            shape.len() == 3 && shape[0] == 1,
            "PP-OCRv6 manga recognizer returned {shape:?}, expected [1, steps, classes]"
        );
        geometry::ctc_decode(&scores, shape[1] as usize, shape[2] as usize, &self.dictionary)
            .map_err(|message| anyhow!(message))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    #[ignore = "downloads the checkpoint and needs libonnxruntime"]
    async fn reads_a_blank_crop_without_failing() {
        let model = PpOcrV6Manga::load(crate::Device::cpu()).await.unwrap();
        model.inference(&DynamicImage::new_rgb8(200, 80)).unwrap();
    }
}
