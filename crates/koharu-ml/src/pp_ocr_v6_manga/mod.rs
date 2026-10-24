//! PP-OCRv6 manga: a manga fine-tune of the PP-OCRv6 tiny detector and small
//! recognizer, run on ONNX Runtime.
//!
//! Upstream: <https://huggingface.co/Kellenok/PP-OCRv6_manga>. The checkpoint is
//! pinned below. The pipeline is a port of the author's reference demo
//! (<https://huggingface.co/spaces/Kellenok/PP-OCR_manga>, `app.py`): the detector
//! finds the text lines inside the crop, each line is cut out upright, the
//! recognizer reads it, and the lines are put back in reading order.
//!
//! The recognizer is sensitive to its input: it expects BGR pixels scaled to
//! `[-1, 1]` at a height of 48 with no padding, and the detector expects the
//! ImageNet statistics. Both differ from what the PP-OCRv6 medium port in
//! `pp_ocr_v6` uses, so nothing is shared with it except the DB contour scoring.
//!
//! Intentional divergences from the demo, kept here so they can be revisited:
//! - The `pyclipper` offset of a contour is replaced by the equivalent computation
//!   on its convex hull (see [`detect`]).
//! - The demo's heuristics that trim furigana and stray margins off narrow vertical
//!   lines (`clean_manga_vertical_crop`) are not ported, and no furigana line is
//!   dropped (the demo does not drop them by default either).
//! - The FP32 ONNX files are used instead of the FP16 ones.
//! - Lines are joined into one text; the demo lists them separately.
//! - Koharu hands over one text-block crop at a time, so the detector uses the
//!   demo's "Single bubble" preset (`0.20`, `0.45`, `1.6`). Set
//!   `KOHARU_PPOCR_DB=threshold,box_threshold,unclip_ratio` to try other values.
//!
//! Set `KOHARU_PPOCR_DEBUG` to a directory to save the detector's map and the
//! detected boxes for every call.

mod debug;
mod detect;
mod geometry;
mod layout;
mod session;

use std::path::PathBuf;

use anyhow::{Context, Result, anyhow, ensure};
use image::{DynamicImage, RgbImage};

use self::{
    debug::DetectorMap,
    detect::{DbParams, MapLayout},
    layout::LineBox,
    session::OnnxSession,
};

crate::model_repository!("Kellenok/PP-OCRv6_manga" @ "72e97f5f9938b06bcf1dda07b4d49cd0632091ca" {
    DETECTOR = "det/manga_det_v0.2.onnx",
    RECOGNIZER = "rec/manga_rec_v0.2.onnx",
    DICTIONARY = "ppocrv6_dict.txt",
});

// The detector normalizes BGR pixels with the ImageNet statistics (blue first).
const DETECTOR_MEAN: [f32; 3] = [0.485, 0.456, 0.406];
const DETECTOR_STD: [f32; 3] = [0.229, 0.224, 0.225];

/// The demo's "Single bubble" preset.
const DB_PARAMS: DbParams = DbParams {
    threshold: 0.20,
    box_threshold: 0.45,
    unclip_ratio: 1.6,
};
const DB_ENV: &str = "KOHARU_PPOCR_DB";
const DEBUG_ENV: &str = "KOHARU_PPOCR_DEBUG";

#[derive(Debug)]
pub struct PpOcrV6Manga {
    detector: OnnxSession,
    recognizer: OnnxSession,
    dictionary: Vec<String>,
    params: DbParams,
    debug_directory: Option<PathBuf>,
}

fn db_params_from_env() -> DbParams {
    let Ok(value) = std::env::var(DB_ENV) else {
        return DB_PARAMS;
    };
    let parsed = value
        .split(',')
        .map(|part| part.trim().parse::<f32>())
        .collect::<Result<Vec<_>, _>>();
    match parsed {
        Ok(values) if values.len() == 3 => DbParams {
            threshold: values[0],
            box_threshold: values[1],
            unclip_ratio: values[2],
        },
        _ => {
            tracing::warn!(
                "ignoring {}={:?}: expected `threshold,box_threshold,unclip_ratio`",
                DB_ENV,
                value
            );
            DB_PARAMS
        }
    }
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

        let debug_directory = std::env::var_os(DEBUG_ENV).map(PathBuf::from);
        if let Some(directory) = &debug_directory {
            std::fs::create_dir_all(directory)
                .with_context(|| format!("failed to create {}", directory.display()))?;
        }

        Ok(Self {
            detector,
            recognizer,
            dictionary,
            params: db_params_from_env(),
            debug_directory,
        })
    }

    /// Reads one text-block crop and returns its lines in reading order.
    pub fn inference(&self, image: &DynamicImage) -> Result<String> {
        let rgb = image.to_rgb8();
        let (width, height) = (rgb.width() as usize, rgb.height() as usize);
        if width < 2 || height < 2 {
            return Ok(String::new());
        }

        let (boxes, map) = self.detect(&rgb)?;
        let vertical = layout::page_is_vertical(&boxes, width as f32, height as f32);
        let ordered = layout::reading_order(boxes, vertical);

        let mut texts = Vec::with_capacity(ordered.len());
        for line in &ordered {
            texts.push(self.read_line(&rgb, line)?);
        }
        if let Some(directory) = &self.debug_directory {
            debug::dump(directory, &rgb, &map, &ordered, &texts);
        }
        Ok(layout::join_lines(&texts))
    }

    /// Finds the text lines of `rgb` and returns them with the detector's map.
    fn detect(&self, rgb: &RgbImage) -> Result<(Vec<LineBox>, DetectorMap)> {
        let (width, height) = (rgb.width() as usize, rgb.height() as usize);
        let (padded, padded_width, padded_height) =
            geometry::pad_white(rgb.as_raw(), width, height, geometry::DET_MARGIN);
        let (input_width, input_height) = geometry::detector_size(padded_width, padded_height);
        let input = geometry::resize_normalize(
            &padded,
            padded_width,
            padded_height,
            input_width,
            input_height,
            DETECTOR_MEAN,
            DETECTOR_STD,
        );
        let (shape, values) = self
            .detector
            .run_f32([1, 3, input_height as i64, input_width as i64], input)?;
        ensure!(
            values.len() == input_width * input_height,
            "PP-OCRv6 manga detector returned {shape:?} for a {input_width}x{input_height} input"
        );

        let layout = MapLayout {
            map_width: input_width as u32,
            map_height: input_height as u32,
            padded_width,
            padded_height,
            margin: geometry::DET_MARGIN,
            width,
            height,
        };
        let boxes = detect::line_boxes(&values, &layout, self.params);
        let map = DetectorMap {
            values,
            width: input_width as u32,
            height: input_height as u32,
        };
        Ok((boxes, map))
    }

    /// Cuts one line out of the crop, turns it upright and reads it. Returns an
    /// empty string for a line that is too small to read.
    fn read_line(&self, rgb: &RgbImage, line: &LineBox) -> Result<String> {
        let corners = geometry::order_corners(&line.corners);
        let Some((crop, crop_width, crop_height)) = geometry::crop_perspective(
            rgb.as_raw(),
            rgb.width() as usize,
            rgb.height() as usize,
            &corners,
        ) else {
            return Ok(String::new());
        };
        // Vertical lines are read as horizontal text turned counter-clockwise.
        let (pixels, width, height) = if crop_height > crop_width {
            geometry::rotate_ccw(&crop, crop_width, crop_height)
        } else {
            (crop, crop_width, crop_height)
        };
        let Some((input, input_width)) = geometry::rec_input(&pixels, width, height) else {
            return Ok(String::new());
        };

        let (shape, scores) = self
            .recognizer
            .run_f32([1, 3, geometry::REC_HEIGHT as i64, input_width as i64], input)?;
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
