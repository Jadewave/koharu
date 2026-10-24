//! The only file that touches ONNX Runtime.
//!
//! Divergence from the other models: PP-OCRv6 manga ships only Paddle and ONNX
//! weights, so it runs on ONNX Runtime instead of a LibTorch port. The runtime
//! library is loaded dynamically (`load-dynamic`) to avoid linking a second C++
//! runtime into the binary; place `libonnxruntime.so` next to the executable or
//! point `ORT_DYLIB_PATH` at it.

use std::path::Path;

use anyhow::Result;

/// A CPU ONNX Runtime session that maps one `f32` tensor to its first output.
pub(super) struct OnnxSession {
    #[cfg(feature = "ppocr-manga")]
    inner: std::sync::Mutex<ort::session::Session>,
}

impl std::fmt::Debug for OnnxSession {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str("OnnxSession")
    }
}

#[cfg(feature = "ppocr-manga")]
impl OnnxSession {
    pub(super) fn load(path: &Path) -> Result<Self> {
        use anyhow::anyhow;

        let path = path.to_owned();
        // With `load-dynamic`, a missing library panics inside `ort` rather than
        // returning an error; surface it as an ordinary load failure.
        let session = std::panic::catch_unwind(move || {
            ort::session::Session::builder().and_then(|builder| builder.commit_from_file(&path))
        })
        .map_err(|_| {
            anyhow!(
                "failed to load the ONNX Runtime library (libonnxruntime.so); \
                 put it next to the executable or set ORT_DYLIB_PATH"
            )
        })?
        .map_err(|error| anyhow!("failed to create the ONNX Runtime session: {error}"))?;
        Ok(Self {
            inner: std::sync::Mutex::new(session),
        })
    }

    /// Runs the model on one NCHW `f32` input and returns the first output's
    /// shape and values. Inputs are bound by position, so graph input names do
    /// not matter.
    pub(super) fn run_f32(&self, shape: [i64; 4], data: Vec<f32>) -> Result<(Vec<i64>, Vec<f32>)> {
        use anyhow::anyhow;

        let tensor = ort::value::Tensor::from_array((shape, data))
            .map_err(|error| anyhow!("failed to build the input tensor: {error}"))?;
        let mut session = self
            .inner
            .lock()
            .map_err(|_| anyhow!("ONNX Runtime session lock is poisoned"))?;
        let outputs = session
            .run(ort::inputs![tensor])
            .map_err(|error| anyhow!("ONNX Runtime inference failed: {error}"))?;
        let (output_shape, values) = outputs[0]
            .try_extract_tensor::<f32>()
            .map_err(|error| anyhow!("failed to read the output tensor: {error}"))?;
        Ok((output_shape.iter().copied().collect(), values.to_vec()))
    }
}

#[cfg(not(feature = "ppocr-manga"))]
impl OnnxSession {
    pub(super) fn load(_path: &Path) -> Result<Self> {
        anyhow::bail!("koharu-ml was built without the `ppocr-manga` feature")
    }

    pub(super) fn run_f32(&self, _shape: [i64; 4], _data: Vec<f32>) -> Result<(Vec<i64>, Vec<f32>)> {
        anyhow::bail!("koharu-ml was built without the `ppocr-manga` feature")
    }
}
