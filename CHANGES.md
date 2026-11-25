# Changes from Koharu 0.94.0

A combination of 0.94.0 and a 0.83.3 Docker build: 0.94.0's models, pipeline stages, scene and
renderer, with a headless server (`crates/koharu-server`) in place of the desktop shell, PP-OCRv6
Manga brought back, and an image-reading analysis of the whole work before translation.
See `README-SERVER.md` for use. **Nothing here has been compiled** (see the end of that file).

## Added
- `crates/koharu-server`: axum server, a one-file web page, a library of kept jobs (`job.json` +
  `pages.khr` each), upload of images, ZIP/CBZ or a `.khr`, run again, rename, delete, edit
  translations, download CBZ / PSD ZIP / `.khr`.
- `crates/koharu-pipeline/src/stages/analyze/`: the analysis. Windows of 12 pages (2 repeated), pages
  with clean-ups laid over and texts' boxes numbered, WebP quality 90, sent to an OpenAI-compatible
  vision model; characters, per-page summaries, per-line speaker/tone/OCR corrections.
- `crates/koharu-ml/src/pp_ocr_v6_manga/`: ported from the 0.83.3 build (ONNX Runtime, `ort`
  `load-dynamic`), with `polygon_score` copied in; `Recognizer::PpOcrV6Manga`.
- `Dockerfile` (no CEF/GTK; ONNX Runtime kept), `.dockerignore`, `.github/workflows/docker.yml`.

- A browser editor (`editor.html`, `editor.rs`): select/move/resize/add text boxes, an inspector for
  text and style, per-block and per-page re-runs, an inpaint brush, undo/redo, a page rail; the
  analyzer's batch size, overlap, image size and quality as form settings.

## Changed
- `scheduler/run.rs`, `scheduler/mod.rs`, `stages/mod.rs`: with an analyzer set, a batch run's first
  translation waits for every page's recognition and inpainting, makes the analysis once, and the
  rest reuse it.
- `stages/translate/*`: a local translator asks for the most context its training allows; the system
  prompt holds the characters and every page's summary; each turn holds that page's notes, which the
  conversation does not keep; corrected OCR readings are used for the request only; the conversation
  starts again when a new analysis arrives.
- `config.rs`: `[pipeline.analyzer]` (`model`, `pages`, `overlap`, `size`, `quality`, `reasoning`).
- `koharu-config/secrets.rs`: `KOHARU_SECRET_<NAME>` in the environment is read before the OS
  credential store, for containers.
- `koharu-runtime/huggingface.rs`: `Repository::tag`, for a checkpoint published by tag.
- `koharu-scene`: `Session::export`, a consistent `.khr` copy through `VACUUM INTO`.
- `Cargo.toml`/`Cargo.lock`: `ort`, `ndarray`, `smallvec 2.0.0-alpha.10` (via `ort`), `multer`,
  `spin` (axum multipart), and the new crate.

## Left out on purpose
The 0.83.3 build's glossary, pronoun pairs, polish pass, script/loop guards, Vietnamese-specific
rules, its other OCR and detection models, DeepL/Caiyun/Google Cloud, and the web editor.
