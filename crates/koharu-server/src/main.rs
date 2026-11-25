//! Koharu's pipeline behind HTTP, for a container with no screen: a page of
//! its own to upload a chapter to and read the result from, and a small API
//! under `/api`. There is no editor canvas here, only each page as the app
//! would draw it and the translated texts to correct; the desktop app is
//! the editor.
//!
//! What a run uses is Koharu's own `[pipeline]` section, then these
//! environment variables over it, and the form of an upload over those:
//!
//! - `KOHARU_TRANSLATOR_MODEL`, `KOHARU_TARGET_LANGUAGE`, `KOHARU_INSTRUCTIONS`
//! - `KOHARU_RECOGNIZER` (such as `pp-ocr-v6-manga`), `KOHARU_INPAINTER`
//! - `KOHARU_ANALYZER_MODEL` (`vendor/model`, empty for none),
//!   `KOHARU_ANALYZER_PAGES`, `KOHARU_ANALYZER_OVERLAP`,
//!   `KOHARU_ANALYZER_SIZE`, `KOHARU_ANALYZER_QUALITY`
//! - `KOHARU_OPENAI_COMPATIBLE_BASE_URL` and
//!   `KOHARU_OPENAI_COMPATIBLE_CONTEXT_WINDOW`, written to `[providers]`
//!
//! A vendor's API key is the secret `KOHARU_SECRET_<VENDOR>`, such as
//! `KOHARU_SECRET_OPENAI_COMPATIBLE`, as a container has no credential store.

mod editor;
mod jobs;

use std::{
    collections::{BTreeSet, HashMap},
    num::NonZeroU32,
    path::PathBuf,
    sync::Arc,
};

use anyhow::{Context as _, Result};
use axum::{
    Json, Router,
    extract::{DefaultBodyLimit, Multipart, Path, Query, State},
    http::{HeaderValue, StatusCode, header},
    response::{Html, IntoResponse, Response},
    routing::{get, post, put},
};
use clap::Parser;
use koharu_llm::Providers;
use koharu_pipeline::{Pipeline, PipelineConfig, Stage};
use koharu_renderer::{Renderer, RendererConfig};
use koharu_scene::LayerId;
use serde::Deserialize;
use serde_json::{Value, json};
use uuid::Uuid;

use self::jobs::{Edited, Registry, Upload};

#[derive(Debug, Parser)]
#[command(version, about)]
struct Cli {
    /// The address to listen on; `KOHARU_HOST`, else 127.0.0.1.
    #[arg(long)]
    host: Option<String>,

    /// The port to listen on; `KOHARU_PORT`, else 7860.
    #[arg(long)]
    port: Option<u16>,

    /// Where the jobs' projects go; `KOHARU_DATA`, else `~/.koharu/server`.
    #[arg(long, value_name = "DIR")]
    data: Option<PathBuf>,
}

struct App {
    pipeline: Pipeline,
    renderer: Renderer,
    registry: Arc<Registry>,
    config: PipelineConfig,
}

type Shared = Arc<App>;

#[tokio::main]
async fn main() -> Result<()> {
    let filter = tracing_subscriber::EnvFilter::try_from_default_env()
        .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("info"));
    tracing_subscriber::fmt().with_env_filter(filter).init();

    let cli = Cli::parse();
    let host = cli
        .host
        .or_else(|| environment("KOHARU_HOST"))
        .unwrap_or_else(|| "127.0.0.1".to_owned());
    let port = match (cli.port, environment("KOHARU_PORT")) {
        (Some(port), _) => port,
        (None, Some(port)) => port.parse().context("KOHARU_PORT isn't a port")?,
        (None, None) => 7860,
    };
    let data = cli
        .data
        .or_else(|| environment("KOHARU_DATA").map(PathBuf::from))
        .or_else(|| {
            std::env::var_os("HOME").map(|home| PathBuf::from(home).join(".koharu/server"))
        })
        .context("no data folder: pass --data or set HOME")?;
    fs_err::create_dir_all(&data)?;

    let config = configure()?;
    tracing::info!(
        translator = %config.translator.model,
        analyzer = %config.analyzer.model,
        recognizer = %config.recognizer.model,
        "pipeline settings"
    );
    let pipeline = Pipeline::new(config.clone());
    let registry = Arc::new(Registry::new(data.join("jobs"))?);
    registry.follow(pipeline.events());
    let renderer = Renderer::new(koharu_config::get::<RendererConfig>()?);
    let app = Arc::new(App {
        pipeline,
        renderer,
        registry,
        config,
    });

    let router = Router::new()
        .route("/", get(|| async { Html(include_str!("ui.html")) }))
        .route("/api/health", get(|| async { "ok" }))
        .route("/api/config", get(settings))
        .route("/api/jobs", get(list).post(create))
        .route("/api/jobs/{id}", get(status).patch(rename).delete(remove))
        .route("/api/jobs/{id}/run", post(run))
        .route("/api/jobs/{id}/blocks", get(blocks))
        .route("/api/jobs/{id}/blocks/{layer}", put(edit))
        .route("/api/jobs/{id}/pages/{number}/image", get(image))
        .route("/api/jobs/{id}/download", get(download))
        .route("/api/stop", post(stop))
        .route("/editor", get(|| async { Html(include_str!("editor.html")) }))
        .merge(editor::routes())
        .layer(DefaultBodyLimit::max(2 << 30))
        .with_state(app);
    let listener = tokio::net::TcpListener::bind((host.as_str(), port))
        .await
        .with_context(|| format!("couldn't listen on {host}:{port}"))?;
    tracing::info!("listening on http://{host}:{port}");
    axum::serve(listener, router)
        .with_graceful_shutdown(async {
            let _ = tokio::signal::ctrl_c().await;
        })
        .await?;
    Ok(())
}

fn environment(name: &str) -> Option<String> {
    std::env::var(name)
        .ok()
        .map(|value| value.trim().to_owned())
        .filter(|value| !value.is_empty())
}

/// `[pipeline]` with the environment's settings over it, and the OpenAI-
/// compatible server's address in `[providers]`.
fn configure() -> Result<PipelineConfig> {
    let mut config: PipelineConfig = koharu_config::get()?;
    if let Some(model) = environment("KOHARU_TRANSLATOR_MODEL") {
        config.translator.model = model;
    }
    if let Some(target) = environment("KOHARU_TARGET_LANGUAGE") {
        set_target(&mut config, &target)?;
    }
    if let Some(instructions) = environment("KOHARU_INSTRUCTIONS") {
        config.translator.instructions = instructions;
    }
    if let Some(id) = environment("KOHARU_RECOGNIZER") {
        config.recognizer.model = by_id(&id).context("KOHARU_RECOGNIZER")?;
    }
    if let Some(id) = environment("KOHARU_INPAINTER") {
        config.inpainter.model = by_id(&id).context("KOHARU_INPAINTER")?;
    }
    // Set but empty turns the analysis off, so it can't go through
    // `environment`, which reads an empty value as unset.
    if let Ok(model) = std::env::var("KOHARU_ANALYZER_MODEL") {
        config.analyzer.model = model.trim().to_owned();
    }
    for (name, setting) in [
        ("KOHARU_ANALYZER_PAGES", &mut config.analyzer.pages),
        ("KOHARU_ANALYZER_OVERLAP", &mut config.analyzer.overlap),
        ("KOHARU_ANALYZER_SIZE", &mut config.analyzer.size),
    ] {
        if let Some(value) = environment(name) {
            *setting = value.parse().with_context(|| format!("{name} isn't a number"))?;
        }
    }
    if let Some(quality) = environment("KOHARU_ANALYZER_QUALITY") {
        config.analyzer.quality = quality
            .parse()
            .context("KOHARU_ANALYZER_QUALITY isn't a number from 1 to 100")?;
    }

    if let Some(url) = environment("KOHARU_OPENAI_COMPATIBLE_BASE_URL") {
        let url: url::Url = url
            .parse()
            .context("KOHARU_OPENAI_COMPATIBLE_BASE_URL isn't a URL")?;
        let window: Option<NonZeroU32> = environment("KOHARU_OPENAI_COMPATIBLE_CONTEXT_WINDOW")
            .map(|window| window.parse())
            .transpose()
            .context("KOHARU_OPENAI_COMPATIBLE_CONTEXT_WINDOW isn't a number of tokens")?;
        koharu_config::update::<Providers>(|providers| {
            let endpoint = providers
                .0
                .entry("openai-compatible".to_owned())
                .or_default();
            endpoint.base_url = Some(url);
            if window.is_some() {
                endpoint.context_window = window;
            }
        })?;
    }
    Ok(config)
}

/// A value by its id, as the `[pipeline]` section writes it.
fn by_id<T: serde::de::DeserializeOwned>(id: &str) -> Result<T> {
    serde_json::from_value(Value::String(id.to_owned()))
        .with_context(|| format!("`{id}` isn't a known id"))
}

/// Sets the target through the section's own reading of it, which refuses a
/// language CLDR has no English name for.
fn set_target(config: &mut PipelineConfig, target: &str) -> Result<()> {
    let mut translator = serde_json::to_value(&config.translator)?;
    translator["target"] = Value::String(target.to_owned());
    config.translator = serde_json::from_value(translator)
        .with_context(|| format!("`{target}` isn't a language CLDR names in English"))?;
    Ok(())
}

/// What the page's form may change, over `config`.
fn overridden(config: &PipelineConfig, form: &HashMap<String, String>) -> Result<PipelineConfig> {
    let mut config = config.clone();
    let field = |name: &str| {
        form.get(name)
            .map(|value| value.trim())
            .filter(|value| !value.is_empty())
    };
    if let Some(model) = field("translator") {
        config.translator.model = model.to_owned();
    }
    if let Some(target) = field("target") {
        set_target(&mut config, target)?;
    }
    if let Some(instructions) = field("instructions") {
        config.translator.instructions = instructions.to_owned();
    }
    if let Some(model) = field("analyzer") {
        config.analyzer.model = if model == "off" {
            String::new()
        } else {
            model.to_owned()
        };
    }
    if let Some(id) = field("recognizer") {
        config.recognizer.model = by_id(id)?;
    }
    for (name, setting) in [
        ("analyzer_pages", &mut config.analyzer.pages),
        ("analyzer_overlap", &mut config.analyzer.overlap),
        ("analyzer_size", &mut config.analyzer.size),
    ] {
        if let Some(value) = field(name) {
            *setting = value
                .parse()
                .with_context(|| format!("{name} isn't a number"))?;
        }
    }
    if let Some(quality) = field("analyzer_quality") {
        config.analyzer.quality = quality
            .parse()
            .context("analyzer_quality isn't a number from 1 to 100")?;
    }
    Ok(config)
}

struct ApiError(StatusCode, String);

impl ApiError {
    fn bad(error: impl std::fmt::Display) -> Self {
        Self(StatusCode::BAD_REQUEST, format!("{error:#}"))
    }

    fn missing(what: &str) -> Self {
        Self(StatusCode::NOT_FOUND, format!("no such {what}"))
    }
}

impl IntoResponse for ApiError {
    fn into_response(self) -> Response {
        (self.0, Json(json!({"error": self.1}))).into_response()
    }
}

impl<E: Into<anyhow::Error>> From<E> for ApiError {
    fn from(error: E) -> Self {
        let error: anyhow::Error = error.into();
        tracing::warn!("a request failed: {error:#}");
        Self(StatusCode::INTERNAL_SERVER_ERROR, format!("{error:#}"))
    }
}

type Reply<T> = Result<T, ApiError>;

/// The job, opened from the library if it isn't open yet.
async fn job(app: &Shared, id: Uuid) -> Reply<Arc<jobs::Job>> {
    let registry = Arc::clone(&app.registry);
    tokio::task::spawn_blocking(move || registry.get(id))
        .await??
        .ok_or_else(|| ApiError::missing("job"))
}

async fn settings(State(app): State<Shared>) -> Json<Value> {
    let config = &app.config;
    Json(json!({
        "translator": config.translator.model,
        "target": config.translator.target.to_string(),
        "analyzer": config.analyzer.model,
        "recognizer": config.recognizer.model.to_string(),
        "analyzer_pages": config.analyzer.pages,
        "analyzer_overlap": config.analyzer.overlap,
    }))
}

/// What the stages field names, or `default` when it names none.
fn stages_of(field: Option<&str>, default: &[Stage]) -> Result<BTreeSet<Stage>> {
    match field.map(str::trim) {
        None | Some("") => Ok(default.iter().copied().collect()),
        Some(stages) => stages
            .split(',')
            .map(|stage| by_id(stage.trim()))
            .collect::<Result<_>>(),
    }
}

const ALL: [Stage; 4] = [Stage::Detect, Stage::Recognize, Stage::Inpaint, Stage::Translate];

/// An upload: the images, or ZIPs and CBZs of them, as `files`; or one `.khr`
/// project, kept as it is. Optional fields name the job (`name`) and override
/// the pipeline's settings for it: `translator`, `target`, `instructions`,
/// `analyzer` (`off` for none), `recognizer`, and `stages` as a comma-separated
/// list. Images always run, all four stages unless `stages` says otherwise; a
/// project runs only the stages `stages` names, none by default.
async fn create(State(app): State<Shared>, mut form: Multipart) -> Reply<Json<Value>> {
    let mut uploads = Vec::new();
    let mut fields = HashMap::new();
    while let Some(field) = form.next_field().await.map_err(ApiError::bad)? {
        let name = field.name().unwrap_or_default().to_owned();
        if name == "files" {
            let file = field.file_name().unwrap_or("page").to_owned();
            let bytes = field.bytes().await.map_err(ApiError::bad)?.to_vec();
            uploads.push(Upload { name: file, bytes });
        } else {
            fields.insert(name, field.text().await.map_err(ApiError::bad)?);
        }
    }
    let config = overridden(&app.config, &fields).map_err(ApiError::bad)?;
    let stages = fields.get("stages").map(String::as_str);
    let is_project = |upload: &Upload| upload.name.to_lowercase().ends_with(".khr");
    let name = fields
        .get("name")
        .map(|name| name.trim().to_owned())
        .filter(|name| !name.is_empty())
        .or_else(|| {
            let first = uploads.first()?;
            let stem = std::path::Path::new(&first.name).file_stem()?.to_str()?;
            Some(stem.to_owned())
        })
        .unwrap_or_else(|| "Untitled".to_owned());
    let worker = Arc::clone(&app);
    let job = if uploads.iter().any(is_project) {
        if uploads.len() != 1 {
            return Err(ApiError::bad("send one .khr project, or images, not both"));
        }
        let run = match stages {
            None | Some("") => None,
            stages => Some(stages_of(stages, &ALL).map_err(ApiError::bad)?),
        };
        let upload = uploads.remove(0);
        tokio::task::spawn_blocking(move || {
            worker
                .registry
                .import(&worker.pipeline, &name, &upload.bytes, &config, run)
        })
        .await?
    } else {
        let stages = stages_of(stages, &ALL).map_err(ApiError::bad)?;
        tokio::task::spawn_blocking(move || {
            worker
                .registry
                .create(&worker.pipeline, &name, uploads, stages, config)
        })
        .await?
    }
    .map_err(ApiError::bad)?;
    Ok(Json(json!({"id": job.id, "pages": job.pages.len()})))
}

async fn list(State(app): State<Shared>) -> Json<Value> {
    Json(json!(app.registry.list()))
}

#[derive(Deserialize)]
struct Rename {
    name: String,
}

async fn rename(
    State(app): State<Shared>,
    Path(id): Path<Uuid>,
    Json(body): Json<Rename>,
) -> Reply<StatusCode> {
    if body.name.trim().is_empty() {
        return Err(ApiError::bad("a name can't be blank"));
    }
    let registry = Arc::clone(&app.registry);
    let found = tokio::task::spawn_blocking(move || registry.rename(id, &body.name)).await??;
    if found {
        Ok(StatusCode::NO_CONTENT)
    } else {
        Err(ApiError::missing("job"))
    }
}

/// Runs stages over a job in the library again: `stages` as a comma-separated
/// list, `translate` by default, and the same overrides an upload has.
async fn run(
    State(app): State<Shared>,
    Path(id): Path<Uuid>,
    Json(fields): Json<HashMap<String, String>>,
) -> Reply<StatusCode> {
    let job = job(&app, id).await?;
    let config = overridden(&app.config, &fields).map_err(ApiError::bad)?;
    let stages = stages_of(fields.get("stages").map(String::as_str), &[Stage::Translate])
        .map_err(ApiError::bad)?;
    tokio::task::spawn_blocking(move || app.registry.start(&job, &app.pipeline, stages, config))
        .await?
        .map_err(|error| ApiError(StatusCode::CONFLICT, format!("{error:#}")))?;
    Ok(StatusCode::ACCEPTED)
}

async fn status(State(app): State<Shared>, Path(id): Path<Uuid>) -> Reply<Json<Value>> {
    let job = job(&app, id).await?;
    let progress = job.progress.lock().clone();
    let name = app.registry.meta(id).map(|meta| meta.name);
    Ok(Json(json!({
        "id": job.id,
        "name": name,
        "pages": job.pages.len(),
        "progress": progress,
    })))
}

async fn remove(State(app): State<Shared>, Path(id): Path<Uuid>) -> Reply<StatusCode> {
    let registry = Arc::clone(&app.registry);
    if tokio::task::spawn_blocking(move || registry.remove(id)).await?? {
        Ok(StatusCode::NO_CONTENT)
    } else {
        Err(ApiError::missing("job"))
    }
}

async fn stop(State(app): State<Shared>) -> StatusCode {
    app.pipeline.stop();
    StatusCode::ACCEPTED
}

async fn blocks(State(app): State<Shared>, Path(id): Path<Uuid>) -> Reply<Json<Value>> {
    let job = job(&app, id).await?;
    let blocks = tokio::task::spawn_blocking(move || job.blocks()).await??;
    Ok(Json(json!(blocks)))
}

#[derive(Deserialize)]
struct Edit {
    text: String,
}

async fn edit(
    State(app): State<Shared>,
    Path((id, layer)): Path<(Uuid, String)>,
    Json(body): Json<Edit>,
) -> Reply<StatusCode> {
    let job = job(&app, id).await?;
    let layer: LayerId = layer.parse().map_err(|_| ApiError::missing("block"))?;
    if body.text.trim().is_empty() {
        return Err(ApiError::bad("a translation can't be blank"));
    }
    match job.edit(layer, body.text) {
        Ok(()) => Ok(StatusCode::NO_CONTENT),
        Err(Edited::NotTranslated) => Err(ApiError(
            StatusCode::CONFLICT,
            "the block has no translation to edit".to_owned(),
        )),
        Err(Edited::Failed(error)) => Err(error.into()),
    }
}

#[derive(Deserialize)]
struct View {
    /// `original` for the scan as uploaded; the page as drawn otherwise.
    view: Option<String>,
    /// With `original`, a JPEG this many pixels wide, for a thumbnail.
    w: Option<u32>,
}

async fn image(
    State(app): State<Shared>,
    Path((id, number)): Path<(Uuid, usize)>,
    Query(view): Query<View>,
) -> Reply<Response> {
    let job = job(&app, id).await?;
    if number == 0 || number > job.pages.len() {
        return Err(ApiError::missing("page"));
    }
    let original = view.view.as_deref() == Some("original");
    let width = view.w.map(|width| width.clamp(16, 2048));
    let (bytes, kind) = tokio::task::spawn_blocking(move || {
        if original {
            match width {
                Some(width) => job
                    .thumbnail(number, width)
                    .map(|bytes| (bytes, "image/jpeg")),
                None => job.original(number),
            }
        } else {
            job.render(&app.renderer, number)
                .map(|bytes| (bytes, "image/png"))
        }
    })
    .await??;
    Ok(([(header::CONTENT_TYPE, kind), (header::CACHE_CONTROL, "no-store")], bytes).into_response())
}

#[derive(Deserialize)]
struct Format {
    /// `cbz` for PNGs and `psd` for layered files, each in a ZIP; `khr` for
    /// the project, to open in the desktop app.
    format: Option<String>,
}

async fn download(
    State(app): State<Shared>,
    Path(id): Path<Uuid>,
    Query(format): Query<Format>,
) -> Reply<Response> {
    let job = job(&app, id).await?;
    let format = format.format.as_deref().unwrap_or("cbz").to_owned();
    if !["cbz", "psd", "khr"].contains(&format.as_str()) {
        return Err(ApiError::bad(format!("`{format}` isn't cbz, psd or khr")));
    }
    let stem = app
        .registry
        .meta(id)
        .map_or_else(|| "koharu".to_owned(), |meta| meta.name);
    let name = jobs::file_name(&stem);
    let (kind, extension) = match format.as_str() {
        "psd" => ("application/zip", "psd.zip"),
        "khr" => ("application/octet-stream", "khr"),
        _ => ("application/vnd.comicbook+zip", "cbz"),
    };
    let bytes = tokio::task::spawn_blocking(move || match format.as_str() {
        "khr" => job.export(),
        "psd" => job.archive(&app.renderer, true),
        _ => job.archive(&app.renderer, false),
    })
    .await??;
    let disposition = HeaderValue::from_str(&format!(
        "attachment; filename=\"{name}.{extension}\"; filename*=UTF-8''{}",
        percent(&format!("{stem}.{extension}"))
    ))
    .map_err(ApiError::bad)?;
    Ok((
        [
            (header::CONTENT_TYPE, HeaderValue::from_static(kind)),
            (header::CONTENT_DISPOSITION, disposition),
        ],
        bytes,
    )
        .into_response())
}

/// `text` percent-encoded for a header's `filename*`.
fn percent(text: &str) -> String {
    text.bytes()
        .map(|byte| match byte {
            b'A'..=b'Z' | b'a'..=b'z' | b'0'..=b'9' | b'-' | b'.' | b'_' => {
                char::from(byte).to_string()
            }
            _ => format!("%{byte:02X}"),
        })
        .collect()
}
