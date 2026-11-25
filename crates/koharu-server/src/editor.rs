//! What the editor page needs beyond the pages' images: a page's layers as
//! the project holds them, edits of them, undo and redo, the fonts, and runs
//! of stages over blocks or a painted area.
//!
//! An edit is the scene's own `Edit`, so the page can do whatever its ops can:
//! the same JSON the desktop editor and the agents send.

use std::collections::{BTreeSet, HashMap};

use axum::{
    Json, Router,
    extract::{Path, State},
    http::StatusCode,
    routing::{get, post},
};
use base64::Engine as _;
use koharu_pipeline::{Area, Stage, Target};
use koharu_scene::{Edit, LayerId, Rect};
use serde::Deserialize;
use serde_json::{Value, json};
use uuid::Uuid;

use super::{ApiError, Reply, Shared, by_id, job, overridden};

pub fn routes() -> Router<Shared> {
    Router::new()
        .route("/api/fonts", get(fonts))
        .route("/api/jobs/{id}/pages/{number}/scene", get(scene))
        .route("/api/jobs/{id}/edit", post(edit))
        .route("/api/jobs/{id}/undo", post(undo))
        .route("/api/jobs/{id}/redo", post(redo))
        .route("/api/jobs/{id}/process", post(process))
}

async fn fonts() -> Reply<Json<Value>> {
    Ok(Json(json!(koharu_renderer::fonts::list().await?)))
}

/// Page `number`'s size, background and layers, each block with its source and
/// translation, as the editor draws them.
async fn scene(
    State(app): State<Shared>,
    Path((id, number)): Path<(Uuid, usize)>,
) -> Reply<Json<Value>> {
    let job = job(&app, id).await?;
    let page = tokio::task::spawn_blocking(move || job.scene(number))
        .await?
        .map_err(|_| ApiError::missing("page"))?;
    Ok(Json(page))
}

/// One undo step: the scene's `Edit`, `{"label": ..., "ops": [...]}`.
async fn edit(
    State(app): State<Shared>,
    Path(id): Path<Uuid>,
    Json(edit): Json<Edit>,
) -> Reply<StatusCode> {
    if edit.ops.is_empty() {
        return Err(ApiError::bad("an edit needs at least one op"));
    }
    let job = job(&app, id).await?;
    tokio::task::spawn_blocking(move || job.apply(edit))
        .await?
        .map_err(ApiError::bad)?;
    Ok(StatusCode::NO_CONTENT)
}

async fn undo(State(app): State<Shared>, Path(id): Path<Uuid>) -> Reply<Json<Value>> {
    let job = job(&app, id).await?;
    let changed = tokio::task::spawn_blocking(move || job.undo()).await??;
    Ok(Json(json!({"changed": changed})))
}

async fn redo(State(app): State<Shared>, Path(id): Path<Uuid>) -> Reply<Json<Value>> {
    let job = job(&app, id).await?;
    let changed = tokio::task::spawn_blocking(move || job.redo()).await??;
    Ok(Json(json!({"changed": changed})))
}

#[derive(Deserialize)]
struct Process {
    /// Stage ids: `detect`, `recognize`, `inpaint`, `translate`.
    stages: Vec<String>,
    /// Blocks' layer ids; the stages run on these alone.
    #[serde(default)]
    blocks: Vec<String>,
    /// Page numbers from 1; the stages run on these alone. With neither this
    /// nor `blocks`, on every page.
    #[serde(default)]
    pages: Vec<usize>,
    /// A painted area to inpaint: the run is `inpaint` on that and nothing
    /// else.
    #[serde(default)]
    area: Option<Painted>,
    /// The same overrides an upload has.
    #[serde(default)]
    settings: HashMap<String, String>,
}

/// A byte a pixel over a rectangle of a page, in page pixels, painted where it
/// isn't 0, in base 64.
#[derive(Deserialize)]
struct Painted {
    page: usize,
    x: i32,
    y: i32,
    width: u32,
    height: u32,
    pixels: String,
    #[serde(default)]
    prompt: Option<String>,
}

async fn process(
    State(app): State<Shared>,
    Path(id): Path<Uuid>,
    Json(body): Json<Process>,
) -> Reply<StatusCode> {
    let job = job(&app, id).await?;
    let config = overridden(&app.config, &body.settings).map_err(ApiError::bad)?;
    let (target, stages) = if let Some(area) = body.area {
        let page = job.page_id(area.page).map_err(ApiError::bad)?;
        let pixels = base64::engine::general_purpose::STANDARD
            .decode(area.pixels.as_bytes())
            .map_err(ApiError::bad)?;
        if area.width == 0
            || area.height == 0
            || pixels.len() as u64 != u64::from(area.width) * u64::from(area.height)
        {
            return Err(ApiError::bad("the painted area has a byte for each pixel"));
        }
        let rect = Rect {
            x: area.x,
            y: area.y,
            width: area.width,
            height: area.height,
        };
        (
            Target::Area {
                page,
                area: Area { rect, pixels },
                prompt: area.prompt.filter(|prompt| !prompt.trim().is_empty()),
            },
            BTreeSet::from([Stage::Inpaint]),
        )
    } else {
        let stages: BTreeSet<Stage> = body
            .stages
            .iter()
            .map(|stage| by_id(stage.trim()))
            .collect::<anyhow::Result<_>>()
            .map_err(ApiError::bad)?;
        if stages.is_empty() {
            return Err(ApiError::bad("name at least one stage"));
        }
        let target = if !body.blocks.is_empty() {
            let blocks = body
                .blocks
                .iter()
                .map(|block| block.parse::<LayerId>())
                .collect::<Result<Vec<_>, _>>()
                .map_err(|_| ApiError::bad("a block id isn't one"))?;
            Target::Blocks(blocks)
        } else if !body.pages.is_empty() {
            let pages = body
                .pages
                .iter()
                .map(|&number| job.page_id(number))
                .collect::<anyhow::Result<Vec<_>>>()
                .map_err(ApiError::bad)?;
            Target::Pages(pages)
        } else {
            Target::Project
        };
        (target, stages)
    };
    tokio::task::spawn_blocking(move || {
        app.registry
            .start_on(&job, &app.pipeline, target, stages, config)
    })
    .await?
    .map_err(|error| ApiError(StatusCode::CONFLICT, format!("{error:#}")))?;
    Ok(StatusCode::ACCEPTED)
}
