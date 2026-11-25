//! Jobs: an upload as a project of pages on disk, the pipeline run over it,
//! and what the page shows of it as the run goes.

use std::{
    collections::{BTreeMap, BTreeSet, HashMap},
    io::{Cursor, Read as _, Write as _},
    path::PathBuf,
    sync::Arc,
    time::{SystemTime, UNIX_EPOCH},
};

use anyhow::{Context as _, Result, bail, ensure};
use image::ImageFormat;
use koharu_pipeline::{Event, Outcome, Pipeline, PipelineConfig, Request, Stage, Target};
use koharu_psd::Labels;
use koharu_renderer::Renderer;
use koharu_scene::{Content, Edit, Hash, LayerId, Op, PageId, Session};
use parking_lot::Mutex;
use serde::{Deserialize, Serialize};
use tokio::sync::broadcast;
use uuid::Uuid;

/// A file as it came in the upload.
pub struct Upload {
    pub name: String,
    pub bytes: Vec<u8>,
}

/// Where a job is, for the page to show.
#[derive(Clone, Debug, Serialize)]
pub struct Progress {
    /// `running`, `done`, `failed` or `stopped`.
    pub state: &'static str,
    /// The page stages the run does, and how many have finished.
    pub steps: usize,
    pub done: usize,
    /// Pages finished, for each stage.
    pub stages: BTreeMap<String, usize>,
    /// What the run started last, such as `page 3: translate`.
    pub activity: String,
    pub error: Option<String>,
}

impl Progress {
    fn running() -> Self {
        Self {
            state: "running",
            steps: 0,
            done: 0,
            stages: BTreeMap::new(),
            activity: "queued".to_owned(),
            error: None,
        }
    }

    /// Where a saved job stands: what its last run ended in, or `interrupted`
    /// when the server stopped while it ran.
    fn saved(state: &str) -> Self {
        let state = match state {
            "failed" => "failed",
            "stopped" => "stopped",
            "running" | "interrupted" => "interrupted",
            _ => "saved",
        };
        Self {
            state,
            activity: state.to_owned(),
            ..Self::running()
        }
    }
}

pub struct Job {
    pub id: Uuid,
    pub dir: PathBuf,
    pub session: Arc<Mutex<Session>>,
    pub pages: Vec<PageId>,
    pub progress: Mutex<Progress>,
}

impl Job {
    fn new(id: Uuid, dir: PathBuf, session: Session, progress: Progress) -> Self {
        let pages = session.project().pages().map(|page| page.id).collect();
        Self {
            id,
            dir,
            session: Arc::new(Mutex::new(session)),
            pages,
            progress: Mutex::new(progress),
        }
    }
}

/// What the library keeps of a job beside its project, in `job.json`.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Meta {
    pub id: Uuid,
    pub name: String,
    /// Seconds since the epoch.
    pub created: u64,
    pub pages: usize,
    /// How the job last stood: `running`, `done`, `failed`, `stopped` or
    /// `saved`, which a project nobody has run here stands in.
    pub state: String,
    pub translator: String,
    pub target: String,
}

/// A block of a page, as the page lists it.
#[derive(Debug, Serialize)]
pub struct BlockView {
    pub page: usize,
    pub id: String,
    pub source: String,
    pub translation: Option<String>,
}

pub struct Registry {
    dir: PathBuf,
    /// Every job the library holds, opened or not.
    metas: Mutex<HashMap<Uuid, Meta>>,
    /// The jobs opened since the server started; a saved job opens on demand.
    jobs: Mutex<HashMap<Uuid, Arc<Job>>>,
    /// The runs under way, by their number, so an event finds its job.
    runs: Mutex<HashMap<u64, Arc<Job>>>,
}

impl Registry {
    /// The library in `dir`: a folder for each job, holding its `pages.khr`
    /// and `job.json`.
    pub fn new(dir: PathBuf) -> Result<Self> {
        fs_err::create_dir_all(&dir)?;
        let mut metas = HashMap::new();
        for entry in fs_err::read_dir(&dir)?.filter_map(|entry| entry.ok()) {
            let folder = entry.path();
            let Ok(text) = fs_err::read_to_string(folder.join("job.json")) else {
                continue;
            };
            match serde_json::from_str::<Meta>(&text) {
                Ok(meta) if folder.join("pages.khr").is_file() => {
                    metas.insert(meta.id, meta);
                }
                Ok(_) => {}
                Err(error) => {
                    tracing::warn!(folder = %folder.display(), "an unreadable job: {error}");
                }
            }
        }
        tracing::info!(jobs = metas.len(), "the library's jobs");
        Ok(Self {
            dir,
            metas: Mutex::new(metas),
            jobs: Mutex::default(),
            runs: Mutex::default(),
        })
    }

    /// The jobs, the newest first, each as it stands now.
    pub fn list(&self) -> Vec<Meta> {
        let jobs = self.jobs.lock();
        let mut metas: Vec<Meta> = self
            .metas
            .lock()
            .values()
            .cloned()
            .map(|mut meta| {
                if let Some(job) = jobs.get(&meta.id) {
                    meta.state = job.progress.lock().state.to_owned();
                }
                meta
            })
            .collect();
        metas.sort_by(|a, b| b.created.cmp(&a.created).then_with(|| a.name.cmp(&b.name)));
        metas
    }

    pub fn meta(&self, id: Uuid) -> Option<Meta> {
        self.metas.lock().get(&id).cloned()
    }

    /// The job, opened from its project file if it isn't yet; `None` when the
    /// library has none. Opening reads the file, so call it off the runtime.
    pub fn get(&self, id: Uuid) -> Result<Option<Arc<Job>>> {
        let mut jobs = self.jobs.lock();
        if let Some(job) = jobs.get(&id) {
            return Ok(Some(Arc::clone(job)));
        }
        let Some(meta) = self.meta(id) else {
            return Ok(None);
        };
        let dir = self.dir.join(id.to_string());
        let session = Session::open(dir.join("pages.khr"))
            .with_context(|| format!("couldn't open the project of {}", meta.name))?;
        let job = Arc::new(Job::new(id, dir, session, Progress::saved(&meta.state)));
        jobs.insert(id, Arc::clone(&job));
        Ok(Some(job))
    }

    /// Writes `meta` beside its project and keeps it.
    fn keep(&self, meta: Meta) -> Result<()> {
        let path = self.dir.join(meta.id.to_string()).join("job.json");
        fs_err::write(path, serde_json::to_vec_pretty(&meta)?)?;
        self.metas.lock().insert(meta.id, meta);
        Ok(())
    }

    pub fn rename(&self, id: Uuid, name: &str) -> Result<bool> {
        let Some(mut meta) = self.meta(id) else {
            return Ok(false);
        };
        meta.name = name.trim().to_owned();
        self.keep(meta)?;
        Ok(true)
    }

    fn set_state(&self, id: Uuid, state: &str) {
        let Some(mut meta) = self.meta(id) else {
            return;
        };
        meta.state = state.to_owned();
        if let Err(error) = self.keep(meta) {
            tracing::warn!("couldn't record a job's state: {error:#}");
        }
    }

    fn new_dir(&self) -> Result<(Uuid, PathBuf)> {
        let id = Uuid::new_v4();
        let dir = self.dir.join(id.to_string());
        fs_err::create_dir_all(&dir)?;
        Ok((id, dir))
    }

    /// Makes a project of the uploaded images, in name order, keeps it in the
    /// library, and starts the run over it.
    pub fn create(
        &self,
        pipeline: &Pipeline,
        name: &str,
        uploads: Vec<Upload>,
        stages: BTreeSet<Stage>,
        config: PipelineConfig,
    ) -> Result<Arc<Job>> {
        let images = images(uploads)?;
        ensure!(!images.is_empty(), "the upload holds no images");
        let (id, dir) = self.new_dir()?;
        let made = (|| {
            let mut session = Session::create(dir.join("pages.khr"))?;
            let mut ops = Vec::new();
            for (index, bytes) in images.into_iter().enumerate() {
                let background = Some(Hash::of(&bytes));
                ops.push(Op::AddFile { bytes });
                ops.push(Op::AddPage {
                    id: None,
                    index,
                    size: None,
                    background,
                });
            }
            session.apply(Edit {
                label: "Import".to_owned(),
                ops,
            })?;
            anyhow::Ok(session)
        })();
        let session = match made {
            Ok(session) => session,
            Err(error) => {
                let _ = fs_err::remove_dir_all(&dir);
                return Err(error);
            }
        };
        let job = Arc::new(Job::new(id, dir, session, Progress::running()));
        self.keep(Meta {
            id,
            name: name.to_owned(),
            created: now(),
            pages: job.pages.len(),
            state: "running".to_owned(),
            translator: config.translator.model.clone(),
            target: config.translator.target.to_string(),
        })?;
        self.jobs.lock().insert(id, Arc::clone(&job));
        self.start(&job, pipeline, stages, config)?;
        Ok(job)
    }

    /// Keeps an uploaded `.khr` in the library as it came, and runs `run`'s
    /// stages over it if there are any.
    pub fn import(
        &self,
        pipeline: &Pipeline,
        name: &str,
        project: &[u8],
        config: &PipelineConfig,
        run: Option<BTreeSet<Stage>>,
    ) -> Result<Arc<Job>> {
        let (id, dir) = self.new_dir()?;
        let opened = fs_err::write(dir.join("pages.khr"), project)
            .map_err(anyhow::Error::from)
            .and_then(|()| Ok(Session::open(dir.join("pages.khr"))?));
        let session = match opened {
            Ok(session) => session,
            Err(error) => {
                let _ = fs_err::remove_dir_all(&dir);
                return Err(error.context("that isn't a Koharu project this build can open"));
            }
        };
        let job = Arc::new(Job::new(id, dir, session, Progress::saved("saved")));
        if job.pages.is_empty() {
            drop(job);
            let _ = fs_err::remove_dir_all(self.dir.join(id.to_string()));
            bail!("the project has no pages");
        }
        self.keep(Meta {
            id,
            name: name.to_owned(),
            created: now(),
            pages: job.pages.len(),
            state: "saved".to_owned(),
            translator: config.translator.model.clone(),
            target: config.translator.target.to_string(),
        })?;
        self.jobs.lock().insert(id, Arc::clone(&job));
        if let Some(stages) = run {
            self.start(&job, pipeline, stages, config.clone())?;
        }
        Ok(job)
    }

    /// Starts `stages` over the whole project, unless a run is under way on it.
    pub fn start(
        &self,
        job: &Arc<Job>,
        pipeline: &Pipeline,
        stages: BTreeSet<Stage>,
        config: PipelineConfig,
    ) -> Result<()> {
        self.start_on(job, pipeline, Target::Project, stages, config)
    }

    /// Starts `stages` over `target`, unless a run is under way on the job.
    pub fn start_on(
        &self,
        job: &Arc<Job>,
        pipeline: &Pipeline,
        target: Target,
        stages: BTreeSet<Stage>,
        config: PipelineConfig,
    ) -> Result<()> {
        let mut progress = job.progress.lock();
        ensure!(progress.state != "running", "a run is under way on this job");
        // The lock is held through the run's start, so no event of it is
        // heard before its job can be found.
        let mut runs = self.runs.lock();
        let run = pipeline.run(
            Arc::clone(&job.session),
            Request {
                target,
                stages,
                config: Some(config.clone()),
            },
        )?;
        runs.insert(run.0, Arc::clone(job));
        drop(runs);
        *progress = Progress::running();
        drop(progress);
        if let Some(mut meta) = self.meta(job.id) {
            meta.state = "running".to_owned();
            meta.translator = config.translator.model;
            meta.target = config.translator.target.to_string();
            if let Err(error) = self.keep(meta) {
                tracing::warn!("couldn't record a job's state: {error:#}");
            }
        }
        Ok(())
    }

    /// Forgets a job and deletes its folder, project included.
    pub fn remove(&self, id: Uuid) -> Result<bool> {
        let Some(meta) = self.metas.lock().remove(&id) else {
            return Ok(false);
        };
        let job = self.jobs.lock().remove(&id);
        self.runs.lock().retain(|_, found| found.id != id);
        // The project's file is locked while a session holds it.
        drop(job);
        fs_err::remove_dir_all(self.dir.join(meta.id.to_string()))?;
        Ok(true)
    }

    /// Follows the pipeline's events for good, on a task of its own.
    pub fn follow(self: &Arc<Self>, mut events: broadcast::Receiver<Event>) {
        let registry = Arc::clone(self);
        tokio::spawn(async move {
            loop {
                match events.recv().await {
                    Ok(event) => registry.hear(&event),
                    Err(broadcast::error::RecvError::Lagged(missed)) => {
                        tracing::warn!(missed, "the progress of a run fell behind");
                    }
                    Err(broadcast::error::RecvError::Closed) => break,
                }
            }
        });
    }

    fn hear(&self, event: &Event) {
        let (run, change): (u64, Box<dyn FnOnce(&Job, &mut Progress)>) = match event.clone() {
            Event::Queued { run, steps } => (
                run.0,
                Box::new(move |_, progress| progress.steps = steps),
            ),
            Event::Started { run, page, stage } => (
                run.0,
                Box::new(move |job, progress| {
                    let number = job.pages.iter().position(|&found| found == page);
                    progress.activity = match number {
                        Some(number) => format!("page {}: {stage}", number + 1),
                        None => stage.to_string(),
                    };
                }),
            ),
            Event::Finished { run, stage, .. } => (
                run.0,
                Box::new(move |_, progress| {
                    progress.done += 1;
                    *progress.stages.entry(stage.to_string()).or_default() += 1;
                }),
            ),
            Event::Ended { run, outcome } => (
                run.0,
                Box::new(move |_, progress| match outcome {
                    Outcome::Done => {
                        progress.state = "done";
                        progress.activity = "done".to_owned();
                    }
                    Outcome::Stopped => {
                        progress.state = "stopped";
                        progress.activity = "stopped".to_owned();
                    }
                    Outcome::Failed { stage, reason, .. } => {
                        progress.state = "failed";
                        progress.activity = format!("{stage} failed");
                        progress.error = Some(reason);
                    }
                }),
            ),
        };
        let ended = matches!(event, Event::Ended { .. });
        let job = {
            let mut runs = self.runs.lock();
            if ended {
                runs.remove(&run)
            } else {
                runs.get(&run).cloned()
            }
        };
        if let Some(job) = job {
            let state = {
                let mut progress = job.progress.lock();
                change(&job, &mut progress);
                progress.state
            };
            if ended {
                self.set_state(job.id, state);
            }
        }
    }
}

fn now() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |elapsed| elapsed.as_secs())
}

/// The images of an upload, in name order: each file that is one, and the
/// images inside each ZIP or CBZ, so a whole chapter can go up as one file.
fn images(uploads: Vec<Upload>) -> Result<Vec<Vec<u8>>> {
    let mut found: Vec<(String, Vec<u8>)> = Vec::new();
    for upload in uploads {
        let name = upload.name.to_lowercase();
        if name.ends_with(".zip") || name.ends_with(".cbz") {
            let mut archive = zip::ZipArchive::new(Cursor::new(upload.bytes))
                .with_context(|| format!("{} isn't a ZIP or CBZ", upload.name))?;
            for index in 0..archive.len() {
                let mut entry = archive.by_index(index)?;
                let Ok(entry_name) = entry.name().map(|name| name.into_owned()) else {
                    continue;
                };
                if entry.is_dir() || ImageFormat::from_path(&entry_name).is_err() {
                    continue;
                }
                let mut bytes = Vec::new();
                entry.read_to_end(&mut bytes)?;
                found.push((entry_name, bytes));
            }
        } else {
            found.push((upload.name, upload.bytes));
        }
    }
    found.sort_by(|(a, _), (b, _)| alphanumeric_sort::compare_str(a, b));
    let mut images = Vec::with_capacity(found.len());
    for (name, bytes) in found {
        let reader = image::ImageReader::new(Cursor::new(&bytes))
            .with_guessed_format()?;
        reader
            .into_dimensions()
            .with_context(|| format!("{name} isn't an image"))?;
        images.push(bytes);
    }
    Ok(images)
}

impl Job {
    fn page(&self, number: usize) -> Result<PageId> {
        number
            .checked_sub(1)
            .and_then(|index| self.pages.get(index))
            .copied()
            .with_context(|| format!("the job has no page {number}"))
    }

    /// Page `number`'s id, counting from 1.
    pub fn page_id(&self, number: usize) -> Result<PageId> {
        self.page(number)
    }

    /// Page `number`'s layers and values as the project holds them, as the
    /// editor reads them.
    pub fn scene(&self, number: usize) -> Result<serde_json::Value> {
        let page = self.page(number)?;
        let session = self.session.lock();
        let project = session.project();
        Ok(serde_json::to_value(project.page(page)?)?)
    }

    /// Commits an edit as one undo step.
    pub fn apply(&self, edit: Edit) -> Result<()> {
        self.session.lock().apply(edit)?;
        Ok(())
    }

    /// Undoes the last step; whether there was one.
    pub fn undo(&self) -> Result<bool> {
        Ok(self.session.lock().undo()?.is_some())
    }

    /// Redoes the step last undone; whether there was one.
    pub fn redo(&self) -> Result<bool> {
        Ok(self.session.lock().redo()?.is_some())
    }

    /// Page `number`'s scan, `width` pixels wide, as a JPEG.
    pub fn thumbnail(&self, number: usize, width: u32) -> Result<Vec<u8>> {
        let (bytes, _) = self.original(number)?;
        let scan = image::load_from_memory(&bytes)?;
        let height = (u64::from(scan.height()) * u64::from(width) / u64::from(scan.width().max(1)))
            .max(1) as u32;
        let small = scan.thumbnail(width.max(16), height.max(16)).to_rgb8();
        let mut out = Cursor::new(Vec::new());
        small.write_to(&mut out, ImageFormat::Jpeg)?;
        Ok(out.into_inner())
    }

    /// Page `number`, from 1, as the app draws it, as a PNG.
    pub fn render(&self, renderer: &Renderer, number: usize) -> Result<Vec<u8>> {
        let page = self.page(number)?;
        let frame = renderer.render(&self.session.lock(), page)?;
        png(&frame.rasterize()?.image)
    }

    /// Page `number`'s scan as it was uploaded, and its type.
    pub fn original(&self, number: usize) -> Result<(Vec<u8>, &'static str)> {
        let page = self.page(number)?;
        let session = self.session.lock();
        let project = session.project();
        let background = project
            .page(page)?
            .background
            .context("the page has no scan")?;
        let bytes = session.file(&background.file)?;
        let kind = match image::guess_format(&bytes) {
            Ok(ImageFormat::Jpeg) => "image/jpeg",
            Ok(ImageFormat::WebP) => "image/webp",
            Ok(ImageFormat::Gif) => "image/gif",
            _ => "image/png",
        };
        Ok((bytes, kind))
    }

    /// Every block with text, page by page.
    pub fn blocks(&self) -> Result<Vec<BlockView>> {
        let session = self.session.lock();
        let project = session.project();
        let mut views = Vec::new();
        for (index, &page) in self.pages.iter().enumerate() {
            for (id, block) in project.blocks(page)? {
                let source = block.source.content.text();
                if source.trim().is_empty() {
                    continue;
                }
                views.push(BlockView {
                    page: index + 1,
                    id: id.to_string(),
                    source,
                    translation: block
                        .translation
                        .as_ref()
                        .map(|translation| translation.content.text()),
                });
            }
        }
        Ok(views)
    }

    /// Sets a translated block's text; the block's place stays as it was.
    pub fn edit(&self, layer: LayerId, text: String) -> Result<(), Edited> {
        let mut session = self.session.lock();
        let translated = matches!(
            session.project().layer(layer).map(|layer| &layer.content),
            Ok(Content::Block(block)) if block.translation.is_some()
        );
        if !translated {
            return Err(Edited::NotTranslated);
        }
        session
            .apply(Edit {
                label: "Edit translation".to_owned(),
                ops: vec![Op::SetText {
                    layer: layer.into(),
                    text,
                }],
            })
            .map_err(|error| Edited::Failed(error.into()))?;
        Ok(())
    }

    /// The project as a `.khr` file, a consistent copy made while the job's
    /// session stays open.
    pub fn export(&self) -> Result<Vec<u8>> {
        let copy = self.dir.join(format!("export-{}.khr", Uuid::new_v4()));
        let made = self.session.lock().export(&copy);
        let bytes = made
            .map_err(anyhow::Error::from)
            .and_then(|()| Ok(fs_err::read(&copy)?));
        let _ = fs_err::remove_file(&copy);
        bytes
    }

    /// Every page in one file: a CBZ of PNGs, or a ZIP of PSDs.
    pub fn archive(&self, renderer: &Renderer, psd: bool) -> Result<Vec<u8>> {
        let digits = self.pages.len().to_string().len().max(3);
        let mut archive = zip::ZipWriter::new(Cursor::new(Vec::new()));
        // PNGs are compressed already, and a PSD barely more than its layers.
        let options = zip::write::SimpleFileOptions::default()
            .compression_method(zip::CompressionMethod::Stored);
        let labels = Labels {
            background: "Background".to_owned(),
            cleanup: "Clean-up".to_owned(),
            outline: "Outline".to_owned(),
        };
        for (index, &page) in self.pages.iter().enumerate() {
            let number = format!("{:0digits$}", index + 1);
            if psd {
                let bytes = koharu_psd::export(&self.session.lock(), renderer, page, &labels)?;
                archive.start_file(format!("{number}.psd"), options)?;
                archive.write_all(&bytes)?;
            } else {
                archive.start_file(format!("{number}.png"), options)?;
                archive.write_all(&self.render(renderer, index + 1)?)?;
            }
        }
        Ok(archive.finish()?.into_inner())
    }
}

pub enum Edited {
    NotTranslated,
    Failed(anyhow::Error),
}

fn png(image: &image::RgbaImage) -> Result<Vec<u8>> {
    let mut bytes = Cursor::new(Vec::new());
    image.write_to(&mut bytes, ImageFormat::Png)?;
    Ok(bytes.into_inner())
}

/// A file name for a download, in ASCII so it is safe in a header; `filename*`
/// carries the real one.
pub fn file_name(name: &str) -> String {
    let name: String = sanitize_filename::sanitize(name)
        .chars()
        .map(|character| {
            if character.is_ascii_graphic() && character != '"' && character != '\\' {
                character
            } else {
                '_'
            }
        })
        .collect();
    if name.trim_matches('_').is_empty() {
        "koharu".to_owned()
    } else {
        name
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn saved(dir: &std::path::Path, name: &str, state: &str) -> Uuid {
        let id = Uuid::new_v4();
        let folder = dir.join(id.to_string());
        fs_err::create_dir_all(&folder).unwrap();
        fs_err::write(folder.join("pages.khr"), b"").unwrap();
        let meta = Meta {
            id,
            name: name.to_owned(),
            created: 7,
            pages: 3,
            state: state.to_owned(),
            translator: "koharu/index-translate-9b".to_owned(),
            target: "English".to_owned(),
        };
        fs_err::write(folder.join("job.json"), serde_json::to_vec(&meta).unwrap()).unwrap();
        id
    }

    #[test]
    fn the_library_lists_the_jobs_a_restart_finds() {
        let dir = tempfile::tempdir().unwrap();
        let id = saved(dir.path(), "One Piece 1", "done");
        let registry = Registry::new(dir.path().to_owned()).unwrap();
        let jobs = registry.list();
        assert_eq!(jobs.len(), 1);
        assert_eq!((jobs[0].id, jobs[0].name.as_str(), jobs[0].pages), (id, "One Piece 1", 3));
    }

    #[test]
    fn a_folder_without_a_readable_job_is_left_out() {
        let dir = tempfile::tempdir().unwrap();
        saved(dir.path(), "Kept", "done");
        let broken = dir.path().join("broken");
        fs_err::create_dir_all(&broken).unwrap();
        fs_err::write(broken.join("job.json"), b"{not json").unwrap();
        fs_err::write(broken.join("pages.khr"), b"").unwrap();
        fs_err::create_dir_all(dir.path().join("no-job-file")).unwrap();
        let registry = Registry::new(dir.path().to_owned()).unwrap();
        assert_eq!(registry.list().len(), 1);
    }

    #[test]
    fn a_rename_is_kept_beside_the_project() {
        let dir = tempfile::tempdir().unwrap();
        let id = saved(dir.path(), "Old", "done");
        let registry = Registry::new(dir.path().to_owned()).unwrap();
        assert!(registry.rename(id, "  New  ").unwrap());
        let again = Registry::new(dir.path().to_owned()).unwrap();
        assert_eq!(again.meta(id).unwrap().name, "New");
        assert!(!registry.rename(Uuid::new_v4(), "x").unwrap());
    }

    #[test]
    fn a_job_the_server_left_running_comes_back_interrupted() {
        assert_eq!(Progress::saved("running").state, "interrupted");
        assert_eq!(Progress::saved("failed").state, "failed");
        assert_eq!(Progress::saved("done").state, "saved");
    }

    #[test]
    fn a_download_name_is_ascii() {
        assert_eq!(file_name("Dr. STONE"), "Dr._STONE");
        assert_eq!(file_name("ワンピース"), "koharu");
    }

    #[test]
    fn a_removed_job_leaves_the_library() {
        let dir = tempfile::tempdir().unwrap();
        let id = saved(dir.path(), "Gone", "done");
        let registry = Registry::new(dir.path().to_owned()).unwrap();
        assert!(registry.remove(id).unwrap());
        assert!(registry.list().is_empty());
        assert!(!dir.path().join(id.to_string()).exists());
        assert!(!registry.remove(id).unwrap());
    }
}
