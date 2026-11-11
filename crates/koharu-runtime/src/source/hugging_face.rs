use std::{fmt::Write as _, path::PathBuf, sync::OnceLock};

use anyhow::Context as _;
use hf_hub::{HFClient, HFRepository, RepoType, repository::download::HFByteStream, split_id};

use crate::{Store, download, network};

static CLIENT: OnceLock<HFClient> = OnceLock::new();

fn client() -> anyhow::Result<HFClient> {
    if let Some(client) = CLIENT.get() {
        return Ok(client.clone());
    }
    let max_retries = network::config()?.max_retries as usize;
    let http = network::http()?;
    let client = HFClient::builder()
        .client(http)
        .cache_enabled(false)
        .retry_max_attempts(max_retries)
        .build()?;
    Ok(CLIENT.get_or_init(|| client).clone())
}

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
enum RepositoryKind {
    Model,
    Dataset,
}

/// An immutable file snapshot hosted by Hugging Face.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct HuggingFaceFile<'a> {
    kind: RepositoryKind,
    repository: &'a str,
    revision: &'a str,
    filename: &'a str,
}

impl<'a> HuggingFaceFile<'a> {
    #[must_use]
    pub const fn pinned(repository: &'a str, revision: &'a str, filename: &'a str) -> Self {
        Self {
            kind: RepositoryKind::Model,
            repository,
            revision,
            filename,
        }
    }

    #[must_use]
    pub const fn pinned_dataset(repository: &'a str, revision: &'a str, filename: &'a str) -> Self {
        Self {
            kind: RepositoryKind::Dataset,
            ..Self::pinned(repository, revision, filename)
        }
    }

    #[must_use]
    pub fn exists(self) -> bool {
        self.path().is_file()
    }

    #[tracing::instrument(skip_all)]
    pub async fn resolve(self) -> anyhow::Result<PathBuf> {
        Store::file(self.path(), move |stage| async move {
            let hub = async {
                let client = client()?;
                let (owner, name) = split_id(self.repository);
                download::receive(self.filename, &stage, async {
                    // hf-hub 1.0 builds metadata URLs with T::default(), so use typed repositories.
                    match self.kind {
                        RepositoryKind::Model => self.download(client.model(owner, name)).await,
                        RepositoryKind::Dataset => self.download(client.dataset(owner, name)).await,
                    }
                })
                .await
            }
            .await;
            let Err(hub_error) = hub else {
                return Ok(());
            };
            // hf-hub asks for the file's metadata first and turns any 404 of that request
            // into "Entry not found", also when the file exists and the answer came from a
            // mirror (`HF_ENDPOINT`), a proxy or the CDN behind the redirect. The plain URL
            // of the file does not depend on that request.
            tracing::warn!(
                file = self.filename,
                error = %format!("{hub_error:#}"),
                "hf-hub could not download the file; trying its plain resolve URL"
            );
            let url = self.direct_url();
            match download::fetch(&url, &stage).await {
                Ok(()) => Ok(()),
                Err(fetch_error) => {
                    let endpoint = std::env::var("HF_ENDPOINT").ok();
                    let diagnosis = diagnose(
                        endpoint.as_deref(),
                        self.repository,
                        self.revision,
                        self.filename,
                        &url,
                    )
                    .await;
                    Err(fetch_error.context(format!(
                        "hf-hub said: {hub_error:#}; Hugging Face answers: {diagnosis}"
                    )))
                    .with_context(|| {
                        format!(
                            "failed to download {} from {}",
                            self.filename, self.repository
                        )
                    })
                }
            }
        })
        .await
    }

    /// The plain URL of the file: `{endpoint}/{repository}/resolve/{revision}/{filename}`.
    fn direct_url(self) -> String {
        let endpoint = std::env::var("HF_ENDPOINT").ok();
        direct_url(
            endpoint.as_deref(),
            matches!(self.kind, RepositoryKind::Dataset),
            self.repository,
            self.revision,
            self.filename,
        )
    }

    async fn download(
        self,
        repository: HFRepository<impl RepoType>,
    ) -> anyhow::Result<(Option<u64>, HFByteStream)> {
        repository
            .download_file_stream()
            .filename(self.filename)
            .revision(self.revision)
            .send()
            .await
            .map_err(Into::into)
    }

    fn path(self) -> PathBuf {
        Store::root()
            .join("hugging-face")
            .join(match self.kind {
                RepositoryKind::Model => "models",
                RepositoryKind::Dataset => "datasets",
            })
            .join(self.repository.replace(['/', '\\'], "--"))
            .join("snapshots")
            .join(self.revision)
            .join(self.filename)
    }
}

fn response_header(response: &reqwest::Response, name: &str) -> String {
    response
        .headers()
        .get(name)
        .and_then(|value| value.to_str().ok())
        .unwrap_or("-")
        .to_owned()
}

/// What Hugging Face says about a file that could not be downloaded, for the error message: the
/// answer for the file itself (with the error code and message that Hugging Face sends along
/// with a 404) and whether the repository is visible at the revision and lists the file. It
/// tells a missing file from a repository this machine cannot see, and both from an answer that
/// did not come from Hugging Face (a proxy). Best effort: a failure here only shortens it.
async fn diagnose(
    endpoint: Option<&str>,
    repository: &str,
    revision: &str,
    filename: &str,
    url: &str,
) -> String {
    let Ok(client) = network::http() else {
        return "no HTTP client".to_owned();
    };
    let token = ["HF_TOKEN", "HUGGING_FACE_HUB_TOKEN"]
        .iter()
        .any(|name| std::env::var_os(name).is_some());
    let mut report = format!("token_env_set={token}");
    match client
        .get(url)
        .header(reqwest::header::RANGE, "bytes=0-0")
        .send()
        .await
    {
        Ok(response) => {
            let _ = write!(
                report,
                "; file: status={} url={} x-error-code={} x-error-message={:?} server={} via={} x-request-id={}",
                response.status().as_u16(),
                response.url(),
                response_header(&response, "x-error-code"),
                response_header(&response, "x-error-message"),
                response_header(&response, "server"),
                response_header(&response, "via"),
                response_header(&response, "x-request-id"),
            );
        }
        Err(error) => {
            let _ = write!(report, "; file request failed: {error:#}");
        }
    }
    let base = endpoint
        .map(str::trim)
        .filter(|endpoint| !endpoint.is_empty())
        .unwrap_or("https://huggingface.co")
        .trim_end_matches('/');
    let api = format!("{base}/api/models/{repository}/revision/{}", revision.replace('/', "%2F"));
    match client.get(&api).send().await {
        Ok(response) => {
            let status = response.status().as_u16();
            let error_code = response_header(&response, "x-error-code");
            let body = response.text().await.unwrap_or_default();
            let sha = body
                .split_once("\"sha\":\"")
                .map(|(_, rest)| rest.chars().take(12).collect::<String>())
                .unwrap_or_else(|| "-".to_owned());
            let _ = write!(
                report,
                "; repository: status={status} x-error-code={error_code} sha={sha} lists_the_file={}",
                body.contains(filename)
            );
        }
        Err(error) => {
            let _ = write!(report, "; repository request failed: {error:#}");
        }
    }
    report
}

/// `{endpoint}/{repository}/resolve/{revision}/{filename}`, with `datasets/` before the
/// repository of a dataset. The endpoint is Hugging Face unless `endpoint` says otherwise.
fn direct_url(
    endpoint: Option<&str>,
    dataset: bool,
    repository: &str,
    revision: &str,
    filename: &str,
) -> String {
    let endpoint = endpoint
        .map(str::trim)
        .filter(|endpoint| !endpoint.is_empty())
        .unwrap_or("https://huggingface.co")
        .trim_end_matches('/');
    let prefix = if dataset { "datasets/" } else { "" };
    let encode = |text: &str| text.replace('%', "%25").replace(' ', "%20").replace('#', "%23");
    let filename = filename.split('/').map(encode).collect::<Vec<_>>().join("/");
    format!(
        "{endpoint}/{prefix}{repository}/resolve/{}/{filename}",
        encode(revision).replace('/', "%2F")
    )
}

#[cfg(test)]
mod tests {
    use super::direct_url;

    #[test]
    fn the_plain_url_of_a_file_follows_the_hub_layout() {
        assert_eq!(
            direct_url(None, false, "Kellenok/PP-OCRv6_manga", "main", "det/manga_det_v0.2.onnx"),
            "https://huggingface.co/Kellenok/PP-OCRv6_manga/resolve/main/det/manga_det_v0.2.onnx"
        );
        assert_eq!(
            direct_url(Some("https://mirror.example/"), true, "o/d", "v1", "a b#.bin"),
            "https://mirror.example/datasets/o/d/resolve/v1/a%20b%23.bin"
        );
        assert_eq!(
            direct_url(Some(" "), false, "o/m", "feature/x", "f.bin"),
            "https://huggingface.co/o/m/resolve/feature%2Fx/f.bin"
        );
    }
}
