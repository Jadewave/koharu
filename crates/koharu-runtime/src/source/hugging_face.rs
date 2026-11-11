use std::{path::PathBuf, sync::OnceLock};

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
            download::fetch(&self.direct_url(), &stage)
                .await
                .with_context(|| {
                    format!(
                        "failed to download {} from {} (hf-hub said: {hub_error:#})",
                        self.filename, self.repository
                    )
                })
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
