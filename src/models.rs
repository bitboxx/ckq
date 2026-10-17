//! The embedding models ckq knows, and where their weights live.
//!
//! GGUF only, run by llama.cpp: the five aliases, their Hugging Face repo
//! and file, dimensions, pooling, prompts and default thresholds.
//! `gemma-q4` is the default.

use anyhow::{Result, anyhow, bail};
use std::path::PathBuf;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Pooling {
    Mean,
    /// Last token, for decoder models such as Qwen3-Embedding.
    Last,
}

#[derive(Debug, Clone)]
pub struct ModelSpec {
    pub alias: &'static str,
    /// Hugging Face repository, also the model's name in the index metadata.
    pub repo: &'static str,
    pub file: &'static str,
    pub dims: usize,
    /// The model's trained context, in tokens.
    pub context: usize,
    pub pooling: Pooling,
    pub query_prefix: &'static str,
    pub document_prefix: &'static str,
    /// Default `--threshold` for `--sem`; scores are not comparable across models.
    pub threshold: f32,
    /// First 16 hex of the blake3 of `file` as the registry downloads it, where
    /// known. An embedded copy of exactly that file keeps the plain build's
    /// signature; any other weights under the same alias do not. Read only by
    /// builds with the `embedded-model` feature.
    #[cfg_attr(not(feature = "embedded-model"), allow(dead_code))]
    pub blake3: Option<&'static str>,
}

/// The default model: `gemma-q4`, or the embedded alias in a build that
/// carries one inside the binary.
#[cfg(not(feature = "embedded-model"))]
pub const DEFAULT_ALIAS: &str = "gemma-q4";
#[cfg(feature = "embedded-model")]
pub const DEFAULT_ALIAS: &str = env!("CKQ_EMBED_ALIAS");

const GEMMA_Q4: ModelSpec = ModelSpec {
    alias: "gemma-q4",
    repo: "ggml-org/embeddinggemma-300M-qat-q4_0-GGUF",
    file: "embeddinggemma-300M-qat-Q4_0.gguf",
    dims: 768,
    context: 2048,
    pooling: Pooling::Mean,
    // EmbeddingGemma is trained with a task instruction on the query and a
    // title slot on the document. "none" is what the model card uses when
    // there is no title, and ck chunks have none to give.
    query_prefix: "task: search result | query: ",
    document_prefix: "title: none | text: ",
    threshold: 0.40,
    blake3: Some("7311e127d8aa9cb9"),
};

const GEMMA_Q8: ModelSpec = ModelSpec {
    alias: "gemma-gguf",
    repo: "ggml-org/embeddinggemma-300M-GGUF",
    file: "embeddinggemma-300M-Q8_0.gguf",
    dims: 768,
    context: 2048,
    pooling: Pooling::Mean,
    query_prefix: "task: search result | query: ",
    document_prefix: "title: none | text: ",
    threshold: 0.40,
    blake3: None,
};

const GRANITE: ModelSpec = ModelSpec {
    alias: "granite-gguf",
    repo: "bartowski/granite-embedding-278m-multilingual-GGUF",
    file: "granite-embedding-278m-multilingual-Q8_0.gguf",
    dims: 768,
    context: 512,
    pooling: Pooling::Mean,
    query_prefix: "",
    document_prefix: "",
    threshold: 0.55,
    blake3: None,
};

const BGE_M3: ModelSpec = ModelSpec {
    alias: "bge-m3-gguf",
    repo: "gpustack/bge-m3-GGUF",
    file: "bge-m3-Q8_0.gguf",
    dims: 1024,
    context: 8192,
    pooling: Pooling::Mean,
    query_prefix: "",
    document_prefix: "",
    threshold: 0.35,
    blake3: None,
};

const QWEN3: ModelSpec = ModelSpec {
    alias: "qwen3-gguf",
    repo: "Qwen/Qwen3-Embedding-0.6B-GGUF",
    file: "Qwen3-Embedding-0.6B-Q8_0.gguf",
    dims: 1024,
    context: 8192,
    pooling: Pooling::Last,
    // Qwen3 takes an instruction on the query only. The task sentence is the
    // one from the model card; the documents go in bare.
    query_prefix: "Instruct: Given a search query, retrieve relevant passages that answer it\nQuery: ",
    document_prefix: "",
    threshold: 0.35,
    blake3: None,
};

/// The registry. `gemma-q4`, `gemma-gguf`, `granite-gguf`, `bge-m3-gguf`, `qwen3-gguf`.
pub fn all() -> &'static [ModelSpec] {
    &[GEMMA_Q4, GEMMA_Q8, GRANITE, BGE_M3, QWEN3]
}

/// An alias, or a repo name as stored in an index, to its spec.
pub fn resolve(name: &str) -> Result<&'static ModelSpec> {
    all()
        .iter()
        .find(|m| m.alias == name || m.repo == name)
        .ok_or_else(|| {
            let aliases: Vec<&str> = all().iter().map(|m| m.alias).collect();
            anyhow!(
                "Unknown model '{name}'. Available models: {}",
                aliases.join(", ")
            )
        })
}

/// The GGUF on disk: `CKQ_GGUF_FILE` when set (for trying another build of the
/// same model), else the Hugging Face cache under `~/.cache/ck/models`,
/// downloading on first use. Same cache layout as the old engine, so nothing
/// already downloaded is fetched again.
///
/// In a build that carries the GGUF of this alias inside the binary, the
/// embedded bytes come out to `~/.cache/ck/models/embedded/<blake3>.gguf`
/// instead: written once, offline, never downloaded.
pub fn gguf_path(spec: &ModelSpec) -> Result<PathBuf> {
    if let Some(override_path) = std::env::var_os("CKQ_GGUF_FILE") {
        let path = PathBuf::from(override_path);
        if !path.is_file() {
            bail!("CKQ_GGUF_FILE does not point at a file: {}", path.display());
        }
        return Ok(path);
    }
    #[cfg(feature = "embedded-model")]
    if Some(spec.alias) == embedded_alias() {
        return embedded_gguf_path();
    }
    let cache = hf_hub::api::sync::ApiBuilder::new()
        .with_cache_dir(models_cache_dir())
        .build()
        .map_err(|e| anyhow::anyhow!("hf-hub: {e}"))?;
    let repo = cache.model(spec.repo.to_string());
    let path = repo
        .get(spec.file)
        .map_err(|e| anyhow::anyhow!("downloading {} ({e})", spec.repo))?;
    Ok(path)
}

fn models_cache_dir() -> PathBuf {
    if let Some(home) = dirs::home_dir() {
        return home.join(".cache").join("ck").join("models");
    }
    std::env::current_dir()
        .unwrap_or_default()
        .join(".ck_models")
}

/// The alias whose GGUF this build carries inside the binary
/// (`embedded-model` feature), which is that build's default model. `None`
/// in a normal build, where every model downloads on first use.
#[cfg(feature = "embedded-model")]
pub fn embedded_alias() -> Option<&'static str> {
    Some(embedded::ALIAS)
}

#[cfg(not(feature = "embedded-model"))]
pub fn embedded_alias() -> Option<&'static str> {
    None
}

/// The one GGUF this build carries: the bytes, and the first 16 hex of
/// their blake3, computed by build.rs so a run never re-hashes 278 MB.
#[cfg(feature = "embedded-model")]
mod embedded {
    pub const GGUF: &[u8] = include_bytes!(env!("CKQ_EMBED_GGUF"));
    pub const ALIAS: &str = env!("CKQ_EMBED_ALIAS");
    pub const HASH: &str = env!("CKQ_EMBED_BLAKE3");
}

/// The embedded bytes as a path llama.cpp can load: llama.cpp reads only
/// from a file, so they are written out once, when absent.
#[cfg(feature = "embedded-model")]
fn embedded_gguf_path() -> Result<PathBuf> {
    install_gguf(
        &models_cache_dir().join("embedded"),
        &format!("{}.gguf", embedded::HASH),
        embedded::GGUF,
    )
}

/// `bytes` at `dir/name`, written once: a file already there of the right
/// size is trusted; anything else (absent, wrong size, a half-written temp
/// left behind by a crashed process) is replaced by writing a per-process
/// temp file in the same directory, fsync, and an atomic rename — two
/// processes racing both end up with one complete file.
#[cfg(feature = "embedded-model")]
fn install_gguf(dir: &std::path::Path, name: &str, bytes: &[u8]) -> Result<PathBuf> {
    let path = dir.join(name);
    if let Ok(meta) = std::fs::metadata(&path)
        && meta.len() == bytes.len() as u64
    {
        return Ok(path);
    }
    std::fs::create_dir_all(dir)?;
    let tmp = dir.join(format!(".{name}.{}.tmp", std::process::id()));
    {
        use std::io::Write as _;
        let mut file = std::fs::File::create(&tmp)?;
        file.write_all(bytes)?;
        file.sync_all()?;
    }
    std::fs::rename(&tmp, &path)?;
    Ok(path)
}

/// What has to match for two vectors to be comparable: the model, the
/// prompts, the pooling and the chunker version. Stored in the index; a
/// mismatch refuses `--index` and names `--clean`.
pub fn signature(spec: &ModelSpec) -> String {
    let mut hasher = blake3::Hasher::new();
    hasher.update(spec.repo.as_bytes());
    hasher.update(b"\0");
    hasher.update(spec.file.as_bytes());
    hasher.update(b"\0");
    hasher.update(spec.query_prefix.as_bytes());
    hasher.update(b"\0");
    hasher.update(spec.document_prefix.as_bytes());
    hasher.update(b"\0");
    hasher.update(match spec.pooling {
        Pooling::Mean => b"mean",
        Pooling::Last => b"last",
    });
    hasher.update(b"\0");
    hasher.update(&crate::chunk::TARGET_TOKENS.to_le_bytes());
    hasher.update(&crate::chunk::OVERLAP_TOKENS.to_le_bytes());
    if let Some(marker) = weights_marker(spec) {
        hasher.update(b"\0");
        hasher.update(marker.as_bytes());
    }
    hasher.finalize().to_hex()[..16].to_string()
}

/// What sets the weights apart when they are not the registry's own file:
/// a `CKQ_GGUF_FILE` override (its path, size and mtime, which are cheap), or
/// embedded bytes whose hash differs from the registry's known one, such as a
/// copy with a requantised token table. `None` for the registry's file, so a
/// plain build and a build embedding that same file share an index.
fn weights_marker(spec: &ModelSpec) -> Option<String> {
    weights_marker_with(spec, std::env::var_os("CKQ_GGUF_FILE").as_deref())
}

fn weights_marker_with(
    spec: &ModelSpec,
    gguf_override: Option<&std::ffi::OsStr>,
) -> Option<String> {
    if let Some(path) = gguf_override {
        let meta = std::fs::metadata(path).ok();
        let size = meta.as_ref().map_or(0, |m| m.len());
        let mtime = meta
            .and_then(|m| m.modified().ok())
            .and_then(|t| t.duration_since(std::time::UNIX_EPOCH).ok())
            .map_or(0, |d| d.as_nanos());
        return Some(format!(
            "override:{}:{size}:{mtime}",
            path.to_string_lossy()
        ));
    }
    #[cfg(feature = "embedded-model")]
    if embedded_alias() == Some(spec.alias) && spec.blake3 != Some(embedded::HASH) {
        return Some(format!("embedded:{}", embedded::HASH));
    }
    let _ = spec;
    None
}

#[cfg(all(test, feature = "embedded-model"))]
mod tests {
    use super::install_gguf;
    use tempfile::TempDir;

    /// A directory that does not exist yet is created, and the bytes land
    /// in it complete.
    #[test]
    fn install_creates_the_directory_and_writes_the_bytes() {
        let dir = TempDir::new().unwrap();
        let target = dir.path().join("embedded");
        let path = install_gguf(&target, "hash.gguf", b"gguf bytes").unwrap();
        assert_eq!(std::fs::read(&path).unwrap(), b"gguf bytes");
    }

    /// A file already there of the right size is trusted as-is: it is not
    /// rewritten.
    #[test]
    fn a_right_size_file_is_trusted() {
        let dir = TempDir::new().unwrap();
        // Same length, different content: what a trusted file must keep.
        std::fs::write(dir.path().join("hash.gguf"), "XXXX bytes").unwrap();
        let path = install_gguf(dir.path(), "hash.gguf", b"gguf bytes").unwrap();
        assert_eq!(std::fs::read(&path).unwrap(), b"XXXX bytes");
    }

    /// A half-written temp file a crashed process left behind is not the
    /// target file: install still produces a complete one.
    #[test]
    fn a_leftover_temp_file_does_not_count() {
        let dir = TempDir::new().unwrap();
        std::fs::write(dir.path().join(".hash.gguf.999999.tmp"), "gguf by").unwrap();
        let path = install_gguf(dir.path(), "hash.gguf", b"gguf bytes").unwrap();
        assert_eq!(std::fs::read(&path).unwrap(), b"gguf bytes");
        assert!(dir.path().join(".hash.gguf.999999.tmp").exists());
    }

    /// A file of the wrong size is replaced with the embedded bytes.
    #[test]
    fn a_wrong_size_file_is_replaced() {
        let dir = TempDir::new().unwrap();
        std::fs::write(dir.path().join("hash.gguf"), "truncated").unwrap();
        let path = install_gguf(dir.path(), "hash.gguf", b"gguf bytes").unwrap();
        assert_eq!(std::fs::read(&path).unwrap(), b"gguf bytes");
    }

    /// Installing twice in a row ends with the same one complete file.
    #[test]
    fn install_is_idempotent() {
        let dir = TempDir::new().unwrap();
        let first = install_gguf(dir.path(), "hash.gguf", b"gguf bytes").unwrap();
        let second = install_gguf(dir.path(), "hash.gguf", b"gguf bytes").unwrap();
        assert_eq!(first, second);
        assert_eq!(std::fs::read(&first).unwrap(), b"gguf bytes");
        let entries: Vec<_> = std::fs::read_dir(dir.path())
            .unwrap()
            .flatten()
            .map(|e| e.file_name())
            .collect();
        assert_eq!(entries, ["hash.gguf"]);
    }
}

#[cfg(test)]
mod signature_tests {
    use super::*;

    #[test]
    fn only_weights_other_than_the_registry_file_add_a_marker() {
        let spec = resolve("gemma-q4").unwrap();
        let marker = weights_marker_with(spec, None);
        // A build embedding other bytes under this alias, such as a copy with
        // a requantised token table, must not share an index with the plain one.
        #[cfg(feature = "embedded-model")]
        if embedded_alias() == Some("gemma-q4") && spec.blake3 != Some(embedded::HASH) {
            assert_eq!(marker, Some(format!("embedded:{}", embedded::HASH)));
            return;
        }
        assert_eq!(marker, None);
    }

    #[test]
    fn an_override_file_marks_the_signature() {
        let spec = resolve("gemma-q4").unwrap();
        let marker = weights_marker_with(spec, Some(std::ffi::OsStr::new("/nonexistent/x.gguf")));
        assert!(marker.unwrap().starts_with("override:/nonexistent/x.gguf"));
    }

    #[test]
    fn the_plain_gemma_signature_is_unchanged() {
        // The value existing v2 indexes carry. A change here forces every
        // index to be rebuilt, so it has to be deliberate.
        if std::env::var_os("CKQ_GGUF_FILE").is_none() && embedded_alias() != Some("gemma-q4") {
            assert_eq!(signature(resolve("gemma-q4").unwrap()), "c23fdf7835ede659");
        }
    }
}
