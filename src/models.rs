//! The embedding models ckq knows, and where their weights live.
//!
//! GGUF only, run by llama.cpp. Ported from `ck-models/src/lib.rs`: the five
//! aliases, their Hugging Face repo and file, dimensions, pooling, prompts and
//! default thresholds. `gemma-q4` is the default.

use anyhow::Result;
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
}

pub const DEFAULT_ALIAS: &str = "gemma-q4";

/// The registry. `gemma-q4`, `gemma-gguf`, `granite-gguf`, `bge-m3-gguf`, `qwen3-gguf`.
pub fn all() -> &'static [ModelSpec] {
    todo!("port the five llamacpp entries from ck-models/src/lib.rs, values unchanged")
}

/// An alias, or a repo name as stored in an index, to its spec.
pub fn resolve(name: &str) -> Result<&'static ModelSpec> {
    todo!()
}

/// The GGUF on disk: `CKQ_GGUF_FILE` when set (for trying another build of the
/// same model), else the Hugging Face cache under `~/.cache/ck/models`,
/// downloading on first use. Same cache layout as the old engine, so nothing
/// already downloaded is fetched again.
pub fn gguf_path(spec: &ModelSpec) -> Result<PathBuf> {
    todo!("hf-hub ApiBuilder with cache dir ~/.cache/ck/models")
}

/// What has to match for two vectors to be comparable: the model, the
/// prompts, the pooling and the chunker version. Stored in the index; a
/// mismatch refuses `--index` and names `--clean`.
pub fn signature(spec: &ModelSpec) -> String {
    todo!("blake3 of repo, file, prefixes, pooling and crate::chunk sizes; first 16 hex chars")
}
