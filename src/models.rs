//! The embedding models ckq knows, and where their weights live.
//!
//! GGUF only, run by llama.cpp: the eight aliases, their Hugging Face repo
//! and file, dimensions, pooling, prompts and default thresholds.
//! `gemma-q4` is the default.

use anyhow::{Result, anyhow, bail};
use std::path::PathBuf;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Pooling {
    Mean,
    /// Last token, for decoder models such as Qwen3-Embedding.
    Last,
    /// First token, for encoders that pool the CLS token such as the
    /// ModernBERT Granite-Embedding models.
    Cls,
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
};

const HARRIER: ModelSpec = ModelSpec {
    alias: "harrier-gguf",
    repo: "mradermacher/harrier-oss-v1-0.6b-GGUF",
    file: "harrier-oss-v1-0.6b.Q8_0.gguf",
    dims: 1024,
    context: 32768,
    pooling: Pooling::Last,
    // harrier is a Qwen3-Embedding-style decoder: the web-search instruction
    // from the model card prefixes the query, and documents go in bare.
    query_prefix: "Instruct: Given a web search query, retrieve relevant passages that answer the query\nQuery: ",
    document_prefix: "",
    threshold: 0.40,
};

const GRANITE_R2_311M: ModelSpec = ModelSpec {
    alias: "granite-r2-311m",
    repo: "mykor/granite-embedding-311m-multilingual-r2-GGUF",
    file: "granite-embedding-311M-multilingual-r2-Q8_0.gguf",
    dims: 768,
    context: 32768,
    pooling: Pooling::Cls,
    // The R2 card encodes queries and passages with no prefixes at all.
    query_prefix: "",
    document_prefix: "",
    threshold: 0.80,
};

const GRANITE_R2_97M: ModelSpec = ModelSpec {
    alias: "granite-r2-97m",
    repo: "mykor/granite-embedding-97m-multilingual-r2-GGUF",
    file: "granite-embedding-97M-multilingual-r2-Q8_0.gguf",
    dims: 384,
    context: 32768,
    pooling: Pooling::Cls,
    query_prefix: "",
    document_prefix: "",
    threshold: 0.78,
};

/// The registry. `gemma-q4`, `gemma-gguf`, `granite-gguf`, `granite-r2-311m`,
/// `granite-r2-97m`, `bge-m3-gguf`, `qwen3-gguf`, `harrier-gguf`.
pub fn all() -> &'static [ModelSpec] {
    &[
        GEMMA_Q4,
        GEMMA_Q8,
        GRANITE,
        GRANITE_R2_311M,
        GRANITE_R2_97M,
        BGE_M3,
        QWEN3,
        HARRIER,
    ]
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
pub fn gguf_path(spec: &ModelSpec) -> Result<PathBuf> {
    if let Some(override_path) = std::env::var_os("CKQ_GGUF_FILE") {
        let path = PathBuf::from(override_path);
        if !path.is_file() {
            bail!("CKQ_GGUF_FILE does not point at a file: {}", path.display());
        }
        return Ok(path);
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
        Pooling::Cls => b"cls",
    });
    hasher.update(b"\0");
    hasher.update(&crate::chunk::TARGET_TOKENS.to_le_bytes());
    hasher.update(&crate::chunk::OVERLAP_TOKENS.to_le_bytes());
    hasher.finalize().to_hex()[..16].to_string()
}
