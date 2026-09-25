use anyhow::{Result, bail};
use ck_models::{ModelConfig, ModelRegistry};
#[cfg(feature = "fastembed")]
use std::path::Path;
#[cfg(any(feature = "fastembed", feature = "mixedbread"))]
use std::path::PathBuf;
use std::sync::OnceLock;

pub mod reranker;
pub mod tokenizer;

pub use reranker::{
    RerankResult, Reranker, create_reranker, create_reranker_for_config,
    create_reranker_with_progress,
};
pub use tokenizer::TokenEstimator;

// Unix sockets only. Windows has AF_UNIX since 1803 but std does not expose it,
// so there the model is loaded per process instead of shared.
#[cfg(all(feature = "llamacpp", unix))]
pub mod embed_daemon;
#[cfg(feature = "llamacpp")]
mod llamacpp;

/// Release any model this process loaded itself, before it exits.
///
/// A ckq that talks to the shared daemon holds no model and this does nothing.
/// One that loaded in-process does, and on macOS the GPU resources have to be
/// given back on the thread that took them or ggml aborts the process from a
/// static destructor, after the output has already been printed. Call it last.
#[cfg(feature = "llamacpp")]
pub fn shutdown_embedders() {
    llamacpp::shutdown();
}

/// No in-process llama.cpp model is possible in this build, so nothing to do.
#[cfg(not(feature = "llamacpp"))]
pub fn shutdown_embedders() {}

#[cfg(feature = "mixedbread")]
mod mixedbread;
#[cfg(feature = "mixedbread")]
use mixedbread::MixedbreadEmbedder;

pub trait Embedder: Send + Sync {
    fn id(&self) -> &'static str;
    fn dim(&self) -> usize;
    fn model_name(&self) -> &str;
    fn embed(&mut self, texts: &[String]) -> Result<Vec<Vec<f32>>>;
}

pub type ModelDownloadCallback = Box<dyn Fn(&str) + Send + Sync>;

/// Embedding configuration that used to travel through environment variables.
///
/// `std::env::set_var` racing a concurrent `getenv` on another thread is
/// undefined, so nothing mutates the environment after startup: the CLI reads
/// these once before its worker threads exist and passes the value down, and
/// `EmbedderOptions::process()` serves the library call sites that are too
/// deep to thread a parameter through. The `CKQ_*` variables stay readable so
/// users' existing invocations keep working.
#[derive(Debug, Clone, Default)]
pub struct EmbedderOptions {
    /// `CKQ_GGUF_FILE`: which GGUF to download and load for llama.cpp models.
    pub gguf_file: Option<String>,
    /// `CKQ_MODEL_ALIAS`: the alias clients re-exec the daemon with.
    pub model_alias: Option<String>,
    /// `CKQ_PIN` / `--serve`: ask the daemon to outlive the idle timeout.
    pub pin: bool,
    /// `CKQ_IN_PROCESS`: load the model here rather than via the daemon.
    pub in_process: bool,
}

impl EmbedderOptions {
    /// Snapshot the `CKQ_*` environment. Call once at process start, before
    /// any worker threads exist.
    pub fn from_env() -> Self {
        Self {
            gguf_file: std::env::var("CKQ_GGUF_FILE")
                .ok()
                .filter(|v| !v.is_empty()),
            model_alias: std::env::var("CKQ_MODEL_ALIAS")
                .ok()
                .filter(|v| !v.is_empty()),
            pin: std::env::var("CKQ_PIN").is_ok(),
            in_process: std::env::var("CKQ_IN_PROCESS").is_ok(),
        }
    }

    /// The process-wide snapshot installed by [`init_process_options`], or an
    /// empty override set when nobody installed one (library and test use).
    pub fn process() -> Self {
        PROCESS_OPTIONS.get().cloned().unwrap_or_default()
    }

    /// The GGUF file for a model. Both the client (it hashes this into the
    /// socket path) and the daemon (it binds that path) must derive the file
    /// the same way, or a `CKQ_GGUF_FILE` override changes only one side's
    /// socket and every request then times out.
    pub fn gguf_file_for(&self, config: &ModelConfig) -> String {
        self.gguf_file
            .clone()
            .unwrap_or_else(|| config.gguf_file.clone())
    }
}

static PROCESS_OPTIONS: OnceLock<EmbedderOptions> = OnceLock::new();

/// Install the process-wide [`EmbedderOptions`] snapshot. Write-once, before
/// worker threads start; later calls are ignored.
pub fn init_process_options(options: EmbedderOptions) {
    let _ = PROCESS_OPTIONS.set(options);
}

#[cfg(any(feature = "fastembed", feature = "mixedbread"))]
pub(crate) fn model_cache_root() -> Result<PathBuf> {
    let base = if let Some(cache_home) = std::env::var_os("XDG_CACHE_HOME") {
        PathBuf::from(cache_home).join("ck")
    } else if let Some(home) = std::env::var_os("HOME") {
        PathBuf::from(home).join(".cache").join("ck")
    } else if let Some(appdata) = std::env::var_os("LOCALAPPDATA") {
        PathBuf::from(appdata).join("ck").join("cache")
    } else {
        PathBuf::from(".ck_models")
    };

    Ok(base.join("models"))
}

pub fn create_embedder(model_name: Option<&str>) -> Result<Box<dyn Embedder>> {
    create_embedder_with_progress(model_name, None)
}

pub fn create_embedder_with_progress(
    model_name: Option<&str>,
    progress_callback: Option<ModelDownloadCallback>,
) -> Result<Box<dyn Embedder>> {
    let registry = ModelRegistry::default();
    let (_, config) = registry.resolve(model_name)?;
    create_embedder_for_config(&config, &EmbedderOptions::process(), progress_callback)
}

// `options` is only read on the llamacpp path; an empty feature set leaves it
// unused rather than branching the signature per feature.
#[cfg_attr(not(feature = "llamacpp"), allow(unused_variables))]
#[allow(clippy::needless_return)]
pub fn create_embedder_for_config(
    config: &ModelConfig,
    options: &EmbedderOptions,
    progress_callback: Option<ModelDownloadCallback>,
) -> Result<Box<dyn Embedder>> {
    match config.provider.as_str() {
        "fastembed" => {
            #[cfg(feature = "fastembed")]
            {
                return Ok(Box::new(FastEmbedder::new_with_progress(
                    config.name.as_str(),
                    progress_callback,
                )?));
            }

            #[cfg(not(feature = "fastembed"))]
            {
                if let Some(callback) = progress_callback.as_ref() {
                    callback("fastembed provider unavailable; using dummy embedder");
                }
                return Ok(Box::new(DummyEmbedder::new_with_model(
                    config.name.as_str(),
                )));
            }
        }
        #[cfg(feature = "llamacpp")]
        "llamacpp" => {
            use llama_cpp_2::context::params::LlamaPoolingType;
            // Unspecified means llama.cpp reads the pooling type out of the GGUF
            // metadata. Qwen is causal and wants last-token; Granite and
            // EmbeddingGemma are encoders and want CLS or mean. Hardcoding one
            // would silently corrupt the others, so let the model declare it.
            let gguf = options.gguf_file_for(config);
            // The daemon's own path (it forces this on), and the escape hatch
            // if the socket cannot be used.
            if options.in_process {
                let embedder = llamacpp::LlamaCppEmbedder::new_in_process(
                    config,
                    progress_callback,
                    &gguf,
                    LlamaPoolingType::Unspecified,
                )?;
                return Ok(Box::new(embedder));
            }
            // Otherwise talk to the shared daemon: one model per machine, not
            // one per process. `pin` keeps it alive past the idle timeout for
            // as long as this client holds its connection, which is what
            // `--serve` wants.
            // Not on Windows: the daemon speaks over a unix socket. There each
            // process loads its own copy, which is how ckq worked before the
            // daemon existed. Correct, only heavier when several run at once.
            #[cfg(not(unix))]
            {
                let embedder = llamacpp::LlamaCppEmbedder::new_in_process(
                    config,
                    progress_callback,
                    &gguf,
                    LlamaPoolingType::Unspecified,
                )?;
                return Ok(Box::new(embedder));
            }

            #[cfg(unix)]
            {
                let alias = options
                    .model_alias
                    .clone()
                    .unwrap_or_else(|| config.name.clone());
                let socket = embed_daemon::socket_path(&config.name, &gguf);
                if embed_daemon::reachable(&alias, &socket) {
                    return Ok(Box::new(llamacpp::LlamaCppDaemonClient::new(
                        config,
                        &alias,
                        &gguf,
                        config.dimensions,
                        options.pin,
                    )));
                }
                // The daemon could not be started, which is normal when the
                // running executable is not ckq. Carry on in-process.
                let embedder = llamacpp::LlamaCppEmbedder::new_in_process(
                    config,
                    progress_callback,
                    &gguf,
                    LlamaPoolingType::Unspecified,
                )?;
                Ok(Box::new(embedder))
            }
        }
        #[cfg(feature = "mixedbread")]
        "qwen" => {
            // Same ONNX path as mixedbread, but Qwen3-Embedding is causal: the
            // sentence vector sits at the last real token, not the first.
            let embedder = mixedbread::MixedbreadEmbedder::new_with_pooling(
                config,
                progress_callback,
                mixedbread::Pooling::LastToken,
            )?;
            Ok(Box::new(embedder))
        }
        #[cfg(not(feature = "mixedbread"))]
        "qwen" => {
            bail!(
                "Model '{}' requires the `mixedbread` feature. Rebuild ck with Mixedbread support.",
                config.name
            );
        }
        "mixedbread" => {
            #[cfg(feature = "mixedbread")]
            {
                return Ok(Box::new(MixedbreadEmbedder::new(
                    config,
                    progress_callback,
                )?));
            }
            #[cfg(not(feature = "mixedbread"))]
            {
                bail!(
                    "Model '{}' requires the `mixedbread` feature. Rebuild ck with Mixedbread support.",
                    config.name
                );
            }
        }
        provider => bail!("Unsupported embedding provider '{provider}'"),
    }
}

pub struct DummyEmbedder {
    dim: usize,
    model_name: String,
}

impl Default for DummyEmbedder {
    fn default() -> Self {
        Self::new()
    }
}

impl DummyEmbedder {
    pub fn new() -> Self {
        Self {
            dim: 384, // Match default BGE model
            model_name: "dummy".to_string(),
        }
    }

    pub fn new_with_model(model_name: &str) -> Self {
        Self {
            dim: 384, // Match default BGE model
            model_name: model_name.to_string(),
        }
    }
}

impl Embedder for DummyEmbedder {
    fn id(&self) -> &'static str {
        "dummy"
    }

    fn dim(&self) -> usize {
        self.dim
    }

    fn model_name(&self) -> &str {
        &self.model_name
    }

    fn embed(&mut self, texts: &[String]) -> Result<Vec<Vec<f32>>> {
        Ok(texts.iter().map(|_| vec![0.0; self.dim]).collect())
    }
}

#[cfg(feature = "fastembed")]
pub struct FastEmbedder {
    model: fastembed::TextEmbedding,
    dim: usize,
    model_name: String,
}

#[cfg(feature = "fastembed")]
impl FastEmbedder {
    pub fn new(model_name: &str) -> Result<Self> {
        Self::new_with_progress(model_name, None)
    }

    pub fn new_with_progress(
        model_name: &str,
        progress_callback: Option<ModelDownloadCallback>,
    ) -> Result<Self> {
        use fastembed::{EmbeddingModel, InitOptions, TextEmbedding};

        let model = match model_name {
            // Current models
            "BAAI/bge-small-en-v1.5" => EmbeddingModel::BGESmallENV15,
            "sentence-transformers/all-MiniLM-L6-v2" => EmbeddingModel::AllMiniLML6V2,

            // Enhanced models with longer context
            "nomic-embed-text-v1" => EmbeddingModel::NomicEmbedTextV1,
            "nomic-embed-text-v1.5" => EmbeddingModel::NomicEmbedTextV15,
            "jina-embeddings-v2-base-code" => EmbeddingModel::JinaEmbeddingsV2BaseCode,

            // BGE variants
            "BAAI/bge-base-en-v1.5" => EmbeddingModel::BGEBaseENV15,
            "BAAI/bge-large-en-v1.5" => EmbeddingModel::BGELargeENV15,

            // Multilingual models
            "BAAI/bge-m3" => EmbeddingModel::BGEM3,
            "Xenova/paraphrase-multilingual-MiniLM-L12-v2" => {
                EmbeddingModel::ParaphraseMLMiniLML12V2
            }
            "Xenova/paraphrase-multilingual-mpnet-base-v2" => {
                EmbeddingModel::ParaphraseMLMpnetBaseV2
            }

            // Default to Nomic v1.5 for better performance
            _ => EmbeddingModel::NomicEmbedTextV15,
        };

        // Configure permanent model cache directory
        let model_cache_dir = model_cache_root()?;
        std::fs::create_dir_all(&model_cache_dir)?;

        if let Some(ref callback) = progress_callback {
            callback(&format!("Initializing model: {model_name}"));

            // Check if model already exists
            let model_exists = Self::check_model_exists(&model_cache_dir, model_name);
            if !model_exists {
                callback(&format!(
                    "Downloading model {} to {}",
                    model_name,
                    model_cache_dir.display()
                ));
            } else {
                callback(&format!("Using cached model: {model_name}"));
            }
        }

        // Configure max_length based on model capacity
        let max_length = match model {
            // Small models - keep at 512
            EmbeddingModel::BGESmallENV15 | EmbeddingModel::AllMiniLML6V2 => 512,
            EmbeddingModel::BGEBaseENV15 => 512,

            // Large context models - use their full capacity!
            EmbeddingModel::NomicEmbedTextV1 | EmbeddingModel::NomicEmbedTextV15 => 8192,
            EmbeddingModel::JinaEmbeddingsV2BaseCode => 8192,

            // BGE large can handle more
            EmbeddingModel::BGELargeENV15 => 512, // Conservative for BGE

            // Multilingual
            EmbeddingModel::BGEM3 => 8192,
            EmbeddingModel::ParaphraseMLMiniLML12V2 | EmbeddingModel::ParaphraseMLMpnetBaseV2 => {
                512
            }

            _ => 512, // Safe default
        };

        let init_options = InitOptions::new(model.clone())
            .with_show_download_progress(progress_callback.is_some())
            .with_cache_dir(model_cache_dir)
            .with_max_length(max_length);

        let embedding = TextEmbedding::try_new(init_options)?;

        if let Some(ref callback) = progress_callback {
            callback("Model loaded successfully");
        }

        let dim = match model {
            // Small models (384 dimensions)
            EmbeddingModel::BGESmallENV15 => 384,
            EmbeddingModel::AllMiniLML6V2 => 384,

            // Large context models (768 dimensions)
            EmbeddingModel::NomicEmbedTextV1 => 768,
            EmbeddingModel::NomicEmbedTextV15 => 768,
            EmbeddingModel::JinaEmbeddingsV2BaseCode => 768,
            EmbeddingModel::BGEBaseENV15 => 768,

            // Large models (1024 dimensions)
            EmbeddingModel::BGELargeENV15 => 1024,
            EmbeddingModel::BGEM3 => 1024,

            // Multilingual
            EmbeddingModel::ParaphraseMLMiniLML12V2 => 384,
            EmbeddingModel::ParaphraseMLMpnetBaseV2 => 768,

            _ => 384, // Default to 384 for BGE default
        };

        Ok(Self {
            model: embedding,
            dim,
            model_name: model_name.to_string(),
        })
    }

    fn check_model_exists(cache_dir: &Path, model_name: &str) -> bool {
        // Simple heuristic - check if model directory exists
        let model_dir = cache_dir.join(model_name.replace("/", "_"));
        model_dir.exists()
    }
}

#[cfg(feature = "fastembed")]
impl Embedder for FastEmbedder {
    fn id(&self) -> &'static str {
        "fastembed"
    }

    fn dim(&self) -> usize {
        self.dim
    }

    fn model_name(&self) -> &str {
        &self.model_name
    }

    fn embed(&mut self, texts: &[String]) -> Result<Vec<Vec<f32>>> {
        let text_refs: Vec<&str> = texts.iter().map(std::string::String::as_str).collect();
        let embeddings = self.model.embed(text_refs, None)?;
        Ok(embeddings)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dummy_embedder() {
        let mut embedder = DummyEmbedder::new();

        assert_eq!(embedder.id(), "dummy");
        assert_eq!(embedder.dim(), 384);

        let texts = vec!["hello".to_string(), "world".to_string()];
        let embeddings = embedder.embed(&texts).unwrap();

        assert_eq!(embeddings.len(), 2);
        assert_eq!(embeddings[0].len(), 384);
        assert_eq!(embeddings[1].len(), 384);

        // Dummy embedder should return all zeros
        assert!(embeddings[0].iter().all(|&x| x == 0.0));
        assert!(embeddings[1].iter().all(|&x| x == 0.0));
    }

    #[test]
    fn test_create_embedder_dummy() {
        #[cfg(not(feature = "fastembed"))]
        {
            let embedder = create_embedder(None).unwrap();
            assert_eq!(embedder.id(), "dummy");
            assert_eq!(embedder.dim(), 384);
        }
    }

    #[test]
    fn test_embedder_trait_object() {
        let mut embedder: Box<dyn Embedder> = Box::new(DummyEmbedder::new());

        let texts = vec!["test".to_string()];
        let result = embedder.embed(&texts);
        assert!(result.is_ok());

        let embeddings = result.unwrap();
        assert_eq!(embeddings.len(), 1);
        assert_eq!(embeddings[0].len(), 384);
    }

    #[cfg(feature = "fastembed")]
    #[test]
    fn test_fastembed_creation() {
        // This test requires downloading models, so we'll skip it in CI
        if std::env::var("CI").is_ok() {
            return;
        }

        let embedder = FastEmbedder::new("BAAI/bge-small-en-v1.5");

        // FastEmbed creation might fail due to network issues or missing models
        // In a real test environment, you'd want to ensure models are available
        match embedder {
            Ok(mut embedder) => {
                assert_eq!(embedder.id(), "fastembed");
                assert_eq!(embedder.dim(), 384);

                let texts = vec!["hello world".to_string()];
                let result = embedder.embed(&texts);
                assert!(result.is_ok());

                let embeddings = result.unwrap();
                assert_eq!(embeddings.len(), 1);
                assert_eq!(embeddings[0].len(), 384);

                // Real embeddings should not be all zeros
                assert!(!embeddings[0].iter().all(|&x| x == 0.0));
            }
            Err(_) => {
                // In test environments, FastEmbed might not be available
                // This is acceptable for unit tests
            }
        }
    }

    #[cfg(feature = "fastembed")]
    #[test]
    fn test_create_embedder_fastembed() {
        if std::env::var("CI").is_ok() {
            return;
        }

        let embedder = create_embedder(Some("BAAI/bge-small-en-v1.5"));

        match embedder {
            Ok(embedder) => {
                assert_eq!(embedder.id(), "fastembed");
                assert_eq!(embedder.dim(), 384);
            }
            Err(_) => {
                // Model might not be available in test environment
            }
        }
    }

    #[test]
    fn test_embedder_empty_input() {
        let mut embedder = DummyEmbedder::new();
        let texts: Vec<String> = vec![];
        let embeddings = embedder.embed(&texts).unwrap();
        assert_eq!(embeddings.len(), 0);
    }

    #[test]
    fn test_embedder_single_text() {
        let mut embedder = DummyEmbedder::new();
        let texts = vec!["single text".to_string()];
        let embeddings = embedder.embed(&texts).unwrap();

        assert_eq!(embeddings.len(), 1);
        assert_eq!(embeddings[0].len(), 384);
    }

    #[test]
    fn test_embedder_multiple_texts() {
        let mut embedder = DummyEmbedder::new();
        let texts = vec![
            "first text".to_string(),
            "second text".to_string(),
            "third text".to_string(),
        ];
        let embeddings = embedder.embed(&texts).unwrap();

        assert_eq!(embeddings.len(), 3);
        for embedding in &embeddings {
            assert_eq!(embedding.len(), 384);
        }
    }
}
