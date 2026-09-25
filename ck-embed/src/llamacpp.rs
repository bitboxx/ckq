//! llama.cpp backend.
//!
//! The ONNX path reaches no GPU on a Mac, and it needed four hand-written fixes
//! to run a causal embedder at all: last-token pooling, position ids, an empty
//! KV cache per layer, and a hardcoded head shape. llama.cpp knows the
//! architectures, so all four are its problem rather than ours, and its Metal
//! kernels are the best-tuned on Apple Silicon.

use anyhow::{Context, Result, anyhow};
use ck_models::ModelConfig;
use llama_cpp_2::context::params::{LlamaContextParams, LlamaPoolingType};
use llama_cpp_2::llama_backend::LlamaBackend;
use llama_cpp_2::llama_batch::LlamaBatch;
use llama_cpp_2::model::params::LlamaModelParams;
use llama_cpp_2::model::{AddBos, LlamaModel};
use std::collections::HashMap;
use std::num::NonZeroU32;
use std::sync::mpsc::{Sender, channel};
use std::sync::{Arc, Mutex, OnceLock};
use std::thread;

use crate::mixedbread::download_assets;
use crate::{Embedder, ModelDownloadCallback};

/// Offload every layer. llama.cpp silently keeps on CPU whatever will not fit.
const GPU_LAYERS: u32 = 1000;
/// llama.cpp splits `n_ctx` evenly across `n_seq_max` KV slots, so these two are
/// one decision, not two: each sequence gets `DEFAULT_CTX / MAX_SEQ` tokens and a
/// chunk larger than that fails with `NoKvCacheSlot`. ck targets 1024-token
/// chunks, so 2048 per sequence leaves headroom.
const MAX_SEQ: usize = 4;
const PER_SEQ: u32 = 2048;
const DEFAULT_CTX: u32 = PER_SEQ * MAX_SEQ as u32;

pub struct LlamaCppEmbedder {
    /// `LlamaContext` is neither `Send` nor `Sync`, and ck embeds from rayon
    /// worker threads. So one dedicated thread owns the model and the context
    /// for the life of the process, and callers hand it work over a channel.
    /// Rebuilding the context per call was the alternative and it costs a
    /// 306 MiB allocation each time, measured at 59% of wall time in the kernel.
    tx: Sender<Job>,
    dim: usize,
    model_name: String,
}

type Job = (Vec<String>, Sender<Result<Vec<Vec<f32>>>>);

/// `LlamaBackend::init()` is once per process and loading a model twice would
/// double the VRAM, so workers are shared. ck builds one embedder to index and
/// another to embed the query; both land on the same worker.
///
/// Keyed by model, because a process may legitimately touch two of them and an
/// unkeyed cache would hand the second one the first one's weights.
type Workers = Mutex<HashMap<String, Arc<Result<(Sender<Job>, usize), String>>>>;
static WORKERS: OnceLock<Workers> = OnceLock::new();

impl LlamaCppEmbedder {
    pub fn new(
        config: &ModelConfig,
        progress_callback: Option<ModelDownloadCallback>,
        gguf_file: &str,
        pooling: LlamaPoolingType,
    ) -> Result<Self> {
        if let Some(cb) = progress_callback.as_ref() {
            cb(&format!("Downloading GGUF ({}) if needed...", config.name));
        }
        let (model_path, _) = download_assets(&config.name, gguf_file, gguf_file)?;
        if let Some(cb) = progress_callback.as_ref() {
            cb("Loading llama.cpp model (Metal)...");
        }

        let gguf = gguf_file.to_string();
        let declared = config.dimensions;
        let max_length = config.max_tokens.min(PER_SEQ as usize);
        let name = config.name.clone();

        // One worker per (model, file); `LlamaBackend::init()` inside it is
        // idempotent in llama.cpp once the first has run.
        let key = format!("{}::{}", config.name, gguf);
        let started = {
            let workers = WORKERS.get_or_init(|| Mutex::new(HashMap::new()));
            let mut guard = workers
                .lock()
                .map_err(|_| anyhow!("llama.cpp worker registry is poisoned"))?;
            Arc::clone(guard.entry(key).or_insert_with(|| {
                Arc::new(
                    start_worker(model_path, gguf, name, declared, max_length, pooling)
                        .map_err(|e| e.to_string()),
                )
            }))
        };
        let (tx, dim) = match started.as_ref() {
            Ok(pair) => (pair.0.clone(), pair.1),
            Err(e) => return Err(anyhow!("{e}")),
        };

        Ok(Self {
            tx,
            dim,
            model_name: config.name.clone(),
        })
    }
}

/// One job: tokenize, pack sequences into as few decodes as the context allows,
/// read the pooled vector back per sequence.
fn run(
    model: &LlamaModel,
    ctx: &mut llama_cpp_2::context::LlamaContext<'_>,
    texts: &[String],
    max_length: usize,
    dim: usize,
) -> Result<Vec<Vec<f32>>> {
    let mut encoded: Vec<Vec<_>> = Vec::with_capacity(texts.len());
    for text in texts {
        let mut tokens = model
            .str_to_token(text, AddBos::Always)
            .map_err(|e| anyhow!("tokenize: {e}"))?;
        // Truncate rather than fail: ck chunks upstream, so an over-long chunk
        // is a chunker bug and not a reason to lose the whole file.
        tokens.truncate(max_length);
        encoded.push(tokens);
    }

    let mut out: Vec<Vec<f32>> = Vec::with_capacity(texts.len());
    let mut start = 0usize;
    while start < encoded.len() {
        let mut end = start;
        let mut total = 0usize;
        while end < encoded.len() {
            let len = encoded[end].len().max(1);
            if end > start && (total + len > DEFAULT_CTX as usize || end - start >= MAX_SEQ) {
                break;
            }
            total += len;
            end += 1;
        }

        let mut batch = LlamaBatch::new(total.max(1), (end - start) as i32);
        for (slot, tokens) in encoded[start..end].iter().enumerate() {
            if !tokens.is_empty() {
                batch
                    .add_sequence(tokens, slot as i32, false)
                    .map_err(|e| anyhow!("batch: {e}"))?;
            }
        }

        ctx.clear_kv_cache();
        ctx.decode(&mut batch).map_err(|e| anyhow!("decode: {e}"))?;

        for (slot, tokens) in encoded[start..end].iter().enumerate() {
            if tokens.is_empty() {
                out.push(vec![0.0; dim]);
                continue;
            }
            let embedding = ctx
                .embeddings_seq_ith(slot as i32)
                .map_err(|e| anyhow!("embeddings: {e}"))?;
            out.push(normalize(embedding));
        }
        start = end;
    }
    Ok(out)
}

fn start_worker(
    model_path: std::path::PathBuf,
    _gguf: String,
    name: String,
    declared: usize,
    max_length: usize,
    pooling: LlamaPoolingType,
) -> Result<(Sender<Job>, usize)> {
    let (tx, rx) = channel::<Job>();
    let (ready_tx, ready_rx) = channel::<Result<usize>>();

    // Everything llama.cpp touches stays on this thread for the process lifetime.
    thread::Builder::new()
        .name("ckq-llamacpp".into())
        .spawn(move || {
            let built = (|| -> Result<(LlamaBackend, LlamaModel)> {
                let backend = LlamaBackend::init().map_err(|e| anyhow!("llama backend: {e}"))?;
                let params = LlamaModelParams::default().with_n_gpu_layers(GPU_LAYERS);
                let model = LlamaModel::load_from_file(&backend, &model_path, &params)
                    .with_context(|| format!("loading GGUF {}", model_path.display()))?;
                Ok((backend, model))
            })();
            let (backend, model) = match built {
                Ok(pair) => pair,
                Err(e) => {
                    let _ = ready_tx.send(Err(e));
                    return;
                }
            };

            let dim = model.n_embd() as usize;
            if dim != declared {
                let _ = ready_tx.send(Err(anyhow!(
                    "{name} reports {dim} dims but the registry declares {declared}"
                )));
                return;
            }

            let n_ctx = NonZeroU32::new(DEFAULT_CTX).expect("non-zero context");
            let ctx_params = LlamaContextParams::default()
                .with_n_ctx(Some(n_ctx))
                .with_n_threads_batch(num_cpus::get().max(1) as i32)
                .with_embeddings(true)
                // Without these the context defaults to one sequence and a small
                // batch, and adding more sequences aborts inside llama_decode.
                // Qwen and Granite happened to survive it; EmbeddingGemma did not.
                .with_n_seq_max(MAX_SEQ as u32)
                .with_n_batch(DEFAULT_CTX)
                .with_n_ubatch(DEFAULT_CTX)
                .with_pooling_type(pooling);
            let mut ctx = match model.new_context(&backend, ctx_params) {
                Ok(c) => c,
                Err(e) => {
                    let _ = ready_tx.send(Err(anyhow!("llama context: {e}")));
                    return;
                }
            };
            let _ = ready_tx.send(Ok(dim));

            while let Ok((texts, reply)) = rx.recv() {
                let _ = reply.send(run(&model, &mut ctx, &texts, max_length, dim));
            }
            // Leak the context and model rather than dropping them: teardown
            // races Metal's resource sets and trips a GGML_ASSERT on exit.
            std::mem::forget(ctx);
            std::mem::forget(model);
            std::mem::forget(backend);
        })
        .map_err(|e| anyhow!("spawning llama.cpp thread: {e}"))?;

    let dim = ready_rx
        .recv()
        .map_err(|_| anyhow!("llama.cpp thread died during startup"))??;
    Ok((tx, dim))
}

impl Embedder for LlamaCppEmbedder {
    fn id(&self) -> &'static str {
        "llamacpp"
    }

    fn dim(&self) -> usize {
        self.dim
    }

    fn model_name(&self) -> &str {
        &self.model_name
    }

    fn embed(&mut self, texts: &[String]) -> Result<Vec<Vec<f32>>> {
        if texts.is_empty() {
            return Ok(Vec::new());
        }
        let (reply_tx, reply_rx) = channel();
        self.tx
            .send((texts.to_vec(), reply_tx))
            .map_err(|_| anyhow!("llama.cpp thread is gone"))?;
        reply_rx
            .recv()
            .map_err(|_| anyhow!("llama.cpp thread dropped the job"))?
    }
}

fn normalize(v: &[f32]) -> Vec<f32> {
    let norm = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm > 0.0 {
        v.iter().map(|x| x / norm).collect()
    } else {
        v.to_vec()
    }
}
