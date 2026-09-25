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
use std::sync::Arc;
use std::sync::OnceLock;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::{Sender, channel};
use std::thread;

use parking_lot::Mutex;

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

/// Talks to the shared daemon so the model is loaded once per machine rather
/// than once per process. `LlamaCppEmbedder::new_in_process` is the daemon's own
/// path, and the escape hatch if the socket cannot be used.
#[cfg(unix)]
pub struct LlamaCppDaemonClient {
    socket: std::path::PathBuf,
    alias: String,
    dim: usize,
    model_name: String,
    pin: bool,
}

#[cfg(unix)]
impl LlamaCppDaemonClient {
    pub fn new(config: &ModelConfig, alias: &str, gguf: &str, dim: usize, pin: bool) -> Self {
        Self {
            socket: crate::embed_daemon::socket_path(&config.name, gguf),
            alias: alias.to_string(),
            dim,
            model_name: config.name.clone(),
            pin,
        }
    }
}

#[cfg(unix)]
impl Embedder for LlamaCppDaemonClient {
    fn id(&self) -> &'static str {
        "llamacpp-daemon"
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
        // A daemon can hit its idle timeout in the window between our connect
        // and our request. ck-index catches per-file errors, counts
        // files_errored and still reports success, so healing it here is the
        // difference between one respawn and files silently missing from the
        // index. One retry: connect (respawning if nothing answers), request.
        let mut last_err = None;
        for _ in 0..2 {
            let mut stream = match crate::embed_daemon::try_connect(&self.socket) {
                Some(s) => s,
                None => crate::embed_daemon::spawn(&self.alias, &self.socket)?,
            };
            match crate::embed_daemon::request(&mut stream, texts, self.pin) {
                Ok(embeddings) => return Ok(embeddings),
                Err(e) => last_err = Some(e),
            }
        }
        Err(last_err.unwrap_or_else(|| anyhow!("embed daemon request failed twice")))
    }
}

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
type Workers = Mutex<HashMap<String, Arc<(Sender<Job>, usize)>>>;
static WORKERS: OnceLock<Workers> = OnceLock::new();

impl LlamaCppEmbedder {
    pub fn new_in_process(
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
        let workers = WORKERS.get_or_init(|| Mutex::new(HashMap::new()));
        {
            // Fast path: this process already has the model loaded.
            let guard = workers.lock();
            if let Some(hit) = guard.get(&key) {
                return Ok(Self {
                    tx: hit.0.clone(),
                    dim: hit.1,
                    model_name: config.name.clone(),
                });
            }
        }
        // Load without holding the registry lock — a second model must not
        // block behind this one's download — and cache only successes: caching
        // the Err of one transient failure used to poison the model for the
        // whole process.
        let built = Arc::new(start_worker(
            model_path, gguf, name, declared, max_length, pooling,
        )?);
        let mut guard = workers.lock();
        let hit = Arc::clone(guard.entry(key).or_insert(built));
        // A concurrent winner inserted its own worker; ours sees its sender
        // dropped below and exits (its loaded model leaks with it, but the
        // race is a once-per-process event).
        Ok(Self {
            tx: hit.0.clone(),
            dim: hit.1,
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
    pooling: LlamaPoolingType,
) -> Result<Vec<Vec<f32>>> {
    let mut encoded: Vec<Vec<_>> = Vec::with_capacity(texts.len());
    let mut truncated = 0usize;
    for text in texts {
        let mut tokens = model
            .str_to_token(text, AddBos::Always)
            .map_err(|e| anyhow!("tokenize: {e}"))?;
        // Truncate rather than fail: ck chunks upstream, so an over-long chunk
        // is a chunker bug and not a reason to lose the whole file.
        if tokens.len() > max_length {
            truncated += 1;
            tokens.truncate(max_length);
        }
        encoded.push(tokens);
    }
    if truncated > 0 {
        warn_truncation(truncated, texts.len(), max_length, pooling);
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

/// Truncation must not be silent: for a last-token pooling model the dropped
/// tail is exactly the token that carries the sentence vector, so the result
/// is wrong rather than merely lossy. Once per run is enough — an index job
/// truncates in the thousands and one visible line says it all.
static TRUNCATION_WARNED: AtomicBool = AtomicBool::new(false);

fn warn_truncation(count: usize, total: usize, max_length: usize, pooling: LlamaPoolingType) {
    if TRUNCATION_WARNED.swap(true, Ordering::Relaxed) {
        return;
    }
    let gravity = if pooling == LlamaPoolingType::Last {
        "this model pools its LAST token, so truncation removes the sentence vector itself: the embedding is wrong, not just lossy"
    } else {
        // Pooling came from GGUF metadata, so it may still be last-token.
        "if this model pools its last token, the embedding is wrong, not just lossy"
    };
    eprintln!(
        "ckq: warning: truncated {count} of {total} inputs to {max_length} tokens before \
         embedding ({gravity}; further truncations are not logged)"
    );
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
                let _ = reply.send(run(&model, &mut ctx, &texts, max_length, dim, pooling));
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
