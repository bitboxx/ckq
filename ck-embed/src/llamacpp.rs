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
use std::time::Duration;

use parking_lot::Mutex;

use crate::mixedbread::download_assets;
use crate::{Embedder, ModelDownloadCallback};

/// Offload every layer. llama.cpp silently keeps on CPU whatever will not fit.
const GPU_LAYERS: u32 = 1000;
/// How many sequences go into one decode. Four was measured against sixteen on
/// both models and the difference was nothing, while sixteen slots doubled the
/// memory, so this stays small.
const MAX_SEQ: usize = 4;
/// Ceiling on one sequence, for a model whose own context is longer than
/// anything ck chunks to. Chunks target 1024 tokens, so 2048 leaves headroom.
const PER_SEQ: u32 = 2048;

/// The context a worker asks for: one slot per sequence, each as long as the
/// model's own limit.
///
/// llama.cpp splits `n_ctx` evenly across `n_seq_max` slots, so the two are one
/// decision: a chunk longer than `n_ctx / n_seq_max` fails with
/// `NoKvCacheSlot`. Deriving it from the model rather than fixing it at
/// `PER_SEQ * MAX_SEQ` matters for a short-context model. Granite trains at 512
/// tokens, so a fixed 2048 per slot asked for four times the KV cache it can
/// use and made llama.cpp warn about a training context overflow on every load.
fn ctx_tokens(max_length: usize) -> u32 {
    (max_length as u32).clamp(1, PER_SEQ) * MAX_SEQ as u32
}

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

enum Job {
    Embed(Vec<String>, Sender<Result<Vec<Vec<f32>>>>),
    /// Drop the context and the model on the worker's own thread, then stop.
    ///
    /// Nothing else can do it: `LlamaContext` is not `Send`, so the thread that
    /// built it is the only one allowed to destroy it. See `shutdown`.
    Quit(Sender<()>),
}

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
            if end > start
                && (total + len > ctx_tokens(max_length) as usize || end - start >= MAX_SEQ)
            {
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

/// Stop every in-process model and wait for it to let go of the GPU.
///
/// Call it once, last thing, before the process exits. It is a no-op when no
/// model was loaded here, which is the normal case: the CLI talks to the shared
/// daemon and never holds one.
pub fn shutdown() {
    let Some(workers) = WORKERS.get() else {
        return;
    };
    let draining: Vec<_> = workers.lock().drain().map(|(_, w)| w).collect();
    for worker in draining {
        let (done_tx, done_rx) = channel();
        if worker.0.send(Job::Quit(done_tx)).is_ok() {
            // Bounded: a worker wedged in a decode must not hang the exit.
            let _ = done_rx.recv_timeout(Duration::from_secs(10));
        }
    }
}

/// Run `shutdown` when the process exits, however it exits.
///
/// The explicit call in `main` covers the CLI. This covers everybody else: a
/// test binary, or any program that links ck-embed, has no such place to put
/// it, and without it ggml aborts from a static destructor once the process is
/// already past its last line. Registering after the model is loaded is what
/// makes this work: exit handlers run in reverse order of registration, so this
/// one runs before ggml tears its device down.
#[cfg(unix)]
fn register_exit_hook() {
    static ONCE: std::sync::Once = std::sync::Once::new();
    extern "C" fn on_exit() {
        shutdown();
    }
    ONCE.call_once(|| unsafe {
        libc::atexit(on_exit);
    });
}

#[cfg(not(unix))]
fn register_exit_hook() {}

/// The llama.cpp backend, initialised once for the whole process.
///
/// `LlamaBackend::init()` returns `BackendAlreadyInitialized` on the second
/// call. Each worker used to init its own, so a process that loaded two models,
/// or a test binary that ran two searches, failed on the second one with an
/// error that named the backend rather than the cause. It is a zero-sized
/// guard, so a static costs nothing and every worker borrows the same one.
static BACKEND: OnceLock<LlamaBackend> = OnceLock::new();
static BACKEND_INIT: Mutex<()> = Mutex::new(());

fn backend() -> Result<&'static LlamaBackend> {
    if let Some(backend) = BACKEND.get() {
        return Ok(backend);
    }
    // `OnceLock::get_or_init` cannot carry a failure, so hold a lock across the
    // fallible part: without it two threads race and the loser sees exactly the
    // BackendAlreadyInitialized this function exists to prevent.
    let _guard = BACKEND_INIT.lock();
    if let Some(backend) = BACKEND.get() {
        return Ok(backend);
    }
    // llama.cpp writes its load and teardown chatter straight to stderr. That was
    // invisible while only the daemon ever loaded a model, and it landed on the
    // user's terminal the moment `--index` started loading one itself. Routing
    // it into `tracing` puts it behind RUST_LOG, where the rest of ck's logging
    // already lives.
    llama_cpp_2::send_logs_to_tracing(llama_cpp_2::LogOptions::default().with_logs_enabled(true));
    let created = LlamaBackend::init().map_err(|e| anyhow!("llama backend: {e}"))?;
    Ok(BACKEND.get_or_init(|| created))
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
            let built = (|| -> Result<(&'static LlamaBackend, LlamaModel)> {
                let backend = backend()?;
                let params = LlamaModelParams::default().with_n_gpu_layers(GPU_LAYERS);
                let model = LlamaModel::load_from_file(backend, &model_path, &params)
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

            let ctx_budget = ctx_tokens(max_length);
            let n_ctx = NonZeroU32::new(ctx_budget).expect("non-zero context");
            let ctx_params = LlamaContextParams::default()
                .with_n_ctx(Some(n_ctx))
                .with_n_threads_batch(num_cpus::get().max(1) as i32)
                .with_embeddings(true)
                // Without these the context defaults to one sequence and a small
                // batch, and adding more sequences aborts inside llama_decode.
                // Qwen and Granite happened to survive it; EmbeddingGemma did not.
                .with_n_seq_max(MAX_SEQ as u32)
                .with_n_batch(ctx_budget)
                .with_n_ubatch(ctx_budget)
                .with_pooling_type(pooling);
            let mut ctx = match model.new_context(backend, ctx_params) {
                Ok(c) => c,
                Err(e) => {
                    let _ = ready_tx.send(Err(anyhow!("llama context: {e}")));
                    return;
                }
            };
            register_exit_hook();
            let _ = ready_tx.send(Ok(dim));

            let mut stopping = None;
            while let Ok(job) = rx.recv() {
                match job {
                    Job::Embed(texts, reply) => {
                        let _ = reply.send(run(&model, &mut ctx, &texts, max_length, dim, pooling));
                    }
                    Job::Quit(done) => {
                        stopping = Some(done);
                        break;
                    }
                }
            }
            // Order matters: the context holds the Metal resource sets and the
            // model holds the buffers they point at. Dropping them here, on the
            // thread that made them and while the backend is still alive,
            // leaves ggml's device with nothing outstanding. Skipping it is
            // what used to abort the process at exit, in a static destructor
            // asserting that the resource set count was zero.
            drop(ctx);
            drop(model);
            if let Some(done) = stopping {
                let _ = done.send(());
            }
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
            .send(Job::Embed(texts.to_vec(), reply_tx))
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
