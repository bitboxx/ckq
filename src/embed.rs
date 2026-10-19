//! Text to vectors, with llama.cpp.
//!
//! The model's prompts are applied here and nowhere else: `embed_query` adds
//! the query prefix, `embed_documents` the document prefix. The old engine
//! applied them in its client and again in its daemon, so every query reached
//! the model with both.
//!
//! A search embeds its query through the query daemon when one can be
//! reached (see `daemon`), and so does the daemon itself; a search that
//! falls back to its own model goes through `LlamaEmbedder` below, as does
//! indexing. The GGUF is memory-mapped, so processes that each load the
//! model share the pages instead of a copy. Queries run on the device that
//! is fastest for one short text from a cold start, which may well be the
//! CPU (no Metal shader compilation); indexing runs on the GPU where there
//! is one. One context sized for indexing batches, a small one for a single
//! query, mean or last-token pooling, L2 normalisation. llama.cpp's own log
//! lines go to stderr or nowhere, never to stdout.
//!
//! `LlamaContext` borrows the model it decodes in and is neither `Send` nor
//! `Sync`, so one dedicated thread owns the model and both contexts for the
//! embedder's lifetime. That thread is also the only place where they are
//! dropped — contexts first, then the model, while the backend is still alive
//! — which is the ordering ggml requires at teardown. [`LlamaEmbedder`]
//! therefore holds nothing but a channel to it.

use crate::models::ModelSpec;
use anyhow::{Result, anyhow};
use std::num::NonZeroU32;
use std::sync::mpsc::{Sender, channel};
use std::sync::{LazyLock, Mutex};
use std::thread;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Device {
    /// Every layer on the GPU (Metal on macOS, CPU elsewhere unless built with a GPU backend).
    Gpu,
    Cpu,
}

/// For a search: `CKQ_QUERY_DEVICE=cpu|gpu` when set, else whichever measured
/// faster for a cold single query on this platform. Measured 17 Oct 2026 on
/// an M1 Ultra, gemma-q4, median of 10 cold `--sem` runs over bench/multilingual:
/// gpu 0.41 s, cpu 0.59 s — the GPU wins even cold (Metal shader compilation
/// included), so it is the default.
pub fn query_device() -> Device {
    match std::env::var("CKQ_QUERY_DEVICE").as_deref() {
        Ok("cpu") => Device::Cpu,
        Ok("gpu") => Device::Gpu,
        _ => Device::Gpu,
    }
}

pub trait Embedder {
    /// The model's vector width; carried for callers that need to size buffers.
    #[allow(dead_code)]
    fn dims(&self) -> usize;
    /// Unit vectors, one per text, the document prefix applied.
    fn embed_documents(&mut self, texts: &[&str]) -> Result<Vec<Vec<f32>>>;
    /// A unit vector for one query, the query prefix applied.
    fn embed_query(&mut self, text: &str) -> Result<Vec<f32>>;
}

/// One embedding job for the worker thread: the texts with their prefix
/// already applied, and whether they go through the small query context.
enum Job {
    Embed {
        texts: Vec<String>,
        query: bool,
        reply: Sender<Result<Vec<Vec<f32>>>>,
    },
}

pub struct LlamaEmbedder {
    spec: &'static ModelSpec,
    #[allow(dead_code)]
    dim: usize,
    tx: Sender<Job>,
}

/// Loading a model twice would double the memory, so workers are shared per
/// model within a process; only successful loads are cached. The handle joins
/// the thread at shutdown, which is what lets its drops run in order.
/// One live model: what identifies it in the cache — its alias and its
/// device, because a CPU worker and a GPU worker are two loaded models, and
/// handing back the wrong one silently discarded the caller's `device` —
/// the channel to its thread, its dimensionality and the thread's handle
/// (for the ordered teardown at shutdown).
type Worker = (
    (&'static str, Device),
    Sender<Job>,
    usize,
    thread::JoinHandle<()>,
);
static WORKERS: LazyLock<Mutex<Vec<Worker>>> = LazyLock::new(|| Mutex::new(Vec::new()));

/// The llama.cpp backend, initialised once per process: a second
/// `LlamaBackend::init()` is an error, so the worker thread borrows this one.
static BACKEND: LazyLock<Result<llama_cpp_2::llama_backend::LlamaBackend, String>> =
    LazyLock::new(|| {
        // Silence llama.cpp's load chatter entirely: it is neither results nor a
        // warning, and it must never contaminate stdout.
        llama_cpp_2::send_logs_to_tracing(
            llama_cpp_2::LogOptions::default().with_logs_enabled(false),
        );
        llama_cpp_2::llama_backend::LlamaBackend::init().map_err(|e| e.to_string())
    });

/// The llama.cpp backend for another in-process model (the reranker): one
/// `LlamaBackend::init()` per process is allowed, so every model borrows
/// this one.
pub fn backend() -> Result<&'static llama_cpp_2::llama_backend::LlamaBackend> {
    BACKEND.as_ref().map_err(|e| anyhow!("llama backend: {e}"))
}

/// Offload every layer; llama.cpp silently keeps on CPU whatever will not fit.
const GPU_LAYERS: u32 = 1000;
/// Sequences per decode in the indexing context. Measured against sixteen on
/// both models in the old engine: no difference, double the memory.
const MAX_SEQ: usize = 4;
/// Ceiling on one sequence. Chunks target 1,024 tokens, so 2,048 is headroom.
const PER_SEQ: u32 = 2048;
/// Context for a single short query: decoding in the indexing context costs
/// 150-200 ms even for ten tokens, because the encoder attends over the whole
/// KV cache size.
const QUERY_CTX: u32 = 512;

impl LlamaEmbedder {
    pub fn load(spec: &'static ModelSpec, device: Device) -> Result<Self> {
        let want = (spec.alias, device);
        let cached = WORKERS
            .lock()
            .expect("worker registry")
            .iter()
            .find(|(key, ..)| *key == want)
            .map(|(_, tx, dim, _)| (tx.clone(), *dim));
        if let Some((tx, dim)) = cached {
            return Ok(Self { spec, dim, tx });
        }
        // Load without holding the registry lock, and cache only successes.
        let (tx, dim, handle) = start_worker(spec, device)?;
        let mut workers = WORKERS.lock().expect("worker registry");
        let (tx, dim) = match workers.iter().find(|(key, ..)| *key == want) {
            // A concurrent load of the same model won the race; ours sees its
            // sender dropped below and its thread exits.
            Some(hit) => (hit.1.clone(), hit.2),
            None => {
                workers.push((want, tx, dim, handle));
                let last = workers.last().expect("just pushed");
                (last.1.clone(), last.2)
            }
        };
        Ok(Self { spec, dim, tx })
    }
}

/// Spawn the thread that owns the model and both contexts, and wait for it to
/// report the model's real dimensionality (or its load error).
fn start_worker(
    spec: &'static ModelSpec,
    device: Device,
) -> Result<(Sender<Job>, usize, thread::JoinHandle<()>)> {
    let (tx, rx) = channel::<Job>();
    let (ready_tx, ready_rx) = channel::<Result<usize>>();
    let model_path = crate::models::gguf_path(spec)?;

    let loader = move || {
        let run_all = (|| -> Result<()> {
            let backend = BACKEND
                .as_ref()
                .map_err(|e| anyhow!("llama backend: {e}"))?;
            let params = llama_cpp_2::model::params::LlamaModelParams::default().with_n_gpu_layers(
                match device {
                    Device::Gpu => GPU_LAYERS,
                    Device::Cpu => 0,
                },
            );
            let model =
                llama_cpp_2::model::LlamaModel::load_from_file(backend, &model_path, &params)
                    .map_err(|e| anyhow!("loading GGUF {}: {e}", model_path.display()))?;
            let dims = model.n_embd() as usize;
            if dims != spec.dims {
                anyhow::bail!(
                    "{} reports {dims} dims but the registry declares {}",
                    spec.repo,
                    spec.dims
                );
            }

            // llama.cpp splits `n_ctx` evenly across `n_seq_max` slots, so the
            // two are one decision: a chunk longer than `n_ctx / n_seq_max`
            // fails with `NoKvCacheSlot`.
            let max_length = spec.context.min(PER_SEQ as usize);
            let ctx_budget = max_length as u32 * MAX_SEQ as u32;
            let pooling = pooling_of(spec);
            let mut ctx = model
                .new_context(backend, context_params(ctx_budget, MAX_SEQ as u32, pooling))
                .map_err(|e| anyhow!("llama context: {e}"))?;
            let q_budget = (max_length as u32).clamp(1, QUERY_CTX);
            let mut query_ctx = model
                .new_context(backend, context_params(q_budget, 1, pooling))
                .map_err(|e| anyhow!("llama query context: {e}"))?;

            register_exit_hook();
            ready_tx
                .send(Ok(dims))
                .map_err(|_| anyhow!("caller left before the model loaded"))?;

            while let Ok(Job::Embed {
                texts,
                query,
                reply,
            }) = rx.recv()
            {
                let result = if query {
                    run(&model, &mut query_ctx, &texts, q_budget as usize, dims)
                } else {
                    run(&model, &mut ctx, &texts, max_length, dims)
                };
                let _ = reply.send(result);
            }
            Ok(())
        })();
        if let Err(e) = run_all {
            let _ = ready_tx.send(Err(e));
        }
        // Locals drop here in reverse declaration order: query context,
        // context, model — while the backend is still alive.
    };

    let handle = thread::Builder::new()
        .name("ckq-llamacpp".into())
        .spawn(loader)
        .map_err(|e| anyhow!("spawning llama.cpp thread: {e}"))?;

    let dim = ready_rx
        .recv()
        .map_err(|_| anyhow!("llama.cpp thread died during startup"))??;
    Ok((tx, dim, handle))
}

/// Stop every worker and wait for it to drop its contexts and its model in
/// order, on its own thread, while the backend is still alive. Without this,
/// ggml aborts from a static destructor once the process is already past its
/// last line. Call it once, last thing, before the process exits.
pub fn shutdown() {
    // Poison-tolerant on purpose: a test that panicked while holding the
    // registry must not turn every later teardown into an abort from the
    // exit hook — the workers still need joining either way.
    let recover = |poisoned: std::sync::PoisonError<_>| poisoned.into_inner();
    let mut workers = WORKERS.lock().unwrap_or_else(recover);
    let draining: Vec<thread::JoinHandle<()>> =
        workers.drain(..).map(|(_, _, _, handle)| handle).collect();
    for handle in draining {
        let _ = handle.join();
    }
}

/// Run [`shutdown`] when the process exits, however it exits: a caller that
/// forgets (a test binary, any program that links ckq) still gets the
/// ordered teardown. Registered after the model is loaded so it runs before
/// ggml's own destructor.
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

fn context_params(
    budget: u32,
    seqs: u32,
    pooling: llama_cpp_2::context::params::LlamaPoolingType,
) -> llama_cpp_2::context::params::LlamaContextParams {
    let n = NonZeroU32::new(budget).expect("non-zero context");
    llama_cpp_2::context::params::LlamaContextParams::default()
        .with_n_ctx(Some(n))
        .with_n_threads_batch(threads())
        .with_embeddings(true)
        // Without these the context defaults to one sequence and a small
        // batch, and adding more sequences aborts in decode.
        .with_n_seq_max(seqs)
        .with_n_batch(budget)
        .with_n_ubatch(budget)
        .with_pooling_type(pooling)
}

/// Thread count for a llama.cpp context: every core, as the embedder does.
pub(crate) fn threads() -> i32 {
    std::thread::available_parallelism()
        .map(|n| n.get() as i32)
        .unwrap_or(1)
}

fn pooling_of(spec: &ModelSpec) -> llama_cpp_2::context::params::LlamaPoolingType {
    use llama_cpp_2::context::params::LlamaPoolingType;
    match spec.pooling {
        crate::models::Pooling::Mean => LlamaPoolingType::Mean,
        crate::models::Pooling::Last => LlamaPoolingType::Last,
    }
}

impl Embedder for LlamaEmbedder {
    fn dims(&self) -> usize {
        self.dim
    }

    fn embed_documents(&mut self, texts: &[&str]) -> Result<Vec<Vec<f32>>> {
        if texts.is_empty() {
            return Ok(Vec::new());
        }
        let texts: Vec<String> = texts
            .iter()
            .map(|t| format!("{}{t}", self.spec.document_prefix))
            .collect();
        self.send(texts, false)
    }

    fn embed_query(&mut self, text: &str) -> Result<Vec<f32>> {
        let texts = vec![format!("{}{text}", self.spec.query_prefix)];
        self.send(texts, true)
            .map(|mut v| v.pop().expect("one vector per text"))
    }
}

impl LlamaEmbedder {
    fn send(&self, texts: Vec<String>, query: bool) -> Result<Vec<Vec<f32>>> {
        let (reply_tx, reply_rx) = channel();
        self.tx
            .send(Job::Embed {
                texts,
                query,
                reply: reply_tx,
            })
            .map_err(|_| anyhow!("llama.cpp thread is gone"))?;
        reply_rx
            .recv()
            .map_err(|_| anyhow!("llama.cpp thread dropped the job"))?
    }
}

/// One job: tokenize, pack sequences into as few decodes as the context
/// allows, read the pooled vector back per sequence. Ported from the old
/// engine: batching, truncation with one warning, L2 normalisation.
fn run(
    model: &llama_cpp_2::model::LlamaModel,
    ctx: &mut llama_cpp_2::context::LlamaContext<'_>,
    texts: &[String],
    max_length: usize,
    dim: usize,
) -> Result<Vec<Vec<f32>>> {
    let vocab = model.vocab();
    let mut encoded: Vec<Vec<_>> = Vec::with_capacity(texts.len());
    let mut truncated = 0usize;
    for text in texts {
        // BOS added and special tokens parsed, as `str_to_token(_, AddBos::Always)`
        // did before llama-cpp-2 0.1.159 replaced it: the same tokens, so
        // vectors in existing indexes stay comparable.
        let mut tokens = vocab.tokenize(text.as_bytes(), true, true);
        // Truncate rather than fail: ck chunks upstream, so an over-long chunk
        // is a chunker bug and not a reason to lose the whole file.
        if tokens.len() > max_length {
            truncated += 1;
            tokens.truncate(max_length);
        }
        encoded.push(tokens);
    }
    if truncated > 0 {
        eprintln!(
            "ckq: warning: truncated {truncated} of {} inputs to {max_length} tokens before \
             embedding (if this model pools its last token, the embedding is wrong, not just \
             lossy; further truncations are not logged)",
            texts.len()
        );
    }

    let budget = ctx.n_ctx() as usize;
    let max_seq = (budget / max_length.max(1)).clamp(1, MAX_SEQ);

    let mut out: Vec<Vec<f32>> = Vec::with_capacity(texts.len());
    let mut start = 0usize;
    while start < encoded.len() {
        let mut end = start;
        let mut total = 0usize;
        while end < encoded.len() {
            let len = encoded[end].len().max(1);
            if end > start && (total + len > budget || end - start >= max_seq) {
                break;
            }
            total += len;
            end += 1;
        }

        let mut batch =
            llama_cpp_2::llama_batch::LlamaBatch::new(total.max(1), (end - start) as i32);
        for (slot, tokens) in encoded[start..end].iter().enumerate() {
            if !tokens.is_empty() {
                // `true` marks every token as an output: a pooled embedding
                // needs that, and llama.cpp forces it anyway.
                batch
                    .add_sequence(tokens, slot as i32, true)
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

#[cfg(not(unix))]
fn register_exit_hook() {}

fn normalize(v: &[f32]) -> Vec<f32> {
    let norm = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm > 0.0 {
        v.iter().map(|x| x / norm).collect()
    } else {
        v.to_vec()
    }
}

#[cfg(test)]
mod device_tests {
    use super::*;

    /// The cache is keyed by alias and device: a second load with another
    /// device loads a second worker (the parameter used to be silently
    /// discarded), and a repeat load of the same pair reuses the first one.
    /// In-module, because the worker registry is private and no CLI path
    /// loads two devices in one process.
    #[test]
    #[ignore = "e2e: loads the default model twice; run with --include-ignored"]
    fn the_worker_cache_is_keyed_by_alias_and_device() {
        let spec = crate::models::resolve(crate::models::DEFAULT_ALIAS).unwrap();
        // Keyed on pairs, not on the registry's length: another test in
        // this binary (index::tests embeds a file) loads the model on its
        // own thread while this one runs.
        let keys = || -> Vec<(String, Device)> {
            WORKERS
                .lock()
                .expect("worker registry")
                .iter()
                .map(|((alias, device), ..)| (alias.to_string(), *device))
                .collect()
        };
        let mut cpu = LlamaEmbedder::load(spec, Device::Cpu).unwrap();
        let mut gpu = LlamaEmbedder::load(spec, Device::Gpu).unwrap();
        let after_two = keys();
        assert!(
            after_two.contains(&(spec.alias.to_string(), Device::Cpu)),
            "the CPU load left no worker: {after_two:?}"
        );
        assert!(
            after_two.contains(&(spec.alias.to_string(), Device::Gpu)),
            "a second device reused the first device's worker: {after_two:?}"
        );
        let mut again = LlamaEmbedder::load(spec, Device::Cpu).unwrap();
        assert_eq!(
            keys(),
            after_two,
            "a repeat (alias, device) must reuse its worker"
        );
        // All three handles embed: the two workers answer, and the reused
        // one is wired to a live channel.
        for embedder in [&mut again, &mut cpu, &mut gpu] {
            let v = embedder.embed_query("a query").unwrap();
            assert_eq!(v.len(), spec.dims);
        }
    }
}
