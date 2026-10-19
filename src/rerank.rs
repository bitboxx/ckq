//! The experimental cross-encoder reranker (`--rerank`, hybrid only).
//!
//! A second model reads the query together with each of the top fused
//! results' passages and scores the pair, and the list is reordered by those
//! scores (`pure`) or re-fused with the reranker's rank as a third list
//! (`fuse`). Off by default: without `--rerank` nothing here runs and
//! results are byte-identical.
//!
//! Input format, exactly what the llama.cpp server feeds a rerank endpoint
//! (`format_prompt_rerank` in `tools/server/server-common.cpp`): a model
//! whose GGUF carries a `rerank` chat template gets that template with
//! `{query}` and `{document}` substituted (tokenised with special tokens
//! parsed, no automatic BOS/EOS); every other model gets the vocabulary's
//! own flags — BOS, query, EOS, SEP, document, EOS — with EOS falling back
//! to SEP when the vocabulary has none. The score is the sequence's pooled
//! output's first element under rank pooling (`LlamaPoolingType::Rank`),
//! unnormalised: llama.cpp's server reads exactly `embeddings_seq[seq][0]`.
//! bge-reranker-v2-m3 (XLM-R, add_bos and add_eos, no SEP) therefore reads
//! `<s> query </s> document </s>`. A wrong format produces plausible but
//! meaningless scores, so `tests/e2e_rerank.rs` pins that an obviously
//! relevant passage outscores an obviously irrelevant one in two languages.
//!
//! Like the embedder, the model and its context live on one dedicated thread
//! for the ggml teardown ordering; the reranker loads in the searching
//! process, never in the daemon (the daemon embeds queries and nothing
//! else). `CKQ_TIMING=1` prints the load time and each search's scoring
//! time on stderr, so the warm cost can be read apart from the load.

use crate::models;
use anyhow::{Result, anyhow};
use std::sync::mpsc::{Sender, channel};
use std::thread::{self, JoinHandle};
use std::time::Instant;

/// A reranker the registry knows: GGUF, run with rank pooling, downloaded
/// through the same Hugging Face cache as the embedders.
#[derive(Debug, Clone)]
pub struct RerankerSpec {
    pub alias: &'static str,
    /// Hugging Face repository.
    pub repo: &'static str,
    pub file: &'static str,
    /// The model's trained context, in tokens.
    pub context: usize,
    /// First 16 hex of the blake3 of `file` as the registry downloads it.
    pub blake3: Option<&'static str>,
}

/// BAAI's bge-reranker-v2-m3 (568M, XLM-R, 100+ languages), Apache-2.0 both
/// upstream and in the GGUF repository. Converted with llama.cpp's own
/// BERT path: add_bos, add_eos, no SEP, no rerank template, so the server's
/// flag-driven format above applies. Q8_0.
const BGE_RERANKER: RerankerSpec = RerankerSpec {
    alias: "bge-reranker",
    repo: "gpustack/bge-reranker-v2-m3-GGUF",
    file: "bge-reranker-v2-m3-Q8_0.gguf",
    context: 8192,
    blake3: Some("fc10b1b86a09d71f"),
};

pub fn rerankers() -> &'static [RerankerSpec] {
    &[BGE_RERANKER]
}

/// The default reranker: the registry's first entry.
pub fn default_spec() -> &'static RerankerSpec {
    &rerankers()[0]
}

/// An alias to its spec.
pub fn resolve(name: &str) -> Result<&'static RerankerSpec> {
    rerankers()
        .iter()
        .find(|r| r.alias == name)
        .ok_or_else(|| {
            let aliases: Vec<&str> = rerankers().iter().map(|r| r.alias).collect();
            anyhow!("Unknown reranker '{name}'. Available rerankers: {}", aliases.join(", "))
        })
}

/// `CKQ_RERANKER`'s spec, else the default.
pub fn spec_from_env() -> Result<&'static RerankerSpec> {
    match std::env::var("CKQ_RERANKER") {
        Ok(name) => resolve(name.trim()),
        Err(_) => Ok(default_spec()),
    }
}

/// How `--rerank` orders the candidates (`CKQ_RERANK_MODE`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Mode {
    /// Sort the top N by the reranker's score.
    Pure,
    /// Add the reranker's rank to the hybrid fusion as a third list.
    Fuse,
}

/// The knobs of one `--rerank` run, from the environment. Defaults: N = 30,
/// `pure`, weight 1.0.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Config {
    /// How many of the fused files are scored (`CKQ_RERANK_TOPN`).
    pub topn: usize,
    pub mode: Mode,
    /// The rerank list's weight in `fuse` (`CKQ_RERANK_WEIGHT`).
    pub weight: f32,
}

impl Config {
    pub fn from_env() -> Result<Self> {
        let topn = match std::env::var("CKQ_RERANK_TOPN") {
            Ok(v) => v
                .trim()
                .parse::<usize>()
                .map_err(|e| anyhow!("CKQ_RERANK_TOPN must be a positive integer (got '{v}'): {e}"))?,
            Err(_) => 30,
        };
        if topn == 0 {
            anyhow::bail!("CKQ_RERANK_TOPN must be at least 1");
        }
        let mode = match std::env::var("CKQ_RERANK_MODE") {
            Ok(v) => match v.trim() {
                "pure" => Mode::Pure,
                "fuse" => Mode::Fuse,
                other => anyhow::bail!("CKQ_RERANK_MODE must be 'pure' or 'fuse' (got '{other}')"),
            },
            Err(_) => Mode::Pure,
        };
        let weight = match std::env::var("CKQ_RERANK_WEIGHT") {
            Ok(v) => v
                .trim()
                .parse::<f32>()
                .map_err(|e| anyhow!("CKQ_RERANK_WEIGHT must be a number (got '{v}'): {e}"))?,
            Err(_) => 1.0,
        };
        if !weight.is_finite() || weight <= 0.0 {
            anyhow::bail!("CKQ_RERANK_WEIGHT must be a positive number (got {weight})");
        }
        Ok(Config { topn, mode, weight })
    }
}

/// Ceiling on one (query, passage) pair, in tokens: the passage is cut at
/// 1,024 and the query and any template add the rest.
pub const PASSAGE_TOKENS: usize = 1024;
/// Headroom for the query and a rerank template over the passage ceiling.
const PAIR_SLACK: usize = 256;
/// Sequences per decode, the embedder's measured optimum.
const MAX_SEQ: usize = 4;

struct Job {
    query: String,
    passages: Vec<String>,
    reply: Sender<Result<Vec<f32>>>,
}

/// A loaded reranker: a channel to the thread that owns the model and its
/// rank-pooling context. Dropping it stops that thread, contexts first, then
/// the model — the ordering ggml requires — while the backend is alive.
pub struct Reranker {
    alias: &'static str,
    tx: Option<Sender<Job>>,
    handle: Option<JoinHandle<()>>,
}

impl Reranker {
    /// Download (first use) and load `spec`. With `CKQ_TIMING=1`, one stderr
    /// line carries the load time, so it can be read apart from scoring.
    pub fn load(spec: &'static RerankerSpec) -> Result<Self> {
        let timing = std::env::var_os("CKQ_TIMING").is_some();
        let path = models::fetch_gguf(spec.repo, spec.file, spec.blake3)?;
        let device = crate::embed::query_device();
        let t0 = Instant::now();
        let (tx, rx) = channel::<Job>();
        let (ready_tx, ready_rx) = channel::<Result<()>>();
        let loader = move || {
            let run_all = (|| -> Result<()> {
                let backend = crate::embed::backend()?;
                let params = llama_cpp_2::model::params::LlamaModelParams::default().with_n_gpu_layers(
                    match device {
                        crate::embed::Device::Gpu => 1000,
                        crate::embed::Device::Cpu => 0,
                    },
                );
                let model = llama_cpp_2::model::LlamaModel::load_from_file(backend, &path, &params)
                    .map_err(|e| anyhow!("loading reranker GGUF {}: {e}", path.display()))?;
                let per_seq = (PASSAGE_TOKENS + PAIR_SLACK).min(spec.context) as u32;
                let budget = per_seq * MAX_SEQ as u32;
                let n = std::num::NonZeroU32::new(budget).expect("non-zero context");
                let mut ctx = model
                    .new_context(
                        backend,
                        llama_cpp_2::context::params::LlamaContextParams::default()
                            .with_n_ctx(Some(n))
                            .with_n_threads_batch(crate::embed::threads())
                            .with_embeddings(true)
                            .with_n_seq_max(MAX_SEQ as u32)
                            .with_n_batch(budget)
                            .with_n_ubatch(budget)
                            .with_pooling_type(llama_cpp_2::context::params::LlamaPoolingType::Rank),
                    )
                    .map_err(|e| anyhow!("reranker context: {e}"))?;
                ready_tx
                    .send(Ok(()))
                    .map_err(|_| anyhow!("caller left before the reranker loaded"))?;
                while let Ok(job) = rx.recv() {
                    let _ = job.reply.send(run(&model, &mut ctx, &job.query, &job.passages, per_seq as usize));
                }
                Ok(())
            })();
            if let Err(e) = run_all {
                let _ = ready_tx.send(Err(e));
            }
            // Locals drop here in reverse declaration order — context, then
            // model — while the backend is still alive.
        };
        let handle = thread::Builder::new()
            .name("ckq-rerank".into())
            .spawn(loader)
            .map_err(|e| anyhow!("spawning the reranker thread: {e}"))?;
        match ready_rx.recv() {
            Ok(Ok(())) => {}
            Ok(Err(e)) => {
                handle.join().ok();
                anyhow::bail!("loading the reranker: {e:#}");
            }
            Err(_) => {
                handle.join().ok();
                anyhow::bail!("the reranker thread died during startup");
            }
        }
        if timing {
            eprintln!("ckq: rerank: loaded {} in {:.2} s", spec.alias, t0.elapsed().as_secs_f32());
        }
        Ok(Self { alias: spec.alias, tx: Some(tx), handle: Some(handle) })
    }

    /// One score per passage: higher means more relevant to the query.
    /// With `CKQ_TIMING=1`, one stderr line carries this search's scoring
    /// time over the passages (the model is warm).
    pub fn score(&self, query: &str, passages: &[String]) -> Result<Vec<f32>> {
        if passages.is_empty() {
            return Ok(Vec::new());
        }
        let timing = std::env::var_os("CKQ_TIMING").is_some();
        let t0 = Instant::now();
        let (reply_tx, reply_rx) = channel();
        self.tx
            .as_ref()
            .ok_or_else(|| anyhow!("the reranker is stopped"))?
            .send(Job {
                query: query.to_string(),
                passages: passages.to_vec(),
                reply: reply_tx,
            })
            .map_err(|_| anyhow!("the reranker thread is gone"))?;
        let scores = reply_rx
            .recv()
            .map_err(|_| anyhow!("the reranker thread dropped the job"))??;
        if timing {
            eprintln!(
                "ckq: rerank: scored {} passages with {} in {:.3} s",
                passages.len(),
                self.alias,
                t0.elapsed().as_secs_f32()
            );
        }
        Ok(scores)
    }
}

impl Drop for Reranker {
    fn drop(&mut self) {
        // Dropping the sender ends the thread's receive loop; joining waits
        // for its contexts and model to drop in order.
        self.tx.take();
        if let Some(handle) = self.handle.take() {
            let _ = handle.join();
        }
    }
}

/// Score every (query, passage) pair: tokenize per the llama.cpp server's
/// rerank format, decode in batches, read each sequence's pooled output.
fn run(
    model: &llama_cpp_2::model::LlamaModel,
    ctx: &mut llama_cpp_2::context::LlamaContext<'_>,
    query: &str,
    passages: &[String],
    budget: usize,
) -> Result<Vec<f32>> {
    let vocab = model.vocab();
    let mut encoded: Vec<Vec<_>> = Vec::with_capacity(passages.len());
    let mut truncated = 0usize;
    for passage in passages {
        let mut tokens = pair_tokens(model, &vocab, query, passage);
        if tokens.len() > budget {
            truncated += 1;
            tokens.truncate(budget);
        }
        encoded.push(tokens);
    }
    if truncated > 0 {
        eprintln!(
            "ckq: warning: truncated {truncated} of {} rerank inputs to {budget} tokens \
             (a reranker that pools its last token scores the truncation, not the passage; \
             further truncations are not logged)",
            passages.len()
        );
    }

    let ctx_budget = ctx.n_ctx() as usize;
    let mut out = Vec::with_capacity(passages.len());
    let mut start = 0usize;
    while start < encoded.len() {
        let mut end = start;
        let mut total = 0usize;
        while end < encoded.len() {
            let len = encoded[end].len().max(1);
            if end > start && (total + len > ctx_budget || end - start >= MAX_SEQ) {
                break;
            }
            total += len;
            end += 1;
        }
        let mut batch =
            llama_cpp_2::llama_batch::LlamaBatch::new(total.max(1), (end - start) as i32);
        for (slot, tokens) in encoded[start..end].iter().enumerate() {
            batch
                .add_sequence(tokens, slot as i32, true)
                .map_err(|e| anyhow!("rerank batch: {e}"))?;
        }
        ctx.clear_kv_cache();
        ctx.decode(&mut batch).map_err(|e| anyhow!("rerank decode: {e}"))?;
        for (slot, tokens) in encoded[start..end].iter().enumerate() {
            if tokens.is_empty() {
                out.push(f32::NAN);
                continue;
            }
            // What the llama.cpp server reads for a rerank task: the pooled
            // sequence output's first element, unnormalised.
            let pooled = ctx
                .embeddings_seq_ith(slot as i32)
                .map_err(|e| anyhow!("rerank embeddings: {e}"))?;
            out.push(pooled.first().copied().unwrap_or(f32::NAN));
        }
        start = end;
    }
    Ok(out)
}

/// The (query, passage) input, as llama.cpp's server formats it for a rerank
/// endpoint: the GGUF's `rerank` chat template where the model carries one,
/// else the vocabulary's own BOS/EOS/SEP flags with EOS falling back to SEP.
fn pair_tokens(
    model: &llama_cpp_2::model::LlamaModel,
    vocab: &llama_cpp_2::vocab::LlamaVocab<'_>,
    query: &str,
    doc: &str,
) -> Vec<llama_cpp_2::token::LlamaToken> {
    if let Ok(template) = model.chat_template(Some("rerank")) {
        if let Ok(text) = template.to_str() {
            let prompt = text.replace("{query}", query).replace("{document}", doc);
            // Special tokens parsed (the template is full of them), nothing
            // added: the template is the whole input.
            return vocab.tokenize(prompt.as_bytes(), false, true);
        }
    }
    let eos = if vocab.eos().0 >= 0 { vocab.eos() } else { vocab.sep() };
    let mut tokens = Vec::new();
    if vocab.should_add_bos() {
        tokens.push(vocab.bos());
    }
    tokens.extend(vocab.tokenize(query.as_bytes(), false, false));
    if vocab.should_add_eos() {
        tokens.push(eos);
    }
    if vocab.should_add_sep() {
        tokens.push(vocab.sep());
    }
    tokens.extend(vocab.tokenize(doc.as_bytes(), false, false));
    if vocab.should_add_eos() {
        tokens.push(eos);
    }
    tokens
}
