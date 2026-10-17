//! The three ranked methods: lexical (BM25), semantic (embeddings) and hybrid
//! (both, fused). All read-only, all keyed on `RelPath`.
//!
//! Behaviour carried over from the old engine, because the benchmarks and the
//! vault evaluation were measured on it:
//! - Lexical ranks files. Without `--term` it searches the query's words; with
//!   `--term` it searches the terms instead, each as a phrase, any of them.
//!   Scores are normalised so the best file is 1.0. The span shown is the
//!   file's chunk with the most matches.
//! - Semantic ranks chunks by cosine similarity and keeps those at or above
//!   the threshold (the model's own unless `--threshold`).
//! - Hybrid fuses per file by reciprocal rank, k = 3: each method contributes
//!   weight / (k + rank). The lexical weight is 1.0 with `--term` and 0.15
//!   without (at full weight a question matched same-subject documents in its
//!   own language and rank-1 hits on the bench fell from 35 to 15 of 40).
//!   `--threshold` filters the semantic list before fusion. One result per
//!   file, its span from the semantic method when it found the file. Each hit
//!   carries both ranks and the fused score.

use crate::chunk::Chunk;
use crate::corpus::{Corpus, RelPath, Scope};
use crate::embed::Embedder;
use crate::store::Store;
use anyhow::Result;

pub const RRF_K: f32 = 3.0;
pub const LEX_WEIGHT_WITH_TERMS: f32 = 1.0;
pub const LEX_WEIGHT_QUERY_ONLY: f32 = 0.15;

#[derive(Debug, Clone)]
pub struct Query {
    pub text: String,
    /// `--term` values. When non-empty they replace `text` for the lexical method.
    pub terms: Vec<String>,
    pub topk: usize,
    /// `--threshold`; `None` means the model's default for semantic, none for lexical.
    pub threshold: Option<f32>,
    pub scope: Scope,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Signals {
    pub lex_rank: Option<usize>,
    pub vec_rank: Option<usize>,
    pub rrf_score: f32,
}

#[derive(Debug, Clone, PartialEq)]
pub struct Hit {
    pub rel: RelPath,
    pub chunk: Chunk,
    pub score: f32,
    /// Set by hybrid only.
    pub signals: Option<Signals>,
}

/// The FTS5 MATCH expression: the terms as quoted phrases joined by OR, or the
/// query's words as quoted tokens joined by OR. Quotes inside are doubled.
pub fn fts_query(q: &Query) -> String {
    todo!()
}

pub fn lexical(store: &Store, corpus: &Corpus, q: &Query) -> Result<Vec<Hit>> {
    todo!("store.bm25, normalise, then pick each file's best chunk by reading corpus.abs(rel)")
}

pub fn semantic(store: &Store, embedder: &mut dyn Embedder, q: &Query, default_threshold: f32) -> Result<Vec<Hit>> {
    todo!("embed_query; dot product against store.vectors(scope); threshold; topk")
}

pub fn hybrid(
    store: &Store,
    corpus: &Corpus,
    embedder: &mut dyn Embedder,
    q: &Query,
    default_threshold: f32,
) -> Result<Vec<Hit>> {
    todo!("lexical depth >= 30, semantic depth topk*3; fuse by RelPath")
}
