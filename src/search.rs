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
use std::collections::HashMap;

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
    let mut parts: Vec<String> = Vec::new();
    let phrase = |s: &str| format!("\"{}\"", s.replace('"', "\"\""));
    if q.terms.is_empty() {
        parts.extend(query_tokens(&q.text).into_iter().map(|t| phrase(&t)));
    } else {
        for term in &q.terms {
            // As the old engine did: a quote or backslash becomes a space, so
            // it splits the term rather than corrupting the phrase syntax.
            let clean = term.replace(['"', '\\'], " ");
            let clean = clean.trim();
            if clean.is_empty() {
                continue;
            }
            parts.push(phrase(clean));
            // The words of a multi-word term also count on their own, since
            // an exact phrase is rare; BM25 does the weighting.
            let words: Vec<&str> = clean.split_whitespace().collect();
            if words.len() > 1 {
                parts.extend(
                    words
                        .into_iter()
                        .filter(|w| w.chars().count() > 3)
                        .map(phrase),
                );
            }
        }
    }
    parts.join(" OR ")
}

/// The words a query searches, as the FTS5 tokenizer will see them: split on
/// non-alphanumerics, lowercased.
fn query_tokens(q: &str) -> Vec<String> {
    q.split(|c: char| !c.is_alphanumeric())
        .filter(|t| !t.is_empty())
        .map(str::to_lowercase)
        .collect()
}

pub fn lexical(store: &Store, corpus: &Corpus, q: &Query) -> Result<Vec<Hit>> {
    let expr = fts_query(q);
    if expr.is_empty() {
        return Ok(Vec::new());
    }
    let ranked = store.bm25(&expr, &q.scope, q.topk)?;
    if ranked.is_empty() {
        return Ok(Vec::new());
    }
    let best = ranked[0].1;
    let terms = span_terms(q);
    let mut hits = Vec::with_capacity(ranked.len());
    for (rel, raw) in ranked {
        let score = if best > 0.0 { raw / best } else { raw };
        if q.threshold.is_some_and(|t| score < t) {
            continue;
        }
        let (chunk, _text) = locate_span(corpus, &rel, &terms);
        hits.push(Hit {
            rel,
            chunk,
            score,
            signals: None,
        });
    }
    Ok(hits)
}

/// The content terms span location counts matches of: the terms' words when
/// given, else the query's words.
fn span_terms(q: &Query) -> Vec<String> {
    if q.terms.is_empty() {
        return query_tokens(&q.text);
    }
    q.terms.iter().flat_map(|t| query_tokens(t)).collect()
}

/// The chunk of the file with the most whole-token matches for `terms`, ties
/// to the earliest chunk; the whole first line when nothing matches or the
/// file cannot be read. Ported from the old engine's `locate_lexical_span`.
fn locate_span(corpus: &Corpus, rel: &RelPath, terms: &[String]) -> (Chunk, String) {
    let fallback = Chunk {
        byte_start: 0,
        byte_end: 0,
        line_start: 1,
        line_end: 1,
    };
    let Ok(text) = std::fs::read_to_string(corpus.abs(rel)) else {
        return (fallback, String::new());
    };
    if terms.is_empty() {
        let end = text.lines().next().map(str::len).unwrap_or(0);
        return (
            Chunk {
                byte_end: end,
                ..fallback
            },
            text.lines().next().unwrap_or("").to_string(),
        );
    }
    let chunks = crate::chunk::chunk(rel, &text);
    let mut best: Option<(usize, Chunk)> = None;
    for chunk in chunks {
        let body = &text[chunk.byte_start..chunk.byte_end];
        let hits = query_tokens(body)
            .into_iter()
            .filter(|token| terms.contains(token))
            .count();
        if hits > 0 && best.is_none_or(|(b, _)| hits > b) {
            best = Some((hits, chunk));
        }
    }
    match best {
        Some((_, chunk)) => (chunk, text[chunk.byte_start..chunk.byte_end].to_string()),
        None => (
            Chunk {
                byte_end: text.lines().next().map(str::len).unwrap_or(0),
                ..fallback
            },
            text.lines().next().unwrap_or("").to_string(),
        ),
    }
}

pub fn semantic(
    store: &Store,
    embedder: &mut dyn Embedder,
    q: &Query,
    default_threshold: f32,
) -> Result<Vec<Hit>> {
    let set = store.vectors(&q.scope)?;
    if set.chunks.is_empty() {
        return Ok(Vec::new());
    }
    let query = embedder.embed_query(&q.text)?;
    let threshold = q.threshold.unwrap_or(default_threshold);
    let dims = set.dims;
    let mut scored: Vec<(usize, f32)> = set
        .chunks
        .iter()
        .enumerate()
        .map(|(i, _)| {
            let row = &set.data[i * dims..(i + 1) * dims];
            let score = row.iter().zip(&query).map(|(a, b)| a * b).sum::<f32>();
            (i, score)
        })
        .filter(|(_, score)| *score >= threshold)
        .collect();
    // Score first, then the store's own order (path, then chunk), so equal
    // scores keep a stable order across runs.
    scored.sort_by(|a, b| {
        b.1.partial_cmp(&a.1)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then(a.0.cmp(&b.0))
    });
    scored.truncate(q.topk);
    Ok(scored
        .into_iter()
        .map(|(i, score)| Hit {
            rel: set.chunks[i].rel.clone(),
            chunk: set.chunks[i].chunk,
            score,
            signals: None,
        })
        .collect())
}
pub fn hybrid(
    store: &Store,
    corpus: &Corpus,
    embedder: &mut dyn Embedder,
    q: &Query,
    default_threshold: f32,
) -> Result<Vec<Hit>> {
    // Fusion can only promote what it sees, so each method supplies more
    // candidates than the final cut.
    let depth = (q.topk * 3).max(30);
    let lex_query = Query {
        topk: depth,
        threshold: None,
        scope: q.scope.clone(),
        text: q.text.clone(),
        terms: q.terms.clone(),
    };
    let lexical = lexical(store, corpus, &lex_query)?;
    let sem_query = Query {
        topk: depth * 3,
        threshold: q.threshold,
        scope: q.scope.clone(),
        text: q.text.clone(),
        terms: q.terms.clone(),
    };
    let semantic = semantic(store, embedder, &sem_query, default_threshold)?;

    let lex_weight = if q.terms.is_empty() {
        LEX_WEIGHT_QUERY_ONLY
    } else {
        LEX_WEIGHT_WITH_TERMS
    };
    let fused = fuse_by_file(&lexical, &semantic, lex_weight);
    Ok(fused.into_iter().take(q.topk).collect())
}

/// Reciprocal rank fusion of two rankings at file level, the lexical one
/// scaled by `lexical_weight`. Each list contributes its best rank per file;
/// the semantic list arrives as chunks, best first, so a file's first chunk
/// sets its rank. Ties break on path, for stable output. Ported from the old
/// engine's `fuse_by_file`.
fn fuse_by_file(lexical: &[Hit], semantic: &[Hit], lexical_weight: f32) -> Vec<Hit> {
    struct Entry {
        hit: Hit,
        lex_rank: Option<usize>,
        vec_rank: Option<usize>,
    }
    let mut entries: HashMap<RelPath, Entry> = HashMap::new();

    let mut vec_rank = 0;
    for chunk in semantic {
        if entries.contains_key(&chunk.rel) {
            continue;
        }
        vec_rank += 1;
        entries.insert(
            chunk.rel.clone(),
            Entry {
                hit: chunk.clone(),
                lex_rank: None,
                vec_rank: Some(vec_rank),
            },
        );
    }

    let mut lex_rank = 0;
    for hit in lexical {
        let rank = lex_rank + 1;
        let entry = entries.entry(hit.rel.clone()).or_insert_with(|| Entry {
            hit: hit.clone(),
            lex_rank: None,
            vec_rank: None,
        });
        if entry.lex_rank.is_none() {
            lex_rank = rank;
            entry.lex_rank = Some(rank);
        }
    }

    let rank_score = |rank: Option<usize>| rank.map_or(0.0, |r| 1.0 / (RRF_K + r as f32));
    let mut fused: Vec<Hit> = entries
        .into_values()
        .map(|mut entry| {
            let score = lexical_weight * rank_score(entry.lex_rank) + rank_score(entry.vec_rank);
            entry.hit.score = score;
            entry.hit.signals = Some(Signals {
                lex_rank: entry.lex_rank,
                vec_rank: entry.vec_rank,
                rrf_score: score,
            });
            entry.hit
        })
        .collect();
    fused.sort_by(|a, b| {
        b.score
            .partial_cmp(&a.score)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| a.rel.cmp(&b.rel))
    });
    fused
}

#[cfg(test)]
mod tests {
    use super::*;

    fn query(text: &str, terms: &[&str]) -> Query {
        Query {
            text: text.to_string(),
            terms: terms.iter().map(|s| s.to_string()).collect(),
            topk: 10,
            threshold: None,
            scope: Scope::All,
        }
    }

    #[test]
    fn a_query_becomes_quoted_tokens_joined_by_or() {
        assert_eq!(
            fts_query(&query("volcano erupt", &[])),
            "\"volcano\" OR \"erupt\""
        );
        assert_eq!(fts_query(&query("a-b c", &[])), "\"a\" OR \"b\" OR \"c\"");
    }

    #[test]
    fn terms_are_phrases_plus_long_words() {
        assert_eq!(
            fts_query(&query("irrelevant", &["circuit breaker"])),
            "\"circuit breaker\" OR \"circuit\" OR \"breaker\""
        );
        // A one-word term is just the phrase.
        assert_eq!(fts_query(&query("q", &["quokka"])), "\"quokka\"");
    }

    #[test]
    fn quotes_in_a_term_split_it_like_the_old_engine() {
        // A quote cannot live inside an FTS5 phrase; the old engine replaced
        // it with a space, and this keeps the words searchable.
        assert_eq!(fts_query(&query("q", &["say \"hi\""])), "\"say  hi\"");
    }

    #[test]
    fn punctuation_only_queries_match_nothing() {
        assert_eq!(fts_query(&query("?? --", &[])), "");
    }
}
