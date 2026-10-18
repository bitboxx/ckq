//! The three ranked methods: lexical (BM25), semantic (embeddings) and hybrid
//! (both, fused). All read-only, all keyed on `RelPath`.
//!
//! Behaviour carried over from the old engine, because the benchmarks and a
//! private evaluation on a personal note collection were measured on it:
//! - Lexical ranks files. Without `--term` it searches the query's words; with
//!   `--term` it searches the terms instead, each as a phrase, any of them.
//!   Scores are normalised so the best file is 1.0. The span shown is the
//!   file's chunk with the most matches.
//! - Semantic ranks chunks by cosine similarity and keeps those at or above
//!   the threshold (the model's own unless `--threshold`).
//! - Hybrid fuses per file by CombMAX over weighted reciprocal ranks: a rank r
//!   scores weight / (k + r) with k = 3, a file takes the better of its two
//!   scores, and the other adds 0.01 of itself to break near-ties. Summed RRF
//!   lost on that private evaluation (17 Oct 2026, 48 questions): at every k from
//!   3 to 60 and every lexical weight from 0.4 to 2.0 a note both methods
//!   ranked second or third overtook the note one method ranked first, and
//!   34 to 36 answers came first against 37 for CombMAX. The lexical weight is
//!   1.05 with `--term`, so a keyword first place leads at equal rank (1.0
//!   gave 34 first, 1.3 gave 37 with a lower MRR), and 0.15 without (at full
//!   weight a question matched same-subject documents in its own language and
//!   rank-1 hits on the bench fell from 35 to 15 of 40).
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
pub const LEX_WEIGHT_WITH_TERMS: f32 = 1.05;
pub const AGREEMENT_WEIGHT: f32 = 0.01;
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
    // Fusion needs the lexical ranking, not its spans: `lexical` would read
    // and re-chunk every ranked file, and for each one the semantic side
    // also found, fusion keeps the semantic chunk and drops the lexical
    // hit's. Rank here through the store alone; locate spans afterwards,
    // only for the fused files that still lack one.
    let expr = fts_query(&lex_query);
    let ranked = if expr.is_empty() {
        Vec::new()
    } else {
        store.bm25(&expr, &q.scope, depth)?
    };
    let no_span = || Chunk {
        byte_start: 0,
        byte_end: 0,
        line_start: 1,
        line_end: 1,
    };
    let lexical: Vec<Hit> = ranked
        .into_iter()
        .map(|(rel, _)| Hit {
            rel,
            chunk: no_span(),
            score: 0.0,
            signals: None,
        })
        .collect();
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
    let mut fused = fuse_by_file(&lexical, &semantic, lex_weight);
    fused.truncate(q.topk);
    let terms = span_terms(q);
    for hit in &mut fused {
        if hit.signals.is_some_and(|s| s.vec_rank.is_none()) {
            let (chunk, _) = locate_span(corpus, &hit.rel, &terms);
            hit.chunk = chunk;
        }
    }
    Ok(fused)
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
            let lex = lexical_weight * rank_score(entry.lex_rank);
            let vec = rank_score(entry.vec_rank);
            let score = lex.max(vec) + AGREEMENT_WEIGHT * lex.min(vec);
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

    fn hits(names: &[&str]) -> Vec<Hit> {
        names
            .iter()
            .map(|n| Hit {
                rel: RelPath::new(std::path::Path::new(n)).unwrap(),
                chunk: Chunk {
                    byte_start: 0,
                    byte_end: 1,
                    line_start: 1,
                    line_end: 1,
                },
                score: 1.0,
                signals: None,
            })
            .collect()
    }

    fn position(fused: &[Hit], name: &str) -> usize {
        fused.iter().position(|h| h.rel.as_str() == name).unwrap()
    }

    #[test]
    fn fusion_gives_one_result_per_file_with_both_ranks() {
        let fused = fuse_by_file(
            &hits(&["a.md", "b.md"]),
            &hits(&["b.md", "b.md", "c.md"]),
            LEX_WEIGHT_WITH_TERMS,
        );
        assert_eq!(fused.len(), 3);
        let b = &fused[position(&fused, "b.md")];
        let s = b.signals.unwrap();
        assert_eq!((s.lex_rank, s.vec_rank), (Some(2), Some(1)));
        assert_eq!(s.rrf_score, b.score);
    }

    #[test]
    fn a_keyword_first_place_beats_a_shared_second_place() {
        // Summed RRF would put both.md first; CombMAX must not.
        let fused = fuse_by_file(
            &hits(&["top.md", "both.md"]),
            &hits(&["sem.md", "both.md"]),
            LEX_WEIGHT_WITH_TERMS,
        );
        assert_eq!(position(&fused, "top.md"), 0, "keywords lead at equal rank");
        assert!(position(&fused, "sem.md") < position(&fused, "both.md"));
        assert!(fused[position(&fused, "both.md")].score > 0.0);
    }

    #[test]
    fn agreement_breaks_a_tie_between_equal_best_ranks() {
        // x.md and y.md are third and fourth for the keywords; y.md is also
        // third for the semantic method, which ties it with x.md and lifts it.
        let fused = fuse_by_file(
            &hits(&["l1.md", "l2.md", "x.md", "y.md"]),
            &hits(&["s1.md", "s2.md", "y.md"]),
            1.0,
        );
        assert!(position(&fused, "y.md") < position(&fused, "x.md"));
    }

    #[test]
    fn a_low_lexical_weight_cannot_overturn_the_semantic_first_place() {
        let fused = fuse_by_file(
            &hits(&["wrong.md", "b.md"]),
            &hits(&["right.md", "b.md"]),
            LEX_WEIGHT_QUERY_ONLY,
        );
        assert_eq!(fused[0].rel.as_str(), "right.md");
        let full = fuse_by_file(
            &hits(&["wrong.md", "b.md"]),
            &hits(&["right.md", "b.md"]),
            LEX_WEIGHT_WITH_TERMS,
        );
        assert_eq!(full[0].rel.as_str(), "wrong.md");
    }

    /// An embedder whose query vector is fixed: enough to drive `semantic`
    /// against vectors the test wrote into the store itself.
    struct FixedEmbedder(Vec<f32>);

    impl crate::embed::Embedder for FixedEmbedder {
        fn dims(&self) -> usize {
            self.0.len()
        }
        fn embed_documents(&mut self, _: &[&str]) -> Result<Vec<Vec<f32>>> {
            unreachable!("hybrid embeds only the query")
        }
        fn embed_query(&mut self, _: &str) -> Result<Vec<f32>> {
            Ok(self.0.clone())
        }
    }

    #[test]
    fn hybrid_keeps_semantic_spans_and_locates_only_the_rest() {
        use crate::models;
        use crate::store::{FileRecord, IndexMeta, Store};
        use tempfile::TempDir;

        let dir = TempDir::new().unwrap();
        let corpus = Corpus::for_index(dir.path()).unwrap();
        // Real files, so a span that gets located is located in them.
        std::fs::write(dir.path().join("sem.md"), "needle semantic home\n").unwrap();
        std::fs::write(
            dir.path().join("lex.md"),
            "filler line\nneedle lexical span\n",
        )
        .unwrap();
        let meta = IndexMeta::for_model(models::resolve(models::DEFAULT_ALIAS).unwrap());
        let mut store = Store::open_write(&corpus, &meta).unwrap();
        let rec = |rel: &str, hash: &str| FileRecord {
            rel: RelPath::new(std::path::Path::new(rel)).unwrap(),
            hash: hash.to_string(),
            size: 10,
            mtime_ns: 1,
        };
        let sem_chunk = Chunk {
            byte_start: 0,
            byte_end: 21,
            line_start: 1,
            line_end: 1,
        };
        let lex_chunk = Chunk {
            byte_start: 0,
            byte_end: 26,
            line_start: 1,
            line_end: 2,
        };
        // sem.md has a vector; lex.md is lexical-only.
        store
            .put_file(
                &rec("sem.md", "h1"),
                "needle semantic home\n",
                &[sem_chunk],
                &[vec![0.5; 768]],
            )
            .unwrap();
        store
            .put_file(
                &rec("lex.md", "h2"),
                "filler line\nneedle lexical span\n",
                &[lex_chunk],
                &[],
            )
            .unwrap();

        let q = query("needle", &[]);
        let mut embedder = FixedEmbedder(vec![1.0; 768]);
        let hits = hybrid(&store, &corpus, &mut embedder, &q, -1.0).unwrap();
        assert_eq!(hits.len(), 2, "{:?}", hits);

        // The file the semantic side found keeps the store's span...
        let sem = hits.iter().find(|h| h.rel.as_str() == "sem.md").unwrap();
        assert_eq!(sem.chunk, sem_chunk);
        // ...and the lexical-only file gets a span located in the real file
        // that covers the needle (bytes 12..18 of "filler line\nneedle..."),
        // not the empty placeholder.
        let lex = hits.iter().find(|h| h.rel.as_str() == "lex.md").unwrap();
        assert!(lex.signals.unwrap().vec_rank.is_none());
        assert!(
            lex.chunk.byte_start <= 12 && lex.chunk.byte_end >= 18,
            "span {:?} does not cover the match",
            lex.chunk
        );
    }
}
