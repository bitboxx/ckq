//! Printing results. Every path printed goes through `Corpus::display`.
//!
//! `--jsonl` is the contract a wrapper parses and must not change shape: one
//! object per line,
//!
//! ```text
//! {"path": "...", "span": {"byte_start": 0, "byte_end": 430, "line_start": 1, "line_end": 4},
//!  "language": "markdown", "snippet": "...", "score": 0.67,
//!  "signals": {"lex_rank": 1, "vec_rank": 3, "rrf_score": 0.41}}
//! ```
//!
//! `snippet` is absent with `--no-snippet`; `signals` only on hybrid hits.
//! `--json` and the plain text form keep what `tests/e2e_contract.rs` pins.
//! Nothing but results goes to stdout: progress, warnings and "No matches
//! found" go to stderr.

use crate::corpus::Corpus;
use crate::grep::Match;
use crate::search::Hit;
use anyhow::Result;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Format {
    Text,
    Json,
    Jsonl,
}

#[derive(Debug, Clone, Copy)]
pub struct OutputOptions {
    pub format: Format,
    pub snippet: bool,
    pub scores: bool,
    pub line_numbers: bool,
    pub files_only: bool,
    /// Name of the model, for `--json`'s `model` field; "none" for grep and lexical.
    pub model: &'static str,
}

/// Print ranked hits. Reads each hit's file through `corpus.abs` for the
/// snippet. Returns the number printed.
pub fn print_hits(corpus: &Corpus, hits: &[Hit], opts: &OutputOptions) -> Result<usize> {
    todo!()
}

/// Print regex matches. Returns the number printed.
pub fn print_matches(corpus: &Corpus, matches: &[Match], opts: &OutputOptions) -> Result<usize> {
    todo!()
}
