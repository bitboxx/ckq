//! Regex mode: plain grep over the files, no index needed.
//!
//! The flags the contract tests pin: `-n`, `-i`, `-w`, `-F`, `-l`, `-C N`,
//! `--topk`, plus `--hidden` and `--no-ckignore`. Walks with `walk::walk`, so
//! it skips what indexing skips. Output goes through `output`, like the ranked
//! methods: JSONL hits carry the span of the matching line and score 1.0.

use crate::corpus::{Corpus, RelPath};
use crate::walk::WalkOptions;
use anyhow::Result;

#[derive(Debug, Clone, Default)]
pub struct GrepOptions {
    pub pattern: String,
    pub ignore_case: bool,
    pub word: bool,
    pub fixed: bool,
    pub context: usize,
    pub files_only: bool,
    pub limit: Option<usize>,
    pub walk: WalkOptions,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Match {
    pub rel: RelPath,
    /// 1-based.
    pub line: usize,
    pub byte_start: usize,
    pub byte_end: usize,
    pub text: String,
    /// `-C` lines before and after, with their line numbers.
    pub before: Vec<(usize, String)>,
    pub after: Vec<(usize, String)>,
}

/// Every match, in path then line order. An invalid regex is an error whose
/// message starts "Regex error".
pub fn grep(corpus: &Corpus, opts: &GrepOptions) -> Result<Vec<Match>> {
    todo!()
}
