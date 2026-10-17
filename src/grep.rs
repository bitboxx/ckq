//! Regex mode: plain grep over the files, no index needed.
//!
//! The flags the contract tests pin: `-n`, `-i`, `-w`, `-F`, `-l`, `-C N`,
//! `--topk`, plus `--hidden` and `--no-ckignore`. Walks with `walk::walk`, so
//! it skips what indexing skips. Output goes through `output`, like the ranked
//! methods: JSONL hits carry the span of the matching line and score 1.0.

use crate::corpus::{Corpus, RelPath};
use crate::walk::WalkOptions;
use anyhow::{Result, anyhow};
use regex::Regex;

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
    let re = build_regex(opts)?;
    let files = crate::walk::walk(corpus, &opts.walk)?;
    let mut out = Vec::new();
    'files: for file in files {
        let Ok(text) = std::fs::read_to_string(corpus.abs(&file.rel)) else {
            continue;
        };
        // (line number, byte offset of the line, display text) per line; the
        // display text is the line without its break, a prefix of the raw.
        let lines: Vec<(usize, usize, &str)> = text
            .split_inclusive('\n')
            .scan((1usize, 0usize), |(no, off), raw| {
                let trimmed = raw.trim_end_matches(['\n', '\r']);
                let entry = (*no, *off, trimmed);
                *no += 1;
                *off += raw.len();
                Some(entry)
            })
            .collect();
        for (idx, (line_no, offset, line)) in lines.iter().enumerate() {
            let mut matched = false;
            for m in re.find_iter(line) {
                if let Some(limit) = opts.limit {
                    if out.len() >= limit {
                        break 'files;
                    }
                }
                let before = lines[idx.saturating_sub(opts.context)..idx]
                    .iter()
                    .map(|(n, _, l)| (*n, l.to_string()))
                    .collect();
                let after_start = (idx + 1).min(lines.len());
                let after = lines[after_start..(idx + 1 + opts.context).min(lines.len())]
                    .iter()
                    .map(|(n, _, l)| (*n, l.to_string()))
                    .collect();
                out.push(Match {
                    rel: file.rel.clone(),
                    line: *line_no,
                    byte_start: offset + m.start(),
                    byte_end: offset + m.end(),
                    text: (*line).to_string(),
                    before,
                    after,
                });
                matched = true;
                if opts.files_only {
                    break;
                }
            }
            if matched && opts.files_only {
                break;
            }
        }
    }
    Ok(out)
}

/// The compiled pattern: `-F` escapes it, `-w` wraps it in word boundaries,
/// `-i` case-folds it.
fn build_regex(opts: &GrepOptions) -> Result<Regex> {
    let body = if opts.fixed {
        regex::escape(&opts.pattern)
    } else {
        opts.pattern.clone()
    };
    let body = if opts.word {
        format!(r"\b{body}\b")
    } else {
        body
    };
    let body = if opts.ignore_case {
        format!("(?i){body}")
    } else {
        body
    };
    Regex::new(&body).map_err(|e| anyhow!("Regex error: invalid pattern: {e}"))
}
