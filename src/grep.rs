//! Regex mode: plain grep over the files, no index needed.
//!
//! The flags the contract tests pin: `-n`, `-i`, `-w`, `-F`, `-l`, `-C N`,
//! `--topk`, plus `--hidden` and `--no-ckqignore`. Walks with `walk::walk`, so
//! it skips what indexing skips. Output goes through `output`, like the ranked
//! methods: JSONL hits carry the span of the matching line and score 1.0.

use crate::corpus::{Corpus, RelPath, Scope};
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

/// The result of one regex search: the matches, and how many files in the
/// scope could not be read. grep exits 2 when an error occurred and nothing
/// matched; ckq does the same, and keeps exit 0 when something was found,
/// so a wrapper that reads nonzero as "no hits" still sees its hits.
#[derive(Debug, Clone, Default)]
pub struct GrepOutcome {
    pub matches: Vec<Match>,
    /// Files in the scope that could not be read. Each already produced one
    /// stderr warning naming it.
    pub read_errors: usize,
}

/// Every match, in path then line order, from `scope` only: files outside it
/// are neither read nor counted, so `--topk` limits in-scope matches alone
/// and a file argument pays for itself, not for its parent tree. An invalid
/// regex is an error whose message starts "Regex error".
pub fn grep(corpus: &Corpus, opts: &GrepOptions, scope: &Scope) -> Result<GrepOutcome> {
    let re = build_regex(opts)?;
    let files = crate::walk::walk(corpus, &opts.walk)?;
    let mut out = Vec::new();
    let mut read_errors = 0usize;
    'files: for file in files {
        if !scope.contains(&file.rel) {
            continue;
        }
        if file.unreadable {
            read_errors += 1;
            eprintln!(
                "ckq: warning: cannot read {}",
                corpus.abs(&file.rel).display()
            );
            continue;
        }
        let text = match std::fs::read_to_string(corpus.abs(&file.rel)) {
            Ok(text) => text,
            // Invalid UTF-8 past the probed head is a binary file, not an
            // error; anything else is, and is reported the way grep would.
            Err(e) if e.kind() == std::io::ErrorKind::InvalidData => continue,
            Err(e) => {
                read_errors += 1;
                eprintln!(
                    "ckq: warning: cannot read {}: {e}",
                    corpus.abs(&file.rel).display()
                );
                continue;
            }
        };
        // (line number, byte offset of the line, display text) per line; the
        // display text is the line without its break, a prefix of the raw.
        // `\r` alone breaks a line as well, matching the chunker's line
        // numbers (classic-Mac endings).
        let lines: Vec<(usize, usize, &str)> = text
            .split_inclusive(['\n', '\r'])
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
                if opts.limit.is_some_and(|limit| out.len() >= limit) {
                    break 'files;
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
    Ok(GrepOutcome {
        matches: out,
        read_errors,
    })
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn lone_cr_endings_count_lines_like_the_chunker() {
        let dir = std::env::temp_dir().join(format!("ckq-grep-cr-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(
            dir.join("cr.txt"),
            "alpha needle one\rneedle two\rno match\r",
        )
        .unwrap();
        let (corpus, scope) = Corpus::plain(&dir).unwrap();
        let opts = GrepOptions {
            pattern: "needle".to_string(),
            ..Default::default()
        };
        let out = grep(&corpus, &opts, &scope).unwrap();
        let texts: Vec<&str> = out.matches.iter().map(|m| m.text.as_str()).collect();
        assert_eq!(texts, vec!["alpha needle one", "needle two"]);
        let lines: Vec<usize> = out.matches.iter().map(|m| m.line).collect();
        assert_eq!(lines, vec![1, 2], "CR must end a line, as in the chunker");
        std::fs::remove_dir_all(&dir).unwrap();
    }
}
