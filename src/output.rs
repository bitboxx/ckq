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
//! `--json` and the plain text form keep what `tests/e2e_contract.rs` pins:
//! `--json` objects carry `file`, `span`, `lang` (the capitalised variant
//! name), `symbol` (null), `score`, `signals` (always present, null ranks
//! outside hybrid), `preview` and the literal `"none"` in `model` — a
//! backwards-compatibility quirk of the old engine that consumers read.
//! Nothing but results goes to stdout: progress, warnings and "No matches
//! found" go to stderr.

use crate::corpus::Corpus;
use crate::grep::Match;
use crate::search::Hit;
use anyhow::Result;
use serde::Serialize;

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
    /// Show a file name with every result, as grep does for a directory (or
    /// several paths) but not for a single named file.
    pub filenames: bool,
}

/// The `--json` `lang` value: the capitalised variant names the old engine's
/// enum serialised, where `--jsonl` uses the lowercase display name.
fn json_lang(rel: &crate::corpus::RelPath) -> &'static str {
    match crate::chunk::language(rel) {
        "markdown" => "Markdown",
        "rust" => "Rust",
        "python" => "Python",
        "typescript" => "TypeScript",
        "javascript" => "JavaScript",
        "go" => "Go",
        "c" => "C",
        "cpp" => "Cpp",
        "csharp" => "CSharp",
        "ruby" => "Ruby",
        "haskell" => "Haskell",
        "zig" => "Zig",
        "json" => "Json",
        "yaml" => "Yaml",
        "toml" => "Toml",
        _ => "Text",
    }
}

#[derive(Serialize)]
struct SpanJson {
    byte_start: usize,
    byte_end: usize,
    line_start: usize,
    line_end: usize,
}
fn span_json(c: crate::chunk::Chunk) -> SpanJson {
    SpanJson {
        byte_start: c.byte_start,
        byte_end: c.byte_end,
        line_start: c.line_start,
        line_end: c.line_end,
    }
}
#[derive(Serialize)]
struct JsonlHit<'a> {
    path: String,
    span: SpanJson,
    language: &'a str,
    #[serde(skip_serializing_if = "Option::is_none")]
    snippet: Option<String>,
    score: f32,
    #[serde(skip_serializing_if = "Option::is_none")]
    signals: Option<SignalsJson>,
}

#[derive(Serialize)]
struct JsonHit<'a> {
    file: String,
    span: SpanJson,
    lang: &'a str,
    symbol: Option<String>,
    score: f32,
    signals: SignalsJson,
    preview: String,
    model: &'a str,
}

#[derive(Serialize)]
struct SignalsJson {
    lex_rank: Option<usize>,
    vec_rank: Option<usize>,
    rrf_score: f32,
}


/// Read the hit's snippet out of its file through `corpus.abs`, clamped to the
/// file's real length so a stale span degrades instead of panicking.
fn snippet(corpus: &Corpus, rel: &crate::corpus::RelPath, chunk: &crate::chunk::Chunk) -> String {
    let Ok(bytes) = std::fs::read(corpus.abs(rel)) else {
        return String::new();
    };
    let start = chunk.byte_start.min(bytes.len());
    let end = chunk.byte_end.clamp(start, bytes.len());
    String::from_utf8_lossy(&bytes[start..end]).into_owned()
}

/// Print ranked hits. Reads each hit's file through `corpus.abs` for the
/// snippet. Returns the number printed.
pub fn print_hits(corpus: &Corpus, hits: &[Hit], opts: &OutputOptions) -> Result<usize> {
    let mut printed = 0usize;
    let mut seen_files: Vec<crate::corpus::RelPath> = Vec::new();
    for hit in hits {
        if opts.files_only && seen_files.contains(&hit.rel) {
            continue;
        }
        seen_files.push(hit.rel.clone());
        let display = corpus.display(&hit.rel).display().to_string();
        let text = snippet(corpus, &hit.rel, &hit.chunk);
        match opts.format {
            Format::Jsonl => {
                let line = JsonlHit {
                    path: display,
                    span: span_json(hit.chunk),
                    language: crate::chunk::language(&hit.rel),
                    snippet: opts.snippet.then_some(text.clone()),
                    score: hit.score,
                    signals: hit.signals.map(|s| SignalsJson {
                        lex_rank: s.lex_rank,
                        vec_rank: s.vec_rank,
                        rrf_score: s.rrf_score,
                    }),
                };
                println!("{}", serde_json::to_string(&line)?);
            }
            Format::Json => {
                let line = JsonHit {
                    file: display,
                    span: span_json(hit.chunk),
                    lang: json_lang(&hit.rel),
                    symbol: None,
                    score: hit.score,
                    signals: SignalsJson {
                        lex_rank: hit.signals.and_then(|s| s.lex_rank),
                        vec_rank: hit.signals.and_then(|s| s.vec_rank),
                        rrf_score: hit.signals.map_or(hit.score, |s| s.rrf_score),
                    },
                    preview: text.clone(),
                    // The old engine always wrote the literal "none" here and
                    // the contract tests pin it.
                    model: "none",
                };
                println!("{}", serde_json::to_string(&line)?);
            }
            Format::Text => {
                let score = if opts.scores {
                    format!("[{:.3}] ", hit.score)
                } else {
                    String::new()
                };
                let preview = single_line_preview(&text);
                if opts.files_only {
                    println!("{display}");
                } else if opts.line_numbers && opts.filenames {
                    println!("{score}{}:{}:{preview}", display, hit.chunk.line_start);
                } else if opts.line_numbers {
                    println!("{score}{}:{preview}", hit.chunk.line_start);
                } else if opts.filenames {
                    println!("{score}{display}:\n{preview}");
                } else {
                    println!("{score}{preview}");
                }
            }
        }
        printed += 1;
    }
    Ok(printed)
}

/// The first line of a chunk preview, so a text-mode result stays one line
/// plus its file name, as the old engine's output did.
fn single_line_preview(text: &str) -> String {
    text.lines().next().unwrap_or("").trim_end().to_string()
}

/// Print regex matches. Returns the number printed.
pub fn print_matches(corpus: &Corpus, matches: &[Match], opts: &OutputOptions) -> Result<usize> {
    let mut printed = 0usize;
    let mut seen_files: Vec<crate::corpus::RelPath> = Vec::new();
    for m in matches {
        if opts.files_only && seen_files.contains(&m.rel) {
            continue;
        }
        seen_files.push(m.rel.clone());
        let display = corpus.display(&m.rel).display().to_string();
        match opts.format {
            Format::Jsonl => {
                let line = JsonlHit {
                    path: display,
                    span: SpanJson {
                        byte_start: m.byte_start,
                        byte_end: m.byte_end,
                        line_start: m.line,
                        line_end: m.line,
                    },
                    language: crate::chunk::language(&m.rel),
                    snippet: opts.snippet.then_some(m.text.clone()),
                    score: 1.0,
                    signals: None,
                };
                println!("{}", serde_json::to_string(&line)?);
            }
            Format::Json => {
                let line = JsonHit {
                    file: display,
                    span: SpanJson {
                        byte_start: m.byte_start,
                        byte_end: m.byte_end,
                        line_start: m.line,
                        line_end: m.line,
                    },
                    lang: json_lang(&m.rel),
                    symbol: None,
                    score: 1.0,
                    signals: SignalsJson {
                        lex_rank: None,
                        vec_rank: None,
                        rrf_score: 1.0,
                    },
                    preview: m.text.clone(),
                    model: "none",
                };
                println!("{}", serde_json::to_string(&line)?);
            }
            Format::Text => {
                if opts.files_only {
                    println!("{display}");
                    printed += 1;
                    continue;
                }
                let current = (m.line, m.text.clone());
                let all = m
                    .before
                    .iter()
                    .chain(std::iter::once(&current))
                    .chain(m.after.iter());
                for (no, text) in all {
                    if opts.line_numbers && opts.filenames {
                        println!("{display}:{no}:{text}");
                    } else if opts.line_numbers {
                        println!("{no}:{text}");
                    } else if opts.filenames {
                        println!("{display}:{text}");
                    } else {
                        println!("{text}");
                    }
                }
            }
        }
        printed += 1;
    }
    Ok(printed)
}
