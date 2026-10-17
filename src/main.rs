//! ckq: multilingual search over a folder of text.
//!
//! Three ranked methods over one index (`--lex` BM25, `--sem` embeddings,
//! `--hybrid` both fused) and a plain regex mode that needs no index.
//!
//! The invariants this design exists for:
//! 1. A document is its root-relative path (`corpus::RelPath`). Both ranked
//!    methods return it, fusion compares it, output prints it through
//!    `Corpus::display`. The working directory never matters.
//! 2. The model's prompts are applied in `embed` and nowhere else.
//! 3. Search never writes the index. `--index`, `--add` and `--clean` do.
//! 4. One store: `.ck/index.db`.
//! 5. stdout carries results only.
//!
//! Exit codes: 0 when something was found or a command succeeded, 1 for no
//! match ("No matches found" on stderr) and for errors (message on stderr).

mod chunk;
mod corpus;
mod embed;
mod grep;
mod index;
mod models;
mod output;
mod search;
mod store;
mod walk;

use clap::Parser;
use std::path::PathBuf;
use std::process::ExitCode;

#[derive(Parser, Debug)]
#[command(name = "ckq", version, about = "Multilingual search over a folder of text: BM25 and embeddings, fused")]
struct Cli {
    /// The query: a regex in regex mode, text for --lex, --sem and --hybrid.
    pattern: Option<String>,
    /// Files or directories to search. Default: the current directory.
    paths: Vec<PathBuf>,

    // Mode
    /// Rank by meaning, with the embedding model.
    #[arg(long, group = "mode")]
    sem: bool,
    /// Rank by BM25 over file text and path.
    #[arg(long, group = "mode")]
    lex: bool,
    /// Both, fused by reciprocal rank, one result per file.
    #[arg(long, group = "mode")]
    hybrid: bool,
    /// Plain regex search (the default).
    #[arg(long, group = "mode")]
    regex: bool,

    // Ranked search
    /// A keyword for the lexical method, matched as a phrase; repeatable. Replaces the query there.
    #[arg(long = "term", value_name = "TERM")]
    terms: Vec<String>,
    /// Number of results (default 10 for ranked modes).
    #[arg(long, visible_alias = "limit", value_name = "N")]
    topk: Option<usize>,
    /// Minimum semantic score, 0.0 to 1.0 (default: the model's own).
    #[arg(long)]
    threshold: Option<f32>,
    /// Show scores in text output.
    #[arg(long)]
    scores: bool,

    // Regex mode
    #[arg(short = 'n', long = "line-number")]
    line_number: bool,
    #[arg(short = 'i', long = "ignore-case")]
    ignore_case: bool,
    #[arg(short = 'w', long = "word-regexp")]
    word_regexp: bool,
    #[arg(short = 'F', long = "fixed-strings")]
    fixed_strings: bool,
    #[arg(short = 'l', long = "files-with-matches")]
    files_with_matches: bool,
    #[arg(short = 'C', long = "context", value_name = "N")]
    context: Option<usize>,
    /// Accepted for grep compatibility; directories are always searched recursively.
    #[arg(short = 'r', long = "recursive")]
    recursive: bool,

    // Output
    #[arg(long, group = "format")]
    json: bool,
    #[arg(long, group = "format")]
    jsonl: bool,
    /// Leave the snippet out of JSON and JSONL results.
    #[arg(long = "no-snippet")]
    no_snippet: bool,
    /// No progress or banners on stderr.
    #[arg(short = 'q', long)]
    quiet: bool,

    // Which files
    /// Include dot-files and dot-directories.
    #[arg(long)]
    hidden: bool,
    /// Do not read .gitignore files.
    #[arg(long = "no-ignore")]
    no_ignore: bool,
    /// Do not read .ckignore files.
    #[arg(long = "no-ckignore")]
    no_ckignore: bool,

    // Index
    /// Build or update the index of the given directory.
    #[arg(long, group = "command")]
    index: bool,
    /// Index or re-index one file.
    #[arg(long, value_name = "FILE", group = "command")]
    add: Option<PathBuf>,
    /// Remove the index.
    #[arg(long, group = "command")]
    clean: bool,
    /// Accepted for compatibility; the index cannot hold orphans.
    #[arg(long = "clean-orphans", group = "command")]
    clean_orphans: bool,
    #[arg(long, group = "command")]
    status: bool,
    #[arg(long = "status-verbose", group = "command")]
    status_verbose: bool,
    #[arg(long = "status-json", group = "command")]
    status_json: bool,
    /// Embedding model alias for --index (default: the index's own, else gemma-q4).
    #[arg(long, value_name = "ALIAS")]
    model: Option<String>,
}

fn main() -> ExitCode {
    let cli = Cli::parse();
    match run(cli) {
        Ok(code) => code,
        Err(e) => {
            eprintln!("Error: {e:#}");
            ExitCode::from(1)
        }
    }
}

fn run(cli: Cli) -> anyhow::Result<ExitCode> {
    todo!("dispatch: index commands, then regex mode, then ranked modes; warn on stale index; print")
}
