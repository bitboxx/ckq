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
//! `--sem` and `--hybrid` embed their query through a resident daemon (unix):
//! the first search starts it, later ones connect, `CKQ_IN_PROCESS=1` embeds
//! in the process instead. Indexing never uses it — it embeds documents in
//! the process that runs `--index`.
//!
//! A build with the `embedded-model` feature carries one GGUF inside the
//! binary (build.rs reads `CKQ_EMBED_GGUF` and `CKQ_EMBED_ALIAS`) and that
//! alias becomes the build's default model. `models::gguf_path` then writes
//! the bytes once to `~/.cache/ck/models/embedded/<blake3>.gguf` — temp
//! file, fsync, atomic rename — and trusts a right-size file thereafter, so
//! the one binary works offline on a fresh machine. A normal build has none
//! of this.
//!
//! Exit codes: 0 when something was found or a command succeeded, 1 for no
//! match ("No matches found" on stderr) and for errors (message on stderr).

mod chunk;
mod corpus;
#[cfg(unix)]
mod daemon;
mod embed;
mod grep;
mod index;
mod models;
mod output;
mod search;
mod store;
mod walk;

use clap::Parser;
use output::{Format, OutputOptions};
use std::path::PathBuf;
use std::process::ExitCode;

#[derive(Parser, Debug)]
#[command(
    name = "ckq",
    version = version_line(),
    about = "Multilingual search over a folder of text: BM25 and embeddings, fused"
)]
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

    // Daemon (unix); hidden: started by ckq itself, not by users
    /// Run as the query daemon for `--sem`/`--hybrid` (started by ckq itself).
    #[cfg(unix)]
    #[arg(long, hide = true, value_name = "ALIAS", group = "command")]
    query_daemon: Option<String>,
    /// Print the query daemon's socket path for ALIAS and exit.
    #[cfg(unix)]
    #[arg(long, hide = true, value_name = "ALIAS", group = "command")]
    daemon_socket: Option<String>,
}

fn main() -> ExitCode {
    // A closed pipe must end us the way it ends grep, not with a panic.
    #[cfg(unix)]
    unsafe {
        libc::signal(libc::SIGPIPE, libc::SIG_DFL);
    }
    let cli = Cli::parse();
    let code = match run(cli) {
        Ok(code) => code,
        Err(e) => {
            eprintln!("Error: {e:#}");
            ExitCode::from(1)
        }
    };
    // Last thing before exit: drop the model in the order ggml requires.
    embed::shutdown();
    code
}

fn run(cli: Cli) -> anyhow::Result<ExitCode> {
    let walk_opts = walk::WalkOptions {
        hidden: cli.hidden,
        no_ckignore: cli.no_ckignore,
        no_ignore: cli.no_ignore,
    };

    // ---- Daemon ----------------------------------------------------------
    #[cfg(unix)]
    if let Some(alias) = &cli.query_daemon {
        daemon::serve(alias)?;
        return Ok(ExitCode::SUCCESS);
    }
    #[cfg(unix)]
    if let Some(alias) = &cli.daemon_socket {
        println!("{}", daemon::socket_path(alias).display());
        return Ok(ExitCode::SUCCESS);
    }

    // ---- Index commands -------------------------------------------------
    if cli.index {
        let targets = command_targets(&cli);
        for path in &targets {
            let corpus = corpus::Corpus::for_index(path)?;
            let report = index::update(
                &corpus,
                &index::IndexOptions {
                    model: cli.model.clone(),
                    walk: walk_opts.clone(),
                    quiet: cli.quiet,
                },
            )?;
            if !cli.quiet {
                eprintln!("{}", report.summary());
            }
        }
        return Ok(ExitCode::SUCCESS);
    }

    if let Some(file) = &cli.add {
        let corpus = corpus::Corpus::for_command(file)?;
        let Some(rel) = corpus.rel(file) else {
            anyhow::bail!("{} is not under {}", file.display(), corpus.root.display());
        };
        let report = index::add_file(
            &corpus,
            &rel,
            &index::IndexOptions {
                model: cli.model.clone(),
                walk: walk_opts.clone(),
                quiet: cli.quiet,
            },
        )?;
        if !cli.quiet {
            eprintln!("{}", report.summary());
        }
        return Ok(ExitCode::SUCCESS);
    }

    if cli.clean {
        for path in command_targets(&cli) {
            let corpus = corpus::Corpus::for_command(&path)?;
            index::clean(&corpus)?;
            if !cli.quiet {
                eprintln!("Index cleaned: {}", corpus.index_dir().display());
            }
        }
        return Ok(ExitCode::SUCCESS);
    }

    if cli.clean_orphans {
        // The store cannot hold orphans: each file is written and dropped in
        // one transaction. Accepted for compatibility, and it does what it
        // promises anyway by removing index entries for deleted files.
        for path in command_targets(&cli) {
            let corpus = corpus::Corpus::for_command(&path)?;
            if store::Store::open_read(&corpus)?.is_some() {
                let report = index::update(
                    &corpus,
                    &index::IndexOptions {
                        model: cli.model.clone(),
                        walk: walk_opts.clone(),
                        quiet: cli.quiet,
                    },
                )?;
                if !cli.quiet {
                    eprintln!("{}", report.summary());
                }
            }
        }
        return Ok(ExitCode::SUCCESS);
    }

    if cli.status || cli.status_verbose || cli.status_json {
        return status(&cli);
    }

    // ---- Search ----------------------------------------------------------
    let Some(pattern) = cli.pattern.clone() else {
        anyhow::bail!("a query is required; see ckq --help");
    };
    let paths: Vec<PathBuf> = if cli.paths.is_empty() {
        vec![PathBuf::from(".")]
    } else {
        cli.paths.clone()
    };
    let format = if cli.json {
        Format::Json
    } else if cli.jsonl {
        Format::Jsonl
    } else {
        Format::Text
    };
    let out_opts = OutputOptions {
        format,
        snippet: !cli.no_snippet,
        scores: cli.scores,
        line_numbers: cli.line_number,
        files_only: cli.files_with_matches,
        filenames: false,
    };
    let mut found = 0usize;

    for path in &paths {
        let show_filenames = paths.len() > 1 || path.is_dir();
        let out_opts = OutputOptions {
            filenames: show_filenames,
            ..out_opts
        };

        if cli.sem || cli.lex || cli.hybrid {
            let (corpus, scope) = corpus::Corpus::for_search(path)?;
            let Some(store) = store::Store::open_read(&corpus)? else {
                anyhow::bail!(
                    "No ckq index found at {}; run `ckq --index` first",
                    corpus.root.display()
                );
            };
            if !cli.quiet {
                let stale = index::stale_count(&corpus, &walk_opts)?;
                if stale > 0 {
                    eprintln!(
                        "ckq: warning: index is stale ({stale} files added, changed or gone); \
                         run `ckq --index`"
                    );
                }
            }
            let meta = store.meta()?;
            let spec = models::resolve(&meta.model_repo)?;
            let query = search::Query {
                text: pattern.clone(),
                terms: cli.terms.clone(),
                topk: cli.topk.unwrap_or(10),
                threshold: cli.threshold,
                scope,
            };
            let hits = if cli.lex {
                search::lexical(&store, &corpus, &query)?
            } else {
                let mut embedder = query_embedder(spec)?;
                if cli.sem {
                    search::semantic(&store, &mut *embedder, &query, spec.threshold)?
                } else {
                    search::hybrid(&store, &corpus, &mut *embedder, &query, spec.threshold)?
                }
            };
            found += output::print_hits(&corpus, &hits, &out_opts)?;
        } else {
            let (corpus, scope) = corpus::Corpus::plain(path)?;
            let opts = grep::GrepOptions {
                pattern: pattern.clone(),
                ignore_case: cli.ignore_case,
                word: cli.word_regexp,
                fixed: cli.fixed_strings,
                context: cli.context.unwrap_or(0),
                files_only: cli.files_with_matches,
                limit: cli.topk,
                walk: walk_opts.clone(),
            };
            let mut matches = grep::grep(&corpus, &opts)?;
            // A file argument scopes the search to that file.
            matches.retain(|m| scope.contains(&m.rel));
            found += output::print_matches(&corpus, &matches, &out_opts)?;
        }
    }

    if found == 0 {
        eprintln!("No matches found");
        return Ok(ExitCode::from(1));
    }
    Ok(ExitCode::SUCCESS)
}

/// The paths an index command works on: the arguments, or `.` when none.
/// A command takes no query, so a lone positional argument is its path —
/// clap cannot know that and files it under `pattern`.
fn command_targets(cli: &Cli) -> Vec<PathBuf> {
    if !cli.paths.is_empty() {
        cli.paths.clone()
    } else if let Some(path) = &cli.pattern {
        vec![PathBuf::from(path)]
    } else {
        vec![PathBuf::from(".")]
    }
}

/// The `--version` line: the crate version, plus the model a
/// `embedded-model` build carries inside the binary.
#[cfg(feature = "embedded-model")]
fn version_line() -> &'static str {
    concat!(
        env!("CARGO_PKG_VERSION"),
        " (embedded model: ",
        env!("CKQ_EMBED_ALIAS"),
        ")"
    )
}

#[cfg(not(feature = "embedded-model"))]
fn version_line() -> &'static str {
    env!("CARGO_PKG_VERSION")
}

/// `--status`, `--status-verbose` (stderr) and `--status-json` (stdout).
fn status(cli: &Cli) -> anyhow::Result<ExitCode> {
    if let Some(alias) = models::embedded_alias() {
        eprintln!("Embedded model: {alias}");
    }
    for path in command_targets(cli) {
        let corpus = corpus::Corpus::for_command(&path)?;
        let store = store::Store::open_read(&corpus)?;
        let stats = match &store {
            Some(s) => s.stats()?,
            None => Default::default(),
        };
        let meta = match &store {
            Some(s) => Some(s.meta()?),
            None => None,
        };
        if cli.status_json {
            let json = serde_json::json!({
                "path": corpus.root.display().to_string(),
                "embedded_model": models::embedded_alias(),
                "index_exists": store.is_some(),
                "total_files": stats.files,
                "total_chunks": stats.chunks,
                "embedded_chunks": stats.embedded_chunks,
                "total_size_bytes": stats.source_bytes,
                "index_size_bytes": stats.db_bytes,
                "model": meta.as_ref().map(|m| serde_json::json!({
                    "name": m.model_repo,
                    "alias": m.model_alias,
                    "dimensions": m.dims,
                })),
            });
            println!("{}", serde_json::to_string_pretty(&json)?);
            continue;
        }
        if stats.files == 0 {
            eprintln!("No index found at {}", corpus.root.display());
            eprintln!("Run `ckq --index .` to create an index");
            continue;
        }
        eprintln!("Index location: {}", corpus.index_dir().display());
        eprintln!("Files indexed: {}", stats.files);
        eprintln!("  Total chunks: {}", stats.chunks);
        eprintln!("  Embedded chunks: {}", stats.embedded_chunks);
        if let Some(m) = &meta {
            eprintln!(
                "  Model: {} (alias '{}', {} dims)",
                m.model_repo, m.model_alias, m.dims
            );
        }
        if cli.status_verbose {
            let source_mb = stats.source_bytes as f64 / (1024.0 * 1024.0);
            let index_mb = stats.db_bytes as f64 / (1024.0 * 1024.0);
            eprintln!("  Source size: {source_mb:.1} MB");
            eprintln!("  Index size: {index_mb:.1} MB");
            if stats.source_bytes > 0 {
                let ratio = stats.db_bytes as f64 / stats.source_bytes as f64;
                eprintln!("  Compression: {:.1}x ({:.1}%)", 1.0 / ratio, ratio * 100.0);
            }
        }
    }
    Ok(ExitCode::SUCCESS)
}

/// The embedder a `--sem`/`--hybrid` search embeds its query with: the query
/// daemon's (unix), which applies the query prompt on its side in `embed`, or
/// the model loaded in this process when there is no daemon to reach or
/// `CKQ_IN_PROCESS=1` asks for none.
#[cfg(unix)]
fn query_embedder(spec: &'static models::ModelSpec) -> anyhow::Result<Box<dyn embed::Embedder>> {
    daemon::embedder(spec)
}

#[cfg(not(unix))]
fn query_embedder(spec: &'static models::ModelSpec) -> anyhow::Result<Box<dyn embed::Embedder>> {
    Ok(Box::new(embed::LlamaEmbedder::load(
        spec,
        embed::query_device(),
    )?))
}
