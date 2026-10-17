//! End-to-end tests for scope and path arguments: what a file or directory
//! argument searches, and the paths results print. Regex-mode tests here are
//! offline; the ranked ones need the default model and carry the `e2e:`
//! ignore, so `cargo test` stays quick and `cargo e2e` runs everything.

mod common;

use common::{Corpus, parse_jsonl, run_ok};
use serial_test::serial;

/// A corpus whose every file matches `needle`, so any scope leak shows up as
/// a foreign file in the results.
fn needled() -> Corpus {
    let c = Corpus::new();
    c.write("top.md", "needle at the top of the corpus\n");
    c.write("other.md", "needle in another root file\n");
    c.write("notes/in.md", "needle inside the notes directory\n");
    c.write("notes/in2.md", "needle inside a sibling file\n");
    c.write("notes2.md", "needle in a prefix-sharing sibling\n");
    c
}

/// The `--jsonl` hits of a search whose last argument is `arg`, not `.`.
fn hits_in(corpus: &Corpus, args: &[&str], arg: &str) -> Vec<common::Hit> {
    let out = run_ok(corpus.ckq().args(args).args(["--jsonl", "-q", arg]));
    parse_jsonl(&out.stdout)
}

/// The file names of the hits, deduplicated, sorted: the set of documents a
/// search actually returned.
fn documents(hits: &[common::Hit]) -> Vec<String> {
    let mut names: Vec<String> = hits.iter().map(|h| h.file_name()).collect();
    names.sort();
    names.dedup();
    names
}

// ---------------------------------------------------------------------------
// Ranked modes: a file argument scopes the search to that file (b-F2)
// ---------------------------------------------------------------------------

#[test]
#[serial]
#[ignore = "e2e: needs the default model; run with --include-ignored"]
fn a_file_argument_scopes_a_lexical_search_to_that_file() {
    let corpus = needled();
    corpus.index();

    // The file's parent is the index root: this used to widen to the whole
    // corpus, printing hits with no file names at all.
    let hits = hits_in(&corpus, &["--lex", "needle"], "top.md");
    assert!(!hits.is_empty(), "top.md must match the query");
    assert_eq!(documents(&hits), vec!["top.md".to_string()]);

    // Below the root: the file, not its directory and not its siblings.
    let hits = hits_in(&corpus, &["--lex", "needle"], "notes/in.md");
    assert!(!hits.is_empty(), "notes/in.md must match the query");
    assert_eq!(documents(&hits), vec!["in.md".to_string()]);
}

#[test]
#[serial]
#[ignore = "e2e: needs the default model; run with --include-ignored"]
fn a_directory_argument_does_not_leak_prefix_sharing_siblings() {
    let corpus = needled();
    corpus.index();

    let hits = hits_in(&corpus, &["--lex", "needle"], "notes");
    let docs = documents(&hits);
    assert_eq!(docs, vec!["in.md".to_string(), "in2.md".to_string()]);
}
