//! End-to-end tests for scope and path arguments: what a file or directory
//! argument searches, and the paths results print. Regex-mode tests here are
//! offline; the ranked ones need the default model and carry the `e2e:`
//! ignore, so `cargo test` stays quick and `cargo e2e` runs everything.

mod common;

use common::{Corpus, parse_jsonl, run_exit, run_ok, status_json};
use serial_test::serial;
use std::path::Path;

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
    let out = run_ok(corpus.rogo().args(args).args(["--jsonl", "-q", arg]));
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

// ---------------------------------------------------------------------------
// Ranked modes: printed paths count from the argument (b-NEW)
// ---------------------------------------------------------------------------

#[test]
#[serial]
#[ignore = "e2e: needs the default model; run with --include-ignored"]
fn ranked_paths_print_from_the_argument_down() {
    let corpus = needled();
    corpus.index();

    // A directory argument: real paths, relative the way the argument was
    // given — not `notes/notes/in.md`, which does not exist.
    let hits = hits_in(&corpus, &["--lex", "needle"], "notes");
    assert!(!hits.is_empty());
    for h in &hits {
        assert!(
            h.path == Path::new("notes/in.md") || h.path == Path::new("notes/in2.md"),
            "printed {}",
            h.path.display()
        );
        assert!(
            corpus.path().join(&h.path).is_file(),
            "printed path does not exist: {}",
            h.path.display()
        );
    }

    // A file argument prints as the file itself.
    let hits = hits_in(&corpus, &["--lex", "needle"], "notes/in.md");
    assert!(!hits.is_empty());
    assert!(hits.iter().all(|h| h.path == Path::new("notes/in.md")));

    // An absolute argument prints absolute paths, the way regex mode does.
    let abs = corpus.path().join("notes");
    let hits = hits_in(&corpus, &["--lex", "needle"], &abs.to_string_lossy());
    assert!(!hits.is_empty());
    assert!(
        hits.iter()
            .all(|h| h.path == abs.join("in.md") || h.path == abs.join("in2.md"))
    );
}

// ---------------------------------------------------------------------------
// Regex mode: --topk applies after the scope (b-F1)
// ---------------------------------------------------------------------------

#[test]
fn regex_topk_with_a_file_argument_returns_that_files_matches() {
    let corpus = Corpus::new();
    corpus.write("a.md", "match one\nmatch two\n");
    corpus.write("b.md", "match three\n");

    // The limit used to stop the walk at a.md, whose matches the scope then
    // dropped: "No matches found", exit 1.
    let out = run_ok(corpus.rogo().args(["-F", "--topk", "1", "match", "b.md"]));
    let stdout = String::from_utf8_lossy(&out.stdout).into_owned();
    let lines: Vec<&str> = stdout.lines().collect();
    assert_eq!(lines, vec!["match three"]);

    // The limit still applies within the scope.
    let out = run_ok(corpus.rogo().args(["-F", "--topk", "1", "match", "."]));
    let stdout = String::from_utf8_lossy(&out.stdout).into_owned();
    let lines: Vec<&str> = stdout.lines().collect();
    assert_eq!(lines, vec!["./a.md:", "match one"]);
}

// ---------------------------------------------------------------------------
// --add of a file the walk excludes is refused (a-F4)
// ---------------------------------------------------------------------------

#[test]
#[serial]
#[ignore = "e2e: needs the default model; run with --include-ignored"]
fn add_refuses_a_file_the_walk_excludes() {
    let corpus = Corpus::new();
    corpus.write("keep.md", "plain text\n");
    corpus.index(); // writes the default .rogoignore, which ignores *.json
    corpus.write("data.json", "{\"a\": 1}\n");

    // Used to succeed ("1 added") and then warn stale on every search,
    // until the next --index silently removed the file again.
    let (code, _, stderr) = run_exit(corpus.rogo().args(["--add", "data.json"]));
    assert_eq!(code, Some(1), "{stderr}");
    assert!(stderr.contains("excluded from the index"), "{stderr}");
    assert_eq!(
        status_json(corpus.path())["total_files"],
        serde_json::json!(1),
        "nothing may be added"
    );
    run_ok(
        corpus
            .rogo()
            .args(["--no-rogoignore", "--add", "data.json"]),
    );
    assert_eq!(
        status_json(corpus.path())["total_files"],
        serde_json::json!(2)
    );
}

// ---------------------------------------------------------------------------
// --threshold is validated at parse time (b-F6)
// ---------------------------------------------------------------------------

#[test]
fn threshold_nan_and_out_of_range_are_rejected_at_parse_time() {
    // NaN used to parse, make every score comparison false, and end in a
    // bare "No matches found"; 5.0 behaved the same, and on --lex NaN
    // silently disabled the filter instead.
    for bad in ["nan", "NaN", "5", "-1.5"] {
        let corpus = Corpus::new();
        let mut cmd = corpus.rogo();
        cmd.args(["--lex"])
            .arg(format!("--threshold={bad}"))
            .args(["query", "."]);
        let (code, _, stderr) = run_exit(&mut cmd);
        assert_eq!(code, Some(1), "{bad}: {stderr}");
        assert!(stderr.contains("--threshold must be"), "{bad}: {stderr}");
    }

    // The ends of the range are accepted (finding nothing is fine).
    let (_code, _, stderr) =
        run_exit(
            Corpus::new()
                .rogo()
                .args(["--lex", "--threshold=-1.0", "query", "."]),
        );
    assert!(stderr.contains("No rogo index found"), "{stderr}");
}

// ---------------------------------------------------------------------------
// Flags that do nothing in the chosen mode say so (b-F7)
// ---------------------------------------------------------------------------

fn stderr_of(out: &std::process::Output) -> String {
    String::from_utf8_lossy(&out.stderr).into_owned()
}

#[test]
fn regex_mode_names_an_inert_term_on_stderr() {
    let corpus = Corpus::new();
    corpus.write("a.md", "needle here\n");
    let out = run_ok(corpus.rogo().args(["--term", "zzzz", "needle", "."]));
    assert!(
        stderr_of(&out).contains("rogo: warning: --term has no effect in regex mode"),
        "{}",
        stderr_of(&out)
    );
}

#[test]
#[serial]
#[ignore = "e2e: needs the default model; run with --include-ignored"]
fn ranked_modes_name_the_grep_flags_they_ignore() {
    let corpus = needled();
    corpus.index();
    let out = run_ok(
        corpus
            .rogo()
            .args(["--lex", "-i", "-w", "-F", "-C", "2", "needle", "."]),
    );
    let err = stderr_of(&out);
    for flag in ["-i", "-w", "-F", "-C"] {
        assert!(
            err.contains(&format!("{flag} has no effect in --lex mode")),
            "{err}"
        );
    }

    // --term has no lexical side under --sem.
    let out = run_ok(
        corpus
            .rogo()
            .args(["--sem", "--term", "zzzz", "needle", "."]),
    );
    assert!(
        stderr_of(&out).contains("rogo: warning: --term has no effect in --sem mode"),
        "{}",
        stderr_of(&out)
    );
}

#[test]
#[serial]
#[ignore = "e2e: needs the default model; run with --include-ignored"]
fn the_flags_vq_passes_never_warn() {
    let corpus = needled();
    corpus.index();
    // A wrapper script's flag sets: -q --jsonl --no-snippet with --lex
    // --term, --hybrid --term --limit, --sem, and -n for regex mode. None
    // of these may produce a warning (shown here without -q, which also
    // silences warnings, to prove the flags themselves are clean).
    for args in [
        vec![
            "--lex",
            "--jsonl",
            "--no-snippet",
            "--term",
            "needle",
            "--limit",
            "5",
            "needle",
            ".",
        ],
        vec![
            "--hybrid",
            "--jsonl",
            "--no-snippet",
            "--term",
            "needle",
            "--limit",
            "5",
            "needle",
            ".",
        ],
        vec!["--sem", "--jsonl", "--no-snippet", "needle", "."],
    ] {
        let out = run_ok(corpus.rogo().args(&args));
        assert!(
            !stderr_of(&out).contains("rogo: warning:"),
            "{:?}: {}",
            args,
            stderr_of(&out)
        );
    }

    let out = run_ok(corpus.rogo().args(["-n", "needle", "."]));
    assert!(!stderr_of(&out).contains("rogo: warning:"));
}
