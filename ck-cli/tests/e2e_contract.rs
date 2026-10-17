//! The output contract of the `ckq` binary: what each search mode promises in
//! `--json` and `--jsonl`, what the grep flags (-l, -n, -C, -i, -w, -F, --topk)
//! do, which exit code and message each failure takes, and that the MCP server
//! speaks JSON-RPC over stdio when asked to `--serve`.
//!
//! These tests pin behaviour, not implementation: a change to a field name, a
//! span that no longer covers the matched bytes, or a new exit code for a
//! missing path is a breaking change for every script and agent that reads this
//! output, and must show up here before it ships.

mod common;

use common::{Corpus, Hit};
use serde_json::{Value, json};
use serial_test::serial;
use std::fs;
use std::io::{BufRead, BufReader, Write};
use std::path::{Path, PathBuf};
use std::process::{Child, ChildStdin, Stdio};
use std::sync::mpsc::{self, RecvTimeoutError};
use std::time::{Duration, Instant};

// ---------------------------------------------------------------------------
// Output contract: --jsonl
// ---------------------------------------------------------------------------

/// The four integer fields of a `span` object, failing with the span if any is
/// missing or not an unsigned integer.
fn span_fields(span: &Value) -> (u64, u64, u64, u64) {
    let get = |key: &str| {
        span.get(key)
            .and_then(Value::as_u64)
            .unwrap_or_else(|| panic!("span.{key} missing or not an integer: {span}"))
    };
    (
        get("byte_start"),
        get("byte_end"),
        get("line_start"),
        get("line_end"),
    )
}

/// A result path as a readable file: regex results carry absolute paths, the
/// semantic method relative ones, and both must resolve against the corpus.
fn resolve(corpus: &Corpus, path: &Path) -> PathBuf {
    if path.is_absolute() {
        path.to_path_buf()
    } else {
        corpus.path().join(path)
    }
}

/// Stdout as trimmed, non-empty lines. The temporary Cow must not outlive the
/// statement, so the lines are owned.
fn stdout_lines(out: &std::process::Output) -> Vec<String> {
    String::from_utf8_lossy(&out.stdout)
        .lines()
        .map(str::trim)
        .filter(|l| !l.is_empty())
        .map(ToOwned::to_owned)
        .collect()
}

/// Every regex result's span must be exactly the matched bytes in the file:
/// that is what makes `byte_start`/`byte_end` usable to jump to the match.
#[test]
fn regex_jsonl_span_is_the_match_bytes() {
    let corpus = Corpus::new();
    corpus.write(
        "a.txt",
        "alpha needle one\nsecond line\nthird line with needle\n",
    );
    corpus.write(
        "code/needle_finder.rs",
        "fn find_needle() -> usize {\n    1\n}\n",
    );

    let hits = corpus.search(&["needle"]);
    assert_eq!(hits.len(), 3, "two matches in a.txt, one in the .rs file");

    let mut a_lines = Vec::new();
    for hit in &hits {
        let (byte_start, byte_end, line_start, line_end) = span_fields(&hit.raw["span"]);
        assert!(byte_end > byte_start, "empty span: {}", hit.raw);
        assert!(line_end >= line_start, "inverted line span: {}", hit.raw);

        let bytes = fs::read(resolve(&corpus, &hit.path)).expect("result file readable");
        assert_eq!(
            &bytes[byte_start as usize..byte_end as usize],
            b"needle".as_slice(),
            "the file's bytes at the span must be the match: {}",
            hit.raw
        );

        // The rest of the promised shape.
        assert!(
            hit.raw.get("language").is_some(),
            "language key: {}",
            hit.raw
        );
        assert_eq!(hit.score, Some(1.0), "regex scores are 1.0: {}", hit.raw);
        let snippet = hit
            .raw
            .get("snippet")
            .and_then(Value::as_str)
            .expect("snippet present by default");
        assert!(
            snippet.contains("needle"),
            "snippet covers the match: {}",
            hit.raw
        );
        assert!(hit.raw.get("signals").is_none(), "regex carries no signals");

        if hit.file_name() == "a.txt" {
            a_lines.push((line_start, line_end));
        }
    }
    a_lines.sort();
    assert_eq!(
        a_lines,
        vec![(1, 1), (3, 3)],
        "line numbers of the two a.txt matches"
    );

    let rust = hits
        .iter()
        .find(|h| h.file_name() == "needle_finder.rs")
        .expect("the .rs file is a result");
    assert_eq!(rust.raw["language"].as_str(), Some("rust"));
    assert_eq!(span_fields(&rust.raw["span"]).2, 1);

    // The same search without snippets: the field must be gone entirely, and
    // the span must still be usable to find the bytes.
    let bare = corpus.search(&["--no-snippet", "needle"]);
    assert_eq!(bare.len(), hits.len(), "--no-snippet drops no results");
    for hit in &bare {
        assert!(
            hit.raw.get("snippet").is_none(),
            "--no-snippet must remove the field entirely: {}",
            hit.raw
        );
        let (byte_start, byte_end, _, _) = span_fields(&hit.raw["span"]);
        let bytes = fs::read(resolve(&corpus, &hit.path)).expect("result file readable");
        assert_eq!(
            &bytes[byte_start as usize..byte_end as usize],
            b"needle".as_slice(),
            "span still names the match bytes: {}",
            hit.raw
        );
        assert!(hit.score.is_some(), "score stays: {}", hit.raw);
    }
}

/// `--json` is a different schema from `--jsonl` (`file`/`lang`/`preview`/
/// `model`, with synthesized null signals) and that difference is load-bearing
/// for its consumers, so it is pinned separately.
#[test]
fn regex_json_schema_fields() {
    let corpus = Corpus::new();
    corpus.write(
        "a.txt",
        "alpha needle one\nsecond line\nthird line with needle\n",
    );
    corpus.write(
        "code/needle_finder.rs",
        "fn find_needle() -> usize {\n    1\n}\n",
    );

    let out = common::run_ok(corpus.ckq().args(["-q", "needle", "--json", "."]));
    let lines = stdout_lines(&out);
    assert_eq!(lines.len(), 3, "one JSON object per match");

    for line in &lines {
        let v: Value =
            serde_json::from_str(line).unwrap_or_else(|e| panic!("bad JSON: {e}: {line}"));
        let file = v["file"]
            .as_str()
            .unwrap_or_else(|| panic!("file field: {v}"));
        assert!(
            file.ends_with("a.txt") || file.ends_with("needle_finder.rs"),
            "file names the match: {v}"
        );

        let (byte_start, byte_end, _, _) = span_fields(&v["span"]);
        let path = if Path::new(file).is_absolute() {
            PathBuf::from(file)
        } else {
            corpus.path().join(file)
        };
        let bytes = fs::read(&path).expect("result file readable");
        assert_eq!(
            &bytes[byte_start as usize..byte_end as usize],
            b"needle".as_slice(),
            "span bytes contain the match: {v}"
        );

        assert!(v.get("lang").is_some(), "lang key: {v}");
        assert_eq!(v["score"].as_f64(), Some(1.0), "regex score: {v}");
        assert_eq!(
            v["model"].as_str(),
            Some("none"),
            "no model in regex mode: {v}"
        );
        let preview = v["preview"]
            .as_str()
            .unwrap_or_else(|| panic!("preview: {v}"));
        assert!(preview.contains("needle"), "preview covers the match: {v}");

        let signals = &v["signals"];
        assert!(
            signals["lex_rank"].is_null() && signals["vec_rank"].is_null(),
            "no method ranks in regex mode: {v}"
        );
        assert_eq!(
            signals["rrf_score"].as_f64(),
            Some(1.0),
            "rrf mirrors the score: {v}"
        );
    }

    let rust_line = lines
        .iter()
        .find(|l| l.contains("needle_finder.rs"))
        .expect("the .rs file is a result");
    let v: Value = serde_json::from_str(rust_line).unwrap();
    // `--json` serializes the enum variant ("Rust") where `--jsonl` uses the
    // display name ("rust"); both are pinned as-is so a unification shows up.
    assert_eq!(v["lang"].as_str(), Some("Rust"));
}

/// Three short documents, one per topic, with Spanish among them so the
/// semantic modes are tested across languages, not just English.
fn semantic_corpus() -> Corpus {
    let corpus = Corpus::new();
    corpus.write(
        "telescope.md",
        "The lighthouse keeper tracks incoming ships with a brass telescope.\n",
    );
    corpus.write(
        "tomates.md",
        "El jardinero riega los tomates cada manana antes del calor.\n",
    );
    corpus.write(
        "croissant.md",
        "A pastry chef folds cold butter into croissant dough before dawn.\n",
    );
    corpus
}

/// `--sem --jsonl`: path, span, language, score on every hit, snippet by
/// default and gone with `--no-snippet`, and no `signals` object.
#[test]
#[serial]
#[ignore = "e2e: needs the default model; run with --include-ignored"]
fn sem_jsonl_contract_and_snippet_toggle() {
    let corpus = semantic_corpus();
    corpus.index();

    let hits = corpus.search(&["--sem", "--topk", "3", "watering the tomato garden"]);
    assert!(
        !hits.is_empty(),
        "the tomato document must score above threshold"
    );
    assert_eq!(
        hits[0].file_name(),
        "tomates.md",
        "the Spanish tomato note is the best match, hits: {:?}",
        hits.iter().map(Hit::file_name).collect::<Vec<_>>()
    );

    for hit in &hits {
        let (byte_start, byte_end, line_start, line_end) = span_fields(&hit.raw["span"]);
        assert!(
            byte_end >= byte_start && line_end >= line_start,
            "span: {}",
            hit.raw
        );
        let score = hit.score.expect("score field");
        assert!((0.0..=1.0).contains(&score), "score in range: {}", hit.raw);
        assert_eq!(
            hit.raw["language"].as_str(),
            Some("markdown"),
            "{}",
            hit.raw
        );
        assert!(
            hit.raw.get("signals").is_none(),
            "--sem carries no signals: {}",
            hit.raw
        );
        let snippet = hit
            .raw
            .get("snippet")
            .and_then(Value::as_str)
            .expect("snippet present by default");
        assert!(
            !snippet.trim().is_empty(),
            "snippet is not blank: {}",
            hit.raw
        );
    }

    let bare = corpus.search(&[
        "--sem",
        "--topk",
        "3",
        "--no-snippet",
        "watering the tomato garden",
    ]);
    assert_eq!(bare.len(), hits.len(), "--no-snippet drops no results");
    for hit in &bare {
        assert!(
            hit.raw.get("snippet").is_none(),
            "--no-snippet must remove the field entirely: {}",
            hit.raw
        );
        assert!(
            hit.score.is_some() && hit.raw.get("span").is_some(),
            "rest of the shape stays"
        );
    }
}

/// `--lex --jsonl`: one result per matching file, with a score and no signals.
#[test]
#[serial]
#[ignore = "e2e: needs the default model; run with --include-ignored"]
fn lex_jsonl_contract() {
    let corpus = semantic_corpus();
    corpus.index();

    let hits = corpus.search(&["--lex", "tomates"]);
    assert_eq!(
        hits.len(),
        1,
        "only the tomato note matches: {:?}",
        hits.iter().map(Hit::file_name).collect::<Vec<_>>()
    );
    assert_eq!(hits[0].file_name(), "tomates.md");
    assert_eq!(
        hits[0].score,
        Some(1.0),
        "a full lexical match scores 1.0: {}",
        hits[0].raw
    );
    assert_eq!(hits[0].raw["language"].as_str(), Some("markdown"));
    span_fields(&hits[0].raw["span"]);
    assert!(
        hits[0].raw.get("signals").is_none(),
        "--lex carries no signals (only --hybrid does): {}",
        hits[0].raw
    );
}

/// `--hybrid --jsonl`: every hit carries `signals` with both method ranks
/// (1-based or null) and an `rrf_score` equal to the hit's score.
#[test]
#[serial]
#[ignore = "e2e: needs the default model; run with --include-ignored"]
fn hybrid_jsonl_signals_contract() {
    let corpus = semantic_corpus();
    corpus.index();

    let hits = corpus.search(&[
        "--hybrid",
        "--term",
        "tomates",
        "--topk",
        "5",
        "watering the tomato garden",
    ]);
    assert!(
        hits.iter().any(|h| h.file_name() == "tomates.md"),
        "the document both methods find is in the results"
    );

    for hit in &hits {
        let signals = hit
            .raw
            .get("signals")
            .unwrap_or_else(|| panic!("hybrid results carry signals: {}", hit.raw));
        let rrf = signals["rrf_score"]
            .as_f64()
            .unwrap_or_else(|| panic!("rrf_score is a number: {}", hit.raw));
        let score = hit.score.expect("score field");
        assert!(
            (rrf - score).abs() < 1e-6,
            "score is the fused score: {}",
            hit.raw
        );
        for key in ["lex_rank", "vec_rank"] {
            if !signals[key].is_null() {
                assert!(
                    signals[key].as_u64().unwrap() >= 1,
                    "ranks are 1-based: {}",
                    hit.raw
                );
            }
        }
    }
    assert!(
        hits.iter().any(|h| h.lex_rank.is_some()),
        "the lexical method contributed"
    );
    assert!(
        hits.iter().any(|h| h.vec_rank.is_some()),
        "the semantic method contributed"
    );
}

/// `--json` for the model-backed modes keeps its own schema (`file`, `lang`,
/// `preview`, `model`, and a synthesized `signals` object even where --jsonl
/// would omit it). One corpus, three modes.
#[test]
#[serial]
#[ignore = "e2e: needs the default model; run with --include-ignored"]
fn sem_lex_hybrid_json_schema_fields() {
    let corpus = semantic_corpus();
    corpus.index();

    let json_lines = |args: &[&str]| -> Vec<Value> {
        let out = common::run_ok(corpus.ckq().args(args));
        stdout_lines(&out)
            .iter()
            .map(|l| serde_json::from_str(l).unwrap_or_else(|e| panic!("bad JSON ({e}): {l}")))
            .collect()
    };

    // --sem: one object per result, first is the tomato note.
    let sem = json_lines(&[
        "-q",
        "--sem",
        "--topk",
        "3",
        "--json",
        "watering the tomato garden",
        ".",
    ]);
    assert!(
        !sem.is_empty(),
        "the query matches the corpus above threshold"
    );
    assert!(
        sem[0]["file"]
            .as_str()
            .is_some_and(|f| f.ends_with("tomates.md")),
        "best semantic hit is the tomato note: {}",
        sem[0]
    );
    for v in &sem {
        span_fields(&v["span"]);
        // `--json` serializes the enum variant ("Markdown"), not the display
        // name --jsonl uses ("markdown").
        assert_eq!(v["lang"].as_str(), Some("Markdown"), "{v}");
        assert_eq!(v["model"].as_str(), Some("none"), "model is a literal: {v}");
        assert!(
            v["preview"].as_str().is_some_and(|p| !p.trim().is_empty()),
            "preview present: {v}"
        );
        let score = v["score"].as_f64().expect("score");
        assert!((0.0..=1.0).contains(&score), "{v}");
        let signals = &v["signals"];
        assert!(
            signals["lex_rank"].is_null() && signals["vec_rank"].is_null(),
            "--sem has no method ranks, but the object is synthesized: {v}"
        );
        assert_eq!(
            signals["rrf_score"].as_f64(),
            Some(score),
            "rrf mirrors the score: {v}"
        );
    }

    // --lex: the same schema on the one file that matches the term.
    let lex = json_lines(&["-q", "--lex", "--json", "tomates", "."]);
    assert_eq!(lex.len(), 1, "one matching file: {lex:?}");
    assert!(
        lex[0]["file"]
            .as_str()
            .is_some_and(|f| f.ends_with("tomates.md"))
    );
    assert_eq!(lex[0]["lang"].as_str(), Some("Markdown"));
    assert_eq!(lex[0]["score"].as_f64(), Some(1.0));
    assert!(lex[0]["signals"]["lex_rank"].is_null());

    // --hybrid: real ranks land in the same signals object.
    let hybrid = json_lines(&[
        "-q",
        "--hybrid",
        "--term",
        "tomates",
        "--topk",
        "5",
        "--json",
        "watering the tomato garden",
        ".",
    ]);
    assert!(
        hybrid.iter().any(|v| v["file"]
            .as_str()
            .is_some_and(|f| f.ends_with("tomates.md"))),
        "the tomato note is among the hybrid results: {hybrid:?}"
    );
    for v in &hybrid {
        let score = v["score"].as_f64().expect("score");
        assert_eq!(
            v["signals"]["rrf_score"].as_f64(),
            Some(score),
            "fused score in signals: {v}"
        );
    }
    assert!(
        hybrid
            .iter()
            .any(|v| v["signals"]["lex_rank"].as_u64().is_some()),
        "the lexical method's rank shows up: {hybrid:?}"
    );
}
/// The documented hybrid contract: one entry per file, and a file both
/// methods find carries both ranks. Corroboration must fuse, not duplicate.
#[test]
#[serial]
#[ignore = "e2e: needs the default model; run with --include-ignored"]
fn hybrid_jsonl_one_fused_result_per_file() {
    let corpus = semantic_corpus();
    corpus.index();

    let hits = corpus.search(&["--hybrid", "--topk", "10", "riegar los tomates"]);
    assert!(!hits.is_empty());

    let mut seen = Vec::new();
    for hit in &hits {
        let name = hit.file_name();
        assert!(
            !seen.contains(&name),
            "{name} appears more than once — the two methods must fuse into one entry: {:?}",
            hits.iter().map(Hit::file_name).collect::<Vec<_>>()
        );
        seen.push(name);
    }

    let top = &hits[0];
    assert_eq!(top.file_name(), "tomates.md");
    assert_eq!(
        top.lex_rank.and(top.vec_rank),
        Some(1),
        "a file both methods rank first has both ranks set to 1: {}",
        top.raw
    );
}

// ---------------------------------------------------------------------------
// Flags
// ---------------------------------------------------------------------------

/// `-l` prints each matching file exactly once, whatever its match count.
#[test]
fn files_with_matches_lists_each_file_once() {
    let corpus = Corpus::new();
    corpus.write("a.txt", "one needle\nanother needle\n");
    corpus.write("b.txt", "just one needle\n");
    corpus.write("c.txt", "no match in here\n");

    let out = common::run_ok(corpus.ckq().args(["-l", "needle", "."]));
    let mut names: Vec<String> = stdout_lines(&out)
        .iter()
        .map(|l| {
            Path::new(l)
                .file_name()
                .unwrap_or_else(|| panic!("not a path line: {l}"))
                .to_string_lossy()
                .into_owned()
        })
        .collect();
    names.sort();
    assert_eq!(
        names,
        vec!["a.txt".to_string(), "b.txt".to_string()],
        "a.txt has two matches but is listed once; c.txt not at all"
    );
}

/// `-n` prints the true 1-based line number of every match.
#[test]
fn line_numbers_are_the_true_line_numbers() {
    let corpus = Corpus::new();
    corpus.write(
        "numbered.txt",
        "first line\nthe needle hides on line two\nthird line\nfourth line\nline five has another needle\nsixth line\n",
    );

    let out = common::run_ok(corpus.ckq().args(["-n", "needle", "numbered.txt"]));
    let lines = stdout_lines(&out);
    assert_eq!(
        lines,
        vec![
            "2:the needle hides on line two",
            "5:line five has another needle"
        ],
        "line:content for each match, in file order"
    );
}

/// `-C N` prints exactly the N lines before and after the match, and stops at
/// the file boundary instead of reaching past it.
#[test]
fn context_flag_gives_exactly_n_lines_around() {
    let corpus = Corpus::new();
    corpus.write(
        "f.txt",
        "line 1\nline 2\nline 3\nline four TARGET\nline 5\nline 6\nline 7\n",
    );
    corpus.write("g.txt", "TARGET at the start\nsecond line\nthird line\n");

    let out = common::run_ok(corpus.ckq().args(["-n", "-C", "1", "TARGET", "f.txt"]));
    let lines = stdout_lines(&out);
    assert_eq!(
        lines,
        vec!["4:line 3", "line four TARGET", "line 5"],
        "-C 1 window"
    );

    let out = common::run_ok(corpus.ckq().args(["-n", "-C", "2", "TARGET", "f.txt"]));
    let lines = stdout_lines(&out);
    assert_eq!(
        lines,
        vec!["4:line 2", "line 3", "line four TARGET", "line 5", "line 6"],
        "-C 2 window"
    );

    let out = common::run_ok(corpus.ckq().args(["-n", "-C", "1", "TARGET", "g.txt"]));
    let lines = stdout_lines(&out);
    assert_eq!(
        lines,
        vec!["1:TARGET at the start", "second line"],
        "no context before line 1"
    );
}

/// `-i` case-folds the pattern; without it the match is case-sensitive.
#[test]
fn ignore_case_flag_matches_other_cases() {
    let corpus = Corpus::new();
    corpus.write("c.txt", "cat cot cut\nCat story\nconcatenate the parts\n");

    let out = common::run_ok(corpus.ckq().args(["-n", "cat", "c.txt"]));
    let lines = stdout_lines(&out);
    assert_eq!(
        lines,
        vec!["1:cat cot cut", "3:concatenate the parts"],
        "case-sensitive: 'Cat' does not match, 'concatenate' does"
    );

    let out = common::run_ok(corpus.ckq().args(["-n", "-i", "cat", "c.txt"]));
    let lines = stdout_lines(&out);
    assert_eq!(
        lines,
        vec!["1:cat cot cut", "2:Cat story", "3:concatenate the parts"],
        "-i adds the case-folded hit"
    );
}

/// `-w` matches only on word boundaries.
#[test]
fn word_regexp_flag_matches_whole_words_only() {
    let corpus = Corpus::new();
    corpus.write("c.txt", "cat cot cut\nCat story\nconcatenate the parts\n");

    let out = common::run_ok(corpus.ckq().args(["-n", "-w", "cat", "c.txt"]));
    let lines = stdout_lines(&out);
    assert_eq!(
        lines,
        vec!["1:cat cot cut"],
        "'concatenate' has no boundary before 'cat', and 'Cat' is another case"
    );
}

/// `-F` treats the pattern as bytes, so regex metacharacters match themselves —
/// and the same pattern read as a regex genuinely does not match the file.
#[test]
fn fixed_strings_flag_escapes_metacharacters() {
    let corpus = Corpus::new();
    corpus.write("d.txt", "a+b (test) [x]\nplain ab\n");

    let out = common::run_ok(corpus.ckq().args(["-n", "-F", "a+b (test) [x]", "d.txt"]));
    let lines = stdout_lines(&out);
    assert_eq!(lines, vec!["1:a+b (test) [x]"], "the literal text is found");

    let stderr = common::run_err(corpus.ckq().args(["-n", "a+b (test) [x]", "d.txt"]));
    assert!(
        stderr.contains("No matches found"),
        "as a regex the pattern matches nothing here: {stderr}"
    );
}

/// `--topk N` caps the result list at N, in jsonl and in `-l` alike.
#[test]
fn topk_limits_the_result_count() {
    let corpus = Corpus::new();
    for i in 1..=5 {
        corpus.write(&format!("k{i}.txt"), format!("quokka wobble {i}\n"));
    }

    let hits = corpus.search(&["--topk", "2", "quokka"]);
    assert_eq!(hits.len(), 2, "--topk 2 of 5 matching files");

    let all = corpus.search(&["quokka"]);
    assert_eq!(all.len(), 5, "regex search returns every match by default");

    let out = common::run_ok(corpus.ckq().args(["-l", "--topk", "3", "quokka", "."]));
    let count = String::from_utf8_lossy(&out.stdout)
        .lines()
        .filter(|l| !l.trim().is_empty())
        .count();
    assert_eq!(count, 3, "-l honors --topk");
}

// ---------------------------------------------------------------------------
// Exit codes and messages
// ---------------------------------------------------------------------------

/// A match exits 0 with the match on stdout.
#[test]
fn exit_zero_on_match() {
    let corpus = Corpus::new();
    corpus.write("hit.txt", "the zephyr kite rises\n");
    let out = common::run_ok(corpus.ckq().args(["-q", "zephyr", "."]));
    assert!(
        String::from_utf8_lossy(&out.stdout).contains("the zephyr kite rises"),
        "the matching line is on stdout"
    );
}

/// No match exits 1 (grep convention), with nothing on stdout and the reason
/// on stderr.
#[test]
fn no_match_exits_one_with_stderr_message() {
    let corpus = Corpus::new();
    corpus.write("plain.txt", "nothing to see\n");
    let (code, stdout, stderr) = common::run_exit(corpus.ckq().args(["-q", "zzz-absent", "."]));
    assert_eq!(code, Some(1), "grep-style no-match exit code");
    assert!(stdout.trim().is_empty(), "stdout stays empty: {stdout}");
    assert!(
        stderr.contains("No matches found"),
        "the reason is on stderr: {stderr}"
    );
}

/// A nonexistent search path exits 1 with the path named in the error.
#[test]
fn nonexistent_path_exits_one_with_message() {
    let corpus = Corpus::new();
    corpus.write("plain.txt", "nothing to see\n");
    let missing = corpus.path().join("no/such/dir");
    let (code, _stdout, stderr) =
        common::run_exit(
            corpus
                .ckq()
                .args(["-q", "anything", missing.to_str().unwrap()]),
        );
    assert_eq!(code, Some(1));
    assert!(
        stderr.contains("Path does not exist"),
        "the error names the problem: {stderr}"
    );
    assert!(
        stderr.contains("no/such/dir"),
        "the error names the path: {stderr}"
    );
}

/// An invalid regex exits 1 with a regex error, not a silent failure.
#[test]
fn invalid_regex_exits_one_with_regex_error() {
    let corpus = Corpus::new();
    corpus.write("plain.txt", "nothing to see\n");
    let (code, _stdout, stderr) =
        common::run_exit(corpus.ckq().args(["-q", "[unclosed-class", "."]));
    assert_eq!(code, Some(1));
    assert!(
        stderr.contains("Regex error"),
        "the error says it is the regex: {stderr}"
    );
}

// ---------------------------------------------------------------------------
// MCP over stdio
// ---------------------------------------------------------------------------

/// A `ckq --serve` child spoken to as the MCP spec says: newline-delimited
/// JSON-RPC on stdin/stdout, logs on a file we can read when it dies.
struct McpStdio {
    child: Child,
    stdin: Option<ChildStdin>,
    rx: mpsc::Receiver<String>,
    stderr_path: PathBuf,
}

impl McpStdio {
    fn start(corpus: &Corpus) -> Self {
        let stderr_path = corpus.path().join("serve.stderr.log");
        let err_file = fs::File::create(&stderr_path).expect("create stderr log");
        let mut child = common::ckq()
            .arg("--serve")
            .current_dir(corpus.path())
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::from(err_file))
            .spawn()
            .expect("spawn ckq --serve");
        let stdin = child.stdin.take().expect("piped stdin");
        let stdout = child.stdout.take().expect("piped stdout");
        let (tx, rx) = mpsc::channel();
        std::thread::spawn(move || {
            for line in BufReader::new(stdout).lines() {
                match line {
                    Ok(l) => {
                        if tx.send(l).is_err() {
                            break;
                        }
                    }
                    Err(_) => break,
                }
            }
        });
        Self {
            child,
            stdin: Some(stdin),
            rx,
            stderr_path,
        }
    }

    fn send(&mut self, message: &Value) {
        let line = serde_json::to_string(message).expect("serialize request");
        let stdin = self.stdin.as_mut().expect("stdin open");
        writeln!(stdin, "{line}").expect("write to server stdin");
        stdin.flush().expect("flush server stdin");
    }

    /// The next response with the given id. If the server dies first, its
    /// stderr — where a startup crash explains itself — goes into the failure.
    fn response(&mut self, id: u64, timeout: Duration) -> Value {
        let deadline = Instant::now() + timeout;
        loop {
            let remaining = deadline
                .checked_duration_since(Instant::now())
                .unwrap_or_else(|| self.fail(&format!("timed out waiting for response {id}")));
            match self.rx.recv_timeout(remaining) {
                Ok(line) => {
                    let v: Value = serde_json::from_str(&line).unwrap_or_else(|e| {
                        panic!("not a JSON line from the server ({e}): {line}")
                    });
                    if v.get("id").and_then(Value::as_u64) == Some(id) {
                        return v;
                    }
                    // Anything else (notifications, server requests) is skipped.
                }
                Err(RecvTimeoutError::Timeout) => {
                    self.fail(&format!("timed out waiting for response {id}"))
                }
                Err(RecvTimeoutError::Disconnected) => {
                    self.fail(&format!("the server exited before answering request {id}"))
                }
            }
        }
    }

    fn fail(&mut self, what: &str) -> ! {
        let stderr = fs::read_to_string(&self.stderr_path).unwrap_or_default();
        let status = self
            .child
            .try_wait()
            .expect("child is waitable")
            .map(|s| s.to_string());
        panic!("ckq --serve {what} (status so far: {status:?})\n--- server stderr\n{stderr}");
    }
}

impl Drop for McpStdio {
    fn drop(&mut self) {
        // Closing stdin is the client's goodbye; give the server a moment to
        // exit on it, then kill exactly this child — never a pattern.
        self.stdin.take();
        let deadline = Instant::now() + Duration::from_secs(5);
        while Instant::now() < deadline {
            if self.child.try_wait().expect("child is waitable").is_some() {
                return;
            }
            std::thread::sleep(Duration::from_millis(50));
        }
        let _ = self.child.kill();
        let _ = self.child.wait();
    }
}

/// The full MCP handshake against the real binary: initialize, initialized,
/// tools/list, then tools/call for the semantic and the hybrid tool (with
/// terms) on a real corpus. The response shapes are the contract AI clients
/// code against.
#[test]
#[serial]
#[ignore = "bug: ckq --serve panics at startup ('chunk size must be non-zero', ck-embed/src/embed_daemon.rs:182) so the stdio server never comes up; also needs the default model (run with --include-ignored)"]
fn known_bug_mcp_stdio_end_to_end() {
    let corpus = semantic_corpus();
    corpus.index();
    const TIMEOUT: Duration = Duration::from_secs(300);

    let mut mcp = McpStdio::start(&corpus);

    mcp.send(&json!({
        "jsonrpc": "2.0", "id": 1, "method": "initialize",
        "params": {
            "protocolVersion": "2024-11-05",
            "capabilities": {},
            "clientInfo": { "name": "ckq-e2e-contract", "version": "1.0" }
        }
    }));
    let init = mcp.response(1, TIMEOUT);
    assert!(init.get("error").is_none(), "initialize failed: {init}");
    let result = &init["result"];
    assert!(
        result
            .get("protocolVersion")
            .and_then(Value::as_str)
            .is_some(),
        "initialize answers with a protocol version: {init}"
    );
    let server_name = result["serverInfo"]["name"]
        .as_str()
        .unwrap_or_else(|| panic!("initialize answers with a server name: {init}"));
    assert!(!server_name.is_empty());
    assert!(
        result["capabilities"].get("tools").is_some(),
        "tools capability advertised: {init}"
    );

    mcp.send(&json!({"jsonrpc": "2.0", "method": "notifications/initialized"}));

    mcp.send(&json!({"jsonrpc": "2.0", "id": 2, "method": "tools/list", "params": {}}));
    let listed = mcp.response(2, TIMEOUT);
    assert!(listed.get("error").is_none(), "tools/list failed: {listed}");
    let tools = listed["result"]["tools"]
        .as_array()
        .unwrap_or_else(|| panic!("tools/list returns an array: {listed}"));
    let names: Vec<String> = tools
        .iter()
        .map(|t| {
            t["name"]
                .as_str()
                .unwrap_or_else(|| panic!("tool without a name: {t}"))
                .to_string()
        })
        .collect();
    for expected in [
        "health_check",
        "semantic_search",
        "lexical_search",
        "regex_search",
        "hybrid_search",
        "index_status",
        "reindex",
        "default_ckignore",
    ] {
        assert!(
            names.iter().any(|n| n == expected),
            "tool {expected} listed; got {names:?}"
        );
    }

    // The semantic tool, end to end: query in, structured results out.
    mcp.send(&json!({
        "jsonrpc": "2.0", "id": 3, "method": "tools/call",
        "params": {
            "name": "semantic_search",
            "arguments": { "query": "folding butter into pastry dough", "path": ".", "top_k": 3 }
        }
    }));
    let sem = mcp.response(3, TIMEOUT);
    assert!(sem.get("error").is_none(), "semantic_search errored: {sem}");
    let sc = &sem["result"]["structuredContent"];
    assert_eq!(
        sc["search"]["mode"].as_str(),
        Some("semantic"),
        "the structured content names its mode: {sc}"
    );
    let matches = sc["results"]["matches"]
        .as_array()
        .unwrap_or_else(|| panic!("semantic matches array: {sc}"));
    assert!(
        matches.iter().any(|m| m["file"]["path"]
            .as_str()
            .is_some_and(|p| p.ends_with("croissant.md"))),
        "the croissant document is among the results: {sc}"
    );
    assert!(
        sc["results"]["count"].as_u64().unwrap_or(0) >= 1,
        "the count matches the results: {sc}"
    );
    for m in matches {
        let path = m["file"]["path"]
            .as_str()
            .unwrap_or_else(|| panic!("file.path: {m}"));
        assert!(!path.is_empty());
        assert!(
            m["file"]["language"].as_str().is_some(),
            "file.language present: {m}"
        );
        span_fields(&m["match"]["span"]);
        assert!(
            m["match"]["score"].as_f64().is_some(),
            "semantic matches carry a score: {m}"
        );
        assert_eq!(
            m["type"].as_str(),
            Some("semantic_match"),
            "match type names the mode: {m}"
        );
    }

    // The hybrid tool, with explicit terms driving its lexical method.
    mcp.send(&json!({
        "jsonrpc": "2.0", "id": 4, "method": "tools/call",
        "params": {
            "name": "hybrid_search",
            "arguments": {
                "query": "watering the tomato garden",
                "terms": ["tomates"],
                "path": ".",
                "top_k": 5
            }
        }
    }));
    let hybrid = mcp.response(4, TIMEOUT);
    assert!(
        hybrid.get("error").is_none(),
        "hybrid_search errored: {hybrid}"
    );
    let hc = &hybrid["result"]["structuredContent"];
    assert_eq!(hc["search"]["mode"].as_str(), Some("hybrid"), "mode: {hc}");
    let matches = hc["results"]["matches"]
        .as_array()
        .unwrap_or_else(|| panic!("hybrid matches array: {hc}"));
    assert!(
        matches.iter().any(|m| m["file"]["path"]
            .as_str()
            .is_some_and(|p| p.ends_with("tomates.md"))),
        "the tomato document is among the results: {hc}"
    );
    for m in matches {
        assert_eq!(m["type"].as_str(), Some("hybrid_match"), "type: {m}");
        assert!(
            m["match"]["rrf_score"].as_f64().is_some(),
            "hybrid matches carry the fused score: {m}"
        );
        span_fields(&m["match"]["span"]);
    }
}
