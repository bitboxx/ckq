//! End-to-end tests for the index lifecycle: build, incremental update,
//! deletion, `--add`, `.rogoignore`, hidden files, robustness against odd
//! inputs, and the safety rails around the store's embedding signature.
//! Everything runs the real binary on real files, and almost everything needs
//! the real default model because indexing embeds — hence the `e2e:` ignores.
//! The index is one SQLite file, `.rogo/index.db`; what a test can observe of it
//! is `--status-json`, the `--index` summary line, and what search returns.

mod common;

use common::{Corpus, run_err, run_ok};
use serde_json::Value;
use serial_test::serial;
use std::fs;
use std::path::PathBuf;
use std::process::{Command, Output, Stdio};
use std::time::{Duration, Instant};

/// Files the index claims, from `--status-json`.
fn total_files(corpus: &Corpus) -> u64 {
    common::status_json(corpus.path())["total_files"]
        .as_u64()
        .expect("total_files in --status-json")
}

/// Run a command that must finish within `secs`, or fail the test. Indexing
/// walks the tree it is given, so a walker that followed a symlink loop
/// would hang forever; a plain `output()` would hang the test with it.
fn run_bounded(cmd: &mut Command, secs: u64) -> Output {
    cmd.stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped());
    let mut child = cmd.spawn().expect("rogo should start");
    let deadline = Instant::now() + Duration::from_secs(secs);
    loop {
        match child.try_wait().expect("rogo should be waitable") {
            Some(_) => break,
            None if Instant::now() >= deadline => {
                let _ = child.kill();
                panic!(
                    "rogo {:?} did not finish within {secs}s",
                    cmd.get_args().collect::<Vec<_>>()
                );
            }
            None => std::thread::sleep(Duration::from_millis(100)),
        }
    }
    child.wait_with_output().expect("reap rogo")
}

/// The first integer that follows `label` in a status report line.
fn number_after(report: &str, label: &str) -> usize {
    let i = report
        .find(label)
        .unwrap_or_else(|| panic!("no '{label}' in status output\n{report}"));
    let digits: String = report[i + label.len()..]
        .chars()
        .take_while(|c| c.is_ascii_digit())
        .collect();
    digits
        .parse()
        .unwrap_or_else(|_| panic!("no number after '{label}' in status output\n{report}"))
}

/// `--status-json` of the corpus, parsed.
fn status_json(corpus: &Corpus) -> Value {
    let out = run_ok(corpus.rogo().arg("--status-json"));
    serde_json::from_slice(&out.stdout).expect("--status-json should print one JSON object")
}

/// Run a search that must find nothing. `rogo` exits 1 with "No matches
/// found" like grep, so `run_ok` cannot express this; require exactly that
/// outcome so a real error (bad flag, crash) still fails the test.
fn expect_no_hits(corpus: &Corpus, args: &[&str]) {
    let out = corpus
        .rogo()
        .args(args)
        .args(["--jsonl", "-q", "."])
        .output()
        .expect("rogo should start");
    assert!(
        !out.status.success(),
        "rogo {args:?} should have found nothing\n--- stdout\n{}",
        String::from_utf8_lossy(&out.stdout),
    );
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(
        stderr.contains("No matches found"),
        "expected the no-matches note on stderr, got:\n{stderr}"
    );
    assert!(
        common::parse_jsonl(&out.stdout).is_empty(),
        "no result lines expected"
    );
}

#[test]
#[serial]
#[ignore = "e2e: needs the default model; run with --include-ignored"]
fn index_builds_the_ck_layout_and_status_reports_the_corpus() {
    let corpus = Corpus::new();
    corpus.write(
        "lighthouse.md",
        "The lighthouse keeper counts every ship that passes the cape at dusk.\n\
         Harbour logs record the weather, the tide and the cargo tonnage.\n",
    );
    corpus.write(
        "garden.md",
        "Gardens teach patience: seeds sleep through frost before they wake.\n\
         The gardener waters the rows each morning before the heat arrives.\n",
    );
    corpus.write(
        "nested/harbour.md",
        "The ferry crosses the estuary at noon when the current is weakest.\n",
    );

    corpus.index();

    // One store file holds the whole index: files, chunks, vectors and the
    // BM25 index alike, so a lexical search needs no build step of its own.
    let index_dir = corpus.path().join(".rogo");
    assert!(index_dir.join("index.db").is_file(), "index.db missing");
    corpus.search(&["--lex", "lighthouse"]);
    assert!(
        index_dir.join("index.db").is_file(),
        "a lexical search writes nothing"
    );

    // --status with no path argument reads the current directory; the counts
    // it reports must be the counts the corpus actually has.
    let out = run_ok(corpus.rogo().arg("--status"));
    let report = String::from_utf8_lossy(&out.stderr).into_owned();
    assert!(
        report.contains("Index location"),
        "no location line\n{report}"
    );
    let files = number_after(&report, "Files indexed: ");
    let chunks = number_after(&report, "Total chunks: ");
    let embedded = number_after(&report, "Embedded chunks: ");
    assert_eq!(files, 3, "status file count\n{report}");
    assert!(chunks >= 3, "one chunk per file at least\n{report}");
    assert_eq!(embedded, chunks, "every chunk embedded\n{report}");

    let json = status_json(&corpus);
    assert_eq!(json["total_files"].as_u64(), Some(3));
    assert_eq!(json["total_chunks"].as_u64(), Some(chunks as u64));
    assert_eq!(json["embedded_chunks"].as_u64(), Some(chunks as u64));
    assert_eq!(json["index_exists"].as_bool(), Some(true));

    let out = run_ok(corpus.rogo().arg("--status-verbose"));
    let verbose = String::from_utf8_lossy(&out.stderr).into_owned();
    for label in ["Source size:", "Index size:", "Compression:", "Model:"] {
        assert!(
            verbose.contains(label),
            "--status-verbose lacks '{label}'\n{verbose}"
        );
    }
}

#[test]
#[serial]
#[ignore = "e2e: needs the default model; run with --include-ignored"]
fn incremental_reindex_reembeds_only_the_edited_file() {
    let corpus = Corpus::new();
    corpus.write(
        "meteor.md",
        "The observatory tracks a meteor shower every August over the ridge.\n",
    );
    corpus.write(
        "bakery.md",
        "The bakery opens before sunrise and the street smells of warm bread.\n",
    );
    corpus.write(
        "ferry.md",
        "The night ferry slows down when it enters the narrow channel.\n",
    );
    corpus.index();

    // The observable for "only that file was re-embedded" is the summary of
    // the second run: one file updated, the other two untouched, and only the
    // edited file's chunks back through the embedder.
    corpus.write(
        "bakery.md",
        "The bakery opens before sunrise and the street smells of warm bread.\n\
         By nine the shelves hold cardamom buns and the queue reaches the corner.\n",
    );
    let out = run_ok(corpus.rogo().args(["--index", "."]));
    let summary = common::parse_index_summary(&String::from_utf8_lossy(&out.stderr));
    assert_eq!(summary.added, 0, "nothing new: {summary:?}");
    assert_eq!(summary.updated, 1, "only the edited file: {summary:?}");
    assert_eq!(summary.removed, 0, "{summary:?}");
    assert_eq!(summary.unchanged, 2, "{summary:?}");
    assert!(summary.chunks_embedded > 0, "the edit must be embedded");
    assert_eq!(total_files(&corpus), 3, "entries were lost");

    // The new content is findable, and the unchanged files keep theirs.
    let sem = corpus.search(&["--sem", "cardamom buns fresh from the oven"]);
    assert_eq!(
        common::rank_of(&sem, "bakery"),
        Some(1),
        "edited content not found by --sem"
    );
    let lex = corpus.search(&["--lex", "meteor"]);
    assert_eq!(common::rank_of(&lex, "meteor"), Some(1));
    let lex = corpus.search(&["--lex", "cardamom"]);
    assert_eq!(common::rank_of(&lex, "bakery"), Some(1));
}

#[test]
#[serial]
#[ignore = "e2e: needs the default model; run with --include-ignored"]
fn deleted_file_disappears_from_search_and_the_index_drops_it() {
    let corpus = Corpus::new();
    corpus.write(
        "tide.md",
        "The tide chart lists every high water mark of the season.\n\
         Sailors read the tables before they leave the sheltered bay.\n",
    );
    corpus.write(
        "mushroom.md",
        "The forest floor hides mushrooms after the autumn rain.\n\
         Deer trails wind between the tall wet pines.\n",
    );
    corpus.index();

    let mushroom_queries = [
        ("--sem", "mushrooms under the wet pines"),
        ("--lex", "mushrooms"),
        ("--hybrid", "mushrooms"),
    ];
    for (mode, query) in mushroom_queries {
        let hits = corpus.search(&[mode, query]);
        assert!(
            hits.iter().any(|h| h.stem() == "mushroom"),
            "{mode} should find the file before deletion"
        );
    }

    corpus.remove("mushroom.md");
    let out = run_ok(corpus.rogo().args(["--index", "."]));
    let summary = common::parse_index_summary(&String::from_utf8_lossy(&out.stderr));
    assert_eq!(
        summary.removed, 1,
        "the deleted file must be dropped: {summary:?}"
    );

    for (mode, query) in mushroom_queries {
        expect_no_hits(&corpus, &[mode, query]);
    }
    // The survivor is untouched by all of this.
    let hits = corpus.search(&["--sem", "high water marks this season"]);
    assert_eq!(common::rank_of(&hits, "tide"), Some(1));
    // One store means no orphans can exist: the dead file's rows went with
    // the same run that noticed the deletion. --clean-orphans stays accepted
    // and keeps its promise trivially.
    run_ok(corpus.rogo().args(["--clean-orphans", "."]));
    expect_no_hits(&corpus, &["--lex", "mushrooms"]);
    assert_eq!(total_files(&corpus), 1);
    // --clean ends the lifecycle: the whole index directory goes away. v2
    // never rebuilds an index from a search (search is read-only), so the
    // next lexical search must refuse and say how to start over.
    run_ok(corpus.rogo().args(["--clean", "."]));
    assert!(
        !corpus.path().join(".rogo").exists(),
        "--clean left .rogo behind"
    );
    let stderr = run_err(corpus.rogo().args(["--lex", "tide", "--jsonl", "-q", "."]));
    assert!(
        stderr.contains("--index"),
        "a search after --clean must point at --index, got:\n{stderr}"
    );
}

/// A legacy `.ck` left by the old engine is content-free to rogo: the walker
/// skips it as it skips `.rogo` and `.git`, however the other flags are set,
/// so its files are neither indexed nor found by the regex mode, which walks
/// the same tree. `--hidden` is on so the dot alone cannot be what excludes
/// it.
#[test]
#[serial]
#[ignore = "e2e: needs the default model; run with --include-ignored"]
fn a_legacy_ck_directory_is_never_indexed_or_searched() {
    let corpus = Corpus::new();
    corpus.write(
        "tide.md",
        "The tide clock counts the hours to the next high water.\n",
    );
    corpus.write(
        ".ck/manifest.json",
        "{\"old engine leftover mentions saxifrage\"}\n",
    );
    run_ok(corpus.rogo().args(["--index", "--hidden", "-q", "."]));
    // Two files: tide.md and the .rogoignore the index run wrote (a hidden
    // text file like any other). The legacy .ck would make it three.
    assert_eq!(
        total_files(&corpus),
        2,
        "the legacy .ck must not be indexed"
    );
    expect_no_hits(&corpus, &["--lex", "saxifrage"]);
    expect_no_hits(&corpus, &["--hidden", "saxifrage"]);
}

/// A corpus with only a legacy `.ck/index.db` refuses a ranked search and
/// says the index moved to `.rogo` and how to build it again, rather than
/// reporting a bare missing index.
#[test]
#[serial]
#[ignore = "e2e: needs no model, only the real binary"]
fn a_search_over_only_a_legacy_ck_index_is_told_it_moved() {
    let corpus = Corpus::new();
    corpus.write(
        "tide.md",
        "The tide clock counts the hours to the next high water.\n",
    );
    corpus.write(".ck/index.db", b"old engine bytes");
    let stderr = run_err(corpus.rogo().args(["--lex", "tide", "--jsonl", "-q", "."]));
    assert!(
        stderr.contains("moved") && stderr.contains(".rogo") && stderr.contains("--index"),
        "the error must say the index moved to .rogo and name --index, got:\n{stderr}"
    );
}

/// `--clean` removes `.rogo` only. A legacy `.ck` stays exactly as it was,
/// and the run says so on stderr.
#[test]
#[serial]
#[ignore = "e2e: needs the default model; run with --include-ignored"]
fn clean_removes_rogo_and_leaves_a_legacy_ck_in_place() {
    let corpus = Corpus::new();
    corpus.write(
        "tide.md",
        "The tide clock counts the hours to the next high water.\n",
    );
    corpus.write(".ck/manifest.json", "{\"old\": \"engine\"}\n");
    corpus.index();
    let out = run_ok(corpus.rogo().args(["--clean", "."]));
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(
        !corpus.path().join(".rogo").exists(),
        "--clean left .rogo behind"
    );
    assert!(
        corpus.path().join(".ck").is_dir(),
        "--clean must not touch a legacy .ck"
    );
    assert!(
        stderr.contains("left in place"),
        "--clean must say the legacy .ck stayed, got:\n{stderr}"
    );
}

/// A tree whose only index is the `.ckq` one ckq 2.0 wrote: a ranked search
/// fails with one message that names the directory and says which `mv`
/// carries it over — and nothing is moved or written on the way. `--status`
/// and `--add` take the same path, so it is covered through the search.
#[test]
#[serial]
#[ignore = "e2e: needs the default model; run with --include-ignored"]
fn a_renamed_ckq_index_is_named_with_its_mv_and_never_touched() {
    let corpus = Corpus::new();
    corpus.write(
        "tide.md",
        "The tide clock counts the hours to the next high water.\n",
    );
    corpus.index();
    // The migration the message prescribes, run backwards: the index a
    // ckq 2.0 user has is an index this build just wrote, under its old name.
    fs::rename(corpus.path().join(".rogo"), corpus.path().join(".ckq")).unwrap();
    let root = corpus.path().display().to_string();
    let stderr = run_err(corpus.rogo().args(["--lex", "tide", "--jsonl", "-q", "."]));
    assert!(
        stderr.contains("mv")
            && stderr.contains(&format!("{root}/.ckq"))
            && stderr.contains(&format!("{root}/.rogo"))
            && stderr.contains("no rebuild"),
        "the error must name the .ckq index, the .rogo target and the mv, got:\n{stderr}"
    );
    let stderr = run_err(corpus.rogo().args(["--status", "."]));
    assert!(
        stderr.contains("mv") && stderr.contains(".ckq"),
        "--status over only a .ckq index must say the same thing, got:\n{stderr}"
    );
    assert!(
        corpus.path().join(".ckq/index.db").is_file() && !corpus.path().join(".rogo").exists(),
        "the failing command must leave both the .ckq index and the absent .rogo as they were"
    );
    // The mv the message prescribes is the whole migration: the index
    // searches again, no rebuild.
    fs::rename(corpus.path().join(".ckq"), corpus.path().join(".rogo")).unwrap();
    let out = run_ok(corpus.rogo().args(["--lex", "tide", "--jsonl", "-q", "."]));
    let stdout = String::from_utf8_lossy(&out.stdout);
    assert!(
        stdout.contains("tide.md"),
        "a moved index must search as it is"
    );
}

/// A `.ckq` left beside a live `.rogo` is content-free to rogo, exactly like
/// a legacy `.ck`: the walker never enters it, however the flags are set,
/// and `--clean` removes `.rogo` only, saying the `.ckq` stayed.
#[test]
#[serial]
#[ignore = "e2e: needs the default model; run with --include-ignored"]
fn a_ckq_directory_is_skipped_by_the_walker_and_left_alone_by_clean() {
    let corpus = Corpus::new();
    corpus.write(
        "tide.md",
        "The tide clock counts the hours to the next high water.\n",
    );
    corpus.write(".ckq/index.db", b"ckq 2.0 bytes mentioning saxifrage");
    run_ok(corpus.rogo().args(["--index", "--hidden", "-q", "."]));
    // Two files: tide.md and the .rogoignore the index run wrote. The .ckq
    // would make it three, and its bytes would be searchable.
    assert_eq!(
        total_files(&corpus),
        2,
        "the .ckq directory must not be indexed"
    );
    expect_no_hits(&corpus, &["--lex", "saxifrage"]);
    expect_no_hits(&corpus, &["--hidden", "saxifrage"]);
    let out = run_ok(corpus.rogo().args(["--clean", "."]));
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(
        !corpus.path().join(".rogo").exists(),
        "--clean left .rogo behind"
    );
    assert!(
        corpus.path().join(".ckq/index.db").is_file(),
        "--clean must not touch a .ckq"
    );
    assert!(
        stderr.contains("left in place") && stderr.contains(".ckq"),
        "--clean must say the .ckq stayed, got:\n{stderr}"
    );
}

/// A corpus ckq indexed has its patterns in `.ckqignore`. While that is all
/// it has, rogo reads them — the same files stay excluded, never silently
/// included — and warns once on stderr to rename the file; `--index` does
/// not write a default `.rogoignore` beside it. After the rename the
/// patterns hold and the warning is gone.
#[test]
#[serial]
#[ignore = "e2e: needs the default model; run with --include-ignored"]
fn a_legacy_ckqignore_is_read_with_one_warning_and_not_duplicated() {
    let corpus = Corpus::new();
    corpus.write("tide.md", "The tide clock counts the hours.\n");
    corpus.write("secret_draft.md", "saxifrage grows in the cliff.\n");
    corpus.write(".ckqignore", "secret_*.md\n");
    let out = run_ok(corpus.rogo().args(["--index", "-q", "."]));
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(total_files(&corpus), 1, "the .ckqignore patterns must hold");
    assert!(
        stderr.matches("rogo: warning: reading the legacy").count() == 1
            && stderr.contains(".ckqignore")
            && stderr.contains(".rogoignore"),
        "one warning must name the rename, got:\n{stderr}"
    );
    assert!(
        !corpus.path().join(".rogoignore").exists(),
        "--index must not write a default .rogoignore beside a .ckqignore"
    );
    expect_no_hits(&corpus, &["--lex", "saxifrage"]);
    // The prescribed rename: same patterns, new name, no warning, and now
    // --index does write nothing either (the file exists).
    fs::rename(
        corpus.path().join(".ckqignore"),
        corpus.path().join(".rogoignore"),
    )
    .unwrap();
    let out = run_ok(corpus.rogo().args(["--index", "-q", "."]));
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(
        !stderr.contains("legacy"),
        "a renamed .rogoignore must not warn, got:\n{stderr}"
    );
    expect_no_hits(&corpus, &["--lex", "saxifrage"]);
}

/// A download that starts while ckq's cache is still in place says so first:
/// one stderr line naming the `mv` that would have skipped it. The endpoint
/// is dead so the download cannot succeed or linger; the line is printed
/// before any bytes move.
#[test]
#[serial]
#[ignore = "e2e: needs no model, only the real binary and a scratch HOME"]
fn a_model_download_with_ckq_cache_in_place_says_which_mv_skips_it() {
    let dir = tempfile::TempDir::new().unwrap();
    let home = dir.path().join("home");
    fs::create_dir_all(home.join(".cache/ckq/models")).unwrap();
    fs::write(
        dir.path().join("tide.md"),
        "The tide clock counts the hours.\n",
    )
    .unwrap();
    let out = common::rogo()
        .env("HOME", &home)
        .env("HF_ENDPOINT", "http://127.0.0.1:9")
        .current_dir(&dir)
        .args(["--index", "-q", "."])
        .output()
        .expect("rogo should start");
    assert!(
        !out.status.success(),
        "the download must fail against the dead endpoint"
    );
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(
        stderr.contains("rogo: note: a ckq model cache exists")
            && stderr.contains("mv ~/.cache/ckq ~/.cache/rogo"),
        "the download must say which mv skips it, got:\n{stderr}"
    );
}

#[test]
#[serial]
#[ignore = "e2e: needs the default model; run with --include-ignored"]
fn add_indexes_one_new_file_without_a_full_run() {
    let corpus = Corpus::new();
    corpus.write(
        "meridian.md",
        "The cartographer draws the meridian with a steady hand.\n",
    );
    corpus.write(
        "orchard.md",
        "Apple crates stack along the orchard wall in the autumn.\n",
    );
    corpus.index();

    corpus.write(
        "lantern.md",
        "The lighthouse lantern attracts a colony of gannets after dark.\n\
         Keepers log every seabird that strikes the glass.\n",
    );
    run_ok(corpus.rogo().args(["--add", "lantern.md"]));

    assert_eq!(status_json(&corpus)["total_files"].as_u64(), Some(3));
    let lex = corpus.search(&["--lex", "gannets"]);
    assert_eq!(
        common::rank_of(&lex, "lantern"),
        Some(1),
        "--add did not make the new file lexically findable"
    );
    let sem = corpus.search(&["--sem", "seabirds circling the light at night"]);
    assert_eq!(
        common::rank_of(&sem, "lantern"),
        Some(1),
        "--add did not make the new file semantically findable"
    );
    // The two files from the full run are still searchable.
    let lex = corpus.search(&["--lex", "meridian"]);
    assert_eq!(common::rank_of(&lex, "meridian"), Some(1));
}

#[test]
#[serial]
#[ignore = "e2e: needs the default model; run with --include-ignored"]
fn rogoignore_excludes_files_from_the_start_and_when_added_later() {
    let corpus = Corpus::new();
    corpus.write(".rogoignore", "# test ignore list\nprivate_*.md\n");
    corpus.write(
        "private_journal.md",
        "My parrot learned to whistle the whole anthem in secret.\n",
    );
    corpus.write(
        "draft_letter.md",
        "Draft: the courier will bring a parsnip crate on Thursday.\n",
    );
    corpus.write(
        "public_bulletin.md",
        "The town mosaic depicts a harvest and a river crossing.\n",
    );
    corpus.index();

    // A pattern present before the first index: the file never gets in.
    expect_no_hits(&corpus, &["--lex", "anthem"]);
    assert_eq!(total_files(&corpus), 2, "the ignored file was indexed");
    assert_eq!(
        common::rank_of(&corpus.search(&["--lex", "parsnip"]), "draft_letter"),
        Some(1)
    );

    // The same file, ignored only after the fact, must drop out of results.
    let mut ignore = fs::read_to_string(corpus.path().join(".rogoignore")).unwrap();
    ignore.push_str("draft_letter.md\n");
    fs::write(corpus.path().join(".rogoignore"), ignore).unwrap();
    corpus.index();

    expect_no_hits(&corpus, &["--lex", "parsnip"]);
    assert_eq!(
        common::rank_of(&corpus.search(&["--lex", "mosaic"]), "public_bulletin"),
        Some(1),
        "the unignored file must stay findable"
    );
}

#[test]
#[serial]
#[ignore = "e2e: needs the default model; run with --include-ignored"]
fn hidden_files_are_indexed_only_with_the_hidden_flag() {
    let corpus = Corpus::new();
    corpus.write(
        "hall.md",
        "The great hall hosts the winter market every year.\n",
    );
    corpus.write(
        ".attic/quilts.md",
        "A quiet attic stores the winter quilts.\n",
    );
    corpus.write(
        ".cellar.md",
        "The cellar keeps jars of plum jam and marbles.\n",
    );
    corpus.index();

    assert_eq!(total_files(&corpus), 1, "only the visible file");
    expect_no_hits(&corpus, &["--lex", "quilts"]);

    run_ok(corpus.rogo().args(["--index", "--hidden", "-q", "."]));
    // The walker sees four text files now: the visible one, the dot-file, the
    // dot-dir's file, and the .rogoignore the first run wrote (a hidden text
    // file like any other, exactly as the old engine treated it).
    assert_eq!(
        total_files(&corpus),
        4,
        "the dot-file, dot-dir and .rogoignore join"
    );
    let hits = corpus.search(&["--lex", "quilts", "--hidden"]);
    assert_eq!(
        common::rank_of(&hits, "quilts"),
        Some(1),
        "--hidden did not make the dot-file findable"
    );
    let hits = corpus.search(&["--lex", "marbles", "--hidden"]);
    assert_eq!(
        common::rank_of(&hits, ".cellar"),
        Some(1),
        "stem of a dot-file keeps its leading dot"
    );
}

#[test]
#[serial]
#[ignore = "e2e: needs the default model; run with --include-ignored"]
fn odd_files_index_and_search_safely() {
    let corpus = Corpus::new();
    // A binary file: every byte value, many times over.
    let binary: Vec<u8> = (0..=255u8).cycle().take(256 * 40).collect();
    corpus.write("blob_data.bin", binary);
    corpus.write("empty_page.md", "");
    corpus.write(
        "crlf_pier.md",
        "The harbourmaster checks the mooring lines.\r\n\
         A gull steals breadcrumbs from the pier.\r\n",
    );
    corpus.write(
        "notas del café ☕.md",
        "El café de la mañana sabe a lluvia fresca de la sierra.\n",
    );
    corpus.write("中文笔记.md", "山中的寺庙藏于云雾之间，钟声缓慢而悠远。\n");
    corpus.write(
        "nested/deep/glacier.md",
        "The glacier retreats a little more every summer,\n\
         leaving a raw scar of gravel behind.\n",
    );
    corpus.index();

    // The binary file is refused, everything else is taken, and one odd
    // input must not cost the rest their entries.
    let json = status_json(&corpus);
    assert_eq!(json["total_files"].as_u64(), Some(5), "files: {json}");
    expect_no_hits(&corpus, &["--lex", "AAAA"]);
    assert!(common::rank_of(&corpus.search(&["--lex", "café"]), "notas del café ☕").is_some());
    assert!(common::rank_of(&corpus.search(&["--lex", "sierra"]), "notas del café ☕").is_some());

    // CRLF content: findable, with a span that stays inside the file.
    let pier: PathBuf = corpus.path().join("crlf_pier.md");
    let pier_size = fs::metadata(&pier).unwrap().len();
    let hits = corpus.search(&["--lex", "mooring"]);
    assert_eq!(common::rank_of(&hits, "crlf_pier"), Some(1));
    let span = &hits[0].raw["span"];
    assert!(span["line_start"].as_u64().unwrap() >= 1);
    assert!(span["line_end"].as_u64().unwrap() <= 3);
    assert!(span["byte_end"].as_u64().unwrap() <= pier_size);

    // Non-ASCII file name: the path must round-trip exactly.
    let hits = corpus.search(&["--lex", "lluvia"]);
    assert_eq!(
        common::rank_of(&hits, "notas del café ☕"),
        Some(1),
        "Spanish content under a spaced non-ASCII name"
    );
    assert_eq!(hits[0].file_name(), "notas del café ☕.md");

    // Non-Latin content: lexical search does not tokenize Chinese, but the
    // embedding must still carry it across languages.
    let hits = corpus.search(&["--sem", "a mountain temple bell in the mist"]);
    assert_eq!(
        common::rank_of(&hits, "中文笔记"),
        Some(1),
        "Chinese content should be reachable from an English query"
    );
    assert_eq!(
        common::rank_of(&corpus.search(&["--lex", "glacier"]), "glacier"),
        Some(1)
    );
}

#[test]
#[serial]
#[ignore = "e2e: needs the default model; run with --include-ignored"]
fn big_file_chunks_have_spans_inside_the_file() {
    let corpus = Corpus::new();
    // ~290 KB of numbered lines: far past one chunk, so the chunker has to
    // split it and every split must land inside the real file.
    let mut big = String::new();
    for i in 1..=4200 {
        big.push_str(&format!(
            "Line {i}: the archivist shelves another bound volume of chronicles.\n"
        ));
    }
    let path = corpus.write("big_archive.md", &big);
    let size = fs::metadata(&path).unwrap().len();
    let lines = big.lines().count() as u64;
    assert!(size > 250_000, "fixture should be around 300 KB, is {size}");

    corpus.index();
    let json = status_json(&corpus);
    assert!(
        json["total_chunks"].as_u64().unwrap() > 10,
        "a 300 KB file must produce several chunks, got {json}"
    );

    let check = |mode: &str, query: &str, min_spans: usize| {
        let hits = corpus.search(&[mode, query, "--topk", "100"]);
        let spans: Vec<&Value> = hits
            .iter()
            .filter(|h| h.stem() == "big_archive")
            .map(|h| &h.raw["span"])
            .collect();
        assert!(
            spans.len() >= min_spans,
            "{mode} returned {} chunk spans of the big file, wanted >= {min_spans}",
            spans.len()
        );
        for span in &spans {
            let start = span["line_start"].as_u64().unwrap();
            let end = span["line_end"].as_u64().unwrap();
            let byte_end = span["byte_end"].as_u64().unwrap();
            assert!(start >= 1, "{mode} span starts before line 1: {span}");
            assert!(
                start <= end && end <= lines,
                "{mode} span leaves the file's {lines} lines: {span}"
            );
            assert!(
                byte_end <= size,
                "{mode} span ends past the file's {size} bytes: {span}"
            );
        }
    };
    // Semantic returns one hit per chunk; lexical collapses to the best
    // match per document but must still land inside the file.
    check("--sem", "archivist shelving bound volumes of chronicles", 3);
    check("--lex", "chronicles", 1);
}

/// A symlink loop in the corpus must not hang the walker (or the test).
#[cfg(unix)]
#[test]
#[serial]
#[ignore = "e2e: needs the default model; run with --include-ignored"]
fn symlink_loop_does_not_hang_indexing() {
    use std::os::unix::fs::symlink;
    let corpus = Corpus::new();
    corpus.write(
        "anchor.md",
        "The anchor holds the boat against the evening current.\n",
    );
    let root = corpus.path();
    fs::create_dir_all(root.join("loopdir")).unwrap();
    symlink("../loopdir", root.join("loopdir/self")).unwrap();
    symlink("..", root.join("loopdir/up")).unwrap();

    let out = run_bounded(corpus.rogo().args(["--index", "-q", "."]), 120);
    assert!(
        out.status.success(),
        "index over a symlink loop failed\n{}",
        String::from_utf8_lossy(&out.stderr)
    );
    assert_eq!(total_files(&corpus), 1);
    // Status is command mode, so it walks nothing and must be immediate.
    assert_eq!(status_json(&corpus)["total_files"].as_u64(), Some(1));
}

/// v2 search answers from the store, never by walking the query path, so a
/// symlink loop in the corpus cannot hang it (the old engine expanded the
/// search target with `glob`, which followed looped symlinks forever).
/// Bounded so a regression is a panic, not a hung run.
#[cfg(unix)]
#[test]
#[serial]
#[ignore = "e2e: needs the default model; run with --include-ignored"]
fn lexical_search_survives_a_symlink_loop() {
    use std::os::unix::fs::symlink;
    let corpus = Corpus::new();
    corpus.write(
        "anchor.md",
        "The anchor holds the boat against the evening current.\n",
    );
    let root = corpus.path();
    fs::create_dir_all(root.join("loopdir")).unwrap();
    symlink("../loopdir", root.join("loopdir/self")).unwrap();
    symlink("..", root.join("loopdir/up")).unwrap();
    corpus.index();

    let out = run_bounded(
        corpus
            .rogo()
            .args(["--lex", "anchor", "--jsonl", "-q", "."]),
        60,
    );
    assert!(
        out.status.success(),
        "--lex failed\n{}",
        String::from_utf8_lossy(&out.stderr)
    );
    let hits = common::parse_jsonl(&out.stdout);
    assert_eq!(common::rank_of(&hits, "anchor"), Some(1));
}

/// One unreadable file must be skipped, not allowed to fail the whole index.
#[cfg(unix)]
#[test]
#[serial]
#[ignore = "e2e: needs the default model; run with --include-ignored"]
fn unreadable_file_is_skipped_without_failing_the_index() {
    use std::os::unix::fs::PermissionsExt;
    let corpus = Corpus::new();
    corpus.write(
        "open_hall.md",
        "The open hall fills with afternoon light and visitors.\n",
    );
    let vault = corpus.write("locked_vault.md", "The vault code is written on a card.\n");
    fs::set_permissions(&vault, fs::Permissions::from_mode(0o000)).unwrap();

    corpus.index();
    assert_eq!(total_files(&corpus), 1);
    assert_eq!(
        common::rank_of(&corpus.search(&["--lex", "afternoon"]), "open_hall"),
        Some(1),
        "the readable file must still be searchable"
    );

    // Once readable again it joins the index on the next run.
    fs::set_permissions(&vault, fs::Permissions::from_mode(0o644)).unwrap();
    corpus.index();
    assert_eq!(total_files(&corpus), 2, "the readable-again file joins");
    assert_eq!(
        common::rank_of(&corpus.search(&["--lex", "vault"]), "locked_vault"),
        Some(1)
    );
}

/// The store's embedding signature guards against mixing vectors from
/// different prompt spaces. A `--model` change means a different signature,
/// which must rebuild the whole index rather than merge the two spaces.
#[test]
#[serial]
#[ignore = "e2e: needs the default model; run with --include-ignored"]
fn switching_models_rebuilds_the_whole_index() {
    let corpus = Corpus::new();
    corpus.write(
        "dune.md",
        "The dune moves a hand's width inland with every winter storm.\n",
    );
    corpus.index();
    assert_eq!(
        status_json(&corpus)["model"]["alias"].as_str(),
        Some("gemma-q4")
    );

    // gemma-gguf is the same architecture at another quantisation: the
    // switch must re-embed, and the store must end up wholly the new model's.
    let out = run_ok(
        corpus
            .rogo()
            .args(["--index", "--model", "gemma-gguf", "."]),
    );
    let summary = common::parse_index_summary(&String::from_utf8_lossy(&out.stderr));
    assert_eq!(summary.added, 1, "a rebuild starts over: {summary:?}");
    assert_eq!(
        summary.unchanged, 0,
        "no stale vectors may survive: {summary:?}"
    );
    let json = status_json(&corpus);
    assert_eq!(json["model"]["alias"].as_str(), Some("gemma-gguf"));
    assert_eq!(json["total_files"].as_u64(), Some(1));
    let hits = corpus.search(&[
        "--sem",
        "sand moved by the winter wind",
        "--threshold",
        "0.0",
    ]);
    assert_eq!(
        common::rank_of(&hits, "dune"),
        Some(1),
        "the rebuilt index must still answer"
    );

    // And a plain store whose signature is tampered with must refuse loudly:
    // pinned by a unit test in src/store.rs, which can reach the meta table.
}

/// The vectors live in the store itself now, so there is no derived copy to
/// go stale. The failure mode the cache test guarded against — garbage being
/// served as if it were results — must stay impossible: a corrupt store is an
/// error, never quietly wrong answers, and `--clean` recovers.
#[test]
#[serial]
#[ignore = "e2e: needs the default model; run with --include-ignored"]
fn a_corrupt_store_errors_loudly_instead_of_lying() {
    let corpus = Corpus::new();
    corpus.write(
        "saffron.md",
        "Saffron costs more than silver by weight in the bazaar.\n",
    );
    corpus.write(
        "cedar.md",
        "The cedar shades the fountain where the children play.\n",
    );
    corpus.index();
    let hits = corpus.search(&["--sem", "an expensive spice at the market"]);
    assert_eq!(common::rank_of(&hits, "saffron"), Some(1));

    let db = corpus.path().join(".rogo/index.db");
    fs::write(&db, b"NOT A DATABASE, JUST GARBAGE BYTES 0123456789").unwrap();
    let sem_err = run_err(
        corpus
            .rogo()
            .args(["--sem", "expensive spice", "--jsonl", "-q", "."]),
    );
    assert!(
        sem_err.contains("Error"),
        "a corrupt store must fail loudly, got:\n{sem_err}"
    );
    let lex_err = run_err(
        corpus
            .rogo()
            .args(["--lex", "saffron", "--jsonl", "-q", "."]),
    );
    assert!(lex_err.contains("Error"), "got:\n{lex_err}");

    // And the recovery path is exactly the one the message names.
    run_ok(corpus.rogo().args(["--clean", "."]));
    corpus.index();
    let hits = corpus.search(&["--sem", "a tree giving shade by a fountain"]);
    assert_eq!(
        common::rank_of(&hits, "cedar"),
        Some(1),
        "results must be right again after clean + reindex"
    );
}
