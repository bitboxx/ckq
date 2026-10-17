//! End-to-end tests for the index lifecycle: build, incremental update,
//! deletion, `--add`, `.ckignore`, hidden files, robustness against odd
//! inputs, and the safety rails around the manifest's embedding signature
//! and the vector cache. Everything runs the real binary on real files,
//! and almost everything needs the real default model because indexing
//! embeds — hence the `e2e:` ignores.

mod common;

use common::{Corpus, run_err, run_ok};
use serde_json::Value;
use serial_test::serial;
use std::fs;
use std::path::PathBuf;
use std::process::{Command, Output, Stdio};
use std::time::{Duration, Instant};

/// The files the manifest claims, as corpus-relative paths (`a/b.md`), with
/// the leading `./` the manifest stores stripped off.
fn manifest_rels(corpus: &Corpus) -> Vec<String> {
    common::read_manifest(corpus.path())["files"]
        .as_object()
        .expect("manifest.files should be an object")
        .keys()
        .map(|k| k.trim_start_matches("./").to_string())
        .collect()
}

/// Whether the manifest has an entry for `rel`.
fn manifest_has(corpus: &Corpus, rel: &str) -> bool {
    manifest_rels(corpus).iter().any(|r| r == rel)
}

/// The bytes of the sidecar the index keeps for `rel`, or `None` when there
/// is no sidecar.
fn sidecar(corpus: &Corpus, rel: &str) -> Option<Vec<u8>> {
    fs::read(corpus.path().join(".ck").join(format!("{rel}.ck"))).ok()
}

/// Run a command that must finish within `secs`, or fail the test. Indexing
/// walks the tree it is given, so a walker that followed a symlink loop
/// would hang forever; a plain `output()` would hang the test with it.
fn run_bounded(cmd: &mut Command, secs: u64) -> Output {
    cmd.stdin(Stdio::null());
    let mut child = cmd.spawn().expect("ckq should start");
    let deadline = Instant::now() + Duration::from_secs(secs);
    loop {
        match child.try_wait().expect("ckq should be waitable") {
            Some(_) => break,
            None if Instant::now() >= deadline => {
                let _ = child.kill();
                panic!(
                    "ckq {:?} did not finish within {secs}s",
                    cmd.get_args().collect::<Vec<_>>()
                );
            }
            None => std::thread::sleep(Duration::from_millis(100)),
        }
    }
    child.wait_with_output().expect("reap ckq")
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
    let out = run_ok(corpus.ckq().arg("--status-json"));
    serde_json::from_slice(&out.stdout).expect("--status-json should print one JSON object")
}

/// Run a search that must find nothing. `ckq` exits 1 with "No matches
/// found" like grep, so `run_ok` cannot express this; require exactly that
/// outcome so a real error (bad flag, crash) still fails the test.
fn expect_no_hits(corpus: &Corpus, args: &[&str]) {
    let out = corpus
        .ckq()
        .args(args)
        .args(["--jsonl", "-q", "."])
        .output()
        .expect("ckq should start");
    assert!(
        !out.status.success(),
        "ckq {args:?} should have found nothing\n--- stdout\n{}",
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

    // The index directory and its parts. The tantivy lexical index is built
    // by the first lexical search, not by --index itself.
    let ck = corpus.path().join(".ck");
    assert!(ck.join("manifest.json").is_file(), "manifest.json missing");
    assert!(ck.join("vectors.cache").is_file(), "vectors.cache missing");
    for rel in ["lighthouse.md", "garden.md", "nested/harbour.md"] {
        assert!(
            ck.join(format!("{rel}.ck")).is_file(),
            "sidecar for {rel} missing"
        );
    }
    assert!(!ck.join("tantivy_index").exists());
    corpus.search(&["--lex", "lighthouse"]);
    assert!(
        ck.join("tantivy_index").is_dir(),
        "tantivy_index should exist after a lexical search"
    );

    // --status with no path argument reads the current directory; the counts
    // it reports must be the counts the corpus actually has.
    let out = run_ok(corpus.ckq().arg("--status"));
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

    let out = run_ok(corpus.ckq().arg("--status-verbose"));
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

    // The observable for "only that file was re-embedded" is the sidecar of
    // every other file: an untouched file is skipped, so its sidecar must be
    // byte-for-byte what the first run wrote, while the edited file's changes.
    let before: Vec<(String, Vec<u8>)> = ["meteor.md", "bakery.md", "ferry.md"]
        .iter()
        .map(|rel| ((*rel).to_string(), sidecar(&corpus, rel).expect("sidecar")))
        .collect();

    corpus.write(
        "bakery.md",
        "The bakery opens before sunrise and the street smells of warm bread.\n\
         By nine the shelves hold cardamom buns and the queue reaches the corner.\n",
    );
    corpus.index();

    for (rel, bytes) in &before {
        let now = sidecar(&corpus, rel).unwrap_or_else(|| panic!("sidecar for {rel} vanished"));
        if rel == "bakery.md" {
            assert_ne!(now, *bytes, "edited file's sidecar was not rewritten");
        } else {
            assert_eq!(
                now, *bytes,
                "unchanged file {rel} was re-embedded by an incremental run"
            );
        }
    }
    assert_eq!(manifest_rels(&corpus).len(), 3, "entries were lost");

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
fn deleted_file_disappears_from_search_and_clean_orphans_drops_its_sidecar() {
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
    corpus.index();

    for (mode, query) in mushroom_queries {
        expect_no_hits(&corpus, &[mode, query]);
    }
    // The survivor is untouched by all of this.
    let hits = corpus.search(&["--sem", "high water marks this season"]);
    assert_eq!(common::rank_of(&hits, "tide"), Some(1));

    // --index leaves the dead sidecar (and its manifest entry) behind;
    // --clean-orphans is the tool for exactly that.
    assert!(
        sidecar(&corpus, "mushroom.md").is_some(),
        "expected a leftover sidecar to exercise --clean-orphans"
    );
    run_ok(corpus.ckq().args(["--clean-orphans", "."]));
    assert!(
        sidecar(&corpus, "mushroom.md").is_none(),
        "--clean-orphans did not remove the orphaned sidecar"
    );
    assert!(
        !manifest_has(&corpus, "mushroom.md"),
        "--clean-orphans did not drop the orphaned manifest entry"
    );
    assert!(
        sidecar(&corpus, "tide.md").is_some(),
        "--clean-orphans removed a live sidecar"
    );
    expect_no_hits(&corpus, &["--lex", "mushrooms"]);
    // --clean ends the lifecycle: the whole index directory goes away, and
    // the next lexical search rebuilds it from the files rather than
    // serving anything stale from the old one.
    run_ok(corpus.ckq().args(["--clean", "."]));
    assert!(
        !corpus.path().join(".ck").exists(),
        "--clean left .ck behind"
    );
    let hits = corpus.search(&["--lex", "tide"]);
    assert_eq!(
        common::rank_of(&hits, "tide"),
        Some(1),
        "a search after --clean should rebuild and find the survivor"
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
    run_ok(corpus.ckq().args(["--add", "lantern.md"]));

    assert_eq!(status_json(&corpus)["total_files"].as_u64(), Some(3));
    assert!(manifest_has(&corpus, "lantern.md"));
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
fn ckignore_excludes_files_from_the_start_and_when_added_later() {
    let corpus = Corpus::new();
    corpus.write(".ckignore", "# test ignore list\nprivate_*.md\n");
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
    assert!(!manifest_has(&corpus, "private_journal.md"));
    expect_no_hits(&corpus, &["--lex", "anthem"]);
    assert!(manifest_has(&corpus, "draft_letter.md"));
    assert_eq!(
        common::rank_of(&corpus.search(&["--lex", "parsnip"]), "draft_letter"),
        Some(1)
    );

    // The same file, ignored only after the fact, must drop out of results.
    let mut ignore = fs::read_to_string(corpus.path().join(".ckignore")).unwrap();
    ignore.push_str("draft_letter.md\n");
    fs::write(corpus.path().join(".ckignore"), ignore).unwrap();
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

    assert_eq!(manifest_rels(&corpus), vec!["hall.md"]);
    expect_no_hits(&corpus, &["--lex", "quilts"]);

    run_ok(corpus.ckq().args(["--index", "--hidden", "-q", "."]));
    let rels = manifest_rels(&corpus);
    for rel in [".attic/quilts.md", ".cellar.md", "hall.md"] {
        assert!(rels.contains(&rel.to_string()), "missing {rel} in {rels:?}");
    }
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
    assert!(!manifest_has(&corpus, "blob_data.bin"));
    assert!(sidecar(&corpus, "blob_data.bin").is_none());
    assert!(manifest_has(&corpus, "empty_page.md"));
    assert!(manifest_has(&corpus, "nested/deep/glacier.md"));

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

    let out = run_bounded(corpus.ckq().args(["--index", "-q", "."]), 120);
    assert!(
        out.status.success(),
        "index over a symlink loop failed\n{}",
        String::from_utf8_lossy(&out.stderr)
    );
    assert_eq!(manifest_rels(&corpus), vec!["anchor.md"]);
    // Status is command mode, so it walks nothing and must be immediate.
    assert_eq!(status_json(&corpus)["total_files"].as_u64(), Some(1));
}

/// BUG: a lexical search on a corpus containing a symlink loop never
/// returns. `--index` walks the same tree fine; the search path expands its
/// target with the `glob` crate (`path_utils::run_glob`), which follows the
/// looped directory symlinks forever. Bounded so the failure is a panic,
/// not a hung run.
#[cfg(unix)]
#[test]
#[serial]
#[ignore = "bug: --lex (any search) hangs on a corpus with a symlink loop; run with --include-ignored"]
fn known_bug_lexical_search_survives_a_symlink_loop() {
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
        corpus.ckq().args(["--lex", "anchor", "--jsonl", "-q", "."]),
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
    assert_eq!(manifest_rels(&corpus), vec!["open_hall.md"]);
    assert_eq!(
        common::rank_of(&corpus.search(&["--lex", "afternoon"]), "open_hall"),
        Some(1),
        "the readable file must still be searchable"
    );

    // Once readable again it joins the index on the next run.
    fs::set_permissions(&vault, fs::Permissions::from_mode(0o644)).unwrap();
    corpus.index();
    assert!(manifest_has(&corpus, "locked_vault.md"));
    assert_eq!(
        common::rank_of(&corpus.search(&["--lex", "vault"]), "locked_vault"),
        Some(1)
    );
}

/// The manifest's embedding signature guards against mixing vectors from
/// different prompt spaces. A mismatch must be refused loudly, not merged.
#[test]
#[serial]
#[ignore = "e2e: needs the default model; run with --include-ignored"]
fn embedding_signature_mismatch_refuses_to_reindex() {
    let corpus = Corpus::new();
    corpus.write(
        "dune.md",
        "The dune moves a hand's width inland with every winter storm.\n",
    );
    corpus.index();

    let manifest = corpus.path().join(".ck/manifest.json");
    let mut tampered = common::read_manifest(corpus.path());
    tampered["embedding_signature"] = Value::String("tampered-signature".into());
    fs::write(&manifest, serde_json::to_vec_pretty(&tampered).unwrap()).unwrap();

    let stderr = run_err(corpus.ckq().args(["--index", "-q", "."]));
    assert!(
        stderr.contains("different embedding prompts"),
        "refusal should name the problem, got:\n{stderr}"
    );
    assert!(
        stderr.contains("--clean"),
        "refusal should say what to do, got:\n{stderr}"
    );
    // Refused means untouched: the index it refused to mix is still there.
    assert!(manifest_has(&corpus, "dune.md"));
    assert!(sidecar(&corpus, "dune.md").is_some());
}

/// The vector cache is only ever a copy; garbage or a truncated cache must
/// be ignored or rebuilt without changing what search returns.
#[test]
#[serial]
#[ignore = "e2e: needs the default model; run with --include-ignored"]
fn garbage_and_truncated_vectors_cache_keep_results_right() {
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
    let cache = corpus.path().join(".ck/vectors.cache");
    assert!(cache.is_file());

    // Garbage in the cache: a miss, not a crash, and results stay right.
    let garbage: &[u8] = b"NOT A CACHE, JUST GARBAGE BYTES 0123456789";
    fs::write(&cache, garbage).unwrap();
    let hits = corpus.search(&["--sem", "an expensive spice at the market"]);
    assert_eq!(
        common::rank_of(&hits, "saffron"),
        Some(1),
        "garbage cache changed the results"
    );
    assert_ne!(
        fs::read(&cache).unwrap(),
        garbage,
        "the cache should have been rebuilt over the garbage"
    );

    // A truncated cache: the first bytes of a real one, cut short.
    let rebuilt = fs::read(&cache).unwrap();
    assert!(rebuilt.len() > 200, "rebuilt cache is suspiciously small");
    fs::write(&cache, &rebuilt[..100]).unwrap();
    let hits = corpus.search(&["--sem", "a tree giving shade by a fountain"]);
    assert_eq!(
        common::rank_of(&hits, "cedar"),
        Some(1),
        "truncated cache changed the results"
    );
    // Lexical search never touches the cache and must be unaffected too.
    assert_eq!(
        common::rank_of(&corpus.search(&["--lex", "saffron"]), "saffron"),
        Some(1)
    );
}
