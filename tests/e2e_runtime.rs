//! Runtime behaviour end to end, through the real binary and the real
//! default model: the query daemon's weights identity, its client's timeout
//! and fallback, and the healing of a stolen socket file. These need the
//! model, so they carry the `e2e:` ignore like the rest of the suites.

mod common;

use common::{Corpus, run_err, run_ok};
use serial_test::serial;
#[cfg(unix)]
use std::os::unix::net::UnixListener;
use std::path::PathBuf;
use std::time::{Duration, Instant};

fn runtime_corpus() -> Corpus {
    let corpus = Corpus::new();
    corpus.write("stew.md", "# stew\n\nthe chef simmers the stew for hours\n");
    corpus.write("volcano.md", "# volcano\n\nmagma rises through the vent\n");
    corpus
}

/// The gemma-q4 GGUF the cache holds, in the layout hf-hub leaves it in.
fn cached_gguf() -> PathBuf {
    let home = PathBuf::from(std::env::var("HOME").expect("HOME is set"));
    let snapshots = home
        .join(".cache/ckq/models")
        .join("models--ggml-org--embeddinggemma-300M-qat-q4_0-GGUF")
        .join("snapshots");
    std::fs::read_dir(&snapshots)
        .expect("the model cache is populated (run any e2e suite first)")
        .flatten()
        .map(|rev| rev.path().join("embeddinggemma-300M-qat-Q4_0.gguf"))
        .find(|path| path.is_file())
        .expect("the gemma-q4 GGUF in the cache")
}

/// A replaced `CKQ_GGUF_FILE` must not keep the old daemon's socket key, and
/// a search whose weights no longer match the index must refuse rather than
/// embed the query in one space while the documents live in another. The
/// trial file is a symlink so its path stays constant while the file behind
/// it changes — exactly what the socket key used to miss.
#[test]
#[serial]
#[ignore = "e2e: needs the default model; run with --include-ignored"]
#[cfg(unix)]
fn replaced_weights_get_a_new_daemon_and_a_refused_search() {
    let corpus = runtime_corpus();
    let real = cached_gguf();
    let trial_dir = tempfile::TempDir::new().unwrap();
    let trial = trial_dir.path().join("trial.gguf");
    std::os::unix::fs::symlink(&real, &trial).unwrap();

    let mut cmd = corpus.ckq();
    cmd.env("CKQ_GGUF_FILE", &trial)
        .args(["--index", "-q", "."]);
    run_ok(&mut cmd);

    let mut cmd = corpus.ckq();
    cmd.env("CKQ_GGUF_FILE", &trial)
        .args(["--sem", "stew", "--jsonl", "--no-snippet", "-q", "."]);
    let out = run_ok(&mut cmd);
    assert!(
        String::from_utf8_lossy(&out.stdout).contains("stew.md"),
        "the override search must work while the weights match"
    );
    let mut cmd = corpus.ckq();
    cmd.env("CKQ_GGUF_FILE", &trial)
        .args(["--daemon-socket", "gemma-q4"]);
    let out = run_ok(&mut cmd);
    let with_override = PathBuf::from(String::from_utf8(out.stdout).unwrap().trim());
    assert!(
        with_override.exists(),
        "the override search started a daemon"
    );

    // Replace the file behind the same path: same name, other bytes.
    std::fs::remove_file(&trial).unwrap();
    std::fs::write(&trial, b"these are not the weights that were indexed").unwrap();

    // The socket key follows the weights, so a changed file means a new
    // daemon, not the resident one holding the old bytes.
    let mut cmd = corpus.ckq();
    cmd.env("CKQ_GGUF_FILE", &trial)
        .args(["--daemon-socket", "gemma-q4"]);
    let out = run_ok(&mut cmd);
    let repointed = PathBuf::from(String::from_utf8(out.stdout).unwrap().trim());
    assert_ne!(
        with_override, repointed,
        "a replaced override must change the socket key"
    );

    // And the search refuses, naming the rebuild, instead of mixing.
    let mut cmd = corpus.ckq();
    cmd.env("CKQ_GGUF_FILE", &trial)
        .args(["--sem", "stew", "--jsonl", "--no-snippet", "-q", "."]);
    let stderr = run_err(&mut cmd);
    assert!(
        stderr.contains("different weights or settings"),
        "stderr: {stderr}"
    );
    assert!(stderr.contains("--index"), "stderr: {stderr}");

    common::kill_daemon(&with_override);
    common::kill_daemon(&repointed);
}

/// A daemon that accepts and closes must not fail the search: the client
/// warns and embeds in its own process, the same fallback a daemon that
/// cannot be started gets.
#[test]
#[serial]
#[ignore = "e2e: needs the default model; run with --include-ignored"]
#[cfg(unix)]
fn a_closing_daemon_is_fallen_back_from() {
    let corpus = runtime_corpus();
    corpus.index();
    let socket = common::daemon_socket("gemma-q4");
    common::kill_daemon(&socket);
    let _ = std::fs::remove_file(&socket);

    let listener = UnixListener::bind(&socket).expect("bind the fake daemon");
    let faker = std::thread::spawn(move || {
        if let Ok((stream, _)) = listener.accept() {
            drop(stream); // accept, then close: the half-dead daemon
            std::thread::sleep(Duration::from_secs(5));
        }
    });

    let out = run_ok(
        corpus
            .ckq()
            .args(["--sem", "stew", "--jsonl", "--no-snippet", "-q", "."]),
    );
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(
        stderr.contains("embedding the query in this process"),
        "stderr: {stderr}"
    );
    assert!(
        String::from_utf8_lossy(&out.stdout).contains("stew.md"),
        "the fallback search must still find the file"
    );

    let _ = faker.join();
    let _ = std::fs::remove_file(&socket);
}

/// A daemon that accepts and goes silent must not hang the search: the
/// client's read times out and the same in-process fallback runs.
#[test]
#[serial]
#[ignore = "e2e: needs the default model; run with --include-ignored"]
#[cfg(unix)]
fn a_silent_daemon_times_out_and_falls_back() {
    let corpus = runtime_corpus();
    corpus.index();
    let socket = common::daemon_socket("gemma-q4");
    common::kill_daemon(&socket);
    let _ = std::fs::remove_file(&socket);

    let listener = UnixListener::bind(&socket).expect("bind the fake daemon");
    let faker = std::thread::spawn(move || {
        if let Ok((stream, _)) = listener.accept() {
            use std::io::{BufRead, BufReader};
            let mut line = String::new();
            let _ = BufReader::new(stream).read_line(&mut line); // take the query
            std::thread::sleep(Duration::from_secs(5)); // ... and never answer
        }
    });

    let start = Instant::now();
    let out = run_ok(corpus.ckq().env("CKQ_DAEMON_TIMEOUT_SECS", "2").args([
        "--sem",
        "stew",
        "--jsonl",
        "--no-snippet",
        "-q",
        ".",
    ]));
    // Long enough for the 2 s timeout to have fired, short enough that no
    // 120 s default could have been waited out.
    assert!(
        start.elapsed() < Duration::from_secs(60),
        "the silent daemon was waited out"
    );
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(
        stderr.contains("embedding the query in this process"),
        "stderr: {stderr}"
    );
    assert!(String::from_utf8_lossy(&out.stdout).contains("stew.md"));

    let _ = faker.join();
    let _ = std::fs::remove_file(&socket);
}

/// A live daemon whose socket file is unlinked underneath it must reclaim
/// the path instead of orphaning the model: the file comes back within a
/// couple of socket-check intervals and the daemon keeps serving.
#[test]
#[serial]
#[ignore = "e2e: needs the default model; run with --include-ignored"]
#[cfg(unix)]
fn an_unlinked_daemon_socket_heals() {
    let corpus = runtime_corpus();
    corpus.index();
    let socket = common::daemon_socket("gemma-q4");

    let hits = corpus.search(&["--sem", "stew"]);
    assert!(!hits.is_empty(), "the warm-up search found nothing");
    assert!(socket.exists(), "the search started a daemon");

    // Steal the socket file the way the check-then-unlink races can.
    std::fs::remove_file(&socket).unwrap();

    let deadline = Instant::now() + Duration::from_secs(10);
    while !socket.exists() && Instant::now() < deadline {
        std::thread::sleep(Duration::from_millis(100));
    }
    assert!(
        socket.exists(),
        "the daemon did not reclaim its socket within 10 s"
    );

    let hits = corpus.search(&["--sem", "stew"]);
    assert!(!hits.is_empty(), "the healed daemon no longer serves");
    common::kill_daemon(&socket);
}
