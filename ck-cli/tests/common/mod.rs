//! Shared helpers for the end-to-end tests.
//!
//! The e2e tests run the real `ckq` binary on real files and, where a test says
//! so, on the real default model. Those tests are `#[ignore]`d with a reason
//! that starts `e2e:`, so a plain `cargo test` stays offline and quick. Run
//! everything with
//!
//! ```text
//! cargo test --release -p ck-search -- --include-ignored
//! ```
//!
//! CI does, with the model cache restored. A test that needs the model must
//! assert on what the model returns: a test that passes when the search fails
//! proves nothing, and the old semantic tests did exactly that.

#![allow(dead_code)] // each test file uses a different subset

use serde_json::Value;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::{Command, Output};
use tempfile::TempDir;

pub fn ckq_bin() -> PathBuf {
    PathBuf::from(env!("CARGO_BIN_EXE_ckq"))
}

/// `ckq` with every setting that changes its behaviour cleared, so a test sees
/// the defaults whatever the developer's shell exports. The daemon gets a short
/// idle timeout: each test build is a new executable and so a new daemon.
pub fn ckq() -> Command {
    let mut cmd = Command::new(ckq_bin());
    for var in [
        "CK_INDEX_DIR",
        "CKQ_IN_PROCESS",
        "CKQ_MODEL_ALIAS",
        "CKQ_GGUF_FILE",
        "CKQ_ONNX_FILE",
        "CKQ_PIN",
        "CKQ_TIMING",
        "CKQ_COREML",
    ] {
        cmd.env_remove(var);
    }
    cmd.env("CKQ_DAEMON_IDLE_SECS", "60");
    cmd
}

/// Run a command and require success, with both streams in the failure message.
pub fn run_ok(cmd: &mut Command) -> Output {
    let out = cmd.output().expect("ckq should start");
    assert!(
        out.status.success(),
        "ckq {:?} exited {}\n--- stdout\n{}\n--- stderr\n{}",
        cmd.get_args().collect::<Vec<_>>(),
        out.status,
        String::from_utf8_lossy(&out.stdout),
        String::from_utf8_lossy(&out.stderr),
    );
    out
}

/// Run a command and require failure; returns stderr.
pub fn run_err(cmd: &mut Command) -> String {
    let out = cmd.output().expect("ckq should start");
    assert!(
        !out.status.success(),
        "ckq {:?} should have failed\n--- stdout\n{}",
        cmd.get_args().collect::<Vec<_>>(),
        String::from_utf8_lossy(&out.stdout),
    );
    String::from_utf8_lossy(&out.stderr).into_owned()
}

/// A directory of files that lives as long as the test.
pub struct Corpus {
    pub dir: TempDir,
}

impl Corpus {
    pub fn new() -> Self {
        Self {
            dir: TempDir::new().expect("temp dir"),
        }
    }

    pub fn path(&self) -> &Path {
        self.dir.path()
    }

    /// Write a file relative to the corpus root, creating parent directories.
    pub fn write(&self, rel: &str, content: impl AsRef<[u8]>) -> PathBuf {
        let path = self.path().join(rel);
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent).expect("create parent");
        }
        fs::write(&path, content).expect("write file");
        path
    }

    pub fn remove(&self, rel: &str) {
        fs::remove_file(self.path().join(rel)).expect("remove file");
    }

    /// `ckq` run from the corpus root.
    pub fn ckq(&self) -> Command {
        let mut cmd = ckq();
        cmd.current_dir(self.path());
        cmd
    }

    /// Build or update the index, and require success.
    pub fn index(&self) -> Output {
        run_ok(self.ckq().args(["--index", "-q", "."]))
    }

    /// Run a search with `--jsonl` appended and parse every result line.
    /// `args` holds the mode, the query and any flags; the path is added.
    pub fn search(&self, args: &[&str]) -> Vec<Hit> {
        let out = run_ok(self.ckq().args(args).args(["--jsonl", "-q", "."]));
        parse_jsonl(&out.stdout)
    }
}

impl Default for Corpus {
    fn default() -> Self {
        Self::new()
    }
}

/// One result line of `--jsonl`.
#[derive(Debug, Clone)]
pub struct Hit {
    pub path: PathBuf,
    pub score: Option<f64>,
    pub lex_rank: Option<u64>,
    pub vec_rank: Option<u64>,
    pub raw: Value,
}

impl Hit {
    /// File name without its extension, which is the document id in the bench.
    pub fn stem(&self) -> String {
        self.path
            .file_stem()
            .map(|s| s.to_string_lossy().into_owned())
            .unwrap_or_default()
    }

    pub fn file_name(&self) -> String {
        self.path
            .file_name()
            .map(|s| s.to_string_lossy().into_owned())
            .unwrap_or_default()
    }
}

/// Parse `--jsonl` output. Lines that are not JSON objects fail the test,
/// because the format promises one object per line.
pub fn parse_jsonl(stdout: &[u8]) -> Vec<Hit> {
    String::from_utf8_lossy(stdout)
        .lines()
        .filter(|l| !l.trim().is_empty())
        .map(|line| {
            let raw: Value = serde_json::from_str(line)
                .unwrap_or_else(|e| panic!("not a JSON line ({e}): {line}"));
            let path = raw
                .get("path")
                .or_else(|| raw.get("file"))
                .and_then(Value::as_str)
                .unwrap_or_else(|| panic!("result without a path: {line}"));
            let signals = raw.get("signals");
            Hit {
                path: PathBuf::from(path),
                score: raw.get("score").and_then(Value::as_f64),
                lex_rank: signals
                    .and_then(|s| s.get("lex_rank"))
                    .and_then(Value::as_u64),
                vec_rank: signals
                    .and_then(|s| s.get("vec_rank"))
                    .and_then(Value::as_u64),
                raw,
            }
        })
        .collect()
}

/// 1-based rank of the first result whose file stem is `stem`, counting each
/// file once, as the bench does.
pub fn rank_of(hits: &[Hit], stem: &str) -> Option<usize> {
    let mut seen: Vec<String> = Vec::new();
    for hit in hits {
        let s = hit.stem();
        if !seen.contains(&s) {
            seen.push(s.clone());
            if s == stem {
                return Some(seen.len());
            }
        }
    }
    None
}

pub struct BenchQuery {
    pub q: String,
    pub lang: String,
    pub answer: String,
}

/// The public cross-lingual benchmark in `bench/multilingual`: 40 documents in
/// five languages and 40 queries, each asked in a language other than its
/// answer's. Written out exactly as `run.py` does, so the scores match it.
pub struct Bench {
    pub corpus: Corpus,
    pub queries: Vec<BenchQuery>,
}

pub fn bench() -> Bench {
    let file = Path::new(env!("CARGO_MANIFEST_DIR")).join("../bench/multilingual/corpus.json");
    let data: Value =
        serde_json::from_str(&fs::read_to_string(&file).expect("read bench corpus")).unwrap();
    let corpus = Corpus::new();
    for doc in data["documents"].as_array().unwrap() {
        corpus.write(
            &format!("{}.md", doc["id"].as_str().unwrap()),
            format!(
                "# {}\n\n{}\n",
                doc["title"].as_str().unwrap(),
                doc["text"].as_str().unwrap()
            ),
        );
    }
    let queries = data["queries"]
        .as_array()
        .unwrap()
        .iter()
        .map(|q| BenchQuery {
            q: q["q"].as_str().unwrap().to_string(),
            lang: q["lang"].as_str().unwrap().to_string(),
            answer: q["answer"].as_str().unwrap().to_string(),
        })
        .collect();
    Bench { corpus, queries }
}

/// Rank-1 count and MRR@5 of a search mode over the bench.
pub struct BenchScore {
    pub first: usize,
    pub mrr5: f64,
    pub misses: Vec<String>,
}

pub fn score_bench(bench: &Bench, mode: &[&str]) -> BenchScore {
    let mut first = 0;
    let mut rr = 0.0;
    let mut misses = Vec::new();
    for q in &bench.queries {
        let mut args: Vec<&str> = mode.to_vec();
        args.extend([
            q.q.as_str(),
            "--topk",
            "15",
            "--threshold",
            "0.0",
            "--no-snippet",
        ]);
        let hits = bench.corpus.search(&args);
        match rank_of(&hits, &q.answer) {
            Some(1) => {
                first += 1;
                rr += 1.0;
            }
            Some(r) => {
                if r <= 5 {
                    rr += 1.0 / r as f64;
                }
                misses.push(format!("{} -> {} at {r}", q.q, q.answer));
            }
            None => misses.push(format!("{} -> {} not found", q.q, q.answer)),
        }
    }
    BenchScore {
        first,
        mrr5: rr / bench.queries.len() as f64,
        misses,
    }
}

/// The hits with one entry per file, the first hit of each file kept, in
/// order. Search modes that report chunks can repeat a file; rankings and
/// order comparisons want it once.
pub fn uniq_hits(hits: &[Hit]) -> Vec<Hit> {
    let mut seen: Vec<String> = Vec::new();
    let mut out = Vec::new();
    for hit in hits {
        let stem = hit.stem();
        if !seen.contains(&stem) {
            seen.push(stem);
            out.push(hit.clone());
        }
    }
    out
}

/// The parsed `.ck/manifest.json` of the index rooted at `root`, for tests
/// that assert on which files the index believes it has.
pub fn read_manifest(root: &Path) -> Value {
    let data = fs::read(root.join(".ck").join("manifest.json"))
        .expect(".ck/manifest.json should exist after --index");
    serde_json::from_slice(&data).expect(".ck/manifest.json should be valid JSON")
}
