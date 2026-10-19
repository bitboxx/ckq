//! The reranker through the real binary: the input format is only proven if
//! an obviously relevant passage outscores an obviously irrelevant one, in
//! more than one language, and lands first.

mod common;

use common::{Corpus, Hit, ckq, run_ok};
use serial_test::serial;

/// The toy corpus: two coffee-storage notes (English and Mandarin) and
/// same-shape notes on other subjects, so a cross-encoder that reads the
/// passage cannot be fooled by file names.
fn toy() -> Corpus {
    let corpus = Corpus::new();
    corpus.write(
        "coffee_en.md",
        "# Storing coffee beans\n\nKeep coffee beans in an airtight container away from light, \
         heat and moisture. Whole beans stay fresh for about a month after roasting; ground \
         coffee loses its aroma within days. Never store beans in the refrigerator, where \
         condensation spoils them.\n",
    );
    corpus.write(
        "coffee_zh.md",
        "# 咖啡豆的保存方法\n\n咖啡豆应存放在密封容器中，避光、避热、避潮湿。整颗咖啡豆在烘焙后大约一个月内保持新鲜，\
         而磨好的咖啡粉几天内就会失去香气。不要把咖啡豆放在冰箱里，冷凝水会破坏风味。\n",
    );
    corpus.write(
        "python_gcd.md",
        "# The greatest common divisor in Python\n\nThe Euclidean algorithm computes the \
         greatest common divisor of two integers by repeated modulo reduction. In Python, \
         math.gcd(a, b) does this for you.\n",
    );
    corpus.write(
        "solar_panel.md",
        "# Solar panel installation\n\nA rooftop solar array faces south in the northern \
         hemisphere, tilted at the latitude angle. String inverters convert the DC output \
         to AC.\n",
    );
    corpus
}

fn rerank_hits(corpus: &Corpus, query: &str) -> Vec<Hit> {
    let mut cmd = ckq();
    cmd.args([
        "--hybrid",
        "--rerank",
        "--jsonl",
        "--no-snippet",
        "--topk",
        "4",
        "--threshold",
        "0.0",
        "-q",
        query,
    ])
    .arg(corpus.dir.path())
    .env("CKQ_IN_PROCESS", "1")
    .env_remove("CKQ_RERANK_TOPN")
    .env_remove("CKQ_RERANK_MODE")
    .env_remove("CKQ_RERANK_WEIGHT")
    .env_remove("CKQ_RERANKER");
    let out = run_ok(&mut cmd);
    common::parse_jsonl(&out.stdout)
}

#[test]
#[serial]
#[ignore = "e2e: needs the default model and the bge-reranker; run with --include-ignored"]
fn the_reranker_ranks_the_relevant_toy_passage_first_in_two_languages() {
    let corpus = toy();
    corpus.index();

    for (query, relevant, irrelevant) in [
        ("how to store coffee beans", "coffee_en", "python_gcd"),
        ("咖啡豆怎么保存", "coffee_zh", "solar_panel"),
        // A third language the corpus does not even contain: the
        // cross-encoder reads the passage, not the query's words.
        (
            "cómo conservar los granos de café",
            "coffee_zh",
            "python_gcd",
        ),
    ] {
        let hits = rerank_hits(&corpus, query);
        assert!(!hits.is_empty(), "{query}: no results");
        let first = &hits[0];
        assert_eq!(
            first.stem(),
            relevant,
            "{query}: the relevant note must lead, got {:?}",
            hits.iter().map(|h| h.stem()).collect::<Vec<_>>()
        );
        // The wide-margin proof of the input format: relevant far above
        // irrelevant, not a coin-flip ordering.
        let score_of = |stem: &str| {
            hits.iter()
                .find(|h| h.stem() == stem)
                .and_then(Hit::rerank_score)
                .unwrap_or_else(|| panic!("{query}: {stem} carries no rerank_score"))
        };
        let (good, bad) = (score_of(relevant), score_of(irrelevant));
        assert!(
            good - bad > 3.0,
            "{query}: relevant {good:.3} must beat irrelevant {bad:.3} by a wide margin"
        );
        assert_eq!(
            hits[0].rerank_rank(),
            Some(1),
            "{query}: the leader carries the reranker's first rank"
        );
    }
}
