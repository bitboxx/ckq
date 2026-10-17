//! Retrieval quality on the public cross-lingual bench, through the real binary
//! and the real default model. This is the gate for any change to the model,
//! the prompts, the chunker, the vector store or the fusion: if it moves these
//! numbers, it changed what users get back.

mod common;

use serial_test::serial;

/// gemma-q4 scores 35/40 rank 1 and MRR@5 0.927 here (26 Sep 2026). One query of
/// slack covers float differences between Metal, CPU and other platforms.
const MIN_FIRST: usize = 34;
const MIN_MRR5: f64 = 0.90;

#[test]
#[serial]
#[ignore = "e2e: needs the default model; run with --include-ignored"]
fn semantic_search_holds_the_bench_score() {
    let bench = common::bench();
    bench.corpus.index();
    let s = common::score_bench(&bench, &["--sem"]);
    eprintln!("--sem: {}/40 rank 1, MRR@5 {:.3}", s.first, s.mrr5);
    assert!(
        s.first >= MIN_FIRST && s.mrr5 >= MIN_MRR5,
        "--sem: {}/40 rank 1, MRR@5 {:.3} (need {MIN_FIRST}, {MIN_MRR5})\n{}",
        s.first,
        s.mrr5,
        s.misses.join("\n")
    );
}

#[test]
#[serial]
#[ignore = "e2e: needs the default model; run with --include-ignored"]
fn hybrid_search_holds_the_bench_score() {
    let bench = common::bench();
    bench.corpus.index();
    let s = common::score_bench(&bench, &["--hybrid"]);
    eprintln!("--hybrid: {}/40 rank 1, MRR@5 {:.3}", s.first, s.mrr5);
    assert!(
        s.first >= MIN_FIRST && s.mrr5 >= MIN_MRR5,
        "--hybrid: {}/40 rank 1, MRR@5 {:.3} (need {MIN_FIRST}, {MIN_MRR5})\n{}",
        s.first,
        s.mrr5,
        s.misses.join("\n")
    );
}
