//! Search behaviour end to end, through the real binary and the real default
//! model: what each mode ranks first, how `--term`, `--threshold` and
//! `--topk` shape the results, that a search is deterministic, that two runs
//! of the now single (daemonless) path agree, that concurrent searches agree,
//! and that the index follows an edited file. These replace the three old
//! integration tests that asserted nothing whenever the command failed.

mod common;

use common::{Corpus, Hit, rank_of, uniq_hits};
use serial_test::serial;
use std::process::Stdio;

// ---------------------------------------------------------------------------
// The three modes rank the right file first
// ---------------------------------------------------------------------------

/// `--sem` must rank the one on-topic file first among off-topic files, not
/// merely return something. A regression in the embed call, the vector store
/// or the scoring that shuffled the order would fail here.
#[test]
#[serial]
#[ignore = "e2e: needs the default model; run with --include-ignored"]
fn sem_ranks_the_on_topic_file_first() {
    let corpus = Corpus::new();
    corpus.write(
        "volcano.md",
        "# Volcanoes\n\nMagma rises from the mantle, dissolved gases expand as pressure drops, \
and the vent fractures: that is why a volcano erupts, sending lava and ash over the valley.\n",
    );
    corpus.write(
        "sourdough.md",
        "# Sourdough\n\nFeed the starter with flour and water every morning, and let the dough \
rise overnight before baking the loaf.\n",
    );
    corpus.write(
        "chess.md",
        "# Chess\n\nThe Sicilian defence answers the king's pawn with a pawn of your own, and the \
endgame rewards passed pawns and an active king.\n",
    );
    corpus.write(
        "alpine_flowers.md",
        "# Alpine flowers\n\nGentians and edelweiss flower late in the season, after the snow \
has left the high meadow and the pollinators have arrived.\n",
    );
    corpus.write(
        "shipping.md",
        "# Container shipping\n\nA feeder vessel carries boxes between the hub port and the \
smaller harbours, and the manifest lists every container by number.\n",
    );
    corpus.index();

    let hits = corpus.search(&["--sem", "what makes a volcano erupt", "--threshold", "0.0"]);
    assert_eq!(
        rank_of(&hits, "volcano"),
        Some(1),
        "stems: {:?}",
        stems(&hits)
    );
}

/// `--lex` must rank the file carrying the term first and must not return a
/// file that does not contain it at all. Catches a broken BM25 query or an
/// index built over the wrong corpus.
#[test]
#[serial]
#[ignore = "e2e: needs the default model; run with --include-ignored"]
fn lex_ranks_the_file_with_the_terms_and_returns_no_other() {
    let corpus = Corpus::new();
    corpus.write(
        "marsupials.md",
        "# Quokkas\n\nThe quokka is a small marsupial that lives on islands off the coast; it \
naps in the shade and eats leaves at dusk.\n",
    );
    corpus.write(
        "owls.md",
        "# Owls\n\nThe barn owl hunts mice over the meadow at night, flying on silent wings.\n",
    );
    corpus.write(
        "reef.md",
        "# Reefs\n\nCoral grows slowly in warm, clear water where the current brings food to \
the polyps.\n",
    );
    corpus.index();

    let hits = corpus.search(&["--lex", "quokka"]);
    assert_eq!(
        rank_of(&hits, "marsupials"),
        Some(1),
        "stems: {:?}",
        stems(&hits)
    );
    assert_eq!(
        uniq_hits(&hits)
            .iter()
            .map(|h| h.stem())
            .collect::<Vec<_>>(),
        vec!["marsupials"],
        "a file without the term must not be returned"
    );
}

/// `--hybrid` must return the expected file first: the file both methods can
/// find beats files neither can. Catches a fusion that ignores one method or
/// mis-sorts by incomparable scores.
#[test]
#[serial]
#[ignore = "e2e: needs the default model; run with --include-ignored"]
fn hybrid_returns_the_expected_file_first() {
    let corpus = Corpus::new();
    corpus.write(
        "bike.md",
        "# Flat tyres\n\nTo fix a flat bicycle tyre, take the wheel off, find the puncture, \
roughen the tube and glue on a patch before pumping it up again.\n",
    );
    corpus.write(
        "pastry.md",
        "# Pastry\n\nKeep the butter cold, fold the dough in thirds, and rest it in the fridge \
between each turn.\n",
    );
    corpus.write(
        "tidepools.md",
        "# Tidepools\n\nAt low tide the pools hold anemones, crabs and small fish trapped \
until the water returns.\n",
    );
    corpus.index();

    let hits = corpus.search(&["--hybrid", "how to fix a flat bicycle tyre"]);
    assert_eq!(rank_of(&hits, "bike"), Some(1), "stems: {:?}", stems(&hits));
}

// ---------------------------------------------------------------------------
// Hybrid fusion
// ---------------------------------------------------------------------------

/// The fusion corpus for `--term` tests: `inventory.md` is findable only by
/// the lexical method through the part number, `workshop.md` only by the
/// semantic method (a paraphrase sharing no word with the query), and
/// `guide.md` by both. `query` drives the semantic method.
fn fusion_corpus() -> (Corpus, &'static str) {
    let corpus = Corpus::new();
    let query = "how do I fix a flat bicycle tyre";
    corpus.write(
        "inventory.md",
        "# Warehouse\n\nCrate four holds the hydraulic seal rings ordered under part KX4471; \
inspection is scheduled for Thursday and the crates must stay sealed until then.\n",
    );
    corpus.write(
        "workshop.md",
        "# Home workshop\n\nMending a punctured wheel: remove the wheel, find the hole, \
roughen the rubber, then glue on the patch and reinflate the tube.\n",
    );
    corpus.write(
        "guide.md",
        "# Tyre repair\n\nA flat bicycle tyre is fixed with tyre levers, a patch kit and a \
pump; follow the guide and the repair holds for years.\n",
    );
    corpus.write(
        "endgame.md",
        "# Endgame\n\nRook endings are drawn when the defending king reaches the corner in \
time; count the tempo before you trade.\n",
    );
    corpus.write(
        "soups.md",
        "# Soups\n\nSimmer the bones slowly, skim the fat, and finish the bowl with lemon \
and herbs.\n",
    );
    (corpus, query)
}

/// Hybrid fusion must surface what either method alone can find: the part
/// number file (lexical-only, via `--term`) and the paraphrase file
/// (semantic-only) both appear, one result per file, and `signals.lex_rank`
/// and `signals.vec_rank` agree with separate `--lex`/`--sem` runs. Catches a
/// fusion that drops lexical-only hits, emits duplicates, or reports ranks
/// that do not match the methods it fused.
#[test]
#[serial]
#[ignore = "e2e: needs the default model; run with --include-ignored"]
fn hybrid_fuses_lexical_only_and_semantic_only_hits() {
    let (corpus, query) = fusion_corpus();
    corpus.index();

    let hybrid = corpus.search(&["--hybrid", query, "--term", "KX4471"]);
    let uniq = uniq_hits(&hybrid);
    let stems: Vec<String> = uniq.iter().map(|h| h.stem()).collect();

    // One result per file.
    let mut sorted = stems.clone();
    sorted.sort_unstable();
    sorted.dedup();
    assert_eq!(
        stems.len(),
        sorted.len(),
        "duplicate files in hybrid results"
    );

    // The lexical-only file: found by the term, not by the semantic method.
    let inventory = uniq
        .iter()
        .find(|h| h.stem() == "inventory")
        .expect("hybrid with --term KX4471 must return the part-number file");
    assert_eq!(inventory.lex_rank, Some(1), "stems: {stems:?}");
    assert_eq!(
        inventory.vec_rank, None,
        "the semantic method must not rank the part-number file"
    );

    // The semantic-only file: a paraphrase no query word matches.
    let workshop = uniq
        .iter()
        .find(|h| h.stem() == "workshop")
        .expect("hybrid must return the paraphrase file only the semantic method finds");
    assert_eq!(
        workshop.lex_rank, None,
        "the term must not match the paraphrase file"
    );
    assert!(workshop.vec_rank.is_some(), "stems: {stems:?}");

    // The both-methods file: the semantic method corroborates it, while the
    // lexical one, driven by the term, cannot have found it.
    let guide = uniq
        .iter()
        .find(|h| h.stem() == "guide")
        .expect("hybrid must return the file both methods can find");
    assert!(guide.vec_rank.is_some(), "stems: {stems:?}");

    // The ranks must agree with the methods run separately, on their own.
    let lex_only = corpus.search(&["--lex", query, "--term", "KX4471"]);
    let sem_only = corpus.search(&["--sem", query]);
    assert_eq!(
        rank_of(&lex_only, "inventory"),
        inventory.lex_rank.map(|r| r as usize),
        "hybrid lex_rank of the part-number file disagrees with the --lex run"
    );
    for hit in &uniq {
        if let Some(vr) = hit.vec_rank {
            assert_eq!(
                rank_of(&sem_only, &hit.stem()),
                Some(vr as usize),
                "hybrid vec_rank of {} disagrees with the --sem run",
                hit.stem()
            );
        }
    }
}

/// `--term` replaces the query for the lexical method, in `--hybrid` too: the
/// semantic-only and lexical-only hits of the fusion test must each be
/// genuinely invisible to the other method.
#[test]
#[serial]
#[ignore = "e2e: needs the default model; run with --include-ignored"]
fn hybrid_without_term_misses_the_part_number_file() {
    let (corpus, query) = fusion_corpus();
    corpus.index();

    // Without the term the lexical method searches the query words, so the
    // part number is the only way inventory.md could be found.
    let plain = corpus.search(&["--lex", query]);
    assert_eq!(
        rank_of(&plain, "inventory"),
        None,
        "the query words must not match the part-number file"
    );

    // And the query is semantically unlike a warehouse inventory list, so the
    // semantic method does not return it either.
    let sem = corpus.search(&["--sem", query]);
    assert_eq!(
        rank_of(&sem, "inventory"),
        None,
        "the semantic method must not rank the part-number file"
    );
}

// ---------------------------------------------------------------------------
// --term
// ---------------------------------------------------------------------------

/// A term corpus: each file carries one distinctive token, so a `--term`
/// result identifies exactly which text the lexical method searched.
fn term_corpus() -> Corpus {
    let corpus = Corpus::new();
    corpus.write(
        "harbour.md",
        "# Harbour\n\nThe dredger returns at dawn and the harbour master logs every arrival in \
the ledger by the quay.\n",
    );
    corpus.write(
        "quokka.md",
        "# Quokka\n\nA quokka naps in the shade after dawn and browses leaves at dusk.\n",
    );
    corpus.write(
        "panel.md",
        "# Electrics\n\nThe circuit breaker panel sits in the basement; switch the main \
breaker off before you open the cover.\n",
    );
    corpus.write(
        "scattered.md",
        "# Notes\n\nThe artist drew a wide circuit of copper wire, and a record breaker of a \
heatwave dried the lake that August.\n",
    );
    corpus
}

/// `--term` replaces the query: files matching the query words but not the
/// term must not come back. Catches the term being OR'd onto the query
/// instead of replacing it.
#[test]
#[serial]
#[ignore = "e2e: needs the default model; run with --include-ignored"]
fn term_replaces_the_query_for_lexical_search() {
    let corpus = term_corpus();
    corpus.index();

    let hits = corpus.search(&["--lex", "harbour dredger", "--term", "quokka"]);
    let stems = uniq_hits(&hits);
    assert!(
        stems.iter().any(|h| h.stem() == "quokka"),
        "the term's file must be returned: {:?}",
        all_stems(&stems)
    );
    assert!(
        stems.iter().all(|h| h.stem() != "harbour"),
        "a file matching only the replaced query must not be returned"
    );
}

/// Several `--term` values are all searched: each term's file comes back.
#[test]
#[serial]
#[ignore = "e2e: needs the default model; run with --include-ignored"]
fn several_terms_all_count() {
    let corpus = term_corpus();
    corpus.index();

    let hits = corpus.search(&["--lex", "irrelevant", "--term", "quokka", "--term", "panel"]);
    let stems = all_stems(&uniq_hits(&hits));
    assert!(stems.contains(&"quokka".to_string()), "stems: {stems:?}");
    assert!(stems.contains(&"panel".to_string()), "stems: {stems:?}");
}

/// A multi-word term is matched as a phrase: the file with the exact phrase
/// ranks first, ahead of a file holding the same words apart. Catches the
/// phrase being tokenised into loose words only.
#[test]
#[serial]
#[ignore = "e2e: needs the default model; run with --include-ignored"]
fn multi_word_term_matches_as_a_phrase() {
    let corpus = term_corpus();
    corpus.index();

    let hits = corpus.search(&["--lex", "irrelevant", "--term", "circuit breaker"]);
    assert_eq!(
        rank_of(&hits, "panel"),
        Some(1),
        "the exact phrase must outrank the same words apart: {:?}",
        all_stems(&uniq_hits(&hits))
    );
    assert_eq!(
        rank_of(&hits, "scattered"),
        Some(2),
        "the words of a multi-word term must also count on their own, behind the phrase: {:?}",
        all_stems(&uniq_hits(&hits))
    );
}

// ---------------------------------------------------------------------------
// --threshold, --topk, --limit
// ---------------------------------------------------------------------------

/// A higher `--threshold` must return a subset of a lower one, and every
/// returned score must clear it; `--topk` and `--limit` must cap the result
/// count and keep the top of the ranking. Catches a threshold that filters
/// after truncation or not at all, and a limit applied before ranking.
#[test]
#[serial]
#[ignore = "e2e: needs the default model; run with --include-ignored"]
fn threshold_filters_and_topk_limits() {
    let corpus = Corpus::new();
    corpus.write(
        "volcano.md",
        "# Volcanoes\n\nMagma rises from the mantle, dissolved gases expand as pressure drops, \
and the vent fractures: that is why a volcano erupts, sending lava and ash over the valley.\n",
    );
    corpus.write(
        "earthquake.md",
        "# Earthquakes\n\nWhere plates grind against each other the rock stores strain until \
it slips and the ground shakes.\n",
    );
    corpus.write(
        "geyser.md",
        "# Geysers\n\nHeated groundwater flashes to steam underground and erupts through a \
narrow vent on a regular schedule.\n",
    );
    corpus.write(
        "chess.md",
        "# Chess\n\nThe Sicilian defence answers the king's pawn with a pawn of your own, and \
the endgame rewards passed pawns and an active king.\n",
    );
    corpus.write(
        "sourdough.md",
        "# Sourdough\n\nFeed the starter with flour and water every morning, and let the dough \
rise overnight before baking the loaf.\n",
    );
    corpus.index();

    let low = corpus.search(&[
        "--sem",
        "why do volcanoes erupt",
        "--threshold",
        "0.0",
        "--topk",
        "20",
    ]);
    assert!(!low.is_empty(), "a zero threshold must return results");

    // Every file must carry a score, whatever the threshold.
    for hit in &low {
        assert!(hit.score.is_some(), "no score on {}", hit.stem());
    }

    let high = corpus.search(&[
        "--sem",
        "why do volcanoes erupt",
        "--threshold",
        "0.55",
        "--topk",
        "20",
    ]);
    let high_stems = all_stems(&uniq_hits(&high));
    let low_stems = all_stems(&uniq_hits(&low));
    let low_set: Vec<&String> = low_stems.iter().collect();
    for stem in &high_stems {
        assert!(
            low_set.contains(&stem),
            "{stem} passed the high threshold but missed the low one"
        );
    }
    for hit in &high {
        let score = hit.score.expect("scored hit");
        assert!(
            score >= 0.55 - 1e-6,
            "{} returned at score {score} below the threshold",
            hit.stem()
        );
    }
    assert_eq!(
        rank_of(&high, "volcano"),
        Some(1),
        "the on-topic file must clear any threshold that returns it"
    );

    let top3 = corpus.search(&[
        "--sem",
        "why do volcanoes erupt",
        "--threshold",
        "0.0",
        "--topk",
        "3",
    ]);
    assert!(top3.len() <= 3, "--topk 3 returned {} lines", top3.len());
    let top3_stems = all_stems(&uniq_hits(&top3));
    assert!(
        top3_stems.len() <= low_stems.len(),
        "--topk returned files the untruncated search did not"
    );
    assert_eq!(
        top3_stems,
        low_stems[..top3_stems.len()],
        "--topk must keep the top of the ranking, not re-rank it"
    );

    let limit2 = corpus.search(&[
        "--sem",
        "why do volcanoes erupt",
        "--threshold",
        "0.0",
        "--limit",
        "2",
    ]);
    assert!(
        limit2.len() <= 2,
        "--limit 2 returned {} lines",
        limit2.len()
    );
    let limit2_stems = all_stems(&uniq_hits(&limit2));
    assert_eq!(
        limit2_stems,
        low_stems[..limit2_stems.len()],
        "--limit must truncate the same ranking --topk does"
    );
}

// ---------------------------------------------------------------------------
// Determinism, daemon vs in-process, concurrency
// ---------------------------------------------------------------------------

/// The same search twice must return the same files in the same order with
/// the same scores. Caches, thread races or an unstable tie-break in the
/// vector store would fail here.
#[test]
#[serial]
#[ignore = "e2e: needs the default model; run with --include-ignored"]
fn the_same_search_twice_is_identical() {
    let (corpus, query) = fusion_corpus();
    corpus.index();

    let first = corpus.search(&["--hybrid", query, "--term", "KX4471"]);
    let second = corpus.search(&["--hybrid", query, "--term", "KX4471"]);
    let a = uniq_hits(&first);
    let b = uniq_hits(&second);
    assert_eq!(
        a.iter().map(|h| h.stem()).collect::<Vec<_>>(),
        b.iter().map(|h| h.stem()).collect::<Vec<_>>(),
        "the order changed between identical searches"
    );
    for (x, y) in a.iter().zip(&b) {
        assert_eq!(x.score, y.score, "the score of {} changed", x.stem());
    }
}

/// There is no daemon any more: every search runs the same in-process path,
/// so two runs of the same query must agree exactly — same order, same scores
/// — and stdout must be clean JSONL with no log lines mixed in (a llama.cpp
/// WARN used to land there in the old in-process path). `parse_jsonl` fails
/// the test on any line that is not one JSON object.
#[test]
#[serial]
#[ignore = "e2e: needs the default model; run with --include-ignored"]
fn two_runs_of_the_single_path_agree_and_stdout_is_clean_jsonl() {
    let (corpus, query) = fusion_corpus();
    corpus.index();

    let first = corpus.search(&["--sem", query]);
    let second = corpus.search(&["--sem", query]);
    assert!(!first.is_empty(), "the corpus must answer");

    let a = uniq_hits(&first);
    let b = uniq_hits(&second);
    assert_eq!(
        a.iter().map(|h| h.stem()).collect::<Vec<_>>(),
        b.iter().map(|h| h.stem()).collect::<Vec<_>>(),
        "two identical runs returned different orders"
    );
    for (x, y) in a.iter().zip(&b) {
        assert_eq!(
            x.score, y.score,
            "two identical runs disagree on the score of {}",
            x.stem()
        );
    }
}

/// Six searches started at once, as separate processes, must all succeed and
/// agree. Catches concurrent readers racing on the store or the embedder.
#[test]
#[serial]
#[ignore = "e2e: needs the default model; run with --include-ignored"]
fn six_concurrent_searches_all_succeed_and_agree() {
    let (corpus, query) = fusion_corpus();
    corpus.index();

    let mut children = Vec::new();
    for _ in 0..6 {
        let mut cmd = corpus.ckq();
        cmd.args(["--sem", query, "--jsonl", "--no-snippet", "-q", "."])
            .stdout(Stdio::piped())
            .stderr(Stdio::piped());
        children.push(cmd.spawn().expect("spawn a concurrent search"));
    }

    let mut runs: Vec<Vec<Hit>> = Vec::new();
    for child in children {
        let out = child.wait_with_output().expect("collect the search");
        assert!(
            out.status.success(),
            "a concurrent search failed:\n--- stdout\n{}\n--- stderr\n{}",
            String::from_utf8_lossy(&out.stdout),
            String::from_utf8_lossy(&out.stderr),
        );
        runs.push(common::parse_jsonl(&out.stdout));
    }

    let first = uniq_hits(&runs[0]);
    let first_stems = all_stems(&first);
    assert!(
        !first_stems.is_empty(),
        "the concurrent searches returned nothing"
    );
    for run in &runs[1..] {
        let other = uniq_hits(run);
        assert_eq!(
            all_stems(&other),
            first_stems,
            "a concurrent search disagreed on the order"
        );
        for (x, y) in first.iter().zip(&other) {
            assert_eq!(
                x.score,
                y.score,
                "a concurrent search disagreed on the score of {}",
                x.stem()
            );
        }
    }
}

// ---------------------------------------------------------------------------
// Cross-lingual
// ---------------------------------------------------------------------------

/// An English query must find the Spanish file on its subject first, among
/// Spanish files on neighbouring subjects. Catches a model swap to an
/// English-only embedder or a query/document prompt mix-up.
#[test]
#[serial]
#[ignore = "e2e: needs the default model; run with --include-ignored"]
fn english_query_finds_the_spanish_file_first() {
    let corpus = Corpus::new();
    corpus.write(
        "solar_es.md",
        "# Energía solar\n\nLos paneles solares del tejado convierten la luz del sol en \
electricidad para la casa; el inversor transforma la corriente y la batería guarda el sobrante \
para la noche.\n",
    );
    corpus.write(
        "carbon_es.md",
        "# Minería de carbón\n\nLa mina de carbón cierra el pozo norte y los mineros temen por \
su empleo; el carbón se transporta en tren hasta la central térmica.\n",
    );
    corpus.write(
        "gasolina_es.md",
        "# Precio de la gasolina\n\nEl precio de la gasolina sube otra vez y los conductores \
llenan el depósito antes del fin de semana; las gasolineras de la autopista cobran más caro.\n",
    );
    corpus.write(
        "presas_es.md",
        "# Presas hidroeléctricas\n\nLa presa retiene el río y la central hidroeléctrica \
genera luz cuando las turbinas giran; en verano el embalse baja de nivel.\n",
    );
    corpus.index();

    let hits = corpus.search(&["--sem", "solar panels making electricity for the house"]);
    assert_eq!(
        rank_of(&hits, "solar_es"),
        Some(1),
        "the Spanish solar file must rank first: {:?}",
        all_stems(&uniq_hits(&hits))
    );
}

/// The same in Mandarin, with Mandarin distractors on neighbouring subjects.
#[test]
#[serial]
#[ignore = "e2e: needs the default model; run with --include-ignored"]
fn english_query_finds_the_mandarin_file_first() {
    let corpus = Corpus::new();
    corpus.write(
        "solar_zh.md",
        "# 家用太阳能\n\n屋顶上的太阳能电池板把阳光变成家里的电，逆变器把电流转换好，多余的电存进电池，留到晚上用。\n",
    );
    corpus.write(
        "coal_zh.md",
        "# 煤矿\n\n北边的煤矿要关闭竖井，矿工担心自己的工作；煤用火车运到火电站去发电。\n",
    );
    corpus.write(
        "petrol_zh.md",
        "# 油价\n\n汽油又涨价了，司机赶在周末前把油箱加满，高速公路上的加油站卖得更贵。\n",
    );
    corpus.write(
        "dam_zh.md",
        "# 水电站\n\n大坝拦住河水，水轮机转动时水电站就发电；到了夏天水库的水位会下降。\n",
    );
    corpus.index();

    let hits = corpus.search(&["--sem", "solar panels making electricity for the house"]);
    assert_eq!(
        rank_of(&hits, "solar_zh"),
        Some(1),
        "the Mandarin solar file must rank first: {:?}",
        all_stems(&uniq_hits(&hits))
    );
}

// ---------------------------------------------------------------------------
// Re-index after an edit
// ---------------------------------------------------------------------------

/// After a file is rewritten and `--index` runs again, the new content is
/// what ranks, for `--sem` and for `--lex`; the old content stops ranking.
/// Caches a stale index: chunks, embeddings or the tantivy corpus that are
/// not invalidated by the file's change would fail here.
#[test]
#[serial]
#[ignore = "e2e: needs the default model; run with --include-ignored"]
fn reindex_after_an_edit_ranks_the_new_content() {
    let corpus = Corpus::new();
    corpus.write(
        "telescope.md",
        "# Telescopes\n\nA refractor telescope gathers starlight through a series of glass \
lenses; align the finder at dusk before the planets rise.\n",
    );
    corpus.write(
        "garden.md",
        "# Garden\n\nSow the peas in early spring and mulch the beds before the summer heat \
arrives.\n",
    );
    corpus.index();

    corpus.write(
        "telescope.md",
        "# Coffee\n\nPulling a good espresso shot: dose the basket, tamp it level, and watch \
the crema turn hazelnut brown as the shot lands in the cup.\n",
    );
    corpus.index();

    let sem_new = corpus.search(&[
        "--sem",
        "how to pull an espresso shot",
        "--threshold",
        "0.0",
    ]);
    assert_eq!(
        rank_of(&sem_new, "telescope"),
        Some(1),
        "the rewritten file must rank for its new subject: {:?}",
        all_stems(&uniq_hits(&sem_new))
    );

    let sem_old = corpus.search(&[
        "--sem",
        "observing planets with a telescope",
        "--threshold",
        "0.0",
    ]);
    assert_ne!(
        rank_of(&sem_old, "telescope"),
        Some(1),
        "the deleted content must no longer rank first"
    );

    let lex_new = corpus.search(&["--lex", "espresso"]);
    assert_eq!(
        rank_of(&lex_new, "telescope"),
        Some(1),
        "the lexical index must hold the new text"
    );
    // The CLI reports no matches like grep does: exit 1 and a message on
    // stderr. The deleted word must be gone from the lexical index.
    let mut cmd = corpus.ckq();
    cmd.args(["--lex", "refractor", "--jsonl", "-q", "."]);
    let stderr = common::run_err(&mut cmd);
    assert!(
        stderr.contains("No matches found"),
        "the lexical index must not hold the deleted text; stderr: {stderr}"
    );
}

// ---------------------------------------------------------------------------

/// Distinct file stems of a hit list, in order, for failure messages and
/// order comparisons.
fn all_stems(hits: &[Hit]) -> Vec<String> {
    hits.iter().map(Hit::stem).collect()
}

/// Same, after `uniq_hits`, for messages only; the assertion uses `uniq_hits`
/// directly where order matters.
fn stems(hits: &[Hit]) -> Vec<String> {
    all_stems(&uniq_hits(hits))
}

// ---------------------------------------------------------------------------
// The working directory must not matter
// ---------------------------------------------------------------------------

/// A caller that passes an absolute path from some other directory must get
/// what a caller inside the corpus gets. The old engine returned "No matches
/// found" from any working directory outside the searched one (four path
/// forms that disagreed); v2 keys every document on its root-relative path,
/// so this must hold everywhere.
#[test]
#[serial]
#[ignore = "e2e: needs the default model; run with --include-ignored"]
fn semantic_search_is_the_same_from_another_directory() {
    let corpus = Corpus::new();
    corpus.write(
        "volcano.md",
        "# Volcanoes\n\nMagma rises from the mantle and dissolved gases expand until the vent \
fractures and the volcano erupts.\n",
    );
    corpus.write(
        "sourdough.md",
        "# Sourdough\n\nFeed the starter every morning and let the dough rise overnight.\n",
    );
    corpus.write(
        "chess.md",
        "# Chess\n\nThe Sicilian defence answers the king's pawn with a pawn of your own.\n",
    );
    corpus.index();
    let root = corpus.path().to_str().unwrap().to_string();
    let args = [
        "--sem",
        "what makes a volcano erupt",
        "--threshold",
        "0.0",
        "--jsonl",
        "-q",
    ];

    let inside = common::parse_jsonl(&common::run_ok(corpus.ckq().args(args).arg(&root)).stdout);
    let elsewhere = tempfile::TempDir::new().unwrap();
    let outside = common::parse_jsonl(
        &common::run_ok(
            common::ckq()
                .current_dir(elsewhere.path())
                .args(args)
                .arg(&root),
        )
        .stdout,
    );

    assert_eq!(
        rank_of(&inside, "volcano"),
        Some(1),
        "inside: {:?}",
        stems(&inside)
    );
    assert_eq!(
        stems(&uniq_hits(&outside)),
        stems(&uniq_hits(&inside)),
        "the same search from another working directory returned different files"
    );
}
