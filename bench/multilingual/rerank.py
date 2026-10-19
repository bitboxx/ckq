#!/usr/bin/env python3
"""Rerank experiment on the cross-lingual bench: --hybrid with and without
--rerank, for each mode (pure, fuse) and N in {10, 30}.

    python3 bench_rerank.py /path/to/ckq

Indexes a scratch copy of bench/multilingual once, then runs every variant
over the same index. Prints rank 1, MRR@5, and — from CKQ_TIMING=1 on
stderr — the reranker's load time and the median warm scoring time per
search. CKQ_IN_PROCESS=1 keeps the query daemon out of the picture; the
reranker always loads in the searching process, so its load is paid once
per run and reported separately.
"""

import json
import os
import re
import statistics
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
if not (HERE / "corpus.json").exists():
    HERE = Path(
        "/Users/bitboxx/nara-local-synced/projects/ckq-wt/rerank/bench/multilingual"
    )
CORPUS = json.loads((HERE / "corpus.json").read_text(encoding="utf-8"))
TOPK = 5


def materialise(root: Path) -> None:
    for doc in CORPUS["documents"]:
        (root / f"{doc['id']}.md").write_text(
            f"# {doc['title']}\n\n{doc['text']}\n", encoding="utf-8"
        )


def search(ckq: str, root: Path, query: str, env: dict) -> tuple[list[str], float]:
    """Document ids in rank order (one per file) and the scoring time."""
    e = dict(os.environ)
    e.update(env)
    argv = [ckq, "--hybrid"]
    if "CKQ_RERANKER" in env:
        argv.append("--rerank")
    argv += ["--jsonl", "--no-snippet", "-q", query,
             "--topk", str(TOPK * 3), "--threshold", "0.0", str(root)]
    out = subprocess.run(argv, capture_output=True, text=True, env=e)
    if out.returncode not in (0, 1):
        raise SystemExit(f"search failed: {out.stderr}")
    ranked, seen = [], set()
    for line in out.stdout.splitlines():
        if not line.startswith("{"):
            continue
        hit = json.loads(line)
        doc_id = Path(hit["path"]).stem
        if doc_id and doc_id not in seen:
            seen.add(doc_id)
            ranked.append(doc_id)
    m = re.search(r"rerank: scored \d+ passages with \S+ in ([0-9.]+) s", out.stderr)
    return ranked, float(m.group(1)) if m else 0.0


def load_time(ckq: str, root: Path, env: dict) -> float:
    e = dict(os.environ)
    e.update(env)
    e["CKQ_TIMING"] = "1"
    q = CORPUS["queries"][0]["q"]
    out = subprocess.run(
        [ckq, "--hybrid", "--rerank", "--jsonl", "--no-snippet", "-q", q,
         "--topk", "1", "--threshold", "0.0", str(root)],
        capture_output=True, text=True, env=e,
    )
    m = re.search(r"rerank: loaded \S+ in ([0-9.]+) s", out.stderr)
    return float(m.group(1)) if m else float("nan")


def evaluate(ckq: str, root: Path, env: dict) -> dict:
    first, reciprocal, misses, times = 0, 0.0, [], []
    for q in CORPUS["queries"]:
        ranked, t = search(ckq, root, q["q"], env)
        times.append(t)
        ids = ranked[:TOPK]
        if q["answer"] in ids:
            rank = ids.index(q["answer"]) + 1
            reciprocal += 1 / rank
            first += rank == 1
        else:
            misses.append(q["q"])
    n = len(CORPUS["queries"])
    return {
        "rank1": first, "n": n, "mrr": reciprocal / n,
        "median_s": statistics.median(times),
        "load_s": None, "misses": misses,
    }


def main() -> None:
    ckq = sys.argv[1] if len(sys.argv) > 1 else "ckq"
    ckq = str(Path(ckq).resolve())
    base = {"CKQ_IN_PROCESS": "1"}
    variants = [
        ("hybrid (no rerank)", {}),
        ("bge pure N=10", {"CKQ_RERANK_TOPN": "10"}),
        ("bge pure N=30", {}),
        ("bge fuse N=10", {"CKQ_RERANK_TOPN": "10", "CKQ_RERANK_MODE": "fuse"}),
        ("bge fuse N=30", {"CKQ_RERANK_MODE": "fuse"}),
    ]
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        materialise(root)
        subprocess.run([ckq, "--index", "--model", "gemma-q4", "-q", str(root)], check=True)
        rows = []
        # The load time needs its own cold process; do it first.
        lt = load_time(ckq, root, base)
        for name, extra in variants:
            env = dict(base)
            env.update(extra)
            env["CKQ_TIMING"] = "1"
            if "no rerank" not in name:
                env["CKQ_RERANKER"] = "bge-reranker"
            r = evaluate(ckq, root, env)
            r["name"] = name
            r["load_s"] = lt
            rows.append(r)
    print()
    print(f"{'variant':<22} {'rank 1':>10} {'MRR@5':>7} {'score s':>8} {'load s':>7}")
    for r in rows:
        t = f"{r['median_s']:.3f}" if r["median_s"] else "-"
        l = f"{r['load_s']:.2f}" if r["load_s"] else "-"
        print(f"{r['name']:<22} {r['rank1']:>4}/{r['n']:<5} {r['mrr']:>7.3f} {t:>8} {l:>7}")
    for r in rows:
        if r["misses"]:
            print(f"\n{r['name']} missed outside the top {TOPK}:")
            for q in r["misses"]:
                print(f"  - {q}")


if __name__ == "__main__":
    main()
