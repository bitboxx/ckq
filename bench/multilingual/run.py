#!/usr/bin/env python3
"""Cross-lingual retrieval benchmark for ckq.

Every query is asked in a language other than the one its answer is written in,
against a corpus where several documents cover neighbouring aspects of the same
subject. A model that only matches words cannot score here, and a model that
matches the subject but not the aspect lands near the answer without hitting it.

    python3 run.py                      # the default model
    python3 run.py granite-gguf gemma-q4   # compare models

Reports rank-1 accuracy, MRR@5 and the score range of the correct hits, which
is what decides whether ck's default --threshold 0.6 keeps them.
"""

import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
CORPUS = json.loads((HERE / "corpus.json").read_text(encoding="utf-8"))
TOPK = 5


def materialise(root: Path) -> None:
    for doc in CORPUS["documents"]:
        (root / f"{doc['id']}.md").write_text(
            f"# {doc['title']}\n\n{doc['text']}\n", encoding="utf-8"
        )


def search(root: Path, query: str) -> list[str]:
    """Document ids in rank order, one entry per document."""
    out = subprocess.run(
        ["ckq", "--jsonl", "--no-snippet", "--sem", "-q", query,
         "--topk", str(TOPK * 3), "--threshold", "0.0", str(root)],
        capture_output=True, text=True,
    ).stdout
    seen, ranked = set(), []
    for line in out.splitlines():
        if not line.startswith("{"):
            continue
        hit = json.loads(line)
        doc_id = Path(hit.get("path") or hit.get("file", "")).stem
        if doc_id and doc_id not in seen:
            seen.add(doc_id)
            ranked.append((doc_id, hit.get("score")))
    return ranked


def evaluate(model: str) -> dict:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        materialise(root)
        subprocess.run(["ckq", "--index", "--model", model, "-q", str(root)], check=True)

        first, reciprocal, scores, misses = 0, 0.0, [], []
        for q in CORPUS["queries"]:
            ranked = search(root, q["q"])
            ids = [doc_id for doc_id, _ in ranked][:TOPK]
            if q["answer"] in ids:
                rank = ids.index(q["answer"]) + 1
                reciprocal += 1 / rank
                if rank == 1:
                    first += 1
                score = dict(ranked)[q["answer"]]
                if score is not None:
                    scores.append(score)
            else:
                misses.append(q["q"])

        n = len(CORPUS["queries"])
        return {
            "model": model,
            "rank1": first,
            "n": n,
            "mrr": reciprocal / n,
            "lo": min(scores) if scores else 0.0,
            "hi": max(scores) if scores else 0.0,
            "above_default_threshold": sum(1 for s in scores if s >= 0.6),
            "misses": misses,
        }


def main() -> None:
    models = sys.argv[1:] or ["granite-gguf"]
    results = [evaluate(m) for m in models]
    print()
    print(f"{'model':<16} {'rank 1':>10} {'MRR@5':>7} {'score range':>14} {'>= 0.60':>9}")
    for r in results:
        print(f"{r['model']:<16} {r['rank1']:>4}/{r['n']:<5} "
              f"{r['mrr']:>7.3f} {r['lo']:>6.2f}-{r['hi']:<7.2f} {r['above_default_threshold']:>5}/{r['n']}")
    for r in results:
        if r["misses"]:
            print(f"\n{r['model']} missed outside the top {TOPK}:")
            for q in r["misses"]:
                print(f"  - {q}")


if __name__ == "__main__":
    main()
