# Cross-lingual retrieval benchmark

40 documents, eight each in English, Mandarin, Hindi, Spanish and Arabic. Ten subjects,
four neighbouring aspects each, one aspect per language: storing coffee beans, roasting
curves, grinder burrs and espresso timing are four separate documents in four languages.

40 queries, **each asked in a language other than the one its answer is written in**. The
wrong answers inside a subject are about that same subject, so matching the topic is not
enough; the aspect has to be right as well. Nothing here is drawn from anyone's notes.

```bash
python3 run.py                          # the default model
python3 run.py gemma-q4 granite-gguf    # compare
```

It reports rank-1 accuracy, MRR@5 and the score range of the correct hits, which is what
decides whether a given `--threshold` keeps them.

`corpus.json` holds the documents and queries. `run.py` writes them to a temporary
directory, indexes it with each model, and runs every query.

## Results, 26 Sep 2026

| model | params | rank 1 | MRR@5 | score range |
|---|---:|---:|---:|---:|
| `gemma-q4` | 300M | 35/40 | 0.927 | 0.42-0.74 |
| `bge-m3-gguf` | 568M | 34/40 | 0.904 | 0.38-0.71 |
| `qwen3-gguf` | 600M | 31/40 | 0.851 | 0.36-0.71 |
| `granite-gguf` | 278M | 28/40 | 0.781 | 0.61-0.88 |

Two queries are missed by every model. One asks in Spanish why a pot explodes in the kiln
when air is trapped inside, and the answer is a Hindi document about wedging clay that
never mentions a kiln. The other asks in Hindi how to decide sowing depth, and the answer
is an Arabic document that gives the rule as twice the seed's diameter. Both need a step of
inference rather than a closer match, which is the honest limit of a bi-encoder.

## Results, 17 Oct 2026 — Apache/MIT candidates

Four candidates for replacing gemma-q4 (Gemma Terms of Use) with a ship-safe
licence, on one build: llama-cpp-2 at git `e40decdb` (llama.cpp v0.6.0, the
first vendored version that knows the `modern-bert` architecture; 0.1.157 from
crates.io does not). `gemma-q4` re-run as the control on the same build.
Thresholds for the new models sit just below their lowest correct-hit score.

| model | params | rank 1 | MRR@5 | score range |
|---|---:|---:|---:|---:|
| `gemma-q4` (control, re-run) | 300M | 35/40 | 0.927 | 0.36-0.72 |
| `granite-r2-311m` | 311M | 33/40 | 0.895 | 0.81-0.94 |
| `harrier-gguf` | 600M | 32/40 | 0.883 | 0.42-0.63 |
| `granite-r2-97m` | 97M | 28/40 | 0.788 | 0.79-0.92 |

Through `--hybrid` the same three candidates hold their rank 1 and MRR@5
exactly (33/0.895, 32/0.883, 28/0.788), as does the control (35/0.927).

`google/embeddinggemma-2` (Apache-2.0) was blocked: it needs the
`gemma-embedding2` architecture, added in llama.cpp 4fbc76de on 6 Oct 2026, and
no commit of llama-cpp-rs vendors a llama.cpp that new — the newest submodule
is v0.6.0 (d812350, 5 Oct 2026), one day short. Every other candidate kept its
licence verified on its Hugging Face page: harrier MIT, both granite Apache-2.0.

