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
