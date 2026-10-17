# ckq

ckq is a small command-line tool for searching a folder of text in any language.
It builds one local index of a directory tree and then answers three kinds of
query over it: plain regex (grep-style, no index needed), lexical ranking
(BM25 over file text and path), and semantic ranking (embedding similarity) —
plus a hybrid mode that fuses the two ranked lists. Documents in one language
can be found with a query in another.

## Install

```bash
cargo build --release
```

The binary is `target/release/ckq`. The default embedding model (about 280 MB)
downloads on first use to `~/.cache/ckq/models`; building or running regex and
lexical search needs no model at all.

## GPU

The embedder runs on the GPU through llama.cpp, and the backend is chosen at
build time by cargo features:

- **macOS**: Metal, always — no feature to turn on or off.
- **Linux and Windows**: Vulkan, on by default. The binary then needs a
  Vulkan loader at runtime: `libvulkan1` on Linux (any GPU driver brings it)
  or `vulkan-1.dll` on Windows. A machine with a loader but no usable device
  falls back to the CPU on its own.
- `--no-default-features` builds a CPU-only binary for machines with no
  Vulkan loader at all.
- `--features cuda` builds with CUDA instead, which needs the CUDA toolkit
  at build time and an NVIDIA driver at runtime.

The default feature reaches `llama-cpp-2` through the tiny `gpu/` path crate
(package `ckq-gpu`), because cargo cannot make a default feature
platform-specific and will not accept the same crate twice under two names.
On macOS the crate is not even in the dependency graph.

## One file

`scripts/build-onefile.sh <alias>` builds a ckq with that model's GGUF inside
the binary itself, so one file works offline on a fresh machine:

```bash
scripts/build-onefile.sh gemma-q4
```

The model must already be in `~/.cache/ckq/models` (run `ckq --index` once to
download it). The embedded model becomes that build's default, and the first
use writes it out once to `~/.cache/ckq/models/embedded/`; `--status` and
`--version` say which model a binary carries. A binary that embeds `gemma-q4`
carries Google's Gemma Terms of Use with it and must ship those terms; a
model under Apache-2.0 or MIT (such as granite, bge-m3 or Qwen3-Embedding)
does not have that condition. Which model the releases embed is decided
separately.

## Usage

```bash
ckq --index                     # build or update the index of the current directory
ckq --clean                     # remove the index
ckq --status                    # what is indexed, and with which model
```

Search — one mode flag picks the method:

```bash
ckq pattern                     # regex mode (the default), like grep
ckq --lex how to store coffee   # BM25 over file text and path
ckq --sem 咖啡豆怎么保存        # embedding similarity
ckq --hybrid --term 'coffee beans' how to keep beans fresh   # both, fused
```

- `--hybrid` fuses the lexical and semantic rankings into one list; `--term`
  gives the lexical side an exact phrase (repeatable) instead of the query.
- `--topk N` caps the number of results (default 10 in ranked modes).
- `--threshold 0.5` sets the minimum semantic score, 0.0–1.0 (default: the
  model's own).
- `--model ALIAS` picks the embedding model for `--index` (default: the
  index's own, else `gemma-q4`).
- `--jsonl` (or `--json`) prints one JSON object per result instead of text.

Regex mode takes the grep flags:

```bash
ckq -n -i -w -F -l -C 2 'literal string'
```

`-n` line numbers, `-i` ignore case, `-w` whole words, `-F` fixed strings,
`-l` file names only, `-C N` context lines. Exit status is 0 when something
was found, 1 for no match or an error.

## How it works

The index is one SQLite file, `.ckq/index.db` in the indexed tree: a contentless
FTS5 table gives BM25 over the file text and path, chunks are embedded to f16
vectors in the same database, and a hash of model, prompts, pooling and chunker
version guards comparability. Embeddings run through llama.cpp with the GPU
when one is present. Five model aliases exist — `gemma-q4` (the default),
`gemma-gguf`, `granite-gguf`, `bge-m3-gguf` and `qwen3-gguf` — all GGUF,
downloaded from Hugging Face on first use.

Searches embed their query through a small resident daemon: the first `--sem`
or `--hybrid` starts it and every later one connects, so a day of searching
loads the model once instead of once per call (loading it costs about 0.4 s).
One daemon per model, keyed to the exact binary, so a rebuild never talks to
a stale one; it exits after 15 minutes without a query. The daemon does
nothing but embed queries — indexing always loads the model in the process
that runs it — and `CKQ_IN_PROCESS=1` embeds in the searching process instead.
Sockets live in `~/.cache/ckq/sockets` (unix only).

Hybrid search fuses per file by CombMAX over weighted reciprocal ranks: each
method contributes `weight / (3 + rank)`, a file keeps the better of its two
scores, and agreement between the methods breaks near-ties. Summed RRF was
measured and lost: it let a file both methods ranked second overtake the file
one method ranked first. Search never writes the index — only `--index`,
`--add` and `--clean` do.

## Benchmark

`bench/multilingual`: 40 documents in English, Mandarin, Hindi, Spanish and
Arabic; 40 queries, each asked in a language other than its answer's.

| model | params | rank 1 | MRR@5 | score range |
|---|---:|---:|---:|---:|
| `gemma-q4` | 300M | 35/40 | 0.927 | 0.42-0.74 |
| `bge-m3-gguf` | 568M | 34/40 | 0.904 | 0.38-0.71 |
| `qwen3-gguf` | 600M | 31/40 | 0.851 | 0.36-0.71 |
| `granite-gguf` | 278M | 28/40 | 0.781 | 0.61-0.88 |

## Credits

A fork of [BeaconBay/ck](https://github.com/BeaconBay/ck) by Mike Renwick,
rewritten as a single crate. MIT OR Apache-2.0.
