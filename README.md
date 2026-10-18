# ckq

ckq is a command-line search engine for a folder of text notes. It builds one
local index of a directory tree and answers three kinds of query over it:
plain regex (grep-style, no index needed), lexical ranking (BM25 over file
text and path), and semantic ranking (embedding similarity), plus a hybrid
mode that fuses the two ranked lists. A document in one language can be found
with a query in another.

## Install

Download an archive from the [releases page](https://github.com/bitboxx/ckq/releases)
and put the `ckq` binary on your `PATH`:

| Platform | Archive |
|---|---|
| Linux x86_64 | `ckq-2.0.0-x86_64-unknown-linux-gnu.tar.gz` |
| Linux aarch64 | `ckq-2.0.0-aarch64-unknown-linux-gnu.tar.gz` |
| macOS x86_64 | `ckq-2.0.0-x86_64-apple-darwin.tar.gz` |
| macOS aarch64 | `ckq-2.0.0-aarch64-apple-darwin.tar.gz` |
| Windows x86_64 | `ckq-2.0.0-x86_64-pc-windows-msvc.zip` |

Check the download against the `.sha256` file beside it (or the combined
`SHA256SUMS` file):

```bash
shasum -a 256 -c ckq-2.0.0-aarch64-apple-darwin.tar.gz.sha256
```

On Windows, `CertUtil -hashfile ckq-2.0.0-x86_64-pc-windows-msvc.zip SHA256`
prints the same hash.

Runtime needs: the Linux and Windows binaries are built with Vulkan and need a
Vulkan loader to start, `libvulkan.so.1` on Linux and `vulkan-1.dll` on
Windows. Any GPU driver installs one. A machine with a loader but no usable
GPU falls back to the CPU on its own. A machine with no Vulkan loader at all
cannot run these binaries; build from source with `--no-default-features` for
a CPU-only binary. macOS needs nothing beyond the system, but a binary
downloaded with a browser is quarantined: run `xattr -d com.apple.quarantine ckq`
once before the first start.

With a Rust toolchain:

```bash
cargo install --git https://github.com/bitboxx/ckq
```

or build from source. The cargo features:

- `vulkan`: GPU through Vulkan, the default on Linux and Windows. Building it
  needs the Vulkan headers and loader, `glslc` and SPIRV-Headers
  (`libvulkan-dev glslc spirv-headers` on Ubuntu; the LunarG SDK with
  `VULKAN_SDK` set on Windows). macOS always builds on Metal and ignores the
  feature.
- `cuda`: GPU through CUDA instead. `--no-default-features --features cuda`
  builds CUDA alone, which needs the CUDA toolkit at build time and an NVIDIA
  driver at runtime. `--features cuda` by itself keeps Vulkan as well.
- `--no-default-features`: CPU only, no Vulkan build dependencies.
- `embedded-model`: build one GGUF into the binary so a single file works
  offline on a fresh machine. `scripts/build-onefile.sh <alias>` wraps it; the
  model must already be in the cache.

```bash
git clone https://github.com/bitboxx/ckq
cd ckq
cargo build --release --locked
```

The binary is `target/release/ckq`.

## First use

Run `ckq --index` in the directory you want to search:

```bash
cd ~/notes
ckq --index
```

The first run downloads the default embedding model (gemma-q4, about 280 MB)
into `~/.cache/ckq/models` and then embeds every text file. Later runs only
read files whose size or modification time changed. Regex search needs no
model and no index.

The first `--index` also writes a `.ckqignore` file with default patterns
(images, video, archives, databases) when the directory has none. It uses
gitignore syntax, and `.gitignore` and `.ignore` apply too. A leftover `.ck`
directory from the old ck engine is never read or written; `--clean` removes
`.ckq` only.

## Search modes

One mode flag picks the method:

```bash
ckq 'coarse grind'                  # regex mode (the default), like grep
ckq --lex how to store coffee       # BM25 over file text and path
ckq --sem 咖啡豆怎么保存            # embedding similarity
ckq --hybrid --term 'coffee beans' 'how to keep beans fresh'   # both, fused
```

`--hybrid` fuses the lexical and semantic rankings into one list, one result
per file. `--term` (repeatable) gives the lexical method of `--lex` and
`--hybrid` an exact phrase in place of the query; in `--hybrid` the query
still drives the semantic method.

Flags for the ranked modes:

- `--topk N` (alias `--limit`): number of results, 10 by default.
- `--threshold F`: minimum semantic score, -1.0 to 1.0. Default: the model's
  own value, since scores are not comparable across models.
- `--scores`: show scores in text output.
- `--jsonl`, `--json`, `--no-snippet`: output format, see below.

Regex mode takes the grep flags:

```bash
ckq -n -i -w -F -l -C 2 'grind size'
```

`-n` line numbers, `-i` ignore case, `-w` whole words, `-F` fixed strings
instead of a regex, `-l` file names only, `-C N` context lines. `--regex`
selects this mode explicitly. `--topk` limits matches here too, and `-r` is
accepted and does nothing (directories are always searched recursively).
Flags that do nothing in the chosen mode draw one warning on stderr.

Both modes take `--hidden` (include dot-files), `--no-ignore` (do not read
`.gitignore`), `--no-ckqignore` (do not read `.ckqignore`) and `-q` (no
progress or banners on stderr). A search takes the paths to search as
arguments after the query; the default is the current directory, and a
subdirectory or single file searches within the index of the tree above it.

## JSONL output

`--jsonl` prints one JSON object per result:

```json
{"path": "./recipes/coffee.md", "span": {"byte_start": 41, "byte_end": 49, "line_start": 3, "line_end": 3}, "language": "markdown", "snippet": "Keep beans in an airtight container...", "score": 0.688, "signals": {"lex_rank": 1, "vec_rank": 2, "rrf_score": 0.41}}
```

- `path`: the file as you named it on the command line, with `/` separators.
- `span`: the byte and line range of the best-matching chunk, or of the
  matching line in regex mode.
- `language`: `markdown`, `text`, `rust`, and so on.
- `snippet`: the text of the span. Absent with `--no-snippet`.
- `score`: the semantic score, or 1.0 in regex mode.
- `signals`: hybrid results only. `lex_rank` and `vec_rank` say where each
  method ranked the file (`null` when a method did not find it);
  `rrf_score` is the fused score.

`--json` prints the same results under the old engine's field names (`file`,
`lang`, `preview`, `model`), one object per line, for scripts that read the
old format.

## Keeping the index current

`--index` is the only command that builds the index (`--add` and `--clean`
also write it; a search never does). A ranked search checks the tree first
and prints one warning when files were added, changed or deleted since the
last index:

```text
ckq: warning: index is stale (3 files added, changed or gone); run `ckq --index`
```

- `ckq --index [DIR]`: bring the index in line with the files on disk.
- `ckq --add FILE`: index or re-index one file.
- `ckq --status [DIR]`: what is indexed and with which model. `--status-verbose`
  adds sizes, `--status-json` prints JSON to stdout.
- `ckq --clean [DIR]`: remove the index.
- `ckq --clean-orphans [DIR]`: accepted for compatibility; the store cannot
  hold orphans, so this just runs an index update.

Each model's vectors live in their own space, so the index records a
signature of the model, its prompts, its pooling and the chunker version.
`ckq --index --model ALIAS` switches models and rebuilds the whole index; a
search against an index whose signature does not match the binary's current
settings refuses and names `ckq --index`. The default model is `gemma-q4`,
or the index's own model when one exists.

## The query daemon

On unix, `--sem` and `--hybrid` embed their query through a small resident
daemon instead of loading the model in every search process. The first search
starts it; later searches connect to it, so a day of searching loads the model
once (a load costs about 0.4 s). One daemon per model, keyed to the exact
binary, so a rebuild never talks to a stale one. Sockets live in
`~/.cache/ckq/sockets`. The daemon embeds queries only; indexing always loads
the model in the process that runs it.

The daemon exits after 15 minutes without a query
(`CKQ_DAEMON_IDLE_SECS`). `CKQ_IN_PROCESS=1` skips it and embeds in the
searching process. A client waits 120 s for a reply
(`CKQ_DAEMON_TIMEOUT_SECS`); on a timeout or any daemon failure it falls back
to embedding in its own process, with one warning on stderr.

## Models

All models are GGUF, downloaded from Hugging Face on first use and cached in
`~/.cache/ckq/models`. Where the registry pins a blake3, the download is
verified against it.

| Alias | Dims | Context | Size | Licence |
|---|---:|---:|---:|---|
| `gemma-q4` (default) | 768 | 2048 | 278 MB | Gemma Terms of Use |
| `gemma-gguf` | 768 | 2048 | 333 MB | Gemma Terms of Use |
| `granite-gguf` | 768 | 512 | 303 MB | Apache-2.0 |
| `bge-m3-gguf` | 1024 | 8192 | 634 MB | MIT |
| `qwen3-gguf` | 1024 | 8192 | 640 MB | Apache-2.0 |

## Environment variables

| Variable | Effect |
|---|---|
| `CKQ_IN_PROCESS=1` | Embed queries in the searching process; no daemon. |
| `CKQ_DAEMON_IDLE_SECS` | Daemon idle exit time in seconds. Default 900. |
| `CKQ_DAEMON_TIMEOUT_SECS` | Client wait for a daemon reply in seconds. Default 120; past it the client falls back to its own process. |
| `CKQ_QUERY_DEVICE` | `cpu` or `gpu` for query embedding. Default `gpu`. |
| `CKQ_GGUF_FILE` | Load the GGUF from this file instead of the registry download, for trying another build of the same model. The file's path, size and mtime enter the index signature. |
| `CKQ_TIMING=1` | Print one stderr line of index timings (walk, read and chunk, embed). |
| `CKQ_EMBED_GGUF`, `CKQ_EMBED_ALIAS` | Build-time only, with the `embedded-model` feature: the GGUF to embed in the binary and its alias. |

## Exit codes

- 0: something was found, or the command succeeded.
- 1: no match (with `No matches found` on stderr), or an error.
- 2: a usage error, or, in regex mode, files that could not be read and
  nothing matched.

## Benchmark

`bench/multilingual` holds 40 documents in English, Mandarin, Hindi, Spanish
and Arabic, and 40 queries each asked in a language other than its answer's.
Run it with `python3 bench/multilingual/run.py [aliases]`.

| model | params | rank 1 | MRR@5 | score range |
|---|---:|---:|---:|---:|
| `gemma-q4` | 300M | 35/40 | 0.927 | 0.42-0.74 |
| `bge-m3-gguf` | 568M | 34/40 | 0.904 | 0.38-0.71 |
| `qwen3-gguf` | 600M | 31/40 | 0.851 | 0.36-0.71 |
| `granite-gguf` | 278M | 28/40 | 0.781 | 0.61-0.88 |

`tests/e2e_quality.rs` enforces the bar in CI: both `--sem` and `--hybrid`
must hold 34/40 rank 1 and MRR@5 0.90 with the default model. The one query
of slack covers float drift between Metal, CPU and other backends.

## Testing

```bash
cargo test          # offline tests only
cargo e2e           # plus the end-to-end suites; downloads the default model once
cargo known-bugs    # the tests that pin a known, still-open bug (none today)
```

## Licence and credits

MIT OR Apache-2.0. A fork of [BeaconBay/ck](https://github.com/BeaconBay/ck)
by Mike Renwick, rewritten as a single crate.
