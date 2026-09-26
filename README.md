# ckq

Semantic, lexical and regex search over a directory, for corpora that are not in English.

A fork of [BeaconBay/ck](https://github.com/BeaconBay/ck) by Mike Renwick. Upstream is an
excellent code-search tool whose embedding models are all English-only; ckq swaps them for
multilingual ones, runs them on the GPU through llama.cpp, and shares a single loaded model
across every invocation on the machine. The binary is `ckq`, so it coexists with `ck`.

Everything else, the grep-compatible CLI, BM25, tree-sitter chunking, `--full-section`, the
MCP server, is upstream's work and is unchanged.

## Install

```bash
git clone https://github.com/bitboxx/ckq && cd ckq
cargo build --release -p ck-search --features llamacpp
./target/release/ckq --index ~/notes
```

Needs a Rust toolchain and CMake; llama.cpp is built from source by `llama-cpp-sys-2`.

On macOS that is all: Metal is always on. On Linux and Windows the build is CPU-only
unless you ask for the GPU with `--features vulkan`, which needs the Vulkan SDK installed
and `VULKAN_SDK` set. llama.cpp's build script fails the whole build when the feature is
on and the SDK is missing, which is why it is not the default.
Model weights are fetched from Hugging Face on first use and cached.

```bash
ckq --index ~/notes                       # granite-gguf by default
ckq --sem "when is the boiler serviced" ~/notes
ckq --lex "exact phrase" ~/notes          # BM25
ckq "regex.*here" ~/notes                 # grep-compatible
ckq --serve                               # MCP server; pins the daemon
ckq --index --model gemma-q4 ~/notes      # a different model
```

## Status

Working and used daily, but young. It has been exercised on prose and on Rust source, on
Apple Silicon and on Linux/Vulkan. CUDA is untested. The API is upstream's; the model
registry and the daemon are new here and may still move.

## Why granite

Stock ck ships English-only models. On a nine-note multilingual fixture (three languages,
mixed within the corpus), correct note at rank 1:

| model | params | fixture |
|---|---:|---|
| stock ck `bge-small-en-v1.5` | 33M | 2/4 |
| stock ck `nomic-embed-text-v1.5` | 137M | 1/4 |
| `paraphrase-multilingual-MiniLM` | 118M | 2/4 |
| `gemma-q4` EmbeddingGemma 300M | 300M | 4/4 |
| **`granite-gguf` granite-embedding-278m** | **278M** | **4/4** |
| `bge-m3-gguf` | 568M | 4/4 |
| `qwen3-gguf` Qwen3-Embedding-0.6B | 600M | 4/4 |

Four models tie, so the fixture decides nothing beyond ruling out the English-only ones.
A 1012-note corpus in three languages does decide it. Six queries with a known correct
answer, top 3 each, no threshold:

| query | `gemma-q4` | `granite-gguf` |
|---|---|---|
| Indonesian, "latest news about mama" | right note at 2 | right note at 3 |
| Indonesian, "who helps look after her at home" | topic right, note wrong | **the exact note** |
| Dutch, "when is the boiler serviced" | correct | correct |
| English, "how much do I owe the tax office" | miss | miss |
| English, "court deadline for the claim" | correct | correct |
| English, "when is the car inspection due" | miss | miss |

Four of six each, granite better on two and worse on none. The deciding factor is the score
range rather than the ranking. Granite answers between 0.71 and 0.85 where EmbeddingGemma
answers between 0.47 and 0.57, and ck's default is `--threshold 0.6`. EmbeddingGemma
therefore drops its own correct answers below the cut, and an imperfect search reads as an
empty corpus. Granite's clear it.

EmbeddingGemma wants task prefixes (`task: search result | query: ` on a query,
`title: none | text: ` on a document) and ck applies none, which is the likely cause. Both
sides are unprefixed, so the space is self-consistent but weaker than the model was trained
for. Granite needs no prefix at all, which is why it works as-is. Upstream never applied
nomic's `search_query:` prefix either, and that is probably why nomic scores 1/4 above.

Granite is also Apache-2.0, where the Gemma weights come under the Gemma Terms of Use.
That did not decide it, but it removes a question.

What it costs: granite's context is 512 tokens against EmbeddingGemma's 2048, so the same
corpus produces about 1.8x the chunks and an index about 1.8x the size. Indexing time is
unchanged, because that tracks tokens rather than chunks.

`gemma-q4` remains available and is a reasonable pick for a corpus that is short on disk
and long per document.

**On quantization.** `ggml-org` publishes a QAT Q4_0 of EmbeddingGemma beside the Q8_0;
both score 4/4 and the Q4 separates slightly better. Granite is served at Q8_0. A Q5_K_M
was measured and indexed at exactly the same speed, so there is nothing to win by going
smaller: the work is not bound by weight size.

## Code search is unchanged

Upstream is a code-search tool, and ckq drops its code-specialised model
(`jina-embeddings-v2-base-code`). Measured on eight semantic queries against ckq's own Rust
source, correct file at rank 1:

| | score |
|---|---|
| upstream `bge-small-en-v1.5` | 5/8 |
| ckq `gemma-q4` | 5/8 |

That was measured before granite became the default; granite has not been re-run on code.

Not the same five; they trade. A general text embedder handles code about as well as
`bge-small` does, which was never a code model either. `--full-section`, tree-sitter
chunking and BM25 are model-independent and untouched.

What was given up is unmeasured: `jina-code` was never benchmarked here. If code search
matters as much as prose, it is worth adding back as an alias and testing properly.

## The shared daemon

Each CLI invocation is its own process, so an in-process model cache cannot be shared.
Before the daemon, two agents searching at once loaded the model twice and ran *slower*
than doing it serially, because they contended for the GPU:

| | before | after |
|---|---:|---:|
| single search | 0.40 s | **0.12-0.19 s** |
| two concurrent searches | 0.89 s | **0.15 s** |
| memory, two concurrent | 470 MB | **459 MB, one process** |

How it works. The first invocation finds no socket, re-execs itself as
`ckq --embed-daemon <model>` detached, and waits for it to listen; the daemon
binds its socket **before** it loads the model, so two processes starting
together never both load it — the loser connects to the winner and queues until
it is ready. Every later invocation connects instead. The socket lives in
`$XDG_RUNTIME_DIR` when set (per-user by spec), otherwise `~/.cache/ck/sockets`
(0700), never a world-writable directory; its name is hashed over (model, gguf
file, and the running binary's size and mtime), so a different model is a
different daemon, a stale socket from another model can never be mistaken for a
live one, and a rebuild is not served by the old code. The daemon also refuses
sockets owned by another uid before sending them anything.

The daemon exits after **15 minutes** with no requests. A client can pin it
(`"pin": true` on its connection); the pin lasts exactly as long as that
connection stays open, so a crashed or finished client cannot leave an immortal
daemon behind. `--serve` pins: the MCP server's requests carry the pin, keeping
the daemon warm while the server is actively answering. The daemon's stderr goes to a log beside its socket, and a failed start is read
back out of that log, so a failed start shows its cause
instead of a bare timeout.

Two more edge cases are handled: a socket file left behind by a killed daemon is
removed before binding, and if a daemon exits on its idle timeout in the window
between a client's connect and its request, the client retries once (respawning
the daemon) rather than letting files silently drop out of the index.

`CKQ_IN_PROCESS=1` bypasses the daemon entirely. That is how the daemon itself
loads the model, and the escape hatch if the socket cannot be used.

**Indexing does not use it.** `--index` and `--switch-model` load the model in the
process that runs them. An index run is one long-lived process that uses the model from
start to finish, which is the case the daemon was never for: it exists so that many short
searches share one copy. Going through it costs 26% on a full index, 20.7 s against 15.3 s
on the same corpus over three runs each, and the in-process figure includes loading the
model. The trade is that an index run holds its own copy while a daemon may also be
resident. `CKQ_IN_PROCESS=1` forces the same for everything else.

**Where index time goes**, measured with `CKQ_TIMING=1` on 600 KB of prose: embedding
19.1 s, writing sidecars 0.66 s, chunking 0.26 s, reading files 0.002 s. So 93% is the
model. Neither the decode batch width, nor the model (300M against 278M), nor the
quantization (Q8_0 against Q5_K_M) moved that number at all.

## Why llama.cpp and not ONNX Runtime

ck embeds through `ort`, which has no path to the Apple GPU: its providers are cuda,
tensorrt, openvino, onednn, directml, nnapi, coreml, xnnpack, rocm, acl, armnn, tvm,
migraphx, rknpu, vitis, cann, qnn, webgpu, azure. No MLX, no Metal, and no issue has ever
proposed one. CoreML was tried properly (`MLProgram`, `ComputeUnits::All`,
`FastPrediction`, static shapes, fp16 weights) and every configuration was *slower* than
plain CPU, because the graph splits node by node.

Getting Qwen3-Embedding to run under ONNX at all took four hand-written fixes: last-token
pooling, `position_ids`, an empty KV cache per layer, and a hardcoded head shape. llama.cpp
knows the architectures, so all four disappeared, and `LlamaPoolingType::Unspecified` reads
the pooling out of GGUF metadata, which matters because Qwen is causal (last-token) while
Gemma and Granite are encoders (CLS/mean).

Result on the same 92 KB of notes: ONNX on CPU 45.6 s wall and ~520 s CPU; llama.cpp on
Metal 8.26 s wall and 1.6 s CPU.

Backends are selected per platform in `ck-embed/Cargo.toml`: Metal on macOS, Vulkan
elsewhere. Verified on Arch-family Linux with an AMD Radeon RX 7600 XT, every layer on `Vulkan0`.
CUDA is untested for want of a machine.

## Platforms

| | build | shared daemon |
|---|---|---|
| macOS (Metal) | yes | yes |
| Linux (Vulkan) | yes | yes |
| Windows | yes | **no** |

The daemon speaks over a unix socket, which Rust's std does not expose on Windows, so
there it is compiled out and every process loads its own copy of the model. That is how
ckq worked before the daemon existed: correct, only heavier when several run at once.

CI builds default features on all three and runs the test suite, because the Windows path
is the one nobody here can exercise by hand.

## Traps worth knowing

- **`LlamaContext` is neither `Send` nor `Sync`** and ck embeds from rayon workers, so one
  dedicated thread owns the model and takes jobs over a channel.
- **Rebuilding the context per call costs a 306 MiB allocation**, which was 59% of wall
  time in the kernel. Fixing that was the 23.2 s to 8.26 s win. Batching sequences into one
  decode, the obvious optimisation, changed nothing by comparison.
- **`n_ctx` is split evenly across `n_seq_max` KV slots**, so those two constants are one
  decision. 16 sequences left 512 tokens each and Qwen's 1285-token chunks failed with
  `NoKvCacheSlot`. Now `MAX_SEQ = 4`, `PER_SEQ = 2048`.
- **`LlamaBackend::init()` is once per process**, and a second call trips a Metal teardown
  assert. The worker registry is keyed by model, and leaks its context on exit rather than
  racing Metal's resource sets.
- **A model can be registered in four places** (`ck-models` registry, `ck-embed`'s
  tokenizer table, `ck-chunk`'s chunk config, and the GGUF filename). Missing any leaves a
  default in place and truncates silently. `ck-chunk` has a test over the registry that
  makes it fail loudly.
- **An implausibly fast index means the model never loaded.** A malformed GGUF
  (`key not found in model: bert.context_length`) made ck report a corpus indexed in 0.16 s
  with nothing embedded.

## Relationship to upstream

The three multilingual fastembed models are upstream as
[BeaconBay/ck#199](https://github.com/BeaconBay/ck/pull/199). Everything else here is not,
and should not be: the maintainer has an embedder-trait redesign planned and this fork
cuts straight across it. Upstream also keeps `bge-small` as the default for backwards
compatibility; ckq does not, because English-only is the problem it exists to solve.

## License

MIT OR Apache-2.0, unchanged from upstream. Copyright for the original work remains with
Mike Renwick and the ck contributors; the changes described above are offered under the
same dual licence.

The models are separately licensed: EmbeddingGemma under the Gemma Terms of Use, Granite
under Apache-2.0, BGE-M3 under MIT, Qwen3-Embedding under Apache-2.0. Check them before
commercial use; ckq only downloads them.
