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

On Windows, set `LLAMA_STATIC_CRT=1` before building. `esaxx-rs`, which `tokenizers`
pulls in, compiles its C++ against the static CRT while llama.cpp defaults to the dynamic
one, and linking the two fails with `LNK2038: mismatch detected for 'RuntimeLibrary'`. The
variable tells llama-cpp-sys-2 to match.

On macOS that is all: Metal is always on. On Linux and Windows the build is CPU-only
unless you ask for the GPU with `--features vulkan`, which needs the Vulkan SDK installed
and `VULKAN_SDK` set. llama.cpp's build script fails the whole build when the feature is
on and the SDK is missing, which is why it is not the default.
Model weights are fetched from Hugging Face on first use and cached.

```bash
ckq --index ~/notes                       # gemma-q4 by default
ckq --sem "when is the boiler serviced" ~/notes
ckq --lex "exact phrase" ~/notes          # BM25
ckq "regex.*here" ~/notes                 # grep-compatible
ckq --serve                               # MCP server; pins the daemon
ckq --index --model granite-gguf ~/notes  # a different model
```

## Status

Working and used daily, but young. It has been exercised on prose and on Rust source, on
Apple Silicon and on Linux/Vulkan. CUDA is untested. The API is upstream's; the model
registry and the daemon are new here and may still move.

## Why EmbeddingGemma, and how it is measured

`bench/multilingual/` holds the benchmark: 40 documents, eight each in English,
Mandarin, Hindi, Spanish and Arabic, over ten subjects with four neighbouring aspects
each, and 40 queries. **Every query is asked in a language other than the one its answer
is written in**, and every wrong answer in a subject is a document about the same subject,
so finding the topic is not enough. `python3 bench/multilingual/run.py <model>...`
reproduces the table.

| model | params | rank 1 | MRR@5 | score range of correct hits |
|---|---:|---:|---:|---:|
| **`gemma-q4` EmbeddingGemma 300M** | **300M** | **35/40** | **0.927** | 0.42-0.74 |
| `bge-m3-gguf` | 568M | 34/40 | 0.904 | 0.38-0.71 |
| `qwen3-gguf` Qwen3-Embedding | 600M | 31/40 | 0.851 | 0.36-0.71 |
| `granite-gguf` granite-embedding-278m | 278M | 28/40 | 0.781 | 0.61-0.88 |

EmbeddingGemma ties a model nearly twice its size and beats both larger ones, at 240 MB on
disk. Upstream's own models are not in the table because they are English-only and score
around 2/4 on even a nine-note multilingual fixture; that is what this fork exists to fix.

### Instruction prefixes

Some models are trained with a different instruction on the query than on the document, and
put the two vectors near each other only when both are present. Upstream ck has one
`embed` method with no way to say which side it is embedding, so no such prefix could ever
be applied. ckq's trait takes a `Role`:

```rust
fn embed_with(&mut self, texts: &[String], role: Role) -> Result<Vec<Vec<f32>>>;
```

The prefixes live in the model registry, empty for the models that want none. For
EmbeddingGemma they are `task: search result | query: ` and `title: none | text: `, and
adding them moved it from 34/40 to 35/40 and from 0.896 to 0.927 MRR. Qwen3 gains about
half as much from its instruction. This is also why upstream's `nomic-embed-text-v1.5`
measured *worse* than a model a quarter of its size: it wants `search_query: ` and never
got it.

Changing a prefix changes every vector, and the model name alone cannot see that, so the
index manifest carries a signature over the model and its prompts. A mismatch refuses to
index rather than mixing two vector spaces in one directory.

### Thresholds are per model

Scores are not comparable between models. On the same 40 queries granite answers its
correct hits between 0.61 and 0.88 while EmbeddingGemma answers the same ones between 0.42
and 0.74. Upstream's flat `--threshold 0.6` therefore throws away most of what
EmbeddingGemma gets right, and an imperfect search reads as an empty corpus. Each model
carries its own default, measured from that benchmark; `--threshold` still overrides it.

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
