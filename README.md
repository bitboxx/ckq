# ckq

A fork of [BeaconBay/ck](https://github.com/BeaconBay/ck) for multilingual notes. Binary is
`ckq`, so it sits alongside stock `ck` rather than replacing it.

Two things changed: the embedding models are multilingual and run on the GPU through
llama.cpp, and one shared daemon serves every invocation on the machine.

```bash
cargo build --release -p ck-search --features llamacpp
ckq --index ~/notes            # gemma-q4 by default
ckq --sem "wanneer moet de cv ketel onderhouden worden" ~/notes
ckq --serve                    # MCP server; pins the daemon
```

## Why EmbeddingGemma

Stock ck ships English-only models. On a nine-note English/Dutch/Indonesian fixture,
correct note at rank 1:

| model | params | fixture |
|---|---:|---|
| stock ck `bge-small-en-v1.5` | 33M | 2/4 |
| stock ck `nomic-embed-text-v1.5` | 137M | 1/4 |
| `paraphrase-multilingual-MiniLM` | 118M | 2/4 |
| **`gemma-q4` EmbeddingGemma 300M** | **300M** | **4/4** |
| `granite-gguf` granite-embedding-278m | 278M | 4/4 |
| `bge-m3-gguf` | 568M | 4/4 |
| `qwen3-gguf` Qwen3-Embedding-0.6B | 600M | 4/4 |

Four models tie on accuracy, so the choice is cost. Indexing 347 KB of multilingual text:

| model | Mac Studio (Metal) | nara, AMD RX 7600 XT (Vulkan) |
|---|---:|---:|
| **EmbeddingGemma 300M** | **9.15 s** | 14.21 s |
| granite 278M | 10.21 s | **8.63 s** |
| bge-m3 568M | 15.51 s | - |
| Qwen3 600M | 28.43 s | 53.08 s |

EmbeddingGemma is the fastest of the four on the machine this runs on, at the smallest
size that still scores 4/4.

**The Q4 is quantization-aware trained, and loses nothing.** `ggml-org` publishes a QAT
Q4_0 beside the Q8_0. Both score 4/4, and the Q4 separates *better*: top score 0.604
against Q8's 0.521 on the same query. It is half the size. So `gemma-q4` is the default
and `gemma-gguf` (Q8_0) is kept only for comparison.

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
`ckq --embed-daemon <model>` detached, and waits for it to listen. Every later invocation
connects instead. The socket is at `$TMPDIR/ckq-embed-<hash>.sock`, hashed over
(model, gguf file), so a different model is a different daemon and a stale socket from
another model can never be mistaken for a live one.

The daemon exits after **15 minutes** with no requests. `--serve` sets `CKQ_PIN`, which
tells the daemon to ignore that timeout: an MCP server is long-lived and its next query may
be hours away, so paying a reload would be wrong. A `--serve` that finds a daemon already
running adopts it and pins it rather than starting a second.

Two edge cases are handled: a socket file left behind by a killed daemon is removed before
binding, and two invocations racing to start one both succeed, with the loser connecting to
the winner.

`CKQ_IN_PROCESS=1` bypasses the daemon entirely. That is how the daemon itself loads the
model, and the escape hatch if the socket cannot be used.

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
elsewhere. Verified on Manjaro with an AMD Radeon RX 7600 XT, every layer on `Vulkan0`.
CUDA is untested for want of a machine.

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
