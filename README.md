# ckq

A fork of [BeaconBay/ck](https://github.com/BeaconBay/ck) that reaches embedding models
upstream cannot: multilingual encoders, and Qwen3-Embedding. Binary is `ckq`, so it sits
alongside stock `ck` without replacing it.

Started 25 Sep 2026.

## What it adds over ck 0.7.11

| alias | model | dims | context | notes |
|---|---|---|---|---|
| `bge-m3` | BAAI/bge-m3 | 1024 | 8k | multilingual, prefix-free |
| `paraphrase-multilingual` | MiniLM-L12-v2 | 384 | 512 | small multilingual |
| `paraphrase-multilingual-base` | mpnet-base-v2 | 768 | 512 | |
| `qwen3-embed` | Qwen3-Embedding-0.6B ONNX | 1024 | 8k | causal, last-token pooling |

Plus a `Pooling` strategy on the ONNX provider, `position_ids` support, KV-cache
plumbing for causal exports, and the CoreML execution provider for encoder models.

The first three are also upstream as [BeaconBay/ck#199](https://github.com/BeaconBay/ck/pull/199).
The Qwen work is not, and should not be: it touches the code the maintainer is redesigning.

## Getting Qwen to run at all took three fixes

ck's ONNX provider assumes a BERT-style encoder. Qwen3-Embedding is a causal model, and
its ONNX export is the full LM graph:

1. **Pooling.** ck hardcoded first-token (CLS). Causal embedders put the vector at the
   last real token, found via the attention mask. Added `Pooling::{FirstToken,LastToken}`.
2. **`position_ids`.** Declared by the graph; BERT encoders derive them internally.
3. **`past_key_values.N.key/value`.** The export declares a KV cache input per layer, 28
   of them. A prefill needs all 56 present and empty.

All three fail at runtime with an ONNX error, not at load, so a model looks registered and
then indexes nothing.

**Known shortcut:** the KV head shape is hardcoded to Qwen3-Embedding-0.6B's (8 heads, 128
head_dim). A real implementation reads `config.json` beside the weights. `ort`'s `Outlet`
does not expose the declared tensor shape through the API used here.

## Performance: this is the catch

Measured on 92 KB of real markdown, Mac Studio, 12 cores saturated:

| stack | device | 92 KB | extrapolated to 9.7 MB |
|---|---|---:|---:|
| ckq + qwen3-embed | CPU | 45.6 s | **~80 min** |
| ckq + bge-m3 | CPU | 42.4 s | ~75 min |
| ckq + bge-m3 + CoreML | GPU/ANE | 40.7 s | ~71 min |
| LEANN + Qwen3-Embedding-0.6B | MPS | - | **8 min** (measured) |

**ort has no MLX or Metal path.** Its full provider list is cuda, tensorrt, openvino,
onednn, directml, nnapi, coreml, xnnpack, rocm, acl, armnn, tvm, migraphx, rknpu, vitis,
cann, qnn, webgpu, azure. On macOS that means CoreML or WebGPU and nothing else, so the
GPU that LEANN uses through MLX is simply not reachable from ONNX Runtime.

**CoreML does not rescue it.**

- For `qwen3-embed` it cannot be used at all. The empty KV tensors are the problem:
  `"has a dynamic shape ({-1,8,-1,128}) but the runtime shape ({1,8,0,128}) has zero
  elements. This is not supported by the CoreML EP."` It errors rather than falling back,
  so CoreML is gated to `Pooling::FirstToken` models.
- The first measurement of this was wrong: `bge-m3` has `provider: "fastembed"` and never
  touches the ONNX provider CoreML is registered on, so both runs measured fastembed on
  CPU. `mxbai-xsmall` is the only model on that path.
- Corrected, on `mxbai-xsmall`: CPU-only 6.31 s wall / 2.33 s CPU; tuned CoreML 6.48 s
  wall / **6.76 s CPU**. Slightly slower, and nearly 3x the CPU time, which is CoreML
  compiling the graph and then falling back node by node.
- It was tuned, not left on defaults. The defaults are bad: `ModelFormat` defaults to
  `NeuralNetwork`, which supports fewer operators than `MLProgram`, and `MLComputeUnits`
  is unset so the GPU and ANE are never requested. Setting `MLProgram`, `ComputeUnits::All`
  and `FastPrediction` changed nothing.
- **The untested lever was tried and it is worse.** `CKQ_ONNX_FILE` switches the ONNX
  file, and `with_static_input_shapes(true)` matches the recipe that worked in
  [pykeio/ort#341](https://github.com/pykeio/ort/issues/341). On `mxbai-xsmall`, warm
  cache: CPU only **1.01 s**, CoreML + static shapes 2.41 s, CoreML + static shapes +
  fp16 **5.11 s**. Every CoreML configuration is slower than plain CPU, and fp16 is the
  worst of them.
- **CoreML is therefore off by default** and behind `CKQ_COREML=1`. The EP is definitely
  compiled in: 516 CoreML symbols in the statically linked binary, so this is not the
  missing-EP problem rc.13's notes warn about. It is graph splits plus per-call overhead
  on small models with short sequences, which is exactly what ort's maintainer points at
  in #341: "the most important factor for performance is minimizing graph splits".
- Superseded, kept for the record: ck hardcodes `onnx/model_quantized.onnx`, INT8 with QDQ nodes, which
  CoreML handles poorly. Pointing `EMBED_MODEL_PATH` at `onnx/model_fp16.onnx` is the one
  thing that might still work. Moot for `qwen3-embed`, which CoreML refuses whatever the
  format.

So on macOS, ckq is roughly **10x slower than LEANN for the same model**. LEANN is not
redundant here.

## Linux and Windows

This is where ckq should win, and the reason is the same one that hurts it on macOS.

Everything runs through ONNX Runtime, and `ort` exposes CUDA, TensorRT, DirectML, ROCm and
OpenVINO execution providers. The CoreML block is `#[cfg(target_os = "macos")]`, so nothing
blocks a Linux or Windows build. The CPU bottleneck measured above is a macOS problem, not
an ONNX one: there is no good GPU path for this graph on a Mac.

On Linux with CUDA, or Windows with DirectML, `qwen3-embed` should run on the GPU and the
80 minutes should collapse. Untested; no such machine here.

ckq also ships as **one Rust binary** with weights fetched on demand. LEANN needs Python,
torch and a virtualenv. For anything deployed rather than run on a laptop, that matters
more than the benchmark above.

To enable a GPU provider, add the feature to `ort` in the workspace `Cargo.toml` (`cuda`,
`directml`) and register it in `mixedbread.rs` beside the CoreML block.

## ort upstream, checked 25 Sep 2026

`ort` is at **v2.0.0-rc.13** (28 Jul 2026); ck pins `=2.0.0-rc.11` (7 Jan 2026) because
rc.12 made `SessionOptionsPointer` `!Sync`. Nothing in rc.12 or rc.13 changes the macOS
GPU story:

- No MLX provider, and no issue or PR has ever mentioned MLX.
- No Metal provider. The only macOS GPU paths remain CoreML and WebGPU.
- rc.13 adds a link-time error when `download-binaries` is on and the prebuilt binaries
  lack a requested EP, plus a build.rs rerun when Xcode updates. Useful, not relevant here.
- WebGPU on macOS is the one untried path, and
  [#552](https://github.com/pykeio/ort/issues/552) reports it crashing under concurrent
  multi-threaded inference. ck indexes files in parallel with rayon, so that is a direct
  hazard rather than a theoretical one.

Upgrading ort would not help. The gap is that ONNX Runtime has no path to the Apple GPU
that MLX uses, and that has not changed.

## Next: replace ort with llama-cpp-2

ONNX Runtime is the wrong backend for this on a Mac, and no amount of configuration fixes
that. The replacement candidates, checked 25 Sep 2026:

| crate | version | note |
|---|---|---|
| **llama-cpp-2** | v0.1.157, 22 Sep 2026, 1.4M dl | Metal backend, native embedding support |
| candle | v0.11.0, 8.6M dl | pure Rust, `metal` feature, model must be written |
| mlx-rs | v0.32.0, 12 Sep 2026, 376 stars | unofficial MLX bindings, model must be written |

**llama-cpp-2 removes every hack in this fork rather than adding a fifth:**

| hack here | llama.cpp |
|---|---|
| `Pooling::LastToken` plus mask arithmetic | `LLAMA_POOLING_TYPE_LAST`, a flag (also CLS, MEAN, NONE, RANK) |
| `position_ids` plumbing | knows the architecture |
| 56 empty `past_key_values` tensors | manages the KV cache natively |
| hardcoded 8 heads / 128 head_dim | reads GGUF metadata |

Models are first-party: `Qwen/Qwen3-Embedding-0.6B-GGUF` is published by Qwen, and
`gpustack/bge-m3-GGUF` exists. llama.cpp's Metal kernels are the best-tuned on Apple
Silicon, which is the gap ONNX Runtime cannot close.

Cost: a C++ build dependency, so ckq stops being a pure-Rust single binary. llama.cpp
carries CUDA and Vulkan backends, so portability survives, less cleanly.

candle keeps the single binary but needs the Qwen3-Embedding architecture written in Rust
and has thinner Metal coverage. mlx-rs would match LEANN exactly, being the same runtime,
but is unofficial bindings and also needs the architecture written.

If the llama-cpp-2 swap works, ckq makes LEANN redundant: one binary, GPU-fast,
multilingual, no Python.
