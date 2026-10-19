# Claude Development Guide

This file contains project-specific instructions for Claude and other AI agents working on the rogo codebase.

## What this is

One crate at the repository root: `rogo`, a CLI for multilingual search over a folder of text. The module docs in `src/main.rs` are the design document; read them first. The invariants the design exists for:

1. A document is its root-relative path (`corpus::RelPath`), everywhere: both ranked methods return it, fusion compares it, output prints it. The working directory never matters.
2. The model's prompts are applied in `embed` and nowhere else.
3. Search never writes the index. `--index`, `--add` and `--clean` do.
4. One store: `.rogo/index.db` (SQLite, WAL, contentless FTS5 for BM25, f16
   vectors). A `.ckq` left by ckq 2.0 is legacy, treated like a legacy `.ck`:
   skipped by the walker, left alone by `--clean`, and a command that finds
   one with no `.rogo` fails saying to `mv` it over (the engine never moves
   anything); the index signature carries no name, so a moved index needs no
   rebuild.
5. stdout carries results only; progress and banners go to stderr.
6. The query daemon (`src/daemon.rs`, unix only) embeds queries and nothing
   else: the client sends bare query text, the daemon calls
   `embed::Embedder::embed_query` on its side (invariant 2 holds through it),
   and indexing never uses it. One model per daemon, socket at
   `~/.cache/rogo/sockets/rogo-<alias>-<16 hex of blake3 over the binary's path,
   size and mtime>.sock` so a rebuilt binary never talks to a stale daemon.
   It binds before loading the model (two starting clients load it once) and
   exits after `ROGO_DAEMON_IDLE_SECS` idle (default 900; the tests set 60 in
   `tests/common`). `ROGO_IN_PROCESS=1` skips the daemon; an unreachable one
   falls back to in-process embedding with one stderr warning.

## Build and test

```bash
cargo build --release
cargo test                  # offline tests only
cargo e2e                   # plus the end-to-end suites (needs the default model)
cargo known-bugs            # the tests that pin a still-open bug
cargo fmt && cargo clippy --release --all-targets -- -D warnings
```

The default model (about 280 MB) downloads once into `~/.cache/rogo/models` on first use. Run `cargo e2e` before any change to the model registry, the prompts, the chunker, the store, the fusion or the ranking: it holds the ranking gate on `bench/multilingual`. The enforced bar (in `tests/e2e_quality.rs`) is 34/40 rank 1 and MRR@5 0.90, one query of slack for float drift between Metal, CPU and other backends; gemma-q4 currently scores 35/40 rank 1, MRR@5 0.927.

## GPU backends

macOS builds on Metal. Everywhere else the `vulkan` cargo feature is on by
default, `cuda` is opt-in, and `--no-default-features` is CPU-only. The
feature reaches `llama-cpp-2` through the `gpu/` path crate (`rogo-gpu`, an
empty lib that exists only because cargo cannot make a default feature
platform-specific and refuses one crate twice under two names); on macOS that
crate is not in the graph at all, so the default feature is inert there.
Building with Vulkan needs the Vulkan headers, loader, `glslc` and
SPIRV-Headers (`libvulkan-dev glslc spirv-headers` on Ubuntu; the LunarG SDK,
with `VULKAN_SDK` set, on Windows). The resulting binary needs a Vulkan
loader at runtime (`libvulkan1` / `vulkan-1.dll`) and falls back to the CPU
when there is a loader but no device.

## Test conventions

- End-to-end tests live in `tests/` at the root (`e2e_*.rs`), use `tests/common`, and carry `#[ignore = "e2e: ..."]` so a plain `cargo test` stays offline; `cargo e2e` includes them.
- A test must fail when the search fails: never put its assertions inside `if output.status.success()`.
- A test that pins a known, still-open bug is named `known_bug_*` with `#[ignore = "bug: ..."]`. `cargo e2e` skips those; `cargo known-bugs` runs them. Fixing the bug means dropping both the prefix and the ignore.
- Integration tests in `tests/integration_tests.rs` also carry the `e2e:` ignore: they index real text and need the model.

## Release

A release is a tag `X.Y.Z` (no `v` prefix) that matches the `version` in
`Cargo.toml`. `release.yml` verifies the match, builds the binary for x86_64
and aarch64 Linux, x86_64 and aarch64 macOS, and x86_64 Windows, smoke-tests
each binary in its own job, and attaches the archives, their `.sha256` files
and one combined `SHA256SUMS` to a GitHub release whose notes are the
`CHANGELOG.md` section for that version. A `workflow_dispatch` run builds the
same matrix as a dry run: artifacts only, no release created. A tag run also
attests build provenance for every archive.

crates.io is by hand, after the release is out: `cargo publish` in `gpu/` first
when `rogo-gpu` changed (its version is pinned in the root `Cargo.toml`), then
`cargo publish` at the root. The Homebrew formula lives in `bitboxx/homebrew-tap`
(`Formula/rogo.rb`): bump its version and the four sha256 values (macOS and Linux) from the
release's `SHA256SUMS`.

Before tagging: `cargo clippy --release --all-targets -- -D warnings`,
`cargo fmt`, `cargo test`, `cargo e2e`, all green, and update `CHANGELOG.md`.
