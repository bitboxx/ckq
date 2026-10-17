# Claude Development Guide

This file contains project-specific instructions for Claude and other AI agents working on the ckq codebase.

## What this is

One crate at the repository root: `ckq`, a CLI for multilingual search over a folder of text. The module docs in `src/main.rs` are the design document — read them first. The invariants the design exists for:

1. A document is its root-relative path (`corpus::RelPath`), everywhere: both ranked methods return it, fusion compares it, output prints it. The working directory never matters.
2. The model's prompts are applied in `embed` and nowhere else.
3. Search never writes the index. `--index`, `--add` and `--clean` do.
4. One store: `.ck/index.db` (SQLite, WAL, contentless FTS5 for BM25, f16 vectors).
5. stdout carries results only; progress and banners go to stderr.

## Build and test

```bash
cargo build --release
cargo test                  # offline tests only
cargo e2e                   # plus the end-to-end suites (needs the default model)
cargo known-bugs            # the tests that pin a still-open bug
cargo fmt && cargo clippy --release --all-targets -- -D warnings
```

The default model (about 280 MB) downloads once into `~/.cache/ck/models` on first use. Run `cargo e2e` before any change to the model registry, the prompts, the chunker, the store, the fusion or the ranking: it holds the ranking gate on `bench/multilingual` (gemma-q4: 35/40 rank 1, MRR@5 0.927).

## Test conventions

- End-to-end tests live in `tests/` at the root (`e2e_*.rs`), use `tests/common`, and carry `#[ignore = "e2e: ..."]` so a plain `cargo test` stays offline; `cargo e2e` includes them.
- A test must fail when the search fails: never put its assertions inside `if output.status.success()`.
- A test that pins a known, still-open bug is named `known_bug_*` with `#[ignore = "bug: ..."]`. `cargo e2e` skips those; `cargo known-bugs` runs them. Fixing the bug means dropping both the prefix and the ignore.
- Integration tests in `tests/integration_tests.rs` also carry the `e2e:` ignore: they index real text and need the model.

## Release

A release is a tag `X.Y.Z` (no `v` prefix) that matches the `version` in `Cargo.toml`. `release.yml` verifies the match, builds the binary for x86_64 and aarch64 Linux, x86_64 and aarch64 macOS, and x86_64 Windows, and attaches the archives to a GitHub release. Nothing is published to crates.io or npm: `Cargo.toml` has `publish = false`.

Before tagging: `cargo clippy --release --all-targets -- -D warnings`, `cargo fmt`, `cargo test`, `cargo e2e` — all green, and update `CHANGELOG.md`.
