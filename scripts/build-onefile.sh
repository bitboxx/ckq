#!/usr/bin/env bash
# Build a one-file ckq: the GGUF of <alias> embedded in the binary, so the
# single file works offline on a fresh machine.
#
#   scripts/build-onefile.sh gemma-q4
#
# The GGUF is resolved in ~/.cache/ckq/models (the Hugging Face cache layout
# ckq downloads into) the way hf-hub resolves it -- the revision refs/main
# names, not whichever snapshot sorts first; see scripts/resolve-gguf.sh. If
# it is not there yet, run ckq --index once on any folder to download it.
set -euo pipefail

if [[ $# -ne 1 || -z $1 ]]; then
    echo "usage: $0 <alias>" >&2
    exit 1
fi
alias=$1

root=$(cd "$(dirname "$0")/.." && pwd)
here=$(cd "$(dirname "$0")" && pwd)

gguf=$("$here/resolve-gguf.sh" "$alias")

echo "embedding $alias: $gguf"
CKQ_EMBED_GGUF=$gguf CKQ_EMBED_ALIAS=$alias \
    cargo build --release --locked --manifest-path "$root/Cargo.toml" --features embedded-model

bin=${CARGO_TARGET_DIR:-$root/target}/release/ckq
bytes=$(wc -c < "$bin" | tr -d ' ')
echo "binary: $bin ($bytes bytes, $((bytes / 1024 / 1024)) MiB)"
