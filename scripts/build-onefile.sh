#!/usr/bin/env bash
# Build a one-file rogo: the GGUF of <alias> embedded in the binary, so the
# single file works offline on a fresh machine.
#
#   scripts/build-onefile.sh gemma-q4
#
# The GGUF is resolved in ~/.cache/rogo/models (the Hugging Face cache layout
# rogo downloads into) the way hf-hub resolves it: the revision refs/main
# names, not whichever snapshot sorts first; see scripts/resolve-gguf.sh. If
# it is not there yet, run rogo --index once on any folder to download it.
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
ROGO_EMBED_GGUF=$gguf ROGO_EMBED_ALIAS=$alias \
    cargo build --release --locked --manifest-path "$root/Cargo.toml" --features embedded-model

bin=${CARGO_TARGET_DIR:-$root/target}/release/rogo
bytes=$(wc -c < "$bin" | tr -d ' ')
echo "binary: $bin ($bytes bytes, $((bytes / 1024 / 1024)) MiB)"
