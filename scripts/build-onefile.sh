#!/usr/bin/env bash
# Build a one-file ckq: the GGUF of <alias> embedded in the binary, so the
# single file works offline on a fresh machine.
#
#   scripts/build-onefile.sh gemma-q4
#
# The GGUF is resolved in ~/.cache/ck/models (the Hugging Face cache layout
# ckq downloads into); if it is not there yet, run ckq --index once on any
# folder to download it. The alias's repo and file are read out of
# src/models.rs, the same registry build.rs validates against.
set -euo pipefail

if [[ $# -ne 1 || -z $1 ]]; then
    echo "usage: $0 <alias>" >&2
    exit 1
fi
alias=$1

root=$(cd "$(dirname "$0")/.." && pwd)
models_rs=$root/src/models.rs
cache=$HOME/.cache/ck/models

# The alias -> (repo, file) row of the registry in src/models.rs.
spec=$(awk -v want="$alias" '
    /alias: "/ { sub(/.*alias: "/, ""); sub(/".*/, ""); in_spec = ($0 == want) }
    in_spec && /repo: "/ { sub(/.*repo: "/, ""); sub(/".*/, ""); repo = $0 }
    in_spec && /file: "/ { sub(/.*file: "/, ""); sub(/".*/, ""); file = $0 }
    END { print repo; print file }
' "$models_rs")
repo=${spec%%$'\n'*}
file=${spec#*$'\n'}

if [[ -z $repo || -z $file ]]; then
    known=$(grep -o 'alias: "[^"]*"' "$models_rs" | sed 's/alias: "//; s/"//' | tr '\n' ' ')
    echo "error: unknown alias '$alias'. Available models: ${known% }" >&2
    exit 1
fi

# The Hugging Face cache layout: models--<org>--<name>/snapshots/<rev>/<file>.
snapshot=$cache/models--${repo//\//--}/snapshots
shopt -s nullglob
candidates=("$snapshot"/*/"$file")
shopt -u nullglob

if [[ ${#candidates[@]} -eq 0 ]]; then
    echo "error: $file of $repo is not in $snapshot yet." >&2
    echo "run ckq --index once on any folder to download it, then rebuild" >&2
    exit 1
fi
gguf=${candidates[0]}

echo "embedding $alias: $gguf"
CKQ_EMBED_GGUF=$gguf CKQ_EMBED_ALIAS=$alias \
    cargo build --release --manifest-path "$root/Cargo.toml" --features embedded-model

bin=${CARGO_TARGET_DIR:-$root/target}/release/ckq
bytes=$(wc -c < "$bin" | tr -d ' ')
echo "binary: $bin ($bytes bytes, $((bytes / 1024 / 1024)) MiB)"
