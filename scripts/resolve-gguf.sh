#!/usr/bin/env bash
# Resolve the GGUF of <alias> in the Hugging Face cache the way hf-hub (and
# so a normal ckq download) resolves it: the commit named by refs/main, when
# that ref exists; the plain glob only as a fallback, and only when it is
# unambiguous. Prints the path, or fails saying why.
#
#   scripts/resolve-gguf.sh gemma-q4
#
# The cache root is ~/.cache/ckq/models, or $CKQ_MODELS_CACHE when set (the
# tests use that to point at a fake cache). build-onefile.sh calls this, and
# so does nothing surprising when the cache holds more than one revision --
# which it does after a re-download following an upstream push: picking the
# lexically-first snapshot there embedded the wrong weights.

set -euo pipefail

if [[ $# -ne 1 || -z $1 ]]; then
    echo "usage: $0 <alias>" >&2
    exit 1
fi
alias=$1

root=$(cd "$(dirname "$0")/.." && pwd)
models_rs=$root/src/models.rs
cache=${CKQ_MODELS_CACHE:-$HOME/.cache/ckq/models}

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

repo_dir=$cache/models--${repo//\//--}
snapshot_dir=$repo_dir/snapshots
ref=$repo_dir/refs/main

if [[ -f $ref ]]; then
    # What hf-hub would serve: the revision the ref names.
    rev=$(<"$ref")
    gguf=$snapshot_dir/$rev/$file
    if [[ ! -f $gguf ]]; then
        echo "error: refs/main of $repo points at $rev, but its snapshot holds no $file." >&2
        echo "delete $repo_dir and let ckq download the repo again" >&2
        exit 1
    fi
else
    shopt -s nullglob
    candidates=("$snapshot_dir"/*/"$file")
    shopt -u nullglob
    if [[ ${#candidates[@]} -eq 0 ]]; then
        echo "error: $file of $repo is not in $snapshot_dir yet." >&2
        echo "run ckq --index once on any folder to download it" >&2
        exit 1
    fi
    if [[ ${#candidates[@]} -gt 1 ]]; then
        echo "error: several snapshot revisions hold $file and there is no refs/main to say which:" >&2
        printf '  %s\n' "${candidates[@]}" >&2
        exit 1
    fi
    gguf=${candidates[0]}
fi

printf '%s\n' "$gguf"
