#!/usr/bin/env bash

# List Hugging Face models in the user cache that have an index file
# (model.safetensors.index.json or pytorch_model.bin.index.json).
#
# Usage:
#   bash scripts/list-index-models.sh [HF_CACHE_ROOT]
#
# Resolution order for HF cache root:
#   1) first CLI arg
#   2) $MLXK2_USER_HF_HOME
#   3) $HF_HOME

set -euo pipefail

BASE="${1:-${MLXK2_USER_HF_HOME:-${HF_HOME:-}}}"
if [[ -z "${BASE}" ]]; then
  echo "Usage: $0 [HF_CACHE_ROOT]" >&2
  echo "Hint: export MLXK2_USER_HF_HOME=/path/to/huggingface/cache" >&2
  exit 1
fi

HUB_DIR="${BASE%/}/hub"
if [[ ! -d "${HUB_DIR}" ]]; then
  echo "Error: '${HUB_DIR}' not found. Expected HF cache layout at: ${BASE}" >&2
  exit 2
fi

# Find index files and turn cache directories back into repo ids (org/model)
# models--org--model[/optional/segments]/snapshots/<hash>/...
RESULTS=$(find "${HUB_DIR}" -type f \( -name 'model.safetensors.index.json' -o -name 'pytorch_model.bin.index.json' \) 2>/dev/null \
  | sed -E 's#.*/hub/models--(.*)/snapshots/.*#\1#; s#--#/#g' \
  | sort -u || true)

if [[ -z "${RESULTS}" ]]; then
  echo "No index-bearing models found under: ${HUB_DIR}" >&2
  exit 0
fi

echo "Index-bearing models in cache (${HUB_DIR}):"
echo "${RESULTS}"

