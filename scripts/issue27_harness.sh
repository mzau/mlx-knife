#!/usr/bin/env bash
set -euo pipefail

# Safety harness for validating Issue #27 against 1.x without touching user cache.
#
# - Creates an isolated HF_HOME with a test sentinel
# - Verifies mlx_knife resolves MODEL_CACHE into this isolated location
# - Optionally copies a real model from user cache for mutation-based checks
#
# Usage:
#   USER_HF_HOME=${HF_HOME} ./scripts/issue27_harness.sh [org/model]
#
# Notes:
# - Read-only access to USER_HF_HOME; all writes go to a temp HF_HOME
# - Aborts if verification fails at any step

MODEL_SPEC=${1:-"mlx-community/Mistral-7B-Instruct-v0.2-4bit"}

if [[ -z "${USER_HF_HOME:-}" ]]; then
  echo "Please set USER_HF_HOME to your real HF cache root (the directory that contains 'hub')." >&2
  echo "Example: USER_HF_HOME=$HF_HOME ./scripts/issue27_harness.sh" >&2
  exit 2
fi

if [[ ! -d "$USER_HF_HOME/hub" ]]; then
  echo "USER_HF_HOME/hub not found: $USER_HF_HOME/hub" >&2
  exit 3
fi

echo "[1/5] Creating isolated HF_HOME..."
TMPDIR=$(mktemp -d -t mlxk1_issue27_XXXX)
export HF_HOME="$TMPDIR/hf_home"
mkdir -p "$HF_HOME/hub"

echo "[2/5] Adding test sentinel..."
SENTINEL_DIR="$HF_HOME/hub/models--TEST-CACHE-SENTINEL--mlxk1-safety-check/snapshots/main"
mkdir -p "$SENTINEL_DIR"
echo '{"test_cache": true}' > "$SENTINEL_DIR/config.json"

echo "[3/5] Verifying runtime points to isolated cache..."
PY_CACHE_PATH=$(python - <<'PY'
from mlx_knife import cache_utils
print(cache_utils.MODEL_CACHE)
PY
)
EXPECTED_PATH="$HF_HOME/hub"
echo "Resolved MODEL_CACHE: $PY_CACHE_PATH"
echo "Expected MODEL_CACHE: $EXPECTED_PATH"
if [[ "$PY_CACHE_PATH" != "$EXPECTED_PATH" ]]; then
  echo "❌ MODEL_CACHE mismatch — aborting to protect user cache." >&2
  exit 4
fi

echo "[4/5] Copying model into isolated cache (read-only copy from USER_HF_HOME)..."
CACHE_DIR_NAME=$(python - <<PY
print('models--' + "${MODEL_SPEC}".replace('/', '--'))
PY
)
SRC="$USER_HF_HOME/hub/$CACHE_DIR_NAME"
DST="$HF_HOME/hub/$CACHE_DIR_NAME"
if [[ ! -d "$SRC" ]]; then
  echo "Source model not found in USER_HF_HOME: $SRC" >&2
  exit 5
fi
rsync -a "$SRC/" "$DST/"

echo "[5/5] Sanity list in isolated cache..."
echo "HF_HOME=$HF_HOME"
mlxk list --all || true

cat <<MSG

Isolated environment ready.
- To run health:        mlxk list --all --health | grep -i "${MODEL_SPEC##*/}" || true
- To mutate shards:
  SNAP=$(ls -1d "$DST/snapshots"/* | head -n1)
  rm "$SNAP"/model-00002-of-*.safetensors  # example: delete one shard
  : > "$SNAP"/model-00001-of-*.safetensors  # example: truncate one shard
  echo "version https://git-lfs..." > "$SNAP"/model-00003-of-*.safetensors  # LFSify shard
- Then re-check health: mlxk list --all --health | grep -i "${MODEL_SPEC##*/}" || true

IMPORTANT: This harness aborts if MODEL_CACHE != HF_HOME/hub to prevent user cache writes.
MSG

exit 0

