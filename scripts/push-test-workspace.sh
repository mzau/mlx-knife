#!/usr/bin/env bash
set -euo pipefail

# Simple helper to push a local test workspace to Hugging Face.
# Usage: scripts/push-test-workspace.sh <org/model> [branch] [commit_message]

REPO_ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WS_DIR="${REPO_ROOT_DIR}/mymodel_test_workspace"

REPO_ID=${1:-}
BRANCH=${2:-main}
COMMIT_MSG=${3:-"mlx-knife push (test workspace)"}

if [[ -z "${REPO_ID}" ]]; then
  echo "Usage: $0 <org/model> [branch] [commit_message]" >&2
  exit 2
fi

if [[ -z "${HF_TOKEN:-}" ]]; then
  echo "HF_TOKEN is not set; export a write-enabled token" >&2
  exit 2
fi

# Prepare workspace (ignored by Git via .gitignore)
mkdir -p "${WS_DIR}"
if [[ ! -f "${WS_DIR}/README.md" ]]; then
  cat >"${WS_DIR}/README.md" <<'EOF'
# Test Workspace for mlxk2 push

This folder is intentionally lightweight and git-ignored.
It is safe to push to a personal HF test repo for validation.
EOF
fi

# Reasonable default exclude rules (merged with hard excludes in code)
cat >"${WS_DIR}/.hfignore" <<'EOF'
.DS_Store
__pycache__/
*.tmp
*.log
*.zip
*.tar
*.tar.gz
.venv/
venv/
EOF

echo "Pushing ${WS_DIR} -> ${REPO_ID}@${BRANCH}"
mlxk2 push "${WS_DIR}" "${REPO_ID}" --create --branch "${BRANCH}" --commit "${COMMIT_MSG}"

