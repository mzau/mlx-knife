#!/usr/bin/env bash
set -euo pipefail

# Non-invasive test of local hooks in a temporary worktree.

if ! git rev-parse --is-inside-work-tree >/dev/null 2>&1; then
  echo "Not inside a Git repository." >&2
  exit 1
fi

ROOT=$(git rev-parse --show-toplevel)
echo "Repo: $ROOT"

echo "[1/3] Testing pre-commit in temp worktree..."
WT=$(mktemp -d)
cleanup() {
  git worktree remove --force "$WT" >/dev/null 2>&1 || true
}
trap cleanup EXIT

git worktree add -f "$WT" HEAD >/dev/null
git -C "$WT" config user.email "local@test"
git -C "$WT" config user.name "Local Test"

(
  cd "$WT"
  echo "test" > AGENTS.md
  git add -f AGENTS.md
  if git commit -m "should be blocked by pre-commit" >/dev/null 2>&1; then
    echo "ERROR: pre-commit did NOT block committing AGENTS.md" >&2
    exit 2
  else
    echo "OK: pre-commit blocked AGENTS.md commit"
  fi
  git restore --staged AGENTS.md >/dev/null 2>&1 || true
  rm -f AGENTS.md
)

echo "[2/3] Testing pre-push blocking..."
HOOKS=$(git rev-parse --git-path hooks)
BR=$(git rev-parse --abbrev-ref HEAD)

if printf "refs/heads/%s 0 refs/heads/%s 0\n" "$BR" "$BR" | "$HOOKS/pre-push" >/dev/null 2>&1; then
  echo "ERROR: pre-push did NOT block current branch" >&2
  exit 3
else
  echo "OK: pre-push blocked current branch"
fi

echo "[3/3] Testing pre-push override..."
if ALLOW_PUSH=1 printf "refs/heads/%s 0 refs/heads/%s 0\n" "$BR" "$BR" | "$HOOKS/pre-push" >/dev/null 2>&1; then
  echo "OK: pre-push override allowed"
else
  echo "ERROR: pre-push override failed" >&2
  exit 4
fi

echo "All hook tests passed."

