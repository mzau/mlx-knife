#!/usr/bin/env bash
set -euo pipefail

# Simple local/CI guard: if the spec docs or schema changed, require bump in mlxk2/spec.py
# Bypass: include [no-spec-bump] or [skip-spec-bump] in the latest commit message, or set SPEC_BUMP_BYPASS=1

BASE_REF=${1:-}

if [[ -z "${BASE_REF}" ]]; then
  # Try to find a reasonable base (main branch); fall back to first commit
  if git show-ref --verify --quiet refs/heads/main; then
    BASE_REF="main"
  elif git show-ref --verify --quiet refs/remotes/origin/main; then
    BASE_REF="origin/main"
  else
    BASE_REF=$(git rev-list --max-parents=0 HEAD)
  fi
fi

changed_files=$(git diff --name-only "${BASE_REF}"...HEAD)

spec_changed=false
spec_files=("docs/json-api-specification.md" "docs/json-api-schema.json")
for f in ${spec_files[@]}; do
  if echo "${changed_files}" | grep -q "^${f}$"; then
    spec_changed=true
  fi
done

if [[ "${spec_changed}" != "true" ]]; then
  echo "Spec files unchanged relative to ${BASE_REF}. OK."
  exit 0
fi

if [[ "${SPEC_BUMP_BYPASS:-}" == "1" ]]; then
  echo "Bypass via SPEC_BUMP_BYPASS=1. Skipping spec bump check."
  exit 0
fi

last_msg=$(git log -1 --pretty=%B)
if echo "${last_msg}" | grep -Eqi "\[(no-spec-bump|skip-spec-bump)\]"; then
  echo "Bypass via commit message token [no-spec-bump]/[skip-spec-bump]."
  exit 0
fi

if ! echo "${changed_files}" | grep -q "^mlxk2/spec.py$"; then
  echo "ERROR: Spec docs or schema changed without version bump in mlxk2/spec.py" >&2
  echo " - Changed spec files: $(echo "${changed_files}" | grep -E '^(docs/json-api-specification.md|docs/json-api-schema.json)$' | tr '\n' ' ')" >&2
  echo " - Please update JSON_API_SPEC_VERSION in mlxk2/spec.py and adjust tests accordingly." >&2
  echo " - To bypass for editorial changes, add [no-spec-bump] to the commit message or set SPEC_BUMP_BYPASS=1." >&2
  exit 1
fi

echo "Spec change detected and version bump present in mlxk2/spec.py. OK."

