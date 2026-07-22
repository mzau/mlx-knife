#!/usr/bin/env bash
set -euo pipefail

# Vision → Text pipeline example (photo archive workflow).
#
# Practical goal:
# - Produce stable Image numbers (Image 1..N) that you can reference later in prompts
# - Pipe the result into a larger chat-only model for synthesis (tags, table, summary)
#
# Notes:
# - This is a CLI workflow (`mlxk run` / `mlx-run`), not the server.
# - Today, this script uses a one-image-at-a-time loop to stay Metal-safe.
# - With ADR-012 Phase 1c (CLI internal batch processing), a single `mlxk run --image ...` invocation
#   should be able to process thousands of images while keeping global Image numbering across chunks.

: "${VISION_MODEL:=pixtral}"
: "${TEXT_MODEL:=Qwen3-Next}"

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 <images...>" >&2
  echo "Example: $0 collection1/*.jpeg" >&2
  exit 2
fi

# Enable stdin pipe semantics (ADR-014 Phase 1)
export MLXK2_ENABLE_PIPES=1

captions_file="$(mktemp -t mlxk-vision-captions.XXXXXX)"
trap 'rm -f "$captions_file"' EXIT

idx=0
for img in "$@"; do
  idx=$((idx + 1))

  # Ask for a short caption only; the script prefixes with a stable Image number.
  caption="$({
    mlx-run "$VISION_MODEL" \
      "Write ONE short caption for this image (<= 20 words)." \
      --temperature 0 \
      --image "$img" \
      --no-stream \
      --no-reasoning \
      2>/dev/null
  } | tr -d '\r' | sed -e 's/[[:space:]]\+$//')"

  printf 'Image %d: %s\n' "$idx" "$caption" >> "$captions_file"
done

# Pipe captions into a chat-only model for structure/archiving.
cat "$captions_file" | mlx-run "$TEXT_MODEL" - \
  "Create a Markdown table with columns: Image, Description, Tags.\n\nThen add a short guess: what island/region could these photos be from?\n\nKeep Image numbers stable and refer to them explicitly." \
  2>/dev/null
