#!/bin/bash
# RAG Pipeline using reusable tools
#
# Usage: ./rag-pipeline.sh <query> [index] [top-k]
#
# Environment:
#   EMBED_MODEL - Embedding model (default: bge-small-en-v1.5-4bit)
#
# Note: mlxk embed is experimental in 2.0.7; this script sets
#       MLXK2_ENABLE_ALPHA_FEATURES=1 for its own embed call.
#
# Example:
#   ./rag-pipeline.sh "How does auth work?" index.jsonl 5
#   EMBED_MODEL=bge-small ./rag-pipeline.sh "query" index.jsonl 3

set -euo pipefail

QUERY="${1:-}"
INDEX="${2:-index.jsonl}"
TOP_K="${3:-3}"
EMBED_MODEL="${EMBED_MODEL:-bge-small-en-v1.5-4bit}"

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

if [ -z "$QUERY" ]; then
    echo "Usage: $0 <query> [index] [top-k]" >&2
    exit 1
fi

# Pipeline: Query → Embed → Search → Retrieve
echo "$QUERY" \
  | MLXK2_ENABLE_ALPHA_FEATURES=1 mlxk embed "$EMBED_MODEL" - \
  | python3 "$SCRIPT_DIR/cosine-search.py" "$INDEX" - --top-k "$TOP_K" --output-json \
  | python3 "$SCRIPT_DIR/retrieve-files.py" --include-score
