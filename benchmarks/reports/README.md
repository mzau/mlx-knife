# Test Reports

This directory contains JSONL test reports from mlx-knife's E2E test suite.

## File Naming Convention

```
YYYY-MM-DD-vX.Y.Z.jsonl
```

Example: `2025-11-16-v2.0.3.jsonl`

## Format

- **One JSON object per line** (JSONL)
- **Schema version:** Each report has `schema_version` field
- **Appending:** New releases append new files (never edit old ones)

## Historical Data

Reports are git-tracked to preserve historical trends:
- Performance changes over releases
- Model compatibility evolution
- Stop token workaround stability

## Analysis Examples

```bash
# Count tests by outcome
jq -r '.outcome' benchmarks/reports/*.jsonl | sort | uniq -c

# Average tokens/sec by model family
jq -r 'select(.performance) | "\(.model.family) \(.performance.tokens_per_sec)"' \
  benchmarks/reports/*.jsonl | \
  awk '{sum[$1]+=$2; count[$1]++} END {for (f in sum) print f, sum[f]/count[f]}'

# List models with workarounds
jq -r 'select(.stop_tokens.workaround != "none") | "\(.model.id): \(.stop_tokens.workaround)"' \
  benchmarks/reports/*.jsonl | sort -u

# Performance regression detection
jq -r 'select(.performance) | "\(.timestamp) \(.model.id) \(.performance.tokens_per_sec)"' \
  benchmarks/reports/*.jsonl | sort
```

## Schema Version

Current: **0.1.0** (Phase 0 - Experimental)

See `../schemas/report-v0.1.schema.json` for details.
