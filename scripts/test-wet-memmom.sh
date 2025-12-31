#!/bin/bash
# ⚠️  DEBUG TOOL - NOT FOR REGULAR TESTING ⚠️
#
# Session 56-57 Memory Debugging Script
# Purpose: Analyze memory behavior with memmon.py + benchmark reporting
# Created: 2025-12-28 for investigating macOS Tahoe 26.2 RAM tax
#
# ❌ DO NOT USE for regular development testing!
# ✅ USE scripts/test-wet-umbrella.sh instead
#
# This script adds significant overhead:
# - memmon.py (memory monitoring every 200ms)
# - Benchmark JSONL reporting (schema v0.2.0)
# - Memplot generation for timeline visualization
#
# Historical context:
# - Session 56: Investigated 71 GB swap issue
# - Session 57: Discovered macOS Tahoe 26.2 consumes 25 GB more baseline RAM
# - Result: No MLX bug, just Apple TB5 RDMA support overhead
#
# Usage: $0 <signature>
# Example: $0 v4
# Output: benchmarks/reports/YYYY-MM-DD-wet-{memory,benchmark}-<signature>.jsonl

if [ -z "$1" ]; then
  echo "Usage: $0 <signature>"
  echo "Example: $0 v4"
  exit 1
fi

SIGNATURE="$1"
DATE=$(date +%Y-%m-%d)

python -u benchmarks/tools/memmon.py \
  --output benchmarks/reports/${DATE}-wet-memory-${SIGNATURE}.jsonl -- \
  venv310/bin/pytest -m wet -v -s --tb=no --report-output=benchmarks/reports/${DATE}-wet-benchmark-${SIGNATURE}.jsonl
