#!/bin/bash
# Benchmark Memory Monitoring Script
# Created: 2026-01-13 (Session 95)
# Purpose: Pre-Phase for ADR-013 Community Model Quality Database
#
# Evolution Roadmap:
# ==================
# Phase 0 (COMPLETE): Fork of wet-memmon with all wet tests
#   - Baseline: All 161 wet tests (98 with model, 63 infrastructure)
#   - Purpose: Establish benchmark methodology with memmon integration
#   - Output: JSONL + memplot for analysis
#
# Phase 1 (CURRENT): Pure inference tests only
#   - Goal: Filter out infrastructure/fixture tests
#   - Target: ~94 real inference tests (duration >= 0.5s)
#   - Method: Use @pytest.mark.benchmark_inference filter
#   - Selection criteria:
#     * Model loaded + inference performed
#     * Meaningful prompt + response validation
#     * Representative of real-world usage
#     * Not infrastructure (portfolio discovery, fixture validation)
#
# Phase 2 (Refinement): Curate benchmark test set
#   - Goal: Select subset of high-value tests for benchmarking
#   - Criteria:
#     * Stop token detection (critical quality metric)
#     * Performance representative (tokens/second)
#     * Vision vs Text coverage
#     * Model size distribution (small/medium/large)
#   - Target: ~30-40 tests (balanced portfolio)
#
# Phase 3 (Template): Independent benchmark suite
#   - Goal: Create standalone mlxk-benchmark package (ADR-013 Phase 1)
#   - Features:
#     * Dedicated benchmark command (mlxk benchmark)
#     * Standardized prompts (deterministic, temperature=0)
#     * JSON report generation (schema v1.0)
#     * Community contribution workflow
#   - Separation: Benchmark ≠ E2E tests (different purposes)
#
# Related:
# - ADR-013: Community Model Quality Database (PROPOSED)
# - wet-memmon.sh: Parent script (Session 56-57, memory debugging)
# - wet-umbrella.sh: E2E test suite (161 tests)
#
# Usage:
#   ./scripts/benchmark-memmon.sh <signature>
#
# Example:
#   ./scripts/benchmark-memmon.sh baseline-v1
#
# Output:
#   benchmarks/reports/YYYY-MM-DD-benchmark-memory-<signature>.jsonl
#   benchmarks/reports/YYYY-MM-DD-benchmark-benchmark-<signature>.jsonl
#   (Note: 'benchmark-benchmark' naming will be refined in Phase 1)

if [ -z "$1" ]; then
  echo "Usage: $0 <signature>"
  echo "Example: $0 baseline-v1"
  echo ""
  echo "Creates benchmark reports with memory monitoring (Pre-Phase for ADR-013)"
  exit 1
fi

SIGNATURE="$1"
DATE=$(date +%Y-%m-%d)

echo "=== Benchmark Memory Monitoring ==="
echo "Phase: 1 (Filtered - pure inference only)"
echo "Signature: ${SIGNATURE}"
echo "Date: ${DATE}"
echo ""
echo "Output files:"
echo "  - benchmarks/reports/${DATE}-benchmark-memory-${SIGNATURE}.jsonl"
echo "  - benchmarks/reports/${DATE}-benchmark-benchmark-${SIGNATURE}.jsonl"
echo ""
echo "Running tests with memory monitoring..."
echo ""

# Run filtered inference tests with memory monitoring
# Phase 1: Use benchmark_inference marker to filter pure inference tests (~94 tests)
# Phase 0 (baseline): Use -m wet for all 161 tests (no longer default)
env MLXK2_ENABLE_PIPES=1 python -u benchmarks/tools/memmon.py \
  --output benchmarks/reports/${DATE}-benchmark-memory-${SIGNATURE}.jsonl -- \
  pytest -m "wet and benchmark_inference" -v -s --tb=no --report-output=benchmarks/reports/${DATE}-benchmark-benchmark-${SIGNATURE}.jsonl -o addopts=""

echo ""
echo "=== Benchmark Complete ==="
echo ""
echo "Next steps:"
echo "1. Generate markdown report:"
echo "   python benchmarks/generate_benchmark_report.py benchmarks/reports/${DATE}-benchmark-benchmark-${SIGNATURE}.jsonl"
echo "3. Analyze memory timeline:"
echo "   python benchmarks/tools/memplot.py benchmarks/reports/${DATE}-benchmark-memory-${SIGNATURE}.jsonl"
echo ""
echo "Phase 1 complete: Filtered pure inference tests (~94 tests)"
echo "Next: Phase 2 (Curation) - select 30-40 high-value tests for dedicated benchmarking"
echo "See BENCHMARK-EVOLUTION.md for roadmap: benchmarks/BENCHMARK-EVOLUTION.md"
