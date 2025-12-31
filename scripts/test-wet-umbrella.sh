#!/bin/bash
# Run all "real tests" (wet umbrella + resumable)
# Memory-optimized for large test suites (154+ tests)
set -e

echo "🌂 Wet Umbrella: Running all real tests..."

# Memory-saving pytest options:
# --tb=no: No tracebacks (routine runs expect all PASSED, debug failures individually)
# --capture=sys: System-level capture only (less buffering than --capture=fd)
PYTEST_OPTS="--tb=no --capture=sys"

# Run 1: Compatible live tests (User Cache READ)
echo ""
echo "📦 Phase 1: User Cache READ tests (wet umbrella)..."
MLXK2_ENABLE_ALPHA_FEATURES=1 pytest -m wet -v $PYTEST_OPTS

# Run 2: Isolated Cache WRITE tests (incompatible with Portfolio)
echo ""
echo "📥 Phase 2: Isolated Cache WRITE tests (resumable)..."
MLXK2_TEST_RESUMABLE_DOWNLOAD=1 pytest -m live_resumable -v $PYTEST_OPTS

echo ""
echo "✅ All real tests completed!"
