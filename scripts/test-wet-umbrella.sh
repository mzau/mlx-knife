#!/bin/bash
# Run all "real tests" (wet umbrella + isolated cache tests)
# Memory-optimized for large test suites (154+ tests)
set -e

echo "🌂 Wet Umbrella: Running all real tests..."

# Memory-saving pytest options:
# --tb=no: No tracebacks (routine runs expect all PASSED, debug failures individually)
# --capture=sys: System-level capture only (less buffering than --capture=fd)
# For verbose output with portfolio info, run with: pytest -s ...
PYTEST_OPTS="--tb=no --capture=sys"

# Run 1: Compatible live tests (User Cache READ + Workspace)
echo ""
echo "📦 Phase 1: User Cache READ tests (wet umbrella)..."
MLXK2_ENABLE_ALPHA_FEATURES=1 pytest -m wet -v $PYTEST_OPTS

# Run 2: Isolated Cache WRITE - Pull (incompatible with Portfolio)
echo ""
echo "📥 Phase 2: Isolated Cache WRITE - Pull tests..."
MLXK2_TEST_RESUMABLE_DOWNLOAD=1 pytest -m live_pull -v $PYTEST_OPTS

# Run 3: Isolated Cache WRITE - Clone (incompatible with Portfolio)
echo ""
echo "🔄 Phase 3: Isolated Cache WRITE - Clone tests..."
# Note: live_clone tests are opt-in (require env vars), will skip if not configured
MLXK2_ENABLE_ALPHA_FEATURES=1 pytest -m live_clone -v $PYTEST_OPTS

# Run 4: Vision→Geo Pipe Integration
echo ""
echo "🖼️  Phase 4: Vision→Geo Pipe tests..."
# Note: Requires vision model (e.g., pixtral) + text model (e.g., Qwen3-Next)
# Will skip if models not found in cache (graceful degradation)
MLXK2_ENABLE_PIPES=1 pytest -m live_vision_pipe -v $PYTEST_OPTS

echo ""
echo "✅ All real tests completed!"
