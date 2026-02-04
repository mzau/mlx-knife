#!/bin/bash
# Run all "real tests" (wet umbrella + isolated cache tests)
# Memory-optimized for large test suites (154+ tests)
#
# Exit code handling: Collects exit codes from all phases, reports at end.
# This allows all phases to run even if earlier phases have failures.

echo "🌂 Wet Umbrella: Running all real tests..."

# Memory-saving pytest options:
# --tb=no: No tracebacks (routine runs expect all PASSED, debug failures individually)
# --capture=sys: System-level capture only (less buffering than --capture=fd)
# For verbose output with portfolio info, run with: pytest -s ...
PYTEST_OPTS="--tb=no --capture=sys"

# Collect exit codes for summary
declare -a PHASE_NAMES=("Phase 1: User Cache READ" "Phase 2: Pull" "Phase 3: Clone" "Phase 4: Vision→Geo Pipe")
declare -a PHASE_EXITS=()

# Run 1: Compatible live tests (User Cache READ + Workspace)
echo ""
echo "📦 Phase 1: User Cache READ tests (wet umbrella)..."
# Override addopts to allow live tests (pytest.ini has -m "not live" for default run)
pytest -m wet -v $PYTEST_OPTS -o addopts=""
PHASE_EXITS+=(${PIPESTATUS[0]:-$?})

# Run 2: Isolated Cache WRITE - Pull (incompatible with Portfolio)
echo ""
echo "📥 Phase 2: Isolated Cache WRITE - Pull tests..."
MLXK2_TEST_RESUMABLE_DOWNLOAD=1 pytest -m live_pull -v $PYTEST_OPTS -o addopts=""
PHASE_EXITS+=(${PIPESTATUS[0]:-$?})

# Run 3: Isolated Cache WRITE - Clone (incompatible with Portfolio)
echo ""
echo "🔄 Phase 3: Isolated Cache WRITE - Clone tests..."
# Note: live_clone tests are opt-in (require env vars), will skip if not configured
pytest -m live_clone -v $PYTEST_OPTS -o addopts=""
PHASE_EXITS+=(${PIPESTATUS[0]:-$?})

# Run 4: Vision→Geo Pipe Integration
echo ""
echo "🖼️  Phase 4: Vision→Geo Pipe tests..."
# Note: Requires vision model (e.g., pixtral) + text model (e.g., Qwen3-Next)
# Will skip if models not found in cache (graceful degradation)
MLXK2_ENABLE_PIPES=1 pytest -m live_vision_pipe -v $PYTEST_OPTS -o addopts=""
PHASE_EXITS+=(${PIPESTATUS[0]:-$?})

# Summary
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "📊 Wet Umbrella Summary:"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

TOTAL_FAILURES=0
for i in "${!PHASE_NAMES[@]}"; do
    EXIT=${PHASE_EXITS[$i]}
    NAME=${PHASE_NAMES[$i]}
    if [ "$EXIT" -eq 0 ]; then
        echo "  ✅ $NAME: PASSED"
    else
        echo "  ❌ $NAME: FAILED (exit $EXIT)"
        TOTAL_FAILURES=$((TOTAL_FAILURES + 1))
    fi
done

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

if [ "$TOTAL_FAILURES" -eq 0 ]; then
    echo "✅ All phases completed successfully!"
    exit 0
else
    echo "❌ $TOTAL_FAILURES phase(s) had failures"
    exit 1
fi
