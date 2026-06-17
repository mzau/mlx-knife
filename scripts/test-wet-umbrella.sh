#!/bin/bash
# Run all "real tests" (wet umbrella + isolated cache tests)
# Memory-optimized for large test suites (154+ tests)
#
# Exit code handling: Collects exit codes from all phases, reports at end.
# This allows all phases to run even if earlier phases have failures.
#
# CRASH-SAFETY — Phase 1 is path-scoped (see TESTING-DETAILS.md §Known limitation):
# A full-tree `pytest -m wet` aborts fatally (nanobind: duplicate key "cpu" in
# "mlx.core.DeviceType") because marker collection imports the unit-test STUB mlx.core
# (tests_2.0/stubs/) into the same process where in-process-real-mlx tests then init the
# REAL mlx.core (via _use_real_mlx_modules). Native extensions can't re-init. So each
# in-process-real-mlx phase is scoped to its own path, never co-collected with the stub tree:
#   1a = tests_2.0/live              → real world; unit-stub tree never collected
#   1b = the non-live wet remainder  → stub tree only (issue27 + issue37); no real mlx
#   1c = test_stop_tokens_live.py    → real world; stop-token regression (Issue #32),
#        run via -m live_stop_tokens (they SKIP under -m wet), per-model skip if absent.
# 1a+1b+1c cover -m wet plus the opt-in stop-tokens. Phases 2-4 use marker selection
# (separate processes; Phase 2 file-scoped since test_resumable_pull swaps real mlx).

echo "🌂 Wet Umbrella: Running all real tests..."

# Memory-saving pytest options:
# --tb=no: No tracebacks (routine runs expect all PASSED, debug failures individually)
# --capture=sys: System-level capture only (less buffering than --capture=fd)
# For verbose output with portfolio info, run with: pytest -s ...
PYTEST_OPTS="--tb=no --capture=sys"

# Collect exit codes for summary
declare -a PHASE_NAMES=(
    "Phase 1a: wet — live dir (User Cache READ + Workspace)"
    "Phase 1b: wet — non-live (issue27 + issue37)"
    "Phase 1c: stop-tokens regression (-m live_stop_tokens; per-model skip)"
    "Phase 2: Pull"
    "Phase 3: Clone"
    "Phase 4: Vision→Geo Pipe"
)
declare -a PHASE_EXITS=()

# Override addopts to allow live tests (pytest.ini has -m "not live" for default run)

# Phase 1 (wet) is split into three path-scoped runs (see CRASH-SAFETY above).
echo ""
echo "📦 Phase 1a: wet — live dir tests..."
pytest -m wet tests_2.0/live -v $PYTEST_OPTS -o addopts=""
PHASE_EXITS+=(${PIPESTATUS[0]:-$?})

echo ""
echo "📦 Phase 1b: wet — non-live tests..."
pytest -m wet tests_2.0 --ignore=tests_2.0/live --ignore=tests_2.0/test_stop_tokens_live.py -v $PYTEST_OPTS -o addopts=""
PHASE_EXITS+=(${PIPESTATUS[0]:-$?})

echo ""
echo "🛑 Phase 1c: stop-tokens regression (Issue #32; runs when models present)..."
# Opt-in marker so they actually RUN (under -m wet they skip). File-scoped: in-process
# real mlx, must not collect the unit-stub tree. Per-model skip if model/RAM absent.
pytest -m live_stop_tokens tests_2.0/test_stop_tokens_live.py -v $PYTEST_OPTS -o addopts=""
PHASE_EXITS+=(${PIPESTATUS[0]:-$?})

# Run 2: Isolated Cache WRITE - Pull (in-process real mlx → file-scoped, not full tree)
echo ""
echo "📥 Phase 2: Isolated Cache WRITE - Pull tests..."
MLXK2_TEST_RESUMABLE_DOWNLOAD=1 pytest -m live_pull tests_2.0/test_resumable_pull.py -v $PYTEST_OPTS -o addopts=""
PHASE_EXITS+=(${PIPESTATUS[0]:-$?})

# Run 3: Isolated Cache WRITE - Clone (live dir)
echo ""
echo "🔄 Phase 3: Isolated Cache WRITE - Clone tests..."
# Note: live_clone tests are opt-in (require env vars), will skip if not configured
pytest -m live_clone tests_2.0/live -v $PYTEST_OPTS -o addopts=""
PHASE_EXITS+=(${PIPESTATUS[0]:-$?})

# Run 4: Vision→Geo Pipe Integration (live dir)
echo ""
echo "🖼️  Phase 4: Vision→Geo Pipe tests..."
# Note: Requires vision model (e.g., pixtral) + text model (e.g., Qwen3-Next)
# Will skip if models not found in cache (graceful degradation)
MLXK2_ENABLE_PIPES=1 pytest -m live_vision_pipe tests_2.0/live -v $PYTEST_OPTS -o addopts=""
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
