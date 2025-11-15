# ADR-011: E2E Live Test Architecture

**Status:** ✅ IMPLEMENTED (2.0.1+)
**Date:** 2025-10-21 (Updated: 2025-11-12)
**Supersedes:** 1.1.1 `test_end_token_issue.py` comprehensive testing
**Affects:** Test Suite (Stable 2.0.1+)
**Related:** ADR-009 (Stop Token Detection - provides Portfolio Discovery infrastructure)

---

## Context

### Problem Statement

**1.1.1 Had Comprehensive E2E Testing:**
- `test_end_token_issue.py` validated full model portfolio
- Server/HTTP API endpoints tested
- Streaming vs. Non-Streaming parity (Issue #20)
- CLI integration (`run`, `show`)
- RAM-aware portfolio testing

**2.0 Beta Gaps:**
- 95%+ unit tests with mocks/stubs
- <5% live tests (3 hard-coded models in `test_stop_tokens_live.py`)
- No E2E validation for Server/HTTP/CLI paths
- No systematic portfolio coverage beyond stop tokens

**Risk:**
Without E2E live tests, we cannot validate production behavior before Stable release:
- Server API correctness across model portfolio
- Streaming vs. batch parity (Issue #20 regression)
- CLI integration with real models
- Real-world usage patterns

---

## Decision

### E2E Live Test Suite for Stable 2.0

**Reuses ADR-009 Infrastructure:**
- Portfolio discovery (`discover_mlx_models_in_cache()`)
- RAM gating (`get_safe_ram_budget_gb()`, `should_skip_model()`)
- Test fixtures (`_use_real_mlx_modules`, `requires_hf_home`)

**New Test Areas:**

#### 1. Server/HTTP API Validation
```python
# tests_2.0/live/test_server_e2e.py

@pytest.mark.live_e2e
def test_server_streaming_portfolio():
    """Validate /v1/chat/completions SSE streaming over portfolio."""
    for model in discover_portfolio():
        with LocalServer(model) as server:
            response = requests.post(f"{server.url}/v1/chat/completions",
                                    json={"stream": True, ...})
            # Validate SSE format, stop tokens, no visible EOS
```

#### 2. Streaming vs. Non-Streaming Parity (Issue #20)
```python
# tests_2.0/live/test_streaming_parity.py

@pytest.mark.live_e2e
def test_streaming_nonstreaming_parity_portfolio():
    """Validate streaming and non-streaming produce identical output (Issue #20)."""
    for model in discover_portfolio():
        runner = MLXRunner(model)
        batch_output = runner.generate_batch(prompt, max_tokens=50)
        stream_output = "".join(runner.generate_streaming(prompt, max_tokens=50))

        # Issue #20: non-streaming previously had visible stop tokens
        assert batch_output == stream_output
```

#### 3. CLI Integration
```python
# tests_2.0/live/test_cli_e2e.py

@pytest.mark.live_e2e
def test_run_command_portfolio():
    """Validate mlxk run across portfolio."""
    for model in discover_portfolio():
        result = subprocess.run(
            ["mlxk", "run", model.id, "--prompt", "Test"],
            capture_output=True, text=True
        )
        assert result.returncode == 0
        assert "<|end|>" not in result.stdout
```

### Safety Requirements

**Read-Only Cache Access:**
- No pull/rm operations during tests
- Sentinel protection (`TEST-CACHE-SENTINEL` abort)
- Reuses ADR-007 CoW constraints

**RAM Gating:**
- Progressive budget (40%-70%, already implemented in ADR-009)
- Auto-skip models exceeding available RAM

---

## Dependencies

**Requires ADR-009 (Beta.6):**
- Portfolio discovery infrastructure
- RAM gating logic
- Test fixtures

**Relationship:**
- **ADR-009:** Develops portfolio infrastructure, tests Runner-level stop tokens
- **ADR-011:** Reuses portfolio infrastructure, tests E2E APIs

**No overlap:** ADR-009 = Runner tests, ADR-011 = E2E tests

---

## Implementation Plan

**Priority:** HIGH (Required for Stable 2.0)
**Timeline:** Post-Beta.6, before Stable release

**Tasks:**
1. ✅ **Implement ADR-009 Portfolio Discovery** (prerequisite for E2E tests)
   - `discover_mlx_models_in_cache()` helper
   - RAM gating logic (`should_skip_model()`)
2. ✅ **Server E2E Tests** (`test_server_e2e.py`)
   - HTTP API validation
   - SSE streaming format
3. ✅ **Streaming Parity Tests** (`test_streaming_parity.py`)
   - Issue #20 regression protection
4. ✅ **CLI Integration Tests** (`test_cli_e2e.py`)
   - `mlxk run` validation
   - Exit codes, error messages
5. ✅ **Documentation Updates**
   - TESTING.md: E2E test coverage section

---

## Implementation Status

**Status: ✅ COMPLETED** (2025-11-13)

E2E test suite implemented and validated with 17 real MLX chat models.

### Current Results

**Command:** `HF_HOME=/path/to/cache pytest -m live_e2e -v`

- ✅ **72/80 tests passing** (Portfolio: 17 models discovered, 15 testable, 2 RAM-skipped, 8 total skipped)
- ✅ Server E2E: 35 tests (health, models, chat completions, completions - batch + streaming)
- ✅ CLI Integration: 30 tests (text + JSON output, exit codes, stop token filtering)
- ✅ Streaming Parity: 6 tests (Issue #20 protection)
- ✅ Exit Code Tests: 2 tests (Issue #38 validation)
- ⏱️ Duration: ~6-7 minutes

### Key Fixes Applied

For detailed bug analysis and fixes, see CHANGELOG.md 2.0.2 section. Summary:

1. **Stop token ordering bug** (production bug) - Both `generate_batch()` and `generate_streaming()` now filter by earliest position in text
2. **Test temperature flakiness** (test fix) - E2E tests use `temperature=0.0` for deterministic results
3. **Portfolio Discovery collection** (test fix) - Marker check before discovery (keeps default `pytest` fast)
4. **SSE sentinel validation** (test fix) - Explicit `[DONE]` check prevents client hangs
5. **CLI subprocess args** (test fix) - Positional argument instead of `--prompt` flag
6. **MXFP4 reasoning parity** (documented) - Removed from parity tests (ADR-010 known issue)

### Quality Infrastructure

- **Verbose Mode:** `mlxk run --verbose` shows token generation details including multiple EOS token warnings
- **Quality Database:** Known Model Quality Issues tracked in TESTING-DETAILS.md
- **Philosophy:** No hidden workarounds - broken models fail tests and are documented
- **Note:** Initial `MLXK2_DEBUG_TOKENS` E2E test support removed (caused false positives matching metadata)

---

## Implementation Status (2025-11-12)

**Status: ✅ COMPLETED - E2E Test Suite Refactored & Validated**

E2E test suite successfully refactored with production-grade architecture. Validated with 17 real MLX models (15 passed, 2 skipped).

### Refactoring Summary

**1. Portfolio Discovery Refactored** (~70 LOC eliminated)
- **Before:** Duplicated `mlxk list` logic (cache scanning, build_model_object, filtering)
- **After:** Uses `mlxk list --json` via subprocess (production command)
- **Location:** `tests_2.0/test_stop_tokens_live.py` Lines 167-234
- **Benefit:** Tests use production code, automatically benefit from fixes

**2. Test Architecture Fixed** (5 files refactored)
- **Before:** `for model in portfolio:` loops → Server RAM leaks → System freeze
- **After:** `@pytest.mark.parametrize` → One server per test → Clean lifecycle
- **Files Modified:**
  - `tests_2.0/live/server_context.py` - Timeout 5s→30s
  - `tests_2.0/live/conftest.py` - pytest_generate_tests hook + model_info fixture
  - `tests_2.0/live/test_server_e2e.py` - 2 tests parametrized
  - `tests_2.0/live/test_streaming_parity.py` - 2 tests parametrized
  - `tests_2.0/live/test_cli_e2e.py` - 2 tests parametrized

**3. Marker-Required Fixture Added**
- **File:** `tests_2.0/live/conftest.py` - Autouse fixture `_skip_unless_live_e2e_marker`
- **Effect:** E2E tests skipped in default `pytest -v` run (marker-required 🔒)
- **Test counts:** 306 passed, 46 skipped (26 E2E tests auto-skipped)

### Validation Results

**Test Execution:** `TestChatCompletionsBatch` with 17 discovered MLX models
- ✅ **15 passed, 2 skipped** in 82.75s
- ✅ No system freeze, clean RAM cleanup
- ✅ Models tested: Qwen2.5 0.5B, Llama 3.2 3B, Mistral 7B, Mixtral 8x7B, etc.

### Performance Comparison

| Metric | OLD (Broken) | NEW (Fixed) |
|--------|--------------|-------------|
| Architecture | Loop-based | Parametrized |
| Server timeout | 5s → SIGKILL | 30s → clean |
| Test isolation | RAM leaks | Clean per test |
| Discovery | 70 LOC duplicated | `mlxk list --json` |
| Test success | 6/14 → freeze | **15/17 → success** |
| Code reduction | — | ~500 LOC removed |

### Infrastructure Delivered

- ✅ `tests_2.0/live/server_context.py` - LocalServer with 30s timeout
- ✅ `tests_2.0/live/sse_parser.py` - SSE parser utilities
- ✅ `tests_2.0/live/conftest.py` - pytest_generate_tests + marker-required fixture
- ✅ `tests_2.0/live/test_utils.py` - Utility functions
- ✅ `tests_2.0/live/test_server_e2e.py` - Server E2E (parametrized)
- ✅ `tests_2.0/live/test_streaming_parity.py` - Streaming parity (parametrized)
- ✅ `tests_2.0/live/test_cli_e2e.py` - CLI integration (parametrized)
- ✅ Marker: `live_e2e` added to pytest.ini
- ✅ Documentation: TESTING-DETAILS.md updated

### Reused Infrastructure

- ✅ Portfolio Discovery from ADR-009 (refactored to `mlxk list --json`)
- ✅ RAM gating logic
- ✅ MLX modules fixture

**Total Effort:** ~4 hours (refactoring + debugging + validation)

---

## Test Organization

**File Structure:**
```
tests_2.0/
├── test_stop_tokens_live.py       # ADR-009: Runner stop tokens + portfolio
├── live/
│   ├── conftest.py                # Shared fixtures + pytest_generate_tests
│   ├── server_context.py          # LocalServer context manager (30s timeout)
│   ├── sse_parser.py              # SSE parsing utilities
│   ├── test_utils.py              # Utility functions
│   ├── test_server_e2e.py         # ADR-011: Server/HTTP (parametrized)
│   ├── test_streaming_parity.py   # ADR-011: Issue #20 (parametrized)
│   └── test_cli_e2e.py            # ADR-011: CLI (parametrized)
```

**Markers:**
```python
@pytest.mark.live_e2e        # E2E tests (ADR-011)
@pytest.mark.live_stop_tokens # Stop token tests (ADR-009)
@pytest.mark.slow            # Both
```

**Run Strategy** (see TESTING.md for details):
```bash
pytest -m live_stop_tokens  # ADR-009 only
pytest -m live_e2e          # ADR-011 only
pytest                      # Unit tests (skips all live)
```

---

## Consequences

### Positive
- ✅ Production confidence before Stable release
- ✅ 1.1.1 test parity restored
- ✅ Issue #20/#32 regression protection
- ✅ Portfolio coverage (not limited to 3 models)
- ✅ Reusable infrastructure from ADR-009

### Negative
- ⚠️ Portfolio tests take ~80s for 17 models (marker-required to avoid slowing default suite)
- ⚠️ Maintenance overhead if Server API changes
- ⚠️ Requires 30s timeout per test (longer than typical unit tests)

### Trade-offs

**Accepted:**
- Live tests remain opt-in (see TESTING.md)
- Portfolio limited to user's cache (not all HF models)

**Rejected:**
- Testing all HuggingFace Hub models (unrealistic)
- Hard-coding model lists (not scalable)

---

## References

### Related Issues
- Issue #20: End token filtering (streaming vs. non-streaming)
- Issue #32: Stop token detection (ADR-009)

### Related ADRs
- ADR-009: Stop Token Detection Fix (provides portfolio infrastructure)
- ADR-007: Clone Implementation (CoW constraints)
- ADR-004: Enhanced Error Handling (error envelope validation)

### 1.1.1 Test Suite
- `test_end_token_issue.py`: Original comprehensive test (reference)

---

## Success Criteria

**Beta.6 → Stable Transition:**
1. ✅ ADR-009 portfolio discovery implemented
2. ✅ Server E2E tests cover ≥3 models (MXFP4, Qwen, Llama)
3. ✅ Streaming parity validated (Issue #20)
4. ✅ CLI integration tested
5. ✅ Documentation updated

**Definition of Done:**
```bash
pytest -m live_e2e -v  # All tests pass or skip gracefully
```

**✅ ACHIEVED (2025-11-12):**
- 15/17 models tested successfully (2 skipped due to RAM)
- No test failures, only passes and graceful skips
- System stability validated (no freeze, clean RAM cleanup)
- Production-grade architecture (parametrized tests, 30s timeout)
