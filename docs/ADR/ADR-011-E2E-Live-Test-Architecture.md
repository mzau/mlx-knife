# ADR-011: E2E Live Test Architecture

**Status:** Proposed (Planned for Post-Beta.6 / Stable 2.0)
**Date:** 2025-10-21
**Supersedes:** 1.1.1 `test_end_token_issue.py` comprehensive testing
**Affects:** Test Suite (Stable 2.0)
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
1. ⏳ **Implement ADR-009 Portfolio Discovery** (prerequisite for E2E tests)
   - `discover_mlx_models_in_cache()` helper
   - RAM gating logic (`should_skip_model()`)
2. ⏳ **Server E2E Tests** (`test_server_e2e.py`)
   - HTTP API validation
   - SSE streaming format
3. ⏳ **Streaming Parity Tests** (`test_streaming_parity.py`)
   - Issue #20 regression protection
4. ⏳ **CLI Integration Tests** (`test_cli_e2e.py`)
   - `mlxk run` validation
   - Exit codes, error messages
5. ⏳ **Documentation Updates**
   - TESTING.md: E2E test coverage section

---

## Implementation Status (2025-10-21)

**Status: NOT STARTED**

All tasks above are pending. This ADR documents the **planned architecture** for E2E tests.

**Current Reality:**
- No E2E test suite exists (`tests_2.0/live/test_server_e2e.py` etc. not created)
- Portfolio discovery not implemented (hard-coded 3 models in `test_stop_tokens_live.py:174`)
- ADR-009 provides **test plan** for portfolio discovery, but implementation deferred

**Blocker:**
- Requires Portfolio Discovery implementation (ADR-009 Step 1, currently incomplete)

**Next Steps:**
1. Complete ADR-009 Portfolio Discovery (Beta.6 scope)
2. Implement E2E test suite (Post-Beta.6, pre-Stable 2.0)

**Estimated Effort:** 2-3 sessions (reuses ADR-009 infrastructure)

---

## Test Organization

**File Structure:**
```
tests_2.0/
├── test_stop_tokens_live.py       # ADR-009: Runner stop tokens + portfolio
├── live/
│   ├── test_server_e2e.py         # ADR-011: Server/HTTP
│   ├── test_streaming_parity.py   # ADR-011: Issue #20
│   └── test_cli_e2e.py            # ADR-011: CLI
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
- ⚠️ Portfolio tests may take 10-30 minutes (10-50 models)
- ⚠️ Maintenance overhead if Server API changes

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

No failures - only passes or skips (RAM/availability).
