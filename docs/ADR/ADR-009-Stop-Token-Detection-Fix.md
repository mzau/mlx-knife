# ADR-009: Stop Token Detection Fix

**Status:** Accepted
**Date:** 2025-10-21
**Supersedes:** Issue #32 discussions (September 2025)
**Affects:** Runner (Beta.6)
**Related:** ADR-010 (Reasoning Content API - Future)

---

## Context

### Problem Statement

Issue #32 requests migration from model-specific workarounds to **generic stop token detection** using native chat templates and mlx-lm APIs.

**Current State:**
- ✅ MXFP4 works (via hardcoded `<|end|>` skip in `stop_tokens.py:49`)
- ❌ Not state-of-the-art (model-specific "Gebastel")
- ❌ Every new model needs custom pattern
- ❌ Runner uses singular `eos_token_id` instead of `eos_token_ids` Set

**Goal:**
Use **mlx-lm TokenizerWrapper APIs** as primary mechanism, fall back to model-specific handling only when needed.

### Root Cause Analysis

**Runner Bug (mlxk2/core/runner/__init__.py:468, 589):**
```python
# CURRENT (checks only singular ID)
if token_id == self.tokenizer.eos_token_id:
    break

# SHOULD BE (checks Set of IDs)
if token_id in self.tokenizer.eos_token_ids:
    break
```

**Why `eos_token_ids` is better:**
- mlx-lm `TokenizerWrapper` normalizes `eos_token_id` → `eos_token_ids` (Set)
- Handles models with multiple EOS tokens (e.g., Llama 3: `[128001, 128009]`)
- Generic mechanism, no model-specific code needed

**Example (MXFP4):**
```python
# HuggingFace config (upstream bug)
tokenizer.eos_token_id = 200002  # Only <|return|>, missing 200007 (<|end|>)

# But added_tokens_decoder has both:
{
  200002: "<|return|>",
  200007: "<|end|>"
}

# Current workaround (stop_tokens.py:49):
if token_content == '<|end|>':
    continue  # Skip adding to stop_tokens

# Hypothesis: 2-LOC fix may be sufficient
# If not, fallback to add_eos_token():
tokenizer.add_eos_token("<|end|>")  # Adds 200007 to eos_token_ids set
```

### Current Workarounds

**Model-Specific Code:**
1. `stop_tokens.py:49` - Hardcoded `<|end|>` skip for MXFP4
2. `stop_tokens.py:92` - Hardcoded `<|return|>` add for gpt-oss
3. `reasoning.py:22-33` - MXFP4 reasoning patterns

**These work, but are not scalable for future models.**

### Constraints

1. **Generic First:** Use mlx-lm APIs, avoid model-specific code when possible
2. **Pragmatic Fallback:** Keep model-specific handling if needed (not all models are perfect)
3. **No Breaking Changes:** Existing models must continue working
4. **Focus Models Only:** Test MXFP4, Qwen 2.5, Llama 3.2 (not all models)

---

## Decision

### Test-Driven Fix Strategy

**Step 1: Implement Real-Model Test Suite**

Required before any code changes - we need empirical data to validate the fix.

```python
# tests_2.0/test_stop_tokens_live.py (see Test Strategy section below)
```

**Step 2: Baseline Measurement**

Document current behavior with existing workarounds:
- MXFP4: Does `<|end|>` appear in output? (expected: NO, via workaround)
- Qwen 2.5: Does self-conversation occur? (expected: document pattern)
- Llama 3.2: Does generation work correctly? (expected: YES)

**Step 3: Apply 2-LOC Fix**

```python
# mlxk2/core/runner/__init__.py:468 (generate_streaming)
if token_id in self.tokenizer.eos_token_ids:  # Changed: == to in
    break

# mlxk2/core/runner/__init__.py:589 (generate_batch)
if token_id in self.tokenizer.eos_token_ids:  # Changed: == to in
    break
```

**Step 4: Re-Test & Evaluate**

Run test suite again. Three possible outcomes:

| Outcome | Action |
|---------|--------|
| ✅ All tests pass | Remove obsolete workarounds, ship Beta.6 |
| ⚠️ Some tests fail | Investigate: Need `add_eos_token()` integration? |
| ❌ Tests still fail | Document findings, implement targeted fixes |

**Step 5: Conditional Cleanup**

```python
# stop_tokens.py:49 - Remove IF tests pass without it
if token_content == '<|end|>':
    continue  # ← DELETE if generic fix works

# stop_tokens.py:92 - Keep IF still needed
if model_type == 'gpt-oss':
    stop_tokens.add('<|return|>')  # Keep with comment: "Upstream config bug"
```

**Step 6 (Optional): add_eos_token() Integration**

If tests reveal that `eos_token_ids` doesn't contain all necessary EOS tokens:

```python
# Option A: In stop_tokens.py:extract_stop_tokens() (after line 55)
# When we find EOS-like tokens in added_tokens_decoder, register them:
if token_content in ['<|end|>', '<|return|>']:  # Derived from added_tokens_decoder values we flag as EOS
    tokenizer.add_eos_token(token_content)
    # NOTE: Modifies tokenizer state, but needed for upstream config bugs

# Option B: In runner/__init__.py:load_model() (after line 192)
# Model-specific fixes after tokenizer load:
if 'mxfp4' in str(model_path).lower():
    self.tokenizer.add_eos_token("<|end|>")
```

**Decision Point:** Only implement Step 6 if empirical testing shows it's necessary.

**Philosophy:**
- **Test-driven** (measure before fixing)
- **Generic first** (2-LOC fix should work for most models)
- **Pragmatic fallback** (`add_eos_token()` only if needed)
- **Not our job to fix all models** (focus on priority models)

### Implementation Status (2025-10-21)

**Steps 1-4: ✅ COMPLETED**
- Real-model test suite implemented in `tests_2.0/test_stop_tokens_live.py` (4 tests, 3 models)
- 2-LOC fix applied in `mlxk2/core/runner/__init__.py:468,590`
- Empirical validation executed (see `stop_token_config_report.json`)
- Results: Generic fix alone **not sufficient** - MXFP4 still requires `add_eos_token()` workaround

**Step 5: ⏸️ SKIPPED**
- Conditional cleanup deferred (workarounds still active)
- Rationale: Step 6 became necessary, re-evaluate cleanup after stabilization

**Step 6: 🔧 ACTIVE (Deterministic Guard)**
- `add_eos_token()` implemented in `mlxk2/core/runner/stop_tokens.py:49-56`
- **Implementation differs from "optional" plan:**
  - Originally planned: "only if empirical tests show it's needed"
  - Actually implemented: Unconditional call whenever `<|end|>` appears in config
  - Rationale: Deterministic guard for MXFP4-class models (pragmatic workaround)
- No tokenizer state mutation side-effects observed (callable check + exception guard)

**Outstanding Work:**
- Portfolio discovery not yet implemented (hard-coded 3 models in test suite)
- Workaround cleanup evaluation (lines 49, 99 in `stop_tokens.py`)
- Empirical validation scope expansion (currently 3 models, aim for full cache coverage)

### Non-Goals (Beta.6)

- ❌ Test all models (unrealistic)
- ❌ Remove all workarounds (only remove obsolete ones)
- ❌ Fix upstream HuggingFace configs (report issues, but don't block on them)
- ❌ Reasoning API changes (see ADR-010)

### Test Strategy

**Real-Model Test Suite Required:**

```python
# tests_2.0/test_stop_tokens_live.py

@pytest.mark.live_stop_tokens
def test_mxfp4_stop_tokens():
    """Verify <|end|> doesn't appear in output."""
    runner = MLXRunner("mlx-community/gpt-oss-20b-MXFP4-Q8")
    response = runner.generate_batch("Write one sentence about cats.", max_tokens=50)

    assert "<|end|>" not in response  # Should be filtered
    assert "<|return|>" not in response  # Should stop before this

@pytest.mark.live_stop_tokens
def test_qwen_self_conversation():
    """Verify model stops before generating turn-taking markers (no self-conversation).

    Self-conversation occurs when the model generates the next user turn prompt
    instead of stopping after its own response. This manifests as chat template
    role markers appearing in the output (e.g., "\\nUser:", "\\nHuman:").

    Expected behavior: Model stops cleanly after its response, before any role markers.
    """
    runner = MLXRunner("mlx-community/Qwen2.5-0.5B-Instruct-4bit")

    # Test with simple prompt that might trigger multi-turn continuation
    response = runner.generate_batch("Hello", max_tokens=50)

    # Assert no role markers from chat template appear in output
    # (These would indicate the model is generating the next turn)
    chat_turn_markers = [
        '\nUser:', '\nHuman:', '\nYou:', '\nAssistant:',
        '\n\nUser:', '\n\nHuman:', '\n\nYou:', '\n\nAssistant:',
        '<|im_start|>user', '<|im_start|>assistant'  # Qwen-specific markers
    ]

    for marker in chat_turn_markers:
        assert marker not in response, (
            f"Self-conversation detected: Found '{marker}' in response. "
            f"Model should stop before generating next turn."
        )

    # Baseline: Verify we got a non-empty response
    assert response.strip(), "Response should not be empty"

@pytest.mark.live_stop_tokens
def test_llama_regression():
    """Ensure Llama still works (control)."""
    runner = MLXRunner("mlx-community/Llama-3.2-3B-Instruct-4bit")
    response = runner.generate_batch("Hi", max_tokens=20)

    assert response  # Should generate something
    assert "<|eot_id|>" not in response  # Stop token filtered
```

**Test Phases:**
1. **Baseline:** Document current behavior (with workarounds)
2. **Generic Fix:** Apply 2-LOC change, test all 3 models
3. **Cleanup:** Remove obsolete workarounds if tests pass

**See:** `docs/ADR/appendix/ADR-009-test-plan.md` for details

---

## Consequences

### Positive

- ✅ **State-of-the-Art:** Uses mlx-lm APIs (same as reference implementation)
- ✅ **Minimal Code Change:** 2 LOC fix (`==` → `in`, twice)
- ✅ **Scalable:** New models automatically supported if configs are correct
- ✅ **Pragmatic:** Model-specific code stays if needed (with clear comments)
- ✅ **Non-Breaking:** Existing models continue working

### Negative

- ⚠️ **Upstream Bugs Remain:** HuggingFace configs may be incomplete
- ⚠️ **Test Dependency:** Requires real models (~3GB download for CI)
- ⚠️ **Partial Coverage:** Only focus models validated, not all models

### Risks & Mitigation

| Risk | Mitigation |
|------|------------|
| Generic approach breaks MXFP4 | Keep workaround if tests fail |
| Unknown models have issues | Users report, we fix incrementally |
| CI becomes slow (model downloads) | Use cached models, mark tests as slow |

### Trade-offs

**Accepted:**
- Not testing all models (focus on priority models only)
- Keeping some workarounds if needed (pragmatism over purity)
- Incremental improvement (not perfect, but better than status quo)

**Rejected:**
- Testing all models (unrealistic)
- Removing all workarounds blindly (risky)
- Waiting for upstream fixes (blocks progress)

---

## Implementation Plan

**Priority:** CRITICAL (Issue #32 open since September)

**Tasks:**
1. ✅ Research findings documented (this ADR)
2. ⏳ Implement real-model test suite (`test_stop_tokens_live.py`)
3. ⏳ Baseline measurement (document current behavior with all 3 models)
4. ⏳ Apply 2-LOC fix (runner/__init__.py:468, 589)
5. ⏳ Re-test & evaluate (compare before/after behavior)
6. ⏳ Conditional: Implement `add_eos_token()` integration (ONLY if tests fail)
7. ⏳ Conditional: Remove obsolete workarounds (ONLY if tests pass without them)
8. ⏳ Update TESTING.md + CHANGELOG.md + close Issue #32

**Estimated Effort:** 2-3 sessions (test suite implementation is non-trivial)
**Blocker for:** 2.0.0 stable release

**Key Decision Gate:** Step 5 → Step 6 (empirical testing determines if `add_eos_token()` is needed)

---

## References

### mlx-lm APIs Used

**TokenizerWrapper** (returned by `mlx_lm.load()`):
```python
# Property
tokenizer.eos_token_ids -> set[int]  # All EOS token IDs

# Method
tokenizer.add_eos_token(token: str) -> None  # Add token to EOS set
```

**Source:**
- `mlx_lm/tokenizer_utils.py:254` (TokenizerWrapper class)
- `mlx_lm/generate.py:701` (usage example: `if token in tokenizer.eos_token_ids`)

### Internal Documents

- **Research Findings (historical background):** `docs/ADR/appendix/ADR-009-research-findings.md`
- **Live Test Plan (authoritative):** `docs/ADR/appendix/ADR-009-test-plan.md`
- **Historical Transcripts:** `docs/ADR/appendix/ADR-009-gpt-oss-interview.md`, `docs/ADR/appendix/ADR-009-september-reasoning-discussion.md`

### External References

- **mlx-lm Source:** https://github.com/ml-explore/mlx-lm
- **HuggingFace MXFP4:** https://huggingface.co/mlx-community/gpt-oss-20b-MXFP4-Q8

### Related Issues

- **GitHub Issue #32:** Replace custom chat format with native Chat Templates
- **Issue #20:** End-Token filtering (defense-in-depth)
- **ADR-010:** Reasoning Content API (Phase 2)

---

**Next Review:** After test suite implementation
**Decision Makers:** Project maintainer
**Stakeholders:** Beta.6 testers, downstream users
