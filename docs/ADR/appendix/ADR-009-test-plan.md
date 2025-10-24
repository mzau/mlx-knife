# ADR-009 Appendix: Test Plan

**Status:** Active – authoritative live-test blueprint for ADR-009  
**Related:** ADR-009 Stop Token Detection Fix  
**Purpose:** Real-model validation strategy for Beta.6

---

## Test Models

### Representative Models (Initial Validation)

| Model | ID | Expected Issue | Purpose |
|-------|----|----|---------|
| **MXFP4** | `mlx-community/gpt-oss-20b-MXFP4-Q8` | `<|end|>` visible in output | Validate stop token fix |
| **Qwen 2.5** | `mlx-community/Qwen2.5-0.5B-Instruct-4bit` | Self-conversation (?) | Validate chat template handling |
| **Llama 3.2** | `mlx-community/Llama-3.2-3B-Instruct-4bit` | None (control) | Regression testing |

**Note:** These 3 models serve as initial validation. Full portfolio testing (below) extends coverage to all MLX models in user cache.

### Portfolio Discovery (Production Validation)

Instead of hard-coded models, iterate over all MLX-compatible models in user cache:

```python
def discover_mlx_models_in_cache(hf_home: str) -> List[ModelInfo]:
    """Scan HF_HOME/hub/models--*/snapshots/* for MLX models.

    Filters:
    - MLX-compatible: Has safetensors + config.json
    - RAM-aware: Estimates model size, skips if exceeds budget

    Returns: List of discovered models with metadata
    """
```

**RAM Gating** (already implemented in `test_stop_tokens_live.py`):
- Progressive budget: 40% (16GB), 50% (32GB), 60% (64GB), 70% (96GB+)
- Auto-skip models exceeding available RAM
- See `get_safe_ram_budget_gb()`, `should_skip_model()` helpers

**Safety:**
- Read-only cache access (no pull/rm)
- Sentinel protection (`TEST-CACHE-SENTINEL`)
- See ADR-007 for CoW constraints

---

## Test Phases

### Phase 1: Baseline Measurement

**Goal:** Document current broken behavior

**Test Case:**
```python
prompt = "Write one sentence about cats."
output = runner.generate_streaming(prompt, max_tokens=50)
```

**Collect:**
- Full generated text
- Token IDs (if accessible)
- Stop condition (why stopped?)
- Visible stop tokens

**Expected Baseline Results:**
- MXFP4: `<|end|>` appears in output ✗
- Qwen: TBD (may self-converse) ?
- Llama: Clean output ✓

### Phase 2: Fix Validation

**After implementing fix, same test case**

**Expected After-Fix Results:**
- MXFP4: No stop tokens visible ✓
- Qwen: No self-conversation ✓
- Llama: Still works (no regression) ✓

### Phase 3: Empirical Mapping

**Document tokenizer configs:**
```python
{
  "model": "gpt-oss",
  "configured_eos": ["<|return|>"],     # From tokenizer
  "generated_tokens": ["<|end|>", ...], # Empirically observed
  "workaround_needed": True/False
}
```

---

## Test Implementation

**File:** `tests_2.0/test_stop_tokens_live.py`

**Markers:**
```python
@pytest.mark.live_stop_tokens  # Requires models downloaded
@pytest.mark.slow              # >1 min per model
```

**Run:**
```bash
# Baseline
pytest tests_2.0/test_stop_tokens_live.py::test_baseline -v -m live_stop_tokens

# After fix
pytest tests_2.0/test_stop_tokens_live.py::test_validation -v -m live_stop_tokens
```

---

## Success Criteria

**Initial Validation (3 Models):**
✅ **Phase 1 Complete:** Baseline measurements documented
✅ **Phase 2 Complete:** All 3 models pass validation tests
✅ **Phase 3 Complete:** Empirical mapping generated (test artifact: `stop_token_config_report.json`)

**Portfolio Validation (All Models in Cache):**
⏳ **Portfolio Discovery:** Planned (currently hard-coded 3-model `TEST_MODELS` dict)
⏳ **Cache Iterator:** Planned (`discover_mlx_models_in_cache()` not yet implemented)
⏳ **Dynamic Validation:** Planned (scale to all models in user cache, not just 3)

---

## Related Documentation

- **ADR-009 Main:** Implementation details, 2-LOC fix, `add_eos_token()` fallback
- **ADR-011:** E2E Live Test Architecture (Server/HTTP/CLI validation, reuses portfolio discovery)
- **TESTING.md:** Live test execution, markers, environment setup
