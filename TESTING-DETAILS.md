# MLX Knife Testing - Detailed Documentation

This document contains version-specific details, complete file listings, and implementation specifics for the MLX Knife test suite. For timeless testing philosophy and quick start instructions, see [TESTING.md](TESTING.md).

## Current Status

✅ **2.0.4** — **First stable release with Vision + Audio.** Vision support (CLI + Server, EXIF metadata); Audio transcription (Whisper via mlx-audio); Runtime compatibility accuracy; Server `/v1/audio/transcriptions` endpoint; Probe/Policy architecture; Pipes/Memory-Aware; **Test Portfolio Separation complete**; Workspace Infrastructure (ADR-018 Phase 0a+0b+0c); Convert Operation (ADR-018 Phase 1); Resumable Clone; **Benchmark Schema v0.2.2**.

### Test Results (Official Reference)

**Standard Unit Tests (Multi-Python):**
```
Platform: macOS 26.2 (Tahoe), M2 Max, 64GB RAM
Python 3.10: 697 passed, 13 skipped
Python 3.11: 697 passed, 13 skipped
Python 3.12: 697 passed, 13 skipped
Note: Default suite works on 16GB. Full integration tests: 64GB recommended
```

**Full Integration Tests (`./scripts/test-wet-umbrella.sh`):**
```
Phase 1 (portfolio tests):   179 passed, 73 skipped, 732 deselected
Phase 2-4 (live operations): 3 passed
Total:                       182 passed across all phases
```

✅ **Production verified & reported:** M1, M1 Max, M2 Max in real-world use
✅ **License:** Apache 2.0 (was MIT in 1.x)
✅ **Isolated test system** - user cache stays pristine with temp cache isolation
✅ **3-category test strategy** - optimized for performance and safety
✅ **Portfolio Separation** - Text and Vision models tested independently with separate RAM formulas

### Skipped Tests Breakdown (65 deselected, standard run without HF_HOME)
- **38 Live E2E tests** - Server/HTTP/CLI validation with real models (requires `pytest -m live_e2e`, ADR-011 + Portfolio Separation)
  - **23 Text model tests** - Parametrized across text_portfolio (chat completions batch/streaming)
  - **3 Vision model tests** - Parametrized across vision_portfolio (multimodal, SSE, text-on-vision)
  - **5 Vision CLI E2E tests** - Deterministic vision queries (requires vision model in cache, ADR-012)
  - **11 Audio E2E tests** - Audio transcription (CLI + Server `/v1/audio/transcriptions`) with Whisper models (ADR-020)
  - **3 Non-parametrized tests** - Health, models list, vision→text switching
- **4 Live Stop Tokens tests** - Stop token validation with real models (requires `pytest -m live_stop_tokens`, ADR-009)
- **3 Live Clone tests** - APFS same-volume clone workflow (requires `MLXK2_LIVE_CLONE=1`)
- **2 Issue #37 tests** - Private/org model detection (requires `pytest -m live_run`, Issue #37)
- **2 Runtime Compatibility tests** - Reason chain validation (requires specific model types)
- **1 Live List test** - Tests against user cache (requires HF_HOME with models)
- **1 Live Push test** - Real HuggingFace push (requires `MLXK2_LIVE_PUSH=1`)
- **1 Resumable Pull test** - Real network download with controlled interruption (requires `MLXK2_TEST_RESUMABLE_DOWNLOAD=1`)
- **2 Show Portfolio tests** - Display text/vision portfolios separately (requires HF_HOME)
- **7 Issue #27 tests** - Real-model health validation (requires HF_HOME or MLXK2_USER_HF_HOME setup)

**Portfolio Discovery** (ADR-009) auto-discovers MLX models in user cache using `mlxk list --json`. Validates fixes across the full model portfolio with RAM-aware skipping.

**Note:** Portfolio Discovery only includes **cache models** (HuggingFace cache). Workspace paths (e.g., `./my-workspace`) are not discovered. Models requiring workspace repair (e.g., Gemma-3n for audio) must be tested manually.

---

## Test Execution Guide

| Target | How to Run | Markers / Env | Includes | Network |
|---|---|---|---|---|
| Default suite | `pytest -v` | — | JSON-API (list/show/health), Human-Output, Model-Resolution, Health-Policy, Push Offline (`--check-only`, `--dry-run`), Spec/Schema checks | No |
| Spec only | `pytest -m spec -v` | `spec` | Schema/contract tests, version sync, docs example validation | No |
| Exclude spec | `pytest -m "not spec" -v` | `not spec` | Everything except spec/schema checks | No |
| Push offline | `pytest -k push -v` | — | Push offline tests (`--check-only`, `--dry-run`, error handling); no network, no credentials needed | No |
| Live pipe mode | `MLXK2_ENABLE_PIPES=1 pytest -m live_e2e tests_2.0/live/test_cli_pipe_live.py -v` | `live_e2e`; Env: `HF_HOME`, `MLXK2_ENABLE_PIPES=1` | Stdin `-`, pipe auto-batch, JSON interactive error path, list→run pipe; first eligible model from portfolio discovery | No (uses local cache) |
| Vision→Geo pipe | `MLXK2_ENABLE_PIPES=1 pytest -m live_vision_pipe -v` | `live_vision_pipe` (new marker); Env: `HF_HOME` (requires vision + text models), `MLXK2_ENABLE_PIPES=1`; Optional: `MLXK2_VISION_BATCH_SIZE=N` (default: 1) | **Smoke test for complete Vision→Geo pipeline.** Validates: Vision batch processing (`--chunk 1`), chunk isolation (no state leakage), pipe stdin + `--prompt` combination, geo inference. **PASSED criteria:** Process exits 0, output not empty, output mentions expected terms. Uses `tests_2.0/assets/geo-test/` (9 JPEGs with EXIF). | No (uses local cache) |
| Live push | `pytest -m live_push -v` | `live_push` (subset of `wet`) + Env: `MLXK2_LIVE_PUSH=1`, `HF_TOKEN`, `MLXK2_LIVE_REPO`, `MLXK2_LIVE_WORKSPACE` | JSON push against the real Hub; on errors the test SKIPs (diagnostic) | Yes |
| Live list | `pytest -m live_list -v` | `live_list` (subset of `wet`) + Env: `HF_HOME` (user cache with models) | Tests list/health against user cache models | No (uses local cache) |
| Clone offline | `pytest -k clone -v` | — | Clone offline tests (APFS validation, temp cache, CoW workflow); no network needed | No |
| Live clone (ADR-007) | `pytest -m live_clone -v` | `live_clone` + Env: `MLXK2_LIVE_CLONE=1`, `HF_TOKEN`, `MLXK2_LIVE_CLONE_MODEL`, `MLXK2_LIVE_CLONE_WORKSPACE` | Real clone workflow: pull→temp cache→APFS same-volume clone→workspace (ADR-007 Phase 1 constraints: same volume + APFS required) | Yes |
| Live stop tokens (ADR-009) | `pytest -m live_stop_tokens -v` | `live_stop_tokens`; Optional: `HF_HOME` | Issue #32: Stop token behavior. Uses Portfolio Discovery or fallback models (see below). | No |
| Live run | `pytest -m live_run -v` | `live_run` + `HF_HOME` (needs Phi-3-mini) | Issue #37: Private/org MLX model framework detection. | No |
| Live E2E (ADR-011) | `pytest -m live_e2e -v` | `live_e2e`; Optional: `HF_HOME`; Requires: `httpx` | Server/HTTP/CLI validation. Uses Portfolio Discovery or fallback models. | No |
| Vision E2E (ADR-012) | `pytest -m live_e2e tests_2.0/live/test_vision*.py -v` | `live_e2e`; Optional: `HF_HOME`; Requires: `mlx-vlm` | Vision CLI + Server. Uses Portfolio Discovery or `pixtral-12b-4bit` fallback. | No |
| Audio E2E (ADR-020) | `pytest -m live_e2e tests_2.0/live/test_audio*.py -v` | `live_e2e`; Optional: `HF_HOME`; Requires: `mlx-audio` | Audio transcription + Server. Uses Portfolio Discovery or `whisper` fallback. | No |
| Resumable Pull | `MLXK2_TEST_RESUMABLE_DOWNLOAD=1 pytest -m live_pull tests_2.0/test_resumable_pull.py -v` | `live_pull` (required) + Env: `MLXK2_TEST_RESUMABLE_DOWNLOAD=1` (opt-in for network test) | **✅ Working:** Real network download with controlled interruption (45s timer). Tests unhealthy detection → `requires_confirmation` status → resume with `force_resume=True` → final health check. Validates resumable pull feature (interrupted downloads can be resumed). Uses isolated cache (no impact on user cache). | Yes (HuggingFace download) |
| Show E2E portfolios | `HF_HOME=/path/to/cache python tests_2.0/show_portfolios.py` OR `pytest -m show_model_portfolio -s` | Env: `HF_HOME` | Displays TEXT and VISION portfolios separately. Shows model keys (text_XX, vision_XX), RAM requirements, and test/skip status. Diagnostic tool for understanding portfolio separation. Use script for detailed output, or pytest marker for quick check. | No (uses local cache) |
| Manual debug mode | `mlxk run <model> "test prompt" --verbose` | Manual CLI usage with `--verbose` flag | Shows token generation details including multiple EOS token warnings. Use this for manual debugging of model quality issues. Output includes `[DEBUG] Token generation analysis` and `⚠️ WARNING: Multiple EOS tokens detected` for broken models. | No (uses local cache) |
| Issue #27 real-model | `pytest -m issue27 tests_2.0/test_issue_27.py -v` | Marker: `issue27`; Env (required): `MLXK2_USER_HF_HOME` or `HF_HOME` (user cache, read-only). Env (optional): `MLXK2_ISSUE27_MODEL`, `MLXK2_ISSUE27_INDEX_MODEL`, `MLXK2_SUBSET_COUNT=0`. | Copies real models from user cache into isolated test cache; validates strict health policy on index-based models (no network) | No (uses local cache) |
| Server tests | `pytest -k server -v` | — | Basic server API tests (minimal, uses MLX stubs) | No |

**Quick reference (not in table above):**
```bash
# All live tests (umbrella marker)
pytest -m wet -v

# Show which models will be tested
pytest -m live_e2e --collect-only -q

# Empirical Mapping (heavy, excluded from wet)
pytest -m live_stop_tokens tests_2.0/test_stop_tokens_live.py::TestStopTokensEmpiricalMapping -v
```

---

## Test Directory Organization

### Fundamental Definitions (Single Source of Truth)

**User Cache (Singleton - ONE instance per system)**
```
What:      The HuggingFace cache directory owned by the user
Location:  Defined by HF_HOME environment variable
           Default: ~/.cache/huggingface
           Example: /Volumes/ExternalSSD/huggingface/cache
Lifecycle: Permanent - survives test runs, system reboots
Ownership: Belongs to USER, NOT to tests
Instances: Exactly ONE per system/user
```

**Isolated Cache (Class with instances - MANY per test run)**
```
What:      Temporary cache directories created by isolated_cache fixture
Location:  System temp OR APFS volume root (for CoW optimization)
           Example: /Volumes/ExternalSSD/.mlxk2_test_isolation/mlxk2_test_xyz123/test_cache/hub
Lifecycle: Temporary - created before test, deleted after test (seconds)
Ownership: Belongs to SPECIFIC TEST, isolated from others
Instances: NEW instance PER test function (pytest scope="function")
           5 tests = 5 separate isolated cache instances
```

**Instance Model:**
- **User Cache:** Singleton pattern - shared resource, READ-only in tests
- **Isolated Cache:** Factory pattern - `isolated_cache()` fixture creates new instance each time

**Lifecycle Diagram:**
```
Test 1: def test_foo(isolated_cache):
    [SETUP]   Create /tmp/mlxk2_test_abc123/  ← Instance 1
    [TEST]    Use isolated cache
    [TEARDOWN] Delete /tmp/mlxk2_test_abc123/  ✓ Instance 1 destroyed

Test 2: def test_bar(isolated_cache):
    [SETUP]   Create /tmp/mlxk2_test_xyz789/  ← Instance 2 (NEW, separate)
    [TEST]    Use isolated cache
    [TEARDOWN] Delete /tmp/mlxk2_test_xyz789/  ✓ Instance 2 destroyed
```

**Sentinel Safety Mechanism:**

Every isolated cache contains a sentinel model: `models--TEST-CACHE-SENTINEL--mlxk2-safety-check`

```python
# Fixture setup (Line 464-468 conftest.py)
sentinel_dir = hub_path / TEST_SENTINEL
sentinel_snapshot = sentinel_dir / "snapshots" / "test123..."
sentinel_snapshot.mkdir(parents=True)
(sentinel_snapshot / "config.json").write_text('{"model_type": "test_sentinel", "test_cache": true}')

# Fixture teardown (Line 498-500)
_safe_rmtree(temp_dir_path, signature_id)  # ← Checks signature before delete
```

**How Sentinel protects User Cache:**
1. Test code tries to delete a directory
2. `_safe_rmtree()` checks: Does this directory contain TEST_SENTINEL?
3. **NO** → ❌ REFUSE deletion (could be User Cache!)
4. **YES** → ✅ OK to delete (is Test Cache)

**What it prevents:**
- Accidental deletion if `HF_HOME` wrongly points to User Cache
- Bugs in test code using wrong paths
- Race conditions between tests
- Catastrophic data loss (User Cache = gigabytes of downloaded models)

**Environment Variables during test:**
```python
# Before test:
HF_HOME = /Volumes/ExternalSSD/huggingface/cache  # User Cache

# During test (isolated_cache fixture):
HF_HOME = /tmp/mlxk2_test_abc123/test_cache  # Isolated Cache (instance 1)
MLXK2_STRICT_TEST_DELETE = 1                 # Enable safety checks

# After test:
HF_HOME = /Volumes/ExternalSSD/huggingface/cache  # Restored
MLXK2_STRICT_TEST_DELETE = <original>         # Restored
```

**Workspace (Separate Concept - NOT a Cache)**

Workspace is semantically distinct from Cache - it's a **self-contained, portable** directory for Clone/Push operations.

```
What:      Self-contained directory with model files (standalone health-check capable)
Purpose:   Clone target (output) OR Push source (input)
Location:  User-specified path (production) OR tmp_path (tests)
           Goal: Any location (USB stick, SMB share, different volumes)
           Phase 1: Same APFS volume as cache (CoW optimization)
Structure: Flat directory with config.json + weights (*.safetensors, *.gguf, etc.)
           NOT HuggingFace cache structure (no hub/snapshots/blobs)
Lifecycle: Permanent (user owns it) OR temporary (tests use tmp_path)
Portable:  Yes (conceptually) - can be copied, moved, shared via USB/SMB
```

**Workspace vs Cache:**
- **Cache**: HuggingFace internal structure (hub/models--org--repo/snapshots/...), **many models**
- **Workspace**: User-facing structure (config.json, model.safetensors, tokenizer.json, ...), **exactly one model**

**Self-contained health check:**
- Workspace contains all files needed for validation
- Can be checked without HuggingFace cache
- `mlxk push --check-only workspace/` validates standalone

**Workspace in production:**
- **Clone**: `mlxk clone org/model /path/to/workspace` → Creates workspace at user-specified path
- **Push**: `mlxk push /path/to/workspace org/model` → Uploads from user-specified path
- **Validation**: Clone requires empty directory, Push requires valid model structure
- **Portability**: Phase 1 requires same APFS volume (limitation), future: any location

**Workspace in tests:**
- Uses pytest's `tmp_path` fixture (NOT `isolated_cache`)
- Pattern: `target_dir = str(tmp_path / "workspace")`
- Example: `test_clone_operation.py` line 489, `test_push_workspace_check.py` line 18

**Workspace safety (Clone operation):**
- Temp cache during clone: `.mlxk2_temp_cache_sentinel` (cleanup protection)
- Sentinel is in temp cache, NOT in final workspace
- Temp cache deleted after successful clone → workspace remains

**Workspace dimension in truth table:**

| Operation | Workspace Location | Cache Type | Allowed? |
|-----------|-------------------|------------|----------|
| **Clone (production)** | User-specified path | Isolated temp cache | ✅ |
| **Clone (tests)** | `tmp_path / "workspace"` | Isolated temp cache | ✅ |
| **Push (production)** | User-specified path | N/A (upload only) | ✅ |
| **Push (tests)** | `tmp_path / "ws"` | N/A (offline `--check-only`) | ✅ |
| **Clone/Push (NEVER)** | Inside User Cache | User Cache | ❌ FORBIDDEN |
| **Clone/Push (wrong)** | Inside Isolated Cache | Isolated Cache | ❌ Semantically wrong |

**Why Workspace ≠ Cache:**
- Different structure (flat vs HF nested)
- Different ownership (user vs HF tooling)
- Different purpose (working directory vs download cache)
- Different lifecycle (permanent vs managed)

---

**CRITICAL RULE:** ❌ **NEVER write to User Cache** ❌ All writes must go to isolated cache or external destinations (HuggingFace, workspace).

### Truth Table: Cache Type × Operation

| Cache Type | Read | Write | Allowed? | Example Tests | Directory |
|------------|------|-------|----------|---------------|-----------|
| **User Cache** | ✅ | ❌ | **READ ONLY** | Portfolio Discovery, E2E with real models | `tests_2.0/live/` |
| **Isolated Cache** | ✅ | ✅ | **Both allowed** | Mock models, fresh downloads, safety copies | `tests_2.0/`, `tests_2.0/spec/` |

### Extended Truth Table: Portfolio Discovery Compatibility

Portfolio Discovery fixtures (`vision_portfolio`, `text_portfolio`) use module scope and run subprocesses, creating import-state side-effects. Most tests are compatible, but some require clean import state.

| Test Category          | Cache Type | Fixtures Used           | Portfolio Compatible? | Marker          | Run Group |
|------------------------|------------|-------------------------|-----------------------|-----------------|-----------|
| Portfolio Discovery    | User READ  | vision/text_portfolio   | ✅ (source)           | live_e2e        | wet       |
| Stop Token Validation  | User READ  | vision/text_portfolio   | ✅ (uses it)          | live_stop_tokens| wet       |
| User Cache Read        | User READ  | isolated_cache (copy)   | ✅ (no conflict)      | live_run/list   | wet       |
| Workspace Operations   | N/A        | tmp_path                | ✅ (cache-agnostic)   | live_push       | wet       |
| Issue Reproduction     | User→Iso   | isolated_cache + copy   | ✅ (no conflict)      | issue27         | wet       |
| Isolated Cache Write (Pull) | Isolated | isolated_cache (fresh) | ❌ (import conflict) | live_pull       | separate  |
| Isolated Cache Write (Clone) | Isolated + tmp_path | temp cache (fresh) | ❌ (import conflict) | live_clone | separate |

**Run Groups:**
- `wet`: Can run together in one pytest invocation
- `separate`: Must run in separate pytest process

**User Experience:**
```bash
./scripts/test-wet-umbrella.sh  # Runs both groups automatically
```

### Decision Tree: Categorizing New Tests

When writing a new test, follow this decision tree:

**Question 1:** Does your test use Portfolio Discovery fixtures (vision/text_portfolio)?
├─ YES → Marker: `live_e2e` or `live_stop_tokens` → Run group: **wet** ✅
└─ NO  → Continue to Question 2

**Question 2:** Does your test need fresh downloads (Isolated Cache WRITE)?
├─ YES (Pull) → Marker: `live_pull` → Run group: **separate** ⚠️
├─ YES (Clone with internal pull) → Marker: `live_clone` → Run group: **separate** ⚠️
└─ NO  → Continue to Question 3

**Question 3:** What does your test use?
├─ User Cache (READ only)     → Markers: `live_run`, `live_list`, `issue27` → Run group: **wet** ✅
├─ Workspace (tmp_path only)   → Marker: `live_push` → Run group: **wet** ✅
└─ Isolated Cache (copy/mock)  → Standard markers → Unit test (not live)

### Writing New Live Tests: Umbrella Marker Convention

**CRITICAL:** All live tests (real models/network/user cache) MUST include `pytest.mark.live`:

```python
# tests_2.0/live/test_my_new_live_test.py
pytestmark = [pytest.mark.live, pytest.mark.live_e2e]  # Umbrella + specific marker

def test_my_feature(text_portfolio):
    pass
```

**Why:** Default test run excludes ALL `live` tests via `pytest -m "not live"` (used in `test-multi-python.sh`). New live tests are automatically excluded without script changes.

### Fixture Guidelines (Schema v0.2.1 - Benchmark Modality Detection)

**CRITICAL:** New live tests MUST use modality-specific fixtures for accurate benchmark reporting:

```python
# ✅ CORRECT - Use modality-specific fixtures
def test_my_text_feature(text_model_key, text_model_info):
    """Text inference test - automatically tagged as 'text' modality."""
    pass

def test_my_vision_feature(vision_model_key, vision_model_info):
    """Vision inference test - automatically tagged as 'vision' modality."""
    pass

# ❌ DEPRECATED - Avoid legacy fixtures
def test_old_style(model_key):  # Don't use - shows as "Unknown (legacy)" in reports
    pass
```

**Available Fixtures:**

| Fixture | Modality | Use Case |
|---------|----------|----------|
| `text_model_key` | Text | Parametrized text model tests |
| `text_model_info` | Text | Access model metadata (size, path) |
| `vision_model_key` | Vision | Parametrized vision model tests |
| `vision_model_info` | Vision | Access vision model metadata |
| `audio_model_key` | Audio | Parametrized audio model tests |
| `audio_model_info` | Audio | Access audio model metadata |

**DEPRECATED Fixtures (do not use in new code):**

| Deprecated | Replacement | Reason |
|------------|-------------|--------|
| `model_key` | `text_model_key` | No modality detection |
| `portfolio_models` | `text_portfolio` | Ambiguous modality |

**How Modality Detection Works (Schema v0.2.1):**

The pytest hooks in `tests_2.0/conftest.py` automatically detect inference modality:

1. **Fixture-based detection:** Tests using `text_model_key` → `inference_modality: "text"`
2. **Fixture-based detection:** Tests using `vision_model_key` → `inference_modality: "vision"`
3. **Fixture-based detection:** Tests using `audio_model_key` → `inference_modality: "audio"`
4. **Explicit override:** Pipe tests can set modality via `request.node.user_properties`
5. **Legacy fallback:** Tests without modality fixtures → `inference_modality: "unknown"`

**Why This Matters:**

Benchmark reports differentiate Vision vs Text inference for mixed-modality models:

```
Model                       Size     Mode   Tests  Time      RAM (GB)
pixtral-12b-8bit           12.6GB   Vision 8      316.0s    17.5-29.1
pixtral-12b-8bit           12.6GB   Text   1       14.3s    20.3
```

Without modality-specific fixtures, tests appear as "Unknown (legacy)" - making reports less useful.

**Non-Parametrized Tests:**

For tests that don't use parametrized fixtures but still need modality reporting:

```python
@pytest.fixture(autouse=True)
def _report_text_modality(request):
    """Explicitly tag non-parametrized tests as text inference."""
    request.node.user_properties.append(("inference_modality", "text"))
```

See `tests_2.0/live/test_cli_pipe_live.py` for an example.

### Schema Field Development (Developer Guide)

**CRITICAL:** When adding new fields to the benchmark report schema, follow these steps carefully. Missing any step will result in silent failures (fields missing from JSONL output).

#### Case Study: Schema v0.2.2 Bug (test_start_ts / test_end_ts)

**What went wrong:**
- Added `test_start_ts`/`test_end_ts` to schema JSON ✅
- Wrote pytest hooks to capture timestamps ✅
- **FORGOT** `@pytest.hookimpl` decorator ❌ → hooks never executed
- **FORGOT** to add fields to whitelist (conftest.py:1534) ❌
- Result: 120 tests ran, **0 had timestamps** in JSONL → schema validation failed

This bug was discovered during beta.9 benchmark run and cost a full re-run.

#### Step-by-Step: Adding New Schema Fields

**1. Update Schema JSON**

```bash
# Create new schema version
cp benchmarks/schemas/report-v0.2.1.schema.json \
   benchmarks/schemas/report-v0.2.2.schema.json

# Edit schema: Add new fields with descriptions
# Update: "title": "MLX Knife Benchmark Report Schema v0.2.2"
```

**2. Register pytest Hooks (CRITICAL)**

If capturing data during test execution, hooks MUST have `@pytest.hookimpl`:

```python
# tests_2.0/live/conftest.py

import pytest
import time

# Define StashKeys for data storage (pytest 7.0+ API)
my_field_key = pytest.StashKey[float]()

@pytest.hookimpl(tryfirst=True)  # ← REQUIRED! Without this, hook is IGNORED
def pytest_runtest_setup(item):
    """Capture data at test start."""
    item.stash[my_field_key] = time.time()

@pytest.hookimpl(tryfirst=True)  # ← REQUIRED!
def pytest_runtest_makereport(item, call):
    """Add data to benchmark report via user_properties.

    CRITICAL: Uses tryfirst=True to run BEFORE conftest.py's
    hookwrapper=True that writes JSONL.
    """
    if call.when == "call":
        my_value = item.stash.get(my_field_key, None)
        if my_value:
            item.user_properties.append(("my_field", my_value))
```

**Without `@pytest.hookimpl`:** Hook is silently ignored, no error, no data.

**3. Update Whitelist in conftest.py**

```python
# tests_2.0/conftest.py (around line 1534)

for key, value in item.user_properties:
    if key in ("model", "performance", "stop_tokens", "system",
               "test_start_ts", "test_end_ts", "my_field"):  # ← Add new field here
        # Top-level keys
        data[key] = value
    else:
        # Everything else → metadata
        data.setdefault("metadata", {})[key] = value
```

**Without whitelist entry:** Field goes to `metadata` instead of top-level.

**4. Document Migration**

```markdown
# benchmarks/schemas/MIGRATIONS.md

### 0.2.3 (YYYY-MM-DD) - My Feature

Added fields:
- `my_field`: Description of what this captures

Purpose: Why we added this field

Breaking changes: None (backward compatible)
```

**5. Update Schema Symlink**

```bash
cd benchmarks/schemas/
rm report-current.schema.json
ln -s report-v0.2.3.schema.json report-current.schema.json
```

**6. Verify in Test Run**

```bash
# Run single test with report output
pytest tests_2.0/live/test_cli_e2e.py::test_run_command -v \
  --report-output /tmp/test.jsonl

# Check if field appears
head -1 /tmp/test.jsonl | python3 -m json.tool | grep my_field
```

**Expected:** `"my_field": 1738328572.96` (or your value)
**If missing:** Check `@pytest.hookimpl` decorator and whitelist!

#### Hook Execution Order

pytest hooks run in specific order. For benchmark fields:

```python
# Early hooks (data capture)
@pytest.hookimpl(tryfirst=True)
def pytest_runtest_setup(item):
    """Runs FIRST - capture start state."""
    pass

@pytest.hookimpl(trylast=True)
def pytest_runtest_teardown(item):
    """Runs LAST - capture end state."""
    pass

# Report hook (data serialization)
@pytest.hookimpl(tryfirst=True)  # ← CRITICAL for makereport
def pytest_runtest_makereport(item, call):
    """Runs BEFORE conftest.py's hookwrapper=True.

    Must add to user_properties BEFORE JSONL is written.
    """
    pass
```

**Why `tryfirst=True` for makereport?**
- conftest.py's `pytest_runtest_makereport` has `hookwrapper=True`
- Hookwrappers run around normal hooks
- `tryfirst=True` ensures data is in user_properties BEFORE JSONL write

#### Testing Checklist

Before committing new schema fields:

- [ ] Schema JSON created with version bump
- [ ] pytest hooks have `@pytest.hookimpl` decorator
- [ ] Fields added to conftest.py whitelist (line ~1534)
- [ ] MIGRATIONS.md updated
- [ ] report-current.schema.json symlink updated
- [ ] Test run confirms fields appear in JSONL
- [ ] Schema validation passes: `python benchmarks/validate_reports.py`

**Pro tip:** Test with a SINGLE test first before running full benchmark suite!

### Compatibility Rule (Technical Background)

**Why separate runs?**

Portfolio Discovery fixtures manipulate import state via subprocesses:
- Subprocess runs: `mlxk list --json [--vision]`
- Side-effect: Imports `mlx_lm`, `mlx_vlm`, `transformers`, `huggingface_hub` into pytest process
- Module scope: Remains active across all tests in run

**Compatible tests:** Can tolerate polluted import state
- Tests using Portfolio fixtures (they expect it)
- Tests with User Cache READ (no import-sensitive operations)
- Tests with Workspace (cache-agnostic)

**Incompatible tests:** Require clean import state
- `live_pull`: HuggingFace Hub needs clean imports for symlink creation
- Fresh downloads fail with polluted `sys.modules`

**Solution:** Separate pytest runs ensure clean import state for incompatible tests.

For pytest implementation details, see Appendix below.

### conftest.py Scope Rules

**Pytest conftest.py files form a hierarchy** - parent conftest applies to all children, but child conftest should ONLY apply to their directory.

**Rule:** Subdirectory `conftest.py` files MUST limit their scope to their own directory to avoid interfering with sibling/parent tests.

**Implementation pattern:**
```python
# tests_2.0/live/conftest.py

@pytest.fixture(scope="function", autouse=True)
def _skip_unless_live_e2e_marker(request):
    """Autouse fixture that ONLY applies to tests in live/ directory."""
    # CRITICAL: Early return for tests outside this directory
    test_path = str(request.node.path)
    if "/live/" not in test_path and "\\live\\" not in test_path:
        return  # Skip fixture for tests outside live/ directory

    # Rest of fixture logic...

def pytest_generate_tests(metafunc):
    """Hook that ONLY applies to tests in live/ directory."""
    # CRITICAL: Early return for tests outside this directory
    test_path = str(metafunc.definition.path)
    if "/live/" not in test_path and "\\live\\" not in test_path:
        return  # Skip hook for tests outside live/ directory

    # Rest of hook logic (Portfolio Discovery, parametrization, etc.)
```

**Rationale:**
- `tests_2.0/live/` contains User Cache tests (Portfolio Discovery)
- `tests_2.0/` contains Isolated Cache tests (fresh downloads, mocks)
- Portfolio Discovery hooks (`pytest_generate_tests`, `autouse` fixtures) should NOT apply to Isolated Cache tests
- Without scope limitation: `live/conftest.py` hooks interfere with `isolated_cache` fixture in parent tests

**Test hierarchy:**
```
tests_2.0/
├── conftest.py              # Global: applies to ALL tests (isolated_cache, assert_is_test_cache, etc.)
├── test_resumable_pull.py   # Uses isolated_cache → must NOT be affected by live/conftest.py
├── live/
│   ├── conftest.py          # Local: MUST limit scope to /live/ only (Portfolio Discovery)
│   └── test_server_e2e.py   # Uses Portfolio Discovery → affected by live/conftest.py
└── spec/
    ├── conftest.py          # (if exists) Local: MUST limit scope to /spec/ only
    └── test_*.py
```

**Verification:**
- Test in `tests_2.0/test_resumable_pull.py` with marker `live_pull` should collect as 1 test (NOT parametrized)
- Test in `tests_2.0/live/test_server_e2e.py` with `text_model_key` should parametrize across portfolio

### Test Categories by Cache Strategy

**Category 1: User Cache READ (Portfolio Discovery)**
- **Location:** `tests_2.0/live/`
- **Cache:** Direct User Cache via `HF_HOME` environment
- **Operation:** READ only (via `mlxk list --json`, server load, etc.)
- **Examples:**
  - `test_server_e2e.py` - Parametrized tests across text_portfolio (23 models)
  - `test_vision_e2e_live.py` - Vision CLI with real models from cache
  - `test_stop_tokens_live.py` - Stop token validation with discovered models
- **Why in live/:** Portfolio Discovery hooks (`pytest_generate_tests`) auto-discover models from User Cache
- **Count:** 10+ tests

**Category 2: Isolated Cache WRITE (Fresh State)**
- **Location:** `tests_2.0/`
- **Cache:** Isolated temp cache (empty at start)
- **Operation:** WRITE (download, create mock models)
- **Examples:**
  - `test_resumable_pull.py` - Downloads fresh from HuggingFace into isolated cache
  - `test_robustness.py` - Creates mock model files for testing
  - `test_integration.py` - Synthetic model creation for integration tests
- **Why in parent:** Needs clean pytest environment without Portfolio Discovery hooks
- **Count:** 15+ tests

**Category 3: Isolated Cache READ (Safety Copies)**
- **Location:** `tests_2.0/`
- **Cache:** Isolated temp cache (copied from User Cache)
- **Operation:** WRITE (copy) then READ (test on copy)
- **Examples:**
  - `test_issue_27.py` - Copies real models from User Cache, mutates copies, tests health
  - `test_issue_37_private_org_regression.py` - Copies model, renames to simulate private repo
- **Why copy:** Protects User Cache from mutations (delete shards, truncate, inject LFS pointers)
- **Count:** 2 tests

**Category 4: Spec/Schema Validation**
- **Location:** `tests_2.0/spec/`
- **Cache:** Isolated cache with minimal mocks
- **Operation:** WRITE (mock creation) + READ
- **Examples:**
  - `test_cli_commands_json_flag.py` - JSON output format validation
  - `test_code_outputs_validate_against_schema.py` - Schema compliance
- **Why isolated:** Fast, deterministic, no real models needed
- **Count:** 7 tests

**Special Cases (External Writes):**
- `test_push_live.py` - Writes to **HuggingFace** (not local cache) ✅ Allowed
- `test_clone_live.py` - Writes to **workspace directory** (not cache) ✅ Allowed

### Why Resumable Pull MUST stay in `tests_2.0/`

**The conflict:**
- Resumable Pull needs **empty isolated cache** (Category 2)
- `tests_2.0/live/` expects **populated User Cache** (Category 1)
- Portfolio Discovery hooks (`pytest_generate_tests`) run during collection, expecting models in HF_HOME
- When test uses `isolated_cache` in `live/`, hooks interfere with cache isolation

**Observed:**
- ✅ `tests_2.0/test_resumable_pull.py` → 2.15GB downloaded, PASS
- ❌ `tests_2.0/live/test_resumable_pull.py` → 0 bytes downloaded, FAIL

### Decision Tree: Where does my test belong?

**Ask yourself:**

1. **Does my test READ from User Cache?**
   - YES, and needs Portfolio Discovery (test many models) → `tests_2.0/live/`
   - YES, but only one specific model → Copy to isolated cache, use `tests_2.0/`
   - NO → Continue to question 2

2. **Does my test WRITE?**
   - Write to **User Cache** → ❌ **FORBIDDEN** - redesign your test
   - Write to **isolated cache** (fresh download, mock creation) → `tests_2.0/`
   - Write to **external** (HuggingFace, workspace) → `tests_2.0/` or `tests_2.0/live/` depending on read source

3. **Does my test use mock/stub models?**
   - YES, schema validation only → `tests_2.0/spec/`
   - YES, but needs integrated testing → `tests_2.0/`
   - NO → See questions 1-2

**Summary:**
- User Cache READ + Many models → `tests_2.0/live/` (Portfolio Discovery)
- User Cache READ + Safety copies → `tests_2.0/` (copy_user_model_to_isolated)
- Isolated Cache WRITE → `tests_2.0/` (fresh state, no hooks)
- Schema/Mock only → `tests_2.0/spec/` (fast validation)

---

## ⚠️ CRITICAL: Sequential Execution Required for E2E Tests

**DO NOT use parallel execution with `-m live_e2e` tests:**

```bash
# ✅ SAFE - Sequential execution (one model at a time)
HF_HOME=/path/to/cache pytest -m live_e2e -v

# 🔥 DEADLY - Parallel execution (multiple large models simultaneously)
HF_HOME=/path/to/cache pytest -m live_e2e -n auto  # ← NEVER DO THIS!
```

**Why parallel execution is dangerous:**

| Risk | Impact | Evidence |
|------|--------|----------|
| **Multiple large models loading simultaneously** | System freeze requiring hardware reset | Experienced 2025-11-12 during development |
| **RAM budget violation** | 8 workers × 20GB models = 160GB peak RAM usage | Even 64GB M2 Max cannot handle this |
| **Metal GPU memory exhaustion** | MLX Metal cache shared across processes | Leads to GPU hang + system unresponsive |

**Architecture protections (sequential mode only):**
- ✅ **One server per test:** No parallel inference within a single test
- ✅ **Active cleanup polling:** Waits for actual process termination (not blind timeout)
- ✅ **Explicit garbage collection:** Forces Python GC + 2s Metal memory buffer
- ✅ **Conservative timeout:** 45s max wait for very large models (>40GB), but polls every 500ms
- ⚠️ **Large model transitions:** Models >20GB may have 10-15s RAM overlap during cleanup

**Safe execution guidelines:**
- Always run `pytest -m live_e2e` without `-n auto` or `-n <workers>`
- If using pytest-xdist, ensure it's NOT active for E2E tests
- Monitor system RAM during first run to understand your hardware limits
- Expected duration: ~7-10 minutes for 15 models (sequential, with cleanup)

**Note on `-n auto` (pytest-xdist):**
- `-n auto`: Spawns one worker per CPU core (e.g., 8 workers on 8-core M2)
- Each worker loads a separate model instance simultaneously
- Safe for unit tests (mocked, no real models), DEADLY for E2E tests (real models)

---

## Python Version Verification Results

**All standard tests validated on Apple Silicon with enhanced isolation**

| Python Version | Status | Tests Passing | Skipped | Notes |
|----------------|--------|---------------|---------|-------|
| 3.9.6 (macOS)  | ✅ Verified | 519/588 | 69 | Vision tests auto-skip (mlx-vlm requires 3.10+) |
| 3.10.x         | ✅ Verified | 528/588 | 60 | Full suite including vision tests |
| 3.11.x         | ✅ Verified | 528/588 | 60 | Full suite including vision tests |
| 3.12.x         | ✅ Verified | 528/588 | 60 | Full suite including vision tests |
| 3.13.x         | ✅ Verified | 528/588 | 60 | Full suite including vision tests |
| 3.14.x         | ✅ Verified | 528/588 | 60 | Full suite including vision tests |

**Note:** 60 skipped tests (69 on Python 3.9) are opt-in (live tests, alpha features). Skipped count may vary by environment:
- Without `HF_HOME`: Standard 60 skipped (69 on Py3.9, live E2E tests use fallback parametrization)
- With `HF_HOME`: Live E2E tests run with discovered models across text_portfolio (23) and vision_portfolio (3)

All versions tested with `isolated_cache` system and MLX stubs for fast execution without model downloads.

## Push Testing Details (2.0)

This section summarizes what our test suite covers for the experimental `push` feature and what still requires live/manual checks.

### Reference: Push CLI and JSON

- Usage: `mlxk2 push <local_dir> <org/model> --private [--create] [--branch main] [--commit <msg>] [--check-only] [--json] [--verbose]`
- Args:
  - `--private` (required in alpha): Safety gate to avoid public uploads.
  - `--create`: Create the repository if it does not exist (model repo).
  - `--branch`: Target branch, default `main`. Missing branches are tolerated; with `--create`, the branch is proactively created (and upload retried once if the hub initially rejects the revision).
  - `--commit`: Commit message, default `"mlx-knife push"`.
  - `--check-only`: Analyze workspace locally; no network call; returns `data.workspace_health`.
  - `--dry-run`: Compare local workspace to the remote branch and summarize changes without uploading (requires repo read access).
  - `--json`: Print JSON response; in JSON mode, logs/progress are suppressed by default.
  - `--verbose`: Human mode — append details (e.g., commit URL). In JSON mode, only toggles console log verbosity; the JSON payload is unchanged.

- JSON fields (`data`):
  - `repo_id: string` — target `org/model`.
  - `branch: string` — target branch.
  - `commit_sha: string|null` — commit id; null when `no_changes:true` or on noop.
  - `commit_url: string|null` — link to commit; null when no commit created.
  - `repo_url: string` — `https://huggingface.co/<org/model>`.
  - `uploaded_files_count: int|null` — number of changed files; set to `0` on `no_changes:true`.
  - `local_files_count: int|null` — approximate local file count scanned.
  - `no_changes: boolean` — true when hub reports an empty commit (preferred signal) or no file operations are detected.
  - `created_repo: boolean` — true when repo was created (with `--create`).
  - `change_summary: {added:int, modified:int, deleted:int}` — optional; derived from hub response when available.
  - `message: string|null` — short human hint; mirrors hub on no-op.
  - `hf_logs: string[]` — buffered hub log lines (not printed in JSON mode unless `--verbose`).
  - `experimental: true` and `disclaimer: string` — feature state markers.
  - `workspace_health: {...}` — present only with `--check-only`:
    - `healthy: bool`, `anomalies: []`, `config`, `weights.index`, `weights.pattern_complete`, etc.
  - `dry_run: true` — present only with `--dry-run`.
  - `dry_run_summary: {added:int, modified:int, deleted:int}` — present with `--dry-run`.
  - `would_create_repo: bool` / `would_create_branch: bool` — planning hints when target does not exist.

- Error types (`error.type`):
  - `dependency_missing` — `huggingface-hub` not installed.
  - `auth_error` — missing `HF_TOKEN` (unless `--check-only`).
  - `workspace_not_found` — local_dir missing/not a directory.
  - `repo_not_found` — repo missing without `--create`.
  - `upload_failed` — hub returned an error (e.g., 403/permission).
  - `push_operation_failed` — unexpected internal failure wrapper.

- Exit codes: success → `0`; any `status:error` → `1`.

### Automated (offline)

- **Token/Workspace errors:** Missing `HF_TOKEN` and missing workspace produce proper JSON errors.
- **CLI args (JSON mode):** Missing positional args emit JSON errors rather than usage text.
- **Schema shape:** Push success/error outputs validate against `docs/json-api-schema.json`.
- **No-op push:** Detects `no_changes: true`, sets `uploaded_files_count: 0`, carries hub message into JSON (`message`/`hf_logs`), and human output shows "no changes" without duplicate logs.
- **Commit path:** Extracts `commit_sha`, `commit_url`, `change_summary` (+/~/−), correct `uploaded_files_count`; human `--verbose` includes URL.
- **Repo/Branch handling:** Missing repo requires `--create`; with `--create` sets `created_repo: true`. Missing branch is tolerated; upload attempts proceed. With `--create`, the branch is proactively created and the upload is retried once if the hub rejects the revision (e.g., "Invalid rev id").
- **Ignore rules:** `.hfignore` is merged with default ignores and forwarded to the hub.

**Files:**
- `tests_2.0/test_cli_push_args.py` (CLI errors and JSON outputs)
- `tests_2.0/test_push_extended.py` (no-op vs commit, branch/repo, .hfignore, human; includes retry on invalid revision with `--create`)
- `tests_2.0/spec/test_push_output_matches_schema.py` (schema success path)

**Run (venv39):**
```bash
source venv39/bin/activate && pip install -e .
pytest -q tests_2.0/test_cli_push_args.py tests_2.0/test_push_extended.py
pytest -q tests_2.0/spec/test_push_output_matches_schema.py
pytest -q tests_2.0/test_push_extended.py::test_push_retry_creates_branch_on_upload_revision_error
```

### Live (opt-in / wet)

- Purpose: sanity-check real HF behavior (auth, no-op vs commit, URLs).
- Defaults: Live tests are skipped. Enable with env vars and markers.
- Env:
  - `MLXK2_LIVE_PUSH=1`
  - `HF_TOKEN` (write-enabled)
  - `MLXK2_LIVE_REPO='org/model'`
  - `MLXK2_LIVE_WORKSPACE='/abs/path/to/workspace'`
- Command:
  - `pytest -q -m wet tests_2.0/live/test_push_live.py`
  - or `pytest -q -m live_push`

## Pull/Preflight (Issue #30)

Goal: Gated/private/not-found repos must not pollute the cache and should fail fast.

- Behavior (2.0):
  - Preflight uses `huggingface_hub.HfApi.model_info()` (metadata only; no download).
  - Gated/Forbidden/Unauthorized/NotFound → `access_denied` before download; clear hint to set `HF_TOKEN`.
  - Network timeouts/unspecific HTTP errors in preflight → degrade to a warning; allow the download layer (to surface meaningful error/timeout paths).
  - Tokens: prefer `HF_TOKEN` (legacy `HUGGINGFACE_HUB_TOKEN` is read, but not promoted).
  - Tests use isolated caches; the user cache is never touched.

- Relevant tests: `tests_2.0/test_issue_30_preflight.py`
  - `test_preflight_private_model_without_token`
  - `test_preflight_nonexistent_model`
  - `test_preflight_integration_in_pull`
  - `test_preflight_prevents_cache_pollution`

- Quick checks:
  - `pytest -q tests_2.0/test_issue_30_preflight.py`
  - CLI: `unset HF_TOKEN HUGGINGFACE_HUB_TOKEN; mlxk-json pull meta-llama/Llama-2-7b-hf --json`

## Runner: Interruption & Recovery

- Semantics (2.0): A new generation resets `_interrupted = False` at the start (recovery behavior). A previous Ctrl-C does not block the next generation.
- Streaming:
  - During an active generation, the runner yields a line `"[Generation interrupted by user]"` and stops.
  - Token diffing in streaming is robust against minimal mocks (no StopIteration due to short `decode` sequences).
- Batch:
  - Resets the flag at the start of a new generation; filters stop tokens; chat stop tokens optional via `use_chat_stop_tokens=True`.
- Relevant tests:
  - `tests_2.0/test_ctrl_c_handling.py` (SIGINT, interruption behavior, interactive)
  - `tests_2.0/test_interruption_recovery.py` (resetting the flag for new generations)
  - `tests_2.0/test_runner_core.py` (consistency/batch/streaming, error handling)

## Server Minimal Tests

- Dependencies: `httpx`, `fastapi`, `uvicorn`, `pydantic` (via `[test]`).
- Scope: OpenAI-compatible endpoints (minimal smoke); no real models required.
- Optional for local verification; in CI currently "nice to have" (Backlog, not part of the 2.0 Guide).

## Known Warnings

- urllib3 LibreSSL notice on macOS Python 3.9
  - Message: "urllib3 v2 only supports OpenSSL 1.1.1+, currently the 'ssl' module is compiled with 'LibreSSL 2.8.3' …"
  - Status: Harmless for our usage; suppressed in production code (see `mlxk2/__init__.py`, `warnings.filterwarnings(...)`).
  - Tests: May still appear in pytest summary if third-party dependencies import `urllib3` before our package.
  - Optional suppression in tests: add to `pytest.ini`:

    ```ini
    filterwarnings =
        ignore:urllib3 v2 only supports OpenSSL 1.1.1+
    ```

## Issue #27 Tests (Real Multi-Shard Model Health)

### Quick Start (Minimal)

```bash
# Set your HF cache (external SSD recommended)
export HF_HOME=/Volumes/your-ssd/huggingface/cache

# Select a model with index file (upstream repo)
export MLXK2_ISSUE27_MODEL="mistralai/Mixtral-8x7B-Instruct-v0.1"

# Optional: Bootstrap index if not in cache
export MLXK2_BOOTSTRAP_INDEX=1

# Run tests
pytest tests_2.0/test_issue_27.py -v
```

### Purpose

These tests validate the strict health policy against real upstream Hugging Face repositories that ship multi-shard safetensors with a `model.safetensors.index.json`. They complement the deterministic unit tests by exercising real-world layouts.

### When to Run

**Run them when:**
- Your user cache contains at least one upstream PyTorch repo with a safetensors index (not MLX/GGUF conversions). Good candidates:
  - `mistralai/Mistral-7B-Instruct-v0.2` or `-v0.3`
  - `Qwen/Qwen1.5-7B-Chat`, `Qwen/Qwen2-7B-Instruct`
  - `teknium/OpenHermes-2.5-Mistral`
  - Gated: `meta-llama/Llama-2-7b-chat-hf`, `meta-llama/Llama-3-8B-Instruct`, `google/gemma-7b-it`
- You want to sanity-check index-based completeness, shard deletion/truncation, and LFS pointer detection against real artifacts.

**They are not useful when:**
- Your cache only has MLX Community models (no `model.safetensors.index.json`) or GGUF models — the index-based tests will skip by design. In that case, rely on `tests_2.0/test_health_multifile.py` for deterministic coverage.

### Environment Setup

```bash
# Set user cache (EITHER)
export MLXK2_USER_HF_HOME=/absolute/path/to/huggingface/cache
# OR
export HF_HOME=/absolute/path/to/huggingface/cache  # Test harness preserves this

# Select model with index file (recommended)
export MLXK2_ISSUE27_MODEL="mistralai/Mixtral-8x7B-Instruct-v0.1"

# Optional: Minimize copy size
export MLXK2_SUBSET_COUNT=1      # Default 1
export MLXK2_MIN_FREE_MB=512     # Default 512 MB

# Run tests
PYTHONPATH=. pytest tests_2.0/test_issue_27.py -v
```

### Optional Bootstrap (Opt-in, Minimal Workflow)

```bash
# Enable index bootstrap (fetches only index files, never modifies user cache)
export MLXK2_BOOTSTRAP_INDEX=1

# Optional: Separate model for index tests
export MLXK2_ISSUE27_INDEX_MODEL="org/model-with-index"

# Run
pytest tests_2.0/test_issue_27.py -v
```

**Note:** Network is only needed if your user cache does not already contain an index file for the chosen repo. If the index exists in your cache, the tests copy it into the isolated cache and no network is required.

### Troubleshooting

**If you see SKIPs:**
- "No safetensors index found" → The chosen model snapshot lacks an index file. Pick a model that has `model.safetensors.index.json` (or `pytorch_model.bin.index.json`).
- "Not enough free space" → Free disk space; tests create a subset copy into an isolated temp cache.
- "User model not found" → Verify your model exists in the user cache and `MLXK2_USER_HF_HOME` points to the `.../huggingface/cache` root.

**Quick helper to list index-bearing models in your user cache:**

```bash
find "$MLXK2_USER_HF_HOME/hub" -type f \
  \( -name 'model.safetensors.index.json' -o -name 'pytorch_model.bin.index.json' \) \
| sed 's#.*/hub/models--\(.*\)/snapshots/.*#\1#; s#--#/#g' | sort -u
```

### Resource Considerations

- **Disk:** Tests copy a minimal subset of files into an isolated cache (index + 1 smallest shard, or 1 Pattern-Shard).
- **Network:** If you need to fetch a candidate model first, prefer downloading only `config.json`, `model.safetensors.index.json`, and 1-2 small shards to keep it light.

### Copy-on-Write (CoW) Optimization

Test model copies use CoW on macOS/APFS for instant, disk-free clones.

**How it works:**
- Volume detection: `_get_volume_root()` finds mount point, `_is_apfs_volume()` verifies APFS
- External volumes: Creates `.mlxk2_test_isolation/` on volume root for temp dirs
- System volume: Falls back to `/var/folders` (same APFS container, CoW still works!)
- `copy_user_model_to_isolated()` uses `cp -c` (clonefile) for instant CoW copies
- On non-APFS or cross-volume scenarios, it falls back to regular `shutil.copy2()`

**Benefits:**
- Vision model tests (~24GB) complete in **< 1 second** instead of minutes
- No disk space consumed for CoW copies (blocks shared until mutation)
- `MLXK2_SUBSET_COUNT=999` now safe to use (copies all shards instantly)

**Requirements for CoW:**
- macOS with APFS filesystem
- User cache and test temp dir on the **same** volume
- If user cache is on external SSD and temp uses system disk, falls back to regular copy

### Safety Signature Mechanism

**CRITICAL:** The test infrastructure includes a safety signature mechanism to prevent accidental deletion of user data.

**How it works:**
1. `_create_isolated_temp_dir()`: **Atomically** creates temp directory + signature file
2. Signature contains: Magic string, UUID, path hash, creation timestamp
3. `_safe_rmtree()`: **REFUSES** to delete unless signature verification passes
4. If signature creation fails, the directory is immediately cleaned up

**Safety checks before deletion:**
- Signature file must exist
- Magic string must match (`MLXK2_ISOLATED_TEST_CACHE_V1`)
- UUID must match the one returned at creation
- Path hash must match current path
- Path must contain `mlxk2_test_` marker

**Why this matters:**
- Temp directories are created on the **same volume** as user cache (for CoW)
- Without this mechanism, a bug could accidentally delete user model data
- The signature verification provides multiple layers of protection

### Vision Model Health Tests (ADR-012 Phase 2)

Real vision model health validation with controlled mutations.

```bash
# Set user cache
export MLXK2_USER_HF_HOME=/path/to/huggingface/cache

# Optional: Override default vision model (11B, ~24GB)
export MLXK2_VISION_MODEL="mlx-community/Llama-3.2-11B-Vision-Instruct-4bit"

# Run vision health tests only
pytest tests_2.0/test_issue_27.py::TestIssue27Exploration::test_vision_model_missing_preprocessor_is_unhealthy -v
pytest tests_2.0/test_issue_27.py::TestIssue27Exploration::test_vision_model_invalid_preprocessor_is_unhealthy -v
pytest tests_2.0/test_issue_27.py::TestIssue27Exploration::test_vision_model_missing_tokenizer_json_is_unhealthy -v
pytest tests_2.0/test_issue_27.py::TestIssue27Exploration::test_vision_model_complete_is_healthy -v

# Or run all Issue #27 tests (including vision)
pytest tests_2.0/test_issue_27.py -m issue27 -v
```

**Vision model mutations tested:**
- `remove_preprocessor` - Deletes `preprocessor_config.json` (should be unhealthy)
- `inject_invalid_preprocessor` - Creates invalid JSON in `preprocessor_config.json` (should be unhealthy)
- `remove_tokenizer_json` - Deletes `tokenizer.json` while `tokenizer_config.json` exists (should be unhealthy)

**Default model:** `mlx-community/Llama-3.2-11B-Vision-Instruct-4bit` (~24GB)

**Requirements:**
- Vision model must exist in user cache (pull it first if needed)
- Python 3.10+ (mlx-vlm dependency - tests skip on Python 3.9)
- CoW (macOS/APFS, same volume) eliminates disk space concerns; otherwise ~24GB free space needed

## Manual MLX Chat Model Smoke Test (2.0)

Goal: Pull a small MLX chat model, verify classification, prepare a local workspace, validate it offline, and push to a private repo while preserving chat intent. This helps issuers validate iOS-focused workflows.

**Model choice (example):**
- `mlx-community/Qwen2.5-0.5B-Instruct-4bit` (small, chat-oriented)

### Steps

1. **Pull (venv39):**
   ```bash
   mlxk2 pull mlx-community/Qwen2.5-0.5B-Instruct-4bit
   ```

2. **Verify in cache:**
   ```bash
   mlxk2 list --health "Qwen2.5-0.5B-Instruct-4bit"
   # Expect: Framework MLX, Type chat, capabilities include chat
   ```

3. **Prepare local workspace from cache (dereference symlinks):**
   ```bash
   # Ensure HF_HOME points to your HF cache
   # Compute cache path: $HF_HOME/models--mlx-community--Qwen2.5-0.5B-Instruct-4bit
   # Find latest snapshot hash under snapshots/
   # Copy to workspace and dereference symlinks:
   rsync -aL "$HF_HOME/models--mlx-community--Qwen2.5-0.5B-Instruct-4bit/snapshots/<HASH>/" ./mymodel_test_workspace/
   ```

4. **Recommended README front-matter (to preserve intent on push):**
   - Include YAML with tags and pipeline tag, e.g.:
     - `tags: [mlx, chat]`
     - `pipeline_tag: text-generation`
     - `base_model: <upstream_base>`
   - Keep model name containing `Instruct` or `chat` to aid chat detection

5. **Offline validation (no network):**
   ```bash
   mlxk2 push --check-only ./mymodel_test_workspace <org/model> --json
   # Expect: workspace_health.healthy: true
   ```

6. **Push to private repo:**
   ```bash
   mlxk2 push --private --create ./mymodel_test_workspace <org/model> --json
   # Re-push without changes should show no_changes: true
   ```

7. **Post-push verification:**
   ```bash
   mlxk2 list --all --health <org/model>
   # Current limitation: Framework may show PyTorch for non-mlx-community orgs
   # This does not affect content; future M1 will parse model card tags (mlx)
   ```

## Real-Model Testing (Implemented in 2.0.1)

**Status:** ✅ Live in 2.0.1 (Portfolio Discovery, ADR-009)

### Portfolio Discovery

Auto-discovers and tests all MLX chat models in user cache.

**Location:** `test_stop_tokens_live.py` (Category 2: Live Tests)
**Marker:** `live_stop_tokens`

**Usage:**
```bash
# With HF_HOME: Auto-discovers all MLX chat models
export HF_HOME=/path/to/cache
pytest -m live_stop_tokens -v

# Without HF_HOME: Uses 3 predefined models (must exist in cache)
pytest -m live_stop_tokens -v  # → Runs if models present, else fails
```

**Features:**
- ✅ **Model Filtering:** MLX + healthy + runtime_compatible + chat only
- ✅ **Portfolio Discovery:** Uses `mlxk list --json` to discover all qualifying models (refactored: production command, ~70 LOC eliminated)
- ✅ **RAM-Aware:** Progressive budgets prevent OOM (40%-70% of system RAM)
- ✅ **Empirical Report:** Generates `stop_token_config_report.json` with findings
- ✅ **Fallback:** Uses predefined models when no qualifying models discovered (regardless of HF_HOME setting)

**Required Models for Live Tests:**

Live tests use **either** Portfolio Discovery **or** these fallback models:

| Scenario | Models tested |
|----------|---------------|
| Portfolio Discovery finds models | Only discovered models (dynamic) |
| Portfolio Discovery finds nothing | Only fallback models (this list) |

**Fallback models** (only needed when Discovery finds nothing — any qualifying MLX model in cache replaces these):

| Type | Model | RAM | Fallback for |
|------|-------|-----|--------------|
| Text | `mlx-community/gpt-oss-20b-MXFP4-Q8` | ~12 GB | Text tests |
| Text | `mlx-community/Qwen2.5-0.5B-Instruct-4bit` | ~1 GB | Text tests |
| Text | `mlx-community/Llama-3.2-3B-Instruct-4bit` | ~4 GB | Text tests |
| Vision | `mlx-community/pixtral-12b-4bit` | ~7 GB | Vision tests (or any vision model) |
| Audio | `mlx-community/whisper-large-v3-turbo-4bit` | ~1.5 GB | Audio tests (or any audio model) |

```bash
# Pull all minimum required models (~25 GB total)
mlxk pull mlx-community/gpt-oss-20b-MXFP4-Q8
mlxk pull mlx-community/Qwen2.5-0.5B-Instruct-4bit
mlxk pull mlx-community/Llama-3.2-3B-Instruct-4bit
mlxk pull mlx-community/pixtral-12b-4bit
mlxk pull mlx-community/whisper-large-v3-turbo-4bit
```

**Note:** These models are defined in `tests_2.0/live/test_utils.py` (`TEST_MODELS`, `VISION_TEST_MODELS`, `AUDIO_TEST_MODELS`) and `tests_2.0/test_stop_tokens_live.py` (`TEST_MODELS`).

### E2E Tests with Portfolio Separation (ADR-011 + Portfolio Separation)

**Status:** ✅ Working (Portfolio Separation complete)

Auto-discovers and validates Server/HTTP/CLI interfaces with real models, separated into text and vision portfolios.

**Location:** `tests_2.0/live/` (test_server_e2e.py, test_vision_server_e2e.py, test_cli_e2e.py, test_streaming_parity.py)
**Marker:** `live_e2e`

**Usage:**
```bash
# With HF_HOME: Auto-discovers all MLX chat models (separated into text/vision)
export HF_HOME=/path/to/cache
pytest -m live_e2e -v

# See which TEXT models will be tested
pytest tests_2.0/live/test_server_e2e.py::TestChatCompletionsBatch --collect-only -q

# See which VISION models will be tested
pytest tests_2.0/live/test_vision_server_e2e.py::TestVisionServerE2E --collect-only -q

# Show portfolios before running tests
HF_HOME=/path/to/cache python tests_2.0/show_portfolios.py

# ⚠️ IMPORTANT: Always test collection before release
pytest -m live_e2e --collect-only  # Should work without errors
```

**Architecture:**
- ✅ **Separate Portfolio Discovery:** `discover_text_models()` and `discover_vision_models()` filter by capability
- ✅ **Production Command:** Uses `mlxk list --json` instead of duplicating cache logic (~70 LOC eliminated)
- ✅ **Parametrized Tests:** text_XX (23 text models), vision_XX (3 vision models) - deterministic indices
- ✅ **Independent RAM Formulas:** Text uses 1.2x multiplier, Vision uses 0.70 threshold (ADR-016)
- ✅ **Clean Lifecycle:** Each test gets its own server instance (45s timeout for MLX cleanup)
- ✅ **Disjoint Portfolios:** No model appears in both text and vision portfolios
- ✅ **Current result:** 136/136 tests passing (23 text + 3 vision models, deterministic discovered_XX replaced)

**Tests Covered:**
- **Text Portfolio:** Server health/metadata, chat completions (batch/streaming), text completions, CLI run, streaming parity, stop tokens
- **Vision Portfolio:** Multimodal chat (Base64 images), SSE graceful degradation, text-only on vision models, Vision→Text switching

### RAM-Aware Model Selection (Portfolio Separation)

**Implementation:** `calculate_text_model_ram_gb()`, `calculate_vision_model_ram_gb()`, `get_safe_ram_budget_gb()`, `should_skip_model()`

**Text Model RAM Calculation:**
- Formula: `size_bytes * 1.2` (accounts for KV cache + inference overhead)
- Progressive budgets apply (40%-70% based on system RAM)

**Vision Model RAM Calculation:**
- Formula: `size_bytes` (no multiplier) with 0.70 threshold gate
- Vision Encoder overhead: Crashes above 70% system RAM (ADR-016)
- Models >70% → `ram_needed_gb = float('inf')` (auto-skip)

**Progressive RAM Budgets (Text Models):**

| System RAM | Budget | Available for Models |
|------------|--------|---------------------|
| 16GB | 40% | 6.4GB |
| 32GB | 50% | 16GB |
| 64GB | 60% | 38.4GB |
| 96GB+ | 70% | 67GB+ |

**Vision Model Threshold (0.70 = 70%):**

| System RAM | 70% Threshold | Example Vision Model |
|------------|---------------|----------------------|
| 64GB | 44.8GB | Llama-3.2-11B-Vision (5.6GB) ✅ RUN |
| 64GB | 44.8GB | Llama-3.2-90B-Vision (46.4GB) ⏭️ SKIP |

**Rationale:**
- Text: OS overhead is ~4-6GB (constant), larger systems have more headroom
- Vision: Vision Encoder overhead is unpredictable, conservative 70% gate prevents OOM crashes

**Behavior:**
- Models exceeding budget/threshold → Auto-skipped
- Skip reason: "Model requires XGB but only YGB available" or "Vision model exceeds 70% threshold"
- Empirical report tracks skipped models

**Example (64GB system):**
```python
# TEXT MODELS (1.2x multiplier, 60% budget = 38.4GB)
# Qwen-0.5B (0.6GB RAM) → ✅ RUN
# Llama-3.2-3B (4.8GB RAM) → ✅ RUN
# Mistral-7B (10GB RAM) → ✅ RUN
# Mixtral-8x7B (38.4GB RAM) → ✅ RUN (exactly at budget)

# VISION MODELS (no multiplier, 70% threshold = 44.8GB)
# Llama-3.2-11B-Vision (5.6GB, 8.75% ratio) → ✅ RUN
# pixtral-12b (12.6GB, 19.6% ratio) → ✅ RUN
# Llama-3.2-90B-Vision (46.4GB, 72.5% ratio) → ⏭️ SKIP (exceeds 70%)
```

### max_tokens Strategy: Vision vs Text

**Problem:** Vision and text models have fundamentally different context management strategies.

**Text Models (MLXRunner):**
- **Shift-Window Context:** Maintain conversation history in context buffer
- **Server Default:** `context_length / 2` (reserve half for history, half for generation)
- **CLI Default:** `context_length` (full context, no reservation)
- **Example:** Llama-3.2-3B (128K context) → Server: 64K max_tokens
- **Implementation:** `get_effective_max_tokens(runner, requested_max_tokens, server_mode)`

**Vision Models (VisionRunner):**
- **Stateless Processing:** Each request is independent (Metal memory limitations prevent context preservation)
- **No Shift-Window:** History not maintained in model context
- **Server/CLI Default:** `2048` tokens (conservative, works for all vision models)
- **Rationale:**
  - No need for `/2` division (no history to reserve)
  - Vision inference is slow → 2048 adequate for image descriptions
  - Prevents accidentally generating 64K+ tokens on large-context models
- **Example:** Llama-3.2-11B-Vision (128K context) → Default: 2048 max_tokens
- **Implementation:** `get_effective_max_tokens_vision(runner, requested_max_tokens)`

**Batch Processing:**
- Vision: Processes multiple images → Batched stateless (each image independent)
- Text: Receives ALL vision outputs → Full shift-window context for complex queries
- Example: "Compare Image 1 and Image 15" requires text model with full history

### Text Portfolio E2E Tests

**Status:** ✅ Complete (Portfolio Separation)

**Location:** `tests_2.0/live/test_server_e2e.py`
**Fixture:** `text_portfolio` (provides text-only models)
**Parametrization:** `text_model_key` (text_00, text_01, ..., text_22)

**Test Classes:**
1. **TestServerHealthEndpoints** - Basic server functionality
   - `test_health_endpoint` - Server liveness check
   - `test_v1_models_list` - Model metadata endpoint

2. **TestChatCompletionsBatch** - Non-streaming chat (parametrized)
   - `test_chat_completions_batch[text_XX]` - OpenAI-compatible batch responses
   - Validates: Response structure, stop token filtering (Issue #32)
   - **23 tests** (one per text model in portfolio)

3. **TestChatCompletionsStreaming** - SSE streaming chat (parametrized)
   - `test_chat_completions_streaming[text_XX]` - Server-Sent Events format
   - Validates: SSE format, chunk structure, stop token filtering, completion
   - **23 tests** (one per text model in portfolio)

4. **TestCompletionsBatch** - Non-streaming text completion
   - `test_completions_batch_basic` - Basic `/v1/completions` endpoint

5. **TestCompletionsStreaming** - SSE streaming text completion
   - `test_completions_streaming_basic` - Streaming text completions

**RAM Gating:**
- Uses `calculate_text_model_ram_gb()` (1.2x multiplier)
- Progressive budgets: 40%-70% based on system RAM
- Models exceeding budget auto-skipped with clear reason

**Example:**
```python
# 64GB system → 38.4GB budget (60%)
# text_00: Qwen-0.5B (0.6GB) → ✅ RUN
# text_10: Mistral-7B (10GB) → ✅ RUN
# text_22: Mixtral-8x7B (38.4GB) → ✅ RUN (at budget limit)
```

### Vision Portfolio E2E Tests

**Status:** ✅ Complete (Portfolio Separation)

**Location:** `tests_2.0/live/test_vision_server_e2e.py`
**Fixture:** `vision_portfolio` (provides vision-capable models)
**Parametrization:** `vision_model_key` (vision_00, vision_01, vision_02)

**Test Class: TestVisionServerE2E**

1. **test_single_image_chat_completion[vision_XX]** (parametrized)
   - Multimodal chat with Base64 image data
   - Validates: Vision model describes image correctly
   - OpenAI Vision API format: `content: [{"type": "text"}, {"type": "image_url"}]`
   - **3 tests** (one per vision model in portfolio)

2. **test_streaming_graceful_degradation[vision_XX]** (parametrized)
   - Vision request with `stream=True`
   - Validates: SSE emulation (mlx-vlm doesn't support true streaming)
   - Returns HTTP 200 with SSE events (not HTTP 400 rejection)
   - **3 tests** (one per vision model in portfolio)

3. **test_text_request_still_works_on_vision_model[vision_XX]** (parametrized)
   - Text-only request on vision model server
   - Validates: Vision models handle pure text requests
   - No image data, just string content
   - **3 tests** (one per vision model in portfolio)

4. **test_vision_to_text_model_switch_filters_images** (special integration test)
   - Tests Vision→Text model switching with conversation history
   - Server filters `image_url` content for text models
   - Validates: Multimodal history filtering
   - **1 test** (uses both portfolios)

**RAM Gating:**
- Uses `calculate_vision_model_ram_gb()` (0.70 threshold, no multiplier)
- Models >70% system RAM → `ram_needed_gb = float('inf')` (auto-skip)
- Conservative gate prevents Vision Encoder OOM crashes

**Example:**
```python
# 64GB system → 44.8GB threshold (70%)
# vision_00: Llama-3.2-11B-Vision (5.6GB, 8.75%) → ✅ RUN
# vision_01: pixtral-12b (12.6GB, 19.6%) → ✅ RUN
# vision_02: Llama-3.2-90B-Vision (46.4GB, 72.5%) → ⏭️ SKIP (exceeds 70%)
```

**Why Separate Portfolios:**
- Text and Vision models have different RAM characteristics
- Vision models crash unpredictably above 70% (Vision Encoder overhead)
- Deterministic test indices (text_XX, vision_XX) replace flaky discovered_XX
- Tests validate model-type-specific behavior (e.g., text-only on vision models)

### Audio Portfolio E2E Tests

**Status:** ✅ Complete (ADR-020, Portfolio Separation)

**Location:** `tests_2.0/live/test_audio_e2e_live.py`
**Fixture:** `audio_portfolio` (provides audio-capable models)
**Parametrization:** `audio_model_key` (audio_00, audio_01, ...)

**Test Class: TestAudioTranscription**

1. **test_transcribe_short_audio_wav[audio_XX]** (parametrized)
   - Short audio transcription (~3 seconds)
   - Validates: Key semantic content (universe, sir, exist)
   - **N tests** (one per audio model in portfolio)

2. **test_transcribe_longer_audio_wav[audio_XX]** (parametrized)
   - Longer audio transcription (~14 seconds)
   - Validates: At least 2 keywords from passage (royal, cavern, throne, crown, gong)
   - **N tests** (one per audio model in portfolio)

3. **test_transcribe_mp3_format[audio_XX]** (parametrized)
   - MP3 format support validation
   - Validates: Key semantic content (universe, sir, exist) - same as WAV test
   - **N tests** (one per audio model in portfolio)

4. **test_audio_output_not_empty[audio_XX]** (parametrized)
   - Basic sanity test
   - Validates: Output length > 10 characters
   - **N tests** (one per audio model in portfolio)

**Test Class: TestAudioSegments**

5. **test_segment_metadata_optional[audio_XX]** (parametrized)
   - Validates: No segment metadata without `MLXK2_AUDIO_SEGMENTS=1`
   - **N tests** (one per audio model in portfolio)

**Test Class: TestAudioTranscriptionsServer** (Server `/v1/audio/transcriptions` endpoint)

6. **test_transcription_endpoint_json[audio_XX]** (parametrized)
   - JSON response format validation
   - **N tests** (one per audio model in portfolio)

7. **test_transcription_endpoint_text_format[audio_XX]** (parametrized)
   - Plain text response format validation
   - **N tests** (one per audio model in portfolio)

8. **test_transcription_endpoint_verbose_json[audio_XX]** (parametrized)
   - Verbose JSON with task/duration fields
   - **N tests** (one per audio model in portfolio)

9. **test_transcription_endpoint_mp3[audio_XX]** (parametrized)
   - MP3 format support via server endpoint
   - **N tests** (one per audio model in portfolio)

10. **test_transcription_endpoint_with_language[audio_XX]** (parametrized)
    - Explicit language parameter (`language: "en"`)
    - **N tests** (one per audio model in portfolio)

11. **test_transcription_endpoint_rejects_oversized_audio[audio_XX]** (parametrized)
    - Validates: HTTP 413 for files > 50 MB (MAX_AUDIO_SIZE_BYTES)
    - Prevents resource exhaustion from large uploads
    - **N tests** (one per audio model in portfolio)

**RAM Gating:**
- Uses AudioRunner with Memory Gate (4 GB threshold)
- Whisper models: ~0.4 GB model + ~17 GB runtime (Audio-Decoder, Mel-Spectrogram, librosa)

**Server Security:**
- `/v1/audio/transcriptions` enforces 50 MB upload limit (`MAX_AUDIO_SIZE_BYTES`)
- Returns HTTP 413 for oversized files

**Known Limitations (ADR-020):**
- Upload limit: 50 MB (server endpoint)
- CLI `run`: No file size limit (local files)
- MP3/M4A recommended for long audio (10:1 compression vs WAV)
- **Gemma-3n (mlx-vlm):** ~30 seconds max (multimodal architecture constraint, ADR-019/beta.8)

**Example:**
```python
# 64GB system with whisper-large-v3-turbo-4bit
# Model: 0.4 GB, Runtime: ~17 GB → ✅ RUN
```

---

## Appendix

### A1. Portfolio Discovery Fixture Compatibility

**Implementation:** `tests_2.0/conftest.py`

The `pytest_collection_modifyitems` hook auto-assigns the `wet` marker based on test markers and location:

```python
# Declarative compatibility set
LIVE_MARKERS_FOR_WET = {
    # Portfolio Discovery users
    "live_e2e",           # Uses Portfolio fixtures
    "live_stop_tokens",   # Uses Portfolio fixtures

    # User Cache READ
    "live_run",           # User Cache READ only
    "live_list",          # User Cache READ only
    "issue27",            # User Cache READ + copy

    # Workspace operations
    "live_push",          # Workspace only (tmp_path)
}

def pytest_collection_modifyitems(config, items):
    """Auto-assign wet marker based on compatibility."""
    for item in items:
        test_markers = {m.name for m in item.iter_markers()}
        test_path = str(item.path)
        is_in_live_dir = "/live/" in test_path or "\\live\\" in test_path

        # Wet marker for compatible tests
        if (test_markers & LIVE_MARKERS_FOR_WET) or is_in_live_dir:
            # EXCLUDE Isolated Cache WRITE tests (incompatible)
            if "live_pull" not in test_markers and "live_clone" not in test_markers:
                item.add_marker(pytest.mark.wet)
```

### A2. Why sys.modules Pollution Matters

**pytest runs in single Python process:**
- All tests share: `sys.modules`, `sys.path`
- Module-scoped fixtures remain active across tests
- Import pollution persists until pytest exit

**Portfolio Discovery side-effects:**
```python
# live/conftest.py:vision_portfolio (module scope)
def vision_portfolio():
    discover_vision_models()  # Runs: mlxk list --json --vision
    # Subprocess imports pollute pytest process sys.modules
```

**Why live_pull breaks:**
```python
# test_resumable_pull.py:190
pull_operation(model, force_resume=True)  # In pytest process
# Imports huggingface_hub → finds polluted version in sys.modules
# Symlink creation fails
```

**Evidence:**
- Solo run: ✅ No Portfolio Discovery → clean imports → works
- Multi-test run: ❌ Portfolio Discovery → polluted imports → fails

### A3. Why Not Function Scope for Portfolio?

**Current (module scope):**
- Portfolio Discovery runs 1x per module
- 26 tests → 2x subprocess (text + vision) → fast

**Alternative (function scope):**
- Portfolio Discovery per test
- 26 tests → 52x subprocess → 10-20x slower
- Subprocess overhead: Model enumeration + RAM calculation

**Decision:** Module scope is performance optimization, not bug.

---

### A4. Test Environment Variables Reference

This section documents environment variables used exclusively for testing and development. For user-facing configuration, see the "Configuration Reference" section in [README.md](README.md).

#### A4.1 Test Control Variables

These variables control test behavior and should only be set when running the test suite:

| Variable | Description | Usage |
|----------|-------------|-------|
| `MLXK2_DEBUG` | Enable debug logging in server internals | Set to `1` for verbose debugging during test failures |
| `MLXK2_STRICT_TEST_DELETE` | Enforce strict deletion checks in `rm` tests | Set to `1` in test suite to validate error handling |
| `HF_HUB_DISABLE_PROGRESS_BARS` | Disable HuggingFace download progress bars | Set automatically by MLX Knife; don't set manually |

#### A4.2 Live Test Environment Variables

These variables enable optional live tests that interact with real models or external services:

| Variable | Description | Required For |
|----------|-------------|--------------|
| `MLXK2_LIVE_PUSH` | Enable live push tests (requires HF credentials) | `pytest -m live_push` |
| `MLXK2_LIVE_REPO` | Target repository for push tests | `pytest -m live_push` |
| `MLXK2_LIVE_WORKSPACE` | Workspace path for push tests | `pytest -m live_push` |
| `MLXK2_LIVE_CLONE` | Enable live clone tests | `pytest -m live_clone` |
| `MLXK2_LIVE_CLONE_MODEL` | Model to clone in live tests | `pytest -m live_clone` |
| `MLXK2_LIVE_CLONE_WORKSPACE` | Clone destination path | `pytest -m live_clone` |
| `MLXK2_USER_HF_HOME` | User cache path for read-only tests | `pytest -m issue27` or `pytest -m live_run` |
| `MLXK2_ISSUE27_MODEL` | Specific model for Issue #27 tests | `pytest -m issue27` |
| `MLXK2_ISSUE27_INDEX_MODEL` | Index-based model for Issue #27 | `pytest -m issue27` |
| `MLXK2_SUBSET_COUNT` | Limit Issue #27 test count | `pytest -m issue27` |
| `MLXK2_BOOTSTRAP_INDEX` | Auto-download model for Issue #27 | `pytest -m issue27` |
| `MLXK2_TEST_RESUMABLE_DOWNLOAD` | Enable resumable pull tests (requires network) | `pytest -m live_pull tests_2.0/test_resumable_pull.py` |

**Example:**
```bash
# Enable debug logging for troubleshooting
MLXK2_DEBUG=1 pytest tests_2.0/test_server_base.py -v

# Run live push tests with credentials
MLXK2_LIVE_PUSH=1 \
  MLXK2_LIVE_REPO="test-org/test-model" \
  MLXK2_LIVE_WORKSPACE="/tmp/workspace" \
  HF_TOKEN=hf_... \
  pytest -m live_push -v
```

**Important:** Never set these variables in production environments or user-facing documentation. They are intended exclusively for the test suite.

---

### A5. Complete Test File Structure (2.0.4-beta.10)

```
scripts/
└── test-wet-umbrella.sh           # Single entry point for all real tests (wet + resumable), memory-optimized (--tb=no --capture=sys)

tests_2.0/
├── __init__.py
├── conftest.py                        # Isolated test cache (HF_HOME override), safety sentinel, core fixtures, wet marker hook, memory cleanup (live_e2e+wet), pytest_addoption (--report-output)
├── conftest_runner.py                 # Runner-specific fixtures/mocks
├── show_portfolios.py                 # Diagnostic tool: Display text/vision portfolios with RAM estimates
├── stubs/                             # Minimal mlx/mlx_lm/mlx_vlm stubs for unit/spec tests
│   ├── mlx/
│   │   └── core.py
│   ├── mlx_lm/
│   │   ├── __init__.py
│   │   ├── generate.py
│   │   └── sample_utils.py
│   └── mlx_vlm/
│       └── __init__.py               # Vision stub (load, generate)
├── spec/                              # JSON API spec/contract validation
│   ├── test_cli_commands_json_flag.py         # CLI JSON flag behavior
│   ├── test_cli_version_output.py             # Version command JSON shape
│   ├── test_code_outputs_validate_against_schema.py  # Code outputs validate against schema
│   ├── test_push_error_matches_schema.py      # Push error output matches schema
│   ├── test_push_output_matches_schema.py     # Push success output matches schema
│   ├── test_spec_doc_examples_validate.py     # Docs examples validate against JSON schema
│   └── test_spec_version_sync.py              # Code/docs version consistency check
├── live/                              # Opt-in live tests (markers)
│   ├── __init__.py
│   ├── conftest.py                              # Shared fixtures for live E2E tests (text_portfolio, vision_portfolio, audio_portfolio, pytest_generate_tests hook)
│   ├── server_context.py                       # LocalServer context manager for E2E testing (45s timeout for MLX cleanup)
│   ├── sse_parser.py                           # SSE parsing utilities for streaming validation
│   ├── test_utils.py                           # Portfolio Discovery (text/vision/audio separation), RAM calculation modularization, RAM gating utilities
│   ├── test_audio_e2e_live.py                  # Audio E2E tests with Whisper models (ADR-020: CLI + Server transcriptions + size limit, parametrized: audio_XX)
│   ├── test_cli_e2e.py                         # CLI integration E2E tests (ADR-011, parametrized)
│   ├── test_cli_pipe_live.py                   # Pipe-mode E2E (stdin '-', JSON interactive error, list→run pipe) using first eligible model
│   ├── test_clone_live.py                      # Live clone flow (requires MLXK2_LIVE_CLONE, HF_TOKEN)
│   ├── test_list_human_live.py                 # Live list/health against user cache (requires HF_HOME)
│   ├── test_pipe_vision_geo.py                 # Vision→Geo pipe integration tests (marker: live_vision_pipe: batch processing, complete pipe, chunk isolation)
│   ├── test_portfolio_fixtures.py              # Portfolio separation validation tests (fixture behavior, disjoint check)
│   ├── test_push_live.py                       # Live push flow (requires MLXK2_LIVE_PUSH, HF_TOKEN)
│   ├── test_server_e2e.py                      # Server E2E tests with TEXT models (ADR-011 + Portfolio Separation, parametrized: text_XX)
│   ├── test_show_portfolio.py                  # Portfolio display (marker: show_model_portfolio, requires HF_HOME)
│   ├── test_streaming_parity.py                # Streaming vs batch parity tests (Issue #20, ADR-011, parametrized)
│   ├── test_vision_e2e_live.py                 # Vision CLI E2E tests with real models (ADR-012, 5 deterministic vision queries)
│   ├── test_vision_server_e2e.py               # Vision Server E2E tests with VISION models (ADR-012 Phase 3 + Portfolio Separation, parametrized: vision_XX)
│   └── test_vm_stat_parsing.py                 # vm_stat output parsing validation (macOS memory metrics)
├── test_adr004_error_logging.py       # ADR-004 error logging and redaction (tokens, paths)
├── test_audio_cli.py                  # Audio CLI argument tests (ADR-020 Phase 2: --audio parsing, file validation, capability checks, backend detection)
├── test_capabilities.py               # Probe/Policy architecture (ADR-012, ADR-016)
├── test_cli_log_json_flag.py          # CLI --log-json flag behavior and JSON log format
├── test_cli_push_args.py              # Push CLI args and JSON error/output handling (offline)
├── test_cli_run_exit_codes.py         # CLI exit codes + pipe/JSON regressions, stdin '-', non-TTY batch, interactive JSON error, SIGPIPE, BrokenPipeError
├── test_cli_run_wrapper.py            # mlx-run wrapper argv injection
├── test_clone_operation.py            # Clone operations with APFS optimization
├── test_ctrl_c_handling.py            # SIGINT handling during run/interactive flows
├── test_detection_readme_tokenizer.py # README/tokenizer-based framework detection
├── test_edge_cases_adr002.py          # Naming/health edge cases (ADR-002)
├── test_health_multifile.py           # Multi-file health completeness (index vs pattern)
├── test_health_vision.py              # Vision model health checks (ADR-012 Phase 2, preprocessor_config.json validation)
├── test_human_output.py               # Human rendering of list/health views
├── test_integration.py                # Model resolution and health integration
├── test_interactive_mode.py           # Interactive CLI mode prompts/history/streaming
├── test_interruption_recovery.py      # Recovery semantics after interruption (flag reset)
├── test_issue_27.py                   # Health policy exploration with real models (marker: issue27)
├── test_issue_30_preflight.py         # Preflight for gated/private/not-found repos (Issue #30)
├── test_issue_37_private_org_regression.py  # Issue #37 private/org MLX model detection (marker: live_run)
├── test_json_api_list.py              # JSON API list contract (shape/fields)
├── test_json_api_show.py              # JSON API show contract (base/files/config)
├── test_legacy_formats.py             # Legacy model format detection (Issue #37)
├── test_model_naming.py               # Conversion rules, bijection, parsing
├── test_model_resolution_workspace.py # Workspace path resolution tests (ADR-018, explicit path detection, prefix matching)
├── test_multimodal_filtering.py       # Multimodal history filtering (Vision→Text model switching)
├── test_portfolio_discovery.py        # Portfolio separation discovery tests (text/vision filtering, RAM formulas)
├── test_push_dry_run.py               # Push dry-run diff planning (added/modified/deleted)
├── test_push_extended.py              # Extended push: no-op vs commit, branch/retry, .hfignore
├── test_push_minimal.py               # Minimal push scenarios (offline)
├── test_push_workspace_check.py       # Push check-only: workspace validation without network
├── test_ram_calculation.py            # RAM calculation unit tests (text 1.2x, vision 0.70 threshold, system memory)
├── test_resumable_pull.py             # Resumable download tests (real network download with controlled interruption)
├── test_robustness.py                 # Robustness for rm/pull/disk/timeout/concurrency
├── test_run_complete.py               # End-to-end run command (stream/batch/params)
├── test_run_vision.py                 # Vision runner unit tests (ADR-012 Phase 1b, VisionRunner routing, default prompt)
├── test_runner_core.py                # MLXRunner core generation/memory/stop tokens
├── test_runtime_compatibility_reason_chain.py  # Runtime compatibility reason field decision chain (Issue #36)
├── test_server_api_minimal.py         # Minimal OpenAI-compatible server endpoints (SSE, JSON)
├── test_server_api.py.disabled        # Disabled server API tests (WIP/expanded scenarios)
├── test_server_audio.py               # Audio server unit tests (ADR-020 Phase 4: request detection, Base64 decoding, format validation)
├── test_server_models_and_errors.py   # Server model loading and error handling
├── test_server_streaming_minimal.py   # Server SSE streaming functionality
├── test_server_token_limits_api.py    # Server token limit enforcement
├── test_server_vision.py              # Vision server unit tests (ADR-012 Phase 3: ChatMessage, image detection, helpers)
├── test_stop_tokens_live.py           # Stop token validation with real models (marker: live_stop_tokens, ADR-009)
├── test_token_limits.py               # Dynamic token calculation; server vs run policies
├── test_vision_adapter.py             # Vision HTTP adapter unit tests (Base64 decoding, OpenAI format parsing, sequential images, image ID persistence)
├── test_vision_chunk_streaming.py     # Vision chunk streaming tests (SSE format, multi-chunk streaming, single-chunk routing, generator integration)
├── test_vision_exif.py                # EXIF extraction tests (GPS, DateTime, Camera, collapsible table, privacy controls)
├── test_workspace_sentinel.py         # Workspace infrastructure tests (ADR-018 Phase 0a: sentinel primitives, atomic write, managed/unmanaged detection, health checks, CLI integration)
└── test_convert_repair_index.py       # Convert operation tests (ADR-018 Phase 1: rebuild_safetensors_index, cache sanctity, workspace sentinels, validation)
```

---

## Known Model Quality Issues

Models with documented quality issues discovered during testing. Tests **will fail** when these issues occur (no workarounds).

### Multiple EOS Token Generation

| Model | Token IDs | Status | Evidence Date |
|-------|-----------|--------|---------------|
| Phi-3-mini-4k-instruct-4bit | 32007=`<\|end\|>`, 32000=`<\|endoftext\|>` | Fixed 2.0.2 | 2025-11-13 |

**Issue:** Model generates multiple EOS tokens instead of stopping at first.
**Detection:** Use `mlxk run --verbose` to see token generation details.
**Fix:** MLX-Knife 2.0.2+ filters by earliest position in text (not list order).

**Example usage:**
```bash
# Manual debugging with verbose mode
mlxk run mlx-community/Phi-3-mini-4k-instruct-4bit "Write one sentence about cats." --verbose

# Look for:
# [DEBUG] Token generation analysis:
# [DEBUG]   Last 3 tokens: ["29889='.'", "32007='<|end|>'", "32000='<|endoftext|>'"]
# [DEBUG]   ⚠️ WARNING: Multiple EOS tokens detected (2) - model quality issue
```

---

*MLX-Knife 2.0 Testing Details*
