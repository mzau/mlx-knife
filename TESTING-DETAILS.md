# MLX Knife Testing - Detailed Documentation

This document contains version-specific details, complete file listings, and implementation specifics for the MLX Knife test suite. For timeless testing philosophy and quick start instructions, see [TESTING.md](TESTING.md).

## Current Status

✅ **308/308 unit tests passing** (November 2025) — 2.0.3 Stable; 35 skipped (opt-in)
✅ **73/81 E2E tests passing** (November 2025) — ADR-011 completed; 8 skipped (RAM budget)
✅ **Test environment:** macOS 14.x, M2 Max, Python 3.9-3.13
✅ **Production verified & reported:** M1, M1 Max, M2 Max in real-world use
✅ **License:** Apache 2.0 (was MIT in 1.x)
✅ **Isolated test system** - user cache stays pristine with temp cache isolation
✅ **3-category test strategy** - optimized for performance and safety

### Skipped Tests Breakdown (35 total, standard run without HF_HOME)
- **20 Live E2E tests** - Server/HTTP/CLI validation with real models (requires `pytest -m live_e2e`, ADR-011)
- **4 Live Stop Tokens tests** - Stop token validation with real models (requires `pytest -m live_stop_tokens`, ADR-009)
- **3 Live Clone tests** - APFS same-volume clone workflow (requires `MLXK2_LIVE_CLONE=1`)
- **2 Issue #37 tests** - Private/org model detection (requires `pytest -m live_run`, Issue #37)
- **2 Runtime Compatibility tests** - Reason chain validation (requires specific model types)
- **1 Live List test** - Tests against user cache (requires HF_HOME with models)
- **1 Live Push test** - Real HuggingFace push (requires `MLXK2_LIVE_PUSH=1`)
- **1 Show Portfolio test** - Convenience test to display E2E test models (requires HF_HOME)
- **7 Issue #27 tests** - Real-model health validation (requires HF_HOME or MLXK2_USER_HF_HOME setup)

**Portfolio Discovery** (ADR-009) is implemented in `tests_2.0/test_stop_tokens_live.py`. When `HF_HOME` is set, tests auto-discover all MLX chat models in user cache using `mlxk list --json` (production command). This ensures Issue #32 fix is validated across the full model portfolio. **Current validation:** 17 models discovered, 15 testable (60% RAM budget), 73/81 tests passing, 0 failures. Portfolio includes: Phi-3, DeepSeek-R1, GPT-oss, Llama, Qwen, Mistral, Mixtral families.

For complete test file structure, see [Appendix](#complete-test-file-structure-201).

---

## Test Execution Guide

| Target | How to Run | Markers / Env | Includes | Network |
|---|---|---|---|---|
| Default 2.0 suite | `pytest -v` | — | JSON-API (list/show/health), Human-Output, Model-Resolution, Health-Policy, Push Offline (`--check-only`, `--dry-run`), Spec/Schema checks | No |
| Spec-only | `pytest -m spec -v` | `spec` | Schema/contract tests, version sync, docs example validation | No |
| Exclude Spec | `pytest -m "not spec" -v` | `not spec` | Everything except spec/schema checks | No |
| Push offline | `pytest -k push -v` | — | Push offline tests (tests alpha feature: `--check-only`, `--dry-run`, error handling); no network, no credentials needed | No |
| ⏭️ Live Push | `MLXK2_ENABLE_ALPHA_FEATURES=1 pytest -m live_push -v` | `live_push` (subset of `wet`) + Env: `MLXK2_ENABLE_ALPHA_FEATURES=1`, `MLXK2_LIVE_PUSH=1`, `HF_TOKEN`, `MLXK2_LIVE_REPO`, `MLXK2_LIVE_WORKSPACE` | JSON push against the real Hub; on errors the test SKIPs (diagnostic) | Yes |
| ⏭️ Live List | `pytest -m live_list -v` | `live_list` (subset of `wet`) + Env: `HF_HOME` (user cache with models) | Tests list/health against user cache models | No (uses local cache) |
| Clone offline | `pytest -k clone -v` | — | Clone offline tests (tests alpha feature: APFS validation, temp cache, CoW workflow); no network needed | No |
| ⏭️ Live Clone (ADR-007) | `MLXK2_ENABLE_ALPHA_FEATURES=1 pytest -m live_clone -v` | `live_clone` + Env: `MLXK2_ENABLE_ALPHA_FEATURES=1`, `MLXK2_LIVE_CLONE=1`, `HF_TOKEN`, `MLXK2_LIVE_CLONE_MODEL`, `MLXK2_LIVE_CLONE_WORKSPACE` | Real clone workflow: pull→temp cache→APFS same-volume clone→workspace (ADR-007 Phase 1 constraints: same volume + APFS required) | Yes |
| 🔒 Live Stop Tokens (ADR-009) | `pytest -m live_stop_tokens -v` | `live_stop_tokens` (required); Optional: `HF_HOME` (enables portfolio discovery) | Issue #32: Validates stop token behavior with real models. **With HF_HOME:** Portfolio Discovery auto-discovers all MLX chat models (filter: MLX+healthy+runtime+chat), RAM-aware skip, empirical report. **Without HF_HOME:** Uses 3 predefined models (see "Optional Setup" section for model requirements). | No (uses local cache) |
| ⏭️ Live Run | `pytest -m live_run -v` | `live_run` + Env: `MLXK2_USER_HF_HOME` or `HF_HOME` (user cache with `mlx-community/Phi-3-mini-4k-instruct-4bit`) | Regression tests for Issue #37: Validates private/org MLX model framework detection in run command (renames Phi-3 to simulate private-org model) | No (uses local cache) |
| 🔒 Live E2E (ADR-011) | `HF_HOME=/path/to/cache pytest -m live_e2e -v` | `live_e2e` (required) + Env: `HF_HOME` (optional, enables Portfolio Discovery); Requires: `httpx` installed | **✅ Working:** Server/HTTP/CLI validation with real models. Portfolio Discovery auto-discovers all MLX chat models via `mlxk list --json` (filter: MLX+healthy+runtime+chat), parametrized tests (one server per model), RAM-aware skip. | No (uses local cache) |
| 🔍 Show E2E Portfolio | `HF_HOME=/path/to/cache pytest -m show_model_portfolio -s` | `show_model_portfolio` + Env: `HF_HOME` | **Convenience:** Displays which models would be tested by `live_e2e` tests. Shows table with model keys (discovered_XX), RAM requirements, and test/skip status. No actual testing performed - just displays portfolio. | No (uses local cache) |
| 🔍 Manual Debug Mode | `mlxk run <model> "test prompt" --verbose` | Manual CLI usage with `--verbose` flag | **Quality Analysis:** Shows token generation details including multiple EOS token warnings. Use this for manual debugging of model quality issues. Output includes `[DEBUG] Token generation analysis` and `⚠️ WARNING: Multiple EOS tokens detected` for broken models. | No (uses local cache) |
| ⏭️ Issue #27 real-model | `pytest -m issue27 tests_2.0/test_issue_27.py -v` | Marker: `issue27`; Env (required): `MLXK2_USER_HF_HOME` or `HF_HOME` (user cache, read-only). Env (optional): `MLXK2_ISSUE27_MODEL`, `MLXK2_ISSUE27_INDEX_MODEL`, `MLXK2_SUBSET_COUNT=0`. | Copies real models from user cache into isolated test cache; validates strict health policy on index-based models (no network) | No (uses local cache) |
| Server tests (included) | `pytest -k server -v` | — | Basic server API tests (minimal, uses MLX stubs) | No |

**Useful commands:**
```bash
# Only Spec
pytest -m spec -v

# Push tests (offline)
pytest -k "push and not live" -v

# Clone tests (offline)
pytest -k "clone and not live" -v

# Exclude Spec
pytest -m "not spec" -v

# Live Push only
MLXK2_ENABLE_ALPHA_FEATURES=1 MLXK2_LIVE_PUSH=1 HF_TOKEN=... MLXK2_LIVE_REPO=... MLXK2_LIVE_WORKSPACE=... pytest -m live_push -v

# Live Clone only
MLXK2_ENABLE_ALPHA_FEATURES=1 MLXK2_LIVE_CLONE=1 HF_TOKEN=... MLXK2_LIVE_CLONE_MODEL=... MLXK2_LIVE_CLONE_WORKSPACE=... pytest -m live_clone -v

# Live List only
HF_HOME=/path/to/user/cache pytest -m live_list -v

# Live Stop Tokens only (ADR-009)
pytest -m live_stop_tokens -v  # Optional: HF_HOME=/path/to/cache for portfolio discovery

# Live Run only
HF_HOME=/path/to/user/cache pytest -m live_run -v

# Live E2E only (ADR-011)
HF_HOME=/path/to/user/cache pytest -m live_e2e -v  # See model list: pytest tests_2.0/live/test_server_e2e.py::TestChatCompletionsBatch --collect-only -q

# Issue #27 only
MLXK2_USER_HF_HOME=/path/to/user/cache pytest -m issue27 tests_2.0/test_issue_27.py -v

# All live tests (umbrella)
MLXK2_ENABLE_ALPHA_FEATURES=1 pytest -m wet -v
```

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

| Python Version | Status | Tests Passing | Skipped |
|----------------|--------|---------------|---------|
| 3.9.6 (macOS)  | ✅ Verified | 308/308 | 35 |
| 3.10.x         | ✅ Verified | 308/308 | 35 |
| 3.11.x         | ✅ Verified | 308/308 | 35 |
| 3.12.x         | ✅ Verified | 308/308 | 35 |
| 3.13.x         | ✅ Verified | 308/308 | 35 |

**Note:** 35 skipped tests are opt-in (live tests, alpha features). Skipped count may vary by environment:
- Without `HF_HOME`: Standard 35 skipped (live E2E tests use fallback parametrization)
- With `HF_HOME`: Live E2E tests run with discovered models (20+ additional tests executed)

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
- ✅ **Fallback:** Uses 3 predefined models (MXFP4, Qwen, Llama) if HF_HOME not set - models must exist in HF cache

**Required models for fallback (without HF_HOME):**
```bash
mlxk pull mlx-community/gpt-oss-20b-MXFP4-Q8         # ~12GB RAM
mlxk pull mlx-community/Qwen2.5-0.5B-Instruct-4bit   # ~1GB RAM
mlxk pull mlx-community/Llama-3.2-3B-Instruct-4bit   # ~4GB RAM
```

### E2E Tests with Portfolio Discovery (ADR-011)

**Status:** ✅ Working (refactored Nov 2025)

Auto-discovers and validates Server/HTTP/CLI interfaces with real models.

**Location:** `tests_2.0/live/` (test_server_e2e.py, test_cli_e2e.py, test_streaming_parity.py)
**Marker:** `live_e2e`

**Usage:**
```bash
# With HF_HOME: Auto-discovers all MLX chat models
export HF_HOME=/path/to/cache
pytest -m live_e2e -v

# See which models will be tested
pytest tests_2.0/live/test_server_e2e.py::TestChatCompletionsBatch --collect-only -q

# ⚠️ IMPORTANT: Always test collection before release
pytest -m live_e2e --collect-only  # Should work without errors
```

**Architecture:**
- ✅ **Portfolio Discovery via `mlxk list --json`:** Uses production command instead of duplicating cache logic (~70 LOC eliminated)
- ✅ **Parametrized Tests:** One pytest test per model (prevents RAM leaks from loop-based architecture)
- ✅ **model_key parametrization:** Collection regression fixed with fallback to empty list
- ✅ **Clean Lifecycle:** Each test gets its own server instance (45s timeout for MLX cleanup)
- ✅ **RAM-Aware:** Same progressive budgets as stop token tests (40%-70%)
- ✅ **Current result:** 73/81 tests passing (17 models discovered, 15 testable, 8 skipped: RAM budget) - no system freeze

**Tests Covered:**
- Server health/metadata endpoints
- Chat completions (batch + streaming)
- Text completions (batch + streaming)
- CLI `mlxk run` (text + JSON output)
- Streaming parity validation (Issue #20)
- Stop token filtering (Issue #32)

### RAM-Aware Model Selection

**Implementation:** `get_safe_ram_budget_gb()`, `should_skip_model()`

**Progressive RAM Budgets:**

| System RAM | Budget | Available for Models |
|------------|--------|---------------------|
| 16GB | 40% | 6.4GB |
| 32GB | 50% | 16GB |
| 64GB | 60% | 38.4GB |
| 96GB+ | 70% | 67GB+ |

**Rationale:** OS overhead is ~4-6GB (constant), larger systems have more headroom.

**Behavior:**
- Models exceeding budget → Auto-skipped
- Skip reason: "Model requires XGB but only YGB available"
- Empirical report tracks skipped models

**Example:**
```python
# 32GB system → 16GB budget
# Qwen-0.5B (1GB) → ✅ RUN
# Llama-3.2-3B (4GB) → ✅ RUN
# Mistral-7B (8GB) → ✅ RUN
# Mixtral-8x7B (32GB) → ⏭️ SKIP (exceeds 16GB budget)
```

## Future: Server E2E Testing (ADR-011)

**Status:** Planned for post-2.0.1

### Scope

End-to-end validation of Server/HTTP/CLI with real models:
- **HTTP API:** `/v1/chat/completions` (streaming + non-streaming)
- **SSE Format:** Server-Sent Events validation
- **CLI Integration:** `mlxk run`, `mlxk server` subprocess tests
- **Streaming Parity:** Issue #20 regression protection

### Planned Implementation

**Location:** `tests_2.0/live/test_server_e2e.py`, `test_streaming_parity.py`, `test_cli_e2e.py`
**Marker:** `live_e2e` (future)
**Infrastructure:** Reuses Portfolio Discovery + RAM-Aware logic from ADR-009

**Example:**
```python
@pytest.mark.live_e2e
def test_server_streaming_portfolio(portfolio_models):
    """Validate /v1/chat/completions SSE streaming across portfolio."""
    for model in portfolio_models:
        with LocalServer(model) as server:
            response = requests.post(f"{server.url}/v1/chat/completions",
                                    json={"stream": True, ...})
            # Validate SSE format, stop tokens, no visible EOS
```

**See:** ADR-011 for detailed architecture

---

## Appendix

### Complete Test File Structure (2.0.2)

```
tests_2.0/
├── __init__.py
├── conftest.py                        # Isolated test cache (HF_HOME override), safety sentinel, core fixtures
├── conftest_runner.py                 # Runner-specific fixtures/mocks
├── stubs/                             # Minimal mlx/mlx_lm stubs for unit/spec tests
│   ├── mlx/
│   │   └── core.py
│   └── mlx_lm/
│       ├── __init__.py
│       ├── generate.py
│       └── sample_utils.py
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
│   ├── conftest.py                              # Shared fixtures for live E2E tests (portfolio_models, pytest_generate_tests hook)
│   ├── server_context.py                       # LocalServer context manager for E2E testing (30s timeout for MLX cleanup)
│   ├── sse_parser.py                           # SSE parsing utilities for streaming validation
│   ├── test_utils.py                           # Portfolio Discovery (via mlxk list --json), RAM gating utilities
│   ├── test_cli_e2e.py                         # CLI integration E2E tests (ADR-011, parametrized)
│   ├── test_clone_live.py                      # Live clone flow (requires MLXK2_LIVE_CLONE, HF_TOKEN)
│   ├── test_list_human_live.py                 # Live list/health against user cache (requires HF_HOME)
│   ├── test_push_live.py                       # Live push flow (requires MLXK2_LIVE_PUSH, HF_TOKEN)
│   ├── test_server_e2e.py                      # Server E2E tests with real models (ADR-011, parametrized)
│   └── test_streaming_parity.py                # Streaming vs batch parity tests (Issue #20, ADR-011, parametrized)
├── test_adr004_error_logging.py       # ADR-004 error logging and redaction (tokens, paths)
├── test_cli_log_json_flag.py          # CLI --log-json flag behavior and JSON log format
├── test_cli_push_args.py              # Push CLI args and JSON error/output handling (offline)
├── test_cli_run_exit_codes.py         # CLI exit codes for run command errors (Issue #38)
├── test_clone_operation.py            # Clone operations with APFS optimization
├── test_ctrl_c_handling.py            # SIGINT handling during run/interactive flows
├── test_detection_readme_tokenizer.py # README/tokenizer-based framework detection
├── test_edge_cases_adr002.py          # Naming/health edge cases (ADR-002)
├── test_health_multifile.py           # Multi-file health completeness (index vs pattern)
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
├── test_push_dry_run.py               # Push dry-run diff planning (added/modified/deleted)
├── test_push_extended.py              # Extended push: no-op vs commit, branch/retry, .hfignore
├── test_push_minimal.py               # Minimal push scenarios (offline)
├── test_push_workspace_check.py       # Push check-only: workspace validation without network
├── test_robustness.py                 # Robustness for rm/pull/disk/timeout/concurrency
├── test_run_complete.py               # End-to-end run command (stream/batch/params)
├── test_runner_core.py                # MLXRunner core generation/memory/stop tokens
├── test_runtime_compatibility_reason_chain.py  # Runtime compatibility reason field decision chain (Issue #36)
├── test_server_api_minimal.py         # Minimal OpenAI-compatible server endpoints (SSE, JSON)
├── test_server_api.py.disabled        # Disabled server API tests (WIP/expanded scenarios)
├── test_server_models_and_errors.py   # Server model loading and error handling
├── test_server_streaming_minimal.py   # Server SSE streaming functionality
├── test_server_token_limits_api.py    # Server token limit enforcement
├── test_stop_tokens_live.py           # Stop token validation with real models (marker: live_stop_tokens, ADR-009)
└── test_token_limits.py               # Dynamic token calculation; server vs run policies
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

### Version History

### 2.0.3 (2025-11-17)
- ✅ **Test updates for stderr separation:** 4 test files modified to verify errors go to stderr (human mode)
  - `test_interactive_mode.py`: 2 tests patching stderr for ERROR messages
  - `test_run_complete.py`: 2 tests validating stderr error handling
  - `test_cli_run_exit_codes.py`: 3 tests checking stdout/stderr separation in JSON mode
  - `test_cli_push_args.py`: 2 tests verifying push stdout/stderr returns
- ✅ **Benchmark reporting infrastructure:** 4 live test files updated with benchmark fixtures
  - `live/conftest.py`: +225 lines - `report_benchmark()` fixture, `_parse_model_family()` helper
  - `live/test_cli_e2e.py`: Benchmark metadata reporting (family, variant, stop_tokens)
  - `live/test_server_e2e.py`: Benchmark metadata + performance (usage) data
  - `live/test_streaming_parity.py`: Portfolio Discovery refactoring (uses `mlxk list --json`)
- ✅ **Interactive mode reasoning control:** 2 new tests added (review_report.md)
  - `test_interactive_mode.py`: 1 test for `hide_reasoning` parameter passing
  - `test_run_complete.py`: 1 test for `TestRunReasoningControl` class
- ✅ **Test count:** 308 passed, 35 skipped (+2 tests from review_report.md fixes)

### 2.0.2 (2025-11-14)
- ✅ Test infrastructure hardening (TOKENIZERS_PARALLELISM, active polling, gc.collect())
- ✅ Portfolio Discovery validation complete (73/81 E2E tests, 17 models discovered)
- ✅ Sequential execution warning added (TESTING-DETAILS.md CRITICAL section)
- ✅ RAM logging infrastructure added (server_context.py, for future benchmark tooling)

### 2.0.2-dev (2025-11-13)
- ✅ Stop token ordering bugs fixed (batch + streaming modes)
- ✅ E2E test suite completed (ADR-011)
- ✅ 72/80 E2E tests passing baseline (15 models tested)
- ✅ TESTING.md restructuring (timeless policy + version-specific details split)

### 2.0.1 (2025-11-28)
- ✅ Portfolio Discovery (Issue #32, ADR-009)
- ✅ CLI exit code propagation (Issue #38)
- ✅ 306/306 tests passing

### 2.0.0 (2025-11-06)
- ✅ Complete rewrite with Apache 2.0 license
- ✅ JSON-first architecture
- ✅ Isolated test system with sentinel protection
- ✅ MLX stubs for fast testing without model downloads
- ✅ 3-category test strategy

---

*MLX-Knife 2.0.3 Testing Details*
