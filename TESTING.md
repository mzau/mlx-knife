# MLX Knife Testing Guide

## Current Status

✅ **98/98 tests passing** (September 2025) — 2.0.0-alpha.3; 9 skipped (opt-in)  
✅ **Apple Silicon verified** (M1/M2/M3)  
✅ **Python 3.9-3.13 compatible**  
✅ **Alpha (CLI/JSON)** — default suite green locally (no inference)
✅ **Isolated test system** - user cache stays pristine with temp cache isolation
✅ **3-category test strategy** - optimized for performance and safety

## Quick Start (2.0 Default)

```bash
# Install package + tests
pip install -e .[test]

# Download test model (optional; most 2.0 tests use isolated cache)
# Only needed for opt-in live tests or local experiments
# mlxk pull mlx-community/Phi-3-mini-4k-instruct-4bit

# Run 2.0 tests (default discovery: tests_2.0/)
pytest -v

# Live tests (opt-in; not part of default):
# - Live push (requires env):
#   export MLXK2_LIVE_PUSH=1
#   export HF_TOKEN=...; export MLXK2_LIVE_REPO=org/model; export MLXK2_LIVE_WORKSPACE=/abs/path
#   pytest -q -m live_push
# - Live list (uses your HF_HOME; requires at least one MLX chat + one MLX base in cache):
#   export HF_HOME=/path/to/huggingface/cache
#   pytest -q -m live_list

# Before committing
ruff check mlxk2/ --fix && mypy mlxk2/ && pytest -v
```

## Why Local Testing?

MLX Knife tests fall into two categories for 2.0:

- CLI/JSON tests (default): Run on any supported Python on macOS; no model inference required; use an isolated HF cache (no network).
- Live/Inference tests (opt-in; future RC for server/run): Require Apple Silicon (M1/M2/M3) and real models.

For push/list live tests in 2.0 alpha, see the opt-in commands above.

## Test Structure

### 2.0 Test Structure (default)

```
tests_2.0/
├── __init__.py
├── conftest.py                          # Isolated test cache, fixtures
├── test_human_output.py                 # Human rendering (list/health)
├── test_detection_readme_tokenizer.py   # Issue #31 (README/tokenizer detection)
├── test_json_api_list.py                # JSON API (list contract)
├── test_json_api_show.py                # JSON API (show contract)
├── test_edge_cases_adr002.py            # Edge-case naming, ADR-002
├── test_health_multifile.py             # Multi-file health completeness
├── test_integration.py                  # Model resolution, health integration
├── test_issue_27.py                     # Health policy consistency
├── test_model_naming.py                 # Pattern/@hash parsing and resolution
├── test_robustness.py                   # General robustness tests
├── test_cli_push_args.py                # Push CLI args (offline)
├── test_push_minimal.py                 # Push minimal (offline)
├── test_push_extended.py                # Push extended (offline)
├── test_push_dry_run.py                 # Push dry-run planning (offline)
├── test_push_workspace_check.py         # Push check-only (offline)
├── spec/
│   ├── test_cli_version_output.py               # version command JSON shape
│   ├── test_spec_doc_examples_validate.py       # docs examples vs schema
│   ├── test_spec_version_sync.py                # docs version == code constant
│   ├── test_push_error_matches_schema.py        # push error schema
│   └── test_push_output_matches_schema.py       # push success schema
└── live/                                       # Opt-in live tests (markers)
    ├── test_push_live.py                        # requires MLXK2_LIVE_PUSH, HF_TOKEN
    └── test_list_human_live.py                  # requires HF_HOME
```

Note: Live tests are opt-in via markers (`-m live_push`, `-m live_list`) and environment. Default `pytest` discovery runs only the offline suite above.

## Push Testing (2.0)

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
  - `message: string|null` — short human hint; mirrors hub on no‑op.
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

Notes on output verbosity and behavior
- JSON is quiet by default: only the final JSON object is printed. Use `--verbose` to allow hub logs/progress to reach the console (the JSON payload remains unchanged). For assertions, prefer `data.hf_logs`.
- Human mode is chatty by default: progress + one‑liner summary. `--verbose` appends the commit URL when present.
- No‑changes detection: If the hub reports “No files have been modified… Skipping to prevent empty commit.”, JSON sets `no_changes: true`, `uploaded_files_count: 0`, and nulls `commit_sha`/`commit_url`. Human shows “— no changes”. This hub signal is preferred over inferring from file lists.
 - `--dry-run` human output: prints a concise plan line `dry-run: +A ~M -D` (modifications are an approximation and may be `~?` in rare cases).
 - Branch creation with `--create`: Even if the push is a no‑op, the target branch is created upfront.

Examples (expected)
- No‑op re‑push (JSON): `commit_sha: null`, `commit_url: null`, `uploaded_files_count: 0`, `no_changes: true`, `message` mirrors hub text, `hf_logs` contains hub lines.
- Commit (JSON): `commit_sha`/`commit_url` populated; `uploaded_files_count == sum(change_summary.values())`; `message` summarizes counts.

- Dry-run (existing repo/branch, no changes) — JSON:
  ```json
  {
    "status": "success",
    "command": "push",
    "error": null,
    "data": {
      "repo_id": "org/model",
      "branch": "main",
      "commit_sha": null,
      "commit_url": null,
      "repo_url": "https://huggingface.co/org/model",
      "uploaded_files_count": 0,
      "local_files_count": 11,
      "no_changes": true,
      "created_repo": false,
      "message": "Dry-run: no changes",
      "experimental": true,
      "disclaimer": "Experimental feature (M0: upload only). No validation/filters; review results on the Hub.",
      "dry_run": true,
      "dry_run_summary": {"added": 0, "modified": null, "deleted": 0},
      "change_summary": {"added": 0, "modified": 0, "deleted": 0},
      "would_create_repo": false,
      "would_create_branch": false,
      "added_files": [],
      "deleted_files": []
    }
  }
  ```

- Dry-run (existing repo/branch, changes present) — JSON:
  ```json
  {
    "status": "success",
    "command": "push",
    "error": null,
    "data": {
      "repo_id": "org/model",
      "branch": "main",
      "commit_sha": null,
      "commit_url": null,
      "repo_url": "https://huggingface.co/org/model",
      "uploaded_files_count": 0,
      "local_files_count": 11,
      "no_changes": false,
      "created_repo": false,
      "message": "Dry-run: +2 ~? -1",
      "experimental": true,
      "disclaimer": "Experimental feature (M0: upload only). No validation/filters; review results on the Hub.",
      "dry_run": true,
      "dry_run_summary": {"added": 2, "modified": null, "deleted": 1},
      "change_summary": {"added": 2, "modified": 0, "deleted": 1},
      "would_create_repo": false,
      "would_create_branch": false,
      "added_files": ["new.txt", "weights/model.safetensors"],
      "deleted_files": ["old.txt"]
    }
  }
  ```

- Dry-run — Human output:
  ```
  push (experimental): org/model@main — dry-run: no changes
  push (experimental): org/model@main — dry-run: +2 ~? -1
  ```

Spec/Schema
- The JSON API spec version and schema live in `mlxk2/spec.py` and `docs/json-api-specification.md`. The docs schema includes support for `command: "push"` and its fields. Keep tests in sync with those sources of truth.

**Automated (offline)**
- **Token/Workspace errors:** Missing `HF_TOKEN` and missing workspace produce proper JSON errors.
- **CLI args (JSON mode):** Missing positional args emit JSON errors rather than usage text.
- **Schema shape:** Push success/error outputs validate against `docs/json-api-schema.json`.
- **No-op push:** Detects `no_changes: true`, sets `uploaded_files_count: 0`, carries hub message into JSON (`message`/`hf_logs`), and human output shows "no changes" without duplicate logs.
- **Commit path:** Extracts `commit_sha`, `commit_url`, `change_summary` (+/~/−), correct `uploaded_files_count`; human `--verbose` includes URL.
- **Repo/Branch handling:** Missing repo requires `--create`; with `--create` sets `created_repo: true`. Missing branch is tolerated; upload attempts proceed. With `--create`, the branch is proactively created and the upload is retried once if the hub rejects the revision (e.g., “Invalid rev id”).
- **Ignore rules:** `.hfignore` is merged with default ignores and forwarded to the hub.

Files:
- `tests_2.0/test_cli_push_args.py` (CLI errors and JSON outputs)
- `tests_2.0/test_push_extended.py` (no-op vs commit, branch/repo, .hfignore, human; includes retry on invalid revision with `--create`)
- `tests_2.0/spec/test_push_output_matches_schema.py` (schema success path)

Run (venv39):
- `source venv39/bin/activate && pip install -e .`
- `pytest -q tests_2.0/test_cli_push_args.py tests_2.0/test_push_extended.py`
- `pytest -q tests_2.0/spec/test_push_output_matches_schema.py`
- Targeted retry test: `pytest -q tests_2.0/test_push_extended.py::test_push_retry_creates_branch_on_upload_revision_error`

**Live (opt-in / wet)**
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
- Notes:
  - Live test does not use `--create` (safety). If the repo does not exist, create it once manually.
  - Manual create example: `mlxk2 push --private --create "$MLXK2_LIVE_WORKSPACE" "$MLXK2_LIVE_REPO" --json`

**Manual Checklist (Live)**
- **Create repo (first time):** `--private --create` → expect `created_repo: true`, private repo on HF.
- **No-op re-push:** identical workspace → `no_changes: true`, `uploaded_files_count: 0`, concise human "no changes".
- **Commit after change:** edit a small file → push shows `commit_sha`, `commit_url`, `change_summary` matches expectations.
- **.hfignore behavior:** add ignores (e.g., `.idea/`, `.vscode/`, `*.ipynb`) → verify excluded on HF.
- Optional errors: invalid token or missing rights → JSON `error` (`upload_failed` / auth error), clear message.

Human vs JSON:
- Human output is derived from JSON only; hub logs are not printed directly.
- Use `--verbose` with human output to append the commit URL or short message; JSON content stays the same structurally.

## Manual MLX Chat Model Smoke Test (2.0)

Goal: Pull a small MLX chat model, verify classification, prepare a local workspace, validate it offline, and push to a private repo while preserving chat intent. This helps issuers validate iOS‑focused workflows.

Model choice (example)
- `mlx-community/Qwen2.5-0.5B-Instruct-4bit` (small, chat‑oriented)

Steps
- Pull (venv39):
  - `mlxk2 pull mlx-community/Qwen2.5-0.5B-Instruct-4bit`
- Verify in cache:
  - `mlxk2 list --health "Qwen2.5-0.5B-Instruct-4bit"`
  - Expect: Framework MLX, Type chat, capabilities include chat
- Prepare local workspace from cache (dereference symlinks):
  - Ensure `HF_HOME` points to your HF cache (optional, but recommended)
  - Compute cache path: `$HF_HOME/models--mlx-community--Qwen2.5-0.5B-Instruct-4bit`
  - Find latest snapshot hash under `snapshots/`
  - Copy to workspace and dereference symlinks:
    - `rsync -aL "$HF_HOME/models--mlx-community--Qwen2.5-0.5B-Instruct-4bit/snapshots/<HASH>/" ./mymodel_test_workspace/`
- Recommended README front‑matter (to preserve intent on push):
  - Include YAML with tags and pipeline tag, e.g.
    - `tags: [mlx, chat]`
    - `pipeline_tag: text-generation`
    - `base_model: <upstream_base>`
  - Keep model name containing `Instruct` or `chat` to aid chat detection
- Offline validation (no network):
  - `mlxk2 push --check-only ./mymodel_test_workspace <org/model> --json`
  - Expect: `workspace_health.healthy: true`; ensure tokenizer present (`tokenizer.json` or `tokenizer.model`) and at least one non‑empty weight file
- Push to private repo:
  - `mlxk2 push --private --create ./mymodel_test_workspace <org/model> --json`
  - Re‑push without changes should show `no_changes: true`
- Post‑push verification:
  - `mlxk2 list --all --health <org/model>`
  - Current limitation: Framework may show `PyTorch` for non‑`mlx-community` orgs due to conservative detection. This does not affect content; future M1 will parse model card tags (`mlx`) to classify MLX across orgs.

Notes
- Ensure tokenizer files exist (tokenizer.json/tokenizer.model) and optional generation_config.json for runnable chat contexts.
- Avoid pushing unwanted files; use `.hfignore` for project‑specific filters.

### 1.x Legacy Test Suite (separate)

- Location: `tests/` (stable 1.x release on `main`).
- Not part of the 2.0 default run; execute explicitly with `pytest tests/ -v`.
- Contains extensive integration/server tests unrelated to the 2.0 JSON CLI.

## Legacy 1.x: 3-Category Test Strategy (main)

MLX Knife uses a **3-category test strategy** to balance test isolation, performance, and user cache protection:

### 🏠 CATEGORY 1: ISOLATED CACHE (Most Tests)
**✅ User cache stays pristine** - Tests use temporary isolated caches with automatic cleanup

**Implemented Tests (78 tests):**
- ✅ `test_real_model_lifecycle.py` - Full model lifecycle with `tiny-random-gpt2` (~12MB download)
- ✅ `test_core_functionality.py` - Basic CLI operations with `patch_model_cache` isolation  
- ✅ `test_process_lifecycle.py` - Process management with isolated cache + MODEL_CACHE patching
- ✅ `test_run_command_advanced.py` - Run command edge cases with `mock_model_cache` in isolation
- ✅ `test_lock_cleanup_bug.py` - Lock cleanup testing with temporary MODEL_CACHE override
- ✅ `test_health_checks.py` - Mock corruption testing with isolated `temp_cache_dir`

**Technical Pattern:**
```python
@pytest.mark.usefixtures("temp_cache_dir")
class TestBasicLifecycle:
    def test_something(self, temp_cache_dir, patch_model_cache):
        with patch_model_cache(temp_cache_dir / "hub"):
            # Test operates in complete isolation
            # User cache never touched, automatic cleanup
```

**Benefits:** 
- ✅ **Clean User Cache**: No test artifacts or broken models ever
- ✅ **Parallel Testing**: No cache conflicts between test runs  
- ✅ **Reproducible**: No dependency on existing models in user cache
- ✅ **Fast CI**: Small models (12MB vs 4GB) for most tests

### 🏥 CATEGORY 2: USER CACHE (Framework Diversity)
**📋 Reserved for future** - Real model diversity that cannot be mocked

**Future Framework Validation Tests:**
- Multiple framework detection (MLX + PyTorch + Tokenizer-only models)
- Health check diversity testing with naturally corrupted models
- Cross-framework model compatibility validation

**Currently**: All health/framework tests use `mock_model_cache` and are Category 1 (isolated)

### 🖥️ CATEGORY 3: SERVER CACHE (Performance Tests)  
**🔒 Large models, user cache expected** - Marked with `@pytest.mark.server`

**Server Tests (Excluded from default `pytest`):**
- 🔒 `test_issue_14.py` - Chat self-conversation regression tests
- 🔒 `test_issue_15_16.py` - Dynamic token limit validation  
- 🔒 `test_end_token_issue.py` - End-token filtering consistency
- 🔒 `test_server_functionality.py` - OpenAI API compliance (basic tests only)

**Technical Pattern:**
```python
@pytest.mark.server  # Excluded from default pytest
def test_server_feature(mlx_server, model_name: str):
    # Uses real models in user cache
    # Requires significant RAM and time
```

**Characteristics:**
- 🔒 **Not run by default** - Must use `pytest -m server`
- 💾 **RAM-aware** - Auto-skip models exceeding available memory
- ⏱️ **Longer execution** - 20-40 minutes for full suite
- 🎯 **Model diversity** - Tests across different model sizes/architectures

## Test Prerequisites

### Required Setup

1. **Apple Silicon Mac** (M1/M2/M3)
2. **Python 3.9 or newer**
3. **Test dependencies installed** (includes jsonschema for Spec tests):
   ```bash
   pip install -e .[test]
   ```

Notes:
- Spec validation requires `jsonschema`. Installing `.[test]` ensures it is available.
- Without `jsonschema`, Spec example validation is skipped (you will see one extra SKIPPED test).
- With `jsonschema` installed, expect one additional PASS in the `-m spec` and `tests_2.0/` totals.

**That's it!** Most tests (Category 1) use isolated caches and download small test models automatically (~12MB).

### Enabling Issue #27 Tests (optional)

By default, several Issue #27 tests are skipped because they require a real multi‑shard safetensors model (with `model.safetensors.index.json`) in your user cache and enough free disk space to create an isolated copy.

- Set your user cache: `export MLXK2_USER_HF_HOME=/absolute/path/to/your/huggingface/cache`
- Ensure the cache contains a model with a safetensors index (common for larger Llama/Mistral models).
- Run the focused tests: `PYTHONPATH=. pytest tests_2.0/test_issue_27.py -v`
- If you see skips:
  - “No safetensors index found” → pick a model that has `model.safetensors.index.json`.
  - “Not enough free space” → free disk space; tests create a subset copy into an isolated temp cache.
  - “User model not found” → verify the exact HF path in your cache and env var points to its `.../huggingface/cache` root.

With a suitable model present and `MLXK2_USER_HF_HOME` set, the Issue #27 tests should run without SKIPs.

### When Issue #27 real‑model tests make sense

Purpose
- These tests validate the strict health policy against real upstream Hugging Face repositories that ship multi‑shard safetensors with a `model.safetensors.index.json`. They complement the deterministic unit tests by exercising real‑world layouts.

Run them when
- Your user cache contains at least one upstream PyTorch repo with a safetensors index (not MLX/GGUF conversions). Good candidates:
  - `mistralai/Mistral-7B-Instruct-v0.2` or `-v0.3`
  - `Qwen/Qwen1.5-7B-Chat`, `Qwen/Qwen2-7B-Instruct`
  - `teknium/OpenHermes-2.5-Mistral`
  - Gated: `meta-llama/Llama-2-7b-chat-hf`, `meta-llama/Llama-3-8B-Instruct`, `google/gemma-7b-it`
- You want to sanity‑check index‑based completeness, shard deletion/truncation, and LFS pointer detection against real artifacts.

They are not useful when
- Your cache only has MLX Community models (no `model.safetensors.index.json`) or GGUF models — the index‑based tests will skip by design. In that case, rely on `tests_2.0/test_health_multifile.py` for deterministic coverage.

Resource considerations
- Disk: tests copy a subset of files into an isolated cache. Tune size/speed with:
  - `export MLXK2_COPY_STRATEGY="index_subset"`
  - `export MLXK2_SUBSET_COUNT="1"`
  - `export MLXK2_MIN_FREE_MB="512"` (or higher)
- Network: if you need to fetch a candidate model first, prefer downloading only `config.json`, `model.safetensors.index.json`, and 1–2 small shards to keep it light.

Summary
- If you have a suitable upstream PyTorch chat/instruct model with an index in your user cache, enable the env vars above and run `tests_2.0/test_issue_27.py` for an extra layer of real‑model assurance. Otherwise, the deterministic tests already validate the policy thoroughly.

### Optional Setup (Server Tests Only)

For server tests (`@pytest.mark.server` - **excluded by default**):
```bash
# Medium model for server testing
mlxk pull mlx-community/Phi-3-mini-4k-instruct-4bit

# Different architecture for variety  
mlxk pull mlx-community/Mistral-7B-Instruct-v0.3-4bit
```

**Note**: Server tests are excluded from default `pytest` and require manual execution with `pytest -m server`.

## Environment & Caches

To keep results reproducible and caches safe on Apple Silicon:

- Preferred Python/venv: Apple‑native 3.9 in a dedicated env
  - Example: `python3.9 -m venv venv39 && source venv39/bin/activate && pip install -e .[test]`
- User cache (persistent): shared, real cache for manual ops and certain advanced/server tests
  - Example (external SSD): `export HF_HOME="/Volumes/SomeExternalSSD/models"`
  - Safe ops: `list`, `health`, `show`; Coordinate `pull`/`rm` (maintenance window)
- Test cache (isolated/default): ephemeral via fixtures; default `pytest` runs must not force the user cache
  - Category 1 tests use temporary caches and should not depend on `HF_HOME`
  - Only server/advanced tests may require user cache and are excluded by default (`-m server`)
  - Deletion safety: tests set `MLXK2_STRICT_TEST_DELETE=1` so delete ops fail if not in test cache

In PRs, please state your Python version and whether you used the user cache or isolated test caches.

## Test Commands

### Basic Test Execution

```bash
# All tests (recommended before commits)
pytest

# Only integration tests (system-level)
pytest tests/integration/

# Only unit tests (fast)
pytest tests/unit/

# Verbose output
pytest -v

# Show test coverage
pytest --cov=mlx_knife --cov-report=html
```

### Specific Test Categories

```bash
# Process lifecycle tests (critical for production)
pytest tests/integration/test_process_lifecycle.py -v

# Health check robustness (model corruption detection)
pytest tests/integration/test_health_checks.py -v

# Core functionality (basic CLI commands)
pytest tests/integration/test_core_functionality.py -v

# Issue #20: End-token filtering consistency (new in 1.1.0-beta2)
pytest tests/integration/test_end_token_issue.py -v

# Advanced run command tests
pytest tests/integration/test_run_command_advanced.py -v

# Server functionality tests
pytest tests/integration/test_server_functionality.py -v

# Lock cleanup bug tests (Issue #23 - new in 1.1.0-beta3)
pytest tests/integration/test_lock_cleanup_bug.py -v
```

### Test Filtering

```bash
# Run only basic operations tests
pytest -k "TestBasicOperations" -v

# Server tests are excluded by default (marked with @pytest.mark.server)
# Run server tests manually (requires large models in user cache)
pytest -m server -v

# Skip server tests explicitly (default behavior)
pytest -m "not server" -v

# Run only process lifecycle tests
pytest -k "process_lifecycle or zombie" -v

# Run health check tests only
pytest -k "health" -v

# Only JSON API contract/spec tests
pytest -m spec -v
```

### Timeout and Performance

```bash
# Set custom timeout (default: 300s)
pytest --timeout=60

# Show slowest tests
pytest --durations=10

# Parallel execution (if pytest-xdist installed)
pytest -n auto
```

### Server Tests (Advanced)

**⚠️ Warning**: Server tests require significant system resources and time.

```bash
# Run comprehensive Issue #20 server tests (48 tests, ~30 minutes)
pytest tests/integration/test_end_token_issue.py -m server -v

# All server-marked tests (includes above + server functionality)
pytest -m server -v

# Quick server functionality test only
pytest tests/integration/test_server_functionality.py -v

# Server tests are RAM-aware - automatically skip models that don't fit
```

**Server Test Requirements:**
- **RAM**: 8GB+ recommended (16GB+ for large models)  
- **Time**: 20-40 minutes for full suite
- **Models**: Multiple 4-bit quantized models (1B-30B parameters)
- **Coverage**: Streaming vs non-streaming consistency, token limits, API compliance

## Python Version Compatibility

### Verification Results (August 2025)

**✅ 150/150 tests passing** - All standard tests validated on Apple Silicon with isolated cache system

| Python Version | Status | Tests Passing |
|----------------|--------|---------------|
| 3.9.6 (macOS)  | ✅ Verified | 150/150 |
| 3.10.x         | ✅ Verified | 150/150 |
| 3.11.x         | ✅ Verified | 150/150 |
| 3.12.x         | ✅ Verified | 150/150 |
| 3.13.x         | ✅ Verified | 150/150 |

All versions tested with isolated cache system.
Real MLX execution verified separately with server/run commands.

### Manual Multi-Python Testing

If you have multiple Python versions installed, you can verify compatibility:

```bash
# Run the multi-Python verification script
./test-multi-python.sh

# Or manually test specific versions
python3.9 -m venv test_39
source test_39/bin/activate
pip install -e . && pip install pytest
pytest
deactivate && rm -rf test_39
```

## Code Quality & Development

### Code Quality Tools

MLX Knife includes comprehensive code quality tools:

```bash
# Install development tools
pip install ruff mypy

# Automatic code formatting and linting
ruff check mlx_knife/ --fix

# Type checking with mypy
mypy mlx_knife/

# Complete development workflow
ruff check mlx_knife/ --fix && mypy mlx_knife/ && pytest
```

## Mini‑Matrix: What runs by default vs markers

| Target | How to Run | Markers / Env | Includes | Network |
|---|---|---|---|---|
| Default 2.0 suite | `pytest -v` | — | JSON‑API (list/show/health), Human‑Output, Model‑Resolution, Health‑Policy, Push Offline (`--check-only`, `--dry-run`), Spec/Schema checks | No |
| Spec‑only | `pytest -m spec -v` | `spec` | Schema/contract tests, version sync, docs example validation | No |
| Exclude Spec | `pytest -m "not spec" -v` | `not spec` | Everything except spec/schema checks | No |
| Live Push (opt‑in) | `pytest -m live_push -v` (or all live tests: `pytest -m wet -v`) | `live_push` (subset of `wet`) + Env: `MLXK2_LIVE_PUSH=1`, `HF_TOKEN`, `MLXK2_LIVE_REPO`, `MLXK2_LIVE_WORKSPACE` | JSON push against the real Hub; on errors the test SKIPs (diagnostic) | Yes |
| Issue #27 real‑model (opt‑in) | `pytest tests_2.0/test_issue_27.py -v` | Env: `MLXK2_USER_HF_HOME` (user cache with multi‑shard models) | Strict health policy on real index‑based models | No (uses local cache) |
| Server/run (separate) | `pytest tests/integration -m server -v` | `server` | Heavy server/run tests, RAM‑dependent, longer duration | No (models local) |

Useful commands
- Only Spec: `pytest -m spec -v`
- Offline Push only: `pytest -k "push and not live" -v`
- Exclude Spec: `pytest -m "not spec" -v`
- Live Push only: `MLXK2_LIVE_PUSH=1 HF_TOKEN=... MLXK2_LIVE_REPO=... MLXK2_LIVE_WORKSPACE=... pytest -m live_push -v`
- All live tests (umbrella): `pytest -m wet -v` (may include future live tests beyond push)

Markers: wet vs live_push
- `wet`: umbrella marker for any opt‑in “live” test that may require network, credentials, or user environment. Use to run all live tests.
- `live_push`: narrow marker for push‑specific live tests only. Use to target push live checks without running other live suites.

Note: Without the required env vars, live tests remain SKIPPED.

### Development Workflow

Before committing changes:

```bash
#!/bin/bash
# pre-commit-check.sh - Run before committing
set -e

echo "🧪 Running MLX Knife pre-commit checks..."

# 1. Code style
echo "Checking code style..."
ruff check mlx_knife/ --fix

# 2. Type checking
echo "Checking types..."
mypy mlx_knife/

# 3. Quick smoke test
echo "Running quick tests..."
pytest tests/unit/ -v

echo "✅ All checks passed. Safe to commit!"
```

## Local Development Testing

### Adding New Tests
1. **Integration tests** go in `tests/integration/`
2. **Unit tests** go in `tests/unit/`
3. Use existing fixtures from `conftest.py`
4. Follow naming: `test_*.py`, `Test*` classes, `test_*` methods

### Test Categories (Markers)
```python
@pytest.mark.integration  # Slower system tests
@pytest.mark.unit         # Fast isolated tests  
@pytest.mark.slow         # Tests >30 seconds
@pytest.mark.requires_model  # Needs actual MLX model
@pytest.mark.network      # Requires internet
@pytest.mark.server       # Requires MLX Knife server (excluded from default pytest)
```

### Mock Utilities
- `mock_model_cache()`: Creates fake model directories
- `mlx_knife_process()`: Manages subprocess lifecycle
- `process_monitor()`: Tracks zombie processes
- `temp_cache_dir()`: Isolated test environment

## Test Philosophy

Following the **"Process Hygiene over Edge-Case Perfection"** principle:

1. **Process Cleanliness**: No zombies, no leaks ✅
2. **Health Checks**: Reliable corruption detection ✅  
3. **Core Operations**: Basic functionality works ✅
4. **Error Handling**: Graceful failures ✅

The test suite validates production readiness with real Apple Silicon hardware and actual MLX models.

## Troubleshooting

### Common Issues

**Tests hang forever:**
```bash
pytest --timeout=60
```

**Import errors:**
```bash
pip install -e . && pip install pytest
```

**Process cleanup issues:**
```bash
ps aux | grep mlx_knife  # Check for zombies
```

**Cache conflicts:**
```bash
export HF_HOME="/tmp/test_cache"
pytest --cache-clear
```

### Test Environment

```bash
# Clean test run
rm -rf .pytest_cache __pycache__
pytest tests/ -v --cache-clear

# Debug specific test
pytest tests/integration/test_health_checks.py::TestHealthCheckRobustness::test_healthy_model_detection -v -s
```

## Contributing Test Results

When submitting PRs, please include:

1. **Your test environment**:
   - macOS version
   - Apple Silicon chip (M1/M2/M3)
   - Python version
   - Which model(s) you tested with

2. **Test results summary (2.0)**:
  ```
  Platform: macOS 14.5, M2 Pro
  Python: 3.11.6
  Results: 98/98 tests passed; 9 skipped (opt-in)
  ```

3. **Any issues encountered** and how you resolved them

## Summary

**Legacy 1.x Testing Status (main):**

✅ **Stable** - 150/150 tests passing  
✅ **Isolated Test System** - User cache stays pristine with temp cache isolation
✅ **3-Category Strategy** - Optimized for performance and safety
✅ **Multi-Python Support** - Python 3.9-3.13 verified  
✅ **Code Quality** - ruff/mypy integration working  
✅ **Real Model Testing** - Server/run commands validated with multiple models
✅ **Memory Management** - Context managers prevent leaks  
✅ **Exception Safety** - Context managers ensure cleanup  
✅ **Cache Directory Fix** - Issue #21: Empty cache crash resolved
✅ **LibreSSL Warning Fix** - Issue #22: macOS Python 3.9 warning suppression
✅ **Lock Cleanup Fix** - Issue #23: Enhanced rm command with lock cleanup

This testing framework validates MLX Knife's stability through isolated testing with automatic model downloads and separate real MLX validation.

## Server-Based Testing (Legacy 1.x; 2.0 RC planned)

In 1.x (main), some tests require a running MLX Knife server with loaded models and are marked with `@pytest.mark.server`. For 2.0, server/run will return in the RC; until then, server tests are legacy-only.

### Why Separate Server Tests?

- **Test count varies** by loaded models (makes CI reporting inconsistent)
- **Large memory requirements** - need different models for different RAM sizes  
- **Longer execution time** - each model needs to load individually
- **Manual setup required** - need to download appropriate models first

### Prerequisites for Server Tests

| System RAM | Recommended Models | Commands |
|------------|-------------------|----------|
| **16GB**   | Small models only | `mlxk pull mlx-community/Qwen2.5-0.5B-Instruct-4bit`<br>`mlxk pull mlx-community/Llama-3.2-1B-Instruct-4bit`<br>`mlxk pull mlx-community/Llama-3.2-3B-Instruct-4bit` |
| **32GB**   | + Medium models | `mlxk pull mlx-community/Phi-3-mini-4k-instruct-4bit`<br>`mlxk pull mlx-community/Mistral-7B-Instruct-v0.2-4bit`<br>`mlxk pull mlx-community/Mixtral-8x7B-Instruct-v0.1-4bit` |
| **64GB**   | + Large models | `mlxk pull mlx-community/Mistral-Small-3.2-24B-Instruct-2506-4bit`<br>`mlxk pull mlx-community/Qwen3-30B-A3B-Instruct-2507-4bit`<br>`mlxk pull mlx-community/Llama-3.3-70B-Instruct-4bit` |
| **96GB+**  | + Huge models | `mlxk pull mlx-community/Qwen3-Coder-480B-A35B-Instruct-4bit` |

### Running Server Tests

**Issue #14 Regression Tests** (Chat Self-Conversation Bug):

```bash
# Set environment
export HF_HOME=/path/to/your/cache

# Smoke test first (see which models are available)
python tests/integration/test_issue_14.py

# Run server tests only (excluded from default pytest)
pytest -m server -v

# Run specific Issue #14 tests
pytest tests/integration/test_issue_14.py -m server -v
```

**Expected Output:**
```
🦫 MLX Knife Issue #14 Test - Smoke Test
==================================================
📊 Safe models for this system: 6
💾 System RAM: 64GB total, 40GB available

  🎯 mlx-community/Mistral-7B-Instruct-v0.2-4bit
     └─ Size: 7B, RAM needed: 8GB
  🎯 mlx-community/Llama-3.2-3B-Instruct-4bit  
     └─ Size: 3B, RAM needed: 4GB
  [...]

========== test session starts ==========
tests/integration/test_issue_14.py::test_server_health[mlx_server] PASSED
tests/integration/test_issue_14.py::test_issue_14_self_conversation_regression_original[mlx-community/Mistral-7B-Instruct-v0.2-4bit-7B-8] PASSED
[...6 more model tests...]
========== 7 passed in 45.23s ==========
```

### Additional Server Tests

**Issues #15 & #16** - Dynamic Token Limits (Implemented in 1.1.0-beta1):
```bash
pytest tests/integration/test_issue_15_16.py -v
```

**Issue #20** - End-Token Filtering (Implemented in 1.1.0-beta2):
```bash
pytest tests/integration/test_end_token_issue.py -m server -v
```

### Troubleshooting Server Tests

**Permission warnings are normal:**
```
WARNING: ⚠️  Cannot scan network connections (permission denied)
INFO: 🔧 Falling back to process-based cleanup only
```
This is expected on macOS - the tests continue with process-based cleanup.

**Memory issues:**
- Tests automatically skip models exceeding 80% available RAM
- Use smaller models if you see consistent memory failures  
- Consider external SSD for model cache to reduce memory pressure

**Server startup failures:**
```bash
# Debug server manually
python -m mlx_knife.cli server --port 8000

# Check model health  
mlxk health

# Verify environment
echo $HF_HOME
```

### Adding New Server Tests

When contributing server-based tests:

```python
@pytest.mark.server
def test_new_feature(mlx_server, model_name: str, size_str: str, ram_needed: int):
    """Test new feature with MLX models.""" 
    # Use mlx_server fixture for automatic server management
    # Test implementation here
```

1. **Mark with `@pytest.mark.server`** - excludes from default `pytest`
2. **Use `mlx_server` fixture** - automatic server lifecycle management
3. **Test RAM requirements** - use `get_safe_models_for_system()` helper
4. **Document in TESTING.md** - add to this guide
