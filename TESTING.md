# MLX Knife Testing Guide

## Current Status

✅ **306/306 tests passing** (November 2025) — 2.0.1 Stable Release; 20 skipped (opt-in)
✅ **Test environment:** macOS 14.x, M2 Max, Python 3.9-3.13
✅ **Production verified & reported:** M1, M1 Max, M2 Max in real-world use
✅ **License:** Apache 2.0 (was MIT in 1.x)
✅ **Isolated test system** - user cache stays pristine with temp cache isolation
✅ **3-category test strategy** - optimized for performance and safety

### Skipped Tests Breakdown (20 total, standard run without HF_HOME)
- **4 Live Stop Tokens tests** - Stop token validation with real models (requires `pytest -m live_stop_tokens`, ADR-009)
- **1 Live Run test** - Private/org model detection (requires `pytest -m live_run`, Issue #37)
- **3 Live Clone tests** - APFS same-volume clone workflow (requires `MLXK2_LIVE_CLONE=1`)
- **1 Live List test** - Tests against user cache (requires HF_HOME with models)
- **1 Live Push test** - Real HuggingFace push (requires `MLXK2_LIVE_PUSH=1`)
- **7 Issue #27 tests** - Real-model health validation (requires HF_HOME or MLXK2_USER_HF_HOME setup)
- **3 Additional opt-in tests** - Various live validation scenarios

## Quick Start (2.0 Default)

```bash
# Install package + development tools (required for ruff/mypy/pytest)
pip install -e ".[dev,test]"

# Download test model (optional; most 2.0 tests use isolated cache)
# Only needed for opt-in live tests or local experiments
# mlxk pull mlx-community/Phi-3-mini-4k-instruct-4bit

# Run 2.0 tests (default discovery: tests_2.0/)
pytest -v  # Runs ~300 tests (isolated, no live downloads)

# Optional: Enable alpha push and clone tests
MLXK2_ENABLE_ALPHA_FEATURES=1 pytest -v  # Activates alpha features (clone/push)

# Live tests (opt-in; not part of default suite):
# - Live stop tokens (ADR-009 - requires models in HF_HOME):
#   pytest -m live_stop_tokens
#   # Tests: MXFP4, Qwen 2.5, Llama 3.2 stop token behavior
# - Live run (requires models in HF_HOME):
#   pytest -m live_run
#   # Tests: Issue #37 private/org model detection
# - Live push (requires alpha features + env):
#   export MLXK2_ENABLE_ALPHA_FEATURES=1
#   export MLXK2_LIVE_PUSH=1
#   export HF_TOKEN=...; export MLXK2_LIVE_REPO=org/model; export MLXK2_LIVE_WORKSPACE=/abs/path
#   pytest -m live_push
# - Live clone (ADR-007 Phase 1 - requires alpha features + env + same volume):
#   export MLXK2_ENABLE_ALPHA_FEATURES=1
#   export MLXK2_LIVE_CLONE=1
#   export HF_TOKEN=...
#   export MLXK2_LIVE_CLONE_MODEL="mlx-community/small-model"
#   export MLXK2_LIVE_CLONE_WORKSPACE="/path/on/same/volume/as/HF_HOME"  # APFS + same volume required
#   pytest -m live_clone
# - Live list (uses your HF_HOME; requires at least one MLX chat + one MLX base in cache):
#   export HF_HOME=/path/to/huggingface/cache
#   pytest -m live_list

# Before committing
ruff check mlxk2/ --fix && mypy mlxk2/ && pytest -v
```

Notes
- Reference environment: venv39 (Apple‑native Python 3.9) is the recommended dev base.
- Extras `[dev,test]` install ruff/mypy (code quality) and pytest/jsonschema (testing).
- For release smoke across multiple Python versions: `./test-multi-python.sh` (logs: `test_results_3_9.log`, `test_results_3_10.log`, ...).
- The macOS Python 3.9 LibreSSL warning from urllib3 is suppressed in tests via `pytest.ini`, and at runtime via package init.

## Why Local Testing?

MLX Knife tests fall into three categories for 2.0:

- **Stable CLI/JSON tests (default)**: Run on any supported Python on macOS; no model inference required; use an isolated HF cache (no network). **206 tests**
- **Alpha features (opt-in)**: Hidden alpha features like `push` and `clone` require environment variables to enable. **+21 tests**
- **Live/Inference tests (opt-in)**: Network-dependent or requiring real models/cache setup. **Various markers/env vars**

**Default test run** covers all stable 2.0 features without experimental or live dependencies.

## Test Structure

### 2.0 Test Structure

Legend
- spec/: JSON API spec/contract validation; stays in sync with docs/schema.
- live/: Opt‑in tests requiring env/config; skipped by default (markers: `live_push`, `live_clone`, `live_list`).
- Live markers: Some test files outside `live/` also have live markers (`live_stop_tokens`, `live_run`, `issue27`) and are opt-in.
- stubs/: Lightweight MLX/MLX‑LM replacements used only in unit/spec tests.
- conftest.py: Isolated HF cache (temp), safety sentinel, core fixtures/helpers.
- conftest_runner.py: Runner‑focused fixtures/mocks for generation tests.
- *.py.disabled: Intentionally disabled suites (WIP/expanded scenarios, not run).

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
│   ├── test_clone_live.py                     # Live clone flow (requires MLXK2_LIVE_CLONE, HF_TOKEN)
│   ├── test_list_human_live.py                # Live list/health against user cache (requires HF_HOME)
│   └── test_push_live.py                      # Live push flow (requires MLXK2_LIVE_PUSH, HF_TOKEN)
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

Note: Live tests are opt-in via markers (`-m live_push`, `-m live_clone`, `-m live_list`, `-m live_stop_tokens`, `-m live_run`, `-m issue27`) and environment. Default `pytest` discovery runs only the offline suite above.

### MLX/MLX‑LM Stubs (fast offline tests)
- Purpose: Unit/spec tests run platform‑neutral and without real MLX/MLX‑LM runtime.
- Mechanics: `tests_2.0/conftest.py` prepends `tests_2.0/stubs/` to `sys.path`, so `import mlx`/`mlx_lm` resolve to minimal stubs.
- Effect: Fast, deterministic tests without GPU/large RAM footprint; live/heavy path remains opt‑in.
- Production: CLI/server still use the real packages; stubs are not installed.
- **Stub Limitations:** Tests requiring real mlx-lm integration (e.g., `_get_classes` API) use `@requires_mlx_lm` marker and skip on CI. Guideline: Avoid mocks when Mock:Production LOC ratio >30% (maintenance burden > value). [Note: Details will be documented in Issue #36 closing comment]

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
      "disclaimer": "Alpha feature (upload only). No validation/filters; review results on the Hub.",
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
      "disclaimer": "Alpha feature (upload only). No validation/filters; review results on the Hub.",
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

## Pull/Preflight (Issue #30)

Goal: Gated/private/not‑found repos must not pollute the cache and should fail fast.

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

- Semantics (2.0): A new generation resets `_interrupted = False` at the start (recovery behavior). A previous Ctrl‑C does not block the next generation.
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
- Scope: OpenAI‑compatible endpoints (minimal smoke); no real models required.
- Optional for local verification; in CI currently “nice to have” (Backlog, not part of the 2.0 Guide).

## Known Warnings

- urllib3 LibreSSL notice on macOS Python 3.9
  - Message: “urllib3 v2 only supports OpenSSL 1.1.1+, currently the 'ssl' module is compiled with 'LibreSSL 2.8.3' …”
  - Status: Harmless for our usage; suppressed in production code (see `mlxk2/__init__.py`, `warnings.filterwarnings(...)`).
  - Tests: May still appear in pytest summary if third‑party dependencies import `urllib3` before our package.
  - Optional suppression in tests: add to `pytest.ini`:

    ```ini
    filterwarnings =
        ignore:urllib3 v2 only supports OpenSSL 1.1.1+
    ```
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

## 2.0 Test Strategy

MLX Knife 2.0 uses a **3-category test strategy** with enhanced isolation and sentinel protection:

### 🏠 CATEGORY 1: ISOLATED CACHE (Default Tests - ~230+ tests)
**✅ User cache stays pristine** - Tests use `isolated_cache` fixture with sentinel protection

**Current 2.0 Test Files:**
- ✅ `test_json_api_*.py` - JSON API contract validation
- ✅ `test_human_output.py` - Human output formatting
- ✅ `test_health_multifile.py` - Multi-file health completeness
- ✅ `test_push_*.py` - Push operations (offline, dry-run, workspace check)
- ✅ `test_clone_operation.py` - Clone operations with APFS optimization
- ✅ `test_run_complete.py` / `test_runner_core.py` - Run command and MLX generation
- ✅ `test_server_*_minimal.py` - Basic server API endpoints
- ✅ `spec/test_*.py` - Schema validation and spec compliance

**Technical Pattern (2.0):**
```python
def test_something(isolated_cache):
    # Test operates in complete isolation with sentinel protection
    # isolated_cache fixture ensures TEST_SENTINEL exists
    # MLX stubs enable platform-neutral testing without real MLX
    assert_is_test_cache(isolated_cache)  # Safety check
    # Test implementation here
```

**Benefits:**
- ✅ **Clean User Cache**: No test artifacts or broken models ever
- ✅ **Parallel Testing**: No cache conflicts between test runs
- ✅ **Reproducible**: No dependency on existing models in user cache
- ✅ **Platform Neutral**: MLX stubs enable testing without real MLX hardware
- ✅ **Sentinel Protection**: `TEST_SENTINEL` prevents accidental user cache modification

### 🌐 CATEGORY 2: LIVE TESTS (Network/User Cache - Opt-in)
**🔒 Require explicit environment setup** - Located in `live/` directory or marked with live markers

**Live Test Files:**
- 🔒 `live/test_push_live.py` - Real HuggingFace push operations (marker: `live_push`)
- 🔒 `live/test_clone_live.py` - APFS same-volume clone workflows (marker: `live_clone`)
- 🔒 `live/test_list_human_live.py` - Tests against user cache models (marker: `live_list`)
- 🔒 `test_stop_tokens_live.py` - Stop token validation with real models (marker: `live_stop_tokens`, ADR-009, Issue #32)
- 🔒 `test_issue_37_private_org_regression.py` - Private/org MLX model detection (marker: `live_run`, Issue #37)
- 🔒 `test_issue_27.py` - Real multi-shard model health validation (marker: `issue27`)

**Markers:** `live_push`, `live_clone`, `live_list`, `live_stop_tokens`, `live_run`, `wet` (umbrella), `issue27`

### 🖥️ CATEGORY 3: SERVER TESTS (2.0 Minimal)
**✅ Basic server functionality** - Lightweight API validation

**Server Test Files:**
- ✅ `test_server_api_minimal.py` - Basic OpenAI-compatible endpoints
- ✅ `test_server_streaming_minimal.py` - SSE streaming functionality
- ✅ `test_server_models_and_errors.py` - Model loading and error handling
- ✅ `test_server_token_limits_api.py` - Token limit enforcement

**Characteristics (2.0):**
- ✅ **Included by default** - Part of standard test suite
- 🏠 **Uses isolated cache** - Same safety as Category 1
- ⚡ **Fast execution** - Uses MLX stubs, no real model loading
- 🎯 **API compliance focus** - OpenAI compatibility validation

**Run specifically:** `pytest -k server -v` (optional, included in default anyway)

**Note:** Heavy server tests with real models documented in "Future" section above

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

Quick start (minimal)
- Best practice: set your HF cache to an external volume before pytest: `export HF_HOME=/Volumes/your-ssd/huggingface/cache`.
- Select a model: `export MLXK2_ISSUE27_MODEL="org/model"`.
  - Tip: choose an upstream repo that provides an index file (`model.safetensors.index.json` or `pytorch_model.bin.index.json`) to avoid SKIPs.
- Optional: if your cache has no index file for this repo, enable isolated index bootstrap (index‑only, no shards): `export MLXK2_BOOTSTRAP_INDEX=1`.
- Run: `pytest tests_2.0/test_issue_27.py -v`.

Notes
- Tests read from your user cache and copy a minimal subset into an isolated test cache.
- Network is only used when `MLXK2_BOOTSTRAP_INDEX=1` and the index file is not present locally.

- Set your user cache:
  - EITHER set `MLXK2_USER_HF_HOME=/absolute/path/to/your/huggingface/cache`
  - OR set `HF_HOME=/absolute/path/to/your/huggingface/cache` before running pytest — the test harness preserves this original value and exposes it to the Issue #27 helpers while still isolating `HF_HOME` for the code under test.
- Select a specific upstream model that includes an index file (strongly recommended):
  - `export MLXK2_ISSUE27_MODEL="mistralai/Mixtral-8x7B-Instruct-v0.1"`
  - or another upstream PyTorch repo that contains `model.safetensors.index.json` or `pytorch_model.bin.index.json`.
  - Note: Many `mlx-community/...` conversions do not ship the upstream safetensors index; prefer the original upstream repo to avoid SKIPs.
- Minimize copy size (optional):
  - `export MLXK2_SUBSET_COUNT=1`  (Default 1; erhöht ggf. Shard‑Anzahl)
  - `export MLXK2_MIN_FREE_MB=512` (Default 512 MB Sicherheitsmarge)
- Run the focused tests: `PYTHONPATH=. pytest tests_2.0/test_issue_27.py -v`

Optional bootstrap (opt-in, minimal workflow):
- Minimal preconditions to run all Issue #27 tests without SKIPs:
  - Select models to test:
    - Healthy check model (read-only): `export MLXK2_ISSUE27_MODEL="org/model"` (should be present and healthy in your user cache; single-shard small models are ideal, e.g., `sshleifer/tiny-gpt2`).
    - Index tests model (optional, can be different): `export MLXK2_ISSUE27_INDEX_MODEL="org/model-with-index"` (upstream repo that lists an index; not required to be fully downloaded locally).
- Ensure your user cache root is set via `MLXK2_USER_HF_HOME` (or provide it via `HF_HOME` before pytest; the harness maps it across).
  - Enable index bootstrap: `export MLXK2_BOOTSTRAP_INDEX=1` (fetches only index files into the ISOLATED test cache; never modifies your user cache).
  - Then: `pytest tests_2.0/test_issue_27.py -v`
  - Note: Network is only needed if your user cache does not already contain an index file for the chosen repo. If the index exists in your cache, the tests copy it into the isolated cache and no network is required.

If you still see SKIPs:
- “No safetensors index found” → The chosen model snapshot lacks an index file. Pick a model that has `model.safetensors.index.json` (or `pytorch_model.bin.index.json`).
- “Not enough free space” → Free disk space; tests create a subset copy into an isolated temp cache.
- “User model not found” → Verify your model exists in the user cache and `MLXK2_USER_HF_HOME` points to the `.../huggingface/cache` root.

Quick helper to list index‑bearing models in your user cache:

```bash
find "$MLXK2_USER_HF_HOME/hub" -type f \
  \( -name 'model.safetensors.index.json' -o -name 'pytorch_model.bin.index.json' \) \
| sed 's#.*/hub/models--\(.*\)/snapshots/.*#\1#; s#--#/#g' | sort -u
```

With a suitable model (i.e., one that includes an upstream safetensors index) present and `MLXK2_USER_HF_HOME` set, the Issue #27 tests should run without SKIPs.

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

- Resource considerations
- Disk: tests copy a minimal subset of files into an isolated cache (index + 1 smallest shard, oder 1 Pattern‑Shard). Optional Tuning:
  - `export MLXK2_SUBSET_COUNT="1"` (Default 1; erhöhe bei Bedarf)
  - `export MLXK2_MIN_FREE_MB="512"` (Default 512 MB; erhöhe bei knappem Platz)
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

### Optional Setup (Live Stop Tokens - ADR-009)

For stop token validation tests (`@pytest.mark.live_stop_tokens` - **excluded by default**, requires `-m live_stop_tokens`):

**Option A: Portfolio Discovery (recommended)**
```bash
# Set HF_HOME to discover all MLX chat models in your cache
export HF_HOME=/path/to/your/huggingface/cache
pytest -m live_stop_tokens -v
```
- Auto-discovers all MLX chat models (filter: MLX + healthy + runtime_compatible + chat)
- RAM-aware skipping (progressive budgets 40-70%)
- Generates empirical report: `stop_token_config_report.json`

**Option B: Hardcoded Fallback (3 models)**
```bash
# Ensure these 3 models exist in your HuggingFace cache:
mlxk pull mlx-community/gpt-oss-20b-MXFP4-Q8         # ~12GB RAM
mlxk pull mlx-community/Qwen2.5-0.5B-Instruct-4bit   # ~1GB RAM
mlxk pull mlx-community/Llama-3.2-3B-Instruct-4bit   # ~4GB RAM

# Run tests (uses default cache if HF_HOME not set)
pytest -m live_stop_tokens -v
```

**Note**: These tests are marker-required (🔒) and excluded from default `pytest` runs. Use `-m live_stop_tokens` to run.

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

### Verification Results (November 2025)

**✅ 306/306 tests passing** - All standard tests validated on Apple Silicon with enhanced isolation

| Python Version | Status | Tests Passing | Skipped |
|----------------|--------|---------------|---------|
| 3.9.6 (macOS)  | ✅ Verified | 306/306 | 20 |
| 3.10.x         | ✅ Verified | 306/306 | 20 |
| 3.11.x         | ✅ Verified | 306/306 | 20 |
| 3.12.x         | ✅ Verified | 306/306 | 20 |
| 3.13.x         | ✅ Verified | 306/306 | 20 |

**Note:** 20 skipped tests are opt-in (live tests, alpha features). Skipped count may vary by environment:
- Without `HF_TOKEN`: +1 skip (live push test)
- Without `MLXK2_ENABLE_ALPHA_FEATURES=1`: +3 skips (alpha feature tests)
- Without `jsonschema`: +1 skip (spec validation test)

All versions tested with `isolated_cache` system and MLX stubs for platform-neutral execution.

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
| Push offline | `pytest -k push -v` | — | Push offline tests (tests alpha feature: `--check-only`, `--dry-run`, error handling); no network, no credentials needed | No |
| ⏭️ Live Push | `MLXK2_ENABLE_ALPHA_FEATURES=1 pytest -m live_push -v` | `live_push` (subset of `wet`) + Env: `MLXK2_ENABLE_ALPHA_FEATURES=1`, `MLXK2_LIVE_PUSH=1`, `HF_TOKEN`, `MLXK2_LIVE_REPO`, `MLXK2_LIVE_WORKSPACE` | JSON push against the real Hub; on errors the test SKIPs (diagnostic) | Yes |
| ⏭️ Live List | `pytest -m live_list -v` | `live_list` (subset of `wet`) + Env: `HF_HOME` (user cache with models) | Tests list/health against user cache models | No (uses local cache) |
| Clone offline | `pytest -k clone -v` | — | Clone offline tests (tests alpha feature: APFS validation, temp cache, CoW workflow); no network needed | No |
| ⏭️ Live Clone (ADR-007) | `MLXK2_ENABLE_ALPHA_FEATURES=1 pytest -m live_clone -v` | `live_clone` + Env: `MLXK2_ENABLE_ALPHA_FEATURES=1`, `MLXK2_LIVE_CLONE=1`, `HF_TOKEN`, `MLXK2_LIVE_CLONE_MODEL`, `MLXK2_LIVE_CLONE_WORKSPACE` | Real clone workflow: pull→temp cache→APFS same-volume clone→workspace (ADR-007 Phase 1 constraints: same volume + APFS required) | Yes |
| 🔒 Live Stop Tokens (ADR-009) | `pytest -m live_stop_tokens -v` | `live_stop_tokens` (required); Optional: `HF_HOME` (enables portfolio discovery) | Issue #32: Validates stop token behavior with real models. **With HF_HOME:** Portfolio Discovery auto-discovers all MLX chat models (filter: MLX+healthy+runtime+chat), RAM-aware skip, empirical report. **Without HF_HOME:** Uses 3 predefined models (see "Optional Setup" section for model requirements). | No (uses local cache) |
| ⏭️ Live Run | `pytest -m live_run -v` | `live_run` + Env: `MLXK2_USER_HF_HOME` or `HF_HOME` (user cache with `mlx-community/Phi-3-mini-4k-instruct-4bit`) | Regression tests for Issue #37: Validates private/org MLX model framework detection in run command (renames Phi-3 to simulate private-org model) | No (uses local cache) |
| ⏭️ Issue #27 real‑model | `pytest -m issue27 tests_2.0/test_issue_27.py -v` | Marker: `issue27`; Env (required): `MLXK2_USER_HF_HOME` or `HF_HOME` (user cache, read‑only). Env (optional): `MLXK2_ISSUE27_MODEL`, `MLXK2_ISSUE27_INDEX_MODEL`, `MLXK2_SUBSET_COUNT=0`. | Copies real models from user cache into isolated test cache; validates strict health policy on index‑based models (no network) | No (uses local cache) |
| Server tests (included) | `pytest -k server -v` | — | Basic server API tests (minimal, uses MLX stubs) | No |

**Legend:**
- No symbol: Runs with `pytest -v` (default suite)
- ⏭️ Skip-unless-env: Collected by `pytest -v` but skipped without required environment variables
- 🔒 Marker-required: Skipped by `pytest -v`; requires explicit `-m marker` to run

Useful commands
- Only Spec: `pytest -m spec -v`
- Push tests (offline): `pytest -k "push and not live" -v`
- Clone tests (offline): `pytest -k "clone and not live" -v`
- Exclude Spec: `pytest -m "not spec" -v`
- Live Push only: `MLXK2_ENABLE_ALPHA_FEATURES=1 MLXK2_LIVE_PUSH=1 HF_TOKEN=... MLXK2_LIVE_REPO=... MLXK2_LIVE_WORKSPACE=... pytest -m live_push -v`
- Live Clone only: `MLXK2_ENABLE_ALPHA_FEATURES=1 MLXK2_LIVE_CLONE=1 HF_TOKEN=... MLXK2_LIVE_CLONE_MODEL=... MLXK2_LIVE_CLONE_WORKSPACE=... pytest -m live_clone -v`
- Live List only: `HF_HOME=/path/to/user/cache pytest -m live_list -v`
- Live Stop Tokens only (ADR-009): `pytest -m live_stop_tokens -v` (optional: `HF_HOME=/path/to/user/cache` for portfolio discovery; otherwise uses 3 hardcoded test models)
- Live Run only: `HF_HOME=/path/to/user/cache pytest -m live_run -v` (requires `mlx-community/Phi-3-mini-4k-instruct-4bit` in cache)
- Issue #27 only: `MLXK2_USER_HF_HOME=/path/to/user/cache pytest -m issue27 tests_2.0/test_issue_27.py -v`
- All live tests (umbrella): `MLXK2_ENABLE_ALPHA_FEATURES=1 pytest -m wet -v` (includes live_push, live_clone, live_list)

Markers: wet vs specific live tests
- `wet`: umbrella marker for any "live" test that may require network, credentials, or user environment. Use to run all live tests.
- `live_push`: narrow marker for push‑specific live tests only. Use to target push live checks without running other live suites.
- `live_clone`: narrow marker for clone‑specific live tests only. Use to target ADR-007 Phase 1 real workflow validation.
- `live_stop_tokens`: narrow marker for stop token validation tests with real models (ADR-009). Use to validate Issue #32 fix (multi-EOS models). **Marker-required (🔒):** Must use `-m live_stop_tokens` to run.
- `live_run`: narrow marker for run command tests with real models. Use to validate Issue #37 framework detection regression fix (private/org MLX models).

Note: ⏭️ tests are collected by default but skip without required env vars. 🔒 tests require explicit markers to run.

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

---

## Real-Model Testing (Implemented)

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
- ✅ **Portfolio Discovery:** Scans `HF_HOME/hub/models--*/` for all qualifying models
- ✅ **RAM-Aware:** Progressive budgets prevent OOM (40%-70% of system RAM)
- ✅ **Empirical Report:** Generates `stop_token_config_report.json` with findings
- ✅ **Fallback:** Uses 3 predefined models (MXFP4, Qwen, Llama) if HF_HOME not set - models must exist in HF cache

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

---

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

2. **Test results summary (2.0)** (example format):
  ```
  Platform: macOS 14.5, M2 Pro
  Python: 3.9.6
  Results: 306 passed, 20 skipped
  ```

3. **Any issues encountered** and how you resolved them

## Summary

**MLX Knife 2.0 Testing Status:**

✅ **Feature Complete** - 300+ tests passing, 20 skipped (2.0.1 Stable)
✅ **Enhanced Isolation** - Sentinel protection with `isolated_cache` fixture
✅ **3-Category Strategy** - Isolated/Live/Server tests optimized for 2.0
✅ **Multi-Python Support** - Python 3.9-3.13 verified
✅ **Platform Neutral** - MLX stubs enable testing without real MLX hardware
✅ **Alpha Feature Separation** - Clean boundaries for beta/alpha functionality
✅ **JSON API Validation** - Complete schema compliance testing
✅ **Clone Implementation** - Full ADR-007 Phase 1 validation (APFS optimization)
✅ **Push Operations** - Comprehensive offline testing (dry-run, workspace check)

This testing framework validates MLX Knife 2.0's JSON-first architecture through comprehensive isolated testing with minimal live dependencies.

## Future: Server E2E Testing (TODO, ADR-011)

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

*MLX-Knife 2.0.1*
