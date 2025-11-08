# Changelog

## 2.0.1 — 2025-11-28

**Bug Fix & Enhancement Release**: CLI exit code propagation fixes + Portfolio Discovery for stop token validation.

### Fixed

- **CLI `run` command exit codes** (GitHub Issue #38): `mlxk run` now correctly returns exit code 1 when model execution fails, enabling proper error detection in shell scripts and automation workflows
  - **Both modes fixed**: Text mode and JSON mode now properly propagate errors
  - **JSON mode**: Returns `{"status": "error", "error": {...}}` with exit code 1
  - **Text mode**: Prints `"Error: ..."` message and returns exit code 1
  - **Affects**: Shell scripts using `mlxk run && next_step`, batch processing, model validation workflows
  - **Root cause**: `run_model()` returned `None` in text mode instead of error strings; CLI had no way to detect text-mode failures
  - **Fix**:
    - Modified `mlxk2/operations/run.py` to return `"Error: ..."` strings in both modes (lines 50-86, 125-129)
    - CLI error detection in `mlxk2/cli.py:273-288` now catches errors for both modes
  - **Examples of fixed scenarios**:
    - Nonexistent model: `mlxk run bad-model "hi"` → exit 1 (was: exit 0)
    - Incompatible model: Runtime version mismatch → exit 1 (was: exit 0)
    - Runtime exceptions: OOM, loading failures → exit 1 (was: exit 0)

- **Stop token validation Portfolio Discovery** (GitHub Issue #32, ADR-009): Live stop token tests now support dynamic model discovery and HF_HOME-optional testing
  - **Portfolio Discovery**: Auto-discovers all MLX chat models in `HF_HOME` cache (filter: MLX + healthy + runtime_compatible + chat)
  - **RAM-Aware Testing**: Progressive RAM budgets (40-70%) prevent OOM during multi-model validation
  - **Empirical Reporting**: Generates `stop_token_config_report.json` with cross-model stop token findings
  - **Fallback Support**: Tests work without `HF_HOME` using 3 predefined models (MXFP4, Qwen, Llama)
  - **Marker-Required**: Tests excluded from default suite, use `pytest -m live_stop_tokens` to run
  - **Implementation**: `tests_2.0/test_stop_tokens_live.py` (~110 LOC for discovery + RAM gating)

### Testing

- 306 passed, 20 skipped (including 4 live stop token tests, marker-required)
- **New test files**:
  - `tests_2.0/test_cli_run_exit_codes.py` validates both text and JSON mode exit codes (+9 tests)
  - `tests_2.0/test_stop_tokens_live.py` implements Portfolio Discovery with HF_HOME-optional fallback (+4 tests)
- **Updated tests**: `tests_2.0/test_run_complete.py` reflects new error contract
- Zero regressions in full test suite

---

## 2.0.0 — 2025-11-06

**Stable Release**: MLX Knife 2.0 replaces 1.x as the primary version. Full feature parity with 1.1.1 achieved plus major enhancements.

### License Change

- **MIT → Apache License 2.0**: Better patent protection, industry-standard licensing
- See [MIGRATION.md](MIGRATION.md) for details on license change and user impact

### Highlights

- **Full 1.x Feature Parity**: All commands from 1.1.1 available (`list`, `show`, `pull`, `rm`, `run`, `server`, `health`)
- **JSON API**: Machine-readable output for automation (`--json` flag on all commands)
- **Enhanced Error Handling**: Structured errors with request IDs, logging levels, JSON logs
- **Runtime Compatibility Checks**: Pre-flight validation prevents loading incompatible models
- **Improved Stop Token Detection**: Multi-EOS support (MXFP4, Qwen, Llama)
- **Better Human Output**: Improved formatting, relative timestamps, runtime status

### Package Changes

- **Package name**: `mlx-knife` (unchanged from 1.x)
- **Primary command**: `mlxk` (replaces `mlxk2` from beta)
- **Aliases**: `mlxk-json`, `mlxk2` (backwards compatibility)

### Breaking Changes

- **Lock file handling**: `mlxk rm` requires `--force` flag when models have active locks (safety improvement)
- See [MIGRATION.md](MIGRATION.md) for complete migration guide from 1.x

### Installation

```bash
# PyPI (recommended)
pip install mlx-knife

# GitHub release
pip install https://github.com/mzau/mlx-knife/releases/download/v2.0.0/mlx_knife-2.0.0-py3-none-any.whl

# Upgrade from 1.x
pip install --upgrade mlx-knife
```

### Testing

- 297 passed, 20 skipped (317 total tests)
- Python 3.9-3.13 compatibility verified
- Apple Silicon (M1/M2) tested

---

## 2.0.0-beta.6 — 2025-10-22

### Fixed
- **Stop token detection for multi-EOS models** (Issue #32, ADR-009): MXFP4 and Qwen models no longer generate visible stop tokens (`<|end|>`) or chat template markers in output
- **Private/org MLX model detection** (Issue #37): `mlxk run` now correctly detects MLX models outside `mlx-community/*` namespace
- **Commit-pinned compatibility checks**: Models with `@commit_hash` syntax now correctly validated before inference
- **Packaging dependencies** (P0): `pip install -e .` now installs all required dependencies (`mlx-lm`, `mlx`, `fastapi`, etc.) via `pyproject.toml`

### Documentation
- Simplified installation instructions in README.md and TESTING.md (consistent `pip install -e ".[dev,test]"` recommendation)

### Testing
- 297 passed, 20 skipped (317 total)
- Added 6 new tests: 4 stop token validation tests (opt-in), 2 compatibility check tests

## 2.0.0-beta.5 — 2025-10-20

**Enhanced Error Handling & Logging (ADR-004)**: Unified error envelope, structured logging with JSON support, and request correlation.

**Legacy Model Format Detection**: Models with outdated weight file formats are detected and marked as runtime-incompatible (Issue #37).

### Added

- **Error envelope and structured logging** (ADR-004 Phase 1):
  - Unified error envelope for CLI/Server: `{"status": "error", "error": {"type", "message", "detail", "retryable"}, "request_id"}`
  - Request correlation via `request_id` (UUID4) in all server responses and logs
  - HTTP status mapping: 400 (validation), 403 (access denied), 404 (not found), 500 (internal), 503 (shutdown)
  - Structured logging with INFO/WARN/ERROR/DEBUG levels (replaces ad-hoc print statements)
  - Optional JSON logs via `MLXK2_LOG_JSON=1` for machine-readable output
  - **Log-level control**: `--log-level` (debug/info/warning/error) controls MLXKLogger, root logger, and Uvicorn access logs
  - **`--log-json` CLI flag**: User-friendly alternative to `MLXK2_LOG_JSON=1` environment variable
  - **Uvicorn JSON formatting**: Access logs (`GET /v1/models`, etc.) also formatted as JSON when `--log-json` is used
  - **Root logger JSON formatting**: External libraries (mlx-lm, transformers) also log as JSON in JSON mode
  - Automatic redaction of sensitive data (HF tokens, user paths)
  - Error rate limiting (max 1 error per 5s for duplicate errors)
  - New modules: `mlxk2/errors.py`, `mlxk2/logging.py`, `mlxk2/context.py`
  - FastAPI middleware: Request ID injection, custom exception handler
  - **User documentation**: README.md "Logging & Debugging" section (log levels, JSON format, redaction examples)
  - Test coverage: 22 new tests in `test_adr004_error_logging.py`

- **Legacy format detection in runtime compatibility check** (Issue #37):
  - Gate 2 in `check_runtime_compatibility()`: Validates weight file naming conventions
  - Detects legacy patterns: `weights.*.safetensors` (e.g., `weights.00.safetensors`), `pytorch_model-*.safetensors`
  - Accepts modern patterns: `model.safetensors`, `model-XXXXX-of-YYYYY.safetensors`
  - Clear error message: `"Legacy format not supported by mlx-lm"`
- **Pre-flight check in `run` command**:
  - Validates runtime compatibility before attempting model load
  - Prevents cryptic mlx-lm errors: `"ERROR:root:No safetensors found in..."`
  - Returns user-friendly error: `"Model 'X' is not compatible: Legacy format not supported by mlx-lm"`
  - Best-effort check: gracefully skips if model not in cache (preserves test compatibility)

### Changed
- **Runtime compatibility validation extended**:
  - Gate 1: Framework check (MLX vs GGUF/PyTorch) - from Beta.4
  - Gate 2: **NEW** - Weight file format check (modern vs legacy patterns)
  - Gate 3: Model type support check (mlx-lm compatibility) - from Beta.4
- **CLI description**: "HuggingFace model management for MLX" (removed "JSON-first" and version number)
- **README reorganization**: Better section flow, merged duplicate sections, removed beta-specific content (550 lines)

### Fixed
- **Legacy format detection** (Issue #37, bug):
  - Models with legacy weight file formats (`weights.*.safetensors`, `pytorch_model-*.safetensors`) now correctly detected as runtime-incompatible
  - Health output: `healthy` (file integrity OK) but `runtime_compatible: false`
  - `reason` field describes incompatibility: `"Legacy format not supported by mlx-lm"`
  - Human output: `healthy*` in compact mode, `healthy | no | Legacy format...` in verbose mode
  - Pre-flight check in `run` command prevents cryptic mlx-lm errors
- **CLI error handling** (regression since 19a6667): Running `mlxk2` without arguments now shows help text (like git/docker) instead of JSON error, `--json` flag properly respected for automation
- **Code quality**: Removed 7 unused imports, ruff checks pass

### Implementation
- `mlxk2/operations/health.py`:
  - `check_runtime_compatibility()` Gate 2 implementation (lines 272-304)
  - Regex patterns for legacy format detection
  - Mixed legacy/modern: prefers modern if both present
- `mlxk2/operations/run.py`:
  - Pre-flight runtime compatibility check (lines 45-89)
  - Clear error messages before mlx-lm loading

### Testing
- **Current Status**: 293 passed, 14 skipped, 1 warning (urllib3/LibreSSL)
- **New Tests** (25 total):
  - `tests_2.0/test_adr004_error_logging.py` (22 tests):
    - Error envelope structure and serialization
    - Error type to HTTP status mapping (8 error types validated)
    - Request ID generation and propagation (UUID4 validation, context nesting)
    - Log redaction (HF tokens, home directory paths)
    - Structured logging (plain text vs JSON modes, log levels, rate limiting)
  - `tests_2.0/test_legacy_formats.py` (3 tests):
    - `test_weights_numeric_safetensors_is_runtime_incompatible`: Validates `weights.00.safetensors` detection
    - `test_pytorch_model_numeric_safetensors_is_runtime_incompatible`: Validates `pytorch_model-*.safetensors` detection
    - `test_modern_model_safetensors_passes_legacy_gate`: Ensures modern formats are not rejected
- **Regression**: All existing tests pass (zero breaking changes)

### Known Issues
- **Missing tests for Issue #36** (Beta.4 gap):
  - No dedicated tests for Gate 1 (framework check)
  - No dedicated tests for Gate 3 (model_type support)
  - Runtime compatibility tested indirectly via Issue #37 tests and schema validation
  - TODO: Add explicit tests for Beta.4 runtime compatibility feature

### User Experience Example
```bash
# Before (Beta.4): Cryptic mlx-lm error
$ mlxk2 run TinyLlama-1.1B-Chat-v1.0-4bit "Hello"
ERROR:root:No safetensors found in /Volumes/.../snapshots/01a7088...

# After (Beta.5): Clear error message
$ mlxk2 run TinyLlama-1.1B-Chat-v1.0-4bit "Hello"
Error: Model 'mlx-community/TinyLlama-1.1B-Chat-v1.0-4bit' is not compatible: Legacy format not supported by mlx-lm

# Health status shows details
$ mlxk2 show TinyLlama-1.1B-Chat-v1.0-4bit
Health: healthy (files OK, runtime incompatible)
Reason: Legacy format not supported by mlx-lm
```

### Notes
- Legacy models are file-complete (healthy integrity) but use outdated naming conventions incompatible with modern mlx-lm
- Pre-flight check improves UX by catching incompatibility before expensive model loading

---

## 2.0.0-beta.4 — 2025-10-18

**Health Check Enhancement**: Separate integrity and runtime compatibility validation (Issue #36).

### Changed
- **JSON API 0.1.5 specification**:
  - Added `runtime_compatible: boolean` field to `modelObject` (always present)
  - Added `reason: string | null` field to `modelObject` (describes first problem found)
  - `list`/`show` JSON output performs both integrity and runtime compatibility checks
  - Gate logic: Runtime check requires integrity check first; `reason` shows first problem (integrity > runtime priority)
- **Health check concepts documented**:
  - Integrity Check (`health` field): File-level validation (required files, no LFS pointers, valid JSON)
  - Runtime Compatibility Check (`runtime_compatible` field): MLX framework + architecture validation with mlx-lm
  - Framework detection: GGUF/PyTorch models marked as runtime-incompatible
  - Architecture detection: Unsupported model types (e.g., `qwen3_next` with mlx-lm < 0.28.0) detected
  - Respects `MODEL_REMAPPING` for aliased architectures (e.g., `mistral` → `llama`)

### Implementation Status
- ✅ **Phase 1 Complete**: JSON API Specification 0.1.5
  - `docs/json-api-schema.json` updated with new fields
  - `docs/json-api-specification.md` extended with health check concepts and examples
- ✅ **Phase 2 Complete**: JSON Implementation
  - `mlxk2/spec.py` bumped to 0.1.5
  - `mlxk2/operations/health.py`: `check_runtime_compatibility()` with gate logic
  - `mlxk2/operations/common.py`: `build_model_object()` always computes `runtime_compatible` + `reason`
  - mlx-lm API compatibility: Supports both 0.27.x (`mlx_lm.utils._get_classes`) and 0.28.x APIs
  - Log suppression: mlx-lm ERROR logs redirected to `reason` field only
- ✅ **Phase 3 Complete**: Human Output Specification
  - Compact mode: `healthy` / `healthy*` / `unhealthy` (single column)
  - Verbose mode: "Integrity" | "Runtime" | "Reason" (split columns)
  - ASCII-only output (no UTF-8 symbols for parsing compatibility)
  - README.md fully documented with examples and design philosophy
  - JSON examples verified for consistency with schema and code
- ✅ **Phase 4 Complete**: Human Output Implementation in `mlxk2/output/human.py`

### Dependencies
- **mlx-lm requirement updated**: `>=0.27.0` → `>=0.28.3`
  - Now uses official mlx-lm 0.28.3 release with Python 3.9 compatibility fixes for `qwen3_next`
  - Adds support for newer architectures (Klear, qwen3_next, etc.)
  - Git pin removed in favor of stable PyPI release

### Validation
- ✅ All 256 tests pass (9 skipped)
- ✅ Runtime compatibility correctly detects:
  - GGUF/PyTorch models → `runtime_compatible: false` (framework mismatch)
  - Supported MLX models → `runtime_compatible: true`
  - Unsupported architectures → `runtime_compatible: false` with descriptive `reason`
  - Klear-46B verified working with mlx-lm 0.28.2


### Notes
- Human output columns controlled by CLI flags (documentation in README.md, separate from JSON spec)
- This addresses the root cause discovered in Issue #36: GGUF models show "healthy" but are not executable with mlx-lm

## 2.0.0-beta.3 — 2025-09-18

**Feature Complete**: Full 1.1.1 parity achieved with Clone implementation (ADR-007 Phase 1) and APFS filesystem detection fixes.

### Added
- **Clone command implementation** (MAJOR):
  - Complete `mlxk2 clone` with ADR-007 Phase 1: Same-Volume APFS strategy
  - APFS Copy-on-Write optimization for instant cloning
  - Isolated temp cache with user cache safety
  - Health check integration via `health_from_cache`
  - Feature-gated behind `MLXK2_ENABLE_ALPHA_FEATURES=1`
- **JSON API 0.1.4 specification**:
  - Clone operation schema and documentation
  - Complete schema validation coverage for all 10 JSON commands
  - Schema tests for `list`, `show`, `health`, `pull`, `rm`, `clone`, `version`, `push`, `run`, `server`

### Fixed
- **APFS filesystem detection**: SMB/network mounts now correctly detected as Non-APFS
- **Push APFS warnings**: Non-APFS cache setups now display filesystem warnings

### Testing
- **Comprehensive test coverage**: 254/254 tests passing, 11 skipped
- **Clone operation tests**: 43 tests covering APFS, volume detection, health integration
- **Live validation**: 3 live clone + push tests with real HuggingFace models

## 2.0.0-beta.3-local — 2025-09-14

**Feature Complete Beta**: 1.x parity achieved. All core functionality implemented with clean experimental separation.

### Added
- **Run command implementation** (MAJOR):
  - Complete `mlxk2 run` with interactive and single-shot modes
  - Streaming and batch generation with parameter controls (`--temperature`, `--top-p`, `--max-tokens`)
  - Chat template integration and conversation history tracking
  - Interrupt handling (Ctrl-C) with graceful recovery and session reset
  - Enhanced run with future features (system prompts, reasoning model support)
- **MLXRunner core engine** (ported from 1.x):
  - `mlxk2.core.runner` package with modular architecture
  - Dynamic token limits (full context for run, half-context for server)
  - Stop token filtering and reasoning model detection
  - Thread-safe model loading, memory management, and cleanup
- **Server implementation**:
  - OpenAI-compatible endpoints (`/v1/completions`, `/v1/chat/completions`, `/v1/models`, `/health`)
  - SSE streaming with SIGINT-robust supervisor mode (deterministic shutdown/restart)
  - Model hot-swapping and thread-safe memory management
  - Half-context token limits for DoS protection
- **Experimental feature separation**:
  - Push command hidden behind `MLXK2_ENABLE_EXPERIMENTAL_PUSH=1` environment variable
  - Clean beta/experimental boundaries for stable release classification

### Changed
- **Feature status**: All core commands now complete
  - README/docs updated: Run status "Pending" → "Complete"
  - Feature parity with 1.x stable releases achieved
  - Stable version reference updated to 1.1.1
- **Test architecture**:
  - Default suite: **184 passed, 30 skipped** (stable features only)
  - Experimental: **205 passed, 9 skipped** (with `MLXK2_ENABLE_EXPERIMENTAL_PUSH=1`)
  - Clean separation ensures beta testing covers stable features only
- **Runner architecture**:
  - Modular design with focused helpers: `token_limits.py`, `chat_format.py`, `reasoning_format.py`, `stop_tokens.py`
  - API compatibility preserved for existing integrations and test patches

### Fixed
- **Pull operation cache pollution (Issue #30)**:
  - Added preflight access check with `preflight_repo_access()` to validate repository accessibility
  - Prevents cache pollution from attempting downloads of gated/private/missing repos
  - Surfaces clear "Access denied" guidance with `HF_TOKEN` hints before any download
  - Robust error handling across different `huggingface_hub` versions
- **Test stability**:
  - Pull network timeout test fixed for environments without `HF_TOKEN`
  - All push tests now properly gated behind environment variable (no unexpected failures)
  - Default test runs require no external dependencies or credentials
- **Documentation accuracy**:
  - Feature status corrected across README/TESTING to reflect actual implementation
  - Test count documentation updated to reflect stable vs experimental separation

### Implementation Milestones
- **Complete 1.x parity**: All core functionality (list, health, show, pull, rm, run, serve) fully implemented
- **Production ready**: Comprehensive testing across Python 3.9-3.13 with isolated cache system
- **Clean architecture**: Experimental features properly isolated, beta definition clarified
- **GitHub issues resolved**: Run implementation, interactive mode, streaming support, feature parity

### Tests & Docs
- **Comprehensive test coverage**: 31+ tests for run command (interactive, parameters, error handling)
- **TESTING.md**: Clear guidance on stable (184) vs experimental (+21) test runs
- **Multi-Python verification**: All tests passing across supported Python versions
- **Skip breakdown documented**: 21 push tests, 1 live test, 8 other opt-in tests

### Notes
- 2.0.0-beta.3 represents **complete feature parity** with 1.x stable releases
- Ready for production use as comprehensive 1.x alternative
- Experimental features cleanly separated for future development

## 2.0.0-alpha.3 — 2025-09-08

Port Issue #31 (lenient MLX detection) to 2.0; refine human list behavior.

Hard split: 1.x code and tests have been removed from this branch to avoid confusion and license duality. Use the `main` branch for 1.x (MIT).

### Added
- Detection helpers (README front‑matter + tokenizer):
  - Framework=MLX when README front‑matter `tags` includes `mlx` or `library_name: mlx`, in addition to `mlx-community/*`.
  - Type=chat when tokenizer has `chat_template`, or name hints (`instruct`/`chat`), or `config.model_type == 'chat'`.
  - Unified `build_model_object(...)` used by `list` and `show` to ensure consistent fields.
- Tests:
  - Offline: front‑matter and tokenizer detection for both `list` and `show`.
  - Human output: verifies default/verbose/all filtering semantics.
  - Live (opt-in): `tests_2.0/live/test_list_human_live.py` checks human list variants against a real HF cache (marker `-m live_list`).
  - Push (offline): branch-missing tolerance and retry on "Invalid rev id" with `--create`.

### Changed
- Human list (default): shows only MLX chat models (safer for run/server selection).
- Human list `--verbose`: shows all MLX models (chat + base).
- Human list `--all`: shows all frameworks (MLX, GGUF, PyTorch).
- `show` uses the same detection helpers as `list`; respects `HF_HOME` via `get_current_model_cache()`.

### Docs
- SECURITY.md: clarified experimental push scope and network behavior (explicit only; no background traffic).
- README.md: added “Privacy & Network” bullet; updated version strings to alpha.3.
 - README.md: noted hard split — 1.x lives on `main` (MIT), this branch is 2.x (Apache‑2.0).

### Notes
- No JSON API schema changes; spec remains 0.1.3.
 
### Fixed
- Push: tolerate missing target branches; with `--create`, proactively create the branch and retry the upload once. No‑op uploads still create the branch when `--create` is provided.

## [1.1.1-beta.2] - 2025-09-06

### Feature: Lenient MLX Detection for Private Repos (Issue #31)
- Problem: `run` only accepted `mlx-community/*` models; private/cloned MLX repos (in MLX format) appeared as "PyTorch | base" and were rejected.
- Solution: Added README/tokenizer-based detection to recognize MLX/chat models outside `mlx-community`.
- Details:
  - Tokenizer: If `tokenizer_config.json` contains a non-empty `chat_template` → Type = `chat` (highest priority).
  - README front matter (YAML, lenient parse):
    - `tags` contains `mlx` OR `library_name: mlx` → Framework = `MLX`.
    - `pipeline_tag: text-generation` OR `tags` contain `chat`/`instruct` → Type = `chat`.
    - `pipeline_tag: sentence-similarity` OR `tags` contain `embedding` → Type = `embedding`.
  - Fallback unchanged: `.gguf` → `GGUF`; else `safetensors/bin` → `PyTorch`; else `Unknown`. Type fallback by name substrings (`instruct/chat` → chat; `embed` → embedding; else base).

### CLI Behavior (Schema Unchanged)
- `mlxk show` now displays `Type: <chat|embedding|base>` when detected.
- `mlxk list --all` includes a `TYPE` column; default `mlxk list` now shows chat-capable MLX models only (strict view).
- `mlxk run` now accepts MLX repos identified via README (not only `mlx-community/*`).

### Implementation
- New helper: `mlx_knife/model_card.py` (no deps) to read README front matter and tokenizer hints; fully fail-safe.
- Updated detection in `mlx_knife/cache_utils.py`:
  - `detect_framework(...)` consults README hints before file-type fallback.
  - New `detect_model_type(...)` implements priority order.
  - `run_model(...)` imports runner module for easier test monkeypatching.

### Tests
- Added unit tests: `tests/unit/test_model_card_detection.py`.
- Server test stability and safety improvements:
  - RAM-aware model gating now combines size-token heuristics with `mlxk show` data (disk size + quantization) for more reliable estimates.
  - Fixed MoE size parsing (prefers tokens like `8x7B` over partial `7B` matches).
  - Robust server process guard ensures clean shutdown on Ctrl-C/SIGTERM (prevents orphaned Python processes using excessive memory).
  - Configurable safety/estimation factors via environment variables (see TESTING.md).
- All tests passing locally on Apple Silicon across Python 3.9–3.13: 166/166.

Note: GitHub tag/version uses `1.1.1-beta.2`. PyPI release uses PEP 440 `1.1.1b2`.

## 2.0.0-alpha.2 — 2025-09-05

Experimental `push` (upload only) and documentation/testing refinements.

### Added
- `push` (experimental, M0): Upload a local folder to Hugging Face using `upload_folder`.
  - Safety: `--private` required in alpha.
  - Quiet JSON: With `--json` (without `--verbose`) suppress progress bars/console logs; hub logs are captured in `data.hf_logs`.
  - No-op detection: Prefer hub signal (“No files have been modified… Skipping…”). Sets `no_changes: true`, clears `commit_sha/commit_url`, and sets `uploaded_files_count: 0`.
  - Offline preflight: `--check-only` analyzes the local workspace and returns `data.workspace_health` (index/weights/LFS/partials) without network.
  - Dry-run planning: `--dry-run` computes a plan vs remote (uses `list_repo_files`), returns `dry_run: true`, `dry_run_summary {added, modified:null, deleted}`, and sample `added_files`/`deleted_files` (up to 20). Honors default ignores and merges `.hfignore`.
  - Uploaded file count: Remains `null` when hub does not return per-file operations; no heuristic guessing.

### Docs
- TESTING.md: Added “Reference: Push CLI and JSON”, `--dry-run` examples, and a mini matrix (default vs markers/opt-in).
- CLAUDE.md: Updated Current Focus/Decisions + session summary for push quiet mode, no-op, `--dry-run`.

### Tests
- Offline push tests added/extended, including dry-run planning; live push remains opt-in via `wet`/`live_push` markers and required env vars.

## [1.1.1-beta.1] - 2025-09-01

### Fix: Strict Health Completeness for Multi‑Shard Models (Issue #27)
- Problem: Health reported some multi‑part downloads as OK with missing/empty shards (false positives).
- Solution: Backported 2.0 health rules to 1.x with index‑aware validation, pattern detection, and robust corruption checks.
- Details:
  - Config validation: `config.json` must exist and be a non‑empty JSON object.
  - Index‑aware: If `model.safetensors.index.json` or `pytorch_model.bin.index.json` exists, every referenced shard must exist, be non‑empty, and not be a Git LFS pointer file.
  - Pattern fallback policy: If pattern shards like `model-XXXXX-of-YYYYY.*` are present but no index file exists, the model is considered unhealthy (parity with 2.0 policy).
  - Partial/tmp markers: Any `*.partial`, `*.tmp`, or names containing `partial` anywhere under the snapshot mark the model as unhealthy.
  - LFS detection: Recursive scan flags suspiciously small files (<200B) that contain the Git LFS pointer header.
  - Single‑file weights: Non‑empty `*.safetensors`, `*.bin`, or `*.gguf` without pattern shards remain supported and healthy if not LFS pointers.
- Impact: “Healthy” now reliably means “complete and usable” for automation and CLI workflows.
- Tests: Added `tests/unit/test_health_multishard.py` covering complete/missing/empty shards, pointer detection, pattern‑without‑index policy, partial markers, and PyTorch index parity.

Note: GitHub tag/version uses `1.1.1-beta.1`. PyPI release uses PEP 440 `1.1.1b1`.

## 2.0.0-alpha.1 — 2025-08-31

- New JSON-first CLI (`mlxk2`, `mlxk-json`); `--json` for machine-readable output (new vs 1.0.0).
- Human output by default: improved formatting, new Type column, relative Modified; MLX-only compact view with `--all`, `--health`, `--verbose` flags.
- Stricter health checks for sharded models (Issue #27); robust model resolution (fuzzy, `@hash`); `rm` cleans whole model and locks.
- Packaging/tooling: dynamic versioning; multi-Python test script; Python 3.9–3.13; timezone-aware datetimes.
- **Not included yet: server and run** (use 1.x).

## [1.1.0] - 2025-08-26 - **STABLE RELEASE** 🚀

### Production Readiness & Enhanced Testing 🧪
- **First Stable Release Since 1.0.4**: Comprehensive beta testing cycle complete
- **Isolated Test System**: 150/150 tests passing with pristine user cache protection
  - **3-Category Test Strategy**: Isolated cache (78 tests) + Server tests (@pytest.mark.server) + Future framework diversity
  - **User Cache Protection**: Tests use temporary isolated caches - user cache stays completely clean
  - **Real Model Validation**: End-to-end tests using `hf-internal-testing/tiny-random-gpt2` (~12MB) in isolation
  - **Automatic Test Downloads**: No manual model setup required for standard test suite
  - **Parallel Testing**: No cache conflicts between test runs, improved CI reliability
- **Multi-Python Support**: Full compatibility verified for Python 3.9, 3.10, 3.11, 3.12, 3.13
- **All Critical Issues Resolved**: Issues #21, #22, #23 thoroughly tested and production-ready

### Technical Improvements 🔧
- **Test Infrastructure Revolution**: Complete migration from mocked tests to isolated real-world validation
- **Cache Isolation System**: `temp_cache_dir` + `patch_model_cache` fixtures ensure test isolation
- **Performance Optimization**: Fast CI with small test models, comprehensive validation with server tests
- **Developer Experience**: Clean setup process - only Python + test dependencies required
- **Test Reliability**: Reproducible results independent of user's existing model cache

---

## [1.1.0-beta3] - 2025-08-25

### Critical Bug Fixes 🐛
- **Issue #21**: Empty Cache Directory Crash - **RESOLVED**
  - **Root Cause**: `mlxk list` crashed with `FileNotFoundError` on fresh installations  
  - **Fix**: Added `MODEL_CACHE.exists()` checks in `list_models()` function
  - **Impact**: MLX-Knife now works correctly on fresh installations without pre-existing cache
  - **Test Coverage**: Added `test_list_models_real_empty_cache()` regression test

- **Issue #22**: urllib3 LibreSSL Warning on macOS Python 3.9 - **RESOLVED**
  - **Root Cause**: Every MLX-Knife command showed SSL compatibility warning on macOS system Python
  - **Fix**: Central warnings suppression in `__init__.py` before any imports that use urllib3
  - **Impact**: Clean command output on macOS system Python 3.9 with LibreSSL
  - **Scope**: Only affects macOS system Python 3.9, no impact on other environments

- **Issue #23**: Double rm Execution Problem - **RESOLVED**
  - **Root Cause**: `mlxk rm model@hash` required two executions - first left broken state, second completed
  - **Fix**: Changed from partial `snapshots/<hash>` deletion to complete model directory removal
  - **Enhancement**: Added intelligent lock cleanup system with user-friendly prompts
  - **Impact**: Single execution removes models completely + optional HuggingFace lock cleanup
  - **Features**: Interactive confirmation, `--force` parameter, robust corrupted model handling

### Enhanced Cache Management 🧹
- **Lock Cleanup System**: Addresses upstream HuggingFace FileLock accumulation issue
  - User-friendly prompt: "Clean up cache files? [Y/n]" 
  - `--force` parameter skips all confirmations for automation
  - Robust error handling with warnings (never fails on lock cleanup issues)
- **Extended rm Command**: Now handles all model states (healthy, corrupted, empty snapshots)
- **Superior UX**: Cleaner cache management than official HuggingFace CLI tools

### Test Infrastructure Improvements 🧪
- **Test Count**: Updated to 140/140 tests passing (+5 new tests for Issue #23)
- **Regression Coverage**: New tests for empty cache, corrupted models, lock cleanup scenarios
- **Force Parameter Testing**: Comprehensive coverage of interactive vs force mode behavior
- **Integration Test Robustness**: All edge cases now covered with real model testing

### Documentation Updates 📚
- **Version Updates**: All documentation updated to reflect 1.1.0-beta3 status
- **Testing Guide**: Updated test counts and new test scenarios in TESTING.md
- **Issue Documentation**: Added HUGGINGFACE_LOCK_ISSUES.md with upstream context
- **Lock Cleanup Documentation**: Clear explanation of MLX-Knife's cache management advantages

## [1.1.0-beta2] - 2025-08-22

### Critical Bug Fixes 🐛
- **Issue #19**: Server Response Truncation at ~1000 Words - **RESOLVED**
  - **Root Cause**: Server hardcoded `--max-tokens 2000` overrode dynamic limits from 1.1.0-beta1
  - **Fix**: Changed CLI `--max-tokens` default from `2000` to `None`, enabling model-aware dynamic limits
  - **Impact**: Large context models (Qwen3-30B, Llama-3.3-70B) now work at full capacity by default
  - **Validation**: Server startup shows "model-aware dynamic limits" instead of hardcoded values

- **Issue #20**: End-Token Visibility in Non-Streaming Mode - **RESOLVED**  
  - **Root Cause**: `generate_batch()` lacked End-Token filtering present in `generate_streaming()`
  - **Fix**: Ported filtering logic with new `_filter_end_tokens_from_response()` method
  - **Affected**: `mlxk run model "prompt" --no-stream` and Server API `"stream": false`
  - **Impact**: No more end tokens appearing in the final output in non-streaming mode

### Enhanced
- Better default for `--max-tokens`: `None` → model-aware limits
- Improved consistency between streaming and non-streaming generation
- Clearer server logs indicating active token policies

### Technical
- 15 new tests across server and CLI to validate token policies
- Internal refactoring for token handling to avoid duplication

## [1.1.0-beta1] - 2025-08-21

### Added
- Dynamic model-aware token limits (context-length sensitive)
- CLI `--max-tokens` default changed to `None` (was 2000)
- Server leverages the same dynamic limits

### Improved
- End-token filtering consistency across streaming and non-streaming modes
- Robustness in model loading and memory management

### Tests
- 114/114 tests passing
- Server tests behind `@pytest.mark.server` (opt-in)

## [1.0.4] - 2025-08-19

### Fixed
- **Issue #14**: Interactive chat self-conversation bug resolved
  - MLX models no longer continue generating conversation turns after their response
  - Added context-sensitive chat stop tokens: `\nHuman:`, `\nAssistant:`, `\nYou:`, `\nUser:` 
  - Smart priority system: native model stop tokens checked first, chat tokens as fallback
  - Affects both `mlxk run` and `mlxk server` modes
  - Comprehensive regression test suite added with 15 tests across 7+ MLX models

### Enhanced
- **Web UI Complete Overhaul** (simple_chat.html):
  - 🦫 Branding update: Replaced 🔪 with 🦫 (Beaver) emoji for friendlier appearance
  - 💾 Model persistence: Selected model survives browser reload via localStorage  
  - 📚 Chat history persistence: Full conversation history preserved across sessions
  - 🔄 Smart model switching: Choice to keep or clear chat history when switching models
  - 🌐 Responsive design: Full viewport height utilization, optimized screen space usage
  - 🎯 Clear UX: "Clear Chat" instead of ambiguous "Clear" button
  - 🏴 English dialogs: Custom modal dialogs replace German OS dialogs

### Added
- **Automated Server Testing Infrastructure**:
  - RAM-aware model filtering: Automatic model selection based on available system RAM
  - Self-contained server management: Automatic MLX Knife server lifecycle in tests
  - macOS compatible: Graceful handling of permission restrictions
  - Opt-in testing: Server tests marked `@pytest.mark.server`, excluded from default `pytest`
  - Comprehensive testing guide with RAM-based model recommendations

### Technical
- Context-aware token decoding maintains backward compatibility
- Native model stop tokens preserved, chat tokens only as fallback
- Exception-safe server test infrastructure with automatic cleanup
- Complete TESTING.md documentation for server-based regression testing
- All existing tests continue to pass (114/114)

## [1.0.3] - 2025-08-18

### Added
- **Issue #13**: Hash-based disambiguation for ambiguous model names
  - Use commit hashes to disambiguate between multiple matching models
  - Example: `mlxk show Llama@de2dfaf5` automatically resolves to `mlx-community/Llama-3.3-70B-Instruct-4bit`
  - Pure local resolution, no external API calls, offline-capable
- **Issue #6**: Repository name length validation for HuggingFace Hub 96-character limit
  - Pre-validation with clear error message before attempting download
  - Better user experience with immediate feedback on invalid repository names

### Fixed
- **Issue #7**: Fixed health check inconsistency in show command with fuzzy model names
  - `mlxk show Phi-3` vs `mlxk show mlx-community/Phi-3-mini-4k-instruct-4bit` now show identical health status
  - Unified health check logic to use resolved model names for consistent results

### Enhanced
- Enhanced short commit hash support with local resolution
- Improved model name disambiguation logic
- Real user workflow support - see hashes in `mlxk list`, use directly in other commands

### Technical
- 9 new comprehensive test cases added (TestIssue6RepositoryNameValidation, TestShowModelHealthConsistency, TestIssue13HashDisambiguation)
- All 114 unit tests passing on Apple Silicon
- Improved error handling and user experience across all model resolution scenarios

## [1.0.2] - 2025-08-18

### Fixed
- **Issue #11**: Fixed HF_HOME environment variable handling - MLX Knife now correctly uses `$HF_HOME/hub` for model storage, consistent with HuggingFace standard
- **Issue #9**: Fixed silent failure when removing corrupted models with empty snapshots directories
- **Cache Consistency**: Unified cache path logic - both default (`~/.cache/huggingface/hub`) and custom (`$HF_HOME/hub`) paths now consistently use `/hub` subdirectory

### Enhanced  
- **Download Throttling**: Improved adaptive throttling for household-friendly downloads (512KB chunks, 2-3s delays for large files)
- **Migration Warning**: Added helpful warning when models are found in legacy cache locations with clear migration instructions
- **Memory Management**: Enhanced exception-safe resource cleanup and baseline tracking

### Technical
- **Dependencies**: Updated to latest tested versions (huggingface-hub 0.34.0+, mlx 0.28.0+, fastapi 0.116.0+)
- **Python Support**: Full compatibility verified on Python 3.9-3.13
- **Test Suite**: All 105 tests passing with real MLX models on Apple Silicon

## [1.0.1] - 2025-08-15

### Changed
- **Description Update**: Changed package description to "ollama-style CLI for MLX models on Apple Silicon"

## [1.0.0] - 2025-08-15

### Changed
- **STABLE RELEASE**: MLX Knife 1.0.0 officially stable and ready for production use
- **PyPI Publication**: Now available on PyPI for easy installation via `pip install mlx-knife`
- **CLI-Only Policy**: Officially designated as CLI-only tool (Python API access not officially supported)
- **Documentation**: Updated all documentation to reflect stable 1.0.0 release status

## [1.0-rc3] - 2025-08-14

### Added
- **Issue 1**: Partial name filtering for `mlxk list` command (e.g., `mlxk list Phi-3`)
- **Issue 2**: Fuzzy matching for single-model commands (`mlxk show Phi-3`, `mlxk run Phi-3`)
- **Issue 3**: Default `mlxk health` behavior (no `--all` flag required)
- Comprehensive test coverage for all new fuzzy matching features
- Smart ambiguity resolution with helpful error messages

### Enhanced
- All single-model commands now support partial name matching
- Case-insensitive model name searching
- Improved user experience with intelligent model resolution
- Expanded test suite from 96 to 104 tests (104/104 passing ✅)

### Fixed
- Health command now works without requiring `--all` flag
- Better error handling for ambiguous model specifications
- Enhanced fuzzy matching logic with fallback mechanisms

## [1.0-rc2] - 2025-08-13

### Enhanced
- Robust exception handling during model loading with guaranteed cleanup
- Protection against nested context manager usage 
- Safe cleanup that handles partial loading failures
- Exception-resilient cache clearing (won't fail if cache operations error)
- Safe tokenizer attribute access using getattr() with defaults
- Graceful memory stats handling when metrics unavailable
- Comprehensive unit test coverage for all memory management edge cases

### Fixed
- Memory management edge cases in MLXRunner context manager
- Exception safety during model loading and cleanup operations
- Improved error handling for partial model loading failures

## [1.0-rc1] - 2025-08-12

### Added
- Initial release candidate
- Full MLX model support for Apple Silicon
- OpenAI-compatible API server
- Web chat interface
- Multi-Python support (3.9-3.13)
- Comprehensive test suite (86/86 passing)

## Known Issues
- See GitHub Issues for tracking
 
## 2.0.0‑beta.3 (local)

- Server robustness and API polish
  - Supervisor default: Uvicorn runs as subprocess in its own process group; Ctrl‑C terminates deterministically and allows immediate restart.
  - HTTP mapping: 404 for unknown/failed model loads; 503 during shutdown; preserve HTTPException codes from helpers.
  - Streaming (SSE):
    - Happy path: initial chunk, per‑token chunks, final chunk, then `[DONE]`.
    - Interrupt path: on `KeyboardInterrupt` emit clear interrupt marker and close promptly.
  - Token limits: server mode uses half of context length; explicit `max_tokens` respected.
  - Noise reduction: chat streaming debug prints gated behind `MLXK2_DEBUG`.

- Testing
  - Added focused server API tests for `/v1/models`, 404/503 mapping, SSE happy/interrupt, and server‑side token limit propagation.
  - Global suppression of macOS Python 3.9 `urllib3` LibreSSL warning in tests; runtime already suppressed.

- Docs
  - README/TESTING touch‑ups pending flip; CLAUDE.md tracks SSE UX roadmap (anti‑buffering headers, optional heartbeats, status/interrupt endpoints).
