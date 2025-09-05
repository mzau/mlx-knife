# Changelog

## 1.1.1 — Pending

- Fix (Issue #27): Strict health completeness for multi-shard models in 1.x:
  - Recognize both safetensors (`model.safetensors.index.json`) and PyTorch (`pytorch_model.bin.index.json`) JSON indices.
  - Validate only the present format’s shards (exist, non-empty, not LFS pointers) to avoid false negatives.
  - Aligns 1.x health behavior with 2.0.0-alpha.1 policy.
 - Planned (Issue #31, under #29): Detect Framework/Type via HF Model Card (README front matter) and tokenizer config for non-`mlx-community` repos (lenient parsing). No CLI/JSON schema changes; focused unit tests; target 1.1.1-b2.

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
## 2.0.0-alpha.2 — 2025-09-04
- Experimental: add `push` command (M0 upload-only) with hard excludes and `.hfignore` support
- Safety: require `--private` in CLI for alpha.2 to avoid accidental public uploads
- JSON: add `push` to schema; examples updated; short experimental disclaimer in responses
- Robustness: early validation for `pull` model names; improved CLI JSON errors for missing args
