# Changelog

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
  - **Impact**: Professional clean output - no visible `</s>`, `<|im_end|>`, `<|end|>` tokens
  - **Test Coverage**: 47/48 comprehensive tests validate fix across all model architectures

### Test Infrastructure Improvements 🧪
- **New Test Suite**: `tests/integration/test_end_token_issue.py` with 48 systematic tests
- **RAM-Aware Testing**: Automatic model selection based on available system memory
- **Flaky Test Fix**: Improved server lifecycle management with proper port cleanup
- **Blocking Read Fix**: Fixed timeout issues in server startup validation tests
- **Test Count**: 132/132 standard tests + 48 server tests (180 total)

### Documentation Updates 📚
- **TESTING.md**: New server test procedures, updated test counts (132/132), comprehensive server test guide
- **Test Categories**: Clear separation of standard tests vs resource-intensive server tests
- **Server Test Documentation**: RAM requirements, timing expectations, model compatibility

### Architecture Quality 🏗️
- **End-Token Consistency**: Streaming and non-streaming pipelines now identical in behavior
- **Clean Code**: Unified filtering logic eliminates code duplication between pipelines  
- **Regression Prevention**: Comprehensive test coverage prevents future End-Token issues
- **Professional Output**: All models and modes produce clean, professional responses
- **Test Stability**: Eliminated flaky tests and timeouts for reliable CI/CD

## [1.1.0-beta1] - 2025-08-21

### Major Features 🚀
- **Issues #15 & #16**: Dynamic Model-Aware Token Limits
  - Eliminated hardcoded 500/2000 token defaults with intelligent model-based limits
  - **Phi-3-mini**: 4096 context → 2048 server tokens, 4096 interactive (8x improvement)
  - **Qwen2.5-30B**: 262,144 context → 131,072 server tokens, 262,144 interactive (524x improvement!)
  - Context-aware policies: Interactive mode uses full context, server mode uses context/2 for DoS protection
  - Automatic adaptation to new models with larger context windows (future-proof)

### Enhanced Web Client 🌐  
- **Model Token Capacity Display**: Shows "Ready with Mistral-7B (32,768 tokens)" in header
- **Enhanced `/v1/models` API**: Now exposes `context_length` field for model capabilities
- **Button State Management**: Clear Chat properly disabled during streaming with CSS styling
- **Streaming Status Tracking**: Added `isStreaming` flag with "Generating response..." feedback

### Interactive Mode Improvements 💡
- **Smart CLI Defaults**: `mlxk run <model> "prompt"` automatically uses optimal token limits per model
- **No Configuration Needed**: Users benefit immediately without changing usage patterns
- **Explicit Control Preserved**: `--max-tokens` arguments still respected and capped at model context
- **Clean Type Safety**: Proper `Optional[int]` handling eliminates fragile CLI guessing

### Technical Architecture 🏗️
- **`get_model_context_length()` function**: Extracts context length from model configs with multiple fallback keys
- **Enhanced MLXRunner**: `get_effective_max_tokens()` method for context-aware token limiting
- **Server API Updates**: All endpoints use model-aware limits with DoS protection
- **Unified Token Logic**: Single source of truth through MLXRunner eliminates duplicate code
- **Backward Compatible**: All existing CLI arguments and APIs work unchanged

### Performance Impact 📊
- **Modern Models Unleashed**: Large-context models can now use their full capabilities
- **Real-World Benefits**: No more artificial 500-token truncation for 100K+ context models  
- **Smart Server Limits**: Automatic DoS protection while maximizing usable context
- **Zero Magic Numbers**: Clean architecture with clear `None` vs explicit value semantics

### Testing & Quality Assurance ✅
- **Comprehensive Coverage**: 131/131 tests passing (expansion from 114 tests)
- **20 new unit tests**: Covering CLI None-handling, model context extraction, effective token calculation
- **5 server integration tests**: Real-world validation with actual MLX models
- **Extreme Model Testing**: Validated with models from 1B to 30B parameters, up to 256K context
- **Edge Case Handling**: Unknown models, missing configs, CLI argument combinations

### Issue #14 Model Compatibility Validation
**Chat Self-Conversation Fix tested across model spectrum:**

| Model | Size | RAM (GB) | Context | Status | Architecture |
|-------|------|----------|---------|--------|-------------|
| **Llama-3.2-1B-Instruct-4bit** | 1B | 2 | 131,072 | ✅ PASSED | Llama |
| **Llama-3.2-3B-Instruct-4bit** | 3B | 4 | 131,072 | ✅ PASSED | Llama |
| **Phi-3-mini-4k-instruct-4bit** | 4B | 5 | 4,096 | ✅ PASSED | Phi-3 |
| **Mistral-7B-Instruct-v0.2-4bit** | 7B | 8 | 32,768 | ✅ PASSED | Mistral |
| **Mixtral-8x7B-Instruct-v0.1-4bit** | 8x7B | 16 | 32,768 | ✅ PASSED | Mixtral MoE |
| **Mistral-Small-3.2-24B-Instruct-2506-4bit** | 24B | 20 | 32,768 | ✅ PASSED | Mistral |
| **Qwen3-30B-A3B-Instruct-2507-4bit** | 30B | 24 | 262,144 | ✅ PASSED | Qwen |

**Validation Results**: 7/7 models passed - comprehensive coverage from 1B to 30B parameters across all major MLX architectures ensures robust chat stop token handling.

### Beta Status Notes ⚠️
- **Core Functionality**: Solid foundation with comprehensive test coverage
- **Known Limitation**: Server deadlock possible under extreme concurrent model loading stress
- **Workaround**: Avoid simultaneous heavy model operations (normal usage unaffected)  
- **Real-World Ready**: Significant improvements ready for community testing and feedback

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
  - 🏴󠁧󠁢󠁥󠁮󠁧󠁿 English dialogs: Custom modal dialogs replace German OS dialogs

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
