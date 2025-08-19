# Changelog

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