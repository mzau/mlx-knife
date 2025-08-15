# Changelog

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