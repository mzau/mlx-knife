# TODO: Issue #26 - Embeddings Implementation Plan

## Overview
Implementation checklist for adding OpenAI-compatible embedding functionality to MLX-Knife with both REST API endpoint and CLI commands.

## Phase 1: Core Infrastructure ⏳

### [ ] Create Core Embedding Module
- [ ] Create `mlx_knife/embedding_utils.py`
- [ ] Implement `embed_model_core()` function
  - [ ] MLX model loading logic
  - [ ] Input preprocessing (string/array handling)
  - [ ] Embedding vector generation
  - [ ] Normalization support
  - [ ] Encoding format support (float/base64)
- [ ] Add error handling for embedding models
- [ ] Add input length limiting with `max_length` parameter

### [ ] Model Compatibility Detection
- [ ] Extend `detect_framework()` for embedding model detection
- [ ] Add embedding model validation in model resolution
- [ ] Research common MLX embedding model patterns

## Phase 2: CLI Implementation ⏳

### [ ] Add CLI Commands
- [ ] Add `embed` subcommand to `mlx_knife/cli.py`
  - [ ] `-m, --model` parameter (required)
  - [ ] `-c, --content` parameter for direct text input
  - [ ] `--input-file` parameter for file input
  - [ ] `--encoding-format` parameter (default: float)
  - [ ] `--normalize` parameter (default: true)
  - [ ] `--max-length` parameter
- [ ] Add `embed-multi` subcommand for batch processing
  - [ ] Stdin input handling
  - [ ] Multiple string processing

### [ ] CLI Integration
- [ ] Add `embed_model()` function to `cache_utils.py`
  - [ ] Follow `run_model()` pattern
  - [ ] Use existing `resolve_single_model()`
  - [ ] Use existing `detect_framework()`
  - [ ] Call `embed_model_core()`
- [ ] Add CLI handler functions
- [ ] Add JSON output formatting for CLI

## Phase 3: Server Endpoint ⏳

### [ ] Add Server Models
- [ ] Create `EmbeddingRequest` Pydantic model
  - [ ] `model: str` field
  - [ ] `input: Union[str, List[str]]` field
  - [ ] `encoding_format: Optional[str]` field
  - [ ] `normalize: Optional[bool]` field  
  - [ ] `max_length: Optional[int]` field
- [ ] Create embedding response models following OpenAI spec

### [ ] Add Server Endpoint
- [ ] Add `@app.post("/v1/embeddings")` to `server.py`
- [ ] Follow `/v1/chat/completions` pattern
- [ ] Use existing `get_or_load_model()` function
- [ ] Call `embed_model_core()` with request parameters
- [ ] Return OpenAI-compatible JSON response
- [ ] Add proper error handling and HTTP status codes

## Phase 4: Testing & Validation ⏳

### [ ] Unit Tests
- [ ] Create `tests/unit/test_embedding_utils.py`
  - [ ] Test `embed_model_core()` function
  - [ ] Test input preprocessing
  - [ ] Test normalization and encoding formats
  - [ ] Test error handling
- [ ] Add embedding tests to existing test files

### [ ] Integration Tests  
- [ ] Create `tests/integration/test_embedding_cli.py`
  - [ ] Test `mlxk embed` command
  - [ ] Test `mlxk embed-multi` command
  - [ ] Test file input functionality
  - [ ] Test various parameter combinations
- [ ] Create `tests/integration/test_embedding_server.py`
  - [ ] Test `/v1/embeddings` endpoint
  - [ ] Test OpenAI compatibility
  - [ ] Test error responses
  - [ ] Test different input formats

### [ ] Real Model Testing
- [ ] Test with actual embedding models
  - [ ] `mxbai-embed-large`
  - [ ] `nomic-embed-text`
  - [ ] Other common MLX embedding models
- [ ] Validate output vector dimensions
- [ ] Verify OpenAI API compatibility

## Phase 5: Documentation & Polish ⏳

### [ ] Documentation Updates
- [ ] Update `README.md` with embedding examples
  - [ ] CLI usage examples
  - [ ] Server endpoint examples
  - [ ] curl command examples
- [ ] Add embedding section to API documentation
- [ ] Update help text and command descriptions

### [ ] Code Quality
- [ ] Add type hints throughout embedding code
- [ ] Add comprehensive docstrings
- [ ] Run linting and formatting
- [ ] Ensure Python 3.9 compatibility

### [ ] Performance & Polish
- [ ] Optimize embedding generation performance
- [ ] Add progress indicators for batch operations
- [ ] Improve error messages and user feedback
- [ ] Add verbose mode support

## Success Criteria ✅

### Functional Requirements
- [ ] `mlxk embed -m "model" -c "text"` generates embeddings
- [ ] `mlxk embed -m "model" --input-file file.txt` processes file input
- [ ] `mlxk embed-multi` handles batch processing
- [ ] `POST /v1/embeddings` returns OpenAI-compatible JSON
- [ ] Both CLI and server use same core logic
- [ ] All embedding models work correctly

### Quality Requirements  
- [ ] 100% test coverage for new code
- [ ] Integration with existing error handling
- [ ] Follows established code patterns
- [ ] Comprehensive documentation
- [ ] Performance acceptable for typical use cases

### Compatibility Requirements
- [ ] OpenAI embedding API compatibility verified
- [ ] Works with common MLX embedding models
- [ ] Integrates cleanly with existing codebase
- [ ] Maintains backwards compatibility

## Implementation Notes

### Architecture Decisions
- **Shared Core**: `embed_model_core()` used by both CLI and server
- **Model Resolution**: Reuse existing `resolve_single_model()` pattern
- **Error Handling**: Follow existing server and CLI error patterns
- **Testing**: Use existing test infrastructure and patterns

### Key Files to Modify
- `mlx_knife/embedding_utils.py` (new)
- `mlx_knife/cache_utils.py` (add embed_model function)
- `mlx_knife/cli.py` (add embed subcommands)
- `mlx_knife/server.py` (add /v1/embeddings endpoint)
- Various test files (new and existing)

### Dependencies
- MLX framework for embedding generation
- Existing model loading and resolution logic
- FastAPI for server endpoint
- Pydantic for request/response models

**Estimated Implementation Time**: 4-6 hours following established patterns