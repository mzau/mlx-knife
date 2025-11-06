# ADR-003: Server and Run Functionality Port from 1.x to 2.0

**Status**: Accepted  
**Date**: 2025-09-10  
**Decision Makers**: mzau, Claude

## Context

The 2.0 branch (`feature/2.0.0-alpha.1`) currently lacks the server and run functionality that has been significantly enhanced in the 1.x branch through versions 1.1.1-beta.2 and 1.1.1-beta.3. This includes:

1. **Server functionality** (1.x: `mlx_knife/server.py`):
   - OpenAI-compatible REST API (`/v1/chat/completions`, `/v1/completions`)
   - Real-time streaming support via SSE
   - Model hot-swapping and caching
   - Dynamic token limits based on model context length

2. **Run functionality** (1.x: `mlx_knife/mlx_runner.py`):
   - Direct MLX model execution with streaming
   - Interactive chat mode with conversation history
   - Memory management with context managers
   - Stop token filtering and handling

3. **Reasoning support** (1.x: `mlx_knife/reasoning_utils.py` - NEW in beta.3):
   - GPT-OSS/MXFP4 reasoning model support
   - Pattern-based reasoning extraction
   - Formatted output with `**[Reasoning]**` / `**[Answer]**` sections
   - `--hide-reasoning` flag for answer-only output

4. **Enhanced features from beta.2/beta.3**:
   - MXFP4 quantization support (requires MLX ≥0.29.0)
   - Lenient MLX detection for private repos (Issue #31)
   - README/tokenizer-based model type detection
   - Strict health checks for multi-shard models (Issue #27)
   - Enhanced `show` command with detailed quantization display:
     - MXFP4 mode detection with version requirements
     - GGUF variants listing with sizes
     - Precision info extraction (int4, int8, gguf, etc.)

The 2.0 architecture already includes:
- Modular structure (`mlxk2/core/`, `mlxk2/operations/`, `mlxk2/output/`)
- JSON-first API with schema versioning
- Human output backend (despite docs suggesting JSON-only for beta)
- Enhanced testing infrastructure with isolated caches

## Decision

We will port the server and run functionality from 1.x to 2.0 following a **test-driven, modular approach** that preserves the 2.0 architecture advantages while incorporating all 1.x enhancements.

### Port Strategy

*Note: "Week 1-4" bezeichnet die logische Reihenfolge, nicht reale Kalenderwochen*

#### Week 1: Test Suite Extraction and Abstraction
1. **Extract test specifications** from 1.x test suite:
   - Server tests: `test_server_functionality.py`, `test_issue_14.py`, `test_issue_15_16.py`, `test_end_token_issue.py`
   - Run tests: `test_run_command_advanced.py`, `test_mlx_runner_memory.py`
   - Reasoning tests: Tests for GPT-OSS/MXFP4 formatting

2. **Create abstract test specifications** in 2.0:
   - Document expected behaviors, not implementation details
   - Define API contracts and edge cases
   - Create test matrices for different model types

3. **Implement 2.0-native tests first**:
   - Write tests against the expected 2.0 API
   - Use 2.0's isolated cache infrastructure
   - Ensure tests fail initially (red phase of TDD)

#### Week 2: Modular Implementation
1. **Core modules** (`mlxk2/core/`):
   - `runner.py`: MLX model execution engine (from `mlx_runner.py`)
   - `reasoning.py`: Reasoning extraction utilities (from `reasoning_utils.py`)
   - `server_base.py`: FastAPI server foundation

2. **Operations modules** (`mlxk2/operations/`):
   - `run.py`: CLI run command implementation (inkl. Interactive Chat; kein separates `chat.py`)
   - `serve.py`: Server startup and management (Supervisor als Default)

3. **Output adaptors** (`mlxk2/output/`):
   - Extend existing JSON/Human output for server responses
   - Add streaming output support for both formats

#### Week 3: Feature Integration
1. **Port enhancements in priority order**:
   - Basic run/server functionality (MVP for 2.0.0-beta.1)
   - Reasoning support (GPT-OSS/MXFP4)
   - Dynamic token limits
   - Enhanced model detection (Issue #31)
   - Strict health checks (already partially in 2.0)

2. **Maintain backward compatibility**:
   - Same CLI interface as 1.x
   - Same OpenAI API endpoints
   - Same web UI (update version strings)

### Test-Driven Approach

```python
# Example: Abstract test specification for server
class ServerAPIContract:
    """Define expected server behaviors independent of implementation"""
    
    def test_chat_completions_streaming(self):
        """Server must support streaming chat completions"""
        # Given: A running server with a loaded model
        # When: POST to /v1/chat/completions with stream=true
        # Then: Receive SSE stream with data: prefixed chunks
        
    def test_model_hot_swapping(self):
        """Server must support switching models without restart"""
        # Given: Server running with model A
        # When: Request with different model B
        # Then: Model B loads and responds correctly
        
    def test_dynamic_token_limits(self):
        """Server must respect model context limits"""
        # Given: Model with 8K context
        # When: No max_tokens specified
        # Then: Use appropriate dynamic limit
```

### Implementation Mapping

| 1.x Component | 2.0 Location | Notes |
|--------------|--------------|-------|
| `mlx_knife/server.py` | `mlxk2/core/server_base.py` + `mlxk2/operations/serve.py` | Split core from CLI |
| `mlx_knife/mlx_runner.py` | `mlxk2/core/runner/` | Core execution engine (modularisiert als Paket) |
| `mlx_knife/reasoning_utils.py` | `mlxk2/core/reasoning.py` | Pattern-based extraction |
| `mlx_knife/cache_utils.py` additions | `mlxk2/core/cache.py` extensions | Model detection + quantization display |
| Server CLI logic | `mlxk2/operations/serve.py` | Command implementation |
| Run CLI logic | `mlxk2/operations/run.py` | Command implementation (inkl. Interactive) |

## Consequences

### Positive
- **Test coverage maintained**: All 1.x test scenarios covered in 2.0
- **Architecture preserved**: 2.0's modular structure enhanced, not compromised
- **Feature parity**: 2.0.0-beta.1 will be feature-complete vs 1.1.1
- **Clean separation**: Core logic separate from CLI/output concerns
- **Future-proof**: Easier to add new output formats or APIs

### Negative
- **Development time**: Test-first approach takes longer initially
- **Temporary duplication**: Some code exists in both branches during transition
- **Complexity**: More files/modules than 1.x monolithic approach

### Neutral
- **Version jump to beta.1**: Justified by feature completeness and "human" backend
- **Push feature**: Remains experimental/undefined as per current state
- **License split**: Maintained (1.x MIT, 2.x Apache-2.0)

## Implementation Checklist

*Chronologische Reihenfolge - kann parallel oder iterativ bearbeitet werden*

### Week 1: Test Infrastructure
- [ ] Extract server test specifications from 1.x
- [ ] Extract run/chat test specifications from 1.x
- [ ] Create abstract test contracts in 2.0
- [ ] Write failing tests for all core features

### Week 2: Core Implementation
- [ ] Implement `mlxk2/core/runner.py`
- [ ] Implement `mlxk2/core/server_base.py`
- [ ] Implement `mlxk2/core/reasoning.py`
- [ ] Extend `mlxk2/core/cache.py` with detection

### Week 3: Operations Layer
- [ ] Implement `mlxk2/operations/run.py`
- [ ] Implement `mlxk2/operations/chat.py`
- [ ] Implement `mlxk2/operations/serve.py`
- [ ] Update CLI in `mlxk2/cli.py`

### Week 4: Integration & Polish
- [x] Integrate output formatters (Human + JSON)
- [x] Full 2.0 default test suite passing (containing server-minimaltests)
- [x] Documentation updates (CLAUDE.md, TESTING.md)

## Release Criteria for 2.0.0-beta.1

Based on this port and existing 2.0 features:

### Must Have (Beta.1)
- ✅ JSON-first API (already in alpha.3)
- ✅ Human output backend (already in alpha.3)
- ✅ Enhanced model detection (already in alpha.3)
- ✅ Server functionality with OpenAI API (Supervisor, SSE, Hot‑Swap)
- ✅ Run command with streaming
- ✅ Interactive chat mode
- ✅ Basic reasoning support (GPT-OSS)
- [ ] 90%+ test coverage

### Should Have (Beta.2)
- [ ] Full reasoning features (hide-reasoning flag)
- [ ] Advanced token management
- [ ] Performance optimizations
- [ ] Extended test coverage (95%+)
- [x] Issue #30 Preflight (premature integration)

### Could Have (Future)
- [ ] Custom reasoning token configuration
- [ ] Multi-model server support
- [ ] Push functionality (currently experimental)
- [ ] Web UI (not part of 2.0‑port)

### Not in Scope for Port
- **System prompt CLI support** (`--system` parameter): This is a future enhancement not yet implemented in 1.x. Decision on this feature will be made after successful server & run functional parity with 1.1.1 is achieved. See CLAUDE.md for ongoing discussion.

## References

- CHANGELOG.md: Complete feature history of 1.1.1-beta.2 and beta.3
- TESTING.md: 1.x test structure and categories
- Issue #27: Strict health checks for multi-shard models
- Issue #31: Lenient MLX detection for private repos
- CLAUDE.md: Current context and TODOs
