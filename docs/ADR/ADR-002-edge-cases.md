# ADR-002: Edge Cases Learned from MLX-Knife 1.x Test Suite

## Status
**Proposed** - 2025-08-26

## Context

MLX-Knife 1.x has 150+ tests covering numerous edge cases discovered during development. These tests represent critical knowledge about real-world usage patterns, failure modes, and subtle requirements that must be preserved in 2.0.

## Extracted Edge Cases by Category

### 1. Model Name Resolution

**Critical Cases:**
- **Short name expansion**: "Phi-3" → "mlx-community/Phi-3-mini-4k-instruct-4bit"
- **Hash disambiguation**: When multiple models match, allow `#abc123` suffix
- **Partial matching**: "Llama" matches all Llama models (ambiguous)
- **Empty/whitespace names**: Must handle gracefully
- **Invalid characters**: Names with multiple slashes, special chars
- **Name length limits**: HuggingFace has 96 character limit

**Implementation Requirements:**
```python
def resolve_model_name(name: str) -> tuple[str, Optional[str]]:
    # Returns (model_name, commit_hash)
    # Handle: "Phi-3", "Phi-3#abc123", "mlx-community/Phi-3", etc.
    # Max 96 chars validation
    # Graceful fallback for unknowns
```

### 2. Cache Directory Management

**Critical Cases:**
- **Round-trip conversion**: HF name ↔ cache dir must be bijective
- **Special characters**: Org names with hyphens, dots
- **Missing snapshots directory**: Model without snapshots/
- **Multiple snapshots**: Same model, different commits
- **Empty model directories**: Leftover from failed downloads
- **Orphaned lock files**: .lock files without corresponding models

**Implementation Requirements:**
```python
def cache_path_operations():
    # Must handle:
    # - models--org--name format
    # - snapshots/<hash>/ structure
    # - refs/ for branch tracking
    # - .lock cleanup on operations
```

### 3. Health Checking

**Critical Cases:**
- **LFS pointer files**: Detect Git LFS placeholders (not actual weights)
- **Truncated safetensors**: Partial downloads appearing valid
- **Missing config.json**: Model without configuration
- **Missing tokenizer files**: No tokenizer_config.json
- **Framework detection**: MLX vs PyTorch vs Tokenizer-only
- **Symlink handling**: Don't follow dangerous symlinks
- **Race conditions**: Health check during active download

**Framework Detection Logic (TRICKY!):**
```python
def detect_framework(model_path, hf_name):
    # Quick win: mlx-community models are always MLX
    if "mlx-community" in hf_name:
        return "MLX"
    
    # Check actual files
    has_safetensors = any(path.glob("*/*.safetensors"))
    has_pytorch = any(path.glob("*/pytorch_model.bin"))
    has_config = any(path.glob("*/config.json"))
    total_size = get_model_size(model_path)
    
    # Edge case: Tokenizer-only "models" (< 10MB)
    if total_size < 10 * 1024 * 1024:  # 10MB threshold
        return "Tokenizer"
    
    # Priority order matters!
    if has_safetensors and has_config:
        return "MLX"  # Assume safetensors = MLX
    elif has_pytorch:
        return "PyTorch"
    else:
        return "Unknown"

# PROBLEM: This heuristic fails for:
# - Non-mlx-community MLX models
# - Mixed framework models
# - Models with both .safetensors and .bin files
```

**For 2.0:** 
- Health checks should work for ALL frameworks
- Don't filter by framework in health command
- Show framework in output but don't block operations

**LFS Pointer Detection Pattern:**
```python
def is_lfs_pointer(file_path):
    # Check for:
    # - File size < 1KB for .safetensors
    # - Content starts with "version https://git-lfs"
    # - "oid sha256:" in first 200 bytes
```

### 4. Delete Operations (rm command)

**Critical Cases (Issue #23 regression):**
- **Force flag behavior**: `-f` must skip ALL confirmations
- **Interactive prompts**: Must respect user input exactly
- **Lock file cleanup**: Remove .lock files with model
- **Partial deletion recovery**: Handle interrupted deletes
- **Permission errors**: Read-only files, system dirs
- **Non-existent models**: Graceful error messages

**Implementation Requirements:**
```python
def remove_model(name: str, force: bool = False):
    # MUST respect force flag completely
    # Clean .lock files ALWAYS
    # Atomic operation or rollback
```

### 5. Server Mode Edge Cases

**Critical Cases (Issues #14, #15, #16):**
- **Token limits**: Respect model's actual context length
- **Self-conversation bug**: Messages accumulating incorrectly
- **Streaming vs non-streaming**: End tokens must match
- **Concurrent requests**: Model loading race conditions
- **Port conflicts**: Handle "address already in use"
- **SIGTERM handling**: Clean shutdown (Issue #18 known limitation)
- **Memory management**: Proper cleanup after each request

**Token Limit Strategy:**
```python
def get_safe_token_limit(model_path: Path, is_server: bool):
    # Extract from config.json:
    # - max_position_embeddings (priority 1)
    # - n_positions (priority 2) 
    # - context_length (priority 3)
    # Server mode: min(model_limit, 8192)  # DOS protection
    # Interactive: model_limit or 4096 default
```

### 6. Download & Network Operations

**Critical Cases:**
- **Network timeouts**: Graceful handling, clear messages
- **Partial downloads**: Resume or clean restart
- **Invalid repo names**: Early validation before network call
- **Rate limiting**: Respect HF rate limits
- **Disk space**: Check before download starts
- **Concurrent downloads**: Prevent duplicate downloads

### 7. Process Lifecycle

**Critical Cases:**
- **Zombie processes**: Clean up on parent crash
- **Resource leaks**: File handles, network connections
- **Lock starvation**: Prevent infinite lock waiting
- **Signal handling**: SIGINT, SIGTERM, SIGKILL
- **Timeout handling**: Commands taking too long

### 8. Test Isolation Requirements

**Critical Cases:**
- **Cache pollution**: Tests must NEVER touch user's ~/.cache/huggingface
- **Temporary test cache**: Use isolated temp directory for ALL tests
- **Parallel execution**: Tests must be independent
- **Cleanup verification**: Ensure complete cleanup after each test
- **Mock boundaries**: What to mock vs real
- **Deterministic output**: Consistent across runs

**Implementation Pattern:**
```python
# conftest.py - CRITICAL for 2.0 tests
import tempfile
import os
from pathlib import Path

@pytest.fixture
def isolated_cache(monkeypatch):
    """EVERY test MUST use this to avoid user cache pollution."""
    with tempfile.TemporaryDirectory() as tmpdir:
        test_cache = Path(tmpdir) / "huggingface/hub"
        test_cache.mkdir(parents=True)
        
        # Override environment for complete isolation
        monkeypatch.setenv("HF_HOME", str(tmpdir / "huggingface"))
        monkeypatch.setenv("TMPDIR", str(tmpdir))
        
        # Also patch any direct references in code
        monkeypatch.setattr("mlxk2.core.cache.CACHE_ROOT", test_cache.parent)
        monkeypatch.setattr("mlxk2.core.cache.MODEL_CACHE", test_cache)
        
        yield test_cache
        
        # Cleanup is automatic with TemporaryDirectory

# EVERY test MUST use it:
def test_list_models(isolated_cache):
    # This test cannot pollute user cache
    result = list_models()
    assert result["models"] == []
```

## JSON-Specific Edge Cases for 2.0

### 1. Output Consistency
- **Error format**: Always valid JSON even on crash
- **Partial results**: Stream vs complete JSON
- **Unicode handling**: Proper escaping in JSON
- **Large outputs**: Streaming JSON for big lists
- **Number precision**: Float representation

### 2. Backward Compatibility
- **Exit codes**: Must match 1.x behavior
- **Error messages**: Similar enough for scripts
- **Model resolution**: Same fuzzy matching
- **Path handling**: Same cache structure

## Implementation Checklist for 2.0

### Phase 1: Core Robustness (alpha0)
- [ ] Model name validation (96 char limit)
- [ ] Cache directory round-trip conversion
- [ ] Basic health checks (file existence)
- [ ] Force flag in rm command
- [ ] JSON error handling

### Phase 2: Advanced Edge Cases (alpha1)
- [ ] LFS pointer detection
- [ ] Hash disambiguation
- [ ] Lock file cleanup
- [ ] Partial match warnings
- [ ] Network timeout handling

### Phase 3: Server Integration (beta1)
- [ ] Token limit extraction
- [ ] Memory cleanup patterns
- [ ] Streaming JSON support
- [ ] Concurrent request handling

## Testing Strategy for 2.0

### Unit Tests (30-40 tests)
Focus on pure functions:
- Name resolution logic
- Path conversions
- JSON serialization
- Error formatting

### Integration Tests (20-30 tests)
Real operations with mock cache:
- Health checks on various states
- Delete operations with locks
- List with mixed frameworks
- Error recovery paths

### No Need to Port
- UI/formatting tests (JSON-only now)
- Server streaming format tests
- Terminal color tests
- Progress bar tests

## Patterns to Preserve

### 1. Fail-Fast with Clear Errors
```python
if len(model_name) > 96:
    return {
        "status": "error",
        "error": {
            "type": "ValidationError",
            "message": f"Model name too long: {len(model_name)}/96"
        }
    }
```

### 2. Defensive File Operations
```python
# Always check exists before operations
if not path.exists():
    return None  # Don't throw, return None
    
# Always use Path, not strings
path = Path(model_path)
```

### 3. Atomic Operations
```python
# Either complete fully or rollback
try:
    shutil.rmtree(model_path)
    remove_lock_files(model_name)
except Exception as e:
    # Log but don't partially delete
    pass
```

## Key Learnings

1. **Users expect fuzzy matching** - "Phi" should find Phi models
2. **Force flags must be absolute** - No prompts when -f is used
3. **Lock files cause problems** - Always clean them up
4. **LFS pointers fool naive checks** - Must detect explicitly
5. **Token limits prevent crashes** - Respect model capabilities
6. **Health checks save debugging time** - Worth the complexity
7. **Network operations fail often** - Timeout and retry logic essential
8. **Cache corruption is common** - Robust detection critical

## Decision Outcome

These edge cases represent hard-won knowledge from production usage. The 2.0 implementation MUST handle these cases correctly to maintain user trust and functionality, even while moving to JSON-only output.

## References
- Issue #14: Self-conversation bug
- Issue #15/16: Token limit race conditions
- Issue #18: Server signal handling
- Issue #23: Force flag regression
- Test suite: 150+ tests in tests/