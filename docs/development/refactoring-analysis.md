# MLX-Knife Refactoring Strategy v1.1.1

## Executive Summary

**Goal**: Refactor `cache_utils.py` (1000+ lines) into modular components while adding Issue #8 (model caching) and Issue #26 (embeddings API) fixes.

**Strategy**: Test-driven refactoring FIRST, then add features on clean codebase.

**Timeline**: 3 days (beta1 → beta2 → stable)

## Current Situation

### Problems
- `cache_utils.py`: 1000+ lines "God Module"
- **Token costs**: ~4000 tokens per Claude request for full file
- **Issue #8**: Models reload on every `mlxk run` (10-30s penalty)
- **Issue #26**: Missing embeddings API for RAG/agent use cases
- **Code coupling**: Everything depends on everything

### Assets
- ✅ **150 passing tests** as safety net
- ✅ Clean public API (CLI commands)
- ✅ Version 1.1.0 stable as fallback
- ✅ Good test coverage

## Dependency Analysis

```python
# CORE FUNCTIONS (must stay together - 150 lines)
hf_to_cache_dir()          # Pure function, no deps
cache_dir_to_hf()          # Pure function, no deps  
expand_model_name()        # Uses above
parse_model_spec()         # Uses expand_model_name

# LAYER 1: Path Resolution (200 lines)
get_model_path()           # Uses CORE
find_matching_models()     # Uses CORE
hash_exists_in_local_cache() # Uses CORE
resolve_single_model()     # Uses all above

# LAYER 2: Model Info (250 lines) - EASILY EXTRACTABLE
get_model_size()           # Standalone
get_model_modified()       # Standalone  
detect_framework()         # Standalone
get_model_hash()          # Standalone

# LAYER 3: Health (150 lines) - EASILY EXTRACTABLE
check_lfs_corruption()     # Standalone
is_model_healthy()         # Uses check_lfs_corruption
check_model_health()       # Uses resolve_single_model + is_model_healthy
check_all_models_health()  # Uses is_model_healthy

# LAYER 4: Operations (200 lines)
list_models()             # Uses LAYER 2 functions
show_model()              # Uses everything (but clean)
rm_model()                # Uses resolve_single_model

# OUTLIER: run_model() - DOESN'T BELONG HERE
run_model()               # Should be in cli.py or runner.py
```

**Entanglement Score: 2/10** (Very easy to refactor!)

## Release Plan

### Version 1.1.1-beta1: Clean Refactoring (Day 1)

#### File Structure After Refactoring
```
mlx_knife/
├── cache_utils.py         # Backward compatibility re-exports only
├── core/
│   ├── __init__.py       
│   ├── paths.py          # Core path functions (150 lines)
│   ├── info.py           # Model information (250 lines)
│   ├── health.py         # Health checks (150 lines)
│   ├── operations.py     # List, show, rm (200 lines)
│   └── model_cache.py    # NEW: LRU cache for Issue #8
├── model_runner.py       # Moved run_model() from cache_utils
├── embeddings.py         # NEW: Embedding extractor
├── mlx_runner.py         # Unchanged
└── server.py            # Updated to use new modules
```

#### Implementation Steps

1. **Backup and prepare**
```bash
git checkout -b feature/1.1.1-refactor
cp cache_utils.py cache_utils_backup.py
pytest tests/ -v  # Baseline: all green
```

2. **Create modular structure**
```bash
mkdir -p mlx_knife/core
touch mlx_knife/core/__init__.py
```

3. **Split files** (manual or scripted)
```python
# mlx_knife/core/paths.py
"""Core path and cache utilities."""
from pathlib import Path
import os

DEFAULT_CACHE_ROOT = Path.home() / ".cache/huggingface"
CACHE_ROOT = Path(os.environ.get("HF_HOME", DEFAULT_CACHE_ROOT))
MODEL_CACHE = CACHE_ROOT / "hub"

def hf_to_cache_dir(hf_name: str) -> str: ...
def cache_dir_to_hf(cache_name: str) -> str: ...
def expand_model_name(model_name): ...
def parse_model_spec(model_spec): ...
# etc.

# mlx_knife/core/info.py
"""Model information utilities."""
def get_model_size(model_path): ...
def get_model_modified(model_path): ...
def detect_framework(model_path, hf_name): ...
def get_model_hash(model_path): ...

# mlx_knife/core/health.py
"""Model health and validation."""
def check_lfs_corruption(model_path): ...
def is_model_healthy(model_spec): ...
def check_model_health(model_spec): ...
def check_all_models_health(): ...

# mlx_knife/core/operations.py
"""Model operations (list, show, remove)."""
def list_models(...): ...
def show_model(...): ...
def rm_model(...): ...
```

4. **Create compatibility shim**
```python
# mlx_knife/cache_utils.py
"""
Backward compatibility module.
All functions re-exported from their new locations.
"""
# Core paths - these stay as module-level exports
from .core.paths import (
    MODEL_CACHE, CACHE_ROOT, DEFAULT_CACHE_ROOT,
    hf_to_cache_dir, cache_dir_to_hf,
    expand_model_name, parse_model_spec,
    get_model_path, find_matching_models,
    hash_exists_in_local_cache, resolve_single_model
)

# Model info functions
from .core.info import (
    get_model_size, get_model_modified,
    detect_framework, get_model_hash
)

# Health checks
from .core.health import (
    check_lfs_corruption, is_model_healthy,
    check_model_health, check_all_models_health
)

# Operations
from .core.operations import (
    list_models, show_model, rm_model
)

# This moves elsewhere but maintain compatibility
from .model_runner import run_model

__all__ = [
    # Paths
    'MODEL_CACHE', 'CACHE_ROOT', 'DEFAULT_CACHE_ROOT',
    'hf_to_cache_dir', 'cache_dir_to_hf',
    'expand_model_name', 'parse_model_spec',
    'get_model_path', 'find_matching_models',
    'hash_exists_in_local_cache', 'resolve_single_model',
    # Info
    'get_model_size', 'get_model_modified',
    'detect_framework', 'get_model_hash',
    # Health
    'check_lfs_corruption', 'is_model_healthy',
    'check_model_health', 'check_all_models_health',
    # Operations
    'list_models', 'show_model', 'rm_model',
    'run_model'
]
```

5. **Validate with tests**
```bash
pytest tests/ -xvs  # Stop on first failure
# Fix any import issues
# Repeat until all 150 tests pass
```

### Version 1.1.1-beta2: Add Features (Day 2)

#### Issue #8: Model Caching

```python
# mlx_knife/core/model_cache.py
"""LRU cache for loaded models to avoid reload overhead."""
import time
from typing import Dict, Optional, Tuple
from ..mlx_runner import MLXRunner

class ModelCache:
    """Simple LRU cache for loaded models."""
    
    def __init__(self, max_models: int = 2):
        self._cache: Dict[str, Tuple[MLXRunner, float]] = {}
        self._max_models = max_models
    
    def get_or_load(self, model_path: str, verbose: bool = False) -> MLXRunner:
        """Get model from cache or load if not cached."""
        if model_path in self._cache:
            runner, _ = self._cache[model_path]
            self._cache[model_path] = (runner, time.time())
            if verbose:
                print(f"[CACHE] Model loaded from cache: {model_path}")
            return runner
        
        # Evict oldest if cache full
        if len(self._cache) >= self._max_models:
            oldest_key = min(self._cache.items(), key=lambda x: x[1][1])[0]
            oldest_runner = self._cache[oldest_key][0]
            oldest_runner.cleanup()
            del self._cache[oldest_key]
            if verbose:
                print(f"[CACHE] Evicted model: {oldest_key}")
        
        # Load new model
        if verbose:
            print(f"[CACHE] Loading new model: {model_path}")
        runner = MLXRunner(model_path, verbose=verbose)
        runner.load_model()
        self._cache[model_path] = (runner, time.time())
        return runner
    
    def clear(self):
        """Clear all cached models."""
        for runner, _ in self._cache.values():
            runner.cleanup()
        self._cache.clear()

# Global cache instance
_model_cache = ModelCache()
```

Update `model_runner.py`:
```python
from .core.model_cache import _model_cache

def run_model(model_spec, prompt=None, ...):
    model_path, model_name, commit_hash = resolve_single_model(model_spec)
    # Use cache instead of creating new runner
    runner = _model_cache.get_or_load(str(model_path), verbose=verbose)
    # ... rest of the function
```

#### Issue #26: Embeddings API

```python
# mlx_knife/embeddings.py
"""Embedding extraction for MLX models."""
import mlx.core as mx
from typing import List, Tuple

class EmbeddingExtractor:
    """Extract embeddings from any MLX model."""
    
    def extract_embeddings(
        self,
        model,
        tokenizer,
        texts: List[str],
        normalize: bool = True,
        max_length: Optional[int] = None,
        verbose: bool = False
    ) -> Tuple[List[List[float]], List[int]]:
        """
        Extract raw embeddings from model.
        
        Returns:
            Tuple of (embeddings, token_counts)
        """
        embeddings = []
        token_counts = []
        
        for text in texts:
            # Tokenize
            tokens = tokenizer.encode(
                text,
                max_length=max_length,
                truncation=True if max_length else False
            )
            token_counts.append(len(tokens))
            
            # Get embeddings
            with mx.no_grad():
                token_array = mx.array([tokens])
                outputs = model(token_array)
                
                # Extract hidden states
                if hasattr(outputs, 'last_hidden_state'):
                    hidden = outputs.last_hidden_state
                elif isinstance(outputs, tuple):
                    hidden = outputs[0]
                else:
                    hidden = outputs
                
                # Mean pooling
                embedding = mx.mean(hidden, axis=1).squeeze()
                
                # Normalize
                if normalize:
                    norm = mx.linalg.norm(embedding)
                    if norm > 0:
                        embedding = embedding / norm
                
                embeddings.append(embedding.tolist())
                
        return embeddings, token_counts
```

Update `server.py`:
```python
from .embeddings import EmbeddingExtractor

@app.post("/v1/embeddings")
async def create_embeddings(request: EmbeddingRequest):
    """Embedding endpoint for rapid prototyping."""
    runner = get_or_load_model(request.model)
    extractor = EmbeddingExtractor()
    
    inputs = request.input if isinstance(request.input, list) else [request.input]
    embeddings, token_counts = extractor.extract_embeddings(
        runner.model,
        runner.tokenizer,
        inputs,
        normalize=request.normalize,
        max_length=request.max_length
    )
    
    return {
        "object": "list",
        "data": [
            {"object": "embedding", "embedding": emb, "index": i}
            for i, emb in enumerate(embeddings)
        ],
        "model": request.model,
        "usage": {
            "prompt_tokens": sum(token_counts),
            "total_tokens": sum(token_counts)
        }
    }
```

### Version 1.1.1: Stable Release (Day 3)

1. **Final testing**
```bash
# Unit tests
pytest tests/ -v

# Integration tests  
pytest tests/ -m integration

# Coverage report
pytest tests/ --cov=mlx_knife --cov-report=term-missing
```

2. **Performance validation**
```bash
# Before (without cache)
time mlxk run Phi-3 "test1"  # ~20s
time mlxk run Phi-3 "test2"  # ~20s

# After (with cache)
time mlxk run Phi-3 "test1"  # ~20s (first load)
time mlxk run Phi-3 "test2"  # ~0.5s (cached!)
```

3. **Update documentation**
- Add embeddings example to README
- Document cache behavior
- Update CHANGELOG

## Benefits

### Token Cost Reduction
- **Before**: ~4000 tokens per file edit
- **After**: ~600-800 tokens per focused module
- **Savings**: 75-85% reduction

### Development Speed
- Faster PR reviews (smaller files)
- Better Claude interactions (focused context)
- Easier debugging (isolated modules)
- Parallel development possible

### Code Quality
- Clear separation of concerns
- No more 1000+ line files
- Better testability
- Easier to understand

## Risk Mitigation

1. **Test Suite**: 150 tests ensure no regressions
2. **Backward Compatibility**: `cache_utils.py` re-exports everything
3. **Incremental Approach**: Beta releases for validation
4. **Fallback Plan**: v1.1.0 stable always available

## Success Metrics

- [ ] All 150 tests passing
- [ ] Coverage > 90%
- [ ] Issue #8 resolved (cache working)
- [ ] Issue #26 implemented (embeddings API)
- [ ] Token costs reduced by >70%
- [ ] No breaking changes in public API

## Future Considerations

### 2.0.0 Decision Point
After 1.1.1 stable, evaluate if 2.0.0 is needed:
- If refactoring is clean enough → continue with 1.x
- If major changes needed → branch to 2.0.0-alpha

### Next Refactoring Targets
1. `server.py` (growing with embeddings)
2. `cli.py` (could use command pattern)
3. `mlx_runner.py` (consider splitting generation/chat)

---

*Document created: 2025-08-26*  
*Target release: MLX-Knife v1.1.1*  
*Refactoring philosophy: Test-driven, incremental, backward-compatible*