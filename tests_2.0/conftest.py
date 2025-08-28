"""Test fixtures for MLX-Knife 2.0 isolated testing."""

import os
import tempfile
import pytest
from pathlib import Path
from typing import Generator
from contextlib import contextmanager


@pytest.fixture
def isolated_cache() -> Generator[Path, None, None]:
    """Create isolated cache for MLX-Knife 2.0 tests - NEVER touches user cache."""
    with tempfile.TemporaryDirectory(prefix="mlxk2_test_") as temp_dir:
        cache_path = Path(temp_dir) / "test_cache"
        cache_path.mkdir()
        
        # Create hub subdirectory (HuggingFace standard structure)
        hub_path = cache_path / "hub"
        hub_path.mkdir()
        
        # Store original HF_HOME
        old_hf_home = os.environ.get("HF_HOME")
        os.environ["HF_HOME"] = str(cache_path)
        
        # CRITICAL: Patch MODEL_CACHE to use our isolated cache
        from mlxk2.core import cache
        original_cache = cache.MODEL_CACHE
        cache.MODEL_CACHE = hub_path
        
        # SAFETY CANARY: Create sentinel model to verify we're in test cache
        sentinel_dir = hub_path / "models--TEST-CACHE-SENTINEL--mlxk2-safety-check"
        sentinel_snapshot = sentinel_dir / "snapshots" / "test123456789abcdef0123456789abcdef0123"
        sentinel_snapshot.mkdir(parents=True)
        (sentinel_snapshot / "config.json").write_text('{"model_type": "test_sentinel", "test_cache": true}')
        
        try:
            yield hub_path  # Return hub path (where models-- directories go)
        finally:
            # Restore everything
            cache.MODEL_CACHE = original_cache
            if old_hf_home:
                os.environ["HF_HOME"] = old_hf_home
            elif "HF_HOME" in os.environ:
                del os.environ["HF_HOME"]


@pytest.fixture 
def mock_models(isolated_cache):
    """Create realistic mock models in isolated cache."""
    
    def create_model(hf_name: str, commit_hash: str = "abcdef123456789", healthy: bool = True):
        """Create a mock model with proper directory structure."""
        from mlxk2.core.cache import hf_to_cache_dir
        
        cache_dir_name = hf_to_cache_dir(hf_name)
        model_base_dir = isolated_cache / cache_dir_name
        
        # Create snapshots directory
        snapshots_dir = model_base_dir / "snapshots"
        snapshot_dir = snapshots_dir / commit_hash
        snapshot_dir.mkdir(parents=True)
        
        if healthy:
            # Create healthy model files
            (snapshot_dir / "config.json").write_text('{"model_type": "test", "hidden_size": 768}')
            (snapshot_dir / "tokenizer.json").write_text('{"version": "1.0"}')
            (snapshot_dir / "model.safetensors").write_bytes(b"fake_model_weights" * 1000)
        else:
            # Create corrupted model (missing files)
            (snapshot_dir / "config.json").write_text('invalid json {')
        
        return model_base_dir, snapshot_dir
    
    # Pre-create diverse test models for framework detection
    models_created = {}
    
    # MLX models (detected by "mlx-community" in name)
    models_created["mlx-community/Phi-3-mini-4k-instruct-4bit"] = create_model(
        "mlx-community/Phi-3-mini-4k-instruct-4bit", 
        "e9675aa3def456789abcdef0123456789abcdef0"
    )
    
    models_created["mlx-community/Qwen3-30B-A3B-Instruct-2507-4bit"] = create_model(
        "mlx-community/Qwen3-30B-A3B-Instruct-2507-4bit",
        "e9675aa3def456789abcdef0123456789abcdef0"  # Same short hash for testing
    )
    
    # Second Qwen model for ambiguous matching tests (mock only - different hash)
    models_created["Qwen/Qwen3-Coder-480B-A35B-Instruct"] = create_model(
        "Qwen/Qwen3-Coder-480B-A35B-Instruct", 
        "beef1234567890abcdef1234567890abcdefbeef"  # Different hash from above
    )
    
    # PyTorch models (detected by .safetensors files)
    pytorch_model = create_model(
        "microsoft/DialoGPT-small",
        "fedcba987654321fedcba987654321fedcba98"
    )
    # Add safetensors file for PyTorch detection
    (pytorch_model[1] / "model.safetensors").write_bytes(b"fake_safetensors" * 100)
    models_created["microsoft/DialoGPT-small"] = pytorch_model
    
    # GGUF model (detected by .gguf files) 
    gguf_model = create_model(
        "TheBloke/Llama-2-7B-Chat-GGUF",
        "1234567890abcdef1234567890abcdef12345678"
    )
    # Add GGUF file
    (gguf_model[1] / "q4_0.gguf").write_bytes(b"fake_gguf_model" * 200)
    models_created["TheBloke/Llama-2-7B-Chat-GGUF"] = gguf_model
    
    # Embeddings model (different model_type in config)
    embed_model = create_model(
        "sentence-transformers/all-MiniLM-L6-v2",
        "abcd1234567890abcdef1234567890abcdef12"
    )
    # Override config for embeddings
    (embed_model[1] / "config.json").write_text('{"model_type": "bert", "task": "feature-extraction"}')
    models_created["sentence-transformers/all-MiniLM-L6-v2"] = embed_model
    
    # Corrupted model for testing tolerance
    models_created["corrupted/model"] = create_model(
        "corrupted/model",
        "corrupted123456789abcdef0123456789abcdef0",
        healthy=False
    )
    
    return models_created


@pytest.fixture
def create_corrupted_cache_entry(isolated_cache):
    """Create corrupted cache entries for testing naming tolerance."""
    
    def create_corrupted(cache_name: str):
        """Create a corrupted cache directory name (violates naming rules)."""
        corrupted_dir = isolated_cache / cache_name
        snapshots_dir = corrupted_dir / "snapshots" / "main"  
        snapshots_dir.mkdir(parents=True)
        
        # Create minimal files so it's detected as model
        (snapshots_dir / "config.json").write_text('{"model_type": "corrupted"}')
        
        return corrupted_dir
    
    return create_corrupted


def test_list_models(cache_path):
    """Test-specific list_models that uses exact cache path provided.
    
    This ensures test operations use the same cache consistently.
    """
    from mlxk2.core.cache import cache_dir_to_hf
    
    # SAFETY CHECK: Ensure we're using test cache, not user cache
    path_str = str(cache_path)
    if "/Volumes/mz-SSD/huggingface" in path_str:
        raise RuntimeError(f"FORBIDDEN: Test tried to use user cache: {path_str}")
    if "/var/folders/" not in path_str or "_test_" not in path_str:
        raise RuntimeError(f"WARNING: Unexpected cache path - should be test cache: {path_str}")
    
    # CANARY CHECK: Verify test cache sentinel exists
    sentinel_dir = cache_path / "models--TEST-CACHE-SENTINEL--mlxk2-safety-check"
    if not sentinel_dir.exists():
        raise RuntimeError(f"MISSING CANARY: Test cache sentinel not found in {cache_path}")
    
    models = []
    
    if not cache_path.exists():
        return {
            "status": "success",
            "command": "list",
            "data": {
                "models": models,
                "count": 0
            },
            "error": None
        }
    
    # Find all model directories in the provided cache path
    for model_dir in cache_path.iterdir():
        if not model_dir.is_dir() or not model_dir.name.startswith("models--"):
            continue
            
        hf_name = cache_dir_to_hf(model_dir.name)
        
        # Get hashes from snapshots
        hashes = []
        snapshots_dir = model_dir / "snapshots"
        if snapshots_dir.exists():
            for snapshot_dir in snapshots_dir.iterdir():
                if snapshot_dir.is_dir() and len(snapshot_dir.name) == 40:
                    hashes.append(snapshot_dir.name)
        
        models.append({
            "name": hf_name,
            "hashes": sorted(hashes),
            "cached": True
        })
    
    # Sort by name for consistent output
    models.sort(key=lambda x: x["name"])
    
    return {
        "status": "success", 
        "command": "list",
        "data": {
            "models": models,
            "count": len(models)
        },
        "error": None
    }


def test_resolve_model_for_operation(cache_path, model_query):
    """Test-specific model resolution that uses exact cache path provided.
    
    This ensures model resolution uses the same cache as other test operations.
    """
    # SAFETY CHECK: Ensure we're using test cache, not user cache
    path_str = str(cache_path)
    if "/Volumes/mz-SSD/huggingface" in path_str:
        raise RuntimeError(f"FORBIDDEN: Test tried to use user cache: {path_str}")
    if "/var/folders/" not in path_str or "_test_" not in path_str:
        raise RuntimeError(f"WARNING: Unexpected cache path - should be test cache: {path_str}")
    
    # CANARY CHECK: Verify test cache sentinel exists
    sentinel_dir = cache_path / "models--TEST-CACHE-SENTINEL--mlxk2-safety-check"
    if not sentinel_dir.exists():
        raise RuntimeError(f"MISSING CANARY: Test cache sentinel not found in {cache_path}")
    
    from mlxk2.core.cache import cache_dir_to_hf
    
    # Parse @hash syntax if present
    if "@" in model_query:
        model_name, requested_hash = model_query.split("@", 1)
        requested_hash = requested_hash.lower()
    else:
        model_name = model_query
        requested_hash = None
    
    # Find matching models in the provided cache path
    matching_models = []
    
    if not cache_path.exists():
        return None, None, []
    
    for model_dir in cache_path.iterdir():
        if not model_dir.is_dir() or not model_dir.name.startswith("models--"):
            continue
            
        hf_name = cache_dir_to_hf(model_dir.name)
        
        # Skip sentinel model
        if "TEST-CACHE-SENTINEL" in hf_name:
            continue
        
        # Check for name match (exact, partial, fuzzy)
        name_matches = False
        if model_name.lower() == hf_name.lower():
            name_matches = True  # Exact match
        elif model_name.lower() in hf_name.lower():
            name_matches = True  # Partial match
        elif any(part.lower() in hf_name.lower() for part in model_name.split("-")):
            name_matches = True  # Fuzzy match
        
        if name_matches:
            # Get available hashes
            snapshots_dir = model_dir / "snapshots"
            available_hashes = []
            if snapshots_dir.exists():
                for snapshot_dir in snapshots_dir.iterdir():
                    if snapshot_dir.is_dir() and len(snapshot_dir.name) == 40:
                        available_hashes.append(snapshot_dir.name)
            
            # Check hash match if requested
            if requested_hash:
                hash_match = any(h.lower().startswith(requested_hash) for h in available_hashes)
                if hash_match:
                    matching_models.append(hf_name)
            else:
                matching_models.append(hf_name)
    
    # Return resolution results
    if len(matching_models) == 0:
        return None, requested_hash, []
    elif len(matching_models) == 1:
        return matching_models[0], requested_hash, None
    else:
        # Ambiguous - return choices
        return None, requested_hash, matching_models


def test_health_check_operation(cache_path, model_query=None):
    """Test-specific health check that uses exact cache path provided.
    
    This ensures health check uses the same cache as other test operations.
    """
    # SAFETY CHECK: Ensure we're using test cache, not user cache
    path_str = str(cache_path)
    if "/Volumes/mz-SSD/huggingface" in path_str:
        raise RuntimeError(f"FORBIDDEN: Test tried to use user cache: {path_str}")
    if "/var/folders/" not in path_str or "_test_" not in path_str:
        raise RuntimeError(f"WARNING: Unexpected cache path - should be test cache: {path_str}")
    
    # CANARY CHECK: Verify test cache sentinel exists
    sentinel_dir = cache_path / "models--TEST-CACHE-SENTINEL--mlxk2-safety-check"
    if not sentinel_dir.exists():
        raise RuntimeError(f"MISSING CANARY: Test cache sentinel not found in {cache_path}")
    
    from mlxk2.core.cache import cache_dir_to_hf
    import json
    
    healthy_models = []
    unhealthy_models = []
    
    if not cache_path.exists():
        return {
            "status": "success",
            "command": "health",
            "data": {
                "healthy": [],
                "unhealthy": [],
                "summary": {"total": 0, "healthy_count": 0, "unhealthy_count": 0}
            },
            "error": None
        }
    
    # Check all models in cache path
    for model_dir in cache_path.iterdir():
        if not model_dir.is_dir() or not model_dir.name.startswith("models--"):
            continue
            
        hf_name = cache_dir_to_hf(model_dir.name)
        
        # Skip sentinel model
        if "TEST-CACHE-SENTINEL" in hf_name:
            continue
        
        # Filter by model_query if specified (supports @hash syntax)
        if model_query:
            # Parse @hash syntax if present
            if "@" in model_query:
                query_name, requested_hash = model_query.split("@", 1)
                requested_hash = requested_hash.lower()
                
                # Check name match
                name_matches = (query_name.lower() in hf_name.lower())
                if not name_matches:
                    continue
                
                # Check hash match
                snapshots_dir = model_dir / "snapshots"
                hash_matches = False
                if snapshots_dir.exists():
                    for snapshot_dir in snapshots_dir.iterdir():
                        if snapshot_dir.is_dir() and len(snapshot_dir.name) == 40:
                            if snapshot_dir.name.lower().startswith(requested_hash):
                                hash_matches = True
                                break
                
                if not hash_matches:
                    continue
            else:
                # Simple name filtering
                if model_query.lower() not in hf_name.lower():
                    continue
        
        # Check model health
        is_healthy = True
        health_issues = []
        
        # Check snapshots directory
        snapshots_dir = model_dir / "snapshots"
        if not snapshots_dir.exists():
            is_healthy = False
            health_issues.append("Missing snapshots directory")
        else:
            # Check for at least one valid snapshot
            valid_snapshots = []
            for snapshot_dir in snapshots_dir.iterdir():
                if snapshot_dir.is_dir() and len(snapshot_dir.name) == 40:
                    # Check for config.json
                    config_file = snapshot_dir / "config.json"
                    if config_file.exists():
                        try:
                            with open(config_file, 'r') as f:
                                json.load(f)
                            valid_snapshots.append(snapshot_dir.name)
                        except (json.JSONDecodeError, IOError):
                            health_issues.append(f"Invalid config.json in {snapshot_dir.name}")
                    else:
                        health_issues.append(f"Missing config.json in {snapshot_dir.name}")
            
            if not valid_snapshots:
                is_healthy = False
                health_issues.append("No valid snapshots found")
        
        # Categorize model
        model_info = {
            "name": hf_name,
            "issues": health_issues
        }
        
        if is_healthy:
            healthy_models.append(model_info)
        else:
            unhealthy_models.append(model_info)
    
    return {
        "status": "success",
        "command": "health", 
        "data": {
            "healthy": healthy_models,
            "unhealthy": unhealthy_models,
            "summary": {
                "total": len(healthy_models) + len(unhealthy_models),
                "healthy_count": len(healthy_models),
                "unhealthy_count": len(unhealthy_models)
            }
        },
        "error": None
    }


@contextmanager
def atomic_cache_context(cache_path: Path, expected_context="test"):
    """Atomic cache switching context manager.
    
    Temporarily switches HF_HOME to use specific cache, with verification.
    """
    from mlxk2.core.cache import verify_cache_context
    
    # Store original HF_HOME
    original_hf_home = os.environ.get("HF_HOME")
    
    try:
        # Switch to specified cache
        if cache_path:
            os.environ["HF_HOME"] = str(cache_path.parent)  # cache_path is hub/, we need parent
        
        # Verify we're in the right context
        verify_cache_context(expected_context)
        
        yield cache_path
        
    finally:
        # Restore original HF_HOME
        if original_hf_home:
            os.environ["HF_HOME"] = original_hf_home
        elif "HF_HOME" in os.environ:
            del os.environ["HF_HOME"]


@contextmanager  
def user_cache_context():
    """Context manager for user cache operations."""
    # User cache doesn't need HF_HOME changes - it's the default
    from mlxk2.core.cache import get_current_model_cache, verify_cache_context
    
    # Just verify we're in user cache context
    verify_cache_context("user")
    
    yield get_current_model_cache()