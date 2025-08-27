"""Test fixtures for MLX-Knife 2.0 isolated testing."""

import os
import tempfile
import pytest
from pathlib import Path
from typing import Generator


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
    
    # Pre-create some realistic test models
    models_created = {}
    
    # MLX models
    models_created["mlx-community/Phi-3-mini-4k-instruct-4bit"] = create_model(
        "mlx-community/Phi-3-mini-4k-instruct-4bit", 
        "e9675aa3def456789abcdef0123456789abcdef0"
    )
    
    models_created["mlx-community/Qwen3-30B-A3B-Instruct-2507-4bit"] = create_model(
        "mlx-community/Qwen3-30B-A3B-Instruct-2507-4bit",
        "e9675aa3def456789abcdef0123456789abcdef0"  # Same short hash for testing
    )
    
    # Non-MLX models  
    models_created["microsoft/DialoGPT-small"] = create_model(
        "microsoft/DialoGPT-small",
        "fedcba987654321fedcba987654321fedcba98"
    )
    
    models_created["Qwen/Qwen3-Coder-480B-A35B-Instruct"] = create_model(
        "Qwen/Qwen3-Coder-480B-A35B-Instruct", 
        "1234567890abcdef1234567890abcdef12345678"
    )
    
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