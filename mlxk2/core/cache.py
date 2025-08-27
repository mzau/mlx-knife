"""Cache management for MLX-Knife 2.0."""

import os
from pathlib import Path

# Cache path constants - copied from mlx_knife/cache_utils.py
DEFAULT_CACHE_ROOT = Path.home() / ".cache/huggingface"
CACHE_ROOT = Path(os.environ.get("HF_HOME", DEFAULT_CACHE_ROOT))
MODEL_CACHE = CACHE_ROOT / "hub"


def hf_to_cache_dir(hf_name: str) -> str:
    """Convert HuggingFace model name to cache directory name.
    
    Universal rule: ALL "/" become "--" (mechanical conversion).
    """
    if hf_name.startswith("models--"):
        return hf_name
    
    # Replace all "/" with "--" for universal conversion
    converted = hf_name.replace("/", "--")
    return f"models--{converted}"


def cache_dir_to_hf(cache_name: str) -> str:
    """Convert cache directory name to HuggingFace model name.
    
    Universal rule: ALL "--" become "/" (mechanical conversion).
    This handles both clean names and corrupted cache entries gracefully.
    """
    if cache_name.startswith("models--"):
        remaining = cache_name[len("models--"):]
        return remaining.replace("--", "/")
    return cache_name


def get_model_path(hf_name: str) -> Path:
    """Get the full path to a model in the cache."""
    cache_dir = hf_to_cache_dir(hf_name)
    return MODEL_CACHE / cache_dir