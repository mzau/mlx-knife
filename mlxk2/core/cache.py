"""Cache management for MLX-Knife 2.0."""

import os
from pathlib import Path
from typing import Optional

# Cache path constants - copied from mlx_knife/cache_utils.py
DEFAULT_CACHE_ROOT = Path.home() / ".cache/huggingface"


def get_workspace_home() -> Optional[Path]:
    """Get workspace home directory from MLXK_WORKSPACE_HOME env var.

    Returns:
        Path to workspace home if set and valid, None otherwise.

    Example:
        export MLXK_WORKSPACE_HOME=~/mlx-models
        → Path("/Users/me/mlx-models")
    """
    workspace_home = os.environ.get("MLXK_WORKSPACE_HOME")
    if not workspace_home:
        return None
    path = Path(workspace_home).expanduser()
    # Only return if directory exists (don't auto-create)
    if path.is_dir():
        return path
    return None


def get_current_cache_root() -> Path:
    """Get current cache root (respects runtime HF_HOME changes).

    Note: Returns DEFAULT_CACHE_ROOT if HF_HOME is unset OR empty string.
    This handles `export HF_HOME=""` edge case gracefully.
    """
    hf_home = os.environ.get("HF_HOME")
    if not hf_home:  # None or ""
        return DEFAULT_CACHE_ROOT
    return Path(hf_home)


def get_current_model_cache() -> Path:
    """Get current model cache path (respects runtime HF_HOME changes)."""
    return get_current_cache_root() / "hub"


# Legacy globals - DEPRECATED: Use get_current_*() functions for consistency
CACHE_ROOT = get_current_cache_root()
MODEL_CACHE = get_current_model_cache()


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
