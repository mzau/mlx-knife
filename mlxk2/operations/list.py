"""List models operation for MLX-Knife 2.0."""

from typing import Dict, Any, Optional, Tuple

from ..core.cache import get_current_model_cache, cache_dir_to_hf
from .common import build_model_object


def _latest_snapshot(model_path) -> Tuple[Optional[str], Optional[object]]:
    """Return (hash, path) for the latest snapshot if any, else (None, None)."""
    snapshots_dir = model_path / "snapshots"
    if not snapshots_dir.exists():
        return None, None
    snapshots = [d for d in snapshots_dir.iterdir() if d.is_dir() and len(d.name) == 40]
    if not snapshots:
        return None, None
    latest = max(snapshots, key=lambda x: x.stat().st_mtime)
    return latest.name, latest


def list_models(pattern: str = None) -> Dict[str, Any]:
    """List all models in cache with JSON output.
    
    Args:
        pattern: Optional pattern to filter models (case-insensitive substring match)
    """
    models = []
    model_cache = get_current_model_cache()
    
    if not model_cache.exists():
        return {
            "status": "success",
            "command": "list",
            "data": {
                "models": models,
                "count": 0
            },
            "error": None
        }
    
    # Find all model directories
    for model_dir in model_cache.iterdir():
        if not model_dir.is_dir() or not model_dir.name.startswith("models--"):
            continue
            
        hf_name = cache_dir_to_hf(model_dir.name)
        # Hide test sentinel directories from listings
        if "TEST-CACHE-SENTINEL" in hf_name:
            continue
        
        # Apply pattern filter if specified
        if pattern and pattern.strip():
            if pattern.lower() not in hf_name.lower():
                continue

        # Select snapshot (prefer latest) and build model object
        _hash, snap_path = _latest_snapshot(model_dir)
        model_obj = build_model_object(hf_name, model_dir, snap_path if snap_path is not None else model_dir)
        models.append(model_obj)
    
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
