"""List models operation for MLX-Knife 2.0."""

from pathlib import Path
from typing import Dict, List, Any

from ..core.cache import MODEL_CACHE, cache_dir_to_hf


def list_models() -> Dict[str, Any]:
    """List all models in cache with JSON output."""
    models = []
    
    if not MODEL_CACHE.exists():
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
    for model_dir in MODEL_CACHE.iterdir():
        if not model_dir.is_dir() or not model_dir.name.startswith("models--"):
            continue
            
        hf_name = cache_dir_to_hf(model_dir.name)
        models.append({
            "name": hf_name,
            "cache_dir": model_dir.name,
            "path": str(model_dir)
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