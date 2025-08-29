"""List models operation for MLX-Knife 2.0."""

from pathlib import Path
from typing import Dict, List, Any

from ..core.cache import get_current_model_cache, cache_dir_to_hf


def get_model_size(model_path):
    """Calculate total model size in human readable format."""
    if not model_path.exists():
        return "unknown"
    
    total_size = 0
    for file in model_path.rglob("*"):
        if file.is_file():
            total_size += file.stat().st_size
    
    if total_size >= 1_000_000_000:
        return f"{total_size / 1_000_000_000:.1f}GB"
    elif total_size >= 1_000_000:
        return f"{total_size / 1_000_000:.1f}MB"
    else:
        return f"{total_size / 1_000:.1f}KB"


def get_model_hashes(model_path):
    """Extract available SHA hashes from snapshots directory."""
    hashes = []
    snapshots_dir = model_path / "snapshots"
    
    if snapshots_dir.exists():
        for snapshot_dir in snapshots_dir.iterdir():
            if snapshot_dir.is_dir() and len(snapshot_dir.name) == 40:
                # Full 40-character SHA hash
                hashes.append(snapshot_dir.name)
    
    return sorted(hashes)


def detect_framework(model_path, hf_name):
    """Detect model framework without exposing internal logic."""
    if "mlx-community" in hf_name:
        return "MLX"
    
    # Check for GGUF files
    if list(model_path.glob("**/*.gguf")):
        return "GGUF"
    
    # Check for common formats
    snapshots_dir = model_path / "snapshots"
    if snapshots_dir.exists():
        has_safetensors = any(snapshots_dir.glob("**/*.safetensors"))
        has_pytorch_bin = any(snapshots_dir.glob("**/pytorch_model.bin"))
        
        if has_safetensors:
            return "PyTorch"
        elif has_pytorch_bin:
            return "PyTorch"
    
    return "Unknown"


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
        
        # Sanitized response - no implementation details or paths
        models.append({
            "name": hf_name,
            "size": get_model_size(model_dir),
            "framework": detect_framework(model_dir, hf_name),
            "cached": True,
            "hashes": get_model_hashes(model_dir)
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
