"""List models operation for MLX-Knife 2.0."""

from datetime import datetime
from typing import Dict, Any, Optional, Tuple

from ..core.cache import get_current_model_cache, cache_dir_to_hf
from .health import is_model_healthy


def _total_size_bytes(model_path) -> int:
    """Calculate total model size in bytes for a given path."""
    if not model_path.exists():
        return 0
    total_size = 0
    for file in model_path.rglob("*"):
        if file.is_file():
            total_size += file.stat().st_size
    return total_size


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


def detect_model_type(hf_name: str) -> str:
    n = hf_name.lower()
    if "embed" in n:
        return "embedding"
    if "instruct" in n or "chat" in n:
        return "chat"
    return "base"


def detect_capabilities(hf_name: str) -> list:
    n = hf_name.lower()
    if "embed" in n:
        return ["embeddings"]
    caps = ["text-generation"]
    if "instruct" in n or "chat" in n:
        caps.append("chat")
    return caps


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

        # Select snapshot (prefer latest) and compute fields
        commit_hash, snap_path = _latest_snapshot(model_dir)
        selected_path = snap_path if snap_path is not None else model_dir
        last_modified = datetime.fromtimestamp(selected_path.stat().st_mtime).strftime("%Y-%m-%dT%H:%M:%SZ")
        size_bytes = _total_size_bytes(selected_path)
        healthy, _reason = is_model_healthy(hf_name)

        # Minimal model object per spec 0.1.2
        models.append({
            "name": hf_name,
            "hash": commit_hash,
            "size_bytes": size_bytes,
            "last_modified": last_modified,
            "framework": detect_framework(model_dir, hf_name),
            "model_type": detect_model_type(hf_name),
            "capabilities": detect_capabilities(hf_name),
            "health": "healthy" if healthy else "unhealthy",
            "cached": True,
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
