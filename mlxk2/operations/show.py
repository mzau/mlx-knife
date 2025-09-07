"""Show model operation for MLX-Knife 2.0."""

import json
from typing import Dict, Any

from ..core.cache import get_current_model_cache, hf_to_cache_dir
from ..core.model_resolution import resolve_model_for_operation
from .common import build_model_object


def get_file_type(file_name):
    """Determine file type based on file name."""
    if file_name == "config.json":
        return "config"
    elif file_name.endswith((".safetensors", ".bin", ".gguf")):
        return "weights"
    elif "tokenizer" in file_name.lower():
        return "tokenizer"
    elif file_name.endswith(".json"):
        return "config"
    elif file_name == "README.md":
        return "readme"
    else:
        return "other"


def get_model_files(model_path):
    """Get list of files in model directory with type classification."""
    files = []
    
    if not model_path.exists():
        return files
        
    for file_path in sorted(model_path.rglob("*")):
        if file_path.is_file():
            size_bytes = file_path.stat().st_size
            if size_bytes >= 1_000_000_000:
                size_str = f"{size_bytes / 1_000_000_000:.1f}GB"
            elif size_bytes >= 1_000_000:
                size_str = f"{size_bytes / 1_000_000:.1f}MB"
            elif size_bytes >= 1_000:
                size_str = f"{size_bytes / 1_000:.1f}KB"
            else:
                size_str = f"{size_bytes}B"
                
            files.append({
                "name": file_path.name,
                "size": size_str,
                "type": get_file_type(file_path.name)
            })
    
    return files


def extract_model_metadata(model_path):
    """Extract metadata from config.json if available."""
    config_path = model_path / "config.json"
    if not config_path.exists():
        return None
        
    try:
        with open(config_path) as f:
            config = json.load(f)
            
        # Extract common metadata fields
        metadata = {}
        
        # Model architecture
        if "model_type" in config:
            metadata["model_type"] = config["model_type"]
        if "architectures" in config and config["architectures"]:
            metadata["architecture"] = config["architectures"][0]
            
        # Quantization info
        if "quantization_config" in config:
            quant = config["quantization_config"]
            if "bits" in quant:
                metadata["quantization"] = f"{quant['bits']}bit"
        
        # Size parameters
        if "max_position_embeddings" in config:
            metadata["context_length"] = config["max_position_embeddings"]
        if "vocab_size" in config:
            metadata["vocab_size"] = config["vocab_size"]
        if "hidden_size" in config:
            metadata["hidden_size"] = config["hidden_size"]
        if "num_attention_heads" in config:
            metadata["num_attention_heads"] = config["num_attention_heads"]
        if "num_hidden_layers" in config:
            metadata["num_hidden_layers"] = config["num_hidden_layers"]
            
        return metadata if metadata else None
        
    except (OSError, json.JSONDecodeError):
        return None


def get_config_content(model_path):
    """Get config.json content as parsed JSON."""
    config_path = model_path / "config.json"
    if not config_path.exists():
        return None
        
    try:
        with open(config_path) as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError):
        return None


def _is_40_hex(s: str) -> bool:
    return len(s) == 40 and all(c in "0123456789abcdef" for c in s.lower())


def show_model_operation(model_pattern: str, include_files: bool = False, include_config: bool = False) -> Dict[str, Any]:
    """Show detailed model information."""
    result = {
        "status": "success",
        "command": "show",
        "data": None,
        "error": None
    }
    
    try:
        # Resolve model name and hash
        resolved_name, commit_hash, ambiguous_matches = resolve_model_for_operation(model_pattern)
        
        if ambiguous_matches:
            result["status"] = "error"
            result["error"] = {
                "type": "ambiguous_match",
                "message": f"Multiple models match '{model_pattern}'",
                "matches": ambiguous_matches
            }
            return result
            
        if not resolved_name:
            result["status"] = "error"
            result["error"] = {
                "type": "model_not_found",
                "message": f"No model found matching '{model_pattern}'"
            }
            return result
            
        # Get model directory
        model_cache_dir = get_current_model_cache() / hf_to_cache_dir(resolved_name)
        if not model_cache_dir.exists():
            result["status"] = "error"
            result["error"] = {
                "type": "model_not_cached",
                "message": f"Model '{resolved_name}' not found in cache"
            }
            return result
            
        # Find the correct snapshot
        snapshots_dir = model_cache_dir / "snapshots"
        model_path = None
        
        if commit_hash and snapshots_dir.exists():
            # Specific hash requested
            hash_path = snapshots_dir / commit_hash
            if hash_path.exists():
                model_path = hash_path
            else:
                result["status"] = "error"
                result["error"] = {
                    "type": "hash_not_found",
                    "message": f"Hash '{commit_hash}' not found for model '{resolved_name}'"
                }
                return result
        elif snapshots_dir.exists():
            # Use latest snapshot
            snapshots = [d for d in snapshots_dir.iterdir() if d.is_dir()]
            if snapshots:
                model_path = max(snapshots, key=lambda x: x.stat().st_mtime)
                commit_hash = model_path.name
        
        if not model_path:
            model_path = model_cache_dir
            
        # Build unified model object
        model_obj = build_model_object(resolved_name, model_cache_dir, model_path)

        # Build response data
        data = {"model": model_obj}
        
        if include_files:
            data["files"] = get_model_files(model_path)
            data["metadata"] = None
        elif include_config:
            data["config"] = get_config_content(model_path)
            data["metadata"] = None
        else:
            data["metadata"] = extract_model_metadata(model_path)
            
        result["data"] = data
        
    except Exception as e:
        result["status"] = "error"
        result["error"] = {
            "type": "show_operation_failed",
            "message": str(e)
        }
        
    return result
