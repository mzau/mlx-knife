import subprocess
import sys
from pathlib import Path
from ..core.cache import MODEL_CACHE, hf_to_cache_dir
from ..core.model_resolution import resolve_model_for_operation
from .health import is_model_healthy


# Pull uses exact user input - HuggingFace resolves model names

def pull_model_with_huggingface_hub(model_name):
    """Use huggingface-hub to pull a model."""
    try:
        # Use direct Python API instead of CLI
        from huggingface_hub import snapshot_download
        
        # Download model to cache (default behavior)
        local_dir = snapshot_download(
            repo_id=model_name,
            local_files_only=False,
            resume_download=True
        )
        
        return True, f"Downloaded to {local_dir}"
    
    except ImportError:
        return False, "huggingface-hub not installed (pip install huggingface-hub)"
    except Exception as e:
        return False, f"Download failed: {str(e)}"


def pull_operation(model_spec):
    """Pull (download) operation for JSON API."""
    result = {
        "status": "success",
        "command": "pull",
        "error": None,
        "data": {
            "model": None,
            "download_status": "unknown",
            "message": "",
            "expanded_name": None
        }
    }
    
    try:
        # Use model resolution for fuzzy matching and expansion
        resolved_name, commit_hash, ambiguous_matches = resolve_model_for_operation(model_spec)
        
        if ambiguous_matches:
            result["status"] = "error"
            result["error"] = {
                "type": "ambiguous_match",
                "message": f"Multiple models match '{model_spec}'",
                "matches": ambiguous_matches
            }
            return result
        elif not resolved_name:
            # No existing model found - use original spec for download as-is
            if "@" in model_spec:
                model_name, commit_hash = model_spec.rsplit("@", 1)
                result["data"]["commit_hash"] = commit_hash
            else:
                model_name = model_spec
                commit_hash = None
            resolved_name = model_name  # Use exact name - let HuggingFace resolve it
        
        result["data"]["model"] = resolved_name
        result["data"]["expanded_name"] = resolved_name if resolved_name != model_spec.split('@')[0] else None
        if commit_hash:
            result["data"]["commit_hash"] = commit_hash
        
        # Check if already exists and is healthy
        cache_dir = MODEL_CACHE / hf_to_cache_dir(resolved_name)
        if cache_dir.exists():
            healthy, _ = is_model_healthy(resolved_name)
            if healthy:
                result["data"]["download_status"] = "already_exists"
                result["data"]["message"] = f"Model {resolved_name} already exists in cache"
                return result
            else:
                # Model exists but unhealthy - suggest rm workflow
                result["status"] = "error"
                result["error"] = {
                    "type": "model_corrupted",
                    "message": f"Model exists but is corrupted. Use 'rm {model_spec}' first, then pull again."
                }
                result["data"]["download_status"] = "corrupted"
                return result
        
        # Attempt download
        result["data"]["download_status"] = "downloading"
        success, message = pull_model_with_huggingface_hub(resolved_name)
        
        if success:
            result["data"]["download_status"] = "success"
            result["data"]["message"] = message
        else:
            result["status"] = "error"
            result["data"]["download_status"] = "failed"
            result["error"] = {
                "type": "download_failed",
                "message": message
            }
    
    except Exception as e:
        result["status"] = "error"
        result["error"] = {
            "type": "pull_operation_failed", 
            "message": str(e)
        }
        result["data"]["download_status"] = "error"
    
    return result