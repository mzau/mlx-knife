import shutil
import os
from ..core.cache import get_current_model_cache, hf_to_cache_dir, cache_dir_to_hf
from ..core.model_resolution import resolve_model_for_operation


def find_matching_models(pattern):
    """Find models that match a partial pattern."""
    model_cache = get_current_model_cache()
    all_models = [d for d in model_cache.iterdir() if d.name.startswith("models--")]
    matches = []
    
    for model_dir in all_models:
        hf_name = cache_dir_to_hf(model_dir.name)
        if pattern.lower() in hf_name.lower():
            matches.append((model_dir, hf_name))
    
    return matches


def resolve_model_for_deletion(model_spec):
    """Resolve model spec to exact model for deletion, with fuzzy matching."""
    if "@" in model_spec:
        model_name, commit_hash = model_spec.rsplit("@", 1)
    else:
        model_name = model_spec
        commit_hash = None
    
    # Try exact match first  
    model_cache = get_current_model_cache()
    base_cache_dir = model_cache / hf_to_cache_dir(model_name)
    if base_cache_dir.exists():
        return base_cache_dir, model_name, commit_hash, False
    
    # Try fuzzy matching
    matches = find_matching_models(model_name)
    
    if not matches:
        return None, None, None, False
    elif len(matches) == 1:
        # Unambiguous match
        found_model_dir, found_hf_name = matches[0]
        return found_model_dir, found_hf_name, commit_hash, True
    else:
        # Ambiguous - return matches for user choice
        return None, None, None, matches


def check_model_locks(model_name):
    """Check if model has active lock files."""
    model_cache = get_current_model_cache()
    locks_dir = model_cache / ".locks"
    model_locks = []
    
    if not locks_dir.exists():
        return []
    
    # Look for lock files related to this model
    for lock_file in locks_dir.glob("**/*.lock"):
        if hf_to_cache_dir(model_name) in str(lock_file):
            model_locks.append(str(lock_file.relative_to(model_cache)))
    
    return model_locks


def cleanup_model_locks(model_name):
    """Clean up HuggingFace lock files for a deleted model."""
    model_cache = get_current_model_cache()
    locks_dir = model_cache / ".locks" / hf_to_cache_dir(model_name)
    
    if not locks_dir.exists():
        return 0
    
    try:
        lock_files = list(locks_dir.iterdir())
        if lock_files:
            shutil.rmtree(locks_dir)
            return len(lock_files)
    except Exception:
        pass
    
    return 0


def rm_operation(model_spec, force=False):
    """Remove (delete) operation for JSON API."""
    result = {
        "status": "success",
        "command": "rm",
        "error": None,
        "data": {
            "model": None,
            "action": "unknown",
            "message": "",
            "requires_confirmation": False,
            "matches": [],
            "lock_files_cleaned": 0
        }
    }
    
    try:
        model_cache = get_current_model_cache()
        if not model_cache.exists():
            result["status"] = "error"
            result["error"] = {
                "type": "cache_not_found",
                "message": "Model cache directory does not exist"
            }
            return result
        
        resolved_name, commit_hash, ambiguous_matches = resolve_model_for_operation(model_spec)
        
        if ambiguous_matches:
            result["status"] = "error"
            result["data"]["action"] = "ambiguous"
            result["data"]["matches"] = ambiguous_matches
            result["error"] = {
                "type": "ambiguous_match",
                "message": f"Multiple models match '{model_spec}'"
            }
            return result
        elif not resolved_name:
            result["status"] = "error"
            result["error"] = {
                "type": "model_not_found", 
                "message": f"No models found matching '{model_spec}'"
            }
            return result
        
        resolved_model_dir = model_cache / hf_to_cache_dir(resolved_name)
        is_fuzzy_match = resolved_name != model_spec.split('@')[0]
        
        result["data"]["model"] = resolved_name
        
        # Check for active locks - requires --force (replaces interactive prompt)
        active_locks = check_model_locks(resolved_name)
        if active_locks and not force:
            result["status"] = "error"
            result["data"]["locks_detected"] = True
            result["data"]["lock_files"] = active_locks
            result["error"] = {
                "type": "locks_present",
                "message": "Model has active locks. Use --force to override."
            }
            return result
        
        # Check if this requires confirmation (fuzzy match)
        if is_fuzzy_match and not force:
            result["data"]["requires_confirmation"] = True
            result["data"]["action"] = "requires_confirmation"  
            result["data"]["message"] = f"Would delete '{resolved_name}' (matched from '{model_spec}')"
            return result
        
        # Handle specific hash deletion
        if commit_hash:
            snapshots_dir = resolved_model_dir / "snapshots"
            if not snapshots_dir.exists():
                result["status"] = "error"
                result["error"] = {
                    "type": "snapshots_not_found",
                    "message": f"No snapshots directory found for {resolved_name}"
                }
                return result
            
            hash_dir = snapshots_dir / commit_hash
            if not hash_dir.exists():
                # List available hashes
                available_hashes = [s.name[:8] for s in snapshots_dir.iterdir() if s.is_dir()]
                result["status"] = "error"
                result["error"] = {
                    "type": "hash_not_found",
                    "message": f"Hash {commit_hash} not found",
                    "available_hashes": available_hashes
                }
                return result
            
            result["data"]["action"] = "delete_hash"
            result["data"]["commit_hash"] = commit_hash
        else:
            result["data"]["action"] = "delete_model"
        
        # Perform deletion (with optional strict test safety)
        if force or not result["data"]["requires_confirmation"]:
            # Optional safety: when running tests, enforce test cache context
            if os.environ.get("MLXK2_STRICT_TEST_DELETE") == "1":
                cache_path = str(get_current_model_cache())
                if "mlxk2_test_" not in cache_path:
                    raise RuntimeError(f"STRICT_TEST_DELETE: Refusing to delete from non-test cache: {cache_path}")
            # MLX-Knife 2.0 Fix: Always delete entire model directory
            # This prevents the Issue #23 double-execution problem
            shutil.rmtree(resolved_model_dir)
            
            # Clean up lock files
            lock_count = cleanup_model_locks(resolved_name)
            result["data"]["lock_files_cleaned"] = lock_count
            
            if commit_hash:
                result["data"]["message"] = f"Deleted {resolved_name}@{commit_hash}"
            else:
                result["data"]["message"] = f"Deleted entire model {resolved_name}"
            
            result["data"]["action"] = "deleted"
    
    except PermissionError as e:
        result["status"] = "error"
        result["error"] = {
            "type": "permission_denied",
            "message": f"Permission denied: Cannot delete {e.filename}"
        }
    except Exception as e:
        result["status"] = "error"
        result["error"] = {
            "type": "deletion_failed",
            "message": str(e)
        }
    
    return result
