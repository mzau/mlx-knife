import json
from ..core.cache import get_current_model_cache, hf_to_cache_dir, cache_dir_to_hf
from ..core.model_resolution import resolve_model_for_operation


def is_model_healthy(model_spec):
    """Framework-agnostic health check accepting model names like 1.1.0."""
    from ..core.model_resolution import resolve_model_for_operation
    
    # Resolve model name to get actual cache directory
    resolved_name, commit_hash, ambiguous_matches = resolve_model_for_operation(model_spec)
    
    if ambiguous_matches or not resolved_name:
        return False, "Could not resolve model spec"
    
    # Get the model cache directory (models--namespace--name)
    model_cache = get_current_model_cache()
    model_cache_dir = model_cache / hf_to_cache_dir(resolved_name)
    if not model_cache_dir.exists():
        return False, "Model not in cache"
    
    # Find the appropriate snapshot to check
    snapshots_dir = model_cache_dir / "snapshots"
    if not snapshots_dir.exists():
        return False, "No snapshots directory found"
    
    snapshots = [d for d in snapshots_dir.iterdir() if d.is_dir()]
    if not snapshots:
        return False, "No snapshots found"
    
    # Use specific hash if provided, otherwise latest snapshot
    if commit_hash:
        model_path = snapshots_dir / commit_hash
        if not model_path.exists():
            return False, f"Specific hash {commit_hash} not found"
    else:
        model_path = max(snapshots, key=lambda x: x.stat().st_mtime)
    
    # Now do the actual health check on the snapshot
    return _check_snapshot_health(model_path)


def _check_snapshot_health(model_path):
    """Check health of a specific snapshot directory.

    Rules (Issue #27 parity):
    - If a multi-file safetensors index exists (model.safetensors.index.json),
      ALL referenced shard files must exist and be non-empty, and none may be LFS pointers.
      A subset must NOT be marked healthy.
    - Without an index, require at least one weight file present and non-empty,
      and ensure none are LFS pointers.
    """
    if not model_path.exists():
        return False, "Model path does not exist"
    
    # Check config.json
    config_path = model_path / "config.json"
    if not config_path.exists():
        return False, "config.json missing"
    
    try:
        with open(config_path) as f:
            config_data = json.load(f)
        if not isinstance(config_data, dict) or len(config_data) == 0:
            return False, "config.json is empty or invalid"
    except (OSError, json.JSONDecodeError):
        return False, "config.json contains invalid JSON"
    
    # If a multi-file safetensors index exists, enforce completeness
    index_file = model_path / "model.safetensors.index.json"
    if index_file.exists():
        try:
            with open(index_file) as f:
                index = json.load(f)
            weight_map = index.get('weight_map') or {}
            if not isinstance(weight_map, dict) or not weight_map:
                return False, "Empty or invalid weight_map in index"
            referenced_files = sorted(set(weight_map.values()))
            missing = [rf for rf in referenced_files if not (model_path / rf).exists()]
            if missing:
                return False, f"Missing weight shards: {', '.join(missing)}"
            empty = [rf for rf in referenced_files if (model_path / rf).stat().st_size == 0]
            if empty:
                return False, f"Empty weight shards: {', '.join(empty)}"
            # LFS pointer check on referenced files
            lfs_bad = []
            for rf in referenced_files:
                fp = (model_path / rf)
                if fp.is_file() and fp.stat().st_size < 200:
                    try:
                        with open(fp, 'rb') as f:
                            header = f.read(100)
                            if b'version https://git-lfs.github.com/spec/v1' in header:
                                lfs_bad.append(rf)
                    except Exception:
                        pass
            if lfs_bad:
                return False, f"LFS pointers instead of files: {', '.join(lfs_bad)}"
            return True, "Multi-file model complete"
        except (OSError, json.JSONDecodeError):
            return False, "Invalid safetensors index file"

    # No index: Check weight files (supports common formats)
    weight_files = (
        list(model_path.glob("*.safetensors")) +
        list(model_path.glob("*.bin")) +
        list(model_path.glob("*.gguf"))
    )
    if not weight_files:
        weight_files = (
            list(model_path.glob("**/*.safetensors")) +
            list(model_path.glob("**/*.bin")) +
            list(model_path.glob("**/*.gguf"))
        )
    # Pattern-based completeness (no index): model-XXXXX-of-YYYYY.safetensors
    # If such shards are present, require full set to be present and non-empty
    if weight_files:
        import re
        shard_regex = re.compile(r"model-(\d{5})-of-(\d{5})\.safetensors$")
        shards = []
        for f in weight_files:
            m = shard_regex.search(f.name)
            if m:
                idx = int(m.group(1))
                total = int(m.group(2))
                shards.append((idx, total, f))
        if shards:
            totals = {t for (_, t, _) in shards}
            if len(totals) != 1:
                return False, "Inconsistent shard totals detected"
            expected_total = next(iter(totals))
            present_indices = {i for (i, _, _) in shards}
            missing_indices = [i for i in range(1, expected_total + 1) if i not in present_indices]
            if missing_indices:
                return False, f"Missing shards by pattern: {len(present_indices)}/{expected_total} present"
            empties = [f.name for (_, _, f) in shards if f.stat().st_size == 0]
            if empties:
                return False, f"Empty shards: {', '.join(empties)}"
    if not weight_files:
        return False, "No model weights found"

    # Partial download markers → unhealthy
    for fp in model_path.rglob("*"):
        if fp.is_file():
            name = fp.name.lower()
            if name.endswith('.partial') or name.endswith('.tmp') or 'partial' in name:
                return False, "Partial download marker detected"

    # Ensure files are non-empty
    if any(f.stat().st_size == 0 for f in weight_files):
        empties = [f.name for f in weight_files if f.stat().st_size == 0]
        return False, f"Empty weight files: {', '.join(empties)}"

    # Pattern-based completeness (no index): model-XXXXX-of-YYYYY.safetensors
    # If such shards are present but no index, mark unhealthy (index required for sharded models)
    import re
    shard_regex = re.compile(r"model-(\d{5})-of-(\d{5})\.safetensors$")
    shards = []
    for f in weight_files:
        m = shard_regex.search(f.name)
        if m:
            idx = int(m.group(1))
            total = int(m.group(2))
            shards.append((idx, total, f))
    if shards:
        totals = {t for (_, t, _) in shards}
        if len(totals) != 1:
            return False, "Inconsistent shard totals detected"
        expected_total = next(iter(totals))
        present_indices = {i for (i, _, _) in shards}
        missing_indices = [i for i in range(1, expected_total + 1) if i not in present_indices]
        if missing_indices:
            return False, f"Missing shards by pattern: {len(present_indices)}/{expected_total} present"
        # Even if complete by pattern, absence of index is unhealthy (robust policy)
        return False, "Safetensors index missing for sharded model"

    # LFS pointer scan (recursive simplified)
    lfs_ok, lfs_msg = check_lfs_corruption(model_path)
    if not lfs_ok:
        return False, lfs_msg

    return True, "Model is healthy"


def check_lfs_corruption(model_path):
    """Check for Git LFS pointer files instead of actual model files (recursive)."""
    corrupted_files = []
    for file_path in model_path.rglob("*"):
        if file_path.is_file() and file_path.stat().st_size < 200:
            try:
                with open(file_path, 'rb') as f:
                    header = f.read(100)
                    if b'version https://git-lfs.github.com/spec/v1' in header:
                        corrupted_files.append(str(file_path.relative_to(model_path)))
            except Exception:
                pass
    
    if corrupted_files:
        return False, f"LFS pointers instead of files: {', '.join(corrupted_files)}"
    return True, "No LFS corruption detected"


def health_check_operation(model_pattern=None):
    """Health check operation for JSON API with model resolution support."""
    result = {
        "status": "success",
        "command": "health",
        "error": None,
        "data": {
            "healthy": [],
            "unhealthy": [],
            "summary": {
                "total": 0,
                "healthy_count": 0,
                "unhealthy_count": 0
            }
        }
    }
    
    try:
        model_cache = get_current_model_cache()
        if not model_cache.exists():
            result["data"]["summary"]["total"] = 0
            return result
        
        # Use model resolution if specific pattern provided
        if model_pattern:
            resolved_name, commit_hash, ambiguous_matches = resolve_model_for_operation(model_pattern)
            
            if ambiguous_matches:
                # Multiple matches - let user choose
                result["status"] = "error"
                result["error"] = {
                    "type": "ambiguous_match",
                    "message": f"Multiple models match '{model_pattern}'",
                    "matches": ambiguous_matches
                }
                return result
            elif not resolved_name:
                # No matches found
                result["data"]["summary"]["total"] = 0
                return result
            else:
                # Single match found - check just this model
                model_cache_dir = model_cache / hf_to_cache_dir(resolved_name)
                if model_cache_dir.exists():
                    models_to_check = [model_cache_dir]
                else:
                    models_to_check = []
        else:
            # No pattern - check all models
            models_to_check = [d for d in model_cache.iterdir() if d.name.startswith("models--")]
        
        result["data"]["summary"]["total"] = len(models_to_check)
        
        for model_dir in sorted(models_to_check, key=lambda x: x.name):
            hf_name = cache_dir_to_hf(model_dir.name)
            
            # Use the new flexible health check
            healthy, reason = is_model_healthy(hf_name)
            
            model_info = {
                "name": hf_name,
                "status": "healthy" if healthy else "unhealthy", 
                "reason": reason
            }
            
            if healthy:
                result["data"]["healthy"].append(model_info)
                result["data"]["summary"]["healthy_count"] += 1
            else:
                result["data"]["unhealthy"].append(model_info)
                result["data"]["summary"]["unhealthy_count"] += 1
    
    except Exception as e:
        result["status"] = "error"
        result["error"] = {
            "type": "health_check_failed",
            "message": str(e)
        }
    
    return result
