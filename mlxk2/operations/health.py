import json
import logging
from pathlib import Path
from typing import Tuple, Optional
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


def _check_auxiliary_assets(model_path, config_data):
    """Check vision and tokenizer auxiliary assets.

    ADR-012 Phase 2: Auxiliary asset validation
    - Vision models require preprocessor_config.json (for image processing)
    - Chat models benefit from tokenizer assets (tokenizer.json, tokenizer_config.json)

    Returns:
        (bool, str): (is_ok, reason_message)
    """
    from .common import detect_vision_capability
    is_vision = detect_vision_capability(model_path, config_data)

    if is_vision:
        # Vision models require preprocessor_config.json for mlx-vlm
        preprocessor_path = model_path / "preprocessor_config.json"
        if not preprocessor_path.exists():
            return False, "Vision model missing preprocessor_config.json"

        try:
            with open(preprocessor_path) as f:
                preprocessor_data = json.load(f)
            if not isinstance(preprocessor_data, dict):
                return False, "preprocessor_config.json invalid"
        except (OSError, json.JSONDecodeError):
            return False, "preprocessor_config.json invalid JSON"

    # Chat models benefit from tokenizer assets (not strict requirement for base models)
    # Check tokenizer_config.json for chat template support
    tokenizer_config_path = model_path / "tokenizer_config.json"
    if tokenizer_config_path.exists():
        try:
            with open(tokenizer_config_path) as f:
                tokenizer_data = json.load(f)
            if not isinstance(tokenizer_data, dict):
                return False, "tokenizer_config.json exists but invalid"
        except (OSError, json.JSONDecodeError):
            return False, "tokenizer_config.json contains invalid JSON"

        # If tokenizer_config exists, tokenizer.json should also exist
        tokenizer_path = model_path / "tokenizer.json"
        if not tokenizer_path.exists():
            return False, "tokenizer_config.json present but tokenizer.json missing"

    return True, "Auxiliary assets OK"


def _check_snapshot_health(model_path):
    """Check health of a specific snapshot directory.

    Rules (Issue #27 parity):
    - If a multi-file safetensors index exists (model.safetensors.index.json),
      ALL referenced shard files must exist and be non-empty, and none may be LFS pointers.
      A subset must NOT be marked healthy.
    - Without an index, require at least one weight file present and non-empty,
      and ensure none are LFS pointers.

    ADR-012 Phase 2: Auxiliary asset validation
    - Vision models require preprocessor_config.json (for image processing)
    - Chat models benefit from tokenizer assets (tokenizer.json, tokenizer_config.json)
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
    
    # Prefer safetensors index; else fall back to PyTorch index
    sft_index = model_path / "model.safetensors.index.json"
    pt_index = model_path / "pytorch_model.bin.index.json"
    has_sft_files = any(model_path.rglob("*.safetensors"))
    has_bin_files = any(model_path.rglob("*.bin"))

    chosen_index = None
    if sft_index.exists() and has_sft_files:
        chosen_index = ("sft", sft_index)
    elif pt_index.exists() and has_bin_files:
        chosen_index = ("pt", pt_index)

    if chosen_index is not None:
        kind, index_file = chosen_index
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
            # ADR-012 Phase 2: Vision/tokenizer checks for indexed models
            aux_ok, aux_msg = _check_auxiliary_assets(model_path, config_data)
            if not aux_ok:
                return False, aux_msg
            return True, "Multi-file model complete"
        except (OSError, json.JSONDecodeError):
            return False, "Invalid index file"

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

    # ADR-012 Phase 2: Vision/tokenizer checks for non-indexed models
    aux_ok, aux_msg = _check_auxiliary_assets(model_path, config_data)
    if not aux_ok:
        return False, aux_msg

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


def health_from_cache(model_spec, cache_dir):
    """Health check for a specific model in a specific cache directory.

    This is used by clone operations to check model health in temporary caches
    without contaminating the user's main cache. Uses the full _check_snapshot_health()
    logic to ensure identical health validation standards.

    Args:
        model_spec: Model name/spec to check (e.g., "microsoft/DialoGPT-small")
        cache_dir: Path to the cache directory containing the model

    Returns:
        (bool, str): (is_healthy, reason_message)
    """
    from pathlib import Path
    from ..core.cache import hf_to_cache_dir

    cache_path = Path(cache_dir)

    # Convert model spec to cache directory format
    model_cache_dir = cache_path / hf_to_cache_dir(model_spec)
    if not model_cache_dir.exists():
        return False, "Model not in cache"

    # Find the appropriate snapshot to check
    snapshots_dir = model_cache_dir / "snapshots"
    if not snapshots_dir.exists():
        return False, "No snapshots directory found"

    snapshots = [d for d in snapshots_dir.iterdir() if d.is_dir()]
    if not snapshots:
        return False, "No snapshots found"

    # Use the latest snapshot (by modification time)
    model_path = max(snapshots, key=lambda x: x.stat().st_mtime)

    # Use the same health check logic as regular health operations
    return _check_snapshot_health(model_path)


def check_runtime_compatibility(model_path: Path, framework: str) -> Tuple[bool, Optional[str]]:
    """Check if model is executable with mlx-lm.

    Gate logic:
    1. Framework must be "MLX" (GGUF/PyTorch → incompatible)
    2. Weight files must use mlx-lm compatible naming (not legacy formats)
    3. model_type must be supported by current mlx-lm version

    Returns:
        (is_compatible, reason): reason is None if compatible, error message otherwise
    """
    # Gate 1: Framework check
    if framework != "MLX":
        return False, f"Incompatible: {framework}"

    # Gate 2: Weight file format check (legacy format detection)
    # mlx-lm only accepts:
    # - model.safetensors (single file)
    # - model-XXXXX-of-YYYYY.safetensors (sharded, with index)
    # Legacy formats are rejected: weights.*.safetensors, pytorch_model-*.safetensors
    import re

    # Check for legacy weight file patterns
    legacy_patterns = [
        re.compile(r'^weights\.\d+\.safetensors$'),  # weights.00.safetensors
        re.compile(r'^pytorch_model-\d+\.safetensors$'),  # pytorch_model-00001.safetensors
    ]

    # Check for valid mlx-lm weight file patterns
    valid_patterns = [
        re.compile(r'^model\.safetensors$'),  # Single file
        re.compile(r'^model-\d{5}-of-\d{5}\.safetensors$'),  # Sharded
    ]

    weight_files = list(model_path.glob("*.safetensors"))
    if weight_files:
        has_valid = any(
            any(pattern.match(f.name) for pattern in valid_patterns)
            for f in weight_files
        )
        has_legacy = any(
            any(pattern.match(f.name) for pattern in legacy_patterns)
            for f in weight_files
        )

        if has_legacy and not has_valid:
            # Found only legacy format files, no valid mlx-lm files
            return False, "Legacy format not supported by mlx-lm"

    # Gate 3: model_type support check via mlx-lm
    config_path = model_path / "config.json"
    if not config_path.exists():
        return False, "config.json missing (required for model_type detection)"

    try:
        with open(config_path) as f:
            config = json.load(f)
        model_type = config.get("model_type")
        if not model_type:
            return False, "config.json missing model_type field"
    except (OSError, json.JSONDecodeError) as e:
        return False, f"Failed to read config.json: {e}"

    # Check if mlx-lm supports this model_type
    try:
        # Suppress mlx-lm's ERROR logs during detection
        # mlx-lm uses root logger, so we need to suppress both mlx_lm and root
        mlx_logger = logging.getLogger("mlx_lm")
        root_logger = logging.getLogger()
        original_mlx_level = mlx_logger.level
        original_root_level = root_logger.level
        mlx_logger.setLevel(logging.CRITICAL)
        root_logger.setLevel(logging.CRITICAL)

        try:
            # Try mlx-lm >= 0.28.0 API first (mlx_lm.models.base._get_classes)
            try:
                from mlx_lm.models.base import _get_classes
                model_class, _ = _get_classes(config=config, model_config=config)
            except ImportError:
                # Fall back to mlx-lm 0.27.x API (mlx_lm.utils._get_classes)
                from mlx_lm.utils import _get_classes
                model_class, _ = _get_classes(config)

            if model_class is None:
                return False, f"model_type '{model_type}' not supported by mlx-lm"

            return True, None
        finally:
            mlx_logger.setLevel(original_mlx_level)
            root_logger.setLevel(original_root_level)

    except Exception as e:
        # Pass through the actual error for debugging
        return False, str(e) if str(e) else "Runtime check failed"


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
