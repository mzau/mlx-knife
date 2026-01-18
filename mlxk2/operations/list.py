"""List models operation for MLX-Knife 2.0."""

from pathlib import Path
from typing import Dict, Any, Optional, Tuple

from ..core.cache import get_current_model_cache, cache_dir_to_hf
from .common import build_model_object
from .workspace import find_matching_workspaces, is_explicit_path


def _compute_display_name(workspace_path: Path, pattern: str) -> str:
    """Compute display name for workspace based on input pattern.

    If pattern is relative (./..., ../...), return relative path.
    If pattern is absolute (/...), return absolute path.

    Args:
        workspace_path: Resolved absolute path to workspace
        pattern: Original input pattern

    Returns:
        Display name matching input pattern style
    """
    if pattern.startswith('/'):
        # Absolute pattern → absolute output
        return str(workspace_path)

    # Relative pattern → relative output
    try:
        pattern_path = Path(pattern).expanduser()
        pattern_resolved = pattern_path.resolve()

        # Case 1: Exact workspace match (pattern points to this workspace)
        # Display name is the workspace directory name
        if pattern_resolved == workspace_path:
            return workspace_path.name

        # Case 2: Directory scan (pattern is parent directory containing workspace)
        # Display name is relative to that directory
        if pattern_path.exists() and pattern_path.is_dir():
            return str(workspace_path.relative_to(pattern_resolved))

        # Case 3: Prefix match (pattern is partial name)
        # Display name is relative to parent directory
        search_dir = pattern_path.parent.resolve()
        return str(workspace_path.relative_to(search_dir))
    except ValueError:
        # Can't compute relative path, fall back to absolute
        return str(workspace_path)


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
        pattern: Optional pattern to filter models (case-insensitive substring match),
                 or a workspace path pattern to list local models.

    Workspace patterns (start with ./, ../, or /):
        - Exact: "./my-model" → list single workspace
        - Prefix: "./gemma-" → list all workspaces starting with "gemma-"
    """
    # Check for workspace path patterns first (ADR-018 Phase 0c compatibility)
    # Explicit paths (./foo, ../foo, /foo) are always treated as workspace patterns
    if pattern and is_explicit_path(pattern):
        workspace_matches = find_matching_workspaces(pattern)
        models = []
        for workspace_path in workspace_matches:
            model_obj = build_model_object(
                str(workspace_path),  # hf_name = absolute path for workspaces
                workspace_path,       # model_root
                workspace_path        # selected_path (no snapshots in workspace)
            )
            # Add display_name for human output (respects input pattern style)
            model_obj["display_name"] = _compute_display_name(workspace_path, pattern)
            models.append(model_obj)
        # Return workspace results (may be empty if no matches)
        # Do NOT fall through to cache search for explicit paths
        return {
            "status": "success",
            "command": "list",
            "data": {
                "models": models,
                "count": len(models)
            },
            "error": None
        }

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
