"""
Model listing handler for /v1/models endpoint.

Extracted from server_base.py as part of Phase 1 refactoring.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Dict, Optional


def _get_logger():
    """Lazy import logger to avoid circular dependencies."""
    from ....logging import get_logger
    return get_logger()


def _best_effort_context_length(model_path) -> Optional[int]:
    """Read the model's context length, returning None on any failure."""
    try:
        from ...runner import get_model_context_length
        return get_model_context_length(str(model_path))
    except Exception:
        return None


async def handle_list_models(
    get_cache_fn: Callable[[], Path],
    preload_model: Optional[str],
) -> Dict[str, Any]:
    """List available MLX models from the HF cache and the workspace home.

    Mirrors the default human `mlxk list` view (Issue #58, ADR-022): runnable
    (healthy + runtime_compatible) models from both the HF cache and
    MLXK_WORKSPACE_HOME. Workspace models are advertised by their directory
    basename — a stable short id that resolves at request time via
    workspace-first resolution — not by their absolute path.

    Returns models sorted with preloaded model first (if set), then alphabetically.

    Args:
        get_cache_fn: Function returning the model cache path
        preload_model: Pre-loaded model name for sorting priority (or None)

    Returns:
        Dict with "object": "list", "data": [...models...]
    """
    from ...cache import cache_dir_to_hf
    from ....operations.common import build_model_object
    from ....operations.list import _latest_snapshot
    from ....operations.workspace import get_workspace_home, is_workspace_path

    logger = _get_logger()
    model_list = []

    # ADR-022: workspace-home models (workspace-first paradigm),
    # same discovery as mlxk2/operations/list.py::list_models
    workspace_home = get_workspace_home()
    # Resolved path -> advertised id (symlinked dirs: advertised under link name)
    listed_workspace_ids: Dict[str, str] = {}
    if workspace_home:
        for ws_dir in workspace_home.iterdir():
            if not ws_dir.is_dir() or not is_workspace_path(str(ws_dir)):
                continue
            try:
                model_obj = build_model_object(str(ws_dir), ws_dir, ws_dir)

                # Filter: healthy AND runtime_compatible
                if model_obj.get("health") != "healthy":
                    continue
                if not model_obj.get("runtime_compatible"):
                    continue

                model_list.append({
                    "id": ws_dir.name,  # Stable short id (resolves workspace-first)
                    "object": "model",
                    "owned_by": "workspace",
                    "permission": [],
                    "context_length": _best_effort_context_length(ws_dir),
                })
                listed_workspace_ids[str(ws_dir.resolve())] = ws_dir.name
            except Exception as e:
                # Skip models that can't be processed
                logger.warning(f"Skipping workspace model {ws_dir.name} from /v1/models: {e}")
                continue

    model_cache = get_cache_fn()

    # Find all model directories (handle missing cache gracefully)
    if not model_cache.exists():
        # Fresh installation or custom cache location - no models yet
        models = []
    else:
        models = [d for d in model_cache.iterdir() if d.name.startswith("models--")]

    for model_dir in models:
        model_name = cache_dir_to_hf(model_dir.name)

        # Hide test sentinel directories from listings (as in list_models)
        if "TEST-CACHE-SENTINEL" in model_name:
            continue

        try:
            # Select snapshot (prefer latest, as in list_models)
            _hash, selected_path = _latest_snapshot(model_dir)

            # Use shared build_model_object (single source of truth)
            model_obj = build_model_object(
                model_name, model_dir,
                selected_path if selected_path is not None else model_dir,
            )

            # Filter: healthy AND runtime_compatible
            if model_obj.get("health") != "healthy":
                continue
            if not model_obj.get("runtime_compatible"):
                continue

            # Get model context length (best effort)
            context_length = None
            if selected_path:
                context_length = _best_effort_context_length(selected_path)

            model_list.append({
                "id": model_name,
                "object": "model",
                "owned_by": "mlx-knife-2.0",
                "permission": [],
                "context_length": context_length,
            })
        except Exception as e:
            # Skip models that can't be processed
            logger.warning(f"Skipping model {model_name} from /v1/models: {e}")
            continue

    # Preloaded workspace model: usually already covered by the workspace scan
    # above (then advertised by basename). An explicit-path preload outside the
    # workspace home is added here — subject to the same runnable filter:
    # clients must only ever see runnable models.
    preload_id = preload_model
    if preload_model and is_workspace_path(preload_model):
        preload_path = Path(preload_model).resolve()
        if str(preload_path) in listed_workspace_ids:
            preload_id = listed_workspace_ids[str(preload_path)]
        else:
            not_runnable_reason = None
            try:
                model_obj = build_model_object(str(preload_path), preload_path, preload_path)
                runnable = (
                    model_obj.get("health") == "healthy"
                    and bool(model_obj.get("runtime_compatible"))
                )
                if not runnable:
                    not_runnable_reason = model_obj.get("reason") or "health check failed"
            except Exception as e:
                logger.warning(f"Skipping preloaded model {preload_model} from /v1/models: {e}")
                runnable = False
            if runnable:
                # Basename only resolves via workspace home; otherwise keep the path
                if workspace_home and preload_path.parent == workspace_home:
                    preload_id = preload_path.name
                else:
                    preload_id = preload_model
                if not any(m["id"] == preload_id for m in model_list):
                    model_list.append({
                        "id": preload_id,
                        "object": "model",
                        "owned_by": "workspace",
                        "permission": [],
                        "context_length": _best_effort_context_length(preload_path),
                    })
            elif not_runnable_reason:
                # The model is loaded and serving, but clients must only ever
                # see runnable models. Surface the mismatch for the operator
                # (e.g. integrity false-negatives) instead of hiding silently.
                logger.warning(
                    f"Preloaded model {preload_model} is loaded but hidden from "
                    f"/v1/models (not runnable: {not_runnable_reason})"
                )

    # Sort: preloaded model first, then alphabetically by id
    if preload_id:
        def sort_key(model: Dict[str, Any]):
            # Preloaded model gets priority (0), others sorted alphabetically
            return (0 if model["id"] == preload_id else 1, model["id"])
        model_list.sort(key=sort_key)
    else:
        # No preload: just alphabetical
        model_list.sort(key=lambda m: m["id"])

    return {"object": "list", "data": model_list}
