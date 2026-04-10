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


async def handle_list_models(
    get_cache_fn: Callable[[], Path],
    preload_model: Optional[str],
) -> Dict[str, Any]:
    """List available MLX models in the cache.

    Returns models sorted with preloaded model first (if set), then alphabetically.
    Filters to healthy + runtime_compatible models.

    Args:
        get_cache_fn: Function returning the model cache path
        preload_model: Pre-loaded model name for sorting priority (or None)

    Returns:
        Dict with "object": "list", "data": [...models...]
    """
    from ...cache import cache_dir_to_hf
    from ....operations.common import build_model_object

    logger = _get_logger()
    model_list = []
    model_cache = get_cache_fn()

    # Find all model directories (handle missing cache gracefully)
    if not model_cache.exists():
        # Fresh installation or custom cache location - no models yet
        models = []
    else:
        models = [d for d in model_cache.iterdir() if d.name.startswith("models--")]

    for model_dir in models:
        model_name = cache_dir_to_hf(model_dir.name)

        try:
            # Get snapshot path
            snapshots_dir = model_dir / "snapshots"
            selected_path = None
            if snapshots_dir.exists():
                snapshots = [d for d in snapshots_dir.iterdir() if d.is_dir()]
                if snapshots:
                    selected_path = snapshots[0]

            # Use shared build_model_object (single source of truth)
            model_obj = build_model_object(model_name, model_dir, selected_path)

            # Filter: healthy AND runtime_compatible
            if model_obj.get("health") != "healthy":
                continue
            if not model_obj.get("runtime_compatible"):
                continue

            # Get model context length (best effort)
            context_length = None
            try:
                if selected_path:
                    from ...runner import get_model_context_length
                    context_length = get_model_context_length(str(selected_path))
            except Exception:
                pass

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

    # Add preloaded workspace if present and not already in list
    if preload_model:
        # Check if it's a workspace path
        from ....operations.workspace import is_workspace_path
        if is_workspace_path(preload_model):
            # Check if already in list (avoid duplicates)
            if not any(m["id"] == preload_model for m in model_list):
                # Get context length
                context_length = None
                try:
                    from ...runner import get_model_context_length
                    context_length = get_model_context_length(preload_model)
                except Exception:
                    pass

                model_list.append({
                    "id": preload_model,  # Original path string
                    "object": "model",
                    "owned_by": "workspace",
                    "permission": [],
                    "context_length": context_length,
                })

    # Sort: preloaded model first, then alphabetically by id
    if preload_model:
        def sort_key(model: Dict[str, Any]):
            # Preloaded model gets priority (0), others sorted alphabetically
            return (0 if model["id"] == preload_model else 1, model["id"])
        model_list.sort(key=sort_key)
    else:
        # No preload: just alphabetical
        model_list.sort(key=lambda m: m["id"])

    return {"object": "list", "data": model_list}
