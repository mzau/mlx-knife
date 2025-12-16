"""Shared utilities for live E2E tests (ADR-011).

Provides:
- Portfolio discovery functions (reused from test_stop_tokens_live.py)
- RAM gating utilities
- Common test constants
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, Any, Tuple

# Import portfolio discovery infrastructure from test_stop_tokens_live.py
_parent_dir = Path(__file__).parent.parent
sys.path.insert(0, str(_parent_dir))

try:
    from test_stop_tokens_live import (
        discover_mlx_models_in_user_cache,
        get_safe_ram_budget_gb,
        get_system_ram_gb,
        should_skip_model,
        TEST_MODELS,
    )
finally:
    sys.path.remove(str(_parent_dir))


# RAM calculation utilities (modularized for different model types)

def calculate_text_model_ram_gb(size_bytes: int) -> float:
    """Calculate RAM requirement for text-only models.

    Text models use 1.2x overhead for inference based on empirical testing.
    This accounts for model weights + KV cache + temporary buffers.

    Args:
        size_bytes: Model size in bytes (from mlxk list --json)

    Returns:
        Estimated RAM needed in GB

    References:
        - ADR-009 (Stop Token Portfolio Discovery)
        - ADR-011 (E2E Test Architecture)
        - TESTING-DETAILS.md "RAM-Aware Model Selection"
    """
    return (size_bytes / (1024**3)) * 1.2


def calculate_vision_model_ram_gb(size_bytes: int, system_memory_bytes: int) -> float:
    """Calculate RAM requirement for vision models.

    Vision models have different memory characteristics:
    - Vision Encoder adds significant overhead (varies by model)
    - Metal OOM crashes occur above 70% system memory (ADR-016)
    - No simple multiplier - use direct threshold check

    Args:
        size_bytes: Model size in bytes (from mlxk list --json)
        system_memory_bytes: Total system RAM in bytes

    Returns:
        Estimated RAM needed in GB. Returns float('inf') if model exceeds
        70% system memory threshold (signals: skip this model).

    References:
        - ADR-012 (Vision Support Roadmap)
        - ADR-016 (Memory-Aware Model Loading)
        - capabilities.py MEMORY_THRESHOLD_PERCENT = 0.70
    """
    if system_memory_bytes == 0:
        return float('inf')  # Cannot determine, skip

    memory_ratio = size_bytes / system_memory_bytes

    # Vision models crash above 70% due to Vision Encoder overhead
    if memory_ratio > 0.70:
        return float('inf')  # Signal: Too large, will be skipped

    # Return actual size (no 1.2x multiplier for vision)
    # Encoder overhead is handled by conservative 0.70 threshold
    return size_bytes / (1024**3)


def get_system_memory_bytes() -> int:
    """Get total system memory in bytes via sysctl (macOS).

    Returns:
        Total physical RAM in bytes, or 0 if unavailable.
    """
    import subprocess

    try:
        result = subprocess.run(
            ["sysctl", "-n", "hw.memsize"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0:
            return int(result.stdout.strip())
    except (subprocess.SubprocessError, ValueError, FileNotFoundError):
        pass

    return 0


def discover_text_models() -> list[Dict[str, Any]]:
    """Discover text-only models (filter out Vision models).

    Uses discover_mlx_models_in_user_cache() and filters out models
    with "vision" in their capabilities list.

    This enables deterministic text-only test portfolios that won't
    change when Vision models are added/removed from cache.

    Returns:
        List of text-only model dicts (same format as discover_mlx_models_in_user_cache):
        [{"model_id": "...", "ram_needed_gb": X.X, "snapshot_path": None, "weight_count": None}, ...]
    """
    import json
    import subprocess
    import os

    # Get all discovered models (already filtered: MLX + healthy + runtime_compatible + chat)
    all_models = discover_mlx_models_in_user_cache()
    if not all_models:
        return []

    # Get capabilities from mlxk list --json
    env = os.environ.copy()
    if not env.get("HF_HOME"):
        return all_models  # Fall back to all models if HF_HOME not set

    try:
        result = subprocess.run(
            [sys.executable, "-m", "mlxk2.cli", "list", "--json"],
            capture_output=True,
            text=True,
            timeout=30,
            env=env
        )

        if result.returncode != 0:
            return all_models  # Fall back to all models

        # Parse JSON and build vision model ID set
        data = json.loads(result.stdout)
        models = data.get("data", {}).get("models", [])

        vision_model_ids = {
            m["name"] for m in models
            if "vision" in m.get("capabilities", [])
        }

        # Filter out vision models
        return [m for m in all_models if m["model_id"] not in vision_model_ids]

    except Exception:
        return all_models  # Fall back to all models on error


def discover_vision_models() -> list[Dict[str, Any]]:
    """Discover vision-capable models only.

    Uses discover_mlx_models_in_user_cache() and filters to only models
    with "vision" in their capabilities list.

    IMPORTANT: Recalculates RAM requirements using Vision-specific formula
    (ADR-016 0.70 threshold instead of 1.2x multiplier).

    This enables deterministic vision-only test portfolios separate from
    text-only tests.

    Returns:
        List of vision-capable model dicts (same format as discover_mlx_models_in_user_cache):
        [{"model_id": "...", "ram_needed_gb": X.X, "snapshot_path": None, "weight_count": None}, ...]
    """
    import json
    import subprocess
    import os

    # Get all discovered models (already filtered: MLX + healthy + runtime_compatible + chat)
    all_models = discover_mlx_models_in_user_cache()
    if not all_models:
        return []

    # Get capabilities and size_bytes from mlxk list --json
    env = os.environ.copy()
    if not env.get("HF_HOME"):
        return []  # Vision models need HF_HOME

    try:
        result = subprocess.run(
            [sys.executable, "-m", "mlxk2.cli", "list", "--json"],
            capture_output=True,
            text=True,
            timeout=30,
            env=env
        )

        if result.returncode != 0:
            return []

        # Parse JSON and build vision model data
        data = json.loads(result.stdout)
        models_list = data.get("data", {}).get("models", [])

        # Build map: model_id -> (is_vision, size_bytes)
        model_info = {}
        for m in models_list:
            model_name = m["name"]
            is_vision = "vision" in m.get("capabilities", [])
            size_bytes = m.get("size_bytes", 0)
            model_info[model_name] = (is_vision, size_bytes)

        # Get system memory for vision RAM calculation
        system_memory_bytes = get_system_memory_bytes()

        # Filter to only vision models + recalculate RAM
        vision_models = []
        for model in all_models:
            model_id = model["model_id"]
            if model_id in model_info:
                is_vision, size_bytes = model_info[model_id]
                if is_vision:
                    # Recalculate RAM using Vision-specific formula
                    ram_gb = calculate_vision_model_ram_gb(size_bytes, system_memory_bytes)

                    # Create new dict with updated RAM
                    vision_model = model.copy()
                    vision_model["ram_needed_gb"] = ram_gb
                    vision_models.append(vision_model)

        return vision_models

    except Exception:
        return []


# Re-export for convenience
__all__ = [
    "discover_mlx_models_in_user_cache",
    "discover_text_models",
    "discover_vision_models",
    "calculate_text_model_ram_gb",
    "calculate_vision_model_ram_gb",
    "get_system_memory_bytes",
    "get_safe_ram_budget_gb",
    "get_system_ram_gb",
    "should_skip_model",
    "TEST_MODELS",
    "TEST_PROMPT",
    "MAX_TOKENS",
    "TEST_TEMPERATURE",
]


# Standard test constants (shared across all E2E tests)
TEST_PROMPT = "Write one sentence about cats."
MAX_TOKENS = 50
TEST_TEMPERATURE = 0.0  # Deterministic sampling for reproducible tests
