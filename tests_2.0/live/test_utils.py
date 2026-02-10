"""Shared utilities for live E2E tests (ADR-011).

Provides:
- Portfolio discovery functions (reused from test_stop_tokens_live.py)
- RAM gating utilities
- Common test constants
"""

from __future__ import annotations

import re
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


# =============================================================================
# KNOWN BROKEN MODELS - Upstream Runtime Bugs
# =============================================================================
# These models pass static health checks (files present, config valid) but fail
# at runtime initialization due to upstream mlx-lm/mlx-vlm bugs. They are
# excluded from portfolio discovery to prevent spurious test failures.
#
# Policy: Add models here ONLY when:
#   1. Static health check passes (healthy files)
#   2. Runtime initialization fails consistently
#   3. Root cause is verified upstream bug (not mlx-knife bug)
#   4. Issue is documented (session notes, upstream issue tracker)
#
# Format: Full HuggingFace model ID (org/name) - org matters for filtering!
# Note: BrokeC/ models are FIXED versions and should NOT be in this list
# =============================================================================

KNOWN_BROKEN_MODELS = {
    # transformers 5.0 video processor bug: "argument of type 'NoneType' is not iterable"
    # Root Cause: transformers sets video_processor=None when torchvision unavailable
    # Upstream: https://github.com/Blaizzy/mlx-vlm/issues/640
    # Details: docs/ISSUES/transformers-5.0-video-processor-bug.md
    # Test: `mlxk run mlx-community/MiMo-VL-7B-RL-bf16 "test" --image foo.jpg` → Error
    # Strategy: Waiting for mlx-vlm Issue #640 resolution
    "mlx-community/MiMo-VL-7B-RL-bf16",

    # transformers 5.0 video processor bug (same as MiMo-VL above)
    # Upstream: https://github.com/Blaizzy/mlx-vlm/issues/640
    # Details: docs/ISSUES/transformers-5.0-video-processor-bug.md
    # Test: `mlxk run mlx-community/Qwen2-VL-7B-Instruct-4bit "test" --image foo.jpg` → Error
    # Note: Image-only processing works, video processing broken
    "mlx-community/Qwen2-VL-7B-Instruct-4bit",

    # transformers 5.0 video processor bug (same as above)
    # Upstream: https://github.com/Blaizzy/mlx-vlm/issues/640
    # Details: docs/ISSUES/transformers-5.0-video-processor-bug.md
    # Test: `mlxk run Qwen3-Omni-30B-A3B-Instruct-4bit "test" --image foo.jpg` → Error
    # Note: Omni model (audio+video+vision) - all multimodal processing broken
    "mlx-community/Qwen3-Omni-30B-A3B-Instruct-4bit",

    # mlx-vlm vision feature mismatch: Image token positions (5476) ≠ features (1369)
    # Status: Upstream mlx-vlm vision encoder/model compatibility bug (separate from #624)
    # Test: `mlxk run ./Mistral-Small-3.1-24B-Instruct-2503-FIXED-4bit "test" --image foo.jpg` → Error
    # Note: --repair-index fixes #624 (index mismatch) but NOT this vision feature bug
    # Note: BrokeC/Mistral-Small-3.1... is the FIXED version (not in this list)
    "mlx-community/Mistral-Small-3.1-24B-Instruct-2503-4bit",

    # transformers 5.0.0rc3 trust_remote_code dialog blocks non-interactive tests
    # Root Cause: Model has custom code, transformers 5.0.0rc3 prompts Y/N dialog
    # Upstream: Needs mlx-lm issue (sharded_load sets trust_remote_code=True, load() doesn't)
    # Test: `mlxk run Klear-46B "test"` → hangs waiting for Y/N input
    # Strategy: Exclude until mlx-lm fixes trust_remote_code handling
    "mlx-community/Klear-46B-A2.5B-Instruct-3bit",

    # transformers 5.0 VoxtralProcessor hardcodes return_tensors="pt" (PyTorch only)
    # Root Cause: processing_voxtral.py line 61,192,327 reject non-PyTorch tensors
    # Error: "Unable to convert output to PyTorch tensors format, PyTorch is not installed."
    # Impact: Voxtral STT requires PyTorch (~2GB) - conflicts with lightweight goal
    # Test: `mlxk run Voxtral-Mini "test" --audio foo.wav` → ImportError
    # Strategy: Deferred - use Whisper for STT (works without PyTorch, excellent quality)
    # Watch: transformers upstream for MLX/NumPy tensor support
    "mlx-community/Voxtral-Mini-3B-2507-bf16",
}


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


def parse_vm_stat_page_size(output: str) -> int:
    """Extract vm_stat page size in bytes, falling back to 4096."""
    match = re.search(r"page size of (\d+) bytes", output)
    if match:
        return int(match.group(1))
    return 4096


def discover_text_models() -> list[Dict[str, Any]]:
    """Discover text-only models (filter out Vision and Audio models).

    Uses discover_mlx_models_in_user_cache() and filters out models
    with "vision" or "audio" in their capabilities list.

    This enables deterministic text-only test portfolios that won't
    change when Vision or Audio models are added/removed from cache.

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

        # Filter out vision AND audio models (text-only portfolio)
        non_text_model_ids = {
            m["name"] for m in models
            if "vision" in m.get("capabilities", []) or "audio" in m.get("capabilities", [])
        }

        # Filter out vision and audio models
        return [m for m in all_models if m["model_id"] not in non_text_model_ids]

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

            # Skip known broken models
            if model_id in KNOWN_BROKEN_MODELS:
                continue

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


def discover_audio_models() -> list[Dict[str, Any]]:
    """Discover audio-capable models only (ADR-020).

    Queries mlxk list --json directly and filters for:
    - model_type == "audio" (STT-only models: Whisper, Voxtral)
    - framework == "MLX" and health == "healthy" and runtime_compatible

    Note: This does NOT use discover_mlx_models_in_user_cache() because
    audio models have model_type="audio", not model_type="chat".

    Returns:
        List of audio model dicts:
        [{"model_id": "...", "ram_needed_gb": X.X, "repo_id": "...", ...}, ...]
    """
    import json
    import subprocess
    import os

    env = os.environ.copy()
    if not env.get("HF_HOME"):
        return []  # Audio discovery requires HF_HOME (see TESTING.md)

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

        data = json.loads(result.stdout)
        models_list = data.get("data", {}).get("models", [])

        # Get system memory for RAM calculation
        system_memory_bytes = get_system_memory_bytes()

        audio_models = []
        for m in models_list:
            # Filter: MLX + healthy + runtime_compatible + audio model_type
            if (m.get("framework") == "MLX" and
                m.get("health") == "healthy" and
                m.get("runtime_compatible") is True and
                m.get("model_type") == "audio"):

                model_name = m["name"]

                # Skip known broken models
                if model_name in KNOWN_BROKEN_MODELS:
                    continue

                # Calculate RAM using vision formula (conservative)
                size_bytes = m.get("size_bytes", 0)
                ram_gb = calculate_vision_model_ram_gb(size_bytes, system_memory_bytes)

                audio_models.append({
                    "model_id": model_name,
                    "repo_id": model_name,
                    "ram_needed_gb": ram_gb,
                    "snapshot_path": None,
                    "weight_count": None,
                })

        return audio_models

    except Exception:
        return []


# =============================================================================
# FALLBACK TEST MODELS - Minimum Required Models for Testing Without HF_HOME
# =============================================================================
# When HF_HOME is not set, Portfolio Discovery returns []. These fallback models
# provide a baseline for testing when the user has these specific models in
# their default cache (~/.cache/huggingface).
#
# These models must be downloaded manually if testing without HF_HOME:
#   mlxk pull mlx-community/gpt-oss-20b-MXFP4-Q8
#   mlxk pull mlx-community/Qwen2.5-0.5B-Instruct-4bit
#   mlxk pull mlx-community/Llama-3.2-3B-Instruct-4bit
#   mlxk pull mlx-community/pixtral-12b-4bit
#   mlxk pull mlx-community/whisper-large-v3-turbo-4bit
# =============================================================================

# Vision fallback model (for tests without HF_HOME)
VISION_TEST_MODELS = {
    "pixtral": {
        "id": "mlx-community/pixtral-12b-4bit",
        "expected_issue": None,
        "description": "Pixtral 12B - general-purpose vision model",
        "ram_needed_gb": 7.0  # 12B 4-bit (~7GB empirical)
    }
}

# Audio fallback model (for tests without HF_HOME)
AUDIO_TEST_MODELS = {
    "whisper": {
        "id": "mlx-community/whisper-large-v3-turbo-4bit",
        "expected_issue": None,
        "description": "Whisper large-v3-turbo - STT baseline",
        "ram_needed_gb": 1.5  # Large-v3 4-bit (~1.5GB)
    }
}


# Re-export for convenience
__all__ = [
    "discover_mlx_models_in_user_cache",
    "discover_text_models",
    "discover_vision_models",
    "discover_audio_models",
    "calculate_text_model_ram_gb",
    "calculate_vision_model_ram_gb",
    "get_system_memory_bytes",
    "get_safe_ram_budget_gb",
    "get_system_ram_gb",
    "should_skip_model",
    "TEST_MODELS",
    "VISION_TEST_MODELS",
    "AUDIO_TEST_MODELS",
    "TEST_PROMPT",
    "MAX_TOKENS",
    "TEST_TEMPERATURE",
]


# Standard test constants (shared across all E2E tests)
TEST_PROMPT = "Write one sentence about cats."
MAX_TOKENS = 50
TEST_TEMPERATURE = 0.0  # Deterministic sampling for reproducible tests
