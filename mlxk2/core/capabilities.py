"""Unified probe/policy architecture for model capabilities (ADR-012, ADR-016).

This module provides a single entry point for capability detection and backend
selection across all code paths (CLI, Server, List/Show).

Architecture:
    Resolve -> Probe (Capabilities, Runtime, Memory) -> Policy (Backend Selection) -> Load -> Run

Core principles (see docs/ARCHITECTURE.md):
    - No silent fallbacks
    - Fail fast, fail clearly
    - Memory gates before load
    - Explicit error codes

All errors are explicit and visible.
"""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


class Backend(Enum):
    """Available model backends."""
    MLX_LM = "mlx_lm"      # Text models via mlx-lm
    MLX_VLM = "mlx_vlm"    # Vision models via mlx-vlm
    UNSUPPORTED = "unsupported"  # Model cannot be loaded


class PolicyDecision(Enum):
    """Policy decision outcomes."""
    ALLOW = "allow"         # Proceed with backend
    WARN = "warn"           # Proceed with warning
    BLOCK = "block"         # Do not proceed


# Memory threshold for pre-load checks (ADR-016)
# Vision models crash above ~70% due to Vision Encoder overhead
MEMORY_THRESHOLD_PERCENT = 0.70

# Vision model types (from config.json model_type field)
VISION_MODEL_TYPES = frozenset({
    "llava",
    "llava_next",
    "pixtral",
    "qwen2_vl",
    "phi3_v",
    "mllama",
    "paligemma",
    "idefics",
    "smolvlm",
})


@dataclass
class ModelCapabilities:
    """Probed model capabilities and runtime information.

    All detection happens during probe phase. No detection during policy/load.
    """
    # Model identity
    model_path: Path
    model_name: str

    # Core capabilities
    is_vision: bool = False
    is_chat: bool = False
    is_embedding: bool = False

    # File integrity
    config_valid: bool = False
    config: Optional[Dict[str, Any]] = None
    model_type: Optional[str] = None

    # Runtime compatibility
    python_version: Tuple[int, int, int] = field(default_factory=lambda: sys.version_info[:3])
    mlx_vlm_available: bool = False
    mlx_lm_available: bool = False

    # Framework and runtime compatibility (for text models)
    framework: str = "Unknown"
    runtime_compatible: bool = True
    runtime_reason: Optional[str] = None  # Reason if not compatible

    # Memory information (ADR-016)
    system_memory_bytes: Optional[int] = None
    model_size_bytes: int = 0
    memory_ratio: float = 0.0  # model_size / system_memory

    # Additional metadata
    capabilities_list: List[str] = field(default_factory=list)
    reason: Optional[str] = None  # Error/warning reason if any


@dataclass
class BackendPolicy:
    """Backend selection policy result.

    Returned by select_backend_policy() to indicate what action to take.
    """
    backend: Backend
    decision: PolicyDecision

    # For WARN/BLOCK decisions
    message: Optional[str] = None

    # HTTP status code for server context
    http_status: Optional[int] = None

    # Specific error type for structured errors
    error_type: Optional[str] = None


def _get_system_memory_bytes() -> Optional[int]:
    """Get total system memory in bytes via sysctl (macOS only).

    Returns:
        Total memory in bytes, or None if unavailable.
    """
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
    return None


def _get_model_size_bytes(model_path: Path) -> int:
    """Calculate total model size in bytes."""
    try:
        total = 0
        for f in model_path.rglob("*"):
            if f.is_file():
                total += f.stat().st_size
        return total
    except Exception:
        return 0


def _format_bytes_gb(size_bytes: int) -> str:
    """Format bytes as human-readable GB string."""
    return f"{size_bytes / (1024**3):.1f} GB"


def _has_any(path: Path, patterns: Tuple[str, ...]) -> bool:
    """Check if any file matching patterns exists under path."""
    try:
        for pat in patterns:
            if any(path.glob(pat)):
                return True
    except Exception:
        return False
    return False


def _detect_vision_from_config(config: Optional[Dict[str, Any]]) -> bool:
    """Detect vision capability from config.json content.

    Video models (AutoVideoProcessor) are excluded as they require PyTorch/Torchvision.
    mlx-vlm only supports image vision models (AutoImageProcessor).
    """
    if not isinstance(config, dict):
        return False

    # Check model_type
    mt = config.get("model_type")
    if isinstance(mt, str) and mt.lower() in VISION_MODEL_TYPES:
        return True

    # Check for image_processor
    if config.get("image_processor"):
        return True

    # Check for embedded preprocessor_config
    preprocessor_cfg = config.get("preprocessor_config")
    if isinstance(preprocessor_cfg, dict):
        # Exclude video processors (requires PyTorch/Torchvision)
        if preprocessor_cfg.get("processor_class") == "AutoVideoProcessor":
            return False
        if "temporal_patch_size" in preprocessor_cfg:
            return False
        return True

    return False


def _detect_vision_from_files(model_path: Path) -> bool:
    """Detect vision capability from file presence.

    Video models (AutoVideoProcessor) are excluded as they require PyTorch/Torchvision.
    mlx-vlm only supports image vision models (AutoImageProcessor).
    """
    # Check if it's a video model (requires PyTorch/Torchvision)
    if (model_path / "video_preprocessor_config.json").exists():
        return False

    if _has_any(
        model_path,
        (
            "preprocessor_config.json",
            "processor_config.json",
            "image_processor_config.json",
            "**/preprocessor_config.json",
            "**/processor_config.json",
            "**/image_processor_config.json",
        ),
    ):
        # Found vision-related files, but check if it's a video processor
        preprocessor_path = model_path / "preprocessor_config.json"
        if preprocessor_path.exists():
            try:
                import json
                with open(preprocessor_path) as f:
                    preprocessor_data = json.load(f)
                if isinstance(preprocessor_data, dict):
                    # Video model indicators
                    if preprocessor_data.get("processor_class") == "AutoVideoProcessor":
                        return False
                    if "temporal_patch_size" in preprocessor_data:
                        return False
            except Exception:
                pass
        return True

    return False


def _check_mlx_vlm_available() -> bool:
    """Check if mlx-vlm package is available."""
    return importlib.util.find_spec("mlx_vlm") is not None


def _check_mlx_lm_available() -> bool:
    """Check if mlx-lm package is available."""
    return importlib.util.find_spec("mlx_lm") is not None


def _check_text_runtime_compatibility(model_path: Path, model_name: str, config: Optional[Dict[str, Any]]) -> Tuple[bool, Optional[str]]:
    """Check if text model is compatible with mlx-lm runtime.

    Uses existing detection functions from operations/common.py and operations/health.py.

    Returns:
        (is_compatible, reason): reason is None if compatible, error message otherwise
    """
    try:
        # Import existing functions to avoid code duplication
        from ..operations.common import detect_framework, read_front_matter
        from ..operations.health import check_runtime_compatibility

        # Detect framework using existing logic
        fm = read_front_matter(model_path)
        framework = detect_framework(model_name, model_path.parent.parent, selected_path=model_path, fm=fm)

        # Check runtime compatibility
        return check_runtime_compatibility(model_path, framework)
    except Exception:
        # If detection fails, assume compatible (let MLXRunner handle it)
        return True, None


def probe_model_capabilities(
    model_path: Path,
    model_name: str,
    config: Optional[Dict[str, Any]] = None,
) -> ModelCapabilities:
    """Probe all model capabilities and runtime information.

    This is the single entry point for capability detection. All detection
    happens here - no detection during policy selection or model loading.

    Args:
        model_path: Path to model snapshot directory
        model_name: Model name/spec for error messages
        config: Pre-loaded config.json (optional, will be loaded if None)

    Returns:
        ModelCapabilities with all probed information
    """
    caps = ModelCapabilities(
        model_path=model_path,
        model_name=model_name,
    )

    # Load config if not provided
    if config is None:
        config_path = model_path / "config.json"
        if config_path.exists():
            try:
                caps.config = json.loads(config_path.read_text(encoding="utf-8", errors="ignore"))
                caps.config_valid = isinstance(caps.config, dict) and len(caps.config) > 0
            except (OSError, json.JSONDecodeError):
                caps.config = None
                caps.config_valid = False
        else:
            caps.config_valid = False
    else:
        caps.config = config
        caps.config_valid = isinstance(config, dict) and len(config) > 0

    # Extract model_type
    if caps.config_valid and caps.config:
        caps.model_type = caps.config.get("model_type")

    # Detect vision capability (from config AND files)
    caps.is_vision = (
        _detect_vision_from_config(caps.config) or
        _detect_vision_from_files(model_path)
    )

    # Detect chat capability
    if caps.config_valid and caps.config:
        mt = caps.model_type
        if isinstance(mt, str):
            if mt.lower() == "chat":
                caps.is_chat = True
            elif mt.lower() in VISION_MODEL_TYPES:
                caps.is_chat = True  # Vision models are chat models

    # Check for chat template in tokenizer
    for fname in ("tokenizer_config.json", "tokenizer.json"):
        fp = model_path / fname
        if fp.exists():
            try:
                obj = json.loads(fp.read_text(encoding="utf-8", errors="ignore"))
                if isinstance(obj, dict):
                    ct = obj.get("chat_template")
                    if isinstance(ct, str) and ct.strip():
                        caps.is_chat = True
                        break
            except Exception:
                pass

    # Check name hints for chat
    name_lower = model_name.lower()
    if "instruct" in name_lower or "chat" in name_lower:
        caps.is_chat = True

    # Detect embedding capability
    if "embed" in name_lower:
        caps.is_embedding = True

    # Build capabilities list (for JSON API compatibility)
    if caps.is_embedding:
        caps.capabilities_list = ["embeddings"]
    else:
        caps.capabilities_list = ["text-generation"]
        if caps.is_chat:
            caps.capabilities_list.append("chat")
        if caps.is_vision:
            caps.capabilities_list.append("vision")

    # Check runtime availability
    caps.python_version = sys.version_info[:3]
    caps.mlx_vlm_available = _check_mlx_vlm_available()
    caps.mlx_lm_available = _check_mlx_lm_available()

    # Check text model runtime compatibility (framework + model_type)
    # Vision models use mlx-vlm which has its own checks
    if not caps.is_vision:
        runtime_ok, runtime_reason = _check_text_runtime_compatibility(
            model_path, model_name, caps.config
        )
        caps.runtime_compatible = runtime_ok
        caps.runtime_reason = runtime_reason

        # Also store framework from the check
        try:
            from ..operations.common import detect_framework, read_front_matter
            fm = read_front_matter(model_path)
            caps.framework = detect_framework(model_name, model_path.parent.parent, selected_path=model_path, fm=fm)
        except Exception:
            caps.framework = "Unknown"

    # Memory information (ADR-016)
    caps.system_memory_bytes = _get_system_memory_bytes()
    caps.model_size_bytes = _get_model_size_bytes(model_path)

    if caps.system_memory_bytes and caps.model_size_bytes:
        caps.memory_ratio = caps.model_size_bytes / caps.system_memory_bytes

    return caps


def select_backend_policy(
    caps: ModelCapabilities,
    context: str = "cli",
    has_images: bool = False,
) -> BackendPolicy:
    """Select backend and determine policy based on probed capabilities.

    This function never detects capabilities - it only makes decisions based
    on the already-probed ModelCapabilities.

    Args:
        caps: Probed model capabilities
        context: Execution context ("cli" or "server")
        has_images: Whether images are being passed to the model

    Returns:
        BackendPolicy indicating backend choice and any warnings/blocks
    """
    # Gate 1: Vision model detection and backend selection
    if caps.is_vision or has_images:
        # Vision path requires mlx-vlm backend

        # Gate 1a: Images provided but model not vision-capable
        # Return early so non-vision models get a capability mismatch instead
        # of unrelated version/dependency errors (e.g., on Python 3.9).
        if has_images and not caps.is_vision:
            return BackendPolicy(
                backend=Backend.UNSUPPORTED,
                decision=PolicyDecision.BLOCK,
                message=f"Model '{caps.model_name}' does not support vision inputs (no vision capability detected)",
                http_status=400,
                error_type="capability_mismatch",
            )

        # Check Python version (mlx-vlm requires 3.10+)
        if caps.python_version < (3, 10):
            return BackendPolicy(
                backend=Backend.UNSUPPORTED,
                decision=PolicyDecision.BLOCK,
                message=f"Vision models require Python 3.10+ (current: {'.'.join(map(str, caps.python_version))})",
                http_status=501,
                error_type="python_version_error",
            )

        # Check mlx-vlm availability
        if not caps.mlx_vlm_available:
            return BackendPolicy(
                backend=Backend.UNSUPPORTED,
                decision=PolicyDecision.BLOCK,
                message="Vision models require mlx-vlm (install extras: vision)",
                http_status=501,
                error_type="missing_dependency",
            )

        # Note: Server vision support enabled in 2.0.4-beta.1 (ADR-012 Phase 3)

        # Gate 3: Memory check for vision models (ADR-016)
        if caps.memory_ratio > MEMORY_THRESHOLD_PERCENT:
            msg = (
                f"Model size ({_format_bytes_gb(caps.model_size_bytes)}) exceeds 70% of system memory "
                f"({_format_bytes_gb(caps.system_memory_bytes or 0)}). Vision models crash with Metal OOM "
                f"due to Vision Encoder overhead."
            )
            return BackendPolicy(
                backend=Backend.UNSUPPORTED,
                decision=PolicyDecision.BLOCK,
                message=msg,
                http_status=507,
                error_type="insufficient_memory",
            )

        # Vision model allowed
        return BackendPolicy(
            backend=Backend.MLX_VLM,
            decision=PolicyDecision.ALLOW,
        )

    # Text model path (mlx-lm backend)

    # Check mlx-lm availability
    if not caps.mlx_lm_available:
        return BackendPolicy(
            backend=Backend.UNSUPPORTED,
            decision=PolicyDecision.BLOCK,
            message="Text models require mlx-lm (pip install mlx-lm)",
            http_status=501,
            error_type="missing_dependency",
        )

    # Gate: Runtime compatibility check (framework + model_type support)
    if not caps.runtime_compatible:
        return BackendPolicy(
            backend=Backend.UNSUPPORTED,
            decision=PolicyDecision.BLOCK,
            message=f"Model '{caps.model_name}' is not compatible: {caps.runtime_reason}",
            http_status=501,
            error_type="runtime_incompatible",
        )

    # Gate 4: Memory check for text models (ADR-016)
    # Text models: warning only (they swap gracefully, no crash)
    if caps.memory_ratio > MEMORY_THRESHOLD_PERCENT:
        msg = (
            f"Model size ({_format_bytes_gb(caps.model_size_bytes)}) exceeds 70% of "
            f"{_format_bytes_gb(caps.system_memory_bytes or 0)} system memory. "
            f"Expect extreme slowness due to swapping."
        )
        return BackendPolicy(
            backend=Backend.MLX_LM,
            decision=PolicyDecision.WARN,
            message=msg,
        )

    # Text model allowed
    return BackendPolicy(
        backend=Backend.MLX_LM,
        decision=PolicyDecision.ALLOW,
    )


def probe_and_select(
    model_path: Path,
    model_name: str,
    config: Optional[Dict[str, Any]] = None,
    context: str = "cli",
    has_images: bool = False,
) -> Tuple[ModelCapabilities, BackendPolicy]:
    """Convenience function to probe capabilities and select policy in one call.

    Args:
        model_path: Path to model snapshot directory
        model_name: Model name/spec for error messages
        config: Pre-loaded config.json (optional)
        context: Execution context ("cli" or "server")
        has_images: Whether images are being passed to the model

    Returns:
        Tuple of (ModelCapabilities, BackendPolicy)
    """
    caps = probe_model_capabilities(model_path, model_name, config)
    policy = select_backend_policy(caps, context, has_images)
    return caps, policy
