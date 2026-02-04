"""Common helpers for model metadata detection (2.0).

Lenient framework/type detection for Issue #31 port:
- Prefer MLX for mlx-community/* or when README front-matter indicates MLX.
- Detect chat type via name, config, or tokenizer chat_template hints.

Parsing is intentionally lightweight (no YAML dependency). Front-matter is
parsed from the first '---' block in README.md when present.
"""

from __future__ import annotations

import json as _json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional
import importlib.util
import sys

# Import from unified capabilities module (ARCHITECTURE.md)
from ..core.capabilities import VISION_MODEL_TYPES, AUDIO_MODEL_TYPES, STT_MODEL_TYPES, Capability, Backend


@dataclass
class FrontMatter:
    tags: list[str]
    library_name: Optional[str]


def read_front_matter(root: Path) -> Optional[FrontMatter]:
    """Best-effort parse of README.md YAML-like front matter.

    Supports:
    - Inline list: tags: [mlx, chat]
    - Block list:
        tags:
          - mlx
          - chat
    - library_name: mlx
    Returns None if README.md or front-matter block missing.
    """
    try:
        readme = root / "README.md"
        if not readme.exists() or not readme.is_file():
            return None
        lines = readme.read_text(encoding="utf-8", errors="ignore").splitlines()
        if not lines or lines[0].strip() != "---":
            return None
        # Extract the first front-matter block
        block: list[str] = []
        for line in lines[1:]:
            if line.strip() == "---":
                break
            block.append(line.rstrip("\n"))
        if not block:
            return None

        tags: list[str] = []
        library_name: Optional[str] = None

        # Simple state machine for tags block list
        in_tags_block = False
        for raw in block:
            s = raw.strip()
            if not s:
                continue
            # library_name: value
            if s.lower().startswith("library_name:"):
                try:
                    library_name = s.split(":", 1)[1].strip().strip('"\'')
                except Exception:
                    pass
                in_tags_block = False
                continue

            # tags: [a, b]
            if s.lower().startswith("tags:") and "[" in s and "]" in s:
                try:
                    inside = s.split("[", 1)[1].rsplit("]", 1)[0]
                    parts = [p.strip().strip('"\'') for p in inside.split(",") if p.strip()]
                    tags.extend([p for p in parts if p])
                except Exception:
                    pass
                in_tags_block = False
                continue

            # tags: (start of block list)
            if s.lower().startswith("tags:"):
                in_tags_block = True
                continue

            if in_tags_block:
                # Expect lines like "- mlx"
                try:
                    if s.startswith("-"):
                        val = s.lstrip("-").strip().strip('"\'')
                        if val:
                            tags.append(val)
                    else:
                        # Any other non-dash line ends the block
                        in_tags_block = False
                except Exception:
                    pass

        return FrontMatter(tags=tags, library_name=library_name)
    except Exception:
        return None


def read_tokenizer_hints(root: Path) -> Dict[str, Any]:
    """Extract lightweight tokenizer hints (e.g., chat_template presence)."""
    hints: Dict[str, Any] = {"chat_template": None}
    try:
        for fname in ("tokenizer_config.json", "tokenizer.json"):
            fp = root / fname
            if fp.exists() and fp.is_file():
                try:
                    obj = _json.loads(fp.read_text(encoding="utf-8", errors="ignore"))
                except Exception:
                    obj = None
                if isinstance(obj, dict):
                    ct = obj.get("chat_template")
                    if isinstance(ct, str) and ct.strip():
                        hints["chat_template"] = ct
                        break
    except Exception:
        pass
    return hints


def _has_any(path: Path, patterns: tuple[str, ...]) -> bool:
    try:
        for pat in patterns:
            if any(path.glob(pat)):
                return True
    except Exception:
        return False
    return False


def detect_framework(hf_name: str, model_root: Path, selected_path: Optional[Path] = None, fm: Optional[FrontMatter] = None) -> str:
    """Lenient framework detection.

    MLX if:
    - org is mlx-community/*, or
    - README front-matter tags include 'mlx', or
    - README front-matter library_name == 'mlx', or
    - config.json contains 'quantization' key (MLX-specific).

    Else GGUF if any *.gguf present under selected_path or snapshots.
    Else PyTorch if any *.safetensors or pytorch_model.bin present under snapshots.
    Else Unknown.
    """
    try:
        if "mlx-community/" in hf_name:
            return "MLX"

        # Search location preference: selected snapshot, else model root
        root = selected_path if selected_path is not None else model_root

        # Read front-matter if not provided (Issue #48: self-contained detection)
        if fm is None:
            fm = read_front_matter(root)

        # Front-matter signals
        if fm is not None:
            tags = [t.lower() for t in (fm.tags or [])]
            lib = (fm.library_name or "").lower()
            if "mlx" in tags or lib == "mlx":
                return "MLX"

        # Config-based detection: 'quantization' key is MLX-specific (Issue #48)
        config = _load_config_json(root)
        if config and "quantization" in config:
            return "MLX"

        if _has_any(root, ("**/*.gguf",)):
            return "GGUF"

        # Look under snapshots for common formats
        snapshots_dir = model_root / "snapshots"
        if _has_any(snapshots_dir, ("**/*.safetensors", "**/pytorch_model.bin")):
            return "PyTorch"
    except Exception:
        pass
    return "Unknown"


def detect_model_type(hf_name: str, config: Optional[Dict[str, Any]], tok_hints: Dict[str, Any], probe: Optional[Path] = None) -> str:
    name = hf_name.lower()
    if "embed" in name:
        return "embedding"
    model_type = (config or {}).get("model_type")
    if isinstance(model_type, str):
        mt_lower = model_type.lower()
        if mt_lower == "chat":
            return "chat"
        if mt_lower in VISION_MODEL_TYPES:
            return "chat"
        # STT/Audio-only models (Whisper, Voxtral) - NOT chat models
        # These models only transcribe audio, they don't generate text or chat
        if mt_lower in STT_MODEL_TYPES:
            return "audio"
    ct = tok_hints.get("chat_template")
    if isinstance(ct, str) and ct.strip():
        return "chat"
    # Check for chat_template.json file (Issue #48: reliable indicator)
    if probe is not None and (probe / "chat_template.json").exists():
        return "chat"
    if "instruct" in name or "chat" in name:
        return "chat"
    return "base"


def detect_vision_capability(probe: Path, config: Optional[Dict[str, Any]]) -> bool:
    """Detect whether the model snapshot supports vision inputs.

    Video models (AutoVideoProcessor) are excluded as they require PyTorch/Torchvision.
    mlx-vlm only supports image vision models (AutoImageProcessor).

    Note: skip_vision flag indicates vision components can be skipped for text-only
    inference, but does NOT mean the model lacks vision capabilities.
    """
    try:
        if isinstance(config, dict):
            # Check for vision_config presence (Mistral-Small 3.1 has vision_config with skip_vision)
            vision_config = config.get("vision_config")
            if isinstance(vision_config, dict):
                # Vision config present = vision model (even if skip_vision=true)
                return True

            mt = config.get("model_type")
            if isinstance(mt, str) and mt.lower() in VISION_MODEL_TYPES:
                return True

            if config.get("image_processor"):
                return True

            preprocessor_cfg = config.get("preprocessor_config")
            if isinstance(preprocessor_cfg, dict):
                # Exclude video processors (requires PyTorch/Torchvision)
                if preprocessor_cfg.get("processor_class") == "AutoVideoProcessor":
                    return False
                return True

        if _has_any(
            probe,
            (
                "preprocessor_config.json",
                "processor_config.json",
                "image_processor_config.json",
                "**/preprocessor_config.json",
                "**/processor_config.json",
                "**/image_processor_config.json",
            ),
        ):
            # Check if it's a video processor (requires PyTorch/Torchvision)
            # Video models have video_preprocessor_config.json or temporal_patch_size
            if (probe / "video_preprocessor_config.json").exists():
                return False

            preprocessor_path = probe / "preprocessor_config.json"
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
    except Exception:
        return False
    return False


def detect_audio_capability(probe: Path, config: Optional[Dict[str, Any]]) -> bool:
    """Detect whether the model snapshot supports audio inputs (ADR-019, ADR-020).

    Detection signals:
    - config.json contains "audio_config" key (Gemma-3n, Voxtral)
    - config.json model_type in AUDIO_MODEL_TYPES (Whisper, Voxtral, Gemma-3n)
    - preprocessor_config.json contains WhisperFeatureExtractor (Whisper variants)
    - processor_config.json contains "audio_seq_length" key (secondary)
    """
    try:
        if isinstance(config, dict):
            # Check for audio_config (Gemma-3n, Voxtral)
            if "audio_config" in config:
                return True

            # Check model_type (Whisper, Voxtral, Gemma-3n)
            mt = config.get("model_type")
            if isinstance(mt, str) and mt.lower() in AUDIO_MODEL_TYPES:
                return True

        # Check preprocessor_config.json for WhisperFeatureExtractor (Whisper variants)
        preprocessor_config_path = probe / "preprocessor_config.json"
        if preprocessor_config_path.exists():
            try:
                proc_data = _json.loads(preprocessor_config_path.read_text(encoding="utf-8", errors="ignore"))
                if isinstance(proc_data, dict):
                    feature_extractor = proc_data.get("feature_extractor_type", "")
                    if isinstance(feature_extractor, str) and "whisper" in feature_extractor.lower():
                        return True
            except Exception:
                pass

        # Check processor_config.json for audio_seq_length (secondary)
        processor_config_path = probe / "processor_config.json"
        if processor_config_path.exists():
            try:
                with open(processor_config_path) as f:
                    processor_data = _json.load(f)
                if isinstance(processor_data, dict) and "audio_seq_length" in processor_data:
                    return True
            except Exception:
                pass
    except Exception:
        pass
    return False


def detect_audio_backend(probe: Path, config: Optional[Dict[str, Any]]) -> Optional[Backend]:
    """Model-agnostic audio backend detection (MLX_AUDIO vs MLX_VLM).

    ADR-020: Config-based detection routes audio models to appropriate backend:
    - STT models (Voxtral, Whisper, VibeVoice) → Backend.MLX_AUDIO
    - Multimodal models (Gemma-3n, Qwen3-Omni) → Backend.MLX_VLM

    Detection priority:
    1. model_type == "voxtral" → MLX_AUDIO (always STT, even with audio_config)
    2. audio_config + populated vision_config → MLX_VLM (multimodal)
    3. model_type contains "whisper" → MLX_AUDIO (Whisper variants)
    4. preprocessor has WhisperFeatureExtractor → MLX_AUDIO (Whisper-based)
    5. Name heuristics (whisper/voxtral/vibevoice) → MLX_AUDIO (fallback)
    6. audio_config alone → MLX_VLM (legacy/unknown multimodal)

    Args:
        probe: Path to model snapshot directory
        config: Pre-loaded config.json (optional)

    Returns:
        Backend.MLX_AUDIO for STT, Backend.MLX_VLM for multimodal, None if not audio model
    """
    if not config:
        return None

    model_type = config.get("model_type", "")
    if isinstance(model_type, str):
        model_type_lower = model_type.lower()
    else:
        model_type_lower = ""

    # Priority 1: Voxtral = Always mlx-audio STT (even with audio_config)
    # Works for both Original Mistral and converted models
    if model_type_lower == "voxtral":
        return Backend.MLX_AUDIO

    # Priority 2: audio_config + populated vision_config = mlx-vlm multimodal
    # Gemma-3n, Qwen3-Omni (Vision + Audio → Text)
    if "audio_config" in config:
        vision_config = config.get("vision_config")
        # Populated = dict with content (not None, not empty dict)
        if isinstance(vision_config, dict) and len(vision_config) > 0:
            return Backend.MLX_VLM

    # Priority 3: Whisper model_type = mlx-audio STT
    if "whisper" in model_type_lower:
        return Backend.MLX_AUDIO

    # Priority 4: WhisperFeatureExtractor in preprocessor = mlx-audio STT
    preprocessor_path = probe / "preprocessor_config.json"
    if preprocessor_path.exists():
        try:
            proc_data = _json.loads(preprocessor_path.read_text(encoding="utf-8", errors="ignore"))
            if isinstance(proc_data, dict):
                feature_extractor = proc_data.get("feature_extractor_type", "")
                if isinstance(feature_extractor, str) and "whisper" in feature_extractor.lower():
                    return Backend.MLX_AUDIO
        except Exception:
            pass

    # Priority 5: Name heuristics = mlx-audio STT (fallback)
    name = probe.name.lower()
    stt_keywords = ["whisper", "voxtral", "vibevoice"]
    if any(kw in name for kw in stt_keywords):
        return Backend.MLX_AUDIO

    # Priority 6: audio_config alone = mlx-vlm (legacy/unknown multimodal)
    # This is the fallback for models that have audio_config but no clear STT signal
    if "audio_config" in config:
        return Backend.MLX_VLM

    # Not an audio model (no audio_config, no model_type match)
    return None


def detect_capabilities(
    model_type: str,
    hf_name: str,
    tok_hints: Dict[str, Any],
    config: Optional[Dict[str, Any]],
    probe: Path,
) -> list[str]:
    if model_type == "embedding":
        return [Capability.EMBEDDINGS.value]
    # STT/Audio-only models (Whisper, Voxtral) - ONLY audio capability
    # These models transcribe audio, they don't generate text or chat
    if model_type == "audio":
        return [Capability.AUDIO.value]
    caps = [Capability.TEXT_GENERATION.value]
    name = hf_name.lower()
    ct = tok_hints.get("chat_template")
    if model_type == "chat" or "instruct" in name or "chat" in name or (isinstance(ct, str) and ct.strip()):
        caps.append(Capability.CHAT.value)
    if detect_vision_capability(probe, config):
        caps.append(Capability.VISION.value)
    if detect_audio_capability(probe, config):
        caps.append(Capability.AUDIO.value)
    return caps


def vision_runtime_compatibility() -> tuple[bool, Optional[str]]:
    """Vision uses mlx-vlm backend; mark compatible only if available."""
    if sys.version_info < (3, 10):
        return False, "Vision requires Python 3.10+ (mlx-vlm dependency)"
    spec = importlib.util.find_spec("mlx_vlm")
    if spec is None:
        return False, "mlx-vlm not installed (install extras: vision)"
    return True, None


def audio_runtime_compatibility(backend: Backend) -> tuple[bool, Optional[str]]:
    """Audio runtime check based on backend (ADR-020).

    Args:
        backend: Backend.MLX_AUDIO (Whisper/Voxtral) or Backend.MLX_VLM (Gemma-3n)

    Returns:
        (is_compatible, reason): reason is None if compatible
    """
    if sys.version_info < (3, 10):
        return False, "Audio requires Python 3.10+"

    if backend == Backend.MLX_AUDIO:
        # STT models (Whisper, Voxtral) need mlx-audio
        spec = importlib.util.find_spec("mlx_audio")
        if spec is None:
            return False, "mlx-audio not installed (pip install mlx-knife[audio])"
        return True, None
    elif backend == Backend.MLX_VLM:
        # Multimodal audio (Gemma-3n) needs mlx-vlm
        return vision_runtime_compatibility()
    else:
        return False, "Unknown audio backend"


def _iso8601_utc_from_mtime(p: Path) -> str:
    try:
        from datetime import datetime
        return datetime.fromtimestamp(p.stat().st_mtime).strftime("%Y-%m-%dT%H:%M:%SZ")
    except Exception:
        return "1970-01-01T00:00:00Z"


def _total_size_bytes(path: Path) -> int:
    try:
        total = 0
        for f in path.rglob("*"):
            if f.is_file():
                total += f.stat().st_size
        return total
    except Exception:
        return 0


def _load_config_json(path: Path) -> Optional[Dict[str, Any]]:
    try:
        fp = path / "config.json"
        if fp.exists():
            return _json.loads(fp.read_text(encoding="utf-8", errors="ignore"))
    except Exception:
        pass
    return None


def build_model_object(hf_name: str, model_root: Path, selected_path: Optional[Path]) -> Dict[str, Any]:
    """Build the common model object for list/show using unified detection.

    selected_path: points at the chosen snapshot directory when available; otherwise
    may be the model_root. Commit hash is taken from selected_path.name if it looks
    like a 40-char hex string, else None.

    ADR-018 Phase 0c: Supports workspace paths (hf_name can be absolute path).
    """
    from ..operations.health import is_model_healthy, check_runtime_compatibility, health_check_workspace
    from ..operations.workspace import is_workspace_path

    # Compute commit hash if selected path is a snapshot dir
    commit_hash: Optional[str] = None
    if selected_path is not None:
        name = selected_path.name
        if len(name) == 40 and all(c in "0123456789abcdef" for c in name.lower()):
            commit_hash = name

    # Read hints from selected snapshot if possible; fall back to model root
    probe = selected_path if selected_path is not None else model_root
    fm = read_front_matter(probe)
    tok = read_tokenizer_hints(probe)
    config = _load_config_json(probe)

    framework = detect_framework(hf_name, model_root, selected_path=selected_path, fm=fm)
    model_type = detect_model_type(hf_name, config, tok, probe)
    capabilities = detect_capabilities(model_type, hf_name, tok, config, probe)
    has_vision = "vision" in capabilities
    has_audio = "audio" in capabilities

    # Detect audio backend for runtime check (ADR-020)
    audio_backend = detect_audio_backend(probe, config) if has_audio else None

    # Health: workspace-aware (ADR-018 Phase 0c)
    if is_workspace_path(hf_name):
        # Workspace path - use workspace health check directly
        healthy, health_reason, _ = health_check_workspace(Path(hf_name))
    else:
        # Cache model - use name-based health check
        healthy, health_reason = is_model_healthy(hf_name)

    # Runtime compatibility: ALWAYS computed (gate logic applies)
    # Gate 1: File integrity must be healthy
    # Gate 2: Framework must be MLX (only backend supported)
    runtime_reason: Optional[str] = None
    if not healthy:
        # File integrity failed → skip runtime check
        runtime_compatible = False
        runtime_reason = None  # health_reason takes precedence
    elif framework != "MLX":
        # Non-MLX frameworks not supported (PyTorch, GGUF, etc.)
        runtime_compatible = False
        runtime_reason = f"Incompatible framework: {framework}"
    elif has_audio and audio_backend is not None:
        # Audio models: check based on backend (ADR-020)
        runtime_compatible, runtime_reason = audio_runtime_compatibility(audio_backend)
    elif has_vision:
        # Vision models: check BOTH backends for full chat+vision support
        # 1. mlx-vlm must be available (vision mode with images)
        vision_ok, vision_reason = vision_runtime_compatibility()
        # 2. mlx-lm must support model_type (text-only mode without images)
        text_ok, text_reason = check_runtime_compatibility(probe, framework)

        if vision_ok and text_ok:
            runtime_compatible = True
            runtime_reason = None
        else:
            runtime_compatible = False
            # Prefer text_reason as it's more specific (model_type not supported)
            runtime_reason = text_reason or vision_reason
    else:
        runtime_compatible, runtime_reason = check_runtime_compatibility(probe, framework)

    # Reason field: First problem encountered (health → runtime)
    reason = health_reason if not healthy else runtime_reason

    # Size/Modified computed from selected path (snapshot preferred)
    base = selected_path if selected_path is not None else model_root

    # Cached flag: True for cache models, False for workspace paths (ADR-018 Phase 0c)
    cached = not is_workspace_path(hf_name)

    model_obj = {
        "name": hf_name,
        "hash": commit_hash,
        "size_bytes": _total_size_bytes(base),
        "last_modified": _iso8601_utc_from_mtime(base),
        "framework": framework,
        "model_type": model_type,
        "capabilities": capabilities,
        "health": "healthy" if healthy else "unhealthy",
        "runtime_compatible": runtime_compatible,
        "reason": reason,
        "cached": cached,
    }
    return model_obj
