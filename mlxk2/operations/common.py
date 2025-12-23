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
from ..core.capabilities import VISION_MODEL_TYPES


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
    """
    try:
        if isinstance(config, dict):
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


def detect_capabilities(
    model_type: str,
    hf_name: str,
    tok_hints: Dict[str, Any],
    config: Optional[Dict[str, Any]],
    probe: Path,
) -> list[str]:
    if model_type == "embedding":
        return ["embeddings"]
    caps = ["text-generation"]
    name = hf_name.lower()
    ct = tok_hints.get("chat_template")
    if model_type == "chat" or "instruct" in name or "chat" in name or (isinstance(ct, str) and ct.strip()):
        caps.append("chat")
    if detect_vision_capability(probe, config):
        caps.append("vision")
    return caps


def vision_runtime_compatibility() -> tuple[bool, Optional[str]]:
    """Vision uses mlx-vlm backend; mark compatible only if available."""
    if sys.version_info < (3, 10):
        return False, "Vision requires Python 3.10+ (mlx-vlm dependency)"
    spec = importlib.util.find_spec("mlx_vlm")
    if spec is None:
        return False, "mlx-vlm not installed (install extras: vision)"
    return True, None


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
    """
    from ..operations.health import is_model_healthy, check_runtime_compatibility  # local import to avoid cycle

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

    # Health: rely on existing operation (name-based)
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
    elif has_vision:
        runtime_compatible, runtime_reason = vision_runtime_compatibility()
    else:
        runtime_compatible, runtime_reason = check_runtime_compatibility(probe, framework)

    # Reason field: First problem encountered (health → runtime)
    reason = health_reason if not healthy else runtime_reason

    # Size/Modified computed from selected path (snapshot preferred)
    base = selected_path if selected_path is not None else model_root
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
        "cached": True,
    }
    return model_obj
