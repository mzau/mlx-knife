"""Tests for lenient MLX detection (Issue #31 port) in 2.0.

Covers:
- Framework=MLX via README front-matter (tags/library_name) for non-mlx-community repos.
- Type=chat via tokenizer chat_template hints.
- Consistency between list and show outputs.
"""

from __future__ import annotations

import sys

from pathlib import Path
from typing import Tuple

from mlxk2.core.cache import hf_to_cache_dir
from mlxk2.operations.list import list_models
from mlxk2.operations.show import show_model_operation


def _mk_snapshot(cache_hub: Path, repo_id: str, hash40: str, config_text: str | None = None) -> Tuple[Path, Path]:
    base = cache_hub / hf_to_cache_dir(repo_id)
    snap = base / "snapshots" / hash40
    snap.mkdir(parents=True, exist_ok=True)
    # Minimal healthy files
    cfg = config_text or '{"model_type": "test"}'
    (snap / "config.json").write_text(cfg, encoding="utf-8")
    (snap / "model.safetensors").write_bytes(b"w" * 1024)
    return base, snap


def test_framework_mlx_from_front_matter(isolated_cache):
    repo = "custom-org/FrontMatter-Model"
    h = "0123456789abcdef0123456789abcdef01234567"
    base, snap = _mk_snapshot(isolated_cache, repo, h)

    # README front-matter indicating MLX
    (snap / "README.md").write_text(
        """---
library_name: mlx
tags: [mlx, chat]
---

# Dummy
""",
        encoding="utf-8",
    )

    out = list_models()
    models = {m["name"]: m for m in out["data"]["models"]}
    assert repo in models, f"Model not listed: {repo}"
    assert models[repo]["framework"] == "MLX"

    s = show_model_operation(repo)
    assert s["status"] == "success"
    assert s["data"]["model"]["framework"] == "MLX"


def test_type_chat_from_tokenizer_chat_template(isolated_cache):
    repo = "custom-org/Tokenizer-Chat-Model"
    h = "89abcdef0123456789abcdef0123456789abcdef"
    base, snap = _mk_snapshot(isolated_cache, repo, h)

    # No chat/instruct in name → rely on tokenizer chat_template
    (snap / "tokenizer_config.json").write_text(
        '{"chat_template": "{{ bos_token }}{{ eos_token }}"}', encoding="utf-8"
    )

    # Also put a front-matter not mentioning mlx to ensure chat comes from tokenizer
    (snap / "README.md").write_text(
        """---
tags: [test]
---
""",
        encoding="utf-8",
    )

    out = list_models()
    models = {m["name"]: m for m in out["data"]["models"]}
    assert repo in models, f"Model not listed: {repo}"
    m = models[repo]
    assert m["model_type"] == "chat"
    assert "chat" in (m.get("capabilities") or [])

    s = show_model_operation(repo)
    assert s["status"] == "success"
    ms = s["data"]["model"]
    assert ms["model_type"] == "chat"
    assert "chat" in (ms.get("capabilities") or [])


def test_vision_capability_from_model_type(isolated_cache):
    repo = "mlx-community/llava-vision-alpha"
    h = "1111111111111111111111111111111111111111"
    _, snap = _mk_snapshot(
        isolated_cache,
        repo,
        h,
        config_text='{"model_type": "llava", "image_processor": {"size": 224}}',
    )
    # ADR-012 Phase 2: Vision models require preprocessor_config.json for health
    (snap / "preprocessor_config.json").write_text('{"size": 224}', encoding="utf-8")
    (snap / "tokenizer_config.json").write_text('{"chat_template": "{{ bos_token }}"}', encoding="utf-8")
    (snap / "tokenizer.json").write_text('{}', encoding="utf-8")

    # Vision models require Python 3.10+ (mlx-vlm dependency)
    expected_runtime_compatible = sys.version_info >= (3, 10)

    out = list_models()
    models = {m["name"]: m for m in out["data"]["models"]}
    assert repo in models
    m = models[repo]
    assert m["model_type"] == "chat"
    assert "vision" in (m.get("capabilities") or [])
    assert "chat" in (m.get("capabilities") or [])
    assert m["runtime_compatible"] is expected_runtime_compatible

    s = show_model_operation(repo)
    assert s["status"] == "success"
    ms = s["data"]["model"]
    assert "vision" in (ms.get("capabilities") or [])
    assert ms["runtime_compatible"] is expected_runtime_compatible


def test_vision_capability_from_preprocessor_file(isolated_cache):
    repo = "mlx-community/pixtral-vision-12b"
    h = "2222222222222222222222222222222222222222"
    # Use pixtral model_type (mlx-lm supported) - vision detected from preprocessor_config.json
    _, snap = _mk_snapshot(isolated_cache, repo, h, config_text='{"model_type": "pixtral"}')
    # ADR-012 Phase 2: Vision models require preprocessor_config.json
    (snap / "preprocessor_config.json").write_text("{}", encoding="utf-8")
    (snap / "tokenizer_config.json").write_text('{"chat_template": "{{ bos_token }}"}', encoding="utf-8")
    (snap / "tokenizer.json").write_text('{}', encoding="utf-8")

    # Vision models require Python 3.10+ (mlx-vlm dependency)
    expected_runtime_compatible = sys.version_info >= (3, 10)

    out = list_models()
    models = {m["name"]: m for m in out["data"]["models"]}
    assert repo in models
    m = models[repo]
    assert m["model_type"] == "chat"
    assert "vision" in (m.get("capabilities") or [])
    assert "chat" in (m.get("capabilities") or [])
    assert m["runtime_compatible"] is expected_runtime_compatible

    s = show_model_operation(repo)
    assert s["status"] == "success"
    ms = s["data"]["model"]
    assert "vision" in (ms.get("capabilities") or [])
    assert ms["runtime_compatible"] is expected_runtime_compatible
