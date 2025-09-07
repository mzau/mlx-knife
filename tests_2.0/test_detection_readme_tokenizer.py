"""Tests for lenient MLX detection (Issue #31 port) in 2.0.

Covers:
- Framework=MLX via README front-matter (tags/library_name) for non-mlx-community repos.
- Type=chat via tokenizer chat_template hints.
- Consistency between list and show outputs.
"""

from __future__ import annotations

from pathlib import Path
from typing import Tuple

from mlxk2.core.cache import hf_to_cache_dir
from mlxk2.operations.list import list_models
from mlxk2.operations.show import show_model_operation


def _mk_snapshot(cache_hub: Path, repo_id: str, hash40: str) -> Tuple[Path, Path]:
    base = cache_hub / hf_to_cache_dir(repo_id)
    snap = base / "snapshots" / hash40
    snap.mkdir(parents=True, exist_ok=True)
    # Minimal healthy files
    (snap / "config.json").write_text('{"model_type": "test"}', encoding="utf-8")
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

