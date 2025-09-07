import json
from pathlib import Path

import pytest

from mlx_knife.cache_utils import detect_framework, detect_model_type, run_model


def _make_base(temp_cache_dir: Path, org: str, name: str) -> Path:
    base = temp_cache_dir / "hub" / f"models--{org}--{name}" / "snapshots" / "main"
    base.mkdir(parents=True, exist_ok=True)
    return base


def test_readme_only_mlx_chat_detection(temp_cache_dir: Path):
    base = _make_base(temp_cache_dir, "private", "my-mlx-chat")
    # Minimal config to look like a model snapshot
    (base / "config.json").write_text(json.dumps({"model_type": "llama"}))
    # README with front matter
    readme = """---
tags: [mlx, chat]
pipeline_tag: text-generation
library_name: mlx
---

# Model Card
"""
    (base / "README.md").write_text(readme)

    framework = detect_framework(base.parent.parent, "private/my-mlx-chat")
    model_type = detect_model_type(base.parent.parent, "private/my-mlx-chat")
    assert framework == "MLX"
    assert model_type == "chat"


def test_tokenizer_only_chat_type(temp_cache_dir: Path):
    base = _make_base(temp_cache_dir, "someone", "no-readme")
    (base / "config.json").write_text(json.dumps({"model_type": "llama"}))
    (base / "tokenizer_config.json").write_text(json.dumps({"chat_template": "{{ bos_token }} {{ messages }}"}))

    framework = detect_framework(base.parent.parent, "someone/no-readme")
    model_type = detect_model_type(base.parent.parent, "someone/no-readme")
    # Framework via fallback; likely Tokenizer/PyTorch/Unknown depending on size and files
    assert model_type == "chat"
    assert framework in {"Tokenizer", "PyTorch", "Unknown", "GGUF", "MLX"}


def test_no_hints_fallbacks(temp_cache_dir: Path):
    base = _make_base(temp_cache_dir, "other", "plain-model")
    (base / "config.json").write_text(json.dumps({"model_type": "bert"}))
    (base / "pytorch_model.bin").write_bytes(b"weights")

    framework = detect_framework(base.parent.parent, "other/plain-model")
    model_type = detect_model_type(base.parent.parent, "other/plain-model")
    assert framework in {"PyTorch", "Tokenizer", "Unknown"}
    assert model_type == "base"


def test_run_model_accepts_mlx_via_readme(monkeypatch, temp_cache_dir: Path):
    base = _make_base(temp_cache_dir, "org", "mlxish")
    (base / "config.json").write_text(json.dumps({"model_type": "llama"}))
    (base / "README.md").write_text("""---
tags: [mlx, chat]
pipeline_tag: text-generation
---
""")

    # Patch resolve_single_model to return our base
    from mlx_knife import cache_utils as cu

    def fake_resolve(spec):
        return base, "org/mlxish", "main"

    called = {"ok": False}

    def fake_run_enhanced(**kwargs):
        called["ok"] = True

    monkeypatch.setattr(cu, "resolve_single_model", fake_resolve)
    import mlx_knife.mlx_runner as mr
    monkeypatch.setattr(mr, "run_model_enhanced", fake_run_enhanced, raising=False)

    # Should not raise or exit; should call enhanced runner
    run_model("org/mlxish", prompt="hi", interactive=False)
    assert called["ok"] is True


def _create_model_with_readme(temp_cache_dir: Path, org: str, name: str, readme_front_matter: str) -> Path:
    base = temp_cache_dir / "hub" / f"models--{org}--{name}" / "snapshots" / "main"
    base.mkdir(parents=True, exist_ok=True)
    (base / "config.json").write_text(json.dumps({"model_type": "llama"}))
    (base / "model.safetensors").write_bytes(b"weights" * 100)
    (base / "README.md").write_text(readme_front_matter)
    return base


def test_list_filters_non_chat_by_default(temp_cache_dir: Path, patch_model_cache, capsys):
    # Create a chat-capable MLX model via README front matter
    chat_front_matter = """---
tags: [mlx, chat]
pipeline_tag: text-generation
library_name: mlx
---
"""
    _create_model_with_readme(temp_cache_dir, "org", "chat-model", chat_front_matter)

    # Create a non-chat MLX model (embedding) via README front matter
    embed_front_matter = """---
tags: [mlx, embedding]
pipeline_tag: sentence-similarity
library_name: mlx
---
"""
    _create_model_with_readme(temp_cache_dir, "org", "embed-model", embed_front_matter)

    from mlx_knife.cache_utils import list_models
    with patch_model_cache(temp_cache_dir / "hub"):
        list_models()  # default strict view
    out = capsys.readouterr().out
    assert "org/chat-model" in out
    assert "org/embed-model" not in out  # non-chat should be hidden in strict view


def test_list_all_includes_non_chat_with_type_column(temp_cache_dir: Path, patch_model_cache, capsys):
    # Reuse the same setup as previous test
    chat_front_matter = """---
tags: [mlx, chat]
pipeline_tag: text-generation
library_name: mlx
---
"""
    _create_model_with_readme(temp_cache_dir, "org2", "chat-model2", chat_front_matter)

    embed_front_matter = """---
tags: [mlx, embedding]
pipeline_tag: sentence-similarity
library_name: mlx
---
"""
    _create_model_with_readme(temp_cache_dir, "org2", "embed-model2", embed_front_matter)

    from mlx_knife.cache_utils import list_models
    with patch_model_cache(temp_cache_dir / "hub"):
        list_models(show_all=True)
    out = capsys.readouterr().out
    # Header contains TYPE column in --all mode
    assert "TYPE" in out.splitlines()[0]
    # Both models appear
    assert "org2/chat-model2" in out
    assert "org2/embed-model2" in out
