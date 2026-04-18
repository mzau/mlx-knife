"""Tests for convert --quantize dispatch policy (ADR-023 Text-First + Verified Multimodal).

These tests lock the policy contract:
- Models with verified model_type route to the correct backend.
- Models with multimodal markers (vision_config / audio_config) but an
  unverified model_type are rejected with ErrorType.UNSUPPORTED_MULTIMODAL
  before any filesystem side effect on the target path.
- STT model types are rejected with NOT_IMPLEMENTED.
- Pure text models route to the text backend.

The backends are monkey-patched to fail loudly; a reject path that accidentally
falls through to a backend would fail the test, not silently produce data.
"""

import json
from pathlib import Path
from unittest.mock import patch

import pytest

from mlxk2.operations.convert import convert_operation, QUANTIZE_BACKENDS


def _write_source_workspace(path: Path, config: dict) -> None:
    """Create a minimal source workspace with the given config.json."""
    path.mkdir(parents=True, exist_ok=True)
    (path / "config.json").write_text(json.dumps(config))
    # mlx-lm / mlx-vlm backends would also require weights; we monkey-patch
    # the backends so weights never actually get touched in these tests.


def _backend_must_not_run(*_args, **_kwargs):
    """Fixture for monkey-patched backends: calling them is a test failure."""
    pytest.fail("backend was called; reject path did not fire before dispatch")


class TestConvertMultimodalReject:
    """Policy Gate: multimodal markers without verified model_type -> hard reject."""

    def test_gemma3n_quantize_hard_rejects(self, tmp_path):
        """gemma3n carries vision_config + audio_config but is not in VISION_QUANTIZE_TYPES.

        Must return error.type == "unsupported_multimodal", no backend call,
        no target directory created.
        """
        src = tmp_path / "gemma3n-src"
        dst = tmp_path / "gemma3n-q4"
        _write_source_workspace(src, {
            "model_type": "gemma3n",
            "vision_config": {"some": "marker"},
            "audio_config": {"some": "marker"},
        })

        with patch.dict(QUANTIZE_BACKENDS, {"text": _backend_must_not_run, "vision": _backend_must_not_run}):
            result = convert_operation(
                source_path=str(src),
                target_path=str(dst),
                mode="quantize",
                mode_opts={"bits": 4},
            )

        assert result["status"] == "error"
        assert result["error"]["type"] == "unsupported_multimodal"
        assert "gemma3n" in result["error"]["message"]
        assert not dst.exists(), "target path must not be created on reject"

    def test_unknown_vision_type_with_vision_config_rejects(self, tmp_path):
        """A hypothetical future vision type not in the whitelist must reject,
        not silently degrade to the text backend."""
        src = tmp_path / "qwen3_vl-src"
        dst = tmp_path / "qwen3_vl-q4"
        _write_source_workspace(src, {
            "model_type": "qwen3_vl",
            "vision_config": {},
        })

        with patch.dict(QUANTIZE_BACKENDS, {"text": _backend_must_not_run, "vision": _backend_must_not_run}):
            result = convert_operation(
                source_path=str(src),
                target_path=str(dst),
                mode="quantize",
                mode_opts={"bits": 4},
            )

        assert result["status"] == "error"
        assert result["error"]["type"] == "unsupported_multimodal"
        assert not dst.exists()

    def test_whisper_quantize_rejects_as_not_implemented(self, tmp_path):
        """Whisper is STT and not currently quantize-supported."""
        src = tmp_path / "whisper-src"
        dst = tmp_path / "whisper-q4"
        _write_source_workspace(src, {"model_type": "whisper"})

        with patch.dict(QUANTIZE_BACKENDS, {"text": _backend_must_not_run, "vision": _backend_must_not_run}):
            result = convert_operation(
                source_path=str(src),
                target_path=str(dst),
                mode="quantize",
                mode_opts={"bits": 4},
            )

        assert result["status"] == "error"
        assert result["error"]["type"] == "not_implemented"
        assert not dst.exists()


class TestConvertRoutesCorrectly:
    """Positive routing: verified types land on the right backend."""

    def test_gemma3_quantize_routes_vision(self, tmp_path):
        """gemma3 is in VISION_QUANTIZE_TYPES and carries vision_config.

        Must dispatch to vision backend, not text backend.
        """
        src = tmp_path / "gemma3-src"
        dst = tmp_path / "gemma3-q4"
        _write_source_workspace(src, {
            "model_type": "gemma3",
            "vision_config": {"some": "marker"},
        })

        called = {"text": False, "vision": False}

        def _mark_text(*args, **kwargs): called["text"] = True
        def _mark_vision(*args, **kwargs): called["vision"] = True

        with patch.dict(QUANTIZE_BACKENDS, {"text": _mark_text, "vision": _mark_vision}), \
             patch("mlxk2.operations.convert.write_workspace_sentinel"), \
             patch("mlxk2.operations.convert.update_workspace_hash", return_value=(True, "deadbeef")):
            result = convert_operation(
                source_path=str(src),
                target_path=str(dst),
                mode="quantize",
                mode_opts={"bits": 4},
                skip_health=True,
            )

        assert result["status"] == "success", f"expected success, got {result}"
        assert called["vision"] is True
        assert called["text"] is False

    def test_pure_text_quantize_routes_text(self, tmp_path):
        """llama has no multimodal markers and routes to the text backend."""
        src = tmp_path / "llama-src"
        dst = tmp_path / "llama-q4"
        _write_source_workspace(src, {"model_type": "llama"})

        called = {"text": False, "vision": False}

        def _mark_text(*args, **kwargs): called["text"] = True
        def _mark_vision(*args, **kwargs): called["vision"] = True

        with patch.dict(QUANTIZE_BACKENDS, {"text": _mark_text, "vision": _mark_vision}), \
             patch("mlxk2.operations.convert.write_workspace_sentinel"), \
             patch("mlxk2.operations.convert.update_workspace_hash", return_value=(True, "deadbeef")):
            result = convert_operation(
                source_path=str(src),
                target_path=str(dst),
                mode="quantize",
                mode_opts={"bits": 4},
                skip_health=True,
            )

        assert result["status"] == "success", f"expected success, got {result}"
        assert called["text"] is True
        assert called["vision"] is False
