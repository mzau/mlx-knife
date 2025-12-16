"""Tests for ADR-012 Phase 2: Vision model auxiliary asset health checks.

Validates that health checks properly verify vision-specific assets:
- preprocessor_config.json (required for vision models)
- tokenizer.json + tokenizer_config.json (required if tokenizer_config exists)
"""

from __future__ import annotations

import sys
import pytest
from pathlib import Path

from mlxk2.operations.health import _check_snapshot_health


def test_vision_model_missing_preprocessor_is_unhealthy(isolated_cache):
    """Vision model without preprocessor_config.json should be unhealthy."""
    snap = isolated_cache / "models--test--vision-no-preprocessor" / "snapshots" / "main"
    snap.mkdir(parents=True)

    # Basic vision model files (missing preprocessor_config.json)
    (snap / "config.json").write_text('{"model_type": "llava"}', encoding="utf-8")
    (snap / "model.safetensors").write_bytes(b"weights")
    (snap / "tokenizer.json").write_text('{}', encoding="utf-8")
    (snap / "tokenizer_config.json").write_text('{"chat_template": "test"}', encoding="utf-8")

    healthy, reason = _check_snapshot_health(snap)
    assert not healthy
    assert "preprocessor_config.json" in reason.lower()


def test_vision_model_with_preprocessor_is_healthy(isolated_cache):
    """Vision model with all required assets should be healthy."""
    snap = isolated_cache / "models--test--vision-complete" / "snapshots" / "main"
    snap.mkdir(parents=True)

    # Complete vision model files
    (snap / "config.json").write_text('{"model_type": "llava"}', encoding="utf-8")
    (snap / "model.safetensors").write_bytes(b"weights")
    (snap / "preprocessor_config.json").write_text('{"size": 224}', encoding="utf-8")
    (snap / "tokenizer.json").write_text('{}', encoding="utf-8")
    (snap / "tokenizer_config.json").write_text('{"chat_template": "test"}', encoding="utf-8")

    healthy, reason = _check_snapshot_health(snap)
    assert healthy, f"Expected healthy but got: {reason}"


def test_vision_model_invalid_preprocessor_is_unhealthy(isolated_cache):
    """Vision model with invalid preprocessor_config.json should be unhealthy."""
    snap = isolated_cache / "models--test--vision-bad-preprocessor" / "snapshots" / "main"
    snap.mkdir(parents=True)

    (snap / "config.json").write_text('{"model_type": "llava"}', encoding="utf-8")
    (snap / "model.safetensors").write_bytes(b"weights")
    (snap / "preprocessor_config.json").write_text('not json', encoding="utf-8")

    healthy, reason = _check_snapshot_health(snap)
    assert not healthy
    assert "preprocessor_config.json" in reason.lower()


def test_chat_model_missing_tokenizer_json_is_unhealthy(isolated_cache):
    """Chat model with tokenizer_config but missing tokenizer.json is unhealthy."""
    snap = isolated_cache / "models--test--chat-no-tokenizer" / "snapshots" / "main"
    snap.mkdir(parents=True)

    (snap / "config.json").write_text('{"model_type": "llama"}', encoding="utf-8")
    (snap / "model.safetensors").write_bytes(b"weights")
    (snap / "tokenizer_config.json").write_text('{"chat_template": "test"}', encoding="utf-8")
    # Missing tokenizer.json

    healthy, reason = _check_snapshot_health(snap)
    assert not healthy
    assert "tokenizer.json missing" in reason.lower()


def test_chat_model_invalid_tokenizer_config_is_unhealthy(isolated_cache):
    """Chat model with invalid tokenizer_config.json should be unhealthy."""
    snap = isolated_cache / "models--test--chat-bad-tokenizer-config" / "snapshots" / "main"
    snap.mkdir(parents=True)

    (snap / "config.json").write_text('{"model_type": "llama"}', encoding="utf-8")
    (snap / "model.safetensors").write_bytes(b"weights")
    (snap / "tokenizer_config.json").write_text('not json', encoding="utf-8")

    healthy, reason = _check_snapshot_health(snap)
    assert not healthy
    assert "tokenizer_config.json" in reason.lower() and "invalid" in reason.lower()


def test_chat_model_without_tokenizer_is_healthy(isolated_cache):
    """Chat model without any tokenizer files can still be healthy (base models)."""
    snap = isolated_cache / "models--test--base-no-tokenizer" / "snapshots" / "main"
    snap.mkdir(parents=True)

    (snap / "config.json").write_text('{"model_type": "llama"}', encoding="utf-8")
    (snap / "model.safetensors").write_bytes(b"weights")

    healthy, reason = _check_snapshot_health(snap)
    assert healthy, f"Expected healthy but got: {reason}"


def test_chat_model_with_complete_tokenizer_is_healthy(isolated_cache):
    """Chat model with complete tokenizer assets should be healthy."""
    snap = isolated_cache / "models--test--chat-complete-tokenizer" / "snapshots" / "main"
    snap.mkdir(parents=True)

    (snap / "config.json").write_text('{"model_type": "llama"}', encoding="utf-8")
    (snap / "model.safetensors").write_bytes(b"weights")
    (snap / "tokenizer.json").write_text('{}', encoding="utf-8")
    (snap / "tokenizer_config.json").write_text('{"chat_template": "test"}', encoding="utf-8")

    healthy, reason = _check_snapshot_health(snap)
    assert healthy, f"Expected healthy but got: {reason}"


@pytest.mark.skipif(
    sys.version_info < (3, 10),
    reason="Vision detection (preprocessor_config.json) works on all Python versions"
)
def test_vision_detection_via_preprocessor_file(isolated_cache):
    """Vision model with preprocessor_config.json but non-vision model_type."""
    snap = isolated_cache / "models--test--vision-via-preprocessor" / "snapshots" / "main"
    snap.mkdir(parents=True)

    # Non-vision model_type, but has preprocessor_config.json
    (snap / "config.json").write_text('{"model_type": "base"}', encoding="utf-8")
    (snap / "model.safetensors").write_bytes(b"weights")
    (snap / "preprocessor_config.json").write_text('{"size": 224}', encoding="utf-8")
    (snap / "tokenizer.json").write_text('{}', encoding="utf-8")
    (snap / "tokenizer_config.json").write_text('{}', encoding="utf-8")

    # Should be detected as vision and require preprocessor_config.json
    healthy, reason = _check_snapshot_health(snap)
    assert healthy, f"Expected healthy but got: {reason}"
