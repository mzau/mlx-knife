"""Tests for legacy model format detection (Issue #37).

Note: These tests focus on legacy format detection only.
Runtime compatibility tests for modern formats (Issue #36) are pending.
"""

import json
from pathlib import Path


def test_weights_numeric_safetensors_is_runtime_incompatible(isolated_cache):
    """Legacy weights.00.safetensors format should be healthy but runtime incompatible."""
    snap = isolated_cache / "models--test--legacy-weights" / "snapshots" / "main"
    snap.mkdir(parents=True)

    # Create config.json (required for health check)
    config = {
        "model_type": "llama",
        "architectures": ["LlamaForCausalLM"]
    }
    (snap / "config.json").write_text(json.dumps(config))

    # Create legacy weight file
    (snap / "weights.00.safetensors").write_bytes(b"fake_weights" * 100)

    from mlxk2.operations.health import _check_snapshot_health, check_runtime_compatibility

    # Health check should pass (files are complete)
    healthy, reason = _check_snapshot_health(snap)
    assert healthy is True, f"Expected healthy, got: {reason}"

    # Runtime compatibility should fail due to legacy format
    compatible, reason = check_runtime_compatibility(snap, "MLX")
    assert compatible is False
    assert "Legacy format not supported by mlx-lm" in reason


def test_pytorch_model_numeric_safetensors_is_runtime_incompatible(isolated_cache):
    """Legacy pytorch_model-00001.safetensors format should be runtime incompatible."""
    snap = isolated_cache / "models--test--legacy-pytorch" / "snapshots" / "main"
    snap.mkdir(parents=True)

    config = {
        "model_type": "llama",
        "architectures": ["LlamaForCausalLM"]
    }
    (snap / "config.json").write_text(json.dumps(config))

    # Create legacy pytorch_model files
    (snap / "pytorch_model-00001.safetensors").write_bytes(b"fake" * 100)
    (snap / "pytorch_model-00002.safetensors").write_bytes(b"fake" * 100)

    from mlxk2.operations.health import _check_snapshot_health, check_runtime_compatibility

    healthy, reason = _check_snapshot_health(snap)
    assert healthy is True

    # Runtime compatibility should fail due to legacy format
    compatible, reason = check_runtime_compatibility(snap, "MLX")
    assert compatible is False
    assert "Legacy format not supported by mlx-lm" in reason


def test_modern_model_safetensors_passes_legacy_gate(isolated_cache):
    """Modern model.safetensors should pass the legacy format gate (Gate 2).

    This test verifies that modern formats are NOT rejected by the legacy format check.
    Full runtime compatibility (Gate 3: model_type check) is not tested here.
    """
    snap = isolated_cache / "models--test--modern" / "snapshots" / "main"
    snap.mkdir(parents=True)

    config = {
        "model_type": "llama",
        "architectures": ["LlamaForCausalLM"]
    }
    (snap / "config.json").write_text(json.dumps(config))

    # Create modern weight file
    (snap / "model.safetensors").write_bytes(b"fake_weights" * 100)

    from mlxk2.operations.health import _check_snapshot_health, check_runtime_compatibility

    healthy, reason = _check_snapshot_health(snap)
    assert healthy is True

    # Should NOT be rejected by legacy format check (Gate 2)
    # Note: May still fail at Gate 3 (model_type) if mlx-lm is not available
    compatible, reason = check_runtime_compatibility(snap, "MLX")

    # If it failed, it should NOT be due to legacy format
    if not compatible:
        assert "Legacy format" not in reason, f"Should not fail due to legacy format, but got: {reason}"
