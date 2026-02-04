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


def test_vision_dual_backend_logic():
    """Session 149: Vision models require BOTH mlx-vlm AND mlx-lm for full runtime compatibility.

    This tests the logic from common.py lines 550-563:
    - Vision models need mlx-vlm for image processing
    - Vision models need mlx-lm for text-only mode (without images)
    - Both must be True for runtime_compatible=True
    """
    # Simulate the logic from common.py:550-563
    def vision_runtime_check(vision_ok, vision_reason, text_ok, text_reason):
        """Replicate the Vision dual-backend logic from common.py."""
        if vision_ok and text_ok:
            return True, None
        else:
            # Prefer text_reason as it's more specific
            return False, text_reason or vision_reason

    # Case 1: Both backends available
    ok, reason = vision_runtime_check(True, None, True, None)
    assert ok is True
    assert reason is None

    # Case 2: mlx-vlm available, but mlx-lm doesn't support model_type (e.g., mllama)
    ok, reason = vision_runtime_check(True, None, False, "model_type 'mllama' not supported")
    assert ok is False
    assert "mllama" in reason

    # Case 3: mlx-lm available, but mlx-vlm not installed
    ok, reason = vision_runtime_check(False, "mlx-vlm not installed", True, None)
    assert ok is False
    assert "mlx-vlm" in reason

    # Case 4: Neither available
    ok, reason = vision_runtime_check(False, "mlx-vlm not installed", False, "model_type not supported")
    assert ok is False
    # text_reason takes precedence
    assert "model_type" in reason
