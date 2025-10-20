"""Tests for runtime compatibility reason field decision chain (Issue #36).

Critical requirement: The reason field must reflect the FIRST problem encountered:
1. Health check failures take precedence over runtime failures
2. Gate 1 (framework) failures take precedence over Gate 2 (legacy format)
3. Gate 2 (legacy format) failures take precedence over Gate 3 (model_type)
4. Only when ALL checks pass should reason be None

This ensures users see the most actionable error message first.
"""

import json
from pathlib import Path
import pytest
import sys

# Check if mlx-lm is available for Gate 3 tests
# Note: Gate 3 tests require a working mlx-lm installation with _get_classes API
# Current implementation has compatibility issues with mlx-lm 0.28.x API changes
try:
    import mlx_lm
    # Try to import the function we actually need
    try:
        from mlx_lm.models.base import _get_classes
        HAS_WORKING_MLX_LM = True
    except ImportError:
        # Try old API
        try:
            from mlx_lm.utils import _get_classes
            HAS_WORKING_MLX_LM = True
        except ImportError:
            HAS_WORKING_MLX_LM = False
except ImportError:
    HAS_WORKING_MLX_LM = False

requires_mlx_lm = pytest.mark.skipif(
    not HAS_WORKING_MLX_LM,
    reason="mlx-lm not available or _get_classes API not found (required for Gate 3)"
)


# ============================================================================
# Test Helpers
# ============================================================================

def _create_config(snap: Path, model_type: str = "llama"):
    """Create a minimal valid config.json."""
    config = {
        "model_type": model_type,
        "architectures": ["LlamaForCausalLM"]
    }
    (snap / "config.json").write_text(json.dumps(config))


def _create_healthy_mlx_model(cache: Path, name: str, weights_pattern: str = "model.safetensors", model_type: str = "llama"):
    """Create a healthy MLX model with specified weight file pattern."""
    snap = cache / f"models--{name.replace('/', '--')}" / "snapshots" / "main"
    snap.mkdir(parents=True)
    _create_config(snap, model_type)
    (snap / weights_pattern).write_bytes(b"fake_weights" * 100)
    return snap


# ============================================================================
# Reason Chain Tests: Health Check Precedence (Highest Priority)
# ============================================================================

def test_reason_chain_health_failure_overrides_framework_failure(isolated_cache):
    """Health failure reason should take precedence over framework incompatibility.

    Scenario: GGUF model (would fail Gate 1) but missing config.json (health failure)
    Expected: reason = "config.json missing" (NOT "Incompatible: GGUF")
    """
    snap = isolated_cache / "models--test--broken-gguf" / "snapshots" / "main"
    snap.mkdir(parents=True)
    # Missing config.json → unhealthy
    (snap / "model.gguf").write_bytes(b"fake" * 100)

    from mlxk2.operations.common import build_model_object
    model_obj = build_model_object("test/broken-gguf", snap.parent.parent, snap)

    assert model_obj["health"] == "unhealthy"
    assert model_obj["runtime_compatible"] is False
    assert "config.json" in model_obj["reason"]
    assert "GGUF" not in model_obj["reason"], "Framework reason should not appear when health check fails"


def test_reason_chain_health_failure_overrides_legacy_format(isolated_cache):
    """Health failure should take precedence over legacy format detection.

    Scenario: Legacy weights but missing config.json
    Expected: reason = "config.json missing" (NOT "Legacy format")
    """
    snap = isolated_cache / "models--test--broken-legacy" / "snapshots" / "main"
    snap.mkdir(parents=True)
    # Missing config.json → unhealthy
    (snap / "weights.00.safetensors").write_bytes(b"fake" * 100)

    from mlxk2.operations.common import build_model_object
    model_obj = build_model_object("test/broken-legacy", snap.parent.parent, snap)

    assert model_obj["health"] == "unhealthy"
    assert model_obj["runtime_compatible"] is False
    assert "config.json" in model_obj["reason"]
    assert "Legacy" not in model_obj["reason"], "Legacy format reason should not appear when health check fails"


# ============================================================================
# Reason Chain Tests: Gate 1 (Framework) Precedence
# ============================================================================

def test_reason_chain_gate1_framework_check_gguf(isolated_cache):
    """Gate 1 failure (GGUF) should be reported even if model has legacy weights.

    Scenario: Healthy GGUF model with legacy-named files
    Expected: reason = "Incompatible: GGUF" (Gate 1 fails, Gate 2 never runs)
    """
    snap = _create_healthy_mlx_model(isolated_cache, "test/gguf-legacy", "weights.00.gguf")

    from mlxk2.operations.health import check_runtime_compatibility
    compatible, reason = check_runtime_compatibility(snap, "GGUF")

    assert compatible is False
    assert reason == "Incompatible: GGUF"
    # Gate 1 fails early, so legacy format detection (Gate 2) never runs


def test_reason_chain_gate1_framework_check_pytorch(isolated_cache):
    """Gate 1 failure (PyTorch) should take precedence."""
    snap = _create_healthy_mlx_model(isolated_cache, "test/pytorch", "model.safetensors")

    from mlxk2.operations.health import check_runtime_compatibility
    compatible, reason = check_runtime_compatibility(snap, "PyTorch")

    assert compatible is False
    assert reason == "Incompatible: PyTorch"


def test_reason_chain_gate1_mlx_framework_passes(isolated_cache):
    """MLX framework should pass Gate 1 (may fail at later gates)."""
    snap = _create_healthy_mlx_model(isolated_cache, "test/mlx", "weights.00.safetensors")  # Legacy format

    from mlxk2.operations.health import check_runtime_compatibility
    compatible, reason = check_runtime_compatibility(snap, "MLX")

    # Should pass Gate 1 but fail at Gate 2 (legacy format)
    assert compatible is False
    assert "Incompatible: MLX" not in reason, "Should not fail at Gate 1 for MLX framework"
    assert "Legacy format" in reason, "Should fail at Gate 2 for legacy weights"


# ============================================================================
# Reason Chain Tests: Gate 2 (Legacy Format) Precedence
# ============================================================================

def test_reason_chain_gate2_legacy_weights_numeric(isolated_cache):
    """Gate 2 should detect weights.NN.safetensors legacy format.

    Even if model_type might be unsupported (Gate 3), legacy format (Gate 2) is reported first.
    """
    snap = isolated_cache / "models--test--legacy-weights" / "snapshots" / "main"
    snap.mkdir(parents=True)
    _create_config(snap, model_type="some_fake_unsupported_type_xyz")  # Would fail Gate 3
    (snap / "weights.00.safetensors").write_bytes(b"fake" * 100)
    (snap / "weights.01.safetensors").write_bytes(b"fake" * 100)

    from mlxk2.operations.health import check_runtime_compatibility
    compatible, reason = check_runtime_compatibility(snap, "MLX")

    assert compatible is False
    assert "Legacy format not supported by mlx-lm" in reason
    # Gate 2 fails, so Gate 3 (model_type check) never runs


def test_reason_chain_gate2_legacy_pytorch_model_numeric(isolated_cache):
    """Gate 2 should detect pytorch_model-NNNNN.safetensors legacy format."""
    snap = isolated_cache / "models--test--legacy-pytorch" / "snapshots" / "main"
    snap.mkdir(parents=True)
    _create_config(snap, model_type="some_fake_type")  # Would fail Gate 3
    (snap / "pytorch_model-00001.safetensors").write_bytes(b"fake" * 100)

    from mlxk2.operations.health import check_runtime_compatibility
    compatible, reason = check_runtime_compatibility(snap, "MLX")

    assert compatible is False
    assert "Legacy format not supported by mlx-lm" in reason


def test_reason_chain_gate2_modern_format_passes(isolated_cache):
    """Modern model.safetensors should pass Gate 2."""
    snap = isolated_cache / "models--test--modern" / "snapshots" / "main"
    snap.mkdir(parents=True)
    _create_config(snap, model_type="llama")
    (snap / "model.safetensors").write_bytes(b"fake" * 100)

    from mlxk2.operations.health import check_runtime_compatibility
    compatible, reason = check_runtime_compatibility(snap, "MLX")

    # Should pass Gates 1 and 2, outcome depends on Gate 3 (model_type check)
    # If mlx-lm supports llama, should be compatible
    # If not compatible, reason should NOT be about legacy format
    if not compatible:
        assert "Legacy format" not in reason, "Modern format should not trigger legacy format error"


def test_reason_chain_gate2_sharded_modern_format_passes(isolated_cache):
    """Modern sharded model-XXXXX-of-YYYYY.safetensors should pass Gate 2."""
    snap = isolated_cache / "models--test--sharded" / "snapshots" / "main"
    snap.mkdir(parents=True)
    _create_config(snap, model_type="llama")
    (snap / "model-00001-of-00002.safetensors").write_bytes(b"fake" * 100)
    (snap / "model-00002-of-00002.safetensors").write_bytes(b"fake" * 100)

    from mlxk2.operations.health import check_runtime_compatibility
    compatible, reason = check_runtime_compatibility(snap, "MLX")

    # Should pass Gate 2
    if not compatible:
        assert "Legacy format" not in reason, "Modern sharded format should not trigger legacy format error"


# ============================================================================
# Reason Chain Tests: Gate 3 (model_type Support)
# ============================================================================

@requires_mlx_lm
def test_reason_chain_gate3_unsupported_model_type(isolated_cache):
    """Gate 3 should only run if Gates 1 and 2 pass.

    This test uses a clearly fake model_type that mlx-lm won't support.
    The error should be about model_type, not about framework or legacy format.
    """
    snap = isolated_cache / "models--test--unsupported-arch" / "snapshots" / "main"
    snap.mkdir(parents=True)
    _create_config(snap, model_type="definitely_not_a_real_architecture_xyz123")
    (snap / "model.safetensors").write_bytes(b"fake" * 100)

    from mlxk2.operations.health import check_runtime_compatibility
    compatible, reason = check_runtime_compatibility(snap, "MLX")

    # Should fail at Gate 3
    assert compatible is False
    assert reason is not None
    # Reason should be about model_type, not framework or legacy format
    assert "Incompatible:" not in reason, "Should not fail at Gate 1"
    assert "Legacy format" not in reason, "Should not fail at Gate 2"
    # Should mention model_type or architecture
    assert ("model_type" in reason.lower() or
            "not supported" in reason.lower() or
            "architecture" in reason.lower()), f"Gate 3 should report model_type issue, got: {reason}"


@requires_mlx_lm
def test_reason_chain_gate3_supported_model_type_llama(isolated_cache):
    """Well-known supported model_type (llama) should pass all gates.

    This is the happy path: MLX framework, modern format, supported architecture.
    """
    snap = isolated_cache / "models--test--llama-supported" / "snapshots" / "main"
    snap.mkdir(parents=True)
    _create_config(snap, model_type="llama")  # Well-known supported type
    (snap / "model.safetensors").write_bytes(b"fake" * 100)

    from mlxk2.operations.health import check_runtime_compatibility
    compatible, reason = check_runtime_compatibility(snap, "MLX")

    # Should pass all gates (assuming mlx-lm supports llama)
    assert compatible is True, f"llama model_type should be supported, got reason: {reason}"
    assert reason is None, "Fully compatible models should have reason=None"


# ============================================================================
# Integration Tests: build_model_object() Reason Field
# ============================================================================

def test_integration_reason_field_unhealthy_model(isolated_cache):
    """Integration: Unhealthy model should show health reason in model object."""
    snap = isolated_cache / "models--test--no-config" / "snapshots" / "main"
    snap.mkdir(parents=True)
    # Missing config.json → unhealthy
    (snap / "model.safetensors").write_bytes(b"fake" * 100)

    from mlxk2.operations.common import build_model_object
    model_obj = build_model_object("test/no-config", snap.parent.parent, snap)

    assert model_obj["health"] == "unhealthy"
    assert model_obj["runtime_compatible"] is False
    assert "config.json" in model_obj["reason"]


def test_integration_reason_field_gguf_model(isolated_cache):
    """Integration: Healthy GGUF model should show framework incompatibility."""
    snap = isolated_cache / "models--test--gguf-healthy" / "snapshots" / "main"
    snap.mkdir(parents=True)
    _create_config(snap)
    (snap / "model.gguf").write_bytes(b"fake" * 100)

    from mlxk2.operations.common import build_model_object
    model_obj = build_model_object("test/gguf-healthy", snap.parent.parent, snap)

    assert model_obj["health"] == "healthy"
    assert model_obj["runtime_compatible"] is False
    assert "Incompatible: GGUF" in model_obj["reason"] or "GGUF" in model_obj["reason"]


def test_integration_reason_field_legacy_mlx_model(isolated_cache):
    """Integration: Healthy MLX model with legacy weights should show legacy format reason.

    Important: Model must be recognized as MLX framework (via mlx-community prefix)
    so it passes Gate 1 and reaches Gate 2 (legacy format check).
    """
    snap = isolated_cache / "models--mlx-community--legacy-test" / "snapshots" / "main"
    snap.mkdir(parents=True)
    _create_config(snap)
    (snap / "weights.00.safetensors").write_bytes(b"fake" * 100)

    from mlxk2.operations.common import build_model_object
    model_obj = build_model_object("mlx-community/legacy-test", snap.parent.parent, snap)

    assert model_obj["health"] == "healthy"
    assert model_obj["framework"] == "MLX", "Model should be detected as MLX framework"
    assert model_obj["runtime_compatible"] is False
    assert "Legacy format" in model_obj["reason"]


def test_integration_reason_field_compatible_mlx_model(isolated_cache):
    """Integration: Fully compatible MLX model should have reason=None."""
    snap = isolated_cache / "models--test--mlx-compatible" / "snapshots" / "main"
    snap.mkdir(parents=True)
    _create_config(snap, model_type="llama")  # Well-known supported type
    (snap / "model.safetensors").write_bytes(b"fake" * 100)

    from mlxk2.operations.common import build_model_object
    model_obj = build_model_object("test/mlx-compatible", snap.parent.parent, snap)

    assert model_obj["health"] == "healthy"
    # Should be compatible (assuming mlx-lm supports llama)
    if model_obj["runtime_compatible"]:
        assert model_obj["reason"] is None, "Fully compatible models must have reason=None"


# ============================================================================
# Edge Cases
# ============================================================================

def test_reason_chain_mixed_legacy_and_modern_weights(isolated_cache):
    """Model with BOTH legacy and modern weights should pass Gate 2.

    Gate 2 logic: `if has_legacy and not has_valid` → fail
    If has_valid=True (modern weights exist), should NOT fail at Gate 2.
    """
    snap = isolated_cache / "models--test--mixed-weights" / "snapshots" / "main"
    snap.mkdir(parents=True)
    _create_config(snap, model_type="llama")
    # Both legacy and modern formats present
    (snap / "weights.00.safetensors").write_bytes(b"fake" * 100)  # Legacy
    (snap / "model.safetensors").write_bytes(b"fake" * 100)  # Modern

    from mlxk2.operations.health import check_runtime_compatibility
    compatible, reason = check_runtime_compatibility(snap, "MLX")

    # Should pass Gate 2 (has valid modern weights)
    if not compatible:
        assert "Legacy format" not in reason, "Should not fail at Gate 2 when modern weights exist"


def test_reason_chain_no_weights_at_all(isolated_cache):
    """Model with config but no weights should fail health check, not runtime check."""
    snap = isolated_cache / "models--test--no-weights" / "snapshots" / "main"
    snap.mkdir(parents=True)
    _create_config(snap)
    # No weight files at all

    from mlxk2.operations.health import _check_snapshot_health
    healthy, reason = _check_snapshot_health(snap)

    # Should fail health check
    assert healthy is False
    assert "weights" in reason.lower() or "No model weights" in reason
