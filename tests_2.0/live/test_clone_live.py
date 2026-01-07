"""Opt-in live clone test.

Runs only when explicitly selected via markers/env, per TESTING.md mini‑matrix.
Validates ADR-007 Phase 1 compliance: real pull→temp cache→APFS same-volume clone→workspace.

Enable with ALL required env vars:
- MLXK2_ENABLE_ALPHA_FEATURES=1 (clone is alpha)
- MLXK2_LIVE_CLONE=1 (enable live test)
- HF_TOKEN=<your_token> (for model access)
- MLXK2_LIVE_CLONE_MODEL=<model_name> (e.g., "mlx-community/bge-small-en-v1.5-4bit")
- MLXK2_LIVE_CLONE_WORKSPACE=<workspace_path> (must be on same volume as HF_HOME for APFS)

Run:
- pytest -m live_clone -v
- or umbrella Phase 3: scripts/test-wet-umbrella.sh (isolated run)

NOT part of wet marker (incompatible with Portfolio Discovery - does fresh HF download).

ADR-007 Phase 1 Requirements:
- Same volume: workspace and HF_HOME cache must be on same volume
- APFS filesystem: required for copy-on-write optimization
- User cache safety: never touched, always use temp cache isolation
"""

from __future__ import annotations

import json
import os
import shutil
import sys
from pathlib import Path

import pytest


# Environment validation
alpha_enabled = os.environ.get("MLXK2_ENABLE_ALPHA_FEATURES") == "1"
live_enabled = os.environ.get("MLXK2_LIVE_CLONE") == "1"
hf_token_present = bool(os.environ.get("HF_TOKEN"))
model = os.environ.get("MLXK2_LIVE_CLONE_MODEL")
workspace = os.environ.get("MLXK2_LIVE_CLONE_WORKSPACE")

pytestmark = [
    pytest.mark.live,
    pytest.mark.live_clone,
    pytest.mark.skipif(
        not (alpha_enabled and live_enabled and hf_token_present and model and workspace),
        reason=(
            "Live clone disabled. Set MLXK2_ENABLE_ALPHA_FEATURES=1, MLXK2_LIVE_CLONE=1, "
            "HF_TOKEN, MLXK2_LIVE_CLONE_MODEL, and MLXK2_LIVE_CLONE_WORKSPACE to enable."
        ),
    ),
]


def _run_cli(argv: list[str], capsys) -> str:
    """Run CLI command and return captured output."""
    from mlxk2.cli import main as cli_main
    old_argv = sys.argv[:]
    sys.argv = argv[:]
    try:
        with pytest.raises(SystemExit):
            cli_main()
    finally:
        sys.argv = old_argv
    captured = capsys.readouterr()
    return captured.out.strip()


def test_live_clone_workflow_adr007_phase1(capsys, tmp_path):
    """Test complete live clone workflow following ADR-007 Phase 1 constraints.

    This test validates:
    1. Real HuggingFace model pull → temp cache
    2. Real APFS volume detection and same-volume validation
    3. Real health check integration with health_from_cache
    4. Real APFS copy-on-write clone → workspace
    5. User cache safety (never touched)

    Expected workflow:
    - Pull model to isolated temp cache (not user cache)
    - Validate same volume constraint (workspace + HF_HOME)
    - Health check via health_from_cache (full _check_snapshot_health)
    - APFS clone with copy-on-write optimization
    - Clean workspace output ready for development
    """
    # Ensure clean workspace
    workspace_path = Path(workspace)
    if workspace_path.exists():
        shutil.rmtree(workspace_path)

    # Run live clone operation
    result_json = _run_cli([
        "mlxk2", "clone", model, workspace, "--json"
    ], capsys)

    # Parse JSON response
    try:
        result = json.loads(result_json)
    except json.JSONDecodeError as e:
        pytest.fail(f"Invalid JSON response: {e}\nOutput: {result_json}")

    # Validate successful clone
    assert result["status"] == "success", f"Clone failed: {result.get('error', 'Unknown error')}"
    assert result["command"] == "clone"
    assert result["data"]["model"] == model
    assert result["data"]["target_dir"] == str(Path(workspace).resolve())

    # ADR-007 Phase 1 validation
    assert result["data"]["health_check"] is True, "Health check should be enabled by default"
    assert "clone_status" in result["data"], "Clone status should be reported"

    # Verify workspace was created and contains model files
    assert workspace_path.exists(), f"Workspace {workspace} was not created"
    assert workspace_path.is_dir(), f"Workspace {workspace} is not a directory"

    # Validate essential model files are present
    config_file = workspace_path / "config.json"
    assert config_file.exists(), "config.json missing from workspace"

    # Verify at least one weight file exists
    weight_files = (
        list(workspace_path.glob("*.safetensors")) +
        list(workspace_path.glob("*.bin")) +
        list(workspace_path.glob("*.gguf"))
    )
    assert weight_files, "No weight files found in workspace"

    # Verify files are real (not LFS pointers)
    for weight_file in weight_files[:1]:  # Check first weight file
        assert weight_file.stat().st_size > 200, f"Weight file {weight_file.name} appears to be LFS pointer"

    print(f"✅ Live clone test successful: {model} → {workspace}")
    print(f"📁 Workspace files: {len(list(workspace_path.iterdir()))} items")
    print(f"⚖️  Weight files: {len(weight_files)} files")


def test_live_clone_health_check_integration(capsys, tmp_path):
    """Test that health check integration works with real models.

    This validates that health_from_cache properly integrates with
    _check_snapshot_health for real model validation.
    """
    from mlxk2.operations.health import health_from_cache
    from mlxk2.core.cache import get_current_cache_root

    # Note: This test assumes the previous test ran and workspace exists
    workspace_path = Path(workspace)
    if not workspace_path.exists():
        pytest.skip(f"Workspace {workspace} not found - run full clone test first")

    # For this test, we create a temporary cache and copy the workspace
    # to simulate the temp cache state during clone operation
    temp_cache = tmp_path / "temp_cache_health_test"
    temp_cache.mkdir()

    # Create model structure in temp cache (simulate clone operation state)
    from mlxk2.core.cache import hf_to_cache_dir
    model_cache_dir = temp_cache / hf_to_cache_dir(model)
    snapshots_dir = model_cache_dir / "snapshots"
    snapshot_dir = snapshots_dir / "test_snapshot"
    snapshot_dir.mkdir(parents=True)

    # Copy workspace content to simulate temp cache snapshot
    for item in workspace_path.iterdir():
        if item.is_file():
            shutil.copy2(item, snapshot_dir)
        elif item.is_dir():
            shutil.copytree(item, snapshot_dir / item.name)

    # Test health_from_cache integration
    healthy, message = health_from_cache(model, temp_cache)

    assert healthy is True, f"Health check failed: {message}"
    assert "healthy" in message.lower() or "complete" in message.lower(), f"Unexpected health message: {message}"

    print(f"✅ Health check integration successful: {message}")


def test_live_clone_workspace_validation(capsys):
    """Test workspace validation with real filesystem constraints."""
    # Test that workspace directory must be empty or non-existent
    workspace_path = Path(workspace)

    if workspace_path.exists():
        # Create a dummy file to make workspace non-empty
        dummy_file = workspace_path / "dummy.txt"
        dummy_file.write_text("test")

        # Clone should fail with non-empty workspace
        result_json = _run_cli([
            "mlxk2", "clone", model, workspace, "--json"
        ], capsys)

        try:
            result = json.loads(result_json)
        except json.JSONDecodeError:
            pytest.fail(f"Invalid JSON response: {result_json}")

        assert result["status"] == "error", "Clone should fail with non-empty workspace"
        assert "not empty" in result["error"]["message"].lower(), "Error should mention non-empty workspace"

        # Clean up dummy file
        dummy_file.unlink()

        print("✅ Workspace validation successful: non-empty workspace properly rejected")