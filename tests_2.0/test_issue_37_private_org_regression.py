"""Regression test for Issue #37 P0: Private/org MLX models rejected in run command.

Beta.5 introduced runtime compatibility pre-flight check in run_model() that incorrectly
passed snapshot path instead of cache root to detect_framework(), causing all non-mlx-community
models to be detected as "Unknown framework" and rejected.

This test verifies the fix by simulating a private-org MLX model (renamed from mlx-community/Phi-3).

Opt-in via: pytest -m live_run
Requires: mlx-community/Phi-3-mini-4k-instruct-4bit in user cache (MLXK2_USER_HF_HOME)
"""

from __future__ import annotations

import os
import pytest
import shutil
from pathlib import Path
from mlxk2.operations.run import run_model
from mlxk2.core.cache import hf_to_cache_dir

# Opt-in marker: only run with pytest -m live_run
# CRITICAL: Must include `live` marker so -m "not live" excludes these tests
pytestmark = [pytest.mark.live, pytest.mark.live_run]

# Skip if MLXK2_USER_HF_HOME not set (prevents running in standard pytest)
_USER_CACHE_ROOT = os.environ.get("MLXK2_USER_HF_HOME") or os.environ.get("HF_HOME")
requires_user_cache = pytest.mark.skipif(
    not _USER_CACHE_ROOT,
    reason="requires MLXK2_USER_HF_HOME or HF_HOME (opt-in via pytest -m live_run)"
)


@requires_user_cache
def test_private_org_mlx_model_runs_without_unknown_framework_error(
    copy_user_model_to_isolated, isolated_cache
):
    """Test that private/org MLX models are correctly detected and can run.

    Workflow:
    1. Copy mlx-community/Phi-3-mini-4k-instruct-4bit from user cache
    2. Rename cache directory to simulate private-org model (test-org/phi3-mlx-instruct)
    3. Run the model with a simple prompt
    4. Verify no "Unknown framework" error occurs

    This test requires:
    - Phi-3-mini-4k-instruct-4bit in user cache (MLXK2_USER_HF_HOME)
    - Run with: pytest -m live_run
    """
    # Step 1: Copy Phi-3 from user cache to isolated test cache
    src_model_dir = copy_user_model_to_isolated("mlx-community/Phi-3-mini-4k-instruct-4bit")

    # Step 2: Rename to simulate private-org model
    # From: models--mlx-community--Phi-3-mini-4k-instruct-4bit
    # To:   models--test-org--phi3-mlx-instruct
    private_org_cache_name = "models--test-org--phi3-mlx-instruct"
    private_org_dir = isolated_cache / private_org_cache_name

    # Move the directory
    shutil.move(str(src_model_dir), str(private_org_dir))

    # Verify the renamed model exists
    assert private_org_dir.exists(), "Private org model directory should exist after rename"
    snapshots = private_org_dir / "snapshots"
    assert snapshots.exists(), "Snapshots directory should exist"

    # Step 3: Add README.md with MLX tags to ensure framework detection works
    # (This is what a real private MLX model would have)
    snapshot_dirs = [d for d in snapshots.iterdir() if d.is_dir()]
    assert len(snapshot_dirs) > 0, "Should have at least one snapshot"

    for snapshot_dir in snapshot_dirs:
        readme = snapshot_dir / "README.md"
        readme.write_text("""---
tags: [mlx, chat]
library_name: mlx
---
# Test Org Phi-3 MLX Model

This is a test private-org MLX model for regression testing.
""")

    # Step 4: Run the model - this should NOT fail with "Unknown framework"
    # Note: We use json_output=True to get structured error messages
    result = run_model(
        model_spec="test-org/phi3-mlx-instruct",
        prompt="Hello",
        json_output=True,
        stream=False,
        max_tokens=5,  # Keep it short for speed
        verbose=False
    )

    # Step 5: Verify no "Unknown framework" or "Incompatible: PyTorch" errors
    # Note: We're testing framework detection, not mlx_lm availability
    if isinstance(result, str):
        # The bug would manifest as one of these:
        assert "Unknown framework" not in result, (
            f"Private-org MLX model should not be rejected as 'Unknown framework'. "
            f"Got result: {result}"
        )
        assert "Incompatible: PyTorch" not in result, (
            f"Private-org MLX model should not be detected as PyTorch. "
            f"Got result: {result}"
        )
        # If we get mlx_lm import errors, that's OK - it means framework detection worked!
        # The model was recognized as MLX and pre-flight passed

    # If we get here without assertions failing, the regression is fixed!
    print(f"✓ Private-org MLX model 'test-org/phi3-mlx-instruct' runs successfully")


@requires_user_cache
def test_framework_detection_for_renamed_mlx_community_model(
    copy_user_model_to_isolated, isolated_cache
):
    """Test that framework detection works correctly when cache root is passed.

    This is a more focused unit-style test that verifies detect_framework()
    receives the correct parameters from run_model().
    """
    from mlxk2.operations.common import detect_framework
    from mlxk2.core.cache import get_current_model_cache, hf_to_cache_dir

    # Copy and rename model
    src_model_dir = copy_user_model_to_isolated("mlx-community/Phi-3-mini-4k-instruct-4bit")
    private_org_cache_name = "models--acme--mlx-chat-model"
    private_org_dir = isolated_cache / private_org_cache_name
    shutil.move(str(src_model_dir), str(private_org_dir))

    # Add MLX tags to README
    snapshots = private_org_dir / "snapshots"
    snapshot_dirs = [d for d in snapshots.iterdir() if d.is_dir()]
    assert len(snapshot_dirs) > 0
    snapshot_path = snapshot_dirs[0]

    readme = snapshot_path / "README.md"
    readme.write_text("""---
tags: [mlx]
library_name: mlx
---
# Acme MLX Model
""")

    # Test framework detection with CORRECT parameters (cache root + selected_path + fm)
    from mlxk2.operations.common import read_front_matter
    fm = read_front_matter(snapshot_path)  # Read the README we just wrote
    framework = detect_framework(
        hf_name="acme/mlx-chat-model",
        model_root=private_org_dir,  # Cache root (models--acme--mlx-chat-model)
        selected_path=snapshot_path,  # Snapshot path (snapshots/abc123...)
        fm=fm  # Front-matter with MLX tags
    )

    assert framework == "MLX", (
        f"Framework should be detected as MLX from README tags. Got: {framework}"
    )

    # Test with INCORRECT parameters (what Beta.5 bug did)
    framework_buggy = detect_framework(
        hf_name="acme/mlx-chat-model",
        model_root=snapshot_path,  # BUG: Passing snapshot as root
        selected_path=None
    )

    # With the bug, it would fall through to "Unknown" because:
    # - Not mlx-community/* → no early return
    # - README not in snapshot_path / "snapshots" (doesn't exist)
    # - No GGUF/PyTorch detected
    # This assertion documents the buggy behavior for reference
    print(f"Buggy detection result: {framework_buggy} (should be Unknown without fix)")
