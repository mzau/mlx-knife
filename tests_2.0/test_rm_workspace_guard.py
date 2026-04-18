"""ADR-022 Workspace Guard for `mlxk rm`.

`mlxk rm` operates on the HF cache. Workspace directories are user-managed
and must not be deleted through `mlxk rm` — the correct operation is
`rm -rf <workspace>`. The guard in `mlxk2/operations/rm.py` detects
workspace paths via `is_workspace_path(resolved_name)` and returns a typed
error (`workspace_model`) with an instructive message.

These tests lock the guard against regression:
- The guard fires on an absolute workspace path (config.json present).
- The workspace directory is not modified by the attempted rm.
- Non-workspace inputs still receive their normal resolution path.
"""

import json
from pathlib import Path

from mlxk2.operations.rm import rm_operation


def _write_workspace(path: Path) -> None:
    """Create a minimal workspace marker (config.json is the is_workspace_path signal)."""
    path.mkdir(parents=True, exist_ok=True)
    (path / "config.json").write_text(json.dumps({"model_type": "llama"}))
    # A realistic workspace has more files; the guard only needs config.json.


class TestRmWorkspaceGuard:
    def test_workspace_absolute_path_is_rejected(self, tmp_path):
        """Absolute path pointing at a workspace -> typed error, no deletion."""
        workspace = tmp_path / "my-workspace-model"
        _write_workspace(workspace)
        assert workspace.exists(), "precondition: workspace created"

        result = rm_operation(str(workspace))

        assert result["status"] == "error"
        assert result["error"]["type"] == "workspace_model"
        assert "rm -rf" in result["error"]["message"]
        assert str(workspace) in result["error"]["message"]
        # Guard must not delete — workspace directory remains intact.
        assert workspace.exists()
        assert (workspace / "config.json").exists()

    def test_workspace_guard_does_not_shadow_model_not_found(self, tmp_path):
        """A plausibly-shaped HF name that does not exist must still surface
        model_not_found, not workspace_model — the guard is path-only."""
        result = rm_operation("mlx-community/this-model-does-not-exist-xyz-42")
        assert result["status"] == "error"
        assert result["error"]["type"] == "model_not_found"
