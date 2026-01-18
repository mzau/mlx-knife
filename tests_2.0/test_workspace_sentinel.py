"""Tests for workspace sentinel primitives (ADR-018 Phase 0a).

Tests workspace infrastructure:
- Sentinel write with atomic rename
- Managed workspace detection
- Health check integration with workspaces
- Backward compatibility with unmanaged workspaces
"""

import json
from pathlib import Path

import pytest

from mlxk2 import __version__
from mlxk2.operations.workspace import (
    write_workspace_sentinel,
    is_managed_workspace,
    is_workspace_path,
    is_explicit_path,
    find_matching_workspaces,
    read_workspace_metadata,
    SENTINEL_FILENAME
)
from mlxk2.operations.health import health_check_workspace


class TestIsWorkspacePath:
    """Test is_workspace_path() helper function (ADR-018 Phase 0c)."""

    def test_is_workspace_path_valid_workspace(self, tmp_path):
        """Test detects valid workspace with config.json."""
        (tmp_path / "config.json").write_text('{"model_type": "llama"}')
        assert is_workspace_path(tmp_path) is True
        assert is_workspace_path(str(tmp_path)) is True  # String path

    def test_is_workspace_path_no_config(self, tmp_path):
        """Test returns False if config.json missing."""
        (tmp_path / "other.json").write_text('{}')
        assert is_workspace_path(tmp_path) is False

    def test_is_workspace_path_nonexistent(self, tmp_path):
        """Test returns False for nonexistent path."""
        assert is_workspace_path(tmp_path / "nonexistent") is False

    def test_is_workspace_path_hf_model_id(self, tmp_path):
        """Test returns False for HF model IDs (not paths)."""
        assert is_workspace_path("mlx-community/Phi-3-mini") is False
        assert is_workspace_path("microsoft/phi-2") is False

    def test_is_workspace_path_file_not_directory(self, tmp_path):
        """Test returns False if path is a file, not directory."""
        config_file = tmp_path / "config.json"
        config_file.write_text('{}')
        assert is_workspace_path(config_file) is False

    def test_is_workspace_path_invalid_input(self):
        """Test handles invalid input gracefully."""
        assert is_workspace_path(None) is False
        assert is_workspace_path(123) is False
        assert is_workspace_path([]) is False


class TestIsExplicitPath:
    """Test is_explicit_path() helper function."""

    def test_explicit_path_relative_dot_slash(self):
        """Test ./ prefix is explicit path."""
        assert is_explicit_path("./model") is True
        assert is_explicit_path("./gemma-3n") is True
        assert is_explicit_path("./") is True

    def test_explicit_path_relative_dot_dot_slash(self):
        """Test ../ prefix is explicit path."""
        assert is_explicit_path("../model") is True
        assert is_explicit_path("../parent/model") is True

    def test_explicit_path_absolute(self):
        """Test / prefix is explicit path."""
        assert is_explicit_path("/abs/path/model") is True
        assert is_explicit_path("/model") is True

    def test_explicit_path_dot_only(self):
        """Test . and .. alone are explicit paths."""
        assert is_explicit_path(".") is True
        assert is_explicit_path("..") is True

    def test_not_explicit_path_hf_model_id(self):
        """Test HF model IDs are NOT explicit paths."""
        assert is_explicit_path("mlx-community/Phi-3") is False
        assert is_explicit_path("microsoft/phi-2") is False

    def test_not_explicit_path_bare_name(self):
        """Test bare names without path prefix are NOT explicit paths."""
        assert is_explicit_path("my-model") is False
        assert is_explicit_path("gemma-3n-E2B") is False

    def test_not_explicit_path_invalid_input(self):
        """Test handles invalid input gracefully."""
        assert is_explicit_path(None) is False
        assert is_explicit_path("") is False
        assert is_explicit_path(123) is False


class TestFindMatchingWorkspaces:
    """Test find_matching_workspaces() prefix matching."""

    def test_find_exact_match(self, tmp_path):
        """Test exact match returns single workspace."""
        ws = tmp_path / "my-model"
        ws.mkdir()
        (ws / "config.json").write_text('{"model_type": "llama"}')

        # Use ./ prefix to make it explicit path
        matches = find_matching_workspaces(f"./{ws.name}")
        # Must be in correct directory for ./ to work
        import os
        old_cwd = os.getcwd()
        try:
            os.chdir(tmp_path)
            matches = find_matching_workspaces("./my-model")
            assert len(matches) == 1
            assert matches[0].name == "my-model"
        finally:
            os.chdir(old_cwd)

    def test_find_prefix_match(self, tmp_path):
        """Test prefix match returns multiple workspaces."""
        # Create multiple workspaces with common prefix
        for name in ["gemma-3n-4bit", "gemma-3n-FIXED-4bit", "gemma-3n-8bit"]:
            ws = tmp_path / name
            ws.mkdir()
            (ws / "config.json").write_text('{"model_type": "gemma"}')

        # Create non-matching workspace
        other = tmp_path / "llama-3"
        other.mkdir()
        (other / "config.json").write_text('{"model_type": "llama"}')

        import os
        old_cwd = os.getcwd()
        try:
            os.chdir(tmp_path)
            matches = find_matching_workspaces("./gemma-")
            assert len(matches) == 3
            assert all("gemma" in m.name for m in matches)
            # Should NOT include llama-3
            assert not any("llama" in m.name for m in matches)
        finally:
            os.chdir(old_cwd)

    def test_find_prefix_match_sorted(self, tmp_path):
        """Test prefix match returns sorted results."""
        for name in ["model-c", "model-a", "model-b"]:
            ws = tmp_path / name
            ws.mkdir()
            (ws / "config.json").write_text('{}')

        import os
        old_cwd = os.getcwd()
        try:
            os.chdir(tmp_path)
            matches = find_matching_workspaces("./model-")
            names = [m.name for m in matches]
            assert names == ["model-a", "model-b", "model-c"]
        finally:
            os.chdir(old_cwd)

    def test_find_no_match(self, tmp_path):
        """Test returns empty list when no workspaces match."""
        import os
        old_cwd = os.getcwd()
        try:
            os.chdir(tmp_path)
            matches = find_matching_workspaces("./nonexistent-")
            assert matches == []
        finally:
            os.chdir(old_cwd)

    def test_find_skips_non_workspaces(self, tmp_path):
        """Test skips directories without config.json."""
        # Valid workspace
        valid = tmp_path / "gemma-valid"
        valid.mkdir()
        (valid / "config.json").write_text('{}')

        # Directory without config.json (not a workspace)
        invalid = tmp_path / "gemma-invalid"
        invalid.mkdir()
        (invalid / "other.txt").write_text("not a workspace")

        import os
        old_cwd = os.getcwd()
        try:
            os.chdir(tmp_path)
            matches = find_matching_workspaces("./gemma-")
            assert len(matches) == 1
            assert matches[0].name == "gemma-valid"
        finally:
            os.chdir(old_cwd)

    def test_find_absolute_path(self, tmp_path):
        """Test works with absolute paths."""
        ws = tmp_path / "model"
        ws.mkdir()
        (ws / "config.json").write_text('{}')

        matches = find_matching_workspaces(str(ws))
        assert len(matches) == 1

    def test_find_not_explicit_path(self):
        """Test returns empty list for non-explicit paths."""
        # HF model ID is not explicit path
        matches = find_matching_workspaces("mlx-community/model")
        assert matches == []

        # Bare name is not explicit path
        matches = find_matching_workspaces("my-model")
        assert matches == []

    def test_find_directory_scan(self, tmp_path):
        """Test directory scan (existing directory, not workspace) finds all workspaces inside."""
        import os

        # Create multiple workspaces in tmp_path
        for name in ["model-a", "model-b", "model-c"]:
            ws = tmp_path / name
            ws.mkdir()
            (ws / "config.json").write_text('{}')

        # Create a non-workspace directory
        other = tmp_path / "not-a-workspace"
        other.mkdir()

        old_cwd = os.getcwd()
        try:
            os.chdir(tmp_path)
            # Pattern "." should find all 3 workspaces
            matches = find_matching_workspaces(".")
            assert len(matches) == 3
            names = [m.name for m in matches]
            assert "model-a" in names
            assert "model-b" in names
            assert "model-c" in names
            # Should NOT include non-workspace directory
            assert "not-a-workspace" not in names
        finally:
            os.chdir(old_cwd)

    def test_find_directory_scan_absolute(self, tmp_path):
        """Test directory scan with absolute path."""
        for name in ["ws1", "ws2"]:
            ws = tmp_path / name
            ws.mkdir()
            (ws / "config.json").write_text('{}')

        # Use absolute path to directory
        matches = find_matching_workspaces(str(tmp_path))
        assert len(matches) == 2


class TestWorkspaceSentinel:
    """Test workspace sentinel write/read primitives."""

    def test_write_sentinel_atomic(self, tmp_path):
        """Test atomic write with rename."""
        metadata = {
            "mlxk_version": __version__,
            "created_at": "2025-12-29T10:30:00Z",
            "source_repo": "mlx-community/Llama-3.2-3B",
            "source_revision": "abc123",
            "managed": True,
            "operation": "clone"
        }

        write_workspace_sentinel(tmp_path, metadata)

        sentinel = tmp_path / SENTINEL_FILENAME
        assert sentinel.exists(), "Sentinel file should exist"

        # Verify JSON is valid and matches input
        data = json.loads(sentinel.read_text())
        assert data["managed"] is True
        assert data["mlxk_version"] == __version__
        assert data["operation"] == "clone"
        assert data["source_repo"] == "mlx-community/Llama-3.2-3B"

    def test_write_sentinel_creates_valid_json(self, tmp_path):
        """Test sentinel is well-formed JSON."""
        metadata = {
            "mlxk_version": __version__,
            "created_at": "2025-12-29T10:30:00Z",
            "managed": True,
            "operation": "convert"
        }

        write_workspace_sentinel(tmp_path, metadata)

        sentinel = tmp_path / SENTINEL_FILENAME
        content = sentinel.read_text()

        # Should be valid JSON
        data = json.loads(content)
        assert isinstance(data, dict)

        # Should have trailing newline (conventional)
        assert content.endswith("\n")

    def test_write_sentinel_requires_path_object(self, tmp_path):
        """Test workspace_path must be Path object."""
        metadata = {"mlxk_version": "2.0.4", "created_at": "2025-12-29T10:30:00Z",
                   "managed": True, "operation": "clone"}

        with pytest.raises(TypeError, match="must be Path"):
            write_workspace_sentinel(str(tmp_path), metadata)

    def test_write_sentinel_validates_required_fields(self, tmp_path):
        """Test metadata must have required fields."""
        # Missing 'mlxk_version'
        incomplete_metadata = {
            "created_at": "2025-12-29T10:30:00Z",
            "managed": True,
            "operation": "clone"
        }

        with pytest.raises(ValueError, match="Missing required metadata fields"):
            write_workspace_sentinel(tmp_path, incomplete_metadata)

    def test_write_sentinel_allows_additional_fields(self, tmp_path):
        """Test forward compatibility: extra fields allowed."""
        metadata = {
            "mlxk_version": __version__,
            "created_at": "2025-12-29T10:30:00Z",
            "managed": True,
            "operation": "clone",
            "custom_field": "future_feature",  # Extra field
            "another_field": 42
        }

        write_workspace_sentinel(tmp_path, metadata)

        data = json.loads((tmp_path / SENTINEL_FILENAME).read_text())
        assert data["custom_field"] == "future_feature"
        assert data["another_field"] == 42


class TestManagedWorkspaceDetection:
    """Test managed vs unmanaged workspace detection."""

    def test_is_managed_workspace_true(self, tmp_path):
        """Test managed workspace detection (has valid sentinel)."""
        metadata = {
            "mlxk_version": __version__,
            "created_at": "2025-12-29T10:30:00Z",
            "managed": True,
            "operation": "clone"
        }
        write_workspace_sentinel(tmp_path, metadata)

        assert is_managed_workspace(tmp_path) is True

    def test_is_managed_workspace_false_no_sentinel(self, tmp_path):
        """Test unmanaged workspace (no sentinel)."""
        # Empty directory, no sentinel
        assert is_managed_workspace(tmp_path) is False

    def test_is_managed_workspace_false_managed_field_false(self, tmp_path):
        """Test sentinel exists but managed=False."""
        metadata = {
            "mlxk_version": __version__,
            "created_at": "2025-12-29T10:30:00Z",
            "managed": False,  # Explicitly unmanaged
            "operation": "manual"
        }
        write_workspace_sentinel(tmp_path, metadata)

        assert is_managed_workspace(tmp_path) is False

    def test_is_managed_workspace_false_invalid_json(self, tmp_path):
        """Test corrupted sentinel (invalid JSON)."""
        sentinel = tmp_path / SENTINEL_FILENAME
        sentinel.write_text("{invalid json")

        assert is_managed_workspace(tmp_path) is False

    def test_is_managed_workspace_handles_non_path_input(self):
        """Test graceful handling of non-Path input."""
        assert is_managed_workspace("not_a_path") is False
        assert is_managed_workspace(None) is False

    def test_read_workspace_metadata_valid(self, tmp_path):
        """Test reading sentinel metadata."""
        metadata = {
            "mlxk_version": __version__,
            "created_at": "2025-12-29T10:30:00Z",
            "source_repo": "mlx-community/Model",
            "managed": True,
            "operation": "clone"
        }
        write_workspace_sentinel(tmp_path, metadata)

        read_data = read_workspace_metadata(tmp_path)
        assert read_data["mlxk_version"] == __version__
        assert read_data["source_repo"] == "mlx-community/Model"

    def test_read_workspace_metadata_no_sentinel(self, tmp_path):
        """Test reading returns empty dict if no sentinel."""
        read_data = read_workspace_metadata(tmp_path)
        assert read_data == {}

    def test_read_workspace_metadata_invalid_json(self, tmp_path):
        """Test reading returns empty dict if JSON invalid."""
        sentinel = tmp_path / SENTINEL_FILENAME
        sentinel.write_text("not json")

        read_data = read_workspace_metadata(tmp_path)
        assert read_data == {}


class TestWorkspaceHealthCheck:
    """Test health check integration with managed/unmanaged workspaces."""

    def test_health_check_workspace_managed(self, tmp_path):
        """Test health check on managed workspace."""
        # Create minimal valid workspace
        (tmp_path / "config.json").write_text('{"model_type": "llama"}')
        metadata = {
            "mlxk_version": __version__,
            "created_at": "2025-12-29T10:30:00Z",
            "managed": True,
            "operation": "clone"
        }
        write_workspace_sentinel(tmp_path, metadata)

        healthy, reason, managed = health_check_workspace(tmp_path)
        assert managed is True, "Should detect managed workspace"
        # Note: May fail health due to missing weights, but managed flag works

    def test_health_check_workspace_unmanaged(self, tmp_path):
        """Test health check on unmanaged workspace."""
        # Create minimal workspace without sentinel
        (tmp_path / "config.json").write_text('{"model_type": "llama"}')

        healthy, reason, managed = health_check_workspace(tmp_path)
        assert managed is False, "Should detect unmanaged workspace"
        # Health check still runs, just flagged as unmanaged

    def test_health_check_workspace_no_config(self, tmp_path):
        """Test workspace without config.json is unhealthy."""
        # Empty directory
        metadata = {
            "mlxk_version": __version__,
            "created_at": "2025-12-29T10:30:00Z",
            "managed": True,
            "operation": "clone"
        }
        write_workspace_sentinel(tmp_path, metadata)

        healthy, reason, managed = health_check_workspace(tmp_path)
        assert healthy is False
        assert "No config.json" in reason
        assert managed is True  # Still recognized as managed

    def test_health_check_workspace_returns_three_values(self, tmp_path):
        """Test health_check_workspace returns (bool, str, bool)."""
        (tmp_path / "config.json").write_text('{"model_type": "llama"}')

        result = health_check_workspace(tmp_path)
        assert isinstance(result, tuple)
        assert len(result) == 3
        healthy, reason, managed = result
        assert isinstance(healthy, bool)
        assert isinstance(reason, str)
        assert isinstance(managed, bool)


class TestHealthCheckOperationWorkspaceIntegration:
    """Test health_check_operation CLI integration with workspace paths."""

    def test_health_check_operation_detects_workspace_path(self, tmp_path):
        """Test health_check_operation recognizes workspace paths."""
        from mlxk2.operations.health import health_check_operation

        # Create minimal workspace
        (tmp_path / "config.json").write_text('{"model_type": "llama"}')
        (tmp_path / "model.safetensors").write_text("fake weights")

        metadata = {
            "mlxk_version": __version__,
            "created_at": "2025-12-29T10:30:00Z",
            "managed": True,
            "operation": "clone"
        }
        write_workspace_sentinel(tmp_path, metadata)

        # Call health_check_operation with workspace path
        result = health_check_operation(str(tmp_path))

        assert result["status"] == "success"
        assert result["data"]["summary"]["total"] == 1

        # Should have checked the workspace (not cache)
        assert len(result["data"]["healthy"]) + len(result["data"]["unhealthy"]) == 1

        # First result should include 'managed' flag
        model_info = (result["data"]["healthy"] + result["data"]["unhealthy"])[0]
        assert "managed" in model_info
        assert model_info["managed"] is True
        assert model_info["name"] == str(tmp_path)

    def test_health_check_operation_workspace_unmanaged(self, tmp_path):
        """Test health_check_operation with unmanaged workspace."""
        from mlxk2.operations.health import health_check_operation

        # Unmanaged workspace (no sentinel)
        (tmp_path / "config.json").write_text('{"model_type": "llama"}')
        (tmp_path / "model.safetensors").write_text("fake weights")

        result = health_check_operation(str(tmp_path))

        assert result["status"] == "success"
        model_info = (result["data"]["healthy"] + result["data"]["unhealthy"])[0]
        assert model_info["managed"] is False

    def test_health_check_operation_workspace_vs_cache(self, tmp_path):
        """Test workspace path takes precedence over cache name resolution."""
        from mlxk2.operations.health import health_check_operation

        # Create workspace with same name as potential cache model
        ws_name = "mlx-community-model"
        workspace = tmp_path / ws_name
        workspace.mkdir()
        (workspace / "config.json").write_text('{"model_type": "llama"}')
        (workspace / "model.safetensors").write_text("fake weights")

        # Call with workspace path
        result = health_check_operation(str(workspace))

        # Should detect as workspace (has 'managed' field), not cache model
        assert result["status"] == "success"
        model_info = (result["data"]["healthy"] + result["data"]["unhealthy"])[0]
        assert "managed" in model_info, "Should include 'managed' field (workspace detected)"
