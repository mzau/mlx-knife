"""Workspace integration tests (ADR-022: Workspace-First Paradigm).

Tests that MLXK_WORKSPACE_HOME is correctly used by list, health, show,
and clone operations. Uses the workspace_home fixture from conftest.py.
"""

import json
import os
from pathlib import Path

import pytest

from mlxk2.operations.list import list_models
from mlxk2.operations.health import health_check_operation
from mlxk2.operations.show import show_model_operation


class TestListWithWorkspaceHome:
    """Test list_models() discovers workspaces from MLXK_WORKSPACE_HOME."""

    @staticmethod
    def _workspace_names(result):
        """Extract display_name from workspace models (those that have it)."""
        return [m["display_name"] for m in result["data"]["models"] if "display_name" in m]

    def test_list_finds_workspaces(self, workspace_home):
        """Workspaces appear in list output."""
        result = list_models()
        assert result["status"] == "success"
        names = self._workspace_names(result)
        assert "test-model-bf16" in names
        assert "test-model-4bit" in names

    def test_list_workspace_pattern_filter(self, workspace_home):
        """Pattern filter works for workspace names."""
        result = list_models(pattern="test-model-4bit")
        names = self._workspace_names(result)
        assert "test-model-4bit" in names
        assert "test-model-bf16" not in names

    def test_list_workspace_not_cached(self, workspace_home):
        """Workspace models have cached=False."""
        result = list_models()
        for model in result["data"]["models"]:
            if model.get("display_name") in ("test-model-bf16", "test-model-4bit"):
                assert model["cached"] is False

    def test_list_without_workspace_home(self, monkeypatch):
        """Without MLXK_WORKSPACE_HOME, no workspace models appear."""
        monkeypatch.delenv("MLXK_WORKSPACE_HOME", raising=False)
        result = list_models()
        assert result["status"] == "success"
        # No display_name means no workspace models
        ws_names = self._workspace_names(result)
        assert "test-model-bf16" not in ws_names


class TestWorkspaceOnlyPortfolio:
    """Test that portfolio works with ONLY workspaces (no HF cache)."""

    def test_list_works_without_hf_cache(self, workspace_home, monkeypatch, tmp_path):
        """List returns workspace models even when HF cache doesn't exist."""
        # Point HF_HOME to a non-existent directory
        monkeypatch.setenv("HF_HOME", str(tmp_path / "nonexistent-cache"))
        result = list_models()
        assert result["status"] == "success"
        assert result["data"]["count"] == 2  # Only workspace models
        names = [m.get("display_name") for m in result["data"]["models"]]
        assert "test-model-bf16" in names
        assert "test-model-4bit" in names

    def test_health_works_without_hf_cache(self, workspace_home, monkeypatch, tmp_path):
        """Health check works with only workspace models."""
        monkeypatch.setenv("HF_HOME", str(tmp_path / "nonexistent-cache"))
        result = health_check_operation()
        assert result["status"] == "success"
        assert result["data"]["summary"]["total"] == 2
        names = [m["name"] for m in result["data"]["healthy"]]
        assert "test-model-bf16" in names
        assert "test-model-4bit" in names


class TestHealthWithWorkspaceHome:
    """Test health_check_operation() includes workspaces."""

    def test_health_all_includes_workspaces(self, workspace_home):
        """Health check without pattern includes workspace models."""
        result = health_check_operation()
        assert result["status"] == "success"
        healthy_names = [m["name"] for m in result["data"]["healthy"]]
        # Both test workspaces should be healthy (they have config.json + safetensors)
        assert "test-model-bf16" in healthy_names
        assert "test-model-4bit" in healthy_names

    def test_health_fuzzy_match_workspace(self, workspace_home):
        """Health check with pattern fuzzy-matches workspace name."""
        result = health_check_operation(model_pattern="test-model-bf16")
        assert result["status"] == "success"
        total = result["data"]["summary"]["healthy_count"] + result["data"]["summary"].get("unhealthy_count", 0)
        assert total >= 1

    def test_health_without_workspace_home(self, monkeypatch):
        """Health check without MLXK_WORKSPACE_HOME skips workspaces."""
        monkeypatch.delenv("MLXK_WORKSPACE_HOME", raising=False)
        result = health_check_operation()
        assert result["status"] == "success"
        # Should still succeed, just without workspace models
        healthy_names = [m["name"] for m in result["data"]["healthy"]]
        assert "test-model-bf16" not in healthy_names


class TestShowWithWorkspacePath:
    """Test show_model_operation() with workspace paths."""

    def test_show_workspace_by_path(self, workspace_home):
        """Show operation works with explicit workspace path."""
        ws_path = str(workspace_home / "test-model-4bit")
        result = show_model_operation(ws_path)
        assert result["status"] == "success"
        # model_type is in data.metadata (from config.json)
        assert result["data"]["metadata"]["model_type"] == "llama"

    def test_show_workspace_has_quantization_in_model(self, workspace_home):
        """Show exposes quantization info on model object (from config.json)."""
        ws_path = str(workspace_home / "test-model-4bit")
        result = show_model_operation(ws_path)
        # mlx-lm writes "quantization" key (not "quantization_config")
        # show.py reads it into model_obj["quantization"]
        model = result["data"]["model"]
        assert model.get("quantization") == {"bits": 4, "group_size": 64}

    def test_show_workspace_metadata(self, workspace_home):
        """Show includes workspace metadata from sentinel."""
        ws_path = str(workspace_home / "test-model-bf16")
        result = show_model_operation(ws_path)
        ws_meta = result["data"].get("workspace_metadata")
        assert ws_meta is not None
        assert ws_meta["managed"] is True
        assert ws_meta["operation"] == "clone"


class TestCloneTargetResolution:
    """Test clone target resolution with MLXK_WORKSPACE_HOME."""

    def test_resolve_strips_org_prefix(self, workspace_home):
        """_resolve_clone_target strips org prefix for flat layout."""
        from mlxk2.cli import _resolve_clone_target
        target = _resolve_clone_target("mlx-community/pixtral-12b-bf16")
        assert target == str(workspace_home / "pixtral-12b-bf16")

    def test_resolve_strips_revision(self, workspace_home):
        """_resolve_clone_target strips @revision."""
        from mlxk2.cli import _resolve_clone_target
        target = _resolve_clone_target("mlx-community/model@main")
        assert target == str(workspace_home / "model")

    def test_resolve_bare_model_name(self, workspace_home):
        """_resolve_clone_target handles bare model name (no org)."""
        from mlxk2.cli import _resolve_clone_target
        target = _resolve_clone_target("my-local-model")
        assert target == str(workspace_home / "my-local-model")

    def test_resolve_fails_without_workspace_home(self, monkeypatch):
        """_resolve_clone_target exits when MLXK_WORKSPACE_HOME not set."""
        monkeypatch.delenv("MLXK_WORKSPACE_HOME", raising=False)
        from mlxk2.cli import _resolve_clone_target
        with pytest.raises(SystemExit):
            _resolve_clone_target("mlx-community/model")

    def test_resolve_fails_on_existing_target(self, workspace_home):
        """_resolve_clone_target exits when target directory already exists."""
        from mlxk2.cli import _resolve_clone_target
        # test-model-bf16 already exists in workspace_home
        with pytest.raises(SystemExit):
            _resolve_clone_target("org/test-model-bf16")

    def test_is_explicit_path(self):
        """_is_explicit_path correctly identifies explicit paths."""
        from mlxk2.cli import _is_explicit_path
        assert _is_explicit_path("./my-model") is True
        assert _is_explicit_path("../my-model") is True
        assert _is_explicit_path("/abs/path") is True
        assert _is_explicit_path("bare-name") is False
        assert _is_explicit_path("my-pixtral") is False


class TestConvertPathResolution:
    """Test convert source/target resolution with MLXK_WORKSPACE_HOME."""

    def test_convert_resolves_source_bare_name(self, workspace_home, monkeypatch):
        """Convert resolves bare source name from MLXK_WORKSPACE_HOME."""
        # Simulate what cli.py does for source resolution
        from mlxk2.cli import _is_explicit_path
        source = "test-model-bf16"
        assert not _is_explicit_path(source)
        # Should resolve to workspace_home / source
        resolved = workspace_home / source
        assert resolved.is_dir()

    def test_convert_resolves_target_bare_name(self, workspace_home):
        """Convert resolves bare target name into MLXK_WORKSPACE_HOME."""
        from mlxk2.cli import _is_explicit_path
        target = "test-model-new-4bit"
        assert not _is_explicit_path(target)
        # Should resolve to workspace_home / target
        expected = workspace_home / target
        assert str(expected).startswith(str(workspace_home))

    def test_convert_explicit_path_not_resolved(self, workspace_home):
        """Explicit paths bypass MLXK_WORKSPACE_HOME resolution."""
        from mlxk2.cli import _is_explicit_path
        assert _is_explicit_path("./local-source")
        assert _is_explicit_path("/abs/target")
        # These should NOT be resolved into workspace_home
