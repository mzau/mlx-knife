"""Tests for JSON API spec v0.1.2: list operation minimal model object.

Covers: size_bytes, last_modified (ISO-8601 Z), framework, model_type,
capabilities, health, hash selection, cached.
"""

from datetime import datetime
from typing import Set
import pytest

from mlxk2.operations.list import list_models


def _is_iso_utc_z(ts: str) -> bool:
    try:
        # Must end with 'Z' and be parseable
        if not ts.endswith("Z"):
            return False
        # Strip Z, attempt parsing
        datetime.fromisoformat(ts.replace("Z", ""))
        return True
    except Exception:
        return False


@pytest.mark.spec
def test_list_minimal_model_object_fields(mock_models, isolated_cache):
    """Each model entry returns the minimal model object with health."""
    result = list_models()
    assert result["status"] == "success"
    assert result["command"] == "list"

    models = result["data"]["models"]
    assert isinstance(models, list)
    assert result["data"]["count"] == len(models)

    # Allowed enums
    allowed_framework: Set[str] = {"MLX", "GGUF", "PyTorch", "Unknown"}
    allowed_model_types: Set[str] = {"chat", "embedding", "base", "unknown"}

    # Verify minimal fields and types
    for m in models:
        # Required fields
        assert set([
            "name", "hash", "size_bytes", "last_modified", "framework",
            "model_type", "capabilities", "health", "cached"
        ]).issubset(m.keys())

        assert isinstance(m["name"], str) and "/" in m["name"]

        # hash: 40-char or None
        h = m["hash"]
        assert (h is None) or (isinstance(h, str) and len(h) == 40)

        # size_bytes integer >= 0
        assert isinstance(m["size_bytes"], int)
        assert m["size_bytes"] >= 0

        # last_modified as ISO-8601 UTC Z
        assert isinstance(m["last_modified"], str)
        assert _is_iso_utc_z(m["last_modified"]) is True

        # framework
        assert m["framework"] in allowed_framework

        # model_type + capabilities
        assert m["model_type"] in allowed_model_types
        assert isinstance(m["capabilities"], list)

        # health
        assert m["health"] in {"healthy", "unhealthy"}

        # cached flag
        assert m["cached"] is True

        # Spec 0.1.2: no human-readable size; ensure we do not expose 'size' or internal paths
        assert "size" not in m
        assert "hashes" not in m


@pytest.mark.spec
def test_list_pattern_filter_case_insensitive(mock_models, isolated_cache):
    """Pattern filters case-insensitively on model name."""
    result = list_models(pattern="llama")
    models = result["data"]["models"]
    assert all("llama" in m["name"].lower() for m in models)

    # A different pattern should yield different subset
    result_q = list_models(pattern="Qwen")
    models_q = result_q["data"]["models"]
    assert all("qwen" in m["name"].lower() for m in models_q)
    # Ensure partition is non-trivial in our fixture
    assert set(m["name"].lower() for m in models).isdisjoint(
        set(m["name"].lower() for m in models_q)
    ) is True


@pytest.mark.spec
def test_list_empty_cache(isolated_cache):
    """Empty cache yields empty list and count 0."""
    # Remove all models (keep canary)
    for d in isolated_cache.iterdir():
        if d.is_dir() and d.name.startswith("models--") and "TEST-CACHE-SENTINEL" not in d.name:
            # Safe in tests; strict delete is enforced by fixture env var
            from shutil import rmtree
            rmtree(d)

    result = list_models()
    assert result["status"] == "success"
    assert result["data"]["models"] == []
    assert result["data"]["count"] == 0


class TestListWorkspacePrefix:
    """Test list_models() with workspace path patterns (Session 103)."""

    def test_list_workspace_exact_match(self, tmp_path):
        """Test list_models with exact workspace path."""
        import os

        ws = tmp_path / "my-model"
        ws.mkdir()
        (ws / "config.json").write_text('{"model_type": "llama"}')

        old_cwd = os.getcwd()
        try:
            os.chdir(tmp_path)
            result = list_models(pattern="./my-model")

            assert result["status"] == "success"
            assert result["data"]["count"] == 1
            assert len(result["data"]["models"]) == 1
            assert "my-model" in result["data"]["models"][0]["name"]
        finally:
            os.chdir(old_cwd)

    def test_list_workspace_prefix_match(self, tmp_path):
        """Test list_models with workspace prefix pattern."""
        import os

        # Create multiple workspaces with common prefix
        for name in ["gemma-3n-4bit", "gemma-3n-FIXED-4bit", "gemma-3n-8bit"]:
            ws = tmp_path / name
            ws.mkdir()
            (ws / "config.json").write_text('{"model_type": "gemma"}')

        # Create non-matching workspace
        other = tmp_path / "llama-3"
        other.mkdir()
        (other / "config.json").write_text('{"model_type": "llama"}')

        old_cwd = os.getcwd()
        try:
            os.chdir(tmp_path)
            result = list_models(pattern="./gemma-")

            assert result["status"] == "success"
            assert result["data"]["count"] == 3
            # All matches should contain gemma
            for m in result["data"]["models"]:
                assert "gemma" in m["name"]
        finally:
            os.chdir(old_cwd)

    def test_list_workspace_prefix_no_match(self, tmp_path):
        """Test list_models with non-matching prefix returns empty list."""
        import os

        old_cwd = os.getcwd()
        try:
            os.chdir(tmp_path)
            result = list_models(pattern="./nonexistent-")

            assert result["status"] == "success"
            assert result["data"]["count"] == 0
            assert result["data"]["models"] == []
        finally:
            os.chdir(old_cwd)

    def test_list_workspace_does_not_fall_through_to_cache(self, tmp_path, isolated_cache):
        """Test explicit path patterns don't fall through to cache search."""
        import os

        old_cwd = os.getcwd()
        try:
            os.chdir(tmp_path)
            # ./gemma- should NOT match cache models even if they contain "gemma"
            # It should only search local workspaces
            result = list_models(pattern="./gemma-")

            assert result["status"] == "success"
            # Should be empty (no local workspaces) not cache models
            assert result["data"]["count"] == 0
        finally:
            os.chdir(old_cwd)

    def test_list_workspace_sorted_by_name(self, tmp_path):
        """Test workspace results are sorted by name."""
        import os

        for name in ["model-c", "model-a", "model-b"]:
            ws = tmp_path / name
            ws.mkdir()
            (ws / "config.json").write_text('{}')

        old_cwd = os.getcwd()
        try:
            os.chdir(tmp_path)
            result = list_models(pattern="./model-")

            names = [m["name"] for m in result["data"]["models"]]
            # Should be sorted (names contain full path, but sorted by name)
            assert "model-a" in names[0]
            assert "model-b" in names[1]
            assert "model-c" in names[2]
        finally:
            os.chdir(old_cwd)

    def test_list_workspace_absolute_path(self, tmp_path):
        """Test list_models with absolute workspace path."""
        ws = tmp_path / "model"
        ws.mkdir()
        (ws / "config.json").write_text('{}')

        result = list_models(pattern=str(ws))

        assert result["status"] == "success"
        assert result["data"]["count"] == 1

    def test_list_workspace_has_all_fields(self, tmp_path):
        """Test workspace model objects have all required fields."""
        import os

        ws = tmp_path / "my-model"
        ws.mkdir()
        (ws / "config.json").write_text('{"model_type": "llama"}')

        old_cwd = os.getcwd()
        try:
            os.chdir(tmp_path)
            result = list_models(pattern="./my-model")

            m = result["data"]["models"][0]
            # Check required fields exist
            required = ["name", "hash", "size_bytes", "last_modified",
                       "framework", "model_type", "capabilities", "health", "cached"]
            for field in required:
                assert field in m, f"Missing field: {field}"

            # Workspace-specific: hash should be None, cached should be False
            assert m["hash"] is None
            assert m["cached"] is False
        finally:
            os.chdir(old_cwd)

    def test_list_workspace_display_name_relative(self, tmp_path):
        """Test display_name is relative for relative input patterns."""
        import os

        ws = tmp_path / "my-model"
        ws.mkdir()
        (ws / "config.json").write_text('{}')

        old_cwd = os.getcwd()
        try:
            os.chdir(tmp_path)
            result = list_models(pattern="./my-model")

            m = result["data"]["models"][0]
            # name should be absolute (for programmatic use)
            assert m["name"].startswith("/")
            # display_name should be relative
            assert m["display_name"] == "my-model"
        finally:
            os.chdir(old_cwd)

    def test_list_workspace_display_name_absolute(self, tmp_path):
        """Test display_name is absolute for absolute input patterns."""
        ws = tmp_path / "my-model"
        ws.mkdir()
        (ws / "config.json").write_text('{}')

        # Use absolute path pattern
        result = list_models(pattern=str(ws))

        m = result["data"]["models"][0]
        # Both name and display_name should be absolute
        assert m["name"].startswith("/")
        assert m["display_name"].startswith("/")
        assert m["display_name"] == str(ws)

    def test_list_workspace_display_name_prefix_match(self, tmp_path):
        """Test display_name works correctly with prefix matching."""
        import os

        for name in ["gemma-a", "gemma-b"]:
            ws = tmp_path / name
            ws.mkdir()
            (ws / "config.json").write_text('{}')

        old_cwd = os.getcwd()
        try:
            os.chdir(tmp_path)
            result = list_models(pattern="./gemma-")

            # Both should have relative display_name
            display_names = [m["display_name"] for m in result["data"]["models"]]
            assert "gemma-a" in display_names
            assert "gemma-b" in display_names
            # None should be absolute
            assert not any(dn.startswith("/") for dn in display_names)
        finally:
            os.chdir(old_cwd)

    def test_list_workspace_directory_scan(self, tmp_path):
        """Test listing all workspaces in a directory (. pattern)."""
        import os

        # Create multiple workspaces
        for name in ["model-a", "model-b"]:
            ws = tmp_path / name
            ws.mkdir()
            (ws / "config.json").write_text('{}')

        old_cwd = os.getcwd()
        try:
            os.chdir(tmp_path)
            result = list_models(pattern=".")

            assert result["status"] == "success"
            assert result["data"]["count"] == 2
            # display_names should be just the directory names
            display_names = [m["display_name"] for m in result["data"]["models"]]
            assert "model-a" in display_names
            assert "model-b" in display_names
        finally:
            os.chdir(old_cwd)

    def test_list_workspace_directory_scan_absolute(self, tmp_path):
        """Test listing all workspaces with absolute directory path."""
        # Create workspaces
        for name in ["model-a", "model-b"]:
            ws = tmp_path / name
            ws.mkdir()
            (ws / "config.json").write_text('{}')

        # Use absolute path
        result = list_models(pattern=str(tmp_path))

        assert result["status"] == "success"
        assert result["data"]["count"] == 2
        # display_names should be absolute (because input was absolute)
        for m in result["data"]["models"]:
            assert m["display_name"].startswith("/")
            # name should also be absolute
            assert m["name"].startswith("/")
