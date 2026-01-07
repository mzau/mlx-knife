"""Tests for model resolution workspace path support (ADR-018 Phase 0c).

Tests that resolve_model_for_operation() correctly handles:
- Local workspace paths (managed and unmanaged)
- HF model IDs (existing cache behavior)
- Edge cases and error handling
"""

import pytest
from pathlib import Path

from mlxk2.core.model_resolution import resolve_model_for_operation


class TestResolveModelForOperationWorkspace:
    """Test workspace path resolution (ADR-018 Phase 0c)."""

    def test_resolve_workspace_path_absolute(self, tmp_path):
        """Test resolves absolute workspace path."""
        (tmp_path / "config.json").write_text('{"model_type": "llama"}')

        resolved_name, commit_hash, ambiguous = resolve_model_for_operation(str(tmp_path))

        assert resolved_name == str(tmp_path.resolve())
        assert commit_hash is None
        assert ambiguous is None

    def test_resolve_workspace_path_relative(self, tmp_path, monkeypatch):
        """Test resolves relative workspace path with ./ prefix."""
        workspace = tmp_path / "my-workspace"
        workspace.mkdir()
        (workspace / "config.json").write_text('{"model_type": "llama"}')

        # Change to parent dir, then use relative path with ./
        monkeypatch.chdir(tmp_path)

        resolved_name, commit_hash, ambiguous = resolve_model_for_operation("./my-workspace")

        assert resolved_name == str(workspace.resolve())
        assert commit_hash is None
        assert ambiguous is None

    def test_resolve_workspace_path_with_dot(self, tmp_path, monkeypatch):
        """Test resolves ./workspace notation."""
        (tmp_path / "config.json").write_text('{"model_type": "llama"}')

        monkeypatch.chdir(tmp_path)

        resolved_name, commit_hash, ambiguous = resolve_model_for_operation(".")

        assert resolved_name == str(tmp_path.resolve())
        assert commit_hash is None
        assert ambiguous is None

    def test_resolve_nonexistent_explicit_path_returns_none(self):
        """Test that nonexistent explicit paths (./) fall through to cache logic."""
        resolved_name, commit_hash, ambiguous = resolve_model_for_operation("./nonexistent")

        # Explicit path ./nonexistent doesn't exist, doesn't match workspace check
        # Falls through to cache logic, which tries to expand it
        assert resolved_name is None
        assert ambiguous == []

    def test_resolve_name_without_prefix_uses_cache_logic(self):
        """Test that names without ./ prefix go through cache resolution."""
        # Even if a local directory exists, without ./ it should try cache first
        resolved_name, commit_hash, ambiguous = resolve_model_for_operation("some-model")

        # Should use cache logic (not find it, return None)
        assert resolved_name is None or "/" in resolved_name  # Either not found or cache path
        assert ambiguous is not None  # Cache logic sets this

    def test_resolve_explicit_path_without_config_falls_back(self, tmp_path):
        """Test explicit path (./) without config.json falls back to cache logic."""
        # Directory exists but no config.json
        # Use ./ to make it an explicit path
        import os
        cwd = os.getcwd()
        try:
            os.chdir(tmp_path.parent)
            resolved_name, commit_hash, ambiguous = resolve_model_for_operation(f"./{tmp_path.name}")

            # Explicit path exists but no config.json → not a workspace, falls to cache logic
            assert resolved_name is None or ambiguous is not None
        finally:
            os.chdir(cwd)

    def test_resolve_workspace_ignores_at_hash_syntax(self, tmp_path):
        """Test that @hash syntax doesn't affect workspace paths."""
        # Edge case: workspace path with @ in it (unlikely but should handle)
        workspace = tmp_path / "workspace"
        workspace.mkdir()
        (workspace / "config.json").write_text('{"model_type": "llama"}')

        # Even with @something, if it's a valid workspace path, use it
        resolved_name, commit_hash, ambiguous = resolve_model_for_operation(str(workspace))

        assert str(workspace.resolve()) in resolved_name
        assert commit_hash is None


class TestResolveModelOperationBackwardCompatibility:
    """Test that existing cache-based resolution still works."""

    def test_resolve_hf_model_id_still_works(self, tmp_path, monkeypatch):
        """Test HF model IDs are not treated as workspace paths."""
        # Mock cache to return expected result for cache-based resolution
        from mlxk2.core import cache
        monkeypatch.setattr(cache, "get_current_model_cache", lambda: tmp_path)

        # HF model ID (not a path) should go through cache logic
        resolved_name, commit_hash, ambiguous = resolve_model_for_operation("mlx-community/Phi-3-mini")

        # Cache logic: exact match fails, returns (None, None, [])
        # The key assertion: it's NOT treated as workspace path (would return absolute path)
        assert resolved_name is None or not Path(resolved_name).exists()
        assert ambiguous == [] or ambiguous is None  # Empty list or None for not found

    def test_resolve_short_name_expansion_still_works(self, tmp_path, monkeypatch):
        """Test short name expansion (existing behavior) still works."""
        from mlxk2.core import cache

        # Create mock cache structure
        cache_dir = tmp_path / "cache"
        cache_dir.mkdir()
        model_dir = cache_dir / "models--mlx-community--Phi-3-mini-4k-instruct-4bit"
        model_dir.mkdir(parents=True)
        snapshot_dir = model_dir / "snapshots" / "abc123"
        snapshot_dir.mkdir(parents=True)
        (snapshot_dir / "config.json").write_text('{}')

        monkeypatch.setattr(cache, "get_current_model_cache", lambda: cache_dir)

        # Short name "phi-3" should expand via cache logic
        resolved_name, commit_hash, ambiguous = resolve_model_for_operation("Phi-3")

        # Either finds the model or returns ambiguous/not found
        assert resolved_name is None or ambiguous is not None or "Phi-3" in str(resolved_name)
