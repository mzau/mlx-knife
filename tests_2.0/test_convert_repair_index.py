"""Tests for convert operation --repair-index (ADR-018 Phase 1).

Tests community repair tool for mlx-vlm #624 affected models:
- Safetensors index rebuild primitive
- Cache sanctity enforcement (hard block)
- Workspace sentinel creation
- Health check integration
"""

import json
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

from mlxk2.operations.convert import rebuild_safetensors_index, convert_operation
from mlxk2.operations.workspace import write_workspace_sentinel, is_managed_workspace


class TestRebuildSafetensorsIndex:
    """Test safetensors index rebuild primitive."""

    def test_rebuild_single_file_returns_false(self, tmp_path):
        """Test single safetensors file returns False (no index needed)."""
        # Create single model file
        (tmp_path / "model.safetensors").write_text("mock safetensors data")

        result = rebuild_safetensors_index(tmp_path)
        assert result is False  # No index needed for single file

    def test_rebuild_no_safetensors_returns_false(self, tmp_path):
        """Test no safetensors files returns False."""
        # Empty directory
        result = rebuild_safetensors_index(tmp_path)
        assert result is False

    def test_rebuild_creates_index_for_multi_shard(self, tmp_path):
        """Test index rebuild for multi-shard model.

        Uses mock safetensors.safe_open to simulate shard reading.
        """
        # Create mock shards
        (tmp_path / "model-00001-of-00003.safetensors").write_text("shard 1")
        (tmp_path / "model-00002-of-00003.safetensors").write_text("shard 2")
        (tmp_path / "model-00003-of-00003.safetensors").write_text("shard 3")

        # Mock safetensors.safe_open to return fake tensor keys
        with patch("safetensors.safe_open") as mock_safe_open:
            # Setup mock context manager
            mock_file1 = MagicMock()
            mock_file1.keys.return_value = ["layer.0.weight", "layer.0.bias"]

            mock_file2 = MagicMock()
            mock_file2.keys.return_value = ["layer.1.weight", "layer.1.bias"]

            mock_file3 = MagicMock()
            mock_file3.keys.return_value = ["layer.2.weight", "layer.2.bias"]

            # safe_open returns different mocks for each shard
            mock_safe_open.side_effect = [
                MagicMock(__enter__=lambda self: mock_file1, __exit__=lambda *args: None),
                MagicMock(__enter__=lambda self: mock_file2, __exit__=lambda *args: None),
                MagicMock(__enter__=lambda self: mock_file3, __exit__=lambda *args: None)
            ]

            result = rebuild_safetensors_index(tmp_path)

            assert result is True  # Index should be rebuilt

            # Verify index file was created
            index_path = tmp_path / "model.safetensors.index.json"
            assert index_path.exists()

            # Verify index structure
            index_data = json.loads(index_path.read_text())
            assert "metadata" in index_data
            assert "weight_map" in index_data

            # Verify weight_map has correct mappings
            weight_map = index_data["weight_map"]
            assert weight_map["layer.0.weight"] == "model-00001-of-00003.safetensors"
            assert weight_map["layer.1.weight"] == "model-00002-of-00003.safetensors"
            assert weight_map["layer.2.weight"] == "model-00003-of-00003.safetensors"

    def test_rebuild_atomic_write(self, tmp_path):
        """Test index write is atomic (tmp + rename)."""
        # Create mock shards
        (tmp_path / "model-00001-of-00002.safetensors").write_text("shard 1")
        (tmp_path / "model-00002-of-00002.safetensors").write_text("shard 2")

        with patch("safetensors.safe_open") as mock_safe_open:
            # Setup mock context managers with DIFFERENT keys per shard
            mock_file1 = MagicMock()
            mock_file1.keys.return_value = ["layer.0.weight"]

            mock_file2 = MagicMock()
            mock_file2.keys.return_value = ["layer.1.weight"]

            # Use side_effect to return different mocks for each shard
            mock_safe_open.side_effect = [
                MagicMock(__enter__=lambda self: mock_file1, __exit__=lambda *args: None),
                MagicMock(__enter__=lambda self: mock_file2, __exit__=lambda *args: None)
            ]

            rebuild_safetensors_index(tmp_path)

            # Verify tmp file doesn't exist after successful write
            tmp_files = list(tmp_path.glob("*.tmp"))
            assert len(tmp_files) == 0  # Tmp file should be renamed


class TestConvertCacheSanctity:
    """Test convert blocks cache writes (ADR-018 core rule)."""

    def test_convert_blocks_cache_source_path(self, tmp_path, monkeypatch):
        """Test convert hard-blocks source in HF cache."""
        # Create mock source workspace
        src = tmp_path / "workspace"
        src.mkdir()
        (src / "config.json").write_text('{"model_type": "llama"}')

        dst = tmp_path / "output"

        # Mock get_current_cache_root to make src appear inside cache
        mock_cache_root = tmp_path
        monkeypatch.setattr(
            "mlxk2.operations.convert.get_current_cache_root",
            lambda: mock_cache_root
        )

        result = convert_operation(
            str(src),
            str(dst),
            mode="repair-index"
        )

        assert result["status"] == "error"
        assert "Source path is in HF cache" in result["error"]["message"]
        assert "Use 'mlxk clone' first" in result["error"]["message"]

    def test_convert_blocks_cache_target_path(self, tmp_path, monkeypatch):
        """Test convert hard-blocks target in HF cache."""
        # Create source OUTSIDE cache (in separate directory)
        src = tmp_path / "outside" / "workspace"
        src.mkdir(parents=True)
        (src / "config.json").write_text('{"model_type": "llama"}')
        (src / "model.safetensors").write_text("mock weights")

        # Target appears inside cache
        cache_dir = tmp_path / "cache"
        cache_dir.mkdir()
        dst = cache_dir / "output"

        # Mock cache root to be cache_dir only (src is outside)
        monkeypatch.setattr(
            "mlxk2.operations.convert.get_current_cache_root",
            lambda: cache_dir
        )

        result = convert_operation(
            str(src),
            str(dst),
            mode="repair-index"
        )

        assert result["status"] == "error"
        assert "Target path is in HF cache" in result["error"]["message"]
        assert "Choose workspace location outside cache" in result["error"]["message"]


class TestConvertWorkspaceSentinel:
    """Test convert creates managed workspaces."""

    def test_convert_creates_managed_workspace(self, tmp_path):
        """Test convert writes workspace sentinel to target."""
        # Create minimal source workspace
        src = tmp_path / "source"
        src.mkdir()
        (src / "config.json").write_text('{"model_type": "llama"}')
        (src / "model.safetensors").write_text("mock weights")

        dst = tmp_path / "target"

        # Mock cache root to be outside tmp_path
        with patch("mlxk2.operations.convert.get_current_cache_root") as mock_cache:
            mock_cache.return_value = Path("/nonexistent/cache")

            result = convert_operation(
                str(src),
                str(dst),
                mode="repair-index",
                skip_health=True  # Skip health for this test
            )

            assert result["status"] == "success"

            # Verify target has sentinel
            assert is_managed_workspace(dst) is True

            # Verify sentinel metadata
            from mlxk2.operations.workspace import read_workspace_metadata
            metadata = read_workspace_metadata(dst)
            assert metadata["managed"] is True
            assert metadata["operation"] == "convert"
            assert metadata["mode"] == "repair-index"

    def test_convert_inherits_source_metadata(self, tmp_path):
        """Test convert preserves source repo info if source is managed."""
        # Create managed source workspace
        src = tmp_path / "source"
        src.mkdir()
        (src / "config.json").write_text('{"model_type": "llama"}')
        (src / "model.safetensors").write_text("mock weights")

        # Write source sentinel with metadata
        src_metadata = {
            "mlxk_version": "2.0.4",
            "created_at": "2025-12-29T10:00:00Z",
            "source_repo": "mlx-community/Llama-3.2-3B",
            "source_revision": "abc123",
            "managed": True,
            "operation": "clone"
        }
        write_workspace_sentinel(src, src_metadata)

        dst = tmp_path / "target"

        with patch("mlxk2.operations.convert.get_current_cache_root") as mock_cache:
            mock_cache.return_value = Path("/nonexistent/cache")

            result = convert_operation(
                str(src),
                str(dst),
                mode="repair-index",
                skip_health=True
            )

            assert result["status"] == "success"

            # Verify target inherits source_repo
            from mlxk2.operations.workspace import read_workspace_metadata
            metadata = read_workspace_metadata(dst)
            assert metadata["source_repo"] == "mlx-community/Llama-3.2-3B"
            assert metadata["source_revision"] == "abc123"
            assert metadata["operation"] == "convert"  # Operation updated


class TestConvertValidation:
    """Test convert validation logic."""

    def test_convert_requires_existing_source(self, tmp_path):
        """Test convert fails if source doesn't exist."""
        result = convert_operation(
            "/nonexistent/source",
            str(tmp_path / "target"),
            mode="repair-index"
        )

        assert result["status"] == "error"
        assert "does not exist" in result["error"]["message"]

    def test_convert_requires_empty_target(self, tmp_path):
        """Test convert requires empty target directory."""
        src = tmp_path / "source"
        src.mkdir()
        (src / "config.json").write_text('{"model_type": "llama"}')

        # Create non-empty target
        dst = tmp_path / "target"
        dst.mkdir()
        (dst / "existing_file.txt").write_text("data")

        with patch("mlxk2.operations.convert.get_current_cache_root") as mock_cache:
            mock_cache.return_value = Path("/nonexistent/cache")

            result = convert_operation(
                str(src),
                str(dst),
                mode="repair-index"
            )

            assert result["status"] == "error"
            assert "not empty" in result["error"]["message"]

    def test_convert_unsupported_mode_fails(self, tmp_path):
        """Test convert fails with unsupported mode."""
        src = tmp_path / "source"
        src.mkdir()
        (src / "config.json").write_text('{"model_type": "llama"}')

        dst = tmp_path / "target"

        with patch("mlxk2.operations.convert.get_current_cache_root") as mock_cache:
            mock_cache.return_value = Path("/nonexistent/cache")

            result = convert_operation(
                str(src),
                str(dst),
                mode="unsupported-mode"
            )

            assert result["status"] == "error"
            assert "Unsupported" in result["error"]["message"]
