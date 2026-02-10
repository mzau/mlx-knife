"""Tests for clone operation following ADR-007 Phase 1: Same-Volume APFS strategy.

Tests for the new isolated temp cache + APFS CoW implementation that replaces
the deprecated ADR-006 approach.

Key test areas:
- APFS filesystem validation
- Temp cache creation with sentinel safety
- Volume-aware placement
- APFS copy-on-write cloning
- Temp cache cleanup safety
- JSON API 0.1.4 compliance
"""

import json
import os
import shutil
import subprocess
import tempfile
import time
from pathlib import Path
from unittest.mock import patch, MagicMock, mock_open
import pytest

from mlxk2.operations.clone import (
    clone_operation,
    _validate_apfs_filesystem,
    _is_apfs_filesystem,
    _create_temp_cache_same_volume,
    _get_deterministic_temp_cache_name,
    _check_temp_cache_resume,
    _get_volume_mount_point,
    _resolve_latest_snapshot,
    _apfs_clone_directory,
    _cleanup_temp_cache_safe,
    FilesystemError
)


class TestAPFSFilesystemValidation:
    """Test suite for APFS filesystem requirement validation."""

    def test_validate_apfs_filesystem_success(self):
        """Test APFS validation passes on APFS filesystem."""
        test_path = Path("/tmp/test")

        with patch('mlxk2.operations.clone._is_apfs_filesystem', return_value=True):
            # Should not raise exception
            _validate_apfs_filesystem(test_path)

    def test_validate_apfs_filesystem_failure(self):
        """Test APFS validation fails on non-APFS filesystem."""
        test_path = Path("/tmp/test")

        with patch('mlxk2.operations.clone._is_apfs_filesystem', return_value=False):
            with pytest.raises(FilesystemError) as exc_info:
                _validate_apfs_filesystem(test_path)

            assert "APFS required for clone operations" in str(exc_info.value)
            assert str(test_path) in str(exc_info.value)

    def test_is_apfs_filesystem_true(self):
        """Test APFS detection returns True - real test on Phase 1 APFS system."""
        # Test current working directory - should be APFS on Phase 1 developer system
        result = _is_apfs_filesystem(Path.cwd())
        assert result is True

        # Test HF_HOME if set - should also be APFS on Phase 1 system
        hf_home = os.environ.get('HF_HOME')
        if hf_home:
            result = _is_apfs_filesystem(Path(hf_home))
            assert result is True

    def test_is_apfs_filesystem_false(self):
        """Test APFS detection returns False for non-APFS."""
        test_path = Path("/mnt/nfs")

        with patch('subprocess.run') as mock_run:
            mock_result = MagicMock()
            mock_result.stdout = '/dev/nfs on /mnt/nfs (nfs, local, nodev, nosuid)\n'
            mock_run.return_value = mock_result

            result = _is_apfs_filesystem(test_path)

            assert result is False

    def test_is_apfs_filesystem_error_fallback(self):
        """Test APFS detection safely falls back on subprocess error."""
        test_path = Path("/invalid/path")

        with patch('subprocess.run') as mock_run:
            mock_run.side_effect = subprocess.CalledProcessError(1, 'stat')

            result = _is_apfs_filesystem(test_path)

            assert result is False  # Safe fallback


class TestVolumeDetection:
    """Test suite for volume mount point detection."""

    def test_get_volume_mount_point_root(self):
        """Test volume detection at filesystem root."""
        test_path = Path("/")

        result = _get_volume_mount_point(test_path)

        assert result == Path("/")

    def test_get_volume_mount_point_same_volume(self):
        """Test volume detection with same device."""
        test_path = Path("/Users/test/workspace")

        with patch.object(Path, 'stat') as mock_stat:
            # All paths have same device ID (same volume)
            mock_stat.return_value.st_dev = 12345

            result = _get_volume_mount_point(test_path)

            # Should traverse to root
            assert result == Path("/")

    def test_get_volume_mount_point_mount_boundary(self):
        """Test volume detection at mount boundary."""
        test_path = Path("/Volumes/External/workspace")

        # Create path-specific mocks
        external_stat = MagicMock()
        external_stat.st_dev = 67890
        volumes_stat = MagicMock()
        volumes_stat.st_dev = 12345
        workspace_stat = MagicMock()
        workspace_stat.st_dev = 67890

        def mock_stat_for_path(self):
            if str(self) == "/Volumes/External":
                return external_stat
            elif str(self) == "/Volumes":
                return volumes_stat
            else:  # workspace or other paths
                return workspace_stat

        with patch.object(Path, 'stat', mock_stat_for_path):
            result = _get_volume_mount_point(test_path)

            assert result == Path("/Volumes/External")

    def test_get_volume_mount_point_permission_error(self):
        """Test volume detection handles permission errors."""
        test_path = Path("/restricted/path")

        with patch.object(Path, 'stat') as mock_stat:
            mock_stat.side_effect = PermissionError("Permission denied")

            result = _get_volume_mount_point(test_path)

            # Should fall back to filesystem root
            assert result == Path("/")


class TestRealFilesystemValidation:
    """Test suite for real filesystem validation without mocks."""

    def test_real_apfs_detection_system_volume(self):
        """Test APFS detection on real system volume (should be APFS on modern macOS)."""
        # Test current working directory (should be on system volume)
        current_path = Path.cwd()

        # This should work without exceptions
        is_apfs = _is_apfs_filesystem(current_path)

        # On modern macOS, system volume should be APFS
        # But we don't assert True to avoid false failures on older systems
        assert isinstance(is_apfs, bool)

    def test_real_apfs_detection_user_home(self):
        """Test APFS detection on user home directory."""
        home_path = Path.home()

        # This should work without exceptions
        is_apfs = _is_apfs_filesystem(home_path)
        assert isinstance(is_apfs, bool)

    def test_real_volume_mount_point_detection_system(self):
        """Test real volume mount point detection for system paths."""
        # Test various system paths
        test_paths = [
            Path.cwd(),
            Path.home(),
            Path("/Users"),
            Path("/tmp"),
        ]

        for path in test_paths:
            if path.exists():
                mount_point = _get_volume_mount_point(path)

                # Mount point should be a valid path
                assert isinstance(mount_point, Path)
                assert mount_point.exists()
                assert mount_point.is_dir()
                # Mount point should be an ancestor of the original path
                assert path.resolve().is_relative_to(mount_point) or path.resolve() == mount_point

    def test_real_same_volume_detection_consistency(self):
        """Test that same-volume detection works consistently for Phase 1."""
        # Test current working directory
        current_path = Path.cwd()
        current_mount = _get_volume_mount_point(current_path)

        # Test a subdirectory in the same location
        if current_path.is_dir():
            subdir_mount = _get_volume_mount_point(current_path / "subdir_test_path")

            # Should have same mount point (same volume)
            assert current_mount == subdir_mount
            print(f"Volume consistency test - Mount: {current_mount}")

    def test_real_apfs_validation_phase1_requirement(self):
        """Test APFS validation for Phase 1 requirement on current workspace."""
        # Phase 1: Only test current working directory (realistic workspace)
        current_path = Path.cwd()

        # APFS detection should work without errors
        is_apfs = _is_apfs_filesystem(current_path)
        assert isinstance(is_apfs, bool)

        if is_apfs:
            # If APFS, validation should pass
            try:
                _validate_apfs_filesystem(current_path)
                print(f"✅ APFS validation passed for: {current_path}")
            except FilesystemError:
                pytest.fail("APFS validation failed on APFS filesystem")
        else:
            # If not APFS, validation should fail (Phase 1 requirement)
            with pytest.raises(FilesystemError) as exc_info:
                _validate_apfs_filesystem(current_path)
            assert "APFS required" in str(exc_info.value)
            print(f"⚠️ Non-APFS detected: {current_path} - Phase 1 will reject")


class TestTempCacheCreation:
    """Test suite for temporary cache creation with sentinel safety."""

    def test_create_temp_cache_same_volume(self, tmp_path):
        """Test temp cache creation on same volume."""
        target_workspace = tmp_path / "workspace"
        model_spec = "test/model"

        with patch('mlxk2.operations.clone._get_volume_mount_point') as mock_volume:
            mock_volume.return_value = tmp_path

            temp_cache, should_download = _create_temp_cache_same_volume(target_workspace, model_spec, False)

            # Verify temp cache is on same volume
            assert temp_cache.parent == tmp_path
            assert temp_cache.exists()
            assert temp_cache.is_dir()
            assert should_download is True  # New temp cache, should download

            # Verify sentinel file exists
            sentinel = temp_cache / ".mlxk2_temp_cache_sentinel"
            assert sentinel.exists()
            assert "mlxk2_temp_cache_created_" in sentinel.read_text()

            # Cleanup
            shutil.rmtree(temp_cache)

    # Note: test_create_temp_cache_unique_names and test_create_temp_cache_includes_pid
    # removed in ADR-018 Phase 0b - replaced by deterministic naming (tested in TestResumableClone)


class TestSentinelSafetyMechanism:
    """Test suite for sentinel-based safety mechanism."""

    def test_cleanup_temp_cache_safe_with_sentinel(self, tmp_path):
        """Test cleanup succeeds when sentinel exists."""
        temp_cache = tmp_path / "temp_cache"
        temp_cache.mkdir()

        # Create sentinel
        sentinel = temp_cache / ".mlxk2_temp_cache_sentinel"
        sentinel.write_text("mlxk2_temp_cache_created_123456789")

        result = _cleanup_temp_cache_safe(temp_cache)

        assert result is True
        assert not temp_cache.exists()

    def test_cleanup_temp_cache_safe_without_sentinel(self, tmp_path):
        """Test cleanup refuses when sentinel missing."""
        temp_cache = tmp_path / "temp_cache"
        temp_cache.mkdir()

        # No sentinel file

        with patch('mlxk2.operations.clone.logger') as mock_logger:
            result = _cleanup_temp_cache_safe(temp_cache)

            assert result is False
            assert temp_cache.exists()  # Should not be deleted
            mock_logger.warning.assert_called_once()
            assert "no sentinel found" in mock_logger.warning.call_args[0][0]

    def test_cleanup_temp_cache_safe_protects_user_cache(self, tmp_path):
        """Test cleanup protects user cache directories without sentinel."""
        # Simulate user cache directory structure
        user_cache = tmp_path / ".cache" / "huggingface" / "hub"
        user_cache.mkdir(parents=True)
        (user_cache / "important_model").mkdir()

        with patch('mlxk2.operations.clone.logger') as mock_logger:
            result = _cleanup_temp_cache_safe(user_cache)

            assert result is False
            assert user_cache.exists()
            assert (user_cache / "important_model").exists()
            mock_logger.warning.assert_called_once()

    def test_cleanup_temp_cache_safe_handles_nonexistent(self):
        """Test cleanup handles non-existent paths gracefully."""
        nonexistent_path = Path("/nonexistent/temp/cache")

        result = _cleanup_temp_cache_safe(nonexistent_path)

        assert result is False


# TestHFHomePatch class removed - _patch_hf_home function no longer exists
# Clone operations now use pull_to_cache with explicit cache_dir parameter


class TestSnapshotResolution:
    """Test suite for latest snapshot resolution in temp cache."""

    def test_resolve_latest_snapshot_success(self, tmp_path):
        """Test successful snapshot resolution."""
        temp_cache = tmp_path / "temp_cache"
        model_name = "mlx-community/Phi-3-mini"

        # Create mock cache structure
        cache_dir = temp_cache / "models--mlx-community--Phi-3-mini"
        snapshots_dir = cache_dir / "snapshots"
        snapshots_dir.mkdir(parents=True)

        # Create mock snapshots with different timestamps
        snapshot1 = snapshots_dir / "abc123"
        snapshot2 = snapshots_dir / "def456"
        snapshot1.mkdir()
        snapshot2.mkdir()

        # Set different modification times
        os.utime(snapshot1, (1000, 1000))
        os.utime(snapshot2, (2000, 2000))  # More recent

        with patch('mlxk2.operations.clone.hf_to_cache_dir') as mock_hf_to_cache:
            mock_hf_to_cache.return_value = "models--mlx-community--Phi-3-mini"

            result = _resolve_latest_snapshot(temp_cache, model_name)

            assert result == snapshot2  # Should return most recent

    def test_resolve_latest_snapshot_no_cache(self, tmp_path):
        """Test snapshot resolution when cache doesn't exist."""
        temp_cache = tmp_path / "temp_cache"
        model_name = "nonexistent/model"

        with patch('mlxk2.operations.clone.hf_to_cache_dir') as mock_hf_to_cache:
            mock_hf_to_cache.return_value = "models--nonexistent--model"

            result = _resolve_latest_snapshot(temp_cache, model_name)

            assert result is None

    def test_resolve_latest_snapshot_no_snapshots(self, tmp_path):
        """Test snapshot resolution when snapshots directory is empty."""
        temp_cache = tmp_path / "temp_cache"
        model_name = "empty/model"

        # Create cache structure but no snapshots
        cache_dir = temp_cache / "models--empty--model"
        snapshots_dir = cache_dir / "snapshots"
        snapshots_dir.mkdir(parents=True)

        with patch('mlxk2.operations.clone.hf_to_cache_dir') as mock_hf_to_cache:
            mock_hf_to_cache.return_value = "models--empty--model"

            result = _resolve_latest_snapshot(temp_cache, model_name)

            assert result is None


class TestAPFSCloneDirectory:
    """Test suite for APFS copy-on-write directory cloning."""

    def test_apfs_clone_directory_success(self, tmp_path):
        """Test successful APFS directory cloning."""
        source = tmp_path / "source"
        target = tmp_path / "target"

        # Create source structure
        source.mkdir()
        (source / "file1.txt").write_text("content1")
        (source / "subdir").mkdir()
        (source / "subdir" / "file2.txt").write_text("content2")

        with patch('subprocess.run') as mock_run:
            mock_run.return_value = MagicMock()  # Successful subprocess calls

            result = _apfs_clone_directory(source, target)

            assert result is True
            # Verify cp -c was called for each file
            assert mock_run.call_count == 2

            # Check calls used clonefile (-c flag)
            for call in mock_run.call_args_list:
                args = call[0][0]
                assert args[0] == 'cp'
                assert '-c' in args

    def test_apfs_clone_directory_subprocess_error(self, tmp_path):
        """Test APFS cloning handles subprocess errors."""
        source = tmp_path / "source"
        target = tmp_path / "target"

        source.mkdir()
        (source / "file.txt").write_text("content")

        with patch('subprocess.run') as mock_run:
            mock_run.side_effect = subprocess.CalledProcessError(1, 'cp')

            result = _apfs_clone_directory(source, target)

            assert result is False

    def test_apfs_clone_directory_creates_target_structure(self, tmp_path):
        """Test APFS cloning creates target directory structure."""
        source = tmp_path / "source"
        target = tmp_path / "target"

        # Create nested source structure
        (source / "deep" / "nested" / "path").mkdir(parents=True)
        (source / "deep" / "nested" / "path" / "file.txt").write_text("content")

        with patch('subprocess.run') as mock_run:
            mock_run.return_value = MagicMock()

            result = _apfs_clone_directory(source, target)

            assert result is True
            # Verify target directory structure would be created
            call_args = mock_run.call_args_list[0][0][0]
            target_file = call_args[-1]  # Last argument should be target file
            assert "deep/nested/path/file.txt" in target_file


class TestCloneOperationIntegration:
    """Integration tests for complete clone operation workflow."""

    def test_clone_operation_success_workflow(self, tmp_path):
        """Test complete successful clone operation workflow."""
        target_dir = str(tmp_path / "workspace")
        model_spec = "mlx-community/Phi-3-mini"

        # Create real temp cache with sentinel for realistic cleanup test
        real_temp_cache = tmp_path / "temp_cache"
        real_temp_cache.mkdir()
        sentinel = real_temp_cache / ".mlxk2_temp_cache_sentinel"
        sentinel.write_text("mlxk2_temp_cache_created_test")

        with patch('mlxk2.operations.clone._validate_apfs_filesystem'), \
             patch('mlxk2.operations.clone._create_temp_cache_same_volume') as mock_create_cache, \
             patch('mlxk2.operations.clone.pull_to_cache') as mock_pull, \
             patch('mlxk2.operations.clone._resolve_latest_snapshot') as mock_resolve, \
             patch('mlxk2.operations.health.health_from_cache') as mock_health, \
             patch('mlxk2.operations.clone._apfs_clone_directory') as mock_clone:

            # Use real temp cache
            mock_create_cache.return_value = (real_temp_cache, True)  # (temp_cache, should_download)
            mock_health.return_value = (True, "Model is healthy")

            mock_pull.return_value = {
                "status": "success",
                "data": {"model": model_spec}
            }

            mock_snapshot = MagicMock()
            mock_snapshot.exists.return_value = True
            mock_resolve.return_value = mock_snapshot

            mock_health.return_value = (True, "Model is healthy")
            mock_clone.return_value = True

            result = clone_operation(model_spec, target_dir)

            # Debug: Print result if error
            if result["status"] != "success":
                print(f"Clone operation failed: {result}")

            # Verify success response
            assert result["status"] == "success"
            assert result["command"] == "clone"
            assert result["data"]["model"] == model_spec
            assert result["data"]["clone_status"] == "success"
            assert result["error"] is None

            # Verify workflow steps
            mock_create_cache.assert_called_once()
            mock_pull.assert_called_once_with(model_spec, real_temp_cache)
            mock_resolve.assert_called_once()
            # Health check is currently skipped in clone operation (TODO: implement health_check_to_cache)
            # mock_health.assert_called_once_with(model_spec)
            mock_clone.assert_called_once()

            # Verify real cleanup happened (temp cache should be deleted)
            assert not real_temp_cache.exists()

    def test_clone_operation_apfs_validation_failure(self, tmp_path):
        """Test clone operation fails APFS validation."""
        target_dir = str(tmp_path / "workspace")
        model_spec = "any/model"

        with patch('mlxk2.operations.clone._validate_apfs_filesystem') as mock_validate:
            mock_validate.side_effect = FilesystemError("APFS required")

            result = clone_operation(model_spec, target_dir)

            assert result["status"] == "error"
            assert result["data"]["clone_status"] == "filesystem_error"
            assert "APFS required" in result["error"]["message"]

    def test_clone_operation_pull_failure(self, tmp_path):
        """Test clone operation handles pull failure."""
        target_dir = str(tmp_path / "workspace")
        model_spec = "invalid/model"

        # Create real temp cache for cleanup test
        real_temp_cache = tmp_path / "temp_cache"
        real_temp_cache.mkdir()
        sentinel = real_temp_cache / ".mlxk2_temp_cache_sentinel"
        sentinel.write_text("mlxk2_temp_cache_created_test")

        with patch('mlxk2.operations.clone._validate_apfs_filesystem'), \
             patch('mlxk2.operations.clone._create_temp_cache_same_volume') as mock_create_cache, \
             patch('mlxk2.operations.clone.pull_to_cache') as mock_pull:

            mock_create_cache.return_value = (real_temp_cache, True)  # (temp_cache, should_download)

            mock_pull.return_value = {
                "status": "error",
                "error": {"message": "Model not found"}
            }

            result = clone_operation(model_spec, target_dir)

            assert result["status"] == "error"
            assert result["data"]["clone_status"] == "pull_failed"
            assert "Pull operation failed" in result["error"]["message"]

            # Verify real cleanup happened even on failure
            assert not real_temp_cache.exists()

    def test_clone_operation_health_check_failure(self, tmp_path):
        """Test clone operation continues despite health check failure (Session 59).

        Unhealthy models must be clonable for community repair workflow.
        """
        target_dir = str(tmp_path / "workspace")
        model_spec = "corrupted/model"

        # Create real temp cache
        real_temp_cache = tmp_path / "temp_cache"
        real_temp_cache.mkdir()
        sentinel = real_temp_cache / ".mlxk2_temp_cache_sentinel"
        sentinel.write_text("mlxk2_temp_cache_created_test")

        with patch('mlxk2.operations.clone._validate_apfs_filesystem'), \
             patch('mlxk2.operations.clone._create_temp_cache_same_volume') as mock_create_cache, \
             patch('mlxk2.operations.clone.pull_to_cache') as mock_pull, \
             patch('mlxk2.operations.clone._resolve_latest_snapshot') as mock_resolve, \
             patch('mlxk2.operations.health.health_from_cache') as mock_health:

            mock_create_cache.return_value = (real_temp_cache, True)  # (temp_cache, should_download)

            mock_pull.return_value = {"status": "success", "data": {"model": model_spec}}

            mock_snapshot = MagicMock()
            mock_snapshot.exists.return_value = True
            mock_resolve.return_value = mock_snapshot

            mock_health.return_value = (False, "Model is corrupted")

            # Need to mock clone and sentinel too (Session 59: unhealthy doesn't block)
            with patch('mlxk2.operations.clone._apfs_clone_directory') as mock_clone, \
                 patch('mlxk2.operations.clone.write_workspace_sentinel'):

                mock_clone.return_value = True

                result = clone_operation(model_spec, target_dir)

                # Session 59 fix: Should succeed despite unhealthy
                assert result["status"] == "success"
                assert result["data"]["clone_status"] == "success"

                # Health status recorded as warning
                assert result["data"]["health"] == "unhealthy"
                assert result["data"]["health_reason"] == "Model is corrupted"

            # Verify real cleanup happened
            assert not real_temp_cache.exists()

    def test_clone_operation_target_not_empty(self, tmp_path):
        """Test clone operation fails when target directory not empty."""
        target_dir = tmp_path / "workspace"
        target_dir.mkdir()
        (target_dir / "existing_file.txt").write_text("content")

        result = clone_operation("any/model", str(target_dir))

        assert result["status"] == "error"
        assert result["data"]["clone_status"] == "error"
        assert "not empty" in result["error"]["message"]

    def test_clone_operation_target_is_file(self, tmp_path):
        """Test clone operation fails when target exists as file."""
        target_file = tmp_path / "workspace.txt"
        target_file.write_text("content")

        result = clone_operation("any/model", str(target_file))

        assert result["status"] == "error"
        assert result["data"]["clone_status"] == "error"
        assert "not a directory" in result["error"]["message"]

    def test_clone_operation_apfs_clone_failure(self, tmp_path):
        """Test clone operation handles APFS clone failure."""
        target_dir = str(tmp_path / "workspace")
        model_spec = "test/model"

        # Create real temp cache
        real_temp_cache = tmp_path / "temp_cache"
        real_temp_cache.mkdir()
        sentinel = real_temp_cache / ".mlxk2_temp_cache_sentinel"
        sentinel.write_text("mlxk2_temp_cache_created_test")

        with patch('mlxk2.operations.clone._validate_apfs_filesystem'), \
             patch('mlxk2.operations.clone._create_temp_cache_same_volume') as mock_create_cache, \
             patch('mlxk2.operations.clone.pull_to_cache') as mock_pull, \
             patch('mlxk2.operations.clone._resolve_latest_snapshot') as mock_resolve, \
             patch('mlxk2.operations.health.health_from_cache') as mock_health, \
             patch('mlxk2.operations.clone._apfs_clone_directory') as mock_clone:

            mock_create_cache.return_value = (real_temp_cache, True)  # (temp_cache, should_download)

            mock_pull.return_value = {"status": "success", "data": {"model": model_spec}}

            mock_snapshot = MagicMock()
            mock_snapshot.exists.return_value = True
            mock_resolve.return_value = mock_snapshot

            mock_health.return_value = (True, "Model is healthy")
            mock_clone.return_value = False  # Clone fails

            result = clone_operation(model_spec, target_dir)

            assert result["status"] == "error"
            assert result["data"]["clone_status"] == "filesystem_error"
            assert "APFS clone operation failed" in result["error"]["message"]

            # ADR-018 Phase 0b: Temp cache is KEPT on failure with complete download (for debugging/retry)
            assert real_temp_cache.exists()
            # Download marker should exist (indicating complete download)
            assert (real_temp_cache / ".mlxk2_download_complete").exists()


@pytest.mark.spec
class TestCloneJSONAPICompliance:
    """Test suite for JSON API 0.1.4 compliance."""

    def test_clone_success_response_schema(self, tmp_path):
        """Test successful clone response matches JSON API 0.1.4 schema."""
        target_dir = str(tmp_path / "workspace")
        model_spec = "mlx-community/Phi-3-mini"

        # Create real temp cache for JSON schema tests
        real_temp_cache = tmp_path / "temp_cache"
        real_temp_cache.mkdir()
        sentinel = real_temp_cache / ".mlxk2_temp_cache_sentinel"
        sentinel.write_text("mlxk2_temp_cache_created_test")

        with patch('mlxk2.operations.clone._validate_apfs_filesystem'), \
             patch('mlxk2.operations.clone._create_temp_cache_same_volume') as mock_create_cache, \
             patch('mlxk2.operations.clone.pull_to_cache') as mock_pull, \
             patch('mlxk2.operations.clone._resolve_latest_snapshot') as mock_resolve, \
             patch('mlxk2.operations.health.health_from_cache') as mock_health, \
             patch('mlxk2.operations.clone._apfs_clone_directory'):

            mock_create_cache.return_value = (real_temp_cache, True)  # (temp_cache, should_download)
            mock_pull.return_value = {"status": "success", "data": {"model": model_spec}}
            mock_health.return_value = (True, "Model is healthy")

            mock_snapshot = MagicMock()
            mock_snapshot.exists.return_value = True
            mock_resolve.return_value = mock_snapshot

            result = clone_operation(model_spec, target_dir)

            # Validate top-level structure
            assert isinstance(result, dict)
            assert set(result.keys()) == {"status", "command", "error", "data"}

            # Validate success response
            assert result["status"] == "success"
            assert result["command"] == "clone"
            assert result["error"] is None

            # Validate data section (per JSON API 0.1.4)
            data = result["data"]
            required_fields = {"model", "clone_status", "message", "target_dir", "health_check"}
            assert set(data.keys()) >= required_fields

            assert data["model"] == model_spec
            assert data["clone_status"] == "success"
            assert isinstance(data["message"], str)
            assert isinstance(data["target_dir"], str)
            assert isinstance(data["health_check"], bool)

    def test_clone_error_response_schema(self, tmp_path):
        """Test error clone response matches JSON API 0.1.4 schema."""
        target_dir = str(tmp_path / "workspace")
        model_spec = "invalid/model"

        with patch('mlxk2.operations.clone._validate_apfs_filesystem') as mock_validate:
            mock_validate.side_effect = FilesystemError("APFS required")

            result = clone_operation(model_spec, target_dir)

            # Validate error response structure
            assert result["status"] == "error"
            assert result["command"] == "clone"
            assert result["error"] is not None

            # Validate error section
            error = result["error"]
            assert "type" in error
            assert "message" in error
            assert isinstance(error["type"], str)
            assert isinstance(error["message"], str)

            # Validate data section still present
            assert "data" in result
            assert "clone_status" in result["data"]
            assert result["data"]["clone_status"] == "filesystem_error"

    def test_clone_response_no_extra_fields(self, tmp_path):
        """Test clone response doesn't include fields not in JSON API 0.1.4."""
        target_dir = str(tmp_path / "workspace")
        model_spec = "test/model"

        with patch('mlxk2.operations.clone._validate_apfs_filesystem'), \
             patch('mlxk2.operations.clone._create_temp_cache_same_volume'), \
             patch('mlxk2.operations.clone.pull_to_cache') as mock_pull, \
             patch('mlxk2.operations.clone._resolve_latest_snapshot') as mock_resolve, \
             patch('mlxk2.operations.health.health_from_cache') as mock_health, \
             patch('mlxk2.operations.clone._apfs_clone_directory'), \
             patch('mlxk2.operations.clone._cleanup_temp_cache_safe'):

            mock_pull.return_value = {"status": "success", "data": {"model": model_spec}}
            mock_health.return_value = (True, "Model is healthy")

            mock_snapshot = MagicMock()
            mock_snapshot.exists.return_value = True
            mock_resolve.return_value = mock_snapshot

            result = clone_operation(model_spec, target_dir)

            # Should not include cache-related fields not in API
            data = result["data"]
            assert "cache_cleanup" not in data
            assert "cache_preserved" not in data
            assert "copy_method" not in data


class TestCloneCoreFeatures:
    """Test suite for core clone features and scenarios."""

    def test_clone_same_model_twice_different_versions(self, tmp_path):
        """Test cloning same model multiple times always gets latest version.

        This test validates the core ADR-007 improvement over ADR-006:
        - User cache is preserved (no destructive deletion)
        - Each clone gets fresh pull (latest version)
        - No version conflicts or outdated snapshots
        """
        target_dir1 = str(tmp_path / "workspace1")
        target_dir2 = str(tmp_path / "workspace2")
        model_spec = "org/model"

        with patch('mlxk2.operations.clone._validate_apfs_filesystem'), \
             patch('mlxk2.operations.clone._create_temp_cache_same_volume') as mock_create_cache, \
             patch('mlxk2.operations.clone.pull_to_cache') as mock_pull, \
             patch('mlxk2.operations.clone._resolve_latest_snapshot') as mock_resolve, \
             patch('mlxk2.operations.health.health_from_cache') as mock_health, \
             patch('mlxk2.operations.clone._apfs_clone_directory'), \
             patch('mlxk2.operations.clone._cleanup_temp_cache_safe') as mock_cleanup:

            # Setup different temp caches for each clone
            temp_cache1 = tmp_path / "temp_cache_1"
            temp_cache1.mkdir()  # Create directory so .exists() returns True
            temp_cache2 = tmp_path / "temp_cache_2"
            temp_cache2.mkdir()  # Create directory so .exists() returns True
            mock_create_cache.side_effect = [(temp_cache1, True), (temp_cache2, True)]  # (temp_cache, should_download)
            mock_health.return_value = (True, "Model is healthy")

            # Setup side effects for both clones
            snapshot1 = MagicMock()
            snapshot1.exists.return_value = True
            snapshot2 = MagicMock()
            snapshot2.exists.return_value = True

            mock_pull.side_effect = [
                {"status": "success", "data": {"model": "org/model@abc123"}},
                {"status": "success", "data": {"model": "org/model@def456"}}
            ]
            mock_resolve.side_effect = [snapshot1, snapshot2]

            result1 = clone_operation(model_spec, target_dir1)
            result2 = clone_operation(model_spec, target_dir2)

            # Both should succeed
            assert result1["status"] == "success"
            assert result2["status"] == "success"

            # Each gets the version that was current at pull time
            assert result1["data"]["model"] == "org/model@abc123"
            assert result2["data"]["model"] == "org/model@def456"

            # Verify separate temp caches were used (isolation)
            assert mock_create_cache.call_count == 2
            assert mock_cleanup.call_count == 2

            # Verify each pull was independent (fresh download)
            assert mock_pull.call_count == 2
            for call in mock_pull.call_args_list:
                assert call[0][0] == model_spec  # Same model spec

    def test_clone_preserves_user_cache_with_existing_model(self, tmp_path):
        """Test clone preserves user cache when model already exists locally.

        ADR-007 core principle: User cache is NEVER touched during clone operations.
        """
        target_dir = str(tmp_path / "workspace")
        model_spec = "existing/model"

        # Simulate existing user cache (this stays untouched)
        user_cache = tmp_path / "user_cache"
        user_cache.mkdir()

        with patch('mlxk2.operations.clone._validate_apfs_filesystem'), \
             patch('mlxk2.operations.clone._create_temp_cache_same_volume') as mock_create_cache, \
             patch('mlxk2.operations.clone.pull_to_cache') as mock_pull, \
             patch('mlxk2.operations.clone._resolve_latest_snapshot') as mock_resolve, \
             patch('mlxk2.operations.health.health_from_cache') as mock_health, \
             patch('mlxk2.operations.clone._apfs_clone_directory'), \
             patch('mlxk2.operations.clone._cleanup_temp_cache_safe') as mock_cleanup:

            # Different temp cache (not user cache)
            temp_cache = tmp_path / "temp_cache"
            temp_cache.mkdir()  # Create directory so .exists() returns True
            mock_create_cache.return_value = (temp_cache, True)  # (temp_cache, should_download)
            mock_health.return_value = (True, "Model is healthy")

            mock_snapshot = MagicMock()
            mock_snapshot.exists.return_value = True
            mock_resolve.return_value = mock_snapshot

            mock_pull.return_value = {"status": "success", "data": {"model": model_spec}}

            result = clone_operation(model_spec, target_dir)

            assert result["status"] == "success"

            # User cache should still exist (untouched)
            assert user_cache.exists()

            # Only temp cache should be cleaned up
            mock_cleanup.assert_called_once_with(temp_cache)
            # User cache path never passed to cleanup
            assert all(call[0][0] != user_cache for call in mock_cleanup.call_args_list)


class TestCloneEdgeCases:
    """Test suite for edge cases and error conditions."""

    def test_clone_operation_without_health_check(self, tmp_path):
        """Test clone operation with health check disabled."""
        target_dir = str(tmp_path / "workspace")
        model_spec = "test/model"

        # Create mock temp cache
        temp_cache = tmp_path / "temp_cache"
        temp_cache.mkdir()

        with patch('mlxk2.operations.clone._validate_apfs_filesystem'), \
             patch('mlxk2.operations.clone._create_temp_cache_same_volume') as mock_create_cache, \
             patch('mlxk2.operations.clone.pull_to_cache') as mock_pull, \
             patch('mlxk2.operations.clone._resolve_latest_snapshot') as mock_resolve, \
             patch('mlxk2.operations.health.health_from_cache') as mock_health, \
             patch('mlxk2.operations.clone._apfs_clone_directory') as mock_clone, \
             patch('mlxk2.operations.clone._cleanup_temp_cache_safe'):

            mock_create_cache.return_value = (temp_cache, True)  # (temp_cache, should_download)
            mock_pull.return_value = {"status": "success", "data": {"model": model_spec}}

            # Mock snapshot resolution
            mock_snapshot = MagicMock()
            mock_snapshot.exists.return_value = True
            mock_resolve.return_value = mock_snapshot

            mock_clone.return_value = True

            result = clone_operation(model_spec, target_dir, health_check=False)

            assert result["status"] == "success"
            assert result["data"]["health_check"] is False

            # Health check should not be called
            mock_health.assert_not_called()

    def test_clone_operation_temp_cache_not_found(self, tmp_path):
        """Test clone operation when temp cache snapshot not found."""
        target_dir = str(tmp_path / "workspace")
        model_spec = "test/model"

        # Create real temp cache
        real_temp_cache = tmp_path / "temp_cache"
        real_temp_cache.mkdir()
        sentinel = real_temp_cache / ".mlxk2_temp_cache_sentinel"
        sentinel.write_text("mlxk2_temp_cache_created_test")

        with patch('mlxk2.operations.clone._validate_apfs_filesystem'), \
             patch('mlxk2.operations.clone._create_temp_cache_same_volume') as mock_create_cache, \
             patch('mlxk2.operations.clone.pull_to_cache') as mock_pull, \
             patch('mlxk2.operations.clone._resolve_latest_snapshot') as mock_resolve:

            mock_create_cache.return_value = (real_temp_cache, True)  # (temp_cache, should_download)

            mock_pull.return_value = {"status": "success", "data": {"model": model_spec}}
            mock_resolve.return_value = None  # Snapshot not found

            result = clone_operation(model_spec, target_dir)

            assert result["status"] == "error"
            assert result["data"]["clone_status"] == "cache_not_found"
            assert "Temp cache snapshot not found" in result["error"]["message"]

            # ADR-018 Phase 0b: Temp cache is KEPT on failure with complete download
            assert real_temp_cache.exists()
            assert (real_temp_cache / ".mlxk2_download_complete").exists()

    def test_clone_operation_target_existing_empty(self, tmp_path):
        """Test clone operation with existing empty target directory."""
        target_dir = tmp_path / "workspace"
        target_dir.mkdir()  # Create empty directory

        # Create mock temp cache
        temp_cache = tmp_path / "temp_cache"
        temp_cache.mkdir()

        with patch('mlxk2.operations.clone._validate_apfs_filesystem'), \
             patch('mlxk2.operations.clone._create_temp_cache_same_volume') as mock_create_cache, \
             patch('mlxk2.operations.clone.pull_to_cache') as mock_pull, \
             patch('mlxk2.operations.clone._resolve_latest_snapshot') as mock_resolve, \
             patch('mlxk2.operations.health.health_from_cache') as mock_health, \
             patch('mlxk2.operations.clone._apfs_clone_directory') as mock_clone, \
             patch('mlxk2.operations.clone._cleanup_temp_cache_safe'):

            mock_create_cache.return_value = (temp_cache, True)  # (temp_cache, should_download)
            mock_pull.return_value = {"status": "success", "data": {"model": "test/model"}}
            mock_health.return_value = (True, "Model is healthy")

            # Mock snapshot resolution
            mock_snapshot = MagicMock()
            mock_snapshot.exists.return_value = True
            mock_resolve.return_value = mock_snapshot

            mock_clone.return_value = True

            result = clone_operation("test/model", str(target_dir))

            # Should succeed with empty directory
            assert result["status"] == "success"

    def test_clone_operation_unexpected_exception(self, tmp_path):
        """Test clone operation handles unexpected exceptions."""
        target_dir = str(tmp_path / "workspace")
        model_spec = "test/model"

        with patch('mlxk2.operations.clone._validate_apfs_filesystem') as mock_validate:
            mock_validate.side_effect = RuntimeError("Unexpected error")

            result = clone_operation(model_spec, target_dir)

            assert result["status"] == "error"
            assert result["data"]["clone_status"] == "error"
            assert result["error"]["type"] == "CloneOperationError"
            assert "Unexpected error" in result["error"]["message"]


class TestUnhealthyModelClone:
    """Test that unhealthy models can still be cloned (Session 59 fix).

    Critical for community repair workflow:
    - mlx-vlm #624 affected models have broken index.json
    - Users need to clone them to repair with `convert --repair-index`
    - Health check should be informational only, not blocking
    """

    @patch('mlxk2.operations.clone._validate_apfs_filesystem')
    @patch('mlxk2.operations.clone._create_temp_cache_same_volume')
    @patch('mlxk2.operations.clone.pull_to_cache')
    @patch('mlxk2.operations.clone._resolve_latest_snapshot')
    @patch('mlxk2.operations.health.health_from_cache')
    @patch('mlxk2.operations.clone._apfs_clone_directory')
    @patch('mlxk2.operations.clone._cleanup_temp_cache_safe')
    @patch('mlxk2.operations.clone.write_workspace_sentinel')
    def test_unhealthy_model_clone_succeeds(
        self, mock_sentinel, mock_cleanup, mock_clone, mock_health,
        mock_snapshot, mock_pull, mock_temp_cache, mock_validate_apfs,
        tmp_path
    ):
        """Test that unhealthy models are still cloned successfully."""
        model_spec = "mlx-community/broken-model"
        target_dir = str(tmp_path / "workspace")

        # Setup mocks
        temp_cache = tmp_path / "temp"
        temp_cache.mkdir()
        mock_temp_cache.return_value = (temp_cache, True)  # (temp_cache, should_download)

        mock_pull.return_value = {
            "status": "success",
            "data": {"model": model_spec, "commit_hash": "abc123"}
        }

        snapshot = temp_cache / "snapshot"
        snapshot.mkdir()
        mock_snapshot.return_value = snapshot

        # Health check returns UNHEALTHY
        mock_health.return_value = (False, "Missing weight shards: model-00001-of-00010.safetensors")

        mock_clone.return_value = True

        # Execute clone
        result = clone_operation(model_spec, target_dir, health_check=True)

        # Should succeed despite unhealthy status
        assert result["status"] == "success"
        assert result["data"]["clone_status"] == "success"

        # Health status should be recorded
        assert result["data"]["health"] == "unhealthy"
        assert "Missing weight shards" in result["data"]["health_reason"]

        # Clone should have been called (workspace created)
        mock_clone.assert_called_once()

        # Sentinel should have been written
        mock_sentinel.assert_called_once()

    @patch('mlxk2.operations.clone._validate_apfs_filesystem')
    @patch('mlxk2.operations.clone._create_temp_cache_same_volume')
    @patch('mlxk2.operations.clone.pull_to_cache')
    @patch('mlxk2.operations.clone._resolve_latest_snapshot')
    @patch('mlxk2.operations.health.health_from_cache')
    @patch('mlxk2.operations.clone._apfs_clone_directory')
    @patch('mlxk2.operations.clone._cleanup_temp_cache_safe')
    @patch('mlxk2.operations.clone.write_workspace_sentinel')
    def test_healthy_model_clone_records_status(
        self, mock_sentinel, mock_cleanup, mock_clone, mock_health,
        mock_snapshot, mock_pull, mock_temp_cache, mock_validate_apfs,
        tmp_path
    ):
        """Test that healthy models record health status correctly."""
        model_spec = "mlx-community/good-model"
        target_dir = str(tmp_path / "workspace")

        # Setup mocks
        temp_cache = tmp_path / "temp"
        temp_cache.mkdir()
        mock_temp_cache.return_value = (temp_cache, True)  # (temp_cache, should_download)

        mock_pull.return_value = {
            "status": "success",
            "data": {"model": model_spec, "commit_hash": "abc123"}
        }

        snapshot = temp_cache / "snapshot"
        snapshot.mkdir()
        mock_snapshot.return_value = snapshot

        # Health check returns HEALTHY
        mock_health.return_value = (True, "Multi-file model complete")

        mock_clone.return_value = True

        # Execute clone
        result = clone_operation(model_spec, target_dir, health_check=True)

        # Should succeed
        assert result["status"] == "success"
        assert result["data"]["clone_status"] == "success"

        # Health status should be recorded as healthy
        assert result["data"]["health"] == "healthy"
        assert result["data"]["health_reason"] == "Multi-file model complete"

    @patch('mlxk2.operations.clone._validate_apfs_filesystem')
    @patch('mlxk2.operations.clone._create_temp_cache_same_volume')
    @patch('mlxk2.operations.clone.pull_to_cache')
    @patch('mlxk2.operations.clone._resolve_latest_snapshot')
    @patch('mlxk2.operations.clone._apfs_clone_directory')
    @patch('mlxk2.operations.clone._cleanup_temp_cache_safe')
    @patch('mlxk2.operations.clone.write_workspace_sentinel')
    def test_no_health_check_skips_health_status(
        self, mock_sentinel, mock_cleanup, mock_clone,
        mock_snapshot, mock_pull, mock_temp_cache, mock_validate_apfs,
        tmp_path
    ):
        """Test that --no-health-check skips health status entirely."""
        model_spec = "mlx-community/model"
        target_dir = str(tmp_path / "workspace")

        # Setup mocks
        temp_cache = tmp_path / "temp"
        temp_cache.mkdir()
        mock_temp_cache.return_value = (temp_cache, True)  # (temp_cache, should_download)

        mock_pull.return_value = {
            "status": "success",
            "data": {"model": model_spec, "commit_hash": "abc123"}
        }

        snapshot = temp_cache / "snapshot"
        snapshot.mkdir()
        mock_snapshot.return_value = snapshot

        mock_clone.return_value = True

        # Execute clone with health_check=False
        result = clone_operation(model_spec, target_dir, health_check=False)

        # Should succeed
        assert result["status"] == "success"
        assert result["data"]["clone_status"] == "success"

        # Health status should NOT be in result
        assert "health" not in result["data"]
        assert "health_reason" not in result["data"]

class TestResumableClone:
    """Test suite for ADR-018 Phase 0b: Resumable Clone functionality."""

    def test_deterministic_temp_cache_name_consistency(self, tmp_path):
        """Test that temp cache name is deterministic for same inputs."""
        model_spec = "mlx-community/Llama-3.2-3B"
        target = tmp_path / "workspace"
        
        # Same inputs should produce same name
        name1 = _get_deterministic_temp_cache_name(model_spec, target)
        name2 = _get_deterministic_temp_cache_name(model_spec, target)
        
        assert name1 == name2
        assert name1.startswith(".mlxk2_temp_")
        assert len(name1) == 28  # ".mlxk2_temp_" + 16 hex chars

    def test_deterministic_temp_cache_name_different_inputs(self, tmp_path):
        """Test that different inputs produce different names."""
        target = tmp_path / "workspace"
        
        name1 = _get_deterministic_temp_cache_name("model-a", target)
        name2 = _get_deterministic_temp_cache_name("model-b", target)
        name3 = _get_deterministic_temp_cache_name("model-a", tmp_path / "other")
        
        assert name1 != name2  # Different model
        assert name1 != name3  # Different target
        assert name2 != name3

    def test_check_temp_cache_resume_no_existing(self, tmp_path):
        """Test resume check when temp cache doesn't exist."""
        temp_cache = tmp_path / "nonexistent"
        
        can_resume, reason, is_healthy = _check_temp_cache_resume(temp_cache, "model")
        
        assert can_resume is False
        assert "No existing download" in reason
        assert is_healthy is False

    def test_check_temp_cache_resume_incomplete_download(self, tmp_path):
        """Test resume check when download is incomplete (no marker) - should be resumable."""
        temp_cache = tmp_path / "temp"
        temp_cache.mkdir()

        # Create some files but no completion marker (simulates Ctrl-C)
        (temp_cache / "partial.bin").write_text("data")

        can_resume, reason, is_healthy = _check_temp_cache_resume(temp_cache, "model")

        # Partial downloads are resumable via HuggingFace snapshot_download
        assert can_resume is True
        assert "Partial download" in reason
        assert "resumable" in reason.lower()
        assert is_healthy is False  # Not complete yet

    def test_check_temp_cache_resume_complete_healthy(self, tmp_path):
        """Test resume check when download is complete and healthy."""
        temp_cache = tmp_path / "temp"
        temp_cache.mkdir()
        
        # Mark as complete
        (temp_cache / ".mlxk2_download_complete").write_text("completed")
        
        # Mock healthy check
        with patch('mlxk2.operations.health.health_from_cache', return_value=(True, "OK")):
            can_resume, reason, is_healthy = _check_temp_cache_resume(temp_cache, "model")
        
        assert can_resume is True
        assert "healthy" in reason.lower()
        assert is_healthy is True

    def test_check_temp_cache_resume_complete_unhealthy(self, tmp_path):
        """Test resume check when download is complete but unhealthy."""
        temp_cache = tmp_path / "temp"
        temp_cache.mkdir()
        
        # Mark as complete
        (temp_cache / ".mlxk2_download_complete").write_text("completed")
        
        # Mock unhealthy check
        with patch('mlxk2.operations.health.health_from_cache', return_value=(False, "Missing weights")):
            can_resume, reason, is_healthy = _check_temp_cache_resume(temp_cache, "model")
        
        assert can_resume is True
        assert "unhealthy" in reason.lower()
        assert is_healthy is False

    def test_create_temp_cache_new(self, tmp_path):
        """Test creating new temp cache when none exists."""
        target = tmp_path / "workspace"
        model_spec = "test/model"
        
        with patch('mlxk2.operations.clone._get_volume_mount_point', return_value=tmp_path):
            temp_cache, should_download = _create_temp_cache_same_volume(target, model_spec, False)
        
        assert temp_cache.exists()
        assert should_download is True
        assert (temp_cache / ".mlxk2_temp_cache_sentinel").exists()

    def test_create_temp_cache_resume_healthy(self, tmp_path):
        """Test resuming healthy existing temp cache."""
        target = tmp_path / "workspace"
        model_spec = "test/model"
        
        # Create existing temp cache with deterministic name
        with patch('mlxk2.operations.clone._get_volume_mount_point', return_value=tmp_path):
            temp_name = _get_deterministic_temp_cache_name(model_spec, target)
            temp_cache = tmp_path / temp_name
            temp_cache.mkdir()
            (temp_cache / ".mlxk2_download_complete").write_text("completed")
            
            # Mock healthy
            with patch('mlxk2.operations.health.health_from_cache', return_value=(True, "OK")):
                result_cache, should_download = _create_temp_cache_same_volume(target, model_spec, False)
        
        assert result_cache == temp_cache
        assert should_download is False  # Skip download - healthy resume

    def test_create_temp_cache_resume_unhealthy_force(self, tmp_path):
        """Test force resuming unhealthy temp cache."""
        target = tmp_path / "workspace"
        model_spec = "test/model"
        
        with patch('mlxk2.operations.clone._get_volume_mount_point', return_value=tmp_path):
            temp_name = _get_deterministic_temp_cache_name(model_spec, target)
            temp_cache = tmp_path / temp_name
            temp_cache.mkdir()
            (temp_cache / ".mlxk2_download_complete").write_text("completed")
            
            # Mock unhealthy
            with patch('mlxk2.operations.health.health_from_cache', return_value=(False, "Broken")):
                result_cache, should_download = _create_temp_cache_same_volume(target, model_spec, force_resume=True)
        
        assert result_cache == temp_cache
        assert should_download is False  # Skip download - forced resume

    def test_create_temp_cache_resume_unhealthy_no_force(self, tmp_path):
        """Test deleting unhealthy temp cache when not forcing."""
        target = tmp_path / "workspace"
        model_spec = "test/model"

        with patch('mlxk2.operations.clone._get_volume_mount_point', return_value=tmp_path):
            temp_name = _get_deterministic_temp_cache_name(model_spec, target)
            temp_cache = tmp_path / temp_name
            temp_cache.mkdir()
            (temp_cache / ".mlxk2_download_complete").write_text("completed")
            (temp_cache / "data.bin").write_text("old data")

            # Mock unhealthy
            with patch('mlxk2.operations.health.health_from_cache', return_value=(False, "Broken")):
                result_cache, should_download = _create_temp_cache_same_volume(target, model_spec, force_resume=False)

        # Should have deleted and recreated
        assert result_cache.exists()
        assert should_download is True
        assert not (result_cache / "data.bin").exists()  # Old data gone

    def test_create_temp_cache_resume_partial_download(self, tmp_path):
        """Test resuming partial download (no completion marker) - should call pull_to_cache."""
        target = tmp_path / "workspace"
        model_spec = "test/model"

        with patch('mlxk2.operations.clone._get_volume_mount_point', return_value=tmp_path):
            temp_name = _get_deterministic_temp_cache_name(model_spec, target)
            temp_cache = tmp_path / temp_name
            temp_cache.mkdir()
            # No completion marker - simulates Ctrl-C during download
            (temp_cache / "partial.bin").write_text("partial data")

            result_cache, should_download = _create_temp_cache_same_volume(target, model_spec, force_resume=False)

        # Should resume partial download via HuggingFace
        assert result_cache == temp_cache
        assert should_download is True  # Call pull_to_cache to resume
        assert (result_cache / "partial.bin").exists()  # Partial data preserved

    def test_keyboard_interrupt_preserves_temp_cache(self, tmp_path):
        """Test that KeyboardInterrupt (Ctrl-C) preserves temp cache for resume."""
        target = tmp_path / "workspace"
        model_spec = "test/model"

        with patch('mlxk2.operations.clone._validate_apfs_filesystem'):
            with patch('mlxk2.operations.clone._get_volume_mount_point', return_value=tmp_path):
                # Mock pull_to_cache to raise KeyboardInterrupt
                    with patch('mlxk2.operations.clone.pull_to_cache', side_effect=KeyboardInterrupt()):
                        result = clone_operation(model_spec, str(target))

        # Check result structure
        assert result["status"] == "error"
        assert result["error"]["type"] == "UserCancelledError"
        assert result["data"]["clone_status"] == "cancelled"

        # Check error message has resume instructions
        error_msg = result["error"]["message"]
        assert "Operation cancelled by user" in error_msg
        assert "Partial download preserved" in error_msg
        assert "To resume: Run the same command again" in error_msg

        # Verify temp cache exists and was NOT deleted
        temp_name = _get_deterministic_temp_cache_name(model_spec, target)
        temp_cache = tmp_path / temp_name
        assert temp_cache.exists(), "Temp cache should be preserved after Ctrl-C"
        assert (temp_cache / ".mlxk2_temp_cache_sentinel").exists()

    def test_keyboard_interrupt_early_no_temp_cache(self, tmp_path):
        """Test KeyboardInterrupt before temp cache creation."""
        target = tmp_path / "workspace"
        model_spec = "test/model"

        with patch('mlxk2.operations.clone._validate_apfs_filesystem'):
            # Raise KeyboardInterrupt during volume mount check
            with patch('mlxk2.operations.clone._get_volume_mount_point', side_effect=KeyboardInterrupt()):
                    result = clone_operation(model_spec, str(target))

        # Should handle gracefully even without temp cache
        assert result["status"] == "error"
        assert result["error"]["type"] == "UserCancelledError"
        assert result["data"]["clone_status"] == "cancelled"

        # Error message should be simple (no temp cache to preserve)
        error_msg = result["error"]["message"]
        assert error_msg == "Operation cancelled by user."
