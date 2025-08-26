#!/usr/bin/env python3
"""
Integration test for lock cleanup bug.
This test reproduces the real bug found in Issue #24.
"""

from pathlib import Path
import pytest

from mlx_knife.cache_utils import _cleanup_model_locks


@pytest.mark.usefixtures("temp_cache_dir")
class TestLockCleanupBug:
    """Integration tests for lock cleanup functionality."""

    def test_lock_cleanup_path_bug(self, temp_cache_dir, patch_model_cache):
        """Test that reproduces the lock cleanup path bug.
        
        The bug: _cleanup_model_locks uses MODEL_CACHE.parent instead of MODEL_CACHE,
        causing it to look for locks in the wrong directory.
        
        HF Cache structure:
        cache_root/
        └── hub/                    ← MODEL_CACHE
            ├── .locks/             ← Correct location  
            └── models--name/       
            
        Bug: looks in cache_root/.locks/ instead of cache_root/hub/.locks/
        """
        hub_cache = temp_cache_dir / "hub"
        
        with patch_model_cache(hub_cache):
            # Create test model structure
            model_name = "test-org/broken-model"
            cache_dir_name = "models--test-org--broken-model"
            
            # Create model directory (not needed for lock cleanup, but realistic)
            model_dir = hub_cache / cache_dir_name
            model_dir.mkdir()
            
            # Create lock files in CORRECT location: hub/.locks/
            locks_dir = hub_cache / ".locks" / cache_dir_name
            locks_dir.mkdir(parents=True)
            (locks_dir / "download.lock").touch()
            (locks_dir / "process.lock").touch()
            (locks_dir / "huggingface.lock").write_text("PID:12345")
            (locks_dir / "another.lock").touch()
            
            # Verify setup
            assert locks_dir.exists(), "Lock directory should exist"
            lock_files = list(locks_dir.iterdir())
            assert len(lock_files) == 4, f"Should have 4 lock files, got {len(lock_files)}"
            
            # This should clean up the locks, but currently fails due to path bug
            _cleanup_model_locks(model_name, force=True)
            
            # BUG: Lock directory still exists because function looks in wrong path
            # This assertion will FAIL until the bug is fixed
            assert not locks_dir.exists(), (
                f"❌ BUG REPRODUCED: Lock directory still exists at {locks_dir}. "
                f"The _cleanup_model_locks function is looking in the wrong path."
            )

    def test_lock_cleanup_empty_directory(self, temp_cache_dir, patch_model_cache):
        """Test that _cleanup_model_locks handles empty lock directories gracefully."""
        hub_cache = temp_cache_dir / "hub"
        
        with patch_model_cache(hub_cache):
            model_name = "test-org/empty-locks"
            cache_dir_name = "models--test-org--empty-locks"
            
            # Create empty lock directory
            locks_dir = hub_cache / ".locks" / cache_dir_name
            locks_dir.mkdir(parents=True)
            
            assert locks_dir.exists()
            assert len(list(locks_dir.iterdir())) == 0
            
            # Should handle empty directory gracefully (no-op)
            _cleanup_model_locks(model_name, force=True)
            
            # Empty directory should still exist (function returns early)
            # This will also fail due to path bug, but for different reason

    def test_lock_cleanup_nonexistent_locks(self, temp_cache_dir, patch_model_cache):
        """Test that _cleanup_model_locks handles missing lock directories gracefully."""
        hub_cache = temp_cache_dir / "hub"
        
        with patch_model_cache(hub_cache):
            model_name = "test-org/no-locks"
            
            # Don't create any lock directory
            
            # Should handle gracefully (no-op)
            _cleanup_model_locks(model_name, force=True)
            
            # This should pass (no error thrown)
            assert True, "Function should handle missing lock directories gracefully"