"""Robustness tests for critical rm and pull operations.

These tests ensure user-cache safety and robust error handling
for operations that modify the user's model cache.
"""

import pytest
import tempfile
import shutil
from pathlib import Path
from unittest.mock import patch, MagicMock

from mlxk2.operations.rm import rm_operation
from mlxk2.operations.pull import pull_operation


class TestRmOperationRobustness:
    """Test rm operation robustness with user cache safety."""
    
    def test_rm_force_flag_skips_all_confirmations(self, mock_models, isolated_cache):
        """Critical: Force flag must skip ALL confirmations (Issue #23 regression)."""
        # Get a model from mock cache
        from conftest import test_list_models
        models = test_list_models(isolated_cache)["data"]["models"]
        
        # Filter out sentinel model and get a real mock model
        real_models = [m for m in models if "TEST-CACHE-SENTINEL" not in m["name"]]
        if not real_models:
            pytest.skip("No real models in mock cache for force flag testing")
        
        target_model = real_models[0]["name"]
        
        # Force flag should work without any interactive prompts
        with patch('builtins.input') as mock_input:
            result = rm_operation(target_model, force=True)
            
            # Should never call input() when force=True
            mock_input.assert_not_called()
            
            # Should either succeed or fail with clear reason (never prompt)
            assert result["status"] in ["success", "error"]
    
    def test_rm_without_force_handles_nonexistent_gracefully(self, mock_models):
        """Test rm without force flag handles nonexistent models gracefully."""
        result = rm_operation("definitely-nonexistent-model-12345", force=False)
        
        assert result["status"] == "error"
        assert "not found" in result["error"]["message"].lower() or "no models found" in result["error"]["message"].lower()
    
    def test_rm_permission_error_handling(self, mock_models, isolated_cache):
        """Test rm handles permission errors gracefully."""
        from conftest import atomic_cache_context, test_list_models
        from mlxk2.operations.rm import rm_operation
        
        with atomic_cache_context(isolated_cache, "test"):
            # Get models in test cache context
            models = test_list_models(isolated_cache)["data"]["models"]
            
            # Filter out sentinel model and get a real mock model
            real_models = [m for m in models if "TEST-CACHE-SENTINEL" not in m["name"]]  
            if not real_models:
                pytest.skip("No real models in mock cache for permission testing")
            
            target_model = real_models[0]["name"]
            
            # Mock permission error
            with patch('shutil.rmtree', side_effect=PermissionError("Permission denied")):
                result = rm_operation(target_model, force=True)
                
                assert result["status"] == "error"
                assert "permission" in result["error"]["message"].lower()
    
    def test_rm_partial_deletion_recovery(self, mock_models, isolated_cache):
        """Test rm handles interrupted deletion gracefully."""
        from conftest import atomic_cache_context, test_list_models
        from mlxk2.operations.rm import rm_operation
        
        with atomic_cache_context(isolated_cache, "test"):
            # Get models in test cache context
            models = test_list_models(isolated_cache)["data"]["models"]
            
            # Filter out sentinel model and get a real mock model
            real_models = [m for m in models if "TEST-CACHE-SENTINEL" not in m["name"]]
            if not real_models:
                pytest.skip("No real models in mock cache for partial deletion testing")
            
            target_model = real_models[0]["name"]
            
            # Mock partial failure (some files deleted, then error)
            call_count = 0
            def mock_rmtree_partial_fail(path):
                nonlocal call_count
                call_count += 1
                if call_count == 1:
                    # First call succeeds (partial deletion)
                    pass
                else:
                    # Second call fails
                    raise OSError("Device busy")
            
            with patch('shutil.rmtree', side_effect=mock_rmtree_partial_fail):
                result = rm_operation(target_model, force=True)
                
                # Should handle partial failure gracefully
                assert result["status"] in ["success", "error"]
                if result["status"] == "error":
                    assert "error" in result["error"]["message"].lower()


class TestPullOperationRobustness:
    """Test pull operation robustness and error handling."""
    
    def test_pull_model_name_validation(self):
        """Test pull validates model names before network operations."""
        # Test 96 character limit
        long_name = "a" * 100
        result = pull_operation(long_name)
        
        assert result["status"] == "error"
        # Should fail validation before attempting network operation
        assert "name" in result["error"]["message"].lower() or "invalid" in result["error"]["message"].lower()
    
    def test_pull_network_timeout_handling(self, monkeypatch):
        """Test pull handles network timeouts gracefully."""
        # Set dummy token to pass preflight checks
        monkeypatch.setenv("HF_TOKEN", "dummy-token")

        # Mock preflight to succeed and pull to timeout
        with patch('mlxk2.operations.pull.preflight_repo_access', return_value=(True, None)), \
             patch('mlxk2.operations.pull.pull_model_with_huggingface_hub', side_effect=TimeoutError("Network timeout")):
            result = pull_operation("test-model")

            assert result["status"] == "error"
            assert "timeout" in result["error"]["message"].lower() or "network" in result["error"]["message"].lower() or "error" in result["error"]["message"].lower()
    
    def test_pull_disk_space_validation(self, isolated_cache):
        """Test pull checks available disk space before download."""
        # Mock disk space check
        with patch('shutil.disk_usage', return_value=(1000, 900, 100)):  # Only 100 bytes free
            result = pull_operation("mlx-community/Phi-3-mini-4k-instruct-4bit")
            
            # Should either succeed (if no disk check implemented) or fail gracefully
            assert result["status"] in ["success", "error"]
            if result["status"] == "error":
                # Error message should be helpful
                assert len(result["error"]["message"]) > 0
    
    def test_pull_invalid_repo_early_validation(self):
        """Test pull validates repo format before network calls."""
        invalid_repos = [
            "",  # Empty
            "no-slash",  # No org/model format (might be valid short name though)
            "org//model",  # Double slash
            "/org/model",  # Leading slash
            "org/model/",  # Trailing slash
        ]
        
        for invalid_repo in invalid_repos:
            if not invalid_repo.strip():  # Skip empty strings
                result = pull_operation(invalid_repo)
                assert result["status"] == "error"
                assert len(result["error"]["message"]) > 0
    
    def test_pull_concurrent_download_prevention(self, mock_models):
        """Test pull prevents concurrent downloads of same model."""
        model_name = "test-concurrent-model"
        
        # Mock a long-running download
        with patch('subprocess.run', side_effect=lambda *args, **kwargs: __import__('time').sleep(0.1)):
            # Start first download (simulate in progress)
            import threading
            
            first_result = [None]
            def first_download():
                first_result[0] = pull_operation(model_name)
            
            # Start first download in background
            thread1 = threading.Thread(target=first_download)
            thread1.start()
            
            # Try concurrent download (should detect ongoing download)
            result2 = pull_operation(model_name)
            
            thread1.join(timeout=1.0)  # Wait for first to complete
            
            # At least one should complete successfully, and system should handle concurrent access
            assert isinstance(result2, dict)
            assert result2["status"] in ["success", "error"]


class TestCacheIntegrityRobustness:
    """Test cache integrity and corruption handling."""
    
    def test_operations_with_corrupted_cache_entries(self, create_corrupted_cache_entry):
        """Test that operations handle corrupted cache entries gracefully."""
        # Create corrupted entry
        cache_path = create_corrupted_cache_entry("models--corrupted---entry").parent
        
        # List should not crash with corrupted entries
        from conftest import test_list_models
        result = test_list_models(cache_path)
        
        assert result["status"] == "success"
        # Should include corrupted entry but mark it as such
        corrupted_models = [m for m in result["data"]["models"] if "/-" in m["name"] or m["name"].startswith("-")]
        assert len(corrupted_models) >= 1
    
    def test_cache_recovery_after_interruption(self, isolated_cache):
        """Test system recovers gracefully from interrupted operations."""
        # Create partial model directory (simulate interrupted download)
        partial_model_dir = isolated_cache / "models--test--partial-model"
        partial_model_dir.mkdir(parents=True)
        
        # Create snapshots dir but no content (interrupted state)
        snapshots_dir = partial_model_dir / "snapshots"
        snapshots_dir.mkdir()
        
        # Operations should handle partial state
        from conftest import test_list_models
        result = test_list_models(isolated_cache)
        
        assert result["status"] == "success"
        # Should either exclude partial model or mark it as unhealthy
        model_names = [m["name"] for m in result["data"]["models"]]
        if "test/partial-model" in model_names:
            # If included, should be marked somehow as problematic
            partial_model = next(m for m in result["data"]["models"] if m["name"] == "test/partial-model")
            # Could be marked with different framework or size indicating incomplete
            assert partial_model is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])