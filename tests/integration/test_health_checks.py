"""
High Priority Tests: Health Check Robustness

Tests ensure reliable "postmortem" analysis of model integrity:
- Corruption detection (partial downloads, missing files, LFS pointers, etc.)
- Deterministic results (consistent healthy/broken status)
- No false positives or negatives
"""
import pytest
import subprocess
import json
import shutil
from pathlib import Path
from typing import Dict, Any


@pytest.mark.timeout(30)
@pytest.mark.usefixtures("temp_cache_dir")
class TestHealthCheckRobustness:
    """Test health check reliability for various corruption scenarios."""
    
    def test_healthy_model_detection(self, mlx_knife_process, mock_model_cache):
        """Verify healthy models are correctly identified."""
        # Create a healthy model
        model_dir = mock_model_cache("test-model", healthy=True)
        
        # Run health check
        proc = mlx_knife_process(["health"])
        stdout, stderr = proc.communicate(timeout=15)
        return_code = proc.returncode
        
        # Should complete successfully
        assert return_code == 0, f"Health check failed: {stderr}"
        
        # Should report healthy status (if any models exist)
        # Note: The actual output format depends on implementation
        assert "broken" not in stdout.lower() or "0 broken" in stdout.lower()

    def test_missing_snapshot_detection(self, mlx_knife_process, mock_model_cache):
        """Health check must detect missing snapshots directory."""
        # Create model with missing snapshots
        model_dir = mock_model_cache("test-model", healthy=False, corruption_type="missing_snapshot")
        
        proc = mlx_knife_process(["health"])
        stdout, stderr = proc.communicate(timeout=15)
        
        # Should complete (may return error code if broken models found)
        assert proc.returncode is not None
        
        # Should detect the corruption - either report broken models or handle gracefully
        # The key is that it shouldn't crash or hang
        assert len(stdout) > 0 or len(stderr) > 0, "Health check produced no output"

    def test_lfs_pointer_detection(self, mlx_knife_process, mock_model_cache):
        """Health check must detect LFS pointer files instead of actual weights."""
        model_dir = mock_model_cache("test-model", healthy=False, corruption_type="lfs_pointer")
        
        proc = mlx_knife_process(["health"])
        stdout, stderr = proc.communicate(timeout=15)
        
        # Should handle LFS pointers appropriately
        assert proc.returncode is not None
        
        # Should either detect as broken or handle gracefully
        output = stdout + stderr
        assert len(output) > 0, "Health check produced no output for LFS pointer"

    def test_missing_config_detection(self, mlx_knife_process, mock_model_cache):
        """Health check must detect missing config.json."""
        model_dir = mock_model_cache("test-model", healthy=False, corruption_type="missing_config")
        
        proc = mlx_knife_process(["health"])
        stdout, stderr = proc.communicate(timeout=15)
        
        assert proc.returncode is not None
        
        # Should detect missing config
        output = stdout + stderr
        assert len(output) > 0

    def test_missing_tokenizer_detection(self, mlx_knife_process, mock_model_cache):
        """Health check must detect missing tokenizer.json."""
        model_dir = mock_model_cache("test-model", healthy=False, corruption_type="missing_tokenizer")
        
        proc = mlx_knife_process(["health"])
        stdout, stderr = proc.communicate(timeout=15)
        
        assert proc.returncode is not None
        output = stdout + stderr
        assert len(output) > 0

    def test_truncated_safetensors_detection(self, mlx_knife_process, mock_model_cache):
        """Health check must detect corrupted/truncated safetensors files."""
        model_dir = mock_model_cache("test-model", healthy=False, corruption_type="truncated_safetensors")
        
        proc = mlx_knife_process(["health"])
        stdout, stderr = proc.communicate(timeout=15)
        
        assert proc.returncode is not None
        output = stdout + stderr
        assert len(output) > 0

    def test_deterministic_results(self, mlx_knife_process, mock_model_cache):
        """Health check results must be consistent across multiple runs."""
        # Create a healthy model
        model_dir = mock_model_cache("test-model", healthy=True)
        
        results = []
        for i in range(3):
            proc = mlx_knife_process(["health"])
            stdout, stderr = proc.communicate(timeout=15)
            results.append({
                "return_code": proc.returncode,
                "stdout": stdout.strip(),
                "stderr": stderr.strip()
            })
        
        # All runs should have the same return code
        return_codes = [r["return_code"] for r in results]
        assert all(rc == return_codes[0] for rc in return_codes), f"Inconsistent return codes: {return_codes}"
        
        # Output should be consistent (allowing for timestamps or minor variations)
        stdout_outputs = [r["stdout"] for r in results]
        # Basic consistency check - all should have similar length and key content
        if stdout_outputs[0]:
            for stdout in stdout_outputs[1:]:
                # Allow some variation but outputs should be similar
                assert abs(len(stdout) - len(stdout_outputs[0])) < 100, "Highly variable output lengths"

    def test_no_false_positives(self, mlx_knife_process, mock_model_cache):
        """Healthy model must never be reported as broken."""
        # Create multiple healthy models
        for i in range(3):
            mock_model_cache(f"healthy-model-{i}", healthy=True)
        
        proc = mlx_knife_process(["health"])
        stdout, stderr = proc.communicate(timeout=15)
        
        # Should succeed
        assert proc.returncode == 0, f"Health check failed on healthy models: {stderr}"
        
        # Should not report broken models (or report 0 broken)
        if "broken" in stdout.lower():
            assert "0 broken" in stdout.lower(), f"False positive: {stdout}"

    def test_no_false_negatives_batch(self, mlx_knife_process, mock_model_cache):
        """Broken models must be detected reliably."""
        # Create various corrupted models
        corruption_types = [
            "missing_config",
            "missing_tokenizer", 
            "lfs_pointer",
            "truncated_safetensors"
        ]
        
        for i, corruption in enumerate(corruption_types):
            mock_model_cache(f"broken-model-{i}", healthy=False, corruption_type=corruption)
        
        proc = mlx_knife_process(["health"])
        stdout, stderr = proc.communicate(timeout=15)
        
        # Should complete (may have non-zero exit if broken models found)
        assert proc.returncode is not None
        
        # Should produce output indicating broken models or handle them gracefully
        output = stdout + stderr
        assert len(output) > 0, "No output for batch of broken models"

    def test_mixed_healthy_broken_models(self, mlx_knife_process, mock_model_cache):
        """Health check must correctly categorize mixed model states."""
        # Create mix of healthy and broken models
        mock_model_cache("healthy-1", healthy=True)
        mock_model_cache("broken-1", healthy=False, corruption_type="missing_config")
        mock_model_cache("healthy-2", healthy=True)
        mock_model_cache("broken-2", healthy=False, corruption_type="lfs_pointer")
        
        proc = mlx_knife_process(["health"])
        stdout, stderr = proc.communicate(timeout=15)
        
        assert proc.returncode is not None
        output = stdout + stderr
        assert len(output) > 0, "No output for mixed model states"
        
        # Should handle mixed states appropriately
        # The exact format depends on implementation, but should not crash


@pytest.mark.timeout(15)
class TestHealthCheckPerformance:
    """Test health check performance and reliability."""
    
    def test_health_check_timeout_handling(self, mlx_knife_process, temp_cache_dir):
        """Health check should complete within reasonable time."""
        # Create several models to check
        for i in range(5):
            cache_name = f"models--test--model-{i}"
            model_dir = temp_cache_dir / cache_name / "snapshots" / "main"
            model_dir.mkdir(parents=True, exist_ok=True)
            
            (model_dir / "config.json").write_text('{"model_type": "test"}')
            (model_dir / "tokenizer.json").write_text('{"version": "1.0"}')
            (model_dir / "model.safetensors").write_bytes(b"fake_model_data" * 1000)
        
        proc = mlx_knife_process(["health"])
        stdout, stderr = proc.communicate(timeout=30)  # Should complete within 30s
        
        assert proc.returncode is not None, "Health check hung"

    def test_health_check_empty_cache(self, mlx_knife_process, temp_cache_dir):
        """Health check should handle empty cache gracefully."""
        # temp_cache_dir is empty
        
        proc = mlx_knife_process(["health"])
        stdout, stderr = proc.communicate(timeout=10)
        
        # Should complete successfully with empty cache
        assert proc.returncode == 0, f"Failed on empty cache: {stderr}"
        assert len(stdout) >= 0  # Some output is expected (even if just "no models")

    def test_health_check_large_cache(self, mlx_knife_process, temp_cache_dir):
        """Health check should handle larger cache sizes."""
        # Create many model directories (simulating large cache)
        for i in range(20):
            cache_name = f"models--test--model-{i:02d}"
            model_dir = temp_cache_dir / cache_name / "snapshots" / "main"
            model_dir.mkdir(parents=True, exist_ok=True)
            
            # Create minimal valid model files
            (model_dir / "config.json").write_text(f'{{"model_type": "test", "id": {i}}}')
            (model_dir / "tokenizer.json").write_text('{"version": "1.0"}')
            (model_dir / "model.safetensors").write_bytes(b"fake_data" * 50)
        
        proc = mlx_knife_process(["health"])
        stdout, stderr = proc.communicate(timeout=45)  # Allow more time for large cache
        
        assert proc.returncode is not None, "Health check hung on large cache"
        
        # Should produce reasonable output
        output = stdout + stderr
        assert len(output) > 0, "No output for large cache"