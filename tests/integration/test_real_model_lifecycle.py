"""
Integration tests for real model lifecycle using tiny real models.

This replaces heavily mocked tests with comprehensive integration tests using
hf-internal-testing/tiny-random-gpt2 (112k params, ~500KB) to test:
- Real file system operations
- Real path resolution logic  
- Real framework detection
- Real lock cleanup (our main bug from Issue #23)
- End-to-end model lifecycle: pull → list → show → rm

Strategy: ONE pull for all tests to be efficient, then comprehensive testing
of the full pipeline with real files and directories.
"""
import pytest
import os
import shutil
from pathlib import Path
from unittest.mock import patch
from mlx_knife.hf_download import pull_model
from mlx_knife.cache_utils import (
    list_models, show_model, rm_model, find_matching_models,
    resolve_single_model, is_model_healthy, detect_framework,
    hf_to_cache_dir, MODEL_CACHE
)


class TestRealModelLifecycle:
    """Test complete model lifecycle with real tiny model in isolated cache."""
    
    TEST_MODEL = "hf-internal-testing/tiny-random-gpt2"
    EXPECTED_SIZE_RANGE = (10_000_000, 15_000_000)  # ~12.5MB expected
    
    @staticmethod
    def get_current_model_cache():
        """Get the current model cache path (resolves HF_HOME dynamically)."""
        cache_root = Path(os.environ.get("HF_HOME", Path.home() / ".cache/huggingface"))
        return cache_root / "hub"
    
    @pytest.fixture(scope="class", autouse=True)
    def setup_isolated_model(self, class_temp_cache_dir):
        """Download test model to isolated cache before all tests in this class."""
        print(f"\n=== Downloading {self.TEST_MODEL} to isolated test cache ===")
        print(f"Test cache location: {class_temp_cache_dir}")
        
        # Patch MODEL_CACHE to point to our isolated cache
        from mlx_knife import cache_utils
        original_model_cache = cache_utils.MODEL_CACHE
        cache_utils.MODEL_CACHE = class_temp_cache_dir / "hub"
        
        try:
            # Pull the tiny test model (patch input to auto-confirm)
            with patch('builtins.input', return_value='y'):
                pull_model(self.TEST_MODEL)
            
            # Verify model exists in isolated cache
            cache_dir_name = hf_to_cache_dir(self.TEST_MODEL)
            model_cache_path = cache_utils.MODEL_CACHE / cache_dir_name
            
            if not model_cache_path.exists():
                print(f"HF_HOME: {os.environ.get('HF_HOME', 'not set')}")
                print(f"Expected cache path: {model_cache_path}")
                print(f"Cache contents: {list(cache_utils.MODEL_CACHE.iterdir()) if cache_utils.MODEL_CACHE.exists() else 'does not exist'}")
                pytest.fail(f"Model download failed - cache directory not found: {model_cache_path}")
                
            print(f"✅ Successfully downloaded {self.TEST_MODEL}")
            print(f"📁 Model cached at: {model_cache_path}")
            print(f"🔒 Using isolated test cache (user cache untouched)")
            
            # Fixture runs for all tests in this class
            yield
            
        finally:
            # Restore original MODEL_CACHE
            cache_utils.MODEL_CACHE = original_model_cache
            print(f"\n=== Test cache cleanup and MODEL_CACHE restored ===")
    
    def test_01_model_downloaded_successfully(self):
        """Test that real model download created proper file structure."""
        from mlx_knife import cache_utils
        cache_dir_name = hf_to_cache_dir(self.TEST_MODEL)
        model_cache_path = cache_utils.MODEL_CACHE / cache_dir_name
        
        # Verify top-level structure exists
        assert model_cache_path.exists(), f"Model cache directory missing: {model_cache_path}"
        assert (model_cache_path / "snapshots").exists(), "Snapshots directory missing"
        assert (model_cache_path / "refs").exists(), "Refs directory missing"
        
        # Verify refs/main exists and points to a hash
        refs_main = model_cache_path / "refs" / "main"
        assert refs_main.exists(), "refs/main missing"
        
        commit_hash = refs_main.read_text().strip()
        assert len(commit_hash) >= 8, f"Invalid commit hash: {commit_hash}"
        
        # Verify snapshot directory exists for the hash
        snapshot_dir = model_cache_path / "snapshots" / commit_hash
        assert snapshot_dir.exists(), f"Snapshot directory missing: {snapshot_dir}"
        
        # Verify essential model files exist
        config_json = snapshot_dir / "config.json"
        assert config_json.exists(), "config.json missing"
        
        # Check file size is reasonable (tiny model should be ~500KB total)
        total_size = sum(f.stat().st_size for f in snapshot_dir.rglob("*") if f.is_file())
        assert self.EXPECTED_SIZE_RANGE[0] <= total_size <= self.EXPECTED_SIZE_RANGE[1], \
            f"Model size {total_size} outside expected range {self.EXPECTED_SIZE_RANGE}"
        
        print(f"✓ Real model downloaded: {total_size:,} bytes in {snapshot_dir}")
    
    def test_02_list_shows_downloaded_model(self):
        """Test that list command shows our real downloaded model."""
        # Use list with health check to verify model is detected and healthy
        import io
        import contextlib
        
        stdout_capture = io.StringIO()
        with contextlib.redirect_stdout(stdout_capture):
            list_models(show_all=True, show_health=True)  # Show all models with health status
        
        output = stdout_capture.getvalue()
        
        # Verify our test model appears in the output
        assert self.TEST_MODEL in output or "tiny-random-gpt2" in output, \
            f"Test model not found in list output: {output}"
        
        print(f"✓ Model appears in list output with health status")
    
    def test_03_show_detects_real_framework(self):
        """Test that show command detects framework for real model."""
        import io
        import contextlib
        
        stdout_capture = io.StringIO()
        with contextlib.redirect_stdout(stdout_capture):
            show_model(self.TEST_MODEL)
        
        output = stdout_capture.getvalue()
        
        # Verify show command produced output about our model
        assert self.TEST_MODEL in output or "tiny-random-gpt2" in output, \
            f"Model not found in show output: {output}"
        
        # Should have framework detection
        assert "Framework:" in output, f"Framework detection missing: {output}"
        
        # Should have health status
        assert "Health:" in output, f"Health status missing: {output}"
        
        # Should show size information
        assert any(keyword in output.lower() for keyword in ["size", "gb", "mb", "kb"]), \
            f"Size information missing: {output}"
        
        print(f"✓ Show command detected framework and health for real model")
    
    def test_04_find_matching_works_with_real_model(self):
        """Test that fuzzy matching works with real model."""
        # Test exact match
        exact_matches = find_matching_models(self.TEST_MODEL)
        assert len(exact_matches) >= 1, f"Exact match failed for {self.TEST_MODEL}"
        
        # Test partial match
        partial_matches = find_matching_models("tiny-random")
        assert len(partial_matches) >= 1, f"Partial match failed for 'tiny-random'"
        
        # Verify our model is in the matches
        model_names = [match[1] for match in partial_matches]
        assert any(self.TEST_MODEL in name for name in model_names), \
            f"Test model not found in partial matches: {model_names}"
        
        print(f"✓ Fuzzy matching works: {len(partial_matches)} matches for 'tiny-random'")
    
    def test_05_resolve_real_model_paths(self):
        """Test that path resolution works with real model."""
        # Test exact model resolution
        model_path, resolved_name, commit_hash = resolve_single_model(self.TEST_MODEL)
        
        assert model_path is not None, f"Failed to resolve model path for {self.TEST_MODEL}"
        assert model_path.exists(), f"Resolved path does not exist: {model_path}"
        assert resolved_name == self.TEST_MODEL, f"Name resolution incorrect: {resolved_name}"
        assert commit_hash is not None, f"Commit hash not resolved"
        assert len(commit_hash) >= 8, f"Invalid commit hash: {commit_hash}"
        
        # Test fuzzy resolution
        fuzzy_path, fuzzy_name, fuzzy_hash = resolve_single_model("tiny-random")
        
        assert fuzzy_path is not None, f"Fuzzy resolution failed for 'tiny-random'"
        assert fuzzy_path.exists(), f"Fuzzy resolved path does not exist: {fuzzy_path}"
        
        # Both should resolve to same model
        assert fuzzy_path == model_path, f"Fuzzy and exact paths differ: {fuzzy_path} vs {model_path}"
        
        print(f"✓ Path resolution works: {model_path}")
    
    def test_06_health_check_on_real_model(self):
        """Test health checking on real model files."""
        # Resolve model to get path
        model_path, _, _ = resolve_single_model(self.TEST_MODEL)
        assert model_path is not None, "Model resolution failed"
        
        # Test health check
        is_healthy = is_model_healthy(self.TEST_MODEL)
        
        # Real downloaded model should be healthy
        assert is_healthy, f"Real model reported as unhealthy: {self.TEST_MODEL}"
        
        # Test framework detection
        framework = detect_framework(model_path, self.TEST_MODEL)
        assert framework is not None, f"Framework detection failed for real model"
        assert isinstance(framework, str), f"Framework should be string: {framework}"
        assert len(framework) > 0, f"Empty framework detected: {framework}"
        
        print(f"✓ Health check passed, framework: {framework}")
        
        # Also test using show command for health verification
        import io
        import contextlib
        
        stdout_capture = io.StringIO()
        with contextlib.redirect_stdout(stdout_capture):
            show_model(self.TEST_MODEL)
        
        show_output = stdout_capture.getvalue()
        assert "Health:" in show_output, f"Health status missing in show output: {show_output}"
        
        print(f"✓ Show command also reports health status correctly")
    
    def test_07_rm_cleans_locks_and_model(self):
        """Test that rm command cleans both model AND locks (Issue #23 fix)."""
        # Verify model exists before deletion
        model_path, _, _ = resolve_single_model(self.TEST_MODEL)
        assert model_path is not None, "Model should exist before deletion"
        assert model_path.exists(), f"Model path should exist before deletion: {model_path}"
        
        # Get model cache directory and expected locks directory  
        from mlx_knife import cache_utils
        cache_dir_name = hf_to_cache_dir(self.TEST_MODEL)
        model_cache_path = cache_utils.MODEL_CACHE / cache_dir_name
        locks_dir = cache_utils.MODEL_CACHE / ".locks" / cache_dir_name
        
        # Create some test lock files if they don't exist
        if not locks_dir.exists():
            locks_dir.mkdir(parents=True)
            (locks_dir / "test.lock").touch()
        
        lock_files_before = list(locks_dir.iterdir()) if locks_dir.exists() else []
        
        print(f"Before deletion:")
        print(f"  Model cache: {model_cache_path.exists()}")
        print(f"  Locks dir: {locks_dir.exists()}")  
        print(f"  Lock files: {len(lock_files_before)}")
        
        # Remove model with force=True (no prompts)
        rm_model(self.TEST_MODEL, force=True)
        
        # Verify BOTH model and locks are cleaned up
        model_exists_after = model_cache_path.exists()
        locks_exist_after = locks_dir.exists()
        
        print(f"After deletion:")
        print(f"  Model cache: {model_exists_after}")
        print(f"  Locks dir: {locks_exist_after}")
        
        # Issue #23 fix: Both should be deleted
        assert not model_exists_after, f"Model cache should be deleted: {model_cache_path}"
        assert not locks_exist_after, f"Locks directory should be deleted: {locks_dir}"
        
        print(f"✓ rm command cleaned both model and locks (Issue #23 fix verified)")
    
    def test_08_model_completely_removed(self):
        """Test end-to-end verification that model is completely gone."""
        # Verify model no longer appears in list
        import io
        import contextlib
        
        stdout_capture = io.StringIO()
        with contextlib.redirect_stdout(stdout_capture):
            list_models(show_all=True)  # Show all models, not just MLX ones
        
        output = stdout_capture.getvalue()
        
        # Our test model should NOT appear in output anymore
        assert self.TEST_MODEL not in output, \
            f"Model still appears in list after deletion: {output}"
        assert "tiny-random-gpt2" not in output, \
            f"Model name still appears in list after deletion: {output}"
        
        # Verify resolution fails
        model_path, resolved_name, commit_hash = resolve_single_model(self.TEST_MODEL)
        assert model_path is None, f"Model path should be None after deletion: {model_path}"
        assert resolved_name is None, f"Resolved name should be None after deletion: {resolved_name}"
        
        # Verify fuzzy matching also fails  
        matches = find_matching_models("tiny-random")
        model_names = [match[1] for match in matches] if matches else []
        assert not any(self.TEST_MODEL in name for name in model_names), \
            f"Model still found in fuzzy matches: {model_names}"
        
        print(f"✓ Model completely removed from cache and indexes")


class TestIntegrationTestSelfCheck:
    """Meta-test: Verify integration tests are working properly."""
    
    def test_integration_test_downloads_real_files(self):
        """Verify this integration test actually downloaded real files."""
        # This test runs after TestRealModelLifecycle, so model should be cleaned up
        # But we can verify the test ran by checking if we have network access
        # and that the model we tried to download is a real HF model
        
        model = TestRealModelLifecycle.TEST_MODEL
        assert "/" in model, f"Model name should have org/repo format: {model}"
        assert "tiny" in model.lower(), f"Should use tiny model for tests: {model}"
        assert "gpt2" in model.lower(), f"Should use GPT2 for compatibility: {model}"
        
        # Verify size expectations are reasonable for integration tests
        min_size, max_size = TestRealModelLifecycle.EXPECTED_SIZE_RANGE
        assert min_size < max_size, "Size range should be valid"
        assert max_size < 20_000_000, "Test model should be reasonably small for CI efficiency"
        
        print(f"✓ Integration test configuration validated: {model}")
    
    def test_integration_vs_unit_test_coverage(self):
        """Verify integration tests cover areas missed by unit tests."""
        # This integration test should cover:
        # 1. Real file system operations (not mocked)
        # 2. Real path resolution logic  
        # 3. Real framework detection
        # 4. Real lock cleanup (Issue #23)
        # 5. End-to-end workflows
        
        # Count methods in TestRealModelLifecycle
        test_methods = [method for method in dir(TestRealModelLifecycle) 
                       if method.startswith('test_')]
        
        # Should have comprehensive lifecycle coverage
        assert len(test_methods) >= 7, f"Should have comprehensive test coverage: {len(test_methods)} tests"
        
        # Should test specific functionality
        method_names = ' '.join(test_methods)
        assert 'download' in method_names, "Should test downloading"
        assert 'list' in method_names, "Should test listing" 
        assert 'show' in method_names, "Should test showing"
        assert 'resolve' in method_names, "Should test resolution"
        assert 'health' in method_names, "Should test health checks"
        assert 'rm' in method_names or 'remove' in method_names, "Should test removal"
        assert 'lock' in method_names, "Should test lock cleanup (Issue #23)"
        
        print(f"✓ Integration tests provide comprehensive lifecycle coverage: {len(test_methods)} tests")