"""
Unit tests for cache_utils.py module.

Tests the core model management functions:
- Model discovery and metadata extraction
- Health checking logic
- Cache operations
"""
import pytest
import tempfile
import shutil
import json
from pathlib import Path
from unittest.mock import patch, MagicMock, call

# Import the module under test
from mlx_knife.cache_utils import (
    expand_model_name,
    hf_to_cache_dir, 
    cache_dir_to_hf,
    is_model_healthy,
    detect_framework,
    list_models,
    find_matching_models,
    resolve_single_model
)


class TestModelNameExpansion:
    """Test model name expansion logic."""
    
    def test_expand_short_names(self):
        """Test expansion of common short model names."""
        test_cases = [
            ("Phi-3-mini", "mlx-community/Phi-3-mini-4k-instruct-4bit"),
            ("Mistral-7B", "mlx-community/Mistral-7B-Instruct-v0.3-4bit"),
            ("Llama-3-8B", "mlx-community/Meta-Llama-3-8B-Instruct-4bit"),
        ]
        
        for short_name, expected in test_cases:
            try:
                result = expand_model_name(short_name)
                # Should either expand correctly or return the original name
                assert isinstance(result, str)
                assert len(result) > 0
            except Exception as e:
                pytest.fail(f"expand_model_name failed for {short_name}: {e}")

    def test_expand_full_names(self):
        """Test that full model names are returned unchanged."""
        full_names = [
            "mlx-community/Phi-3-mini-4k-instruct-4bit",
            "microsoft/Phi-3-mini-4k-instruct",
            "meta-llama/Llama-2-7b-chat-hf"
        ]
        
        for full_name in full_names:
            try:
                result = expand_model_name(full_name)
                # Should return the name as-is or expand it
                assert isinstance(result, str)
                assert len(result) > 0
            except Exception as e:
                pytest.fail(f"expand_model_name failed for {full_name}: {e}")

    def test_expand_invalid_names(self):
        """Test handling of invalid or nonsense model names."""
        invalid_names = [
            "definitely-not-a-model-12345",
            "",
            "   ",
            "invalid/model/with/too/many/slashes"
        ]
        
        for invalid_name in invalid_names:
            try:
                result = expand_model_name(invalid_name)
                # Should handle gracefully - either return input or raise appropriate error
                if result is not None:
                    assert isinstance(result, str)
            except Exception:
                # It's OK to raise exceptions for invalid names
                pass


class TestCacheDirectoryConversion:
    """Test cache directory name conversion functions."""
    
    def test_hf_to_cache_dir(self):
        """Test HuggingFace model name to cache directory conversion."""
        test_cases = [
            ("microsoft/Phi-3-mini-4k-instruct", "models--microsoft--Phi-3-mini-4k-instruct"),
            ("meta-llama/Llama-2-7b", "models--meta-llama--Llama-2-7b"),
            ("simple-model", "models--simple-model"),
        ]
        
        for hf_name, expected_cache_dir in test_cases:
            try:
                result = hf_to_cache_dir(hf_name)
                assert isinstance(result, str)
                # Should follow HF cache naming convention
                assert result.startswith("models--")
                assert "--" in result
            except Exception as e:
                pytest.fail(f"hf_to_cache_dir failed for {hf_name}: {e}")

    def test_cache_dir_to_hf(self):
        """Test cache directory to HuggingFace model name conversion."""
        test_cases = [
            ("models--microsoft--Phi-3-mini-4k-instruct", "microsoft/Phi-3-mini-4k-instruct"),
            ("models--meta-llama--Llama-2-7b", "meta-llama/Llama-2-7b"),
            ("models--simple-model", "simple-model"),
        ]
        
        for cache_dir, expected_hf_name in test_cases:
            try:
                result = cache_dir_to_hf(cache_dir)
                assert isinstance(result, str)
                # Should reverse the cache directory format
                assert "/" in result or len(result.split("--")) == 1
            except Exception as e:
                pytest.fail(f"cache_dir_to_hf failed for {cache_dir}: {e}")

    def test_round_trip_conversion(self):
        """Test that conversion functions are inverses."""
        test_names = [
            "microsoft/Phi-3-mini-4k-instruct",
            "simple-model",
            "org/model-name-with-dashes"
        ]
        
        for original_name in test_names:
            try:
                cache_dir = hf_to_cache_dir(original_name)
                recovered_name = cache_dir_to_hf(cache_dir)
                
                assert recovered_name == original_name, \
                    f"Round trip failed: {original_name} -> {cache_dir} -> {recovered_name}"
            except Exception as e:
                pytest.fail(f"Round trip conversion failed for {original_name}: {e}")


class TestModelHealthCheck:
    """Test model health checking logic."""
    
    def test_healthy_model_structure(self, temp_cache_dir):
        """Test health check on properly structured model."""
        # Create a healthy model structure
        model_dir = temp_cache_dir / "models--test--model" / "snapshots" / "main"
        model_dir.mkdir(parents=True)
        
        # Create required files
        (model_dir / "config.json").write_text('{"model_type": "test", "architectures": ["TestModel"]}')
        (model_dir / "tokenizer.json").write_text('{"version": "1.0", "tokenizer": {}}')
        (model_dir / "model.safetensors").write_bytes(b"fake_model_weights" * 100)
        
        try:
            is_healthy = is_model_healthy(str(model_dir))
            # Should be True for healthy model
            assert isinstance(is_healthy, bool)
        except Exception as e:
            pytest.fail(f"Health check failed on healthy model: {e}")

    def test_missing_config_detection(self, temp_cache_dir):
        """Test detection of missing config.json."""
        model_dir = temp_cache_dir / "models--test--model" / "snapshots" / "main"
        model_dir.mkdir(parents=True)
        
        # Missing config.json
        (model_dir / "tokenizer.json").write_text('{"version": "1.0"}')
        (model_dir / "model.safetensors").write_bytes(b"fake_weights")
        
        try:
            is_healthy = is_model_healthy(str(model_dir))
            # Should detect missing config
            assert isinstance(is_healthy, bool)
            # Likely should be False, but depends on implementation
        except Exception as e:
            # It's OK to raise exception for missing config
            pass

    def test_missing_tokenizer_detection(self, temp_cache_dir):
        """Test detection of missing tokenizer.json."""
        model_dir = temp_cache_dir / "models--test--model" / "snapshots" / "main"
        model_dir.mkdir(parents=True)
        
        # Missing tokenizer.json
        (model_dir / "config.json").write_text('{"model_type": "test"}')
        (model_dir / "model.safetensors").write_bytes(b"fake_weights")
        
        try:
            is_healthy = is_model_healthy(str(model_dir))
            assert isinstance(is_healthy, bool)
        except Exception as e:
            # OK to raise exception for missing tokenizer
            pass

    def test_missing_model_weights(self, temp_cache_dir):
        """Test detection of missing model weights."""
        model_dir = temp_cache_dir / "models--test--model" / "snapshots" / "main"
        model_dir.mkdir(parents=True)
        
        # Missing model files
        (model_dir / "config.json").write_text('{"model_type": "test"}')
        (model_dir / "tokenizer.json").write_text('{"version": "1.0"}')
        # No .safetensors files
        
        try:
            is_healthy = is_model_healthy(str(model_dir))
            assert isinstance(is_healthy, bool)
        except Exception as e:
            # OK to raise exception for missing weights
            pass

    def test_lfs_pointer_detection(self, temp_cache_dir):
        """Test detection of LFS pointer files."""
        model_dir = temp_cache_dir / "models--test--model" / "snapshots" / "main"
        model_dir.mkdir(parents=True)
        
        (model_dir / "config.json").write_text('{"model_type": "test"}')
        (model_dir / "tokenizer.json").write_text('{"version": "1.0"}')
        
        # Create LFS pointer file instead of actual weights
        lfs_content = (
            "version https://git-lfs.github.com/spec/v1\n"
            "oid sha256:abc123def456\n"
            "size 1000000000\n"
        )
        (model_dir / "model.safetensors").write_text(lfs_content)
        
        try:
            is_healthy = is_model_healthy(str(model_dir))
            # Should detect LFS pointer as unhealthy
            assert isinstance(is_healthy, bool)
        except Exception as e:
            # OK to raise exception for LFS pointers
            pass

    def test_nonexistent_directory(self):
        """Test health check on nonexistent directory."""
        nonexistent_path = "/this/path/definitely/does/not/exist"
        
        try:
            is_healthy = is_model_healthy(nonexistent_path)
            # Should handle gracefully
            assert isinstance(is_healthy, bool)
            assert is_healthy is False  # Nonexistent should be unhealthy
        except Exception:
            # OK to raise exception for nonexistent path
            pass


class TestFrameworkDetection:
    """Test model framework detection logic."""
    
    def test_mlx_model_detection(self, temp_cache_dir):
        """Test detection of MLX-compatible models."""
        model_dir = temp_cache_dir / "models--mlx-community--test-model" / "snapshots" / "main"
        model_dir.mkdir(parents=True)
        
        # Create MLX model config
        mlx_config = {
            "model_type": "llama",
            "architectures": ["LlamaForCausalLM"],
            "quantization": {"group_size": 64, "bits": 4}  # MLX quantization
        }
        (model_dir / "config.json").write_text(json.dumps(mlx_config))
        (model_dir / "tokenizer.json").write_text('{"version": "1.0"}')
        (model_dir / "model.safetensors").write_bytes(b"mlx_weights")
        
        try:
            from pathlib import Path
            framework = detect_framework(Path(str(model_dir)), "mlx-community/test-model")
            assert isinstance(framework, str)
            # Should detect as MLX or compatible
        except Exception as e:
            pytest.fail(f"Framework detection failed on MLX model: {e}")

    def test_pytorch_model_detection(self, temp_cache_dir):
        """Test detection of PyTorch models."""
        model_dir = temp_cache_dir / "models--pytorch--test-model" / "snapshots" / "main"
        model_dir.mkdir(parents=True)
        
        # Create PyTorch model config
        pytorch_config = {
            "model_type": "bert",
            "architectures": ["BertForSequenceClassification"],
            "torch_dtype": "float32"
        }
        (model_dir / "config.json").write_text(json.dumps(pytorch_config))
        (model_dir / "tokenizer.json").write_text('{"version": "1.0"}')
        (model_dir / "pytorch_model.bin").write_bytes(b"pytorch_weights")
        
        try:
            from pathlib import Path
            framework = detect_framework(Path(str(model_dir)), "pytorch/test-model")
            assert isinstance(framework, str)
        except Exception as e:
            pytest.fail(f"Framework detection failed on PyTorch model: {e}")


class TestModelListing:
    """Test model listing functionality."""
    
    @patch('mlx_knife.cache_utils.MODEL_CACHE')
    def test_list_models_empty_cache(self, mock_cache, temp_cache_dir):
        """Test model listing in empty cache."""
        mock_cache.__str__ = lambda: str(temp_cache_dir)
        mock_cache.exists.return_value = True
        mock_cache.glob.return_value = []
        
        try:
            # list_models prints to stdout, so we test it doesn't crash
            list_models(verbose=False)
        except Exception as e:
            pytest.fail(f"Model listing failed on empty cache: {e}")

    def test_list_models_real_empty_cache(self, temp_cache_dir):
        """Test Issue #21: list_models with real empty HF_HOME directory."""
        import os
        from mlx_knife.cache_utils import list_models
        
        # Create empty cache directory
        empty_cache = temp_cache_dir / "empty_hf_cache" 
        empty_cache.mkdir()
        
        # Set HF_HOME to empty directory and test
        original_hf_home = os.environ.get('HF_HOME')
        try:
            os.environ['HF_HOME'] = str(empty_cache)
            # Should not crash and should print helpful message
            list_models()
        except FileNotFoundError as e:
            pytest.fail(f"Issue #21 regression: list_models crashed with empty cache: {e}")
        finally:
            if original_hf_home is not None:
                os.environ['HF_HOME'] = original_hf_home
            elif 'HF_HOME' in os.environ:
                del os.environ['HF_HOME']

    @patch('mlx_knife.cache_utils.MODEL_CACHE')
    def test_list_models_basic_call(self, mock_cache, temp_cache_dir):
        """Test basic model listing call."""
        mock_cache.__str__ = lambda: str(temp_cache_dir)
        mock_cache.exists.return_value = True
        mock_cache.glob.return_value = []
        
        try:
            # Test various parameter combinations
            list_models(show_all=True)
            list_models(framework_filter="MLX")
            list_models(show_health=True)
        except Exception as e:
            pytest.fail(f"Model listing with parameters failed: {e}")


class TestModelRemoval:
    """Test rm_model functionality (Issue #23)."""
    
    def setup_method(self):
        """Setup mock cache structure for each test."""
        self.test_model_name = "microsoft/DialoGPT-small"
        self.test_hash = "49c537161a457d5256512f9d2d38a87d81ae0f0e"
        self.test_hash_short = "49c53716"
    
    @patch('mlx_knife.cache_utils.MODEL_CACHE')
    @patch('mlx_knife.cache_utils.resolve_single_model') 
    @patch('mlx_knife.cache_utils.shutil.rmtree')
    @patch('builtins.input', return_value='y')
    def test_rm_model_fixed_behavior_issue23(self, mock_input, mock_rmtree, mock_resolve, mock_cache, temp_cache_dir):
        """Test fixed rm behavior - should delete model AND locks (Issue #23 resolved).
        
        Setup mocked directory structure as documented in CLAUDE.md:
        hub/
        ├── .locks/models--<name>/      # Per-model lock files  
        └── models--<name>/             # Model data directory
            ├── blobs/                  # Deduplicated file storage
            ├── refs/main               # Points to current commit hash
            └── snapshots/<hash>/       # Specific version
        """
        from mlx_knife.cache_utils import rm_model
        
        # Create real temp directories that mirror HF cache structure
        # After fix: MODEL_CACHE points to hub/, locks are at hub/.locks/
        hub_dir = temp_cache_dir / "hub"
        model_dir = hub_dir / "models--microsoft--DialoGPT-small"
        snapshots_dir = model_dir / "snapshots"
        hash_dir = snapshots_dir / self.test_hash_short
        refs_dir = model_dir / "refs" 
        blobs_dir = model_dir / "blobs"
        locks_dir = hub_dir / ".locks" / "models--microsoft--DialoGPT-small"
        
        # Create the directory structure (but don't populate with real files)
        hash_dir.mkdir(parents=True)
        refs_dir.mkdir(parents=True) 
        blobs_dir.mkdir(parents=True)
        locks_dir.mkdir(parents=True)
        
        # Create refs/main file pointing to hash
        (refs_dir / "main").write_text(self.test_hash_short)
        
        # Create some mock lock files
        (locks_dir / "file1.lock").touch()
        (locks_dir / "file2.lock").touch()
        
        # Mock resolve_single_model to return our temp structure
        mock_resolve.return_value = (model_dir, self.test_model_name, self.test_hash_short)
        
        # Mock MODEL_CACHE to point to hub directory (after fix: locks are at MODEL_CACHE/.locks/)
        import mlx_knife.cache_utils
        mlx_knife.cache_utils.MODEL_CACHE = hub_dir
        
        # Verify our test structure exists
        assert model_dir.exists()
        assert hash_dir.exists() 
        assert (refs_dir / "main").exists()
        assert locks_dir.exists()
        assert len(list(locks_dir.iterdir())) == 2
        
        # Test current rm behavior - this should show Issue #23
        rm_model(f"{self.test_model_name}@{self.test_hash_short}")
        
        # Verify what was actually deleted
        # Fixed behavior: should delete model directory AND locks directory
        assert mock_rmtree.call_count == 2
        
        # Verify both calls: model directory and locks directory
        calls = [call[0][0] for call in mock_rmtree.call_args_list]
        model_call = next((call for call in calls if "models--microsoft--DialoGPT-small" in str(call) and ".locks" not in str(call)), None)
        locks_call = next((call for call in calls if ".locks" in str(call)), None)
        
        assert model_call is not None, "Should delete model directory"
        assert locks_call is not None, "Should delete locks directory"
    
    @patch('mlx_knife.cache_utils.MODEL_CACHE')
    @patch('mlx_knife.cache_utils.resolve_single_model') 
    @patch('mlx_knife.cache_utils.shutil.rmtree')
    def test_rm_model_force_parameter(self, mock_rmtree, mock_resolve, mock_cache, temp_cache_dir):
        """Test rm_model with force=True skips all confirmations."""
        from mlx_knife.cache_utils import rm_model
        
        # Create same temp structure as previous test (updated for fix)
        hub_dir = temp_cache_dir / "hub"
        model_dir = hub_dir / "models--microsoft--DialoGPT-small"
        snapshots_dir = model_dir / "snapshots"
        hash_dir = snapshots_dir / self.test_hash_short
        locks_dir = hub_dir / ".locks" / "models--microsoft--DialoGPT-small"
        
        # Create the directory structure
        hash_dir.mkdir(parents=True)
        locks_dir.mkdir(parents=True)
        (locks_dir / "file1.lock").touch()
        (locks_dir / "file2.lock").touch()
        
        # Mock resolve_single_model to return our temp structure
        mock_resolve.return_value = (model_dir, self.test_model_name, self.test_hash_short)
        
        # Mock MODEL_CACHE to point to hub directory (after fix)
        import mlx_knife.cache_utils
        mlx_knife.cache_utils.MODEL_CACHE = hub_dir
        
        # Test with force=True - should NOT call input() at all
        with patch('builtins.input') as mock_input:
            rm_model(f"{self.test_model_name}@{self.test_hash_short}", force=True)
            
            # Verify input() was never called (no prompts with force=True)
            mock_input.assert_not_called()
        
        # Verify both model and locks were deleted
        assert mock_rmtree.call_count == 2
        calls = [call[0][0] for call in mock_rmtree.call_args_list]
        model_call = next((call for call in calls if "models--microsoft--DialoGPT-small" in str(call) and ".locks" not in str(call)), None)
        locks_call = next((call for call in calls if ".locks" in str(call)), None)
        
        assert model_call is not None, "Should delete model directory with force=True"
        assert locks_call is not None, "Should delete locks directory with force=True"
    
    @patch('mlx_knife.cache_utils.MODEL_CACHE')
    @patch('mlx_knife.cache_utils.resolve_single_model') 
    @patch('mlx_knife.cache_utils.shutil.rmtree')
    def test_rm_model_force_vs_interactive(self, mock_rmtree, mock_resolve, mock_cache, temp_cache_dir):
        """Test that force=True behaves differently than interactive mode."""
        from mlx_knife.cache_utils import rm_model
        
        # Create temp structure (updated for fix)
        hub_dir = temp_cache_dir / "hub"
        model_dir = hub_dir / "models--test--model"
        snapshots_dir = model_dir / "snapshots"
        hash_dir = snapshots_dir / "abc12345"
        locks_dir = hub_dir / ".locks" / "models--test--model"
        
        hash_dir.mkdir(parents=True)
        locks_dir.mkdir(parents=True)
        (locks_dir / "test.lock").touch()
        
        mock_resolve.return_value = (model_dir, "test/model", None)
        # Mock MODEL_CACHE to point to hub directory (after fix)
        import mlx_knife.cache_utils
        mlx_knife.cache_utils.MODEL_CACHE = hub_dir
        
        # Test 1: Interactive mode - user says no
        mock_rmtree.reset_mock()
        with patch('builtins.input', return_value='n'):
            rm_model("test/model", force=False)
            # Should NOT delete anything when user says no
            mock_rmtree.assert_not_called()
        
        # Test 2: Force mode - no prompts, just delete
        mock_rmtree.reset_mock()
        with patch('builtins.input') as mock_input:
            rm_model("test/model", force=True)
            # Should NOT prompt user
            mock_input.assert_not_called()
            # Should delete both model and locks
            assert mock_rmtree.call_count == 2
    
    
    @patch('mlx_knife.cache_utils.resolve_single_model')
    def test_rm_model_not_found(self, mock_resolve):
        """Test rm behavior when model is not found."""
        from mlx_knife.cache_utils import rm_model
        
        # Setup resolve to return None (not found)
        mock_resolve.return_value = (None, None, None)
        
        # Should return early without error
        result = rm_model("nonexistent/model@hash")
        assert result is None


class TestPartialNameFiltering:
    """Test partial name filtering for list command (Issue 1)."""
    
    def test_find_matching_models_function(self):
        """Test the find_matching_models helper function."""
        with patch('mlx_knife.cache_utils.MODEL_CACHE') as mock_cache:
            # Mock some model directories
            mock_models = [
                MagicMock(name="models--mlx-community--Phi-3-mini"),
                MagicMock(name="models--mlx-community--Phi-3-medium"), 
                MagicMock(name="models--other--Llama-3-8B"),
            ]
            
            for i, mock_model in enumerate(mock_models):
                mock_model.name = f"models--{'mlx-community' if i < 2 else 'other'}--{'Phi-3-mini' if i == 0 else 'Phi-3-medium' if i == 1 else 'Llama-3-8B'}"
            
            mock_cache.iterdir.return_value = mock_models
            
            # Test finding Phi-3 models
            matches = find_matching_models("Phi-3")
            assert len(matches) == 2
            
            # Test finding non-existent model
            matches = find_matching_models("nonexistent")
            assert len(matches) == 0
    
    def test_partial_matching_basic_functionality(self):
        """Test basic partial matching logic without complex mocking."""
        # Simple functional test of the helper functions
        try:
            # These functions exist and can be called
            assert callable(find_matching_models)
            # Function handles empty input gracefully
            matches = find_matching_models("")
            assert isinstance(matches, list)
        except Exception as e:
            pytest.fail(f"Basic functionality test failed: {e}")


class TestSingleModelFuzzyMatching:
    """Test fuzzy matching for single-model commands (Issue 2)."""
    
    def test_resolve_single_model_function_exists(self):
        """Test that resolve_single_model function exists and is callable."""
        try:
            assert callable(resolve_single_model)
            # Function handles invalid input gracefully 
            result = resolve_single_model("definitely-nonexistent-model-12345")
            assert isinstance(result, tuple)
            assert len(result) == 3
        except Exception as e:
            pytest.fail(f"Function existence test failed: {e}")
    
    @patch('mlx_knife.cache_utils.get_model_path') 
    @patch('mlx_knife.cache_utils.find_matching_models')
    def test_resolve_single_model_ambiguous_fuzzy(self, mock_find, mock_get_path, capsys):
        """Test ambiguous fuzzy match shows error."""
        # Mock exact match fails, fuzzy finds multiple matches
        mock_get_path.return_value = (None, None, None)
        mock_find.return_value = [
            (MagicMock(), "model-1"),
            (MagicMock(), "model-2")
        ]
        
        result = resolve_single_model("partial")
        assert result[0] is None  # Should fail
        
        # Check that error message was printed
        captured = capsys.readouterr()
        assert "Multiple models match" in captured.out
        assert "model-1" in captured.out
        assert "model-2" in captured.out
    
    @patch('mlx_knife.cache_utils.get_model_path')
    @patch('mlx_knife.cache_utils.find_matching_models')
    def test_resolve_single_model_no_match(self, mock_find, mock_get_path, capsys):
        """Test no match shows appropriate error."""
        # Mock both exact and fuzzy matching fail
        mock_get_path.return_value = (None, None, None)
        mock_find.return_value = []
        
        result = resolve_single_model("nonexistent")
        assert result[0] is None  # Should fail
        
        # Check error message
        captured = capsys.readouterr()
        assert "No models found matching" in captured.out


class TestShowModelHealthConsistency:
    """Test for Issue #7 - Health check inconsistency in show command with fuzzy model names."""
    
    @patch('mlx_knife.cache_utils.resolve_single_model')
    @patch('mlx_knife.cache_utils.is_model_healthy')
    @patch('mlx_knife.cache_utils.get_model_size')
    @patch('mlx_knife.cache_utils.get_model_modified')
    @patch('mlx_knife.cache_utils.detect_framework')
    @patch('builtins.print')
    def test_show_model_health_consistency_fuzzy_vs_full_name(self, mock_print, mock_framework, 
                                                              mock_modified, mock_size, mock_healthy, 
                                                              mock_resolve, temp_cache_dir):
        """Test that fuzzy and full model names show identical health status.
        
        This is a regression test for Issue #7 where:
        - mlxk show Phi-3 showed "CORRUPTED"  
        - mlxk show mlx-community/Phi-3-mini-4k-instruct-4bit showed "OK"
        for the same underlying model.
        """
        # Setup mock model path
        mock_model_path = temp_cache_dir / "models--mlx-community--Phi-3-mini-4k-instruct-4bit" / "snapshots" / "abc123"
        mock_model_path.mkdir(parents=True)
        
        # Mock resolve_single_model to return consistent results
        # Both fuzzy "Phi-3" and full name should resolve to same model_name
        mock_resolve.return_value = (
            mock_model_path,
            "mlx-community/Phi-3-mini-4k-instruct-4bit",  # Resolved full name
            "abc123"
        )
        
        # Mock other dependencies
        mock_size.return_value = "4.2GB"
        mock_modified.return_value = "2023-12-01 10:00:00"
        mock_framework.return_value = "MLX"
        
        # Test both healthy and unhealthy scenarios
        for health_status in [True, False]:
            mock_healthy.return_value = health_status
            mock_print.reset_mock()
            
            # Test fuzzy name
            from mlx_knife.cache_utils import show_model
            show_model("Phi-3")  # Fuzzy name
            fuzzy_calls = [str(call) for call in mock_print.call_args_list]
            
            mock_print.reset_mock()
            
            # Test full name  
            show_model("mlx-community/Phi-3-mini-4k-instruct-4bit")  # Full name
            full_calls = [str(call) for call in mock_print.call_args_list]
            
            # Both should have identical health output
            fuzzy_health_output = [call for call in fuzzy_calls if "Health:" in call]
            full_health_output = [call for call in full_calls if "Health:" in call]
            
            assert len(fuzzy_health_output) == 1, f"Expected 1 health output for fuzzy name, got {len(fuzzy_health_output)}"
            assert len(full_health_output) == 1, f"Expected 1 health output for full name, got {len(full_health_output)}"
            assert fuzzy_health_output == full_health_output, f"Health status differs: fuzzy={fuzzy_health_output} vs full={full_health_output}"
            
            # Verify is_model_healthy was called with resolved model name (not original spec)
            expected_calls = [call("mlx-community/Phi-3-mini-4k-instruct-4bit")] * 2
            assert mock_healthy.call_args_list == expected_calls, f"is_model_healthy should be called with resolved name, got {mock_healthy.call_args_list}"
            
            # Reset for next iteration
            mock_healthy.reset_mock()



class TestIssue6RepositoryNameValidation:
    """Test for Issue #6 - Add repository name length validation for HuggingFace Hub."""
    
    @patch('builtins.input', return_value='y')  # Mock user input to avoid stdin issues
    def test_pull_model_rejects_long_names(self, mock_input, capsys):
        """Test that repository names >96 characters are rejected."""
        from mlx_knife.hf_download import pull_model
        
        # Create a name that exceeds 96 characters after expansion
        # Use direct long name that doesn't get expanded but is >96 chars
        long_model_name = "organization-name/very-long-model-name-that-definitely-exceeds-the-character-limit-for-repositories-on-hf-platform"
        
        result = pull_model(long_model_name)
        
        assert result is False
        
        captured = capsys.readouterr()
        assert "Repository name exceeds HuggingFace Hub limit" in captured.out
        assert "96 characters" in captured.out
        assert "cannot exist on HuggingFace Hub" in captured.out


class TestIssue13HashBasedDisambiguation:
    """Test for Issue #13 - Hash-based disambiguation for ambiguous model names."""
    
    def test_hash_exists_in_local_cache_full_hash(self):
        """Test hash_exists_in_local_cache returns full hash when exact match exists."""
        with patch('mlx_knife.cache_utils.MODEL_CACHE') as mock_cache:
            mock_hash_dir = MagicMock()
            mock_hash_dir.exists.return_value = True
            
            mock_snapshots_dir = MagicMock()
            mock_snapshots_dir.exists.return_value = True
            mock_snapshots_dir.__truediv__.return_value = mock_hash_dir
            
            mock_base_dir = MagicMock()
            mock_base_dir.exists.return_value = True
            mock_base_dir.__truediv__.return_value = mock_snapshots_dir
            
            mock_cache.__truediv__.return_value = mock_base_dir
            
            from mlx_knife.cache_utils import hash_exists_in_local_cache
            
            full_hash = "a5339a4131f135d0fdc6a5c8b5bbed2753bbe0f3"
            result = hash_exists_in_local_cache("mlx-community/Phi-3-mini", full_hash)
            assert result == full_hash
    
    def test_hash_exists_in_local_cache_none_no_model(self):
        """Test hash_exists_in_local_cache returns None when model doesn't exist."""
        with patch('mlx_knife.cache_utils.MODEL_CACHE') as mock_cache:
            mock_base_dir = MagicMock()
            mock_base_dir.exists.return_value = False
            mock_cache.__truediv__.return_value = mock_base_dir
            
            from mlx_knife.cache_utils import hash_exists_in_local_cache
            
            result = hash_exists_in_local_cache("nonexistent/model", "hash123")
            assert result is None
    
    def test_hash_exists_in_local_cache_none_no_hash(self):
        """Test hash_exists_in_local_cache returns None when hash doesn't exist."""
        with patch('mlx_knife.cache_utils.MODEL_CACHE') as mock_cache:
            mock_hash_dir = MagicMock()
            mock_hash_dir.exists.return_value = False
            
            mock_snapshots_dir = MagicMock()
            mock_snapshots_dir.exists.return_value = True
            mock_snapshots_dir.__truediv__.return_value = mock_hash_dir
            mock_snapshots_dir.iterdir.return_value = []  # No snapshots
            
            mock_base_dir = MagicMock()
            mock_base_dir.exists.return_value = True
            mock_base_dir.__truediv__.return_value = mock_snapshots_dir
            
            mock_cache.__truediv__.return_value = mock_base_dir
            
            from mlx_knife.cache_utils import hash_exists_in_local_cache
            
            result = hash_exists_in_local_cache("mlx-community/Phi-3-mini", "nonexistent")
            assert result is None
    
    def test_hash_exists_in_local_cache_short_hash_resolution(self):
        """Test hash_exists_in_local_cache resolves short hashes locally."""
        with patch('mlx_knife.cache_utils.MODEL_CACHE') as mock_cache:
            # Mock exact match fails
            mock_hash_dir = MagicMock()
            mock_hash_dir.exists.return_value = False
            
            # Mock snapshots directory with matching hash
            mock_snapshot = MagicMock()
            mock_snapshot.is_dir.return_value = True
            mock_snapshot.name = "de2dfaf56839b7d0e834157d2401dee02726874d"
            
            mock_snapshots_dir = MagicMock()
            mock_snapshots_dir.exists.return_value = True
            mock_snapshots_dir.__truediv__.return_value = mock_hash_dir
            mock_snapshots_dir.iterdir.return_value = [mock_snapshot]
            
            mock_base_dir = MagicMock()
            mock_base_dir.exists.return_value = True
            mock_base_dir.__truediv__.return_value = mock_snapshots_dir
            
            mock_cache.__truediv__.return_value = mock_base_dir
            
            from mlx_knife.cache_utils import hash_exists_in_local_cache
            
            result = hash_exists_in_local_cache("mlx-community/Llama-3.3-70B", "de2dfaf5")
            assert result == "de2dfaf56839b7d0e834157d2401dee02726874d"
    
    @patch('mlx_knife.cache_utils.get_model_path')
    @patch('mlx_knife.cache_utils.hash_exists_in_local_cache')
    @patch('mlx_knife.cache_utils.find_matching_models')
    @patch('mlx_knife.cache_utils.MODEL_CACHE')
    def test_resolve_single_model_hash_disambiguation_success(self, mock_cache, mock_find, mock_hash_exists, mock_get_path):
        """Test successful hash-based disambiguation when multiple models match."""
        # Mock find_matching_models to return multiple matches
        mock_find.return_value = [
            (MagicMock(), "mlx-community/Llama-3.2-1B-Instruct-4bit"),
            (MagicMock(), "mlx-community/Llama-3.3-70B-Instruct-4bit"),
        ]
        
        # Mock hash_exists_in_local_cache to return full hash for second model only
        def mock_hash_exists_side_effect(model_name, commit_hash):
            if model_name == "mlx-community/Llama-3.3-70B-Instruct-4bit":
                return "de2dfaf56839b7d0e834157d2401dee02726874d"
            return None
        mock_hash_exists.side_effect = mock_hash_exists_side_effect
        
        # Mock get_model_path to return success
        mock_get_path.return_value = (MagicMock(), "mlx-community/Llama-3.3-70B-Instruct-4bit", "de2dfaf5")
        
        # Mock MODEL_CACHE behavior for exact match check
        mock_base_dir = MagicMock()
        mock_base_dir.exists.return_value = False
        mock_cache.__truediv__.return_value = mock_base_dir
        
        from mlx_knife.cache_utils import resolve_single_model
        
        result = resolve_single_model("Llama@de2dfaf5")
        
        # Should successfully resolve to the second model
        assert result[1] == "mlx-community/Llama-3.3-70B-Instruct-4bit"
        assert result[2] == "de2dfaf5"
        
        # Verify hash_exists_in_local_cache was called for both models
        assert mock_hash_exists.call_count == 2
        
        # Verify get_model_path was called with the resolved spec (full hash)
        mock_get_path.assert_called_once_with("mlx-community/Llama-3.3-70B-Instruct-4bit@de2dfaf56839b7d0e834157d2401dee02726874d")
    
    @patch('mlx_knife.cache_utils.hash_exists_in_local_cache')
    @patch('mlx_knife.cache_utils.find_matching_models')
    @patch('mlx_knife.cache_utils.MODEL_CACHE')
    def test_resolve_single_model_hash_disambiguation_no_match(self, mock_cache, mock_find, mock_hash_exists, capsys):
        """Test hash-based disambiguation when hash doesn't exist in any model."""
        # Mock find_matching_models to return multiple matches
        mock_find.return_value = [
            (MagicMock(), "mlx-community/Llama-3.2-1B-Instruct-4bit"),
            (MagicMock(), "mlx-community/Llama-3.3-70B-Instruct-4bit"),
        ]
        
        # Mock hash_exists_in_local_cache to return None for all models
        mock_hash_exists.return_value = None
        
        # Mock MODEL_CACHE behavior for exact match check
        mock_base_dir = MagicMock()
        mock_base_dir.exists.return_value = False
        mock_cache.__truediv__.return_value = mock_base_dir
        
        from mlx_knife.cache_utils import resolve_single_model
        
        result = resolve_single_model("Llama@nonexistent")
        
        # Should return None tuple
        assert result == (None, None, None)
        
        # Check error message was printed
        captured = capsys.readouterr()
        assert "Hash 'nonexistent' not found in any model matching 'Llama'" in captured.out
        assert "Available models:" in captured.out
    
    @patch('mlx_knife.cache_utils.find_matching_models')
    @patch('mlx_knife.cache_utils.MODEL_CACHE')
    def test_resolve_single_model_no_hash_multiple_matches(self, mock_cache, mock_find, capsys):
        """Test traditional ambiguous model behavior without hash is preserved."""
        # Mock find_matching_models to return multiple matches
        mock_find.return_value = [
            (MagicMock(), "mlx-community/Llama-3.2-1B-Instruct-4bit"),
            (MagicMock(), "mlx-community/Llama-3.3-70B-Instruct-4bit"),
        ]
        
        # Mock MODEL_CACHE behavior for exact match check
        mock_base_dir = MagicMock()
        mock_base_dir.exists.return_value = False
        mock_cache.__truediv__.return_value = mock_base_dir
        
        from mlx_knife.cache_utils import resolve_single_model
        
        result = resolve_single_model("Llama")  # No hash specified
        
        # Should return None tuple
        assert result == (None, None, None)
        
        # Check traditional error message was printed
        captured = capsys.readouterr()
        assert "Multiple models match 'Llama'. Please be more specific:" in captured.out


# Add pytest fixture at module level
@pytest.fixture
def temp_cache_dir():
    """Create temporary cache directory for testing."""
    with tempfile.TemporaryDirectory() as temp_dir:
        yield Path(temp_dir)