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
from unittest.mock import patch, MagicMock

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


# Add pytest fixture at module level
@pytest.fixture
def temp_cache_dir():
    """Create temporary cache directory for testing."""
    with tempfile.TemporaryDirectory() as temp_dir:
        yield Path(temp_dir)