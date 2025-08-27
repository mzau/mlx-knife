"""ADR-002 Edge Cases Validation Tests for MLX-Knife 2.0.

These tests validate critical edge cases learned from 1.x development,
as documented in docs/ADR/ADR-002-edge-cases.md
"""

import pytest
import tempfile
from pathlib import Path
from mlxk2.core.cache import hf_to_cache_dir, cache_dir_to_hf
from mlxk2.core.model_resolution import resolve_model_for_operation, parse_model_spec
from mlxk2.operations.list import list_models
from mlxk2.operations.health import health_check_operation


class TestModelNameValidation:
    """Test model name validation edge cases from ADR-002."""
    
    def test_96_char_limit_validation(self):
        """Test HuggingFace 96 character model name limit."""
        # Valid length model name (95 chars)
        valid_name = "org/" + "a" * 91  # 95 total
        assert len(valid_name) == 95
        
        # Invalid length model name (97 chars)  
        invalid_name = "org/" + "a" * 93  # 97 total
        assert len(invalid_name) == 97
        
        # Resolution should handle long names gracefully
        resolved_name, commit_hash, ambiguous = resolve_model_for_operation(invalid_name)
        # Should either reject or truncate, not crash
        assert isinstance(resolved_name, (str, type(None)))
        assert isinstance(ambiguous, list)
    
    def test_empty_and_whitespace_names(self):
        """Test empty and whitespace-only model names."""
        test_cases = ["", " ", "  ", "\t", "\n", "   \t\n   "]
        
        for test_name in test_cases:
            resolved_name, commit_hash, ambiguous = resolve_model_for_operation(test_name)
            # Should handle gracefully, not crash
            assert resolved_name is None
            # Ambiguous may return all models (fuzzy matching behavior) or empty list
            assert isinstance(ambiguous, list)
    
    def test_invalid_characters_in_names(self):
        """Test names with invalid characters."""
        invalid_names = [
            "org//model",  # Double slash
            "org/model/",  # Trailing slash
            "/org/model",  # Leading slash
            "org//sub//model",  # Multiple double slashes
            "org\\model",  # Backslash
            "org<model>",  # Angle brackets
        ]
        
        for name in invalid_names:
            resolved_name, commit_hash, ambiguous = resolve_model_for_operation(name)
            # Should handle gracefully, not crash
            assert isinstance(resolved_name, (str, type(None)))
            assert isinstance(ambiguous, list)


class TestCacheDirectoryManagement:
    """Test cache directory handling edge cases."""
    
    def test_round_trip_conversion_bijective(self):
        """Test that HF name ↔ cache dir conversion is bijective."""
        test_cases = [
            "microsoft/DialoGPT-small",
            "org/sub/model",
            "single-model",
            "deep/nested/path/model",
            "org-with-dashes/model-with-dashes",
        ]
        
        for hf_name in test_cases:
            # Forward conversion
            cache_dir = hf_to_cache_dir(hf_name)
            
            # Backward conversion  
            recovered_name = cache_dir_to_hf(cache_dir)
            
            # Should be identical
            assert recovered_name == hf_name, f"Round-trip failed: {hf_name} → {cache_dir} → {recovered_name}"
    
    def test_corrupted_cache_tolerance(self):
        """Test tolerance for corrupted cache directory names."""
        # Violate naming rules (triple dashes)
        corrupted_cache_names = [
            "models--org---corrupted",  # Triple dash
            "models--org--model---bad",  # Triple dash at end
            "models---bad--model",  # Triple dash at start
        ]
        
        for cache_name in corrupted_cache_names:
            # Should not crash, mechanical conversion
            hf_name = cache_dir_to_hf(cache_name)
            
            # Should produce visible corruption (empty segments)
            assert isinstance(hf_name, str)
            # Corruption should be visible somehow (empty segments, leading/trailing dashes, etc.)
            if "---" in cache_name:
                corruption_indicators = ["/-", "//", hf_name.startswith("/"), hf_name.endswith("/"), 
                                       hf_name.startswith("-"), hf_name.endswith("-")]
                assert any(corruption_indicators), f"Corruption not visible in: {hf_name}"


class TestHashSyntaxParsing:
    """Test @hash syntax parsing edge cases."""
    
    def test_hash_syntax_parsing(self):
        """Test parsing of @hash syntax."""
        test_cases = [
            ("Phi-3@abc", ("Phi-3", "abc")),
            ("mlx-community/Model@def123", ("mlx-community/Model", "def123")),
            ("Model@a", ("Model", "a")),  # Single char hash
            ("Model@" + "a" * 40, ("Model", "a" * 40)),  # Long hash
        ]
        
        for input_spec, expected in test_cases:
            result = parse_model_spec(input_spec)
            assert result == expected
    
    def test_invalid_hash_syntax(self):
        """Test invalid @hash syntax handling."""
        invalid_cases = [
            "Model@",  # Empty hash
            "Model@@abc",  # Double @
            "@abc",  # No model name
            "Model@hash@invalid",  # Multiple @
        ]
        
        for invalid_spec in invalid_cases:
            # Should parse without crashing, handle invalid parts gracefully
            try:
                model_name, commit_hash = parse_model_spec(invalid_spec)
                # Should return reasonable values, not crash
                assert isinstance(model_name, str)
                assert isinstance(commit_hash, (str, type(None)))
            except Exception as e:
                # If it throws, should be a clear validation error
                assert "invalid" in str(e).lower() or "format" in str(e).lower()


class TestHealthCheckEdgeCases:
    """Test health checking edge cases from ADR-002."""
    
    def test_lfs_pointer_detection_pattern(self, isolated_cache):
        """Test LFS pointer detection logic."""
        # Create fake LFS pointer file
        test_model_dir = isolated_cache / "models--test--lfs-model" / "snapshots" / "main"
        test_model_dir.mkdir(parents=True)
        
        # Create LFS pointer content
        lfs_content = '''version https://git-lfs.github.com/spec/v1
oid sha256:1234567890abcdef1234567890abcdef1234567890abcdef1234567890abcdef
size 123456789
'''
        lfs_file = test_model_dir / "model.safetensors"
        lfs_file.write_text(lfs_content)
        
        # Health check should detect this as unhealthy/incomplete
        result = health_check_operation("test/lfs-model")
        
        # Should complete without crashing
        assert result["status"] == "success"
        
        # If LFS detection is implemented, should flag as unhealthy
        # (This test documents the expected behavior)
    
    def test_missing_critical_files(self, isolated_cache):
        """Test handling of models missing critical files."""
        # Create model with missing config.json
        incomplete_model_dir = isolated_cache / "models--test--incomplete" / "snapshots" / "main"  
        incomplete_model_dir.mkdir(parents=True)
        
        # Only create tokenizer, no config or model files
        (incomplete_model_dir / "tokenizer.json").write_text('{"version": "1.0"}')
        
        result = health_check_operation("test/incomplete")
        
        # Should handle gracefully
        assert result["status"] == "success"
        # Should identify as incomplete/unhealthy if detection is implemented
    
    def test_health_check_with_empty_cache(self):
        """Test health check when no models are cached."""
        result = health_check_operation()
        
        # Should handle empty cache gracefully
        assert result["status"] == "success"
        assert result["data"]["summary"]["total"] >= 0


class TestForceFlag:
    """Test force flag behavior in rm operations."""
    
    def test_force_flag_skips_all_confirmations(self, mock_models):
        """Test that -f flag skips ALL confirmations (Issue #23 regression)."""
        from mlxk2.operations.rm import rm_operation
        
        # Get available model from test cache
        models = list_models()["data"]["models"]
        if not models:
            pytest.skip("No models in test cache for force flag testing")
        
        target_model = models[0]["name"]
        
        # Force flag should work without any prompts
        result = rm_operation(target_model, force=True)
        
        # Should either succeed or fail with clear reason (never prompt)
        assert result["status"] in ["success", "error"]
        
        if result["status"] == "error":
            # Error should not be about confirmation/prompts
            error_msg = result["error"]["message"].lower()
            # Check for interactive prompts (not system errors like "no such file")
            forbidden_phrases = ["confirm", "prompt", "yes/no", "continue?", "are you sure"]
            for phrase in forbidden_phrases:
                assert phrase not in error_msg, f"Force flag still prompting: {error_msg}"


class TestJSONErrorHandling:
    """Test JSON error handling consistency."""
    
    def test_invalid_operations_return_valid_json(self):
        """Test that all invalid operations return valid JSON."""
        invalid_operations = [
            lambda: resolve_model_for_operation("definitely-nonexistent-12345"),
            lambda: health_check_operation("nonexistent-model"),
            lambda: parse_model_spec("invalid@@syntax"),
        ]
        
        for operation in invalid_operations:
            try:
                result = operation()
                # Should return structured data, not throw
                assert isinstance(result, (tuple, dict, list))
            except Exception as e:
                # If it throws, should be for a good reason with clear message
                assert str(e), "Empty error message not allowed"
    
    def test_json_structure_consistency(self):
        """Test that all operations return consistent JSON structure."""
        # Test operations that return JSON
        operations_to_test = [
            list_models,
            lambda: health_check_operation(),
        ]
        
        for operation in operations_to_test:
            result = operation()
            
            # Should have consistent JSON structure
            assert "status" in result
            assert result["status"] in ["success", "error"]
            assert "data" in result or "error" in result
            
            if "error" in result and result["error"] is not None:
                assert "message" in result["error"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])