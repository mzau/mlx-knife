"""Tests for MLX-Knife 2.0 model naming rules and conversion.

These tests document and verify the critical naming rules we discovered:
1. Universal conversion: -- ↔ / (all occurrences)  
2. Character constraints: single "-" extern, double "--" intern
3. Corrupted cache tolerance: mechanical conversion, problems visible
4. CLI compatibility: short names, @hash syntax, fuzzy matching
"""

import pytest
import sys
from pathlib import Path

# Import MLX-Knife 2.0 modules
sys.path.insert(0, str(Path(__file__).parent.parent))
from mlxk2.core.cache import hf_to_cache_dir, cache_dir_to_hf


class TestNamingConversionRules:
    """Test the fundamental -- ↔ / conversion rules."""
    
    def test_universal_conversion_rule(self):
        """ALL -- ↔ / conversion (not just first occurrence)."""
        # External → Internal: All "/" become "--"
        assert hf_to_cache_dir("org/sub/model") == "models--org--sub--model"
        assert hf_to_cache_dir("deep/nested/path/model") == "models--deep--nested--path--model"
        
        # Internal → External: All "--" become "/"  
        assert cache_dir_to_hf("models--org--sub--model") == "org/sub/model"
        assert cache_dir_to_hf("models--deep--nested--path--model") == "deep/nested/path/model"
    
    def test_bijective_conversion_clean_names(self):
        """Clean names must convert bijectively (no information loss)."""
        clean_names = [
            "microsoft/DialoGPT-small",
            "mlx-community/Phi-3-mini-4k-instruct-4bit",
            "org-name/model-v1",  # Single dashes OK
            "single-model",
            "org/sub/model",  # Multi-level
        ]
        
        for external in clean_names:
            internal = hf_to_cache_dir(external)
            recovered = cache_dir_to_hf(internal)
            assert external == recovered, f"NOT BIJECTIVE: {external} → {internal} → {recovered}"
    
    def test_character_constraint_validation(self):
        """Validate character constraints for clean conversion."""
        # Clean external names: max 1 consecutive dash
        valid_external = [
            "org-name/model-v1",
            "microsoft/DialoGPT-small"
        ]
        
        for external in valid_external:
            assert "--" not in external, f"Double dash in external name: {external}"
            
            internal = hf_to_cache_dir(external)
            # Clean internal: max 2 consecutive dashes (separators only)
            assert "---" not in internal, f"Triple dash in internal: {internal}"
    
    def test_corrupted_cache_mechanical_conversion(self):
        """Corrupted cache entries get mechanical conversion (problems visible)."""
        # These violate the clean naming rules but should convert gracefully
        corrupted_cases = [
            ("models--org--model---corrupted", "org/model/-corrupted"),  # Triple dash → empty segment
            ("models--microsoft--DialogGPT---small", "microsoft/DialogGPT/-small"),  # Problem visible
            ("models--org----model", "org//model"),  # Quadruple dash → empty segment
        ]
        
        for corrupted_internal, expected_external in corrupted_cases:
            result = cache_dir_to_hf(corrupted_internal)
            assert result == expected_external, f"Mechanical conversion failed: {corrupted_internal}"
            # Problem must be visible in result
            assert ("/-" in result or "//" in result), f"Corruption not visible in: {result}"


class TestModelResolutionLogic:
    """Test CLI compatibility features: expansion, @hash, fuzzy matching."""
    
    def test_hash_syntax_parsing(self):
        """@hash syntax must parse correctly."""
        from mlxk2.core.model_resolution import parse_model_spec
        
        # With hash
        model, hash_val = parse_model_spec("Qwen3@e96")
        assert hash_val == "e96"
        assert "@" not in model  # Hash removed from model name
        
        # Without hash  
        model, hash_val = parse_model_spec("Phi-3-mini")
        assert hash_val is None
        assert model == "Phi-3-mini"  # Would be expanded by expand_model_name
    
    def test_short_name_expansion_logic(self):
        """Short names should try mlx-community first, then return as-is."""
        from mlxk2.core.model_resolution import expand_model_name
        
        # Names with org should not be expanded
        assert expand_model_name("microsoft/DialoGPT-small") == "microsoft/DialoGPT-small"
        
        # Single names return as-is (no pattern forcing!)
        assert expand_model_name("nonexistent-model") == "nonexistent-model"
        
        # NOTE: mlx-community expansion requires actual cache, tested in integration tests
    
    def test_fuzzy_matching_pattern(self):
        """Fuzzy matching should be case-insensitive partial matching."""
        from mlxk2.core.model_resolution import find_matching_models
        
        # Empty cache returns empty list
        matches = find_matching_models("anything")
        assert isinstance(matches, list)  # Should not crash
        
        # NOTE: Real fuzzy matching requires actual cache, tested in integration tests


class TestErrorHandlingRobustness:
    """Test that edge cases don't crash the system."""
    
    def test_empty_and_invalid_inputs(self):
        """Empty or invalid inputs should not crash."""
        # Empty strings
        assert hf_to_cache_dir("") == "models--"
        assert cache_dir_to_hf("models--") == ""
        
        # Invalid formats
        assert cache_dir_to_hf("invalid-format") == "invalid-format"
        assert cache_dir_to_hf("models--") == ""
    
    def test_resolution_with_invalid_inputs(self):
        """Model resolution should handle invalid inputs gracefully."""
        from mlxk2.core.model_resolution import resolve_model_for_operation
        
        # Should return some response, not crash
        result = resolve_model_for_operation("")
        assert result is not None
        assert len(result) == 3  # (name, hash, matches)
        
        result = resolve_model_for_operation("nonexistent@invalidhash")  
        assert result is not None
        assert len(result) == 3


if __name__ == "__main__":
    pytest.main([__file__, "-v"])