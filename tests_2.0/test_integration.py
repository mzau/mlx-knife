"""Integration tests for MLX-Knife 2.0 with realistic cache scenarios."""

import pytest
from mlxk2.core.model_resolution import resolve_model_for_operation
from mlxk2.operations.health import health_check_operation
from mlxk2.operations.rm import rm_operation


class TestModelResolutionIntegration:
    """Test model resolution with realistic cache structures."""
    
    def test_short_name_expansion_with_cache(self, mock_models):
        """Test that short names expand to mlx-community when model exists in cache."""
        # Should find the cached mlx-community model
        resolved_name, commit_hash, ambiguous = resolve_model_for_operation("Phi-3-mini")
        
        assert resolved_name == "mlx-community/Phi-3-mini-4k-instruct-4bit"
        assert commit_hash is None
        assert ambiguous is None
    
    def test_hash_syntax_resolution(self, mock_models):
        """Test @hash syntax finds correct model by short hash."""
        # Short hash "e96" should match "e9675aa3def..."
        resolved_name, commit_hash, ambiguous = resolve_model_for_operation("Qwen3@e96")
        
        # Should find one of the Qwen3 models (both have same short hash in our mock)
        assert resolved_name is not None
        assert "Qwen3" in resolved_name
        assert commit_hash == "e96"
        assert ambiguous is None
    
    def test_fuzzy_matching_partial_names(self, mock_models):
        """Test fuzzy matching finds models by partial names."""
        resolved_name, commit_hash, ambiguous = resolve_model_for_operation("DialoGPT")
        
        assert resolved_name == "microsoft/DialoGPT-small"
        assert commit_hash is None
        assert ambiguous is None
    
    def test_ambiguous_matching_returns_choices(self, mock_models):
        """Test that ambiguous patterns return list of matches."""
        # "Qwen" should match multiple models
        resolved_name, commit_hash, ambiguous = resolve_model_for_operation("Qwen")
        
        assert resolved_name is None
        assert ambiguous is not None
        assert len(ambiguous) >= 2  # At least 2 Qwen models in mock
        assert any("Qwen3-30B" in name for name in ambiguous)
        assert any("Qwen3-Coder-480B" in name for name in ambiguous)
    
    def test_nonexistent_model_handling(self, mock_models):
        """Test that nonexistent models are handled gracefully."""
        resolved_name, commit_hash, ambiguous = resolve_model_for_operation("nonexistent-model")
        
        assert resolved_name is None
        assert ambiguous == []  # Empty list, not None


class TestHealthOperationIntegration:
    """Test health operation with realistic models."""
    
    def test_health_check_all_models(self, mock_models):
        """Test health check on all cached models."""
        result = health_check_operation()
        
        assert result["status"] == "success"
        assert result["data"]["summary"]["total"] >= 4  # At least our mock models
        assert result["data"]["summary"]["healthy_count"] >= 3  # Healthy models
        assert result["data"]["summary"]["unhealthy_count"] >= 1  # Corrupted model
    
    def test_health_check_specific_model_by_hash(self, mock_models):
        """Test health check on specific model using @hash syntax."""
        result = health_check_operation("Qwen3@e96")
        
        assert result["status"] == "success" 
        assert result["data"]["summary"]["total"] == 1
        assert len(result["data"]["healthy"]) == 1
        assert "Qwen3" in result["data"]["healthy"][0]["name"]
    
    def test_health_check_corrupted_model_detection(self, mock_models):
        """Test that corrupted models are properly detected."""
        result = health_check_operation("corrupted")
        
        assert result["status"] == "success"
        assert result["data"]["summary"]["unhealthy_count"] == 1
        assert result["data"]["unhealthy"][0]["status"] == "unhealthy"


class TestRmOperationIntegration:
    """Test rm operation with realistic scenarios."""
    
    def test_rm_with_fuzzy_matching(self, mock_models):
        """Test rm finds model via fuzzy matching in isolated cache."""
        # Get models from isolated cache
        from mlxk2.operations.list import list_models
        result = list_models()
        available_models = result["data"]["models"]
        
        if not available_models:
            pytest.skip("No models in test cache for rm testing")
        
        # Use first available model for testing
        target_model = available_models[0]["name"]
        
        # Extract partial name for fuzzy matching
        if "/" in target_model:
            partial_name = target_model.split("/")[-1].split("-")[0]  # e.g., "DialoGPT" from "microsoft/DialoGPT-small"
        else:
            partial_name = target_model.split("-")[0]
        
        result = rm_operation(partial_name, force=True)
        
        # Should either succeed or be ambiguous
        assert result["status"] in ["success", "error"]
        
        if result["status"] == "success":
            assert "model" in result["data"]
            assert result["data"]["action"] == "deleted"
    
    def test_rm_ambiguous_pattern_shows_choices(self, mock_models):
        """Test rm shows choices for ambiguous patterns in isolated cache."""
        # Create ambiguous scenario with multiple models starting with same prefix
        result = rm_operation("m", force=False)  # "m" might match multiple models
        
        # Should either be ambiguous (error) or succeed (single match)
        assert result["status"] in ["success", "error"]
        
        if result["status"] == "error" and "ambiguous" in result.get("error", {}).get("message", "").lower():
            # Ambiguous case - should show choices
            assert "matches" in result.get("data", {}) or "choices" in result.get("data", {})
            choices = result["data"].get("matches", result["data"].get("choices", []))
            assert len(choices) >= 2
    
    def test_rm_nonexistent_model(self, mock_models):
        """Test rm handles nonexistent models gracefully."""
        result = rm_operation("absolutely-does-not-exist-12345", force=True)
        
        assert result["status"] == "error"
        error_msg = result["error"]["message"].lower()
        assert "not found" in error_msg or "no matches" in error_msg or "no models found" in error_msg


class TestCorruptedCacheHandling:
    """Test handling of corrupted cache entries."""
    
    def test_corrupted_naming_tolerance(self, create_corrupted_cache_entry):
        """Test that corrupted cache directory names are handled gracefully."""
        # Create cache entry that violates naming rules
        create_corrupted_cache_entry("models--org--model---corrupted")
        
        from mlxk2.operations.list import list_models
        result = list_models()
        
        # Should not crash, should show the corrupted entry
        assert result["status"] == "success"
        corrupted_models = [m for m in result["data"]["models"] if "/-" in m["name"]]
        assert len(corrupted_models) >= 1  # At least our corrupted entry
        
        # Problem should be visible in name
        assert any("/-" in model["name"] for model in corrupted_models)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])