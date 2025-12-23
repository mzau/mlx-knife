"""Tests for Issue #30: Gated Models Preflight Check"""

import pytest
from mlxk2.operations.pull import preflight_repo_access, pull_operation


def _create_mock_response(status_code=403):
    """Create a mock httpx.Response for huggingface-hub 1.x exceptions.

    Hub 1.x requires response parameter to be a real httpx.Response object.
    """
    try:
        import httpx
        # Create minimal mock response
        request = httpx.Request("GET", "https://huggingface.co/api/models/test")
        return httpx.Response(status_code=status_code, request=request)
    except ImportError:
        # Fallback for older hub versions that don't need it
        return None


def test_preflight_private_model_without_token(monkeypatch):
    """Test preflight check with a known private model without token.
    
    This is the core Issue #30 scenario: user tries to pull private/gated model
    without setting HUGGINGFACE_HUB_TOKEN, should fail fast at preflight.
    
    Uses BrokeC/broken_model - a small private test model.
    """
    # Ensure no token is set for this test
    # Ensure no tokens in environment
    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.delenv("HUGGINGFACE_HUB_TOKEN", raising=False)
    
    try:
        # Verify no token in environment (critical for test validity)
        import os
        assert "HF_TOKEN" not in os.environ
        assert "HUGGINGFACE_HUB_TOKEN" not in os.environ

        # Require huggingface_hub for this test (skip if missing)
        hub = pytest.importorskip("huggingface_hub")
        from huggingface_hub import HfApi
        from huggingface_hub import errors as _hub_errors
        GatedRepoError = _hub_errors.GatedRepoError
        def _fake_model_info(self, repo_id, token=None):
            response = _create_mock_response(status_code=403)
            raise GatedRepoError("Gated/private repository", response=response)
        monkeypatch.setattr(HfApi, "model_info", _fake_model_info, raising=True)

        success, error = preflight_repo_access("org/private-model")
        
        # Should fail fast without token
        assert success is False
        assert error is not None
        assert isinstance(error, str)
        # Should mention access/private/gated/denied
        assert any(keyword in error.lower() for keyword in ["access", "private", "gated", "denied", "token"])
        
    finally:
        pass


def test_preflight_nonexistent_model(monkeypatch):
    """Test preflight check with a non-existent model."""
    # Require huggingface_hub for this test (skip if missing)
    hub = pytest.importorskip("huggingface_hub")
    from huggingface_hub import HfApi
    from huggingface_hub import errors as _hub_errors
    RepositoryNotFoundError = _hub_errors.RepositoryNotFoundError
    def _fake_model_info(self, repo_id, token=None):
        response = _create_mock_response(status_code=404)
        raise RepositoryNotFoundError("Not found", response=response)
    monkeypatch.setattr(HfApi, "model_info", _fake_model_info, raising=True)

    success, error = preflight_repo_access("definitely-not-existing-model-12345-xyz")
    
    assert success is False
    assert error is not None
    # HuggingFace returns "access denied" even for non-existent models (security feature)
    assert any(keyword in error.lower() for keyword in ["not found", "access denied", "denied"])


def test_preflight_integration_in_pull(isolated_cache, monkeypatch):
    """Test that preflight check is properly integrated in pull operation.
    
    Uses isolated_cache fixture which creates:
    - Temporary cache under /var/folders/.../mlxk2_test_XXXXX/
    - Safety sentinel: models--TEST-CACHE-SENTINEL--mlxk2-safety-check
    - Proper HF_HOME override and MODEL_CACHE patching
    """
    # Require huggingface_hub for this test (skip if missing)
    hub = pytest.importorskip("huggingface_hub")
    from huggingface_hub import HfApi
    from huggingface_hub import errors as _hub_errors
    RepositoryNotFoundError = _hub_errors.RepositoryNotFoundError
    def _fake_model_info(self, repo_id, token=None):
        response = _create_mock_response(status_code=404)
        raise RepositoryNotFoundError("Not found", response=response)
    monkeypatch.setattr(HfApi, "model_info", _fake_model_info, raising=True)

    # Test with a non-existent model - should fail at preflight stage
    result = pull_operation("definitely-not-existing-model-12345-xyz")
    
    assert result["status"] == "error"
    assert result["data"]["download_status"] == "access_denied"
    assert result["error"]["type"] == "access_denied"
    # HuggingFace returns "access denied" even for non-existent models
    assert any(keyword in result["error"]["message"].lower() for keyword in ["not found", "access denied", "denied"])


def test_preflight_graceful_degradation():
    """Test that preflight check degrades gracefully on errors."""
    # Test with empty model name - should handle gracefully
    success, error = preflight_repo_access("")
    
    # Should either handle this gracefully or fail predictably
    assert isinstance(success, bool)
    if not success:
        assert isinstance(error, str)
        assert len(error) > 0


def test_preflight_mock_gated_scenario():
    """Test preflight behavior documentation for gated models."""
    # Note: We can't easily test actual gated models without tokens
    # This test documents the expected behavior
    
    # If we had a gated model, the expected flow would be:
    # 1. preflight_repo_access("meta-llama/Llama-2-7b-hf") -> (False, "gated")
    # 2. pull_operation should return access_denied without downloading anything
    
    # For now, we just verify the function exists and is importable
    assert callable(preflight_repo_access)
    
    # The function should handle import errors gracefully
    # (e.g., if huggingface_hub is not installed)
    try:
        success, error = preflight_repo_access("test-model")
        # Should not crash, even if the model doesn't exist
        assert isinstance(success, bool)
        assert error is None or isinstance(error, str)
    except Exception as e:
        pytest.fail(f"preflight_repo_access should not crash: {e}")


def test_preflight_prevents_cache_pollution(isolated_cache, monkeypatch):
    """Test that preflight check prevents cache pollution.
    
    This is the core value of Issue #30: failed access should not leave
    partial downloads in the cache.
    """
    from mlxk2.core.cache import MODEL_CACHE
    from conftest import assert_is_test_cache
    
    # Verify we're using test cache (safety)
    # MODEL_CACHE points to hub/, sentinel is in hub/, so check MODEL_CACHE directly
    assert_is_test_cache(MODEL_CACHE)
    
    # Require huggingface_hub for this test (skip if missing)
    hub = pytest.importorskip("huggingface_hub")
    from huggingface_hub import HfApi
    from huggingface_hub import errors as _hub_errors
    GatedRepoError = _hub_errors.GatedRepoError
    def _fake_model_info(self, repo_id, token=None):
        response = _create_mock_response(status_code=403)
        raise GatedRepoError("Gated/private repository", response=response)
    monkeypatch.setattr(HfApi, "model_info", _fake_model_info, raising=True)

    # Attempt to pull a gated/private model
    result = pull_operation("org/gated-model")
    
    # Should fail at preflight stage
    assert result["status"] == "error"
    assert result["data"]["download_status"] == "access_denied"
    
    # Cache should remain clean (no partial downloads)
    cache_contents = list(MODEL_CACHE.iterdir())
    # Only the sentinel should exist
    sentinel_exists = any("TEST-CACHE-SENTINEL" in item.name for item in cache_contents)
    assert sentinel_exists, "Test sentinel should exist"
    
    # No model directories should be created for the failed model
    model_dirs = [item for item in cache_contents if "gated-model" in item.name]
    assert len(model_dirs) == 0, "No partial model directories should exist after preflight failure"
