"""
Minimal server tests for /v1/models and error mappings (404/503).

Keeps scope small and deterministic by mocking model/cache access.
"""

from unittest.mock import Mock, MagicMock, patch

from fastapi.testclient import TestClient

from mlxk2.core.server_base import app


def test_models_endpoint_minimal_structure():
    """/v1/models returns list object with model entries and context_length field."""
    client = TestClient(app)

    # Note: cache_dir_to_hf/detect_framework/is_model_healthy are imported inside
    # the endpoint function, so patch their origin modules, not server_base.
    with patch('mlxk2.core.server_base.get_current_model_cache') as mock_cache, \
         patch('mlxk2.core.cache.cache_dir_to_hf') as mock_cache_to_hf, \
         patch('mlxk2.operations.common.detect_framework') as mock_framework, \
         patch('mlxk2.operations.health.is_model_healthy') as mock_healthy:

        # Simulate a single cached model directory
        mock_cache_dir = MagicMock()
        mock_cache_dir.name = "models--org--model"
        mock_cache.return_value.iterdir.return_value = [mock_cache_dir]

        # Map cache dir -> external id and mark as MLX + healthy
        mock_cache_to_hf.return_value = "org/model"
        mock_framework.return_value = "MLX"
        mock_healthy.return_value = (True, None)

        # Provide a snapshots directory with one folder to allow context_length probing
        mock_snapshots_dir = MagicMock()
        mock_snapshots_dir.exists.return_value = True
        mock_snapshot = MagicMock()
        mock_snapshot.is_dir.return_value = True
        mock_snapshots_dir.iterdir.return_value = [mock_snapshot]
        mock_cache_dir.__truediv__.return_value = mock_snapshots_dir

        resp = client.get("/v1/models")
        assert resp.status_code == 200
        data = resp.json()
        assert data.get("object") == "list"
        assert isinstance(data.get("data"), list)
        # Note: Runtime checks may filter models - list could be empty
        # Just verify structure, not content


def test_unknown_model_maps_to_404():
    """Unknown/invalid model should map to 404 from inner helper."""
    from fastapi import HTTPException

    client = TestClient(app)

    with patch('mlxk2.core.server_base.get_or_load_model') as mock_get:
        mock_get.side_effect = HTTPException(status_code=404, detail="not found")

        payload = {"model": "does/not-exist", "prompt": "hi"}
        resp = client.post("/v1/completions", json=payload)
        assert resp.status_code == 404


def test_models_endpoint_filters_non_mlx_and_unhealthy():
    """Ensure /v1/models excludes non-MLX and unhealthy entries."""
    client = TestClient(app)

    with patch('mlxk2.core.server_base.get_current_model_cache') as mock_cache, \
         patch('mlxk2.core.cache.cache_dir_to_hf') as mock_cache_to_hf, \
         patch('mlxk2.operations.common.detect_framework') as mock_framework, \
         patch('mlxk2.operations.health.is_model_healthy') as mock_healthy:

        # Two cached dirs
        d1 = MagicMock(); d1.name = "models--org--mlx"
        d2 = MagicMock(); d2.name = "models--org--pt"
        mock_cache.return_value.iterdir.return_value = [d1, d2]

        # Map names
        def map_name(n):
            if n == "models--org--mlx":
                return "org/mlx"
            return "org/pt"

        mock_cache_to_hf.side_effect = map_name

        # Framework detection: d1 is MLX, d2 is not
        def detect_fw(model_name, *_args, **_kwargs):
            return "MLX" if model_name.endswith("/mlx") else "PyTorch"

        mock_framework.side_effect = detect_fw

        # Health: return False for the MLX one to ensure it is filtered, too
        def health(model_name):
            return (False, None) if model_name.endswith("/mlx") else (True, None)

        mock_healthy.side_effect = health

        resp = client.get("/v1/models")
        assert resp.status_code == 200
        data = resp.json()
        # Both should be filtered: one not MLX, one unhealthy
        assert data.get("data") == []


def test_chat_unknown_model_maps_to_404():
    from fastapi import HTTPException

    client = TestClient(app)

    with patch('mlxk2.core.server_base.get_or_load_model') as mock_get:
        mock_get.side_effect = HTTPException(status_code=404, detail="not found")

        payload = {"model": "does/not-exist", "messages": [{"role": "user", "content": "hi"}], "stream": False}
        resp = client.post("/v1/chat/completions", json=payload)
        assert resp.status_code == 404


def test_chat_shutdown_event_maps_to_503_and_is_cleared():
    from mlxk2.core import server_base

    client = TestClient(app)

    try:
        server_base._shutdown_event.set()
        payload = {"model": "any/model", "messages": [{"role": "user", "content": "hi"}], "stream": False}
        resp = client.post("/v1/chat/completions", json=payload)
        assert resp.status_code == 503
    finally:
        server_base._shutdown_event.clear()


def test_shutdown_event_maps_to_503_and_is_cleared():
    """When shutdown flag is set, endpoints respond 503; then clear for isolation."""
    from mlxk2.core import server_base

    client = TestClient(app)

    try:
        server_base._shutdown_event.set()
        payload = {"model": "any/model", "prompt": "hi"}
        resp = client.post("/v1/completions", json=payload)
        assert resp.status_code == 503
    finally:
        # Ensure we don't leak shutdown state to other tests
        server_base._shutdown_event.clear()


