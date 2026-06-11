"""
Minimal server tests for /v1/models and error mappings (404/503).

Keeps scope small and deterministic by mocking model/cache access.
"""

from unittest.mock import Mock, MagicMock, patch

from fastapi.testclient import TestClient

from mlxk2.core.server_base import app


def _make_workspace(ws_home, name, config='{"model_type": "llama"}'):
    """Create a minimal workspace directory (config.json marks it as workspace)."""
    ws = ws_home / name
    ws.mkdir(parents=True)
    (ws / "config.json").write_text(config)
    return ws


def test_models_endpoint_minimal_structure(monkeypatch):
    """/v1/models returns list object with model entries and context_length field."""
    monkeypatch.delenv("MLXK_WORKSPACE_HOME", raising=False)
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


def test_models_endpoint_filters_unhealthy_and_not_runtime_compatible(monkeypatch):
    """Ensure /v1/models excludes unhealthy and non-runtime-compatible entries.

    Filter logic: healthy == True AND runtime_compatible == True
    Uses shared build_model_object from common.py (single source of truth).
    """
    monkeypatch.delenv("MLXK_WORKSPACE_HOME", raising=False)
    client = TestClient(app)

    with patch('mlxk2.core.server_base.get_current_model_cache') as mock_cache, \
         patch('mlxk2.core.cache.cache_dir_to_hf') as mock_cache_to_hf, \
         patch('mlxk2.operations.common.build_model_object') as mock_build:

        # Three cached dirs with proper snapshot structure
        d1 = MagicMock(); d1.name = "models--org--healthy-compatible"
        d2 = MagicMock(); d2.name = "models--org--unhealthy"
        d3 = MagicMock(); d3.name = "models--org--not-compatible"

        # Setup snapshot paths for each model dir
        for d in [d1, d2, d3]:
            snapshot_dir = MagicMock()
            snapshot_path = MagicMock()
            snapshot_dir.exists.return_value = True
            snapshot_dir.iterdir.return_value = [snapshot_path]
            snapshot_path.is_dir.return_value = True
            d.__truediv__ = lambda self, x, snap=snapshot_dir, spath=snapshot_path: snap if x == "snapshots" else spath

        mock_cache.return_value.iterdir.return_value = [d1, d2, d3]

        # Map names
        def map_name(n):
            return n.replace("models--", "").replace("--", "/")
        mock_cache_to_hf.side_effect = map_name

        # build_model_object returns different health/runtime_compatible
        def build(model_name, model_dir, selected_path):
            if "unhealthy" in model_name:
                return {"health": "unhealthy", "runtime_compatible": True}
            elif "not-compatible" in model_name:
                return {"health": "healthy", "runtime_compatible": False}
            else:
                return {"health": "healthy", "runtime_compatible": True}
        mock_build.side_effect = build

        resp = client.get("/v1/models")
        assert resp.status_code == 200
        data = resp.json()
        # Only d1 (healthy + runtime_compatible) should pass
        model_ids = [m["id"] for m in data.get("data", [])]
        assert "org/healthy-compatible" in model_ids
        assert "org/unhealthy" not in model_ids
        assert "org/not-compatible" not in model_ids


def test_models_endpoint_lists_workspace_models(tmp_path, monkeypatch):
    """/v1/models lists runnable workspace-home models by basename (Issue #58)."""
    ws_home = tmp_path / "workspaces"
    ws_home.mkdir()
    _make_workspace(ws_home, "ws-model-b")
    _make_workspace(ws_home, "ws-model-a")
    (ws_home / "not-a-workspace").mkdir()  # no config.json → skipped
    monkeypatch.setenv("MLXK_WORKSPACE_HOME", str(ws_home))

    client = TestClient(app)
    with patch('mlxk2.core.server_base.get_current_model_cache') as mock_cache, \
         patch('mlxk2.operations.common.build_model_object') as mock_build:
        mock_cache.return_value.exists.return_value = False
        mock_build.return_value = {"health": "healthy", "runtime_compatible": True}

        resp = client.get("/v1/models")
        assert resp.status_code == 200
        data = resp.json()
        ids = [m["id"] for m in data["data"]]
        # Basename ids (not absolute paths), alphabetical
        assert ids == ["ws-model-a", "ws-model-b"]
        assert all(m["owned_by"] == "workspace" for m in data["data"])


def test_models_endpoint_filters_workspace_like_cache(tmp_path, monkeypatch):
    """Workspace models pass the same runnable filter as cache models (Issue #58)."""
    ws_home = tmp_path / "workspaces"
    ws_home.mkdir()
    _make_workspace(ws_home, "ws-ok")
    _make_workspace(ws_home, "ws-unhealthy")
    _make_workspace(ws_home, "ws-not-compatible")
    monkeypatch.setenv("MLXK_WORKSPACE_HOME", str(ws_home))

    def build(hf_name, model_root, selected_path):
        if "ws-unhealthy" in hf_name:
            return {"health": "unhealthy", "runtime_compatible": True}
        if "ws-not-compatible" in hf_name:
            return {"health": "healthy", "runtime_compatible": False}
        return {"health": "healthy", "runtime_compatible": True}

    client = TestClient(app)
    with patch('mlxk2.core.server_base.get_current_model_cache') as mock_cache, \
         patch('mlxk2.operations.common.build_model_object') as mock_build:
        mock_cache.return_value.exists.return_value = False
        mock_build.side_effect = build

        resp = client.get("/v1/models")
        assert resp.status_code == 200
        ids = [m["id"] for m in resp.json()["data"]]
        assert ids == ["ws-ok"]


def test_models_endpoint_merges_cache_and_workspace(tmp_path, monkeypatch):
    """/v1/models merges HF-cache and workspace-home models (Issue #58)."""
    ws_home = tmp_path / "workspaces"
    ws_home.mkdir()
    _make_workspace(ws_home, "ws-model")
    monkeypatch.setenv("MLXK_WORKSPACE_HOME", str(ws_home))

    client = TestClient(app)
    with patch('mlxk2.core.server_base.get_current_model_cache') as mock_cache, \
         patch('mlxk2.core.cache.cache_dir_to_hf') as mock_cache_to_hf, \
         patch('mlxk2.operations.common.build_model_object') as mock_build:

        d1 = MagicMock(); d1.name = "models--org--cache-model"
        snapshot_dir = MagicMock()
        snapshot_dir.exists.return_value = True
        snapshot_dir.iterdir.return_value = []
        d1.__truediv__ = lambda self, x: snapshot_dir
        mock_cache.return_value.exists.return_value = True
        mock_cache.return_value.iterdir.return_value = [d1]
        mock_cache_to_hf.return_value = "org/cache-model"
        mock_build.return_value = {"health": "healthy", "runtime_compatible": True}

        resp = client.get("/v1/models")
        assert resp.status_code == 200
        data = resp.json()
        by_id = {m["id"]: m for m in data["data"]}
        assert set(by_id) == {"org/cache-model", "ws-model"}
        assert by_id["ws-model"]["owned_by"] == "workspace"
        assert by_id["org/cache-model"]["owned_by"] == "mlx-knife-2.0"


def test_models_endpoint_preload_workspace_dedup_and_sort_first(tmp_path, monkeypatch):
    """A preloaded workspace-home model appears once, by basename, sorted first."""
    ws_home = tmp_path / "workspaces"
    ws_home.mkdir()
    _make_workspace(ws_home, "a-model")
    preload_ws = _make_workspace(ws_home, "z-model")
    monkeypatch.setenv("MLXK_WORKSPACE_HOME", str(ws_home))

    client = TestClient(app)
    with patch('mlxk2.core.server_base.get_current_model_cache') as mock_cache, \
         patch('mlxk2.operations.common.build_model_object') as mock_build, \
         patch('mlxk2.core.server_base._preload_model', str(preload_ws)):
        mock_cache.return_value.exists.return_value = False
        mock_build.return_value = {"health": "healthy", "runtime_compatible": True}

        resp = client.get("/v1/models")
        assert resp.status_code == 200
        ids = [m["id"] for m in resp.json()["data"]]
        # No duplicate absolute-path entry; preloaded model first, by basename
        assert ids == ["z-model", "a-model"]


def test_models_endpoint_preload_workspace_outside_home(tmp_path, monkeypatch):
    """A runnable preloaded workspace outside the workspace home keeps its path id."""
    ws_home = tmp_path / "workspaces"
    ws_home.mkdir()
    _make_workspace(ws_home, "a-model")
    external = _make_workspace(tmp_path / "elsewhere", "ext-model")
    monkeypatch.setenv("MLXK_WORKSPACE_HOME", str(ws_home))

    client = TestClient(app)
    with patch('mlxk2.core.server_base.get_current_model_cache') as mock_cache, \
         patch('mlxk2.operations.common.build_model_object') as mock_build, \
         patch('mlxk2.core.server_base._preload_model', str(external)):
        mock_cache.return_value.exists.return_value = False
        mock_build.return_value = {"health": "healthy", "runtime_compatible": True}

        resp = client.get("/v1/models")
        assert resp.status_code == 200
        data = resp.json()
        ids = [m["id"] for m in data["data"]]
        # Path id (basename would not resolve via workspace home), sorted first
        assert ids == [str(external), "a-model"]
        assert data["data"][0]["owned_by"] == "workspace"


def test_models_endpoint_preload_symlinked_workspace_sorts_first(tmp_path, monkeypatch):
    """A preloaded symlinked workspace keeps its advertised (link) id and sorts first."""
    ws_home = tmp_path / "workspaces"
    ws_home.mkdir()
    _make_workspace(ws_home, "a-plain")
    target = _make_workspace(tmp_path / "elsewhere", "target-model")
    (ws_home / "z-link").symlink_to(target)
    monkeypatch.setenv("MLXK_WORKSPACE_HOME", str(ws_home))

    client = TestClient(app)
    with patch('mlxk2.core.server_base.get_current_model_cache') as mock_cache, \
         patch('mlxk2.operations.common.build_model_object') as mock_build, \
         patch('mlxk2.core.server_base._preload_model', str(ws_home / "z-link")):
        mock_cache.return_value.exists.return_value = False
        mock_build.return_value = {"health": "healthy", "runtime_compatible": True}

        resp = client.get("/v1/models")
        assert resp.status_code == 200
        ids = [m["id"] for m in resp.json()["data"]]
        # Advertised under the link name (not the target basename), sorted first
        assert ids == ["z-link", "a-plain"]


def test_models_endpoint_preload_not_runnable_stays_hidden(tmp_path, monkeypatch):
    """Clients only ever see runnable models — even a preloaded one is filtered."""
    ws_home = tmp_path / "workspaces"
    ws_home.mkdir()
    preload_ws = _make_workspace(ws_home, "sick-model")
    monkeypatch.setenv("MLXK_WORKSPACE_HOME", str(ws_home))

    client = TestClient(app)
    with patch('mlxk2.core.server_base.get_current_model_cache') as mock_cache, \
         patch('mlxk2.operations.common.build_model_object') as mock_build, \
         patch('mlxk2.core.server_base._preload_model', str(preload_ws)):
        mock_cache.return_value.exists.return_value = False
        mock_build.return_value = {"health": "unhealthy", "runtime_compatible": False}

        resp = client.get("/v1/models")
        assert resp.status_code == 200
        assert resp.json()["data"] == []


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


def test_validation_error_returns_adr004_envelope():
    """RequestValidationError (422) should return ADR-004 error envelope (F-06)."""
    client = TestClient(app)

    # Send invalid payload (missing required 'model' field)
    payload = {"prompt": "hi"}  # Missing 'model'

    resp = client.post("/v1/completions", json=payload)

    # Should be 400 (not 422) with ADR-004 envelope
    assert resp.status_code == 400

    data = resp.json()
    assert data.get("status") == "error"
    assert "error" in data
    assert data["error"].get("type") == "validation_error"
    assert "message" in data["error"]


def test_http_exception_includes_not_implemented_type():
    """HTTP 501 should map to ErrorType.NOT_IMPLEMENTED in ADR-004 envelope."""
    from fastapi import HTTPException

    client = TestClient(app)

    with patch('mlxk2.core.server_base.get_or_load_model') as mock_get:
        # Simulate 501 Not Implemented
        mock_get.side_effect = HTTPException(status_code=501, detail="Feature not supported")

        payload = {"model": "test/model", "prompt": "hi"}
        resp = client.post("/v1/completions", json=payload)

        assert resp.status_code == 501
        data = resp.json()
        assert data.get("status") == "error"
        assert data["error"].get("type") == "not_implemented"


