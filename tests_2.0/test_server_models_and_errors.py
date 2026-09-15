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


def test_models_endpoint_excludes_embedders(monkeypatch):
    """ADR-015 Slice C: serve /v1/models hides runnable embedders.

    bge/qwen3-embedders become runtime_compatible=True (so `mlxk list` shows them), but serve's
    chat-surface /v1/models must not advertise them — the embed-backend merge is deferred and the
    response carries no capability field, so listing them would be a list↔verb contradiction.
    """
    monkeypatch.delenv("MLXK_WORKSPACE_HOME", raising=False)
    client = TestClient(app)

    with patch('mlxk2.core.server_base.get_current_model_cache') as mock_cache, \
         patch('mlxk2.core.cache.cache_dir_to_hf') as mock_cache_to_hf, \
         patch('mlxk2.operations.common.build_model_object') as mock_build:

        d1 = MagicMock(); d1.name = "models--org--chat-model"
        d2 = MagicMock(); d2.name = "models--org--bge-embedder"
        for d in [d1, d2]:
            snapshot_dir = MagicMock()
            snapshot_path = MagicMock()
            snapshot_dir.exists.return_value = True
            snapshot_dir.iterdir.return_value = [snapshot_path]
            snapshot_path.is_dir.return_value = True
            d.__truediv__ = lambda self, x, snap=snapshot_dir, spath=snapshot_path: snap if x == "snapshots" else spath
        mock_cache.return_value.iterdir.return_value = [d1, d2]

        def map_name(n):
            return n.replace("models--", "").replace("--", "/")
        mock_cache_to_hf.side_effect = map_name

        # Both are healthy + runtime_compatible; only the embedder carries the embeddings capability.
        def build(model_name, model_dir, selected_path):
            if "embedder" in model_name:
                return {"health": "healthy", "runtime_compatible": True, "capabilities": ["embeddings"]}
            return {"health": "healthy", "runtime_compatible": True, "capabilities": ["text-generation", "chat"]}
        mock_build.side_effect = build

        resp = client.get("/v1/models")
        assert resp.status_code == 200
        ids = [m["id"] for m in resp.json().get("data", [])]
        assert "org/chat-model" in ids
        assert "org/bge-embedder" not in ids


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


def test_health_answers_ok_not_healthy():
    """ADR-029: the status code is the answer; `healthy` is the CLI's file-integrity word."""
    resp = TestClient(app).get("/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok", "service": "mlx-knife-server-2.0"}
    assert "healthy" not in resp.text


def test_models_rows_mark_the_loaded_workspace_model(tmp_path, monkeypatch):
    """Every row carries `loaded`; only the model in memory is true — none while nothing is."""
    from mlxk2.core.server.model_manager import model_dir_identity

    ws_home = tmp_path / "workspaces"
    ws_home.mkdir()
    _make_workspace(ws_home, "a-model")
    in_memory = _make_workspace(ws_home, "b-model")
    monkeypatch.setenv("MLXK_WORKSPACE_HOME", str(ws_home))

    client = TestClient(app)
    with patch('mlxk2.core.server_base.get_current_model_cache') as mock_cache, \
         patch('mlxk2.operations.common.build_model_object') as mock_build:
        mock_cache.return_value.exists.return_value = False
        mock_build.return_value = {"health": "healthy", "runtime_compatible": True}

        with patch('mlxk2.core.server_base._model_manager', None):
            rows = client.get("/v1/models").json()["data"]
        assert [(m["id"], m["loaded"]) for m in rows] == [("a-model", False), ("b-model", False)]

        # The manager and the row meet on the directory, not on a name
        manager = Mock(loaded_identity=model_dir_identity(in_memory))
        with patch('mlxk2.core.server_base._model_manager', manager):
            rows = client.get("/v1/models").json()["data"]
        assert [(m["id"], m["loaded"]) for m in rows] == [("a-model", False), ("b-model", True)]


def test_models_rows_mark_the_loaded_cache_model(tmp_path, monkeypatch):
    """A cached model's row is marked by its cache directory; a workspace row beside it stays false."""
    ws_home = tmp_path / "workspaces"
    ws_home.mkdir()
    _make_workspace(ws_home, "ws-model")
    monkeypatch.setenv("MLXK_WORKSPACE_HOME", str(ws_home))

    client = TestClient(app)
    with patch('mlxk2.core.server_base.get_current_model_cache') as mock_cache, \
         patch('mlxk2.core.cache.cache_dir_to_hf') as mock_cache_to_hf, \
         patch('mlxk2.operations.common.build_model_object') as mock_build, \
         patch('mlxk2.core.server.handlers.models.model_dir_identity', lambda d: ("dev", d.name)), \
         patch('mlxk2.core.server_base._model_manager', Mock(loaded_identity=("dev", "models--org--cache-model"))):

        d1 = MagicMock()
        d1.name = "models--org--cache-model"
        snapshot_dir = MagicMock()
        snapshot_dir.exists.return_value = True
        snapshot_dir.iterdir.return_value = []
        d1.__truediv__ = lambda self, x: snapshot_dir
        mock_cache.return_value.exists.return_value = True
        mock_cache.return_value.iterdir.return_value = [d1]
        mock_cache_to_hf.return_value = "org/cache-model"
        mock_build.return_value = {"health": "healthy", "runtime_compatible": True}

        by_id = {m["id"]: m["loaded"] for m in client.get("/v1/models").json()["data"]}
        assert by_id == {"org/cache-model": True, "ws-model": False}


def test_models_rows_mark_a_preload_outside_the_home(tmp_path, monkeypatch):
    """A preload listed under its path is marked like any other row."""
    from mlxk2.core.server.model_manager import model_dir_identity

    ws_home = tmp_path / "workspaces"
    ws_home.mkdir()
    _make_workspace(ws_home, "a-model")
    external = _make_workspace(tmp_path / "elsewhere", "ext-model")
    monkeypatch.setenv("MLXK_WORKSPACE_HOME", str(ws_home))

    client = TestClient(app)
    with patch('mlxk2.core.server_base.get_current_model_cache') as mock_cache, \
         patch('mlxk2.operations.common.build_model_object') as mock_build, \
         patch('mlxk2.core.server_base._preload_model', str(external)), \
         patch('mlxk2.core.server_base._model_manager', Mock(loaded_identity=model_dir_identity(external))):
        mock_cache.return_value.exists.return_value = False
        mock_build.return_value = {"health": "healthy", "runtime_compatible": True}

        rows = client.get("/v1/models").json()["data"]
        assert [(m["id"], m["loaded"]) for m in rows] == [(str(external), True), ("a-model", False)]


def test_models_rows_carry_the_window_of_the_model_object(tmp_path, monkeypatch):
    """Each row's `context_length` is the one build_model_object read — for workspace, cache and
    preloaded rows alike. The configs on disk state another number, which a second reading would
    report instead."""
    from pathlib import Path

    stated = '{"model_type": "llama", "max_position_embeddings": 8192}'
    ws_home = tmp_path / "workspaces"
    ws_home.mkdir()
    _make_workspace(ws_home, "ws-model", config=stated)
    external = _make_workspace(tmp_path / "elsewhere", "ext-model", config=stated)
    monkeypatch.setenv("MLXK_WORKSPACE_HOME", str(ws_home))

    windows = {"ws-model": 1001, "org/cache-model": 1002, "ext-model": 1003}

    def build(hf_name, model_root, selected_path):
        window = windows.get(hf_name) or windows[Path(hf_name).name]
        return {"health": "healthy", "runtime_compatible": True, "context_length": window}

    client = TestClient(app)
    with patch('mlxk2.core.server_base.get_current_model_cache') as mock_cache, \
         patch('mlxk2.core.cache.cache_dir_to_hf') as mock_cache_to_hf, \
         patch('mlxk2.operations.common.build_model_object', side_effect=build), \
         patch('mlxk2.core.server_base._preload_model', str(external)):
        d1 = MagicMock()
        d1.name = "models--org--cache-model"
        snapshot_dir = MagicMock()
        snapshot_dir.exists.return_value = True
        snapshot_dir.iterdir.return_value = []
        d1.__truediv__ = lambda self, x: snapshot_dir
        mock_cache.return_value.exists.return_value = True
        mock_cache.return_value.iterdir.return_value = [d1]
        mock_cache_to_hf.return_value = "org/cache-model"

        rows = client.get("/v1/models").json()["data"]

    assert {m["id"]: m["context_length"] for m in rows} == {
        str(external): 1003, "org/cache-model": 1002, "ws-model": 1001,
    }


def test_models_listing_runs_off_the_event_loop():
    """The scan reads every model directory; on the loop, GET /health waited for it."""
    import asyncio

    seen = []

    def listing(**kwargs):
        try:
            asyncio.get_running_loop()
            seen.append("on the loop")
        except RuntimeError:
            seen.append("off the loop")
        return {"object": "list", "data": []}

    with patch('mlxk2.core.server_base._handle_list_models_impl', listing):
        assert TestClient(app).get("/v1/models").status_code == 200
    assert seen == ["off the loop"]


def _manager_with_fake_loads(tmp_path, monkeypatch, names):
    """A ModelManager over a cache of empty model directories, resolving names from `names`."""
    import threading
    from mlxk2.core.cache import hf_to_cache_dir
    from mlxk2.core.server.model_manager import ModelManager

    cache = tmp_path / "hub"
    for hf_name in set(names.values()):
        (cache / hf_to_cache_dir(hf_name)).mkdir(parents=True)
    monkeypatch.setattr("mlxk2.core.cache.get_current_model_cache", lambda: cache)
    monkeypatch.setattr("mlxk2.core.model_resolution.resolve_model_for_operation",
                        lambda spec: (names.get(spec), None, None))
    manager = ModelManager(threading.Event())
    monkeypatch.setattr(manager, "_wait_for_memory", lambda *args: None)
    return manager, cache


def test_model_manager_serves_another_spelling_from_memory(tmp_path, monkeypatch):
    """The cache is keyed by the directory a spec reaches: requesting the listed id of a model
    loaded as `qwen` used to load it a second time."""
    from mlxk2.core.server.model_manager import model_dir_identity

    manager, cache = _manager_with_fake_loads(
        tmp_path, monkeypatch, {"qwen": "org/qwen", "org/qwen": "org/qwen", "llama": "org/llama"})
    loads = []
    monkeypatch.setattr(manager, "_load_text_or_vision_model",
                        lambda spec, verbose: loads.append(spec) or Mock(spec=[]))

    first = manager.get_or_load_model("qwen")
    assert manager.get_or_load_model("org/qwen") is first
    assert manager.get_or_load_model("qwen") is first
    assert loads == ["qwen"]
    assert manager.loaded_identity == model_dir_identity(cache / "models--org--qwen")

    manager.get_or_load_model("llama")
    assert loads == ["qwen", "llama"]
    assert manager.loaded_identity == model_dir_identity(cache / "models--org--llama")


def _fake_audio_runner_class(monkeypatch):
    import sys
    import types

    # The real module patches mlx-audio on import, which the stubbed mlx here cannot take.
    fake = types.ModuleType("mlxk2.core.audio_runner")
    fake.AudioRunner = type("AudioRunner", (), {})
    monkeypatch.setitem(sys.modules, "mlxk2.core.audio_runner", fake)
    return fake.AudioRunner


def test_model_manager_serves_another_spelling_of_an_audio_model_from_memory(tmp_path, monkeypatch):
    AudioRunner = _fake_audio_runner_class(monkeypatch)
    manager, _ = _manager_with_fake_loads(
        tmp_path, monkeypatch, {"whisper": "org/whisper", "org/whisper": "org/whisper"})
    loads = []
    monkeypatch.setattr(manager, "_load_audio_model",
                        lambda spec, verbose: loads.append(spec) or AudioRunner())

    first = manager.get_or_load_audio_model("whisper")
    assert manager.get_or_load_audio_model("org/whisper") is first
    assert loads == ["whisper"]


def test_model_manager_never_hands_a_text_request_the_audio_runner(tmp_path, monkeypatch):
    """One model directory, two runner kinds: a text request for a loaded Whisper goes through
    the text loader, where an AudioRunner would fail it with a 500."""
    AudioRunner = _fake_audio_runner_class(monkeypatch)
    manager, _ = _manager_with_fake_loads(tmp_path, monkeypatch, {"whisper": "org/whisper"})
    monkeypatch.setattr(manager, "_load_audio_model", lambda spec, verbose: AudioRunner())
    text_loads = []
    monkeypatch.setattr(manager, "_load_text_or_vision_model",
                        lambda spec, verbose: text_loads.append(spec) or Mock(spec=[]))

    manager.get_or_load_audio_model("whisper")
    assert not isinstance(manager.get_or_load_model("whisper"), AudioRunner)
    assert text_loads == ["whisper"]


def test_model_manager_resolves_spellings_to_one_workspace(tmp_path, monkeypatch):
    """Resolution itself, unmocked: a workspace named by path, by basename and — on a
    case-insensitive volume — in another case is one cache key, and it is the identity
    GET /v1/models compares its row against."""
    import threading
    from mlxk2.core.server.model_manager import ModelManager, model_dir_identity

    ws_home = tmp_path / "workspaces"
    ws_home.mkdir()
    ws = _make_workspace(ws_home, "Qwen-ws")
    monkeypatch.setenv("MLXK_WORKSPACE_HOME", str(ws_home))
    manager = ModelManager(threading.Event())

    expected = model_dir_identity(ws)
    spellings = [str(ws), "Qwen-ws"]
    if (ws_home / "qwen-WS").exists():  # case-insensitive volume
        spellings.append("qwen-WS")
    assert {manager._resolve(spec)[0] for spec in spellings} == {expected}


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


