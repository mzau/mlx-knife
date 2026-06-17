"""Live E2E: /v1/models lists workspace models (Issue #58, ADR-022).

Server /v1/models must match the default human `mlxk list` view:
runnable (healthy + runtime_compatible) models from BOTH the HF cache
and MLXK_WORKSPACE_HOME. Workspace models are advertised by their
directory basename (stable short id), never by absolute path.

Safe: read-only — no cache writes, no workspace writes.

Requires:
- MLXK_WORKSPACE_HOME set and containing at least one runnable workspace model

Run:
- pytest -m wet -v  (included in wet-umbrella Phase 1)
- pytest tests_2.0/live/test_server_models_workspace_live.py -v
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

try:
    import httpx
except ImportError:
    httpx = None

from .server_context import LocalServer

ws_home = os.environ.get("MLXK_WORKSPACE_HOME")
ws_home_valid = ws_home and Path(ws_home).is_dir()

pytestmark = [
    pytest.mark.live,
    pytest.mark.wet,
    pytest.mark.skipif(
        not ws_home_valid,
        reason="MLXK_WORKSPACE_HOME not set or not a directory",
    ),
    pytest.mark.skipif(httpx is None, reason="httpx required for E2E tests"),
]


def _runnable_view():
    """Default human `mlxk list` view: runnable models from cache + workspace.

    Same filter as render_list (no --all): healthy AND runtime_compatible.
    """
    from mlxk2.operations.list import list_models

    models = list_models()["data"]["models"]
    runnable = [
        m for m in models
        if m.get("health") == "healthy" and m.get("runtime_compatible")
        # Embedders are runnable via `mlxk embed` but a bare `serve` can't serve them, so
        # /v1/models omits them (ADR-015; the GET /v1/models embed-merge is deferred to 2.1).
        # Match serve's predicate exactly (handlers/models.py): exclude "embeddings" capability.
        and "embeddings" not in (m.get("capabilities") or [])
    ]
    workspace = {Path(m["name"]).name for m in runnable if not m.get("cached", True)}
    cache = {m["name"] for m in runnable if m.get("cached", True)}
    return workspace, cache


def _get_models(server_url: str) -> list:
    resp = httpx.get(f"{server_url}/v1/models", timeout=60.0)
    assert resp.status_code == 200
    payload = resp.json()
    assert payload.get("object") == "list"
    return payload["data"]


def _smallest_chat_workspace() -> dict | None:
    """Pick the smallest runnable chat-capable text workspace model for preload."""
    from mlxk2.operations.list import list_models

    models = list_models()["data"]["models"]
    candidates = [
        m for m in models
        if not m.get("cached", True)
        and m.get("health") == "healthy"
        and m.get("runtime_compatible")
        and "chat" in (m.get("capabilities") or [])
        and "vision" not in (m.get("capabilities") or [])
        and "audio" not in (m.get("capabilities") or [])
        and (m.get("size_bytes") or 0) < 2 * 1024**3  # keep preload cheap
    ]
    if not candidates:
        return None
    return min(candidates, key=lambda m: m.get("size_bytes") or 0)


def test_v1_models_matches_default_list_view():
    """/v1/models == runnable cache+workspace models (mlxk list without --all)."""
    expected_ws, expected_cache = _runnable_view()
    if not expected_ws:
        pytest.skip("No runnable workspace models in MLXK_WORKSPACE_HOME")

    with LocalServer(model=None, timeout=60) as server_url:
        data = _get_models(server_url)

    ids = [m["id"] for m in data]
    assert len(ids) == len(set(ids)), f"duplicate ids in /v1/models: {ids}"

    served_ws = {m["id"] for m in data if m["owned_by"] == "workspace"}
    served_cache = {m["id"] for m in data if m["owned_by"] != "workspace"}

    # Workspace models advertised by basename, never absolute path
    leaked_paths = [i for i in served_ws if i.startswith("/")]
    assert not leaked_paths, f"absolute paths leaked into /v1/models: {leaked_paths}"

    assert served_ws == expected_ws
    assert served_cache == expected_cache


def test_v1_models_preloaded_workspace_dedup_and_chat():
    """A preloaded workspace model appears once (basename, sorted first) and serves chat."""
    chosen = _smallest_chat_workspace()
    if chosen is None:
        pytest.skip("No small runnable chat workspace model in MLXK_WORKSPACE_HOME")

    preload_path = chosen["name"]  # absolute path (the Issue #58 repro setup)
    basename = Path(preload_path).name

    with LocalServer(model=preload_path, timeout=120) as server_url:
        data = _get_models(server_url)
        ids = [m["id"] for m in data]

        # Exactly one entry for the preloaded model: basename, not absolute path
        assert ids.count(basename) == 1, f"expected one '{basename}' entry, got: {ids}"
        assert preload_path not in ids, "preloaded workspace advertised by absolute path"
        # Preloaded model is sorted first
        assert ids[0] == basename

        # Client can use the advertised id (workspace-first resolution)
        resp = httpx.post(
            f"{server_url}/v1/chat/completions",
            json={
                "model": basename,
                "messages": [{"role": "user", "content": "Say OK"}],
                "max_tokens": 8,
                "stream": False,
            },
            timeout=120.0,
        )
        assert resp.status_code == 200, resp.text
        payload = resp.json()
        assert payload.get("choices"), payload
