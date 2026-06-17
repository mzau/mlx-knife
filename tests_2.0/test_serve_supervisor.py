"""Regression tests for the _run_supervised_uvicorn refactor (ADR-015 Slice D1 reuse seam).

The refactor added `module` + `extra_env` kwargs. Defaults MUST preserve serve's behavior
(targets mlxk2.core.server_base); embed-serve passes the embed module + extra env. subprocess.Popen
is mocked so nothing actually boots.

Also covers the D2 config bridge: start_server validates --embed-backend and exports it as
MLXK2_EMBED_BACKEND (which _run_supervised_uvicorn then copies into the server subprocess env).
"""

import os

import pytest
from unittest.mock import patch

import mlxk2.operations.serve as serve_mod


class _FakeProc:
    pid = 12345

    def wait(self):
        return 0

    def poll(self):
        return 0


def _run_with_capture(**kwargs):
    captured = {}

    def fake_popen(cmd, env=None, **kw):
        captured["cmd"] = cmd
        captured["env"] = env
        return _FakeProc()

    with patch.object(serve_mod.subprocess, "Popen", side_effect=fake_popen):
        rc = serve_mod._run_supervised_uvicorn("127.0.0.1", kwargs.pop("port", 8000), "info", **kwargs)
    return rc, captured


def test_default_targets_server_base():
    rc, cap = _run_with_capture(port=8000)
    assert rc == 0
    assert cap["cmd"][-2:] == ["-m", "mlxk2.core.server_base"]
    assert cap["env"]["MLXK2_HOST"] == "127.0.0.1"
    assert cap["env"]["MLXK2_PORT"] == "8000"


def test_module_override_targets_embed_server_base():
    rc, cap = _run_with_capture(
        port=8002,
        module="mlxk2.core.embed_server_base",
        extra_env={"MLXK2_EMBED_MODEL": "bge-small", "MLXK2_EMBED_CPU": "1"},
    )
    assert rc == 0
    assert cap["cmd"][-2:] == ["-m", "mlxk2.core.embed_server_base"]
    assert cap["env"]["MLXK2_EMBED_MODEL"] == "bge-small"
    assert cap["env"]["MLXK2_EMBED_CPU"] == "1"
    assert cap["env"]["MLXK2_PORT"] == "8002"


def test_extra_env_absent_by_default():
    _, cap = _run_with_capture(port=8000)
    assert "MLXK2_EMBED_MODEL" not in cap["env"]


# --- D2 config bridge: start_server --embed-backend -> MLXK2_EMBED_BACKEND env ---

def test_start_server_sets_embed_backend_env(monkeypatch):
    monkeypatch.delenv("MLXK2_EMBED_BACKEND", raising=False)
    called = {}

    def fake_supervised(host, port, log_level, reload=False, **kw):
        called["yes"] = True
        return 0

    monkeypatch.setattr(serve_mod, "_run_supervised_uvicorn", fake_supervised)
    serve_mod.start_server(embed_backend="http://127.0.0.1:8002", supervise=True)
    assert os.environ["MLXK2_EMBED_BACKEND"] == "http://127.0.0.1:8002"
    assert called.get("yes")


def test_start_server_rejects_bad_embed_backend_url(monkeypatch):
    monkeypatch.delenv("MLXK2_EMBED_BACKEND", raising=False)
    monkeypatch.setattr(serve_mod, "_run_supervised_uvicorn", lambda *a, **k: 0)
    with pytest.raises(ValueError, match="http"):
        serve_mod.start_server(embed_backend="ftp://nope", supervise=True)
    # Fail-fast: env must NOT be set on a rejected URL.
    assert "MLXK2_EMBED_BACKEND" not in os.environ


def test_start_server_without_embed_backend_leaves_env_unset(monkeypatch):
    monkeypatch.delenv("MLXK2_EMBED_BACKEND", raising=False)
    monkeypatch.setattr(serve_mod, "_run_supervised_uvicorn", lambda *a, **k: 0)
    serve_mod.start_server(supervise=True)
    assert "MLXK2_EMBED_BACKEND" not in os.environ


def test_start_server_clears_ambient_embed_backend(monkeypatch):
    # The flag is the single source of truth: an exported MLXK2_EMBED_BACKEND must NOT silently
    # enable (and un-gate) the proxy when `serve` is started without --embed-backend.
    monkeypatch.setenv("MLXK2_EMBED_BACKEND", "http://stale:9999")
    monkeypatch.setattr(serve_mod, "_run_supervised_uvicorn", lambda *a, **k: 0)
    serve_mod.start_server(supervise=True)  # no embed_backend
    assert "MLXK2_EMBED_BACKEND" not in os.environ
