"""Regression tests for the _run_supervised_uvicorn refactor (ADR-015 Slice D1 reuse seam).

The refactor added `module` + `extra_env` kwargs. Defaults MUST preserve serve's behavior
(targets mlxk2.core.server_base); embed-serve passes the embed module + extra env. subprocess.Popen
is mocked so nothing actually boots.
"""

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
