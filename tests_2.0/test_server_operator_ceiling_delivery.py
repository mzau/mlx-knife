"""The operator ceiling reaches the process that answers requests.

`mlxk serve` supervises, and uvicorn imports `server_base` a second time under its real
name, so the copy that answers is not the copy `run_server()` configured. The ceiling is
therefore handed over through the environment and read in the lifespan hook, beside the
other per-process configuration.

Both ends of that chain are covered here, because each fails silently on its own: without
the export the flag reaches nothing, and without the lifespan read the environment reaches
nothing. The import itself must stay free of it — reading at import would reach the serving
copy too, but also the supervisor parent, `embed-serve`, and the test collection.
"""

import os
import subprocess
import sys

import pytest
import mlxk2.core.server_base as server_base
import mlxk2.operations.serve as serve_mod

GATE = "MLXK2_MAX_TOKENS"


@pytest.fixture(autouse=True)
def clean_ceiling(monkeypatch):
    """No ambient value in, nothing left behind out.

    `start_server` writes `os.environ` itself, which monkeypatch does not undo — a leaked
    ceiling would cut every later test's generation short.
    """
    monkeypatch.delenv(GATE, raising=False)
    monkeypatch.delenv("MLXK2_PRELOAD_MODEL", raising=False)
    before = server_base._default_max_tokens
    yield
    os.environ.pop(GATE, None)
    server_base._default_max_tokens = before


@pytest.fixture
def unsupervised(monkeypatch):
    """start_server without actually spawning uvicorn."""
    monkeypatch.setattr(serve_mod, "_run_supervised_uvicorn", lambda *a, **k: 0)


# --- end 1: the flag reaches the child's environment -------------------------

def test_the_flag_is_handed_over_by_environment(unsupervised):
    serve_mod.start_server(max_tokens=7, supervise=True)
    assert os.environ[GATE] == "7"


def test_without_the_flag_nothing_is_exported(unsupervised):
    serve_mod.start_server(supervise=True)
    assert GATE not in os.environ


# --- end 2: the environment reaches the copy that serves ---------------------

def _probe(code: str, ceiling: str | None) -> list[str]:
    """Run `code` in a fresh interpreter with the ceiling set (or absent); return its words.

    Out of process on purpose: leaving the lifespan context calls _request_global_interrupt(),
    which sets a module global that nothing clears — in-process it would leave every later
    test talking to a server that believes it is shutting down.
    """
    env = dict(os.environ)
    env.pop(GATE, None)
    if ceiling is not None:
        env[GATE] = ceiling
    result = subprocess.run(
        [sys.executable, "-c", code], env=env, capture_output=True, text=True, timeout=120
    )
    assert result.returncode == 0, result.stderr
    return result.stdout.split()


LIFESPAN_PROBE = (
    "from fastapi.testclient import TestClient; "
    "import mlxk2.core.server_base as sb; "
    "c = TestClient(sb.app)\n"
    "with c:\n"
    "    print(sb.get_effective_max_tokens(None), sb.get_effective_max_tokens_vision(None), "
    "sb.get_effective_max_tokens(64))"
)


def test_lifespan_reads_it_in_the_serving_copy():
    assert _probe(LIFESPAN_PROBE, "5") == ["5", "5", "64"]


def test_without_the_environment_the_defaults_stand():
    assert _probe(LIFESPAN_PROBE, None) == ["32768", "2048", "64"]


# --- the import must stay free of it -----------------------------------------

IMPORT_PROBE = (
    "import mlxk2.core.server_base as sb; "
    "print(sb._default_max_tokens, sb.get_effective_max_tokens(None))"
)


def test_importing_the_module_reads_nothing():
    """Reading at import would reach every importer: the parent, embed-serve, collection."""
    assert _probe(IMPORT_PROBE, "5") == ["None", "32768"]


# --- a bad ceiling is refused by the parent, in the right wording ------------

def test_a_flag_below_one_is_refused_in_the_flags_wording(unsupervised):
    with pytest.raises(ValueError, match=r"--max-tokens must be at least 1"):
        serve_mod.start_server(max_tokens=0, supervise=True)
    assert GATE not in os.environ


@pytest.mark.parametrize(
    "value,message",
    [("0", "at least 1"), ("-1", "at least 1"), ("abc", "whole number"), ("1.5", "whole number")],
)
def test_a_bad_ambient_value_is_refused_by_its_own_name(unsupervised, monkeypatch, value, message):
    monkeypatch.setenv(GATE, value)
    with pytest.raises(ValueError, match=message) as raised:
        serve_mod.start_server(supervise=True)
    assert GATE in str(raised.value)
