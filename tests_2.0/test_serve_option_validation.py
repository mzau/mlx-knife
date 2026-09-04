"""`serve` checks its options before it announces anything (audit C1 review, R-02).

The CLI printed the `starting` envelope first and let `start_server` raise afterwards, so
a rejected option produced *two* JSON documents on stdout: a reader that parses the first
one believes a server is coming up. That held for every option `start_server` validates —
`--chunk`, `--embed-backend` and, since the ceiling moved into the parent, `--max-tokens`.
"""

import json
import os
import subprocess
import sys

import pytest

from mlxk2.operations.serve import validate_serve_options

GATE = "MLXK2_MAX_TOKENS"


def _documents(stdout: str) -> list:
    """Every JSON document on stdout, so a second one cannot hide behind the first."""
    decoder, found, index = json.JSONDecoder(), [], 0
    while index < len(stdout):
        while index < len(stdout) and stdout[index].isspace():
            index += 1
        if index >= len(stdout):
            break
        document, index = decoder.raw_decode(stdout, index)
        found.append(document)
    return found


def _serve(args, env_overrides=None):
    env = dict(os.environ, MLXK2_ENABLE_ALPHA_FEATURES="1")
    env.pop(GATE, None)
    env.update(env_overrides or {})
    return subprocess.run(
        [sys.executable, "-m", "mlxk2.cli", "serve", "--port", "1", "--json"] + args,
        capture_output=True, text=True, env=env, timeout=120,
    )


@pytest.mark.parametrize(
    "args,env",
    [
        (["--max-tokens", "0"], {}),
        (["--chunk", "0"], {}),
        (["--embed-backend", "ftp://nope"], {}),
        ([], {GATE: "0"}),
        ([], {GATE: "abc"}),
    ],
)
def test_a_rejected_option_yields_one_error_document(args, env):
    result = _serve(args, env)
    documents = _documents(result.stdout)
    assert len(documents) == 1, documents
    assert documents[0]["status"] == "error"
    assert result.returncode == 1


# --- the reported ceiling is the one that will apply -------------------------

def test_the_validator_reports_the_flag(monkeypatch):
    monkeypatch.delenv(GATE, raising=False)
    assert validate_serve_options(max_tokens=7) == 7


def test_the_validator_falls_back_to_the_environment(monkeypatch):
    monkeypatch.setenv(GATE, "5")
    assert validate_serve_options() == 5


def test_the_flag_outranks_the_environment(monkeypatch):
    monkeypatch.setenv(GATE, "5")
    assert validate_serve_options(max_tokens=7) == 7


def test_the_startup_envelope_carries_the_effective_ceiling():
    """It used to report the flag, so an operator setting the variable saw `null`."""
    result = _serve([], {GATE: "5"})
    assert _documents(result.stdout)[0]["data"]["max_tokens"] == 5


def test_the_validator_has_no_side_effects(monkeypatch):
    """The CLI runs it before printing; it must not export anything on its way."""
    monkeypatch.delenv("MLXK2_EMBED_BACKEND", raising=False)
    monkeypatch.delenv(GATE, raising=False)
    validate_serve_options(max_tokens=7, embed_backend="http://127.0.0.1:8002")
    assert GATE not in os.environ
    assert "MLXK2_EMBED_BACKEND" not in os.environ
