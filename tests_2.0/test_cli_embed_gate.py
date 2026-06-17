"""CLI alpha-gate tests for `mlxk embed` (ADR-015 Slice A).

Subprocess-level, no model required: the gate must reject without
MLXK2_ENABLE_ALPHA_FEATURES=1 and pass through to the operation with it.
"""

import json
import os
import sys
import subprocess

import pytest


def _run(args, env_overrides):
    env = dict(os.environ)
    env.pop("MLXK2_ENABLE_ALPHA_FEATURES", None)
    env.update(env_overrides)
    return subprocess.run(
        [sys.executable, "-m", "mlxk2.cli"] + args,
        input=None,
        text=True,
        capture_output=True,
        env=env,
        timeout=120,
    )


def test_gate_blocks_without_alpha_env():
    p = _run(["embed", "some-model", "hello"], env_overrides={})
    assert p.returncode == 1
    assert "MLXK2_ENABLE_ALPHA_FEATURES" in (p.stdout + p.stderr)


def test_gate_passes_with_alpha_env_then_model_error():
    # With the gate open, a bogus model must reach the operation (not the gate),
    # producing a model/route error — never the gate message.
    p = _run(["embed", "__definitely_no_such_model__", "hello"],
             env_overrides={"MLXK2_ENABLE_ALPHA_FEATURES": "1"})
    assert p.returncode == 1
    combined = p.stdout + p.stderr
    assert "MLXK2_ENABLE_ALPHA_FEATURES" not in combined
    assert ("ModelNotFound" in combined or "NotAnEmbedder" in combined
            or "not found" in combined.lower())


def test_json_flag_renders_error_envelope():
    # With the gate open, a bogus model + --json renders the standard envelope on stdout
    # (status=error, command=embed) — model-free, exercises the --json wiring.
    p = _run(["embed", "__definitely_no_such_model__", "hello", "--json"],
             env_overrides={"MLXK2_ENABLE_ALPHA_FEATURES": "1"})
    assert p.returncode == 1
    env = json.loads(p.stdout)
    assert env["status"] == "error"
    assert env["command"] == "embed"
    assert env["error"]["type"]
