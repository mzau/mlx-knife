"""CLI alpha-gate tests for `mlxk embed-serve` (ADR-015 Slice D1).

Subprocess-level, no model/server required: the gate must reject without
MLXK2_ENABLE_ALPHA_FEATURES=1, before importing the operation or binding a port. With the gate
open, a bogus model must fail at pre-flight validation (in the parent, before any server boot).
"""

import json
import os
import sys
import subprocess


def _run(args, env_overrides):
    env = dict(os.environ)
    env.pop("MLXK2_ENABLE_ALPHA_FEATURES", None)
    env.update(env_overrides)
    return subprocess.run(
        [sys.executable, "-m", "mlxk2.cli"] + args,
        input=None, text=True, capture_output=True, env=env, timeout=120,
    )


def test_gate_blocks_without_alpha_env():
    p = _run(["embed-serve", "some-model"], env_overrides={})
    assert p.returncode == 1
    assert "MLXK2_ENABLE_ALPHA_FEATURES" in (p.stdout + p.stderr)


def test_gate_json_error_envelope():
    p = _run(["embed-serve", "some-model", "--json"], env_overrides={})
    assert p.returncode == 1
    env = json.loads(p.stdout)
    assert env["status"] == "error"
    assert env["error"]["type"] == "CommandError"
    assert "MLXK2_ENABLE_ALPHA_FEATURES" in env["error"]["message"]


def test_alpha_on_bogus_model_fails_at_preflight_not_gate():
    # Gate open + bogus model: must fail pre-flight validation (no gate message, no server boot,
    # no hang). start_embed_serve raises before _run_supervised_uvicorn.
    p = _run(["embed-serve", "__definitely_no_such_model__"],
             env_overrides={"MLXK2_ENABLE_ALPHA_FEATURES": "1"})
    assert p.returncode == 1
    combined = (p.stdout + p.stderr)
    assert "MLXK2_ENABLE_ALPHA_FEATURES" not in combined
    assert ("not found" in combined.lower() or "not recognized" in combined.lower())
