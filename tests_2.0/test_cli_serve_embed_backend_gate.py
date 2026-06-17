"""CLI alpha-gate + URL-validation tests for `mlxk serve --embed-backend` (ADR-015 Slice D2).

Subprocess-level, no model/server required. The `--embed-backend` flag is experimental: it must
reject without MLXK2_ENABLE_ALPHA_FEATURES=1, BEFORE any server boot. Plain `serve` (no flag)
must stay UNGATED. A malformed URL must fail fast (no server boot).
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


def test_gate_blocks_embed_backend_without_alpha():
    p = _run(["serve", "--embed-backend", "http://127.0.0.1:8002"], env_overrides={})
    assert p.returncode == 1
    assert "MLXK2_ENABLE_ALPHA_FEATURES" in (p.stdout + p.stderr)


def test_gate_json_error_envelope():
    p = _run(["serve", "--embed-backend", "http://127.0.0.1:8002", "--json"], env_overrides={})
    assert p.returncode == 1
    env = json.loads(p.stdout)
    assert env["status"] == "error"
    assert env["error"]["type"] == "CommandError"
    assert "MLXK2_ENABLE_ALPHA_FEATURES" in env["error"]["message"]


def test_bad_url_shape_rejected_with_alpha():
    # Gate open + malformed URL: start_server raises fail-fast (no server boot, no hang).
    p = _run(["serve", "--embed-backend", "ftp://127.0.0.1:8002"],
             env_overrides={"MLXK2_ENABLE_ALPHA_FEATURES": "1"})
    assert p.returncode != 0
    assert "http(s) URL" in (p.stdout + p.stderr)


def test_plain_serve_not_gated_by_embed_flag():
    # Regression: plain `serve` (no --embed-backend) must NOT hit the embed gate, even with alpha
    # unset. We can't fully boot a server here, so prove it gets PAST the gate by pairing with a
    # bogus --model that fails at model resolution — a different error, with NO gate message.
    p = _run(["serve", "--model", "__definitely_no_such_model__"], env_overrides={})
    assert p.returncode != 0
    combined = (p.stdout + p.stderr)
    assert "MLXK2_ENABLE_ALPHA_FEATURES" not in combined
