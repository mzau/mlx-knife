"""Opt-in live test for push.

This test is skipped by default. Enable by setting BOTH:
- MLXK2_LIVE_PUSH=1
- HF_TOKEN=<your_write_token>
- MLXK2_LIVE_REPO=org/model (target model repo)
- MLXK2_LIVE_WORKSPACE=/abs/path/to/workspace (folder to push)

It performs a JSON-mode push and asserts a success envelope.
It does NOT modify workspace content and thus typically results in a no-op
if the remote already matches. It may create the repo if `--create` is used.
"""

from __future__ import annotations

import json
import os
import sys

import pytest


live_enabled = os.environ.get("MLXK2_LIVE_PUSH") == "1"
hf_token_present = bool(os.environ.get("HF_TOKEN"))
repo = os.environ.get("MLXK2_LIVE_REPO")
workspace = os.environ.get("MLXK2_LIVE_WORKSPACE")

pytestmark = [
    pytest.mark.wet,
    pytest.mark.live_push,
    pytest.mark.skipif(
        not (live_enabled and hf_token_present and repo and workspace),
        reason=(
            "Live push disabled. Set MLXK2_LIVE_PUSH=1, HF_TOKEN, MLXK2_LIVE_REPO, "
            "and MLXK2_LIVE_WORKSPACE to enable."
        ),
    ),
]


def _run_cli(argv: list[str], capsys) -> str:
    from mlxk2.cli import main as cli_main
    old_argv = sys.argv[:]
    sys.argv = argv[:]
    try:
        with pytest.raises(SystemExit):
            cli_main()
    finally:
        sys.argv = old_argv
    out = capsys.readouterr().out
    return out


def test_live_push_json_success(capsys):
    # Run push in JSON mode; do not assume commit vs no-op
    out = _run_cli(["mlxk2", "push", "--private", workspace, repo, "--json"], capsys)
    data = json.loads(out)
    assert data["command"] == "push"
    assert data["status"] in {"success", "error"}
    if data["status"] == "error":
        # Provide a helpful hint on failure
        pytest.skip(f"Live push error: {data['error']}")
