"""Opt-in live E2E test for human list rendering using the real HF cache.

Per TESTING.md mini‑matrix, this test is collected by default but
only runs when explicitly selected with the `live_list` marker.

Run:
- pytest -m live_list -v
- umbrella: pytest -m wet -v
"""

from __future__ import annotations

import json
import sys
from typing import List, Dict

import pytest

pytestmark = [pytest.mark.wet, pytest.mark.live_list]


def _run_cli(argv: List[str], capsys) -> str:
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


def _json_models(capsys) -> List[Dict]:
    out = _run_cli(["mlxk2", "list", "--json"], capsys)
    data = json.loads(out)
    assert data["status"] == "success" and data["command"] == "list"
    return data["data"]["models"]


def _display_name_for_default(name: str) -> str:
    # In compact default view, we strip mlx-community/ prefix
    return name.split("/", 1)[1] if name.startswith("mlx-community/") else name


def test_live_list_human_variants(capsys, request):
    # Only run when explicitly selected with -m live_list
    selected = request.config.getoption("-m") or ""
    if "live_list" not in selected:
        pytest.skip("Run with -m live_list to enable this end-to-end test")
    models = _json_models(capsys)

    mlx = [m for m in models if m.get("framework") == "MLX"]
    mlx_chat = [m for m in mlx if m.get("model_type") == "chat"]
    mlx_base = [m for m in mlx if m.get("model_type") == "base"]
    other = [m for m in models if m.get("framework") != "MLX"]

    # Fail if the cache doesn't have the necessary models
    assert mlx_chat, "Need at least one MLX chat model in HF cache"
    assert mlx_base, "Need at least one MLX base model in HF cache"

    chat_name = mlx_chat[0]["name"]
    base_name = mlx_base[0]["name"]

    # Default list: only MLX chat
    out_default = _run_cli(["mlxk2", "list"], capsys)
    assert _display_name_for_default(chat_name) in out_default
    assert _display_name_for_default(base_name) not in out_default

    # Verbose: all MLX (chat + base)
    out_verbose = _run_cli(["mlxk2", "list", "--verbose"], capsys)
    assert chat_name in out_verbose
    assert base_name in out_verbose

    # All: all frameworks
    out_all = _run_cli(["mlxk2", "list", "--all"], capsys)
    assert _display_name_for_default(chat_name) in out_all or chat_name in out_all
    assert _display_name_for_default(base_name) in out_all or base_name in out_all

    if other:
        other_name = other[0]["name"]
        # Non-MLX names are never stripped by default rule
        assert other_name in out_all

