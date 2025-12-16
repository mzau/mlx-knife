"""Tests for the mlx-run wrapper entry point."""

import sys
import pytest

from mlxk2.tools.run import main as mlx_run_main


def test_mlx_run_inserts_run_subcommand(monkeypatch):
    """Ensure mlx-run prepends 'run' so CLI parses correctly."""
    captured = {}

    def fake_cli_main():
        captured["argv"] = sys.argv[:]
        raise SystemExit(0)

    monkeypatch.setattr("mlxk2.tools.run.mlxk_main", fake_cli_main)

    original_argv = sys.argv[:]
    sys.argv = ["mlx-run", "test-model", "prompt here"]
    try:
        with pytest.raises(SystemExit) as excinfo:
            mlx_run_main()
        assert excinfo.value.code == 0
    finally:
        sys.argv = original_argv

    assert captured["argv"][0] == "mlx-run"
    assert captured["argv"][1] == "run"
    assert captured["argv"][2:] == ["test-model", "prompt here"]
