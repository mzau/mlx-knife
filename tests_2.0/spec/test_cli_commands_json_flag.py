import sys
import json
import pytest


@pytest.mark.spec
def test_cli_list_accepts_json_after_command(monkeypatch, capsys, isolated_cache):
    from mlxk2 import cli

    monkeypatch.setenv("PYTHONWARNINGS", "ignore")
    monkeypatch.setenv("PYTHONDONTWRITEBYTECODE", "1")

    # Ensure we pass --json after the subcommand, as users would
    monkeypatch.setattr(sys, "argv", ["mlxk2", "list", "--json"]) 
    with pytest.raises(SystemExit) as exc:
        cli.main()
    assert exc.value.code == 0

    out = capsys.readouterr().out.strip()
    data = json.loads(out)
    assert data["status"] == "success"
    assert data["command"] == "list"
    assert data["error"] is None
    assert "data" in data and "models" in data["data"] and "count" in data["data"]

