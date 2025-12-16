import sys
import json
import pytest


@pytest.mark.spec
def test_cli_version_json_output(monkeypatch, capsys):
    from mlxk2 import __version__
    from mlxk2.spec import JSON_API_SPEC_VERSION
    from mlxk2 import cli

    monkeypatch.setenv("PYTHONWARNINGS", "ignore")
    monkeypatch.setenv("PYTHONDONTWRITEBYTECODE", "1")

    monkeypatch.setattr(sys, "argv", ["mlxk2", "--version", "--json"])
    with pytest.raises(SystemExit) as exc:
        cli.main()
    assert exc.value.code == 0

    out = capsys.readouterr().out.strip()
    data = json.loads(out)
    assert data["status"] == "success"
    assert data["command"] == "version"
    assert data["error"] is None
    assert data["data"]["cli_version"] == __version__
    assert data["data"]["json_api_spec_version"] == JSON_API_SPEC_VERSION

    # 0.1.6: system object with memory info (macOS only)
    system = data["data"].get("system")
    assert system is None or isinstance(system, dict), "system must be null or object"
    if system is not None:
        # On macOS, memory_total_bytes should be present
        if "memory_total_bytes" in system:
            assert isinstance(system["memory_total_bytes"], int)
            assert system["memory_total_bytes"] > 0

