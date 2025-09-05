"""CLI-arg tests for experimental push (offline)."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest


def _run_cli(argv: list[str], capsys):
    from mlxk2.cli import main as cli_main

    # Replace sys.argv and run
    old_argv = sys.argv[:]
    sys.argv = argv[:]
    try:
        with pytest.raises(SystemExit):
            cli_main()
    finally:
        sys.argv = old_argv
    out = capsys.readouterr().out
    return out


def test_cli_push_missing_args_json_error(capsys):
    # Missing required positional args but with --json should emit JSON error
    out = _run_cli(["mlxk2", "push", "--private", "--json"], capsys)
    data = json.loads(out)
    assert data["status"] == "error"
    assert data["command"] is None
    assert isinstance(data["error"], dict)


def test_cli_push_workspace_missing_json_error(tmp_path, monkeypatch, capsys):
    # Provide missing workspace; ensure JSON error and specific error type
    monkeypatch.setenv("HF_TOKEN", "dummy")
    missing = str(tmp_path / "nope")
    out = _run_cli(["mlxk2", "push", "--private", missing, "user/repo", "--json"], capsys)
    data = json.loads(out)
    assert data["status"] == "error"
    assert data["command"] == "push"
    assert data["error"]["type"] == "workspace_not_found"


def _install_fake_hf(monkeypatch, mode: str):
    class _Errors:
        class HfHubHTTPError(Exception):
            pass

        class RepositoryNotFoundError(Exception):
            pass

        class RevisionNotFoundError(Exception):
            pass

    class _Api:
        def __init__(self, token=None):
            self.token = token

        def repo_info(self, repo_id: str, repo_type: str, revision: str):
            return {"id": repo_id, "type": repo_type, "rev": revision}

    def upload_folder(**kwargs):  # type: ignore
        if mode == "no_changes":
            # Return an object without commit_id
            return SimpleNamespace()
        else:
            return SimpleNamespace(commit_id="abcdef1234567890abcdef1234567890abcdef12")

    fake = SimpleNamespace(HfApi=_Api, upload_folder=upload_folder, errors=_Errors)
    sys.modules["huggingface_hub"] = fake  # type: ignore
    sys.modules["huggingface_hub.errors"] = _Errors  # type: ignore
    monkeypatch.setitem(sys.modules, "huggingface_hub", fake)
    monkeypatch.setitem(sys.modules, "huggingface_hub.errors", _Errors)


def test_cli_push_no_changes_json_output(tmp_path, monkeypatch, capsys):
    # Setup workspace
    ws = tmp_path / "ws"
    ws.mkdir()
    (ws / "x.txt").write_text("x")
    monkeypatch.setenv("HF_TOKEN", "dummy")

    _install_fake_hf(monkeypatch, mode="no_changes")

    out = _run_cli(["mlxk2", "push", "--private", str(ws), "user/repo", "--json"], capsys)
    data = json.loads(out)
    assert data["status"] == "success"
    assert data["command"] == "push"
    assert data["data"]["no_changes"] is True
    assert data["data"]["uploaded_files_count"] == 0


def test_cli_push_with_changes_json_output(tmp_path, monkeypatch, capsys):
    # Setup workspace
    ws = tmp_path / "ws"
    ws.mkdir()
    (ws / "x.txt").write_text("x")
    monkeypatch.setenv("HF_TOKEN", "dummy")

    _install_fake_hf(monkeypatch, mode="with_changes")

    out = _run_cli(["mlxk2", "push", "--private", str(ws), "user/repo", "--json"], capsys)
    data = json.loads(out)
    assert data["status"] == "success"
    assert data["command"] == "push"
    assert data["data"]["no_changes"] is False
    assert isinstance(data["data"]["commit_sha"], str)

