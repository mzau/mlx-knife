"""Dry-run tests for experimental push (offline, no network).

Covers repo-missing, existing-no-changes, and existing-with-changes cases.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

# Skip all tests if push is not enabled
pytestmark = pytest.mark.skipif(
    not os.getenv("MLXK2_ENABLE_EXPERIMENTAL_PUSH"),
    reason="Push tests require MLXK2_ENABLE_EXPERIMENTAL_PUSH=1"
)
from types import SimpleNamespace

import pytest

from mlxk2.operations.push import push_operation, DEFAULT_PUSH_BRANCH
from mlxk2.output.human import render_push


def _install_fake_hf(monkeypatch, *, repo_exists: bool = True, branch_exists: bool = True, remote_files: list[str] | None = None):
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
            if not repo_exists:
                raise _Errors.RepositoryNotFoundError("not found")
            if not branch_exists:
                raise _Errors.RevisionNotFoundError("rev not found")
            return {"id": repo_id, "type": repo_type, "rev": revision}

        def list_repo_files(self, repo_id: str, repo_type: str, revision: str):
            return list(remote_files or [])

        # create_repo is only called when create=True (not used in dry-run tests)
        def create_repo(self, repo_id: str, repo_type: str, private: bool, exist_ok: bool):
            return {"ok": True}

    fake = SimpleNamespace(HfApi=_Api, upload_folder=None, errors=_Errors)
    # Use monkeypatch to ensure automatic restoration after each test
    monkeypatch.setitem(sys.modules, "huggingface_hub", fake)
    monkeypatch.setitem(sys.modules, "huggingface_hub.errors", _Errors)


def test_dry_run_repo_missing(tmp_path: Path, monkeypatch):
    # Workspace with files; one ignored by default, one via .hfignore
    ws = tmp_path / "ws"
    ws.mkdir()
    (ws / "keep.txt").write_text("x")
    (ws / ".DS_Store").write_text("x")  # default ignore
    (ws / "ignored.log").write_text("x")
    (ws / ".hfignore").write_text("ignored.log\n")

    monkeypatch.setenv("HF_TOKEN", "dummy")
    _install_fake_hf(monkeypatch, repo_exists=False)

    res = push_operation(str(ws), "org/model", branch=DEFAULT_PUSH_BRANCH, dry_run=True)
    assert res["status"] == "success"
    d = res["data"]
    assert d.get("dry_run") is True
    assert d.get("would_create_repo") is True
    assert d.get("would_create_branch") is True
    # Only keep.txt should be counted (others ignored)
    assert d.get("dry_run_summary", {}).get("added") == 1
    # Human line
    line = render_push(res)
    assert "dry-run:" in line


def test_dry_run_existing_no_changes(tmp_path: Path, monkeypatch):
    ws = tmp_path / "ws"
    ws.mkdir()
    (ws / "a.txt").write_text("1")
    (ws / "b.txt").write_text("2")
    monkeypatch.setenv("HF_TOKEN", "dummy")
    _install_fake_hf(monkeypatch, repo_exists=True, branch_exists=True, remote_files=["a.txt", "b.txt"]) 

    res = push_operation(str(ws), "org/model", branch=DEFAULT_PUSH_BRANCH, dry_run=True)
    assert res["status"] == "success"
    d = res["data"]
    assert d.get("dry_run") is True
    assert d.get("no_changes") is True
    assert d.get("dry_run_summary", {}).get("added") == 0
    assert d.get("dry_run_summary", {}).get("deleted") == 0
    assert d.get("message") == "Dry-run: no changes"


def test_dry_run_existing_with_changes(tmp_path: Path, monkeypatch):
    ws = tmp_path / "ws"
    ws.mkdir()
    # Local: a.txt (shared), new.txt (to add)
    (ws / "a.txt").write_text("1")
    (ws / "new.txt").write_text("x")
    monkeypatch.setenv("HF_TOKEN", "dummy")
    # Remote: a.txt (shared), gone.txt (to delete)
    _install_fake_hf(monkeypatch, repo_exists=True, branch_exists=True, remote_files=["a.txt", "gone.txt"]) 

    res = push_operation(str(ws), "org/model", branch=DEFAULT_PUSH_BRANCH, dry_run=True)
    assert res["status"] == "success"
    d = res["data"]
    assert d.get("dry_run") is True
    assert d.get("no_changes") is False
    assert d.get("dry_run_summary", {}).get("added") == 1
    assert d.get("dry_run_summary", {}).get("deleted") == 1
    assert d.get("message") == "Dry-run: +1 ~? -1"
    # Human line should reflect plan
    line = render_push(res)
    assert "dry-run: +1 ~? -1" in line
