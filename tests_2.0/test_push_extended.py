"""Extended offline tests for experimental push.

These tests monkeypatch a fake `huggingface_hub` to avoid network
and validate:
- no-op (no changes) behavior and message/log propagation
- change summary (+/~/-) extraction from returned commit info
- repo/branch existence handling (`--create`, missing branch tolerated)
- .hfignore merge with default ignore patterns
- human output rendering including --verbose extras
"""

from __future__ import annotations

import os
import pytest

# Skip all tests if push is not enabled
# Push tests now run by default (alpha features included in standard test suite)

import logging
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from mlxk2.operations.push import push_operation, DEFAULT_PUSH_BRANCH
from mlxk2.output.human import render_push


class _Errors(SimpleNamespace):
    class HfHubHTTPError(Exception):
        pass

    class RepositoryNotFoundError(Exception):
        pass

    class RevisionNotFoundError(Exception):
        pass


class _FakeHfApi:
    def __init__(self, token: str | None = None) -> None:
        self.token = token
        self.created = False

    def repo_info(self, repo_id: str, repo_type: str, revision: str):
        # Default: repo + branch exist
        return {"id": repo_id, "type": repo_type, "rev": revision}

    def create_repo(self, repo_id: str, repo_type: str, private: bool, exist_ok: bool):
        self.created = True
        return {"created": True, "private": private}


def _install_fake_hub(monkeypatch, *, mode: str, capture_patterns: dict | None = None):
    """Install a fake huggingface_hub into sys.modules.

    mode:
      - "no_changes": upload returns object without commit_id and emits hub log
      - "with_changes": upload returns commit and files ops
    capture_patterns: optional dict to capture kwargs from upload_folder
    """

    api = _FakeHfApi

    def upload_folder(**kwargs):  # type: ignore[override]
        # Record ignore_patterns if requested
        if capture_patterns is not None:
            capture_patterns["ignore_patterns"] = list(kwargs.get("ignore_patterns") or [])

        if mode == "no_changes":
            # Emit a hub-like info message
            logging.getLogger("huggingface_hub").info(
                "No files have been modified since last commit. Skipping to prevent empty commit."
            )
            # Return object without commit id and without files
            return SimpleNamespace()
        elif mode == "with_changes":
            files = [
                SimpleNamespace(operation="add"),
                SimpleNamespace(operation="update"),
                SimpleNamespace(operation="delete"),
            ]
            return SimpleNamespace(
                commit_id="abcdef1234567890abcdef1234567890abcdef12",
                commit_url="https://huggingface.co/user/repo/commit/abcdef1",
                files=files,
            )
        else:
            return SimpleNamespace(commit_id="cafebabe" * 5)

    fake = SimpleNamespace(HfApi=api, upload_folder=upload_folder, errors=_Errors)
    sys.modules["huggingface_hub"] = fake  # type: ignore
    sys.modules["huggingface_hub.errors"] = _Errors  # type: ignore
    monkeypatch.setitem(sys.modules, "huggingface_hub", fake)
    monkeypatch.setitem(sys.modules, "huggingface_hub.errors", _Errors)
    return fake


def test_push_no_changes_offline(tmp_path, monkeypatch):
    monkeypatch.setenv("HF_TOKEN", "dummy")
    ws = tmp_path / "ws"
    ws.mkdir()
    (ws / "README.md").write_text("x")

    _install_fake_hub(monkeypatch, mode="no_changes")

    res = push_operation(str(ws), "user/repo", branch=DEFAULT_PUSH_BRANCH)
    assert res["status"] == "success"
    assert res["data"]["no_changes"] is True
    assert res["data"]["uploaded_files_count"] == 0
    # Hub message should be reflected in JSON message or hf_logs
    msg = res["data"].get("message") or ""
    logs = res["data"].get("hf_logs") or []
    assert isinstance(logs, list)
    assert ("No files have been modified" in msg) or any(
        isinstance(l, str) and "No files have been modified" in l for l in logs
    )

    # Human output should show "no changes" and not duplicate hub logs
    line = render_push(res)
    assert "no changes" in line
    assert "No files have been modified" not in line


def test_push_with_changes_summary_and_url(tmp_path, monkeypatch):
    monkeypatch.setenv("HF_TOKEN", "dummy")
    ws = tmp_path / "ws"
    ws.mkdir()
    (ws / "file.txt").write_text("x")

    _install_fake_hub(monkeypatch, mode="with_changes")

    res = push_operation(str(ws), "user/repo", branch=DEFAULT_PUSH_BRANCH)
    assert res["status"] == "success"
    assert res["data"]["no_changes"] is False
    assert res["data"]["uploaded_files_count"] == 3
    assert res["data"]["change_summary"] == {"added": 1, "modified": 1, "deleted": 1}
    assert res["data"]["commit_url"].startswith("https://huggingface.co/")

    # Human output with verbose includes URL
    verbose_line = render_push(res, verbose=True)
    assert "commit" in verbose_line and "http" in verbose_line


def test_push_repo_not_found_requires_create(tmp_path, monkeypatch):
    monkeypatch.setenv("HF_TOKEN", "dummy")
    ws = tmp_path / "ws"
    ws.mkdir()
    (ws / "file.txt").write_text("x")

    # Fake API that raises repo not found
    class _ApiMissing(_FakeHfApi):
        def repo_info(self, repo_id: str, repo_type: str, revision: str):  # type: ignore[override]
            raise _Errors.RepositoryNotFoundError()

    def upload_folder(**kwargs):  # type: ignore
        return SimpleNamespace(commit_id="deadbeefdeadbeefdeadbeefdeadbeefdeadbeef")

    fake = SimpleNamespace(HfApi=_ApiMissing, upload_folder=upload_folder, errors=_Errors)
    sys.modules["huggingface_hub"] = fake  # type: ignore
    sys.modules["huggingface_hub.errors"] = _Errors  # type: ignore
    monkeypatch.setitem(sys.modules, "huggingface_hub", fake)
    monkeypatch.setitem(sys.modules, "huggingface_hub.errors", _Errors)

    # Without --create → error
    res = push_operation(str(ws), "user/repo", create=False, private=False, branch=DEFAULT_PUSH_BRANCH)
    assert res["status"] == "error"
    assert res["error"]["type"] == "repo_not_found"

    # With --create → success and created_repo True
    res2 = push_operation(str(ws), "user/repo", create=True, private=True, branch=DEFAULT_PUSH_BRANCH)
    assert res2["status"] == "success"
    assert res2["data"]["created_repo"] is True


def test_push_branch_missing_is_tolerated(tmp_path, monkeypatch):
    monkeypatch.setenv("HF_TOKEN", "dummy")
    ws = tmp_path / "ws"
    ws.mkdir()
    (ws / "file.txt").write_text("x")

    class _ApiNoBranch(_FakeHfApi):
        def repo_info(self, repo_id: str, repo_type: str, revision: str):  # type: ignore[override]
            raise _Errors.RevisionNotFoundError()

    def upload_folder(**kwargs):  # type: ignore
        return SimpleNamespace(commit_id="feedfacefeedfacefeedfacefeedfacefeedface")

    fake = SimpleNamespace(HfApi=_ApiNoBranch, upload_folder=upload_folder, errors=_Errors)
    sys.modules["huggingface_hub"] = fake  # type: ignore
    sys.modules["huggingface_hub.errors"] = _Errors  # type: ignore
    monkeypatch.setitem(sys.modules, "huggingface_hub", fake)
    monkeypatch.setitem(sys.modules, "huggingface_hub.errors", _Errors)

    res = push_operation(str(ws), "user/repo", branch=DEFAULT_PUSH_BRANCH)
    assert res["status"] == "success"
    assert isinstance(res["data"].get("commit_sha"), str)


def test_push_hfignore_is_merged_with_defaults(tmp_path, monkeypatch):
    monkeypatch.setenv("HF_TOKEN", "dummy")
    ws = tmp_path / "ws"
    ws.mkdir()
    # Create files and .hfignore
    (ws / "README.md").write_text("x")
    (ws / ".hfignore").write_text(".idea/\n.vscode/\n*.ipynb\n")

    captured: dict = {}
    _install_fake_hub(monkeypatch, mode="with_changes", capture_patterns=captured)

    res = push_operation(str(ws), "user/repo", branch=DEFAULT_PUSH_BRANCH)
    assert res["status"] == "success"
    pats = captured.get("ignore_patterns") or []
    # Ensure core defaults are present
    defaults = {"**/.git/**", "**/.DS_Store", "**/__pycache__/**", "**/.venv/**", "**/venv/**", "**/*.pyc"}
    assert defaults.issubset(set(pats))
    # Ensure .hfignore additions are present
    assert ".idea/" in pats and ".vscode/" in pats and "*.ipynb" in pats


def test_push_retry_creates_branch_on_upload_revision_error(tmp_path, monkeypatch):
    """If upload fails with a revision-not-found style error and --create is set,
    the operation should create the branch and retry once, succeeding offline."""
    monkeypatch.setenv("HF_TOKEN", "dummy")
    ws = tmp_path / "ws"
    ws.mkdir()
    (ws / "file.txt").write_text("x")

    class _ApiOk(_FakeHfApi):
        instance = None  # type: ignore[var-annotated]

        def __init__(self, token: str | None = None) -> None:  # type: ignore[override]
            super().__init__(token)
            self.created_branches: list[tuple[str, str]] = []
            _ApiOk.instance = self

        def create_branch(self, repo_id: str, repo_type: str, branch: str):  # type: ignore[override]
            self.created_branches.append((repo_id, branch))
            return {"ok": True}

    state = {"attempt": 0}

    def upload_folder(**kwargs):  # type: ignore
        # First attempt fails with a hub-like error; second succeeds
        if state["attempt"] == 0:
            state["attempt"] += 1
            raise _Errors.HfHubHTTPError("Invalid rev id: test-branch")
        state["attempt"] += 1
        return SimpleNamespace(commit_id="0123456789abcdef0123456789abcdef01234567")

    fake = SimpleNamespace(HfApi=_ApiOk, upload_folder=upload_folder, errors=_Errors)
    sys.modules["huggingface_hub"] = fake  # type: ignore
    sys.modules["huggingface_hub.errors"] = _Errors  # type: ignore
    monkeypatch.setitem(sys.modules, "huggingface_hub", fake)
    monkeypatch.setitem(sys.modules, "huggingface_hub.errors", _Errors)

    res = push_operation(str(ws), "user/repo", create=True, private=True, branch="test-branch")
    assert res["status"] == "success"
    # Ensure we retried exactly once (two attempts total)
    assert state["attempt"] == 2
    # Ensure branch creation was attempted once
    assert _ApiOk.instance is not None
    assert ("user/repo", "test-branch") in (_ApiOk.instance.created_branches if _ApiOk.instance else [])


def test_push_apfs_warning_added_for_non_apfs_workspace(tmp_path, monkeypatch):
    """Test that push adds APFS warning to message for non-APFS workspaces (ADR-007)."""
    monkeypatch.setenv("HF_TOKEN", "dummy")
    ws = tmp_path / "ws"
    ws.mkdir()
    (ws / "file.txt").write_text("test content")

    # Mock APFS detection to return False (non-APFS workspace)
    with monkeypatch.context() as m:
        m.setattr("mlxk2.operations.push._is_apfs_filesystem", lambda path: False)

        _install_fake_hub(monkeypatch, mode="with_changes")

        res = push_operation(str(ws), "user/repo", branch=DEFAULT_PUSH_BRANCH)

        assert res["status"] == "success"
        assert "Clone operations require APFS filesystem" in res["data"]["message"]


def test_push_no_apfs_warning_for_apfs_workspace(tmp_path, monkeypatch):
    """Test that push does NOT add APFS warning for APFS workspaces (ADR-007)."""
    monkeypatch.setenv("HF_TOKEN", "dummy")
    ws = tmp_path / "ws"
    ws.mkdir()
    (ws / "file.txt").write_text("test content")

    # Mock APFS detection to return True (APFS workspace)
    with monkeypatch.context() as m:
        m.setattr("mlxk2.operations.push._is_apfs_filesystem", lambda path: True)

        _install_fake_hub(monkeypatch, mode="with_changes")

        res = push_operation(str(ws), "user/repo", branch=DEFAULT_PUSH_BRANCH)

        assert res["status"] == "success"
        assert "Clone operations require APFS filesystem" not in res["data"]["message"]
