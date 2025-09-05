"""Minimal offline tests for experimental push operation (M0).

These tests avoid any network access and only validate local preconditions
and JSON envelope/fields.
"""

from pathlib import Path

from mlxk2.operations.push import push_operation, DEFAULT_PUSH_BRANCH


def test_push_requires_token(tmp_path, monkeypatch):
    # Ensure no token present
    monkeypatch.delenv("HF_TOKEN", raising=False)

    d: Path = tmp_path / "workspace"
    d.mkdir()
    (d / "README.md").write_text("hello")

    res = push_operation(str(d), "org/model", branch=DEFAULT_PUSH_BRANCH)
    assert res["command"] == "push"
    assert res["status"] == "error"
    assert res["error"]["type"] == "auth_error"
    assert res["data"]["repo_id"] == "org/model"
    assert res["data"]["branch"] == DEFAULT_PUSH_BRANCH


def test_push_workspace_missing(monkeypatch, tmp_path):
    monkeypatch.setenv("HF_TOKEN", "dummy")
    missing = tmp_path / "nope"
    res = push_operation(str(missing), "org/model", branch=DEFAULT_PUSH_BRANCH)
    assert res["status"] == "error"
    assert res["error"]["type"] == "workspace_not_found"
