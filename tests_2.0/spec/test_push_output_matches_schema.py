"""Validate push(success) output against the JSON schema without network.

We monkeypatch a fake `huggingface_hub` module into sys.modules so that
`push_operation` can run to a success path offline.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from mlxk2.operations.push import push_operation


def _load_validator():
    try:
        from jsonschema import Draft7Validator  # type: ignore
    except Exception:
        pytest.skip("jsonschema not available", allow_module_level=True)
    schema_path = Path("docs/json-api-schema.json")
    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    return Draft7Validator(schema)


class _FakeHfApi:
    def __init__(self, token: str | None = None) -> None:
        self.token = token

    def repo_info(self, repo_id: str, repo_type: str, revision: str):
        # Pretend repo + branch exist
        return {"id": repo_id, "type": repo_type, "rev": revision}

    def create_repo(self, repo_id: str, repo_type: str, private: bool, exist_ok: bool):
        return {"created": True}


def _install_fake_hf_module(monkeypatch):
    class _Errors(SimpleNamespace):
        class HfHubHTTPError(Exception):
            pass

        class RepositoryNotFoundError(Exception):
            pass

        class RevisionNotFoundError(Exception):
            pass

    def upload_folder(**kwargs):
        # Emulate successful upload return with commit_id attribute
        return SimpleNamespace(commit_id="abcdef1234567890abcdef1234567890abcdef12")

    fake = SimpleNamespace(HfApi=_FakeHfApi, upload_folder=upload_folder, errors=_Errors)
    sys.modules["huggingface_hub"] = fake  # type: ignore
    sys.modules["huggingface_hub.errors"] = _Errors  # type: ignore
    monkeypatch.setitem(sys.modules, "huggingface_hub", fake)
    monkeypatch.setitem(sys.modules, "huggingface_hub.errors", _Errors)


def test_push_success_shape_matches_schema(tmp_path, monkeypatch):
    validator = _load_validator()
    # Prepare workspace
    ws = tmp_path / "ws"
    ws.mkdir()
    (ws / "README.md").write_text("ok")
    (ws / ".hfignore").write_text(".DS_Store\n__pycache__/\n")
    monkeypatch.setenv("HF_TOKEN", "dummy")

    # Fake HF module
    _install_fake_hf_module(monkeypatch)

    res = push_operation(str(ws), "user/repo", create=False, private=False, branch="main", commit_message="t")
    assert res["status"] == "success"
    assert res["command"] == "push"
    # Validate against schema
    errors = sorted(e.message for e in validator.iter_errors(res))
    assert not errors, f"Schema validation errors for push success: {errors}"
