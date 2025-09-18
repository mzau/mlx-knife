"""Validate push(error) output (missing HF_TOKEN) against the JSON schema.

Offline test: no network; ensures error envelope conforms to schema.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

# Skip all tests if push is not enabled
# Push tests now run by default (alpha features included in standard test suite)

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


def test_push_missing_token_matches_schema(tmp_path, monkeypatch):
    validator = _load_validator()
    # Ensure no token
    monkeypatch.delenv("HF_TOKEN", raising=False)
    ws = tmp_path / "ws"
    ws.mkdir()
    (ws / "README.md").write_text("x")

    res = push_operation(str(ws), "user/repo", branch="main")
    assert res["status"] == "error"
    assert res["command"] == "push"
    # Validate against schema (top-level error is globally defined)
    errors = sorted(e.message for e in validator.iter_errors(res))
    assert not errors, f"Schema validation errors for push error: {errors}"

