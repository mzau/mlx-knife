"""The JSON API version lives in four places; they must agree (#67).

Code (`mlxk2.spec`), schema title, spec header, and the newest Version History entry.
Hard imports on purpose: a missing `jsonschema` is a broken [test] install, not a reason
to skip the contract check.
"""

import json
import re
from pathlib import Path

import pytest
from jsonschema import Draft7Validator

from mlxk2.spec import JSON_API_SPEC_VERSION

_REPO = Path(__file__).resolve().parents[2]
_SCHEMA_PATH = _REPO / "docs" / "json-api-schema.json"
_SPEC_PATH = _REPO / "docs" / "json-api-specification.md"

_SEMVER = r"([0-9]+\.[0-9]+\.[0-9]+)"


def _schema() -> dict:
    return json.loads(_SCHEMA_PATH.read_text(encoding="utf-8"))


def _spec_text() -> str:
    return _SPEC_PATH.read_text(encoding="utf-8")


def _schema_title_version(schema: dict) -> str:
    m = re.fullmatch(rf"MLX-Knife 2\.0 JSON API {_SEMVER} \(current\)", schema["title"])
    assert m, f"schema title does not carry a version: {schema['title']!r}"
    return m.group(1)


def _spec_header_version(text: str) -> str:
    m = re.search(rf"^\*\*Specification Version:\*\*\s*{_SEMVER}\s*$", text, re.MULTILINE)
    assert m, "spec header has no '**Specification Version:** X.Y.Z' line"
    return m.group(1)


def _spec_history_head_version(text: str) -> str:
    # The first bullet under "## Version History" is the newest entry
    start = text.find("## Version History")
    assert start >= 0, "spec has no '## Version History' section"
    m = re.search(rf"^- \*\*{_SEMVER}\*\*", text[start:], re.MULTILINE)
    assert m, "Version History has no '- **X.Y.Z**' entry"
    return m.group(1)


@pytest.mark.spec
def test_version_triple_agrees():
    schema_version = _schema_title_version(_schema())
    text = _spec_text()
    versions = {
        "mlxk2.spec.JSON_API_SPEC_VERSION": JSON_API_SPEC_VERSION,
        "docs/json-api-schema.json title": schema_version,
        "docs/json-api-specification.md header": _spec_header_version(text),
        "docs/json-api-specification.md Version History": _spec_history_head_version(text),
    }
    assert len(set(versions.values())) == 1, f"JSON API version drift: {versions}"


def _run_success(finish_reason) -> dict:
    # The envelope cli.py builds for `run --json` (no api_version field)
    return {
        "status": "success",
        "command": "run",
        "data": {
            "model": "mlx-community/Phi-3-mini-4k-instruct-4bit",
            "prompt": "Count from 1 to 500.",
            "response": "1\n2\n3",
            "finish_reason": finish_reason,
        },
        "error": None,
    }


@pytest.mark.spec
@pytest.mark.parametrize("finish_reason", ["stop", "length", None])
def test_run_finish_reason_validates(finish_reason):
    errors = list(Draft7Validator(_schema()).iter_errors(_run_success(finish_reason)))
    assert not errors, f"run envelope invalid: {errors[0].message} at {'/'.join(map(str, errors[0].path)) or '<root>'}"


@pytest.mark.spec
def test_run_finish_reason_rejects_unknown_value():
    errors = list(Draft7Validator(_schema()).iter_errors(_run_success("banana")))
    assert errors, "schema accepted finish_reason 'banana'"
    assert any(list(e.path) == ["data", "finish_reason"] for e in errors), [e.message for e in errors]


@pytest.mark.spec
def test_run_context_length_reject_validates_and_is_listed():
    envelope = {
        "status": "error",
        "command": "run",
        "data": None,
        "error": {
            "type": "context_length_exceeded",
            "message": "Prompt is 5000 tokens, but the model's context window is 4096 tokens; "
            "nothing is left to generate. Shorten the prompt.",
        },
    }
    errors = list(Draft7Validator(_schema()).iter_errors(envelope))
    assert not errors, f"run error envelope invalid: {errors[0].message}"
    assert "- `context_length_exceeded`" in _spec_text(), "error type missing from the spec's Error Types list"
