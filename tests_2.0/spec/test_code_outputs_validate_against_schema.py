"""Validate actual command outputs against the JSON schema.

This complements the doc example validation by checking the live outputs
returned from operations and the CLI, using the isolated test cache.
If jsonschema is not installed locally, these tests are skipped.
"""

from __future__ import annotations

import json
from pathlib import Path
import sys
import pytest


def _load_schema():
    try:
        import jsonschema  # noqa: F401
    except Exception:
        pytest.skip("jsonschema not installed; skipping schema validation tests", allow_module_level=True)

    schema_path = Path("docs/json-api-schema.json")
    assert schema_path.exists(), "Schema file docs/json-api-schema.json missing"
    return json.loads(schema_path.read_text(encoding="utf-8"))


def _get_validator():
    try:
        from jsonschema import Draft7Validator
    except Exception:
        pytest.skip("jsonschema not available", allow_module_level=True)
    return Draft7Validator(_load_schema())


@pytest.mark.spec
def test_list_output_matches_schema(mock_models, isolated_cache):
    from mlxk2.operations.list import list_models
    validator = _get_validator()

    data = list_models()
    errors = sorted(validator.iter_errors(data), key=lambda e: e.path)
    assert not errors, f"list output invalid: {errors[0].message} at {'/'.join(map(str, errors[0].path)) or '<root>'}"


@pytest.mark.spec
def test_show_outputs_match_schema(mock_models, isolated_cache):
    from mlxk2.operations.show import show_model_operation
    validator = _get_validator()

    name = "mlx-community/Phi-3-mini-4k-instruct-4bit"

    base = show_model_operation(name)
    files = show_model_operation(name, include_files=True, include_config=False)
    cfg = show_model_operation(name, include_files=False, include_config=True)

    for label, payload in ("base", base), ("files", files), ("config", cfg):
        errors = sorted(_get_validator().iter_errors(payload), key=lambda e: e.path)
        assert not errors, f"show ({label}) output invalid: {errors[0].message} at {'/'.join(map(str, errors[0].path)) or '<root>'}"


@pytest.mark.spec
def test_health_output_matches_schema(mock_models, isolated_cache):
    from mlxk2.operations.health import health_check_operation
    validator = _get_validator()

    data = health_check_operation()
    errors = sorted(validator.iter_errors(data), key=lambda e: e.path)
    assert not errors, f"health output invalid: {errors[0].message} at {'/'.join(map(str, errors[0].path)) or '<root>'}"


@pytest.mark.spec
def test_rm_output_matches_schema(monkeypatch, mock_models, isolated_cache):
    from mlxk2.operations.rm import rm_operation
    validator = _get_validator()

    # Delete an existing model in the isolated cache
    name = "mlx-community/Qwen3-30B-A3B-Instruct-2507-4bit"
    res = rm_operation(name, force=True)
    errors = sorted(validator.iter_errors(res), key=lambda e: e.path)
    assert not errors, f"rm output invalid: {errors[0].message} at {'/'.join(map(str, errors[0].path)) or '<root>'}"


@pytest.mark.spec
def test_pull_output_matches_schema_already_exists(mock_models, isolated_cache):
    from mlxk2.operations.pull import pull_operation
    validator = _get_validator()

    # For an already-cached healthy model, pull should return already_exists
    name = "mlx-community/Phi-3-mini-4k-instruct-4bit"
    res = pull_operation(name)
    errors = sorted(validator.iter_errors(res), key=lambda e: e.path)
    assert not errors, f"pull output invalid: {errors[0].message} at {'/'.join(map(str, errors[0].path)) or '<root>'}"


@pytest.mark.spec
def test_version_output_matches_schema(monkeypatch, capsys):
    from mlxk2 import cli
    validator = _get_validator()

    monkeypatch.setattr(sys, "argv", ["mlxk2", "--version", "--json"]) 
    with pytest.raises(SystemExit) as exc:
        cli.main()
    assert exc.value.code == 0

    out = capsys.readouterr().out.strip()
    payload = json.loads(out)
    errors = sorted(validator.iter_errors(payload), key=lambda e: e.path)
    assert not errors, f"version output invalid: {errors[0].message} at {'/'.join(map(str, errors[0].path)) or '<root>'}"

