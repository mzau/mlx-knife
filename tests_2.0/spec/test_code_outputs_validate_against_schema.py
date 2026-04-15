from __future__ import annotations

"""Validate actual command outputs against the JSON schema.

This complements the doc example validation by checking the live outputs
returned from operations and the CLI, using the isolated test cache.
If jsonschema is not installed locally, these tests are skipped.
"""

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
def test_pull_output_matches_schema_already_exists():
    """Test pull response schema with static example data."""
    validator = _get_validator()

    # Static example of pull operation response for already-cached model
    pull_response = {
        "status": "success",
        "command": "pull",
        "error": None,
        "data": {
            "model": "mlx-community/Phi-3-mini-4k-instruct-4bit",
            "download_status": "already_exists",
            "message": "Model mlx-community/Phi-3-mini-4k-instruct-4bit already exists in cache",
            "expanded_name": None
        }
    }

    errors = sorted(validator.iter_errors(pull_response), key=lambda e: e.path)
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


@pytest.mark.spec
def test_convert_output_matches_schema():
    """Test convert response schema with static example data."""
    validator = _get_validator()

    # Static example: quantize operation (text model)
    convert_response = {
        "status": "success",
        "command": "convert",
        "error": None,
        "data": {
            "source": "/path/to/source-workspace",
            "target": "/path/to/target-workspace",
            "mode": "quantize",
            "health_check": True,
            "message": "Quantized to 4-bit successfully",
            "bits": 4,
            "group_size": 64,
            "health_status": "healthy",
            "health_reason": "Multi-file model complete",
            "content_hash": "bdd5a5be06e3e612bd91ad8ebdcfa787c892e0cf78dc2a585e0a092e32e9ed5f"
        }
    }

    errors = sorted(validator.iter_errors(convert_response), key=lambda e: e.path)
    assert not errors, f"convert output invalid: {errors[0].message} at {'/'.join(map(str, errors[0].path)) or '<root>'}"

    # Static example: quantize with warning (re-quantizing already quantized model)
    convert_warning = {
        "status": "success",
        "command": "convert",
        "error": None,
        "data": {
            "source": "/path/to/4bit-model",
            "target": "/path/to/8bit-model",
            "mode": "quantize",
            "health_check": True,
            "warning": "Source already 4-bit. Quantizing to 8-bit does not increase precision.",
            "message": "Quantized to 8-bit successfully",
            "bits": 8,
            "group_size": 64,
            "health_status": "healthy",
            "health_reason": "Multi-file model complete",
            "content_hash": "abc123"
        }
    }

    errors = sorted(validator.iter_errors(convert_warning), key=lambda e: e.path)
    assert not errors, f"convert (warning) output invalid: {errors[0].message} at {'/'.join(map(str, errors[0].path)) or '<root>'}"

    # Static example: repair-index operation
    repair_response = {
        "status": "success",
        "command": "convert",
        "error": None,
        "data": {
            "source": "/path/to/broken-workspace",
            "target": "/path/to/fixed-workspace",
            "mode": "repair-index",
            "health_check": True,
            "message": "Index rebuilt successfully",
            "health_status": "healthy",
            "health_reason": "Multi-file model complete",
            "content_hash": "def456"
        }
    }

    errors = sorted(validator.iter_errors(repair_response), key=lambda e: e.path)
    assert not errors, f"convert (repair) output invalid: {errors[0].message} at {'/'.join(map(str, errors[0].path)) or '<root>'}"


@pytest.mark.spec
def test_clone_output_matches_schema():
    """Test clone response schema with static example data."""
    validator = _get_validator()

    # Static example: successful clone operation
    clone_response = {
        "status": "success",
        "command": "clone",
        "error": None,
        "data": {
            "model": "mlx-community/Phi-3-mini-4k-instruct-4bit",
            "clone_status": "success",
            "target_dir": "/Users/me/workspaces/Phi-3-mini-4k-instruct-4bit",
            "message": "Cloned mlx-community/Phi-3-mini-4k-instruct-4bit",
            "health_check": True,
            "health": "healthy",
            "health_reason": "Multi-file model complete"
        }
    }

    errors = sorted(validator.iter_errors(clone_response), key=lambda e: e.path)
    assert not errors, f"clone output invalid: {errors[0].message} at {'/'.join(map(str, errors[0].path)) or '<root>'}"

    # Static example: clone error
    clone_error = {
        "status": "error",
        "command": "clone",
        "error": {"type": "not_found", "message": "Model not found"},
        "data": {
            "model": "nonexistent/model",
            "clone_status": "error",
            "target_dir": "/tmp/target",
            "message": "Model not found in HuggingFace"
        }
    }

    errors = sorted(validator.iter_errors(clone_error), key=lambda e: e.path)
    assert not errors, f"clone error output invalid: {errors[0].message} at {'/'.join(map(str, errors[0].path)) or '<root>'}"


@pytest.mark.spec
def test_push_output_matches_schema():
    """Test push response schema with static example data."""
    validator = _get_validator()

    # Static example of push operation response (matches schema requirements)
    push_response = {
        "status": "success",
        "command": "push",
        "error": None,
        "data": {
            "repo_id": "user/custom-model",
            "branch": "main",
            "repo_url": "https://huggingface.co/user/custom-model",
            "uploaded_files_count": 5,
            "experimental": False,
            "disclaimer": "Push completed successfully"
        }
    }

    errors = sorted(validator.iter_errors(push_response), key=lambda e: e.path)
    assert not errors, f"push output invalid: {errors[0].message} at {'/'.join(map(str, errors[0].path)) or '<root>'}"


@pytest.mark.spec
def test_run_output_matches_schema():
    """Test run response schema with static example data."""
    validator = _get_validator()

    # Static example of run operation response (non-streaming)
    run_response = {
        "status": "success",
        "command": "run",
        "error": None,
        "data": {
            "model": "mlx-community/Phi-3-mini-4k-instruct-4bit",
            "prompt": "Hello world",
            "response": "Hello! How can I help you today?",
            "tokens_generated": 8,
            "generation_time_s": 0.95
        }
    }

    errors = sorted(validator.iter_errors(run_response), key=lambda e: e.path)
    assert not errors, f"run output invalid: {errors[0].message} at {'/'.join(map(str, errors[0].path)) or '<root>'}"


# NOTE: serve/server commands don't produce JSON output - they run as server processes
# Only error cases would produce JSON, which are covered by general error handling

