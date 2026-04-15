"""Automated smoke tests: JSON output validation per command.

Validates that every mlxk command with --json produces:
1. CLEAN stdout (no [INFO], tqdm, or other noise before/after JSON)
2. Valid JSON parseable by json.loads() — same strictness as `python -m json.tool`
3. Schema-valid output against docs/json-api-schema.json (0.2.1)

Mirrors SMOKE-TEST-beta3.md Section A.
Runs as part of wet-umbrella Phase 1 (wet marker).
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import uuid
from pathlib import Path

import pytest

try:
    from jsonschema import Draft7Validator
    _schema_path = Path(__file__).parent.parent.parent / "docs" / "json-api-schema.json"
    _schema = json.loads(_schema_path.read_text()) if _schema_path.exists() else None
    _validator = Draft7Validator(_schema) if _schema else None
except ImportError:
    _validator = None

ws_home = os.environ.get("MLXK_WORKSPACE_HOME")
ws_home_valid = ws_home and Path(ws_home).is_dir()

TEST_PREFIX = "mlxk-test-"

pytestmark = [
    pytest.mark.live,
    pytest.mark.wet,
    pytest.mark.skipif(
        not ws_home_valid,
        reason="MLXK_WORKSPACE_HOME not set or not a directory",
    ),
]


def _run_mlxk(*args: str, timeout: int = 60) -> subprocess.CompletedProcess:
    """Run mlxk CLI as subprocess with current env."""
    return subprocess.run(
        [sys.executable, "-m", "mlxk2.cli", *args],
        capture_output=True, text=True, timeout=timeout,
        env=os.environ.copy(),
    )


def _validate_json(stdout: str, expected_command: str):
    """Strict validation: clean stdout + valid JSON + schema compliance.

    Same strictness as `python -m json.tool` — no noise tolerance.
    """
    # 1. stdout must be parseable as JSON directly (no noise)
    text = stdout.strip()
    assert text, "stdout is empty"
    try:
        data = json.loads(text)
    except json.JSONDecodeError as e:
        # Show what's on stdout for debugging
        preview = text[:200] if len(text) > 200 else text
        pytest.fail(
            f"stdout is not clean JSON (same as `python -m json.tool` failure):\n"
            f"  {e}\n"
            f"  stdout preview: {preview!r}"
        )

    # 2. Envelope check
    assert data.get("command") == expected_command, (
        f"Expected command={expected_command}, got {data.get('command')}"
    )

    # 3. Schema validation
    if _validator:
        errors = sorted(_validator.iter_errors(data), key=lambda e: e.path)
        assert not errors, (
            f"Schema validation failed for {expected_command}: "
            f"{errors[0].message} at {'/'.join(map(str, errors[0].path)) or '<root>'}"
        )

    return data


def _find_workspace_model() -> str | None:
    """Find first workspace model in MLXK_WORKSPACE_HOME."""
    if not ws_home:
        return None
    for d in sorted(Path(ws_home).iterdir(), key=lambda x: x.name):
        if d.is_dir() and (d / "config.json").exists():
            return d.name
    return None


class TestJsonSmoke:
    """SMOKE-TEST-beta3.md Section A: JSON output per command (read-only)."""

    def test_a1_list_json(self):
        """mlxk list --json → clean stdout, schema-valid."""
        r = _run_mlxk("list", "--json")
        data = _validate_json(r.stdout, "list")
        assert data["status"] == "success"
        assert isinstance(data["data"]["models"], list)

    def test_a2_show_json(self):
        """mlxk show <workspace> --json → clean stdout, schema-valid."""
        model = _find_workspace_model()
        if not model:
            pytest.skip("No workspace model found")
        r = _run_mlxk("show", str(Path(ws_home) / model), "--json")
        data = _validate_json(r.stdout, "show")
        assert data["status"] == "success"
        assert "model" in data["data"]

    def test_a3_health_json(self):
        """mlxk health --json → clean stdout, schema-valid, finds workspaces."""
        r = _run_mlxk("health", "--json")
        data = _validate_json(r.stdout, "health")
        assert data["status"] == "success"
        assert data["data"]["summary"]["total"] > 0, "No models found"

    def test_a6_health_error_json(self):
        """mlxk health nonexistent --json → clean error JSON."""
        r = _run_mlxk("health", "nonexistent-model-xyz", "--json")
        _validate_json(r.stdout, "health")

    def test_a7_convert_error_json(self):
        """mlxk convert nonexistent --json → clean error JSON, schema-valid."""
        r = _run_mlxk("convert", "/nonexistent/source", "/tmp/x",
                       "--repair-index", "--json")
        data = _validate_json(r.stdout, "convert")
        assert data["status"] == "error"

    def test_a8_version_json(self):
        """mlxk --version --json → clean stdout, schema-valid."""
        r = _run_mlxk("--version", "--json")
        data = _validate_json(r.stdout, "version")
        assert "cli_version" in data["data"]
        assert "json_api_spec_version" in data["data"]


class TestJsonSmokeWrite:
    """SMOKE-TEST-beta3.md Section A4-A5: write operations.

    Uses mlxk-test- prefix targets in MLXK_WORKSPACE_HOME.
    Subprocess-isolated to avoid mlx nanobind double-import.
    """

    @pytest.fixture(autouse=True)
    def _setup_target(self):
        """Create unique target name, clean up after."""
        self._target_name = f"{TEST_PREFIX}smoke-{uuid.uuid4().hex[:8]}"
        self._target_path = Path(ws_home) / self._target_name
        yield
        if self._target_path.exists() and self._target_path.name.startswith(TEST_PREFIX):
            shutil.rmtree(self._target_path)

    def test_a4_clone_error_json(self):
        """mlxk clone collision --json → clean error JSON, schema-valid.

        Clone success requires download → Phase 3 (live_clone).
        Here: validate error path produces clean, schema-valid JSON.
        """
        existing = _find_workspace_model()
        if not existing:
            pytest.skip("No workspace model for collision test")
        r = _run_mlxk("clone", "org/any-model", existing, "--json", timeout=30)
        data = _validate_json(r.stdout, "clone")
        assert data["status"] == "error"

    def test_a5_convert_json(self):
        """mlxk convert --quantize --json → clean stdout, schema-valid."""
        source = _find_workspace_model()
        if not source:
            pytest.skip("No workspace model found")
        # Skip audio models (mlx-lm can't quantize whisper)
        source_path = Path(ws_home) / source
        try:
            config = json.loads((source_path / "config.json").read_text())
            if config.get("model_type", "").lower() in ("whisper", "whisper_v3", "qwen2_audio"):
                pytest.skip(f"Audio model {source} not quantizable")
        except (ValueError, OSError):
            pass

        r = _run_mlxk("convert", source, self._target_name,
                       "--quantize", "8", "--json", timeout=300)
        data = _validate_json(r.stdout, "convert")
        assert data["status"] == "success", f"Convert failed: {data.get('error')}"
        assert data["data"]["mode"] == "quantize"
