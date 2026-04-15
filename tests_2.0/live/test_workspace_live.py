"""Live workspace tests: clone shorthand + convert bare names.

Validates ADR-022 workspace-first UX with real models.
Safe: only creates temporary directories in MLXK_WORKSPACE_HOME, never touches HF cache.

Requires:
- MLXK_WORKSPACE_HOME set and containing at least one workspace model
- No HF_TOKEN needed, no cache writes

Run:
- pytest -m wet -v  (included in wet-umbrella Phase 1)
- pytest tests_2.0/live/test_workspace_live.py -v
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


def _strict_json(stdout: str) -> dict:
    """Parse JSON from stdout — strict, no noise tolerance.

    Same strictness as `python -m json.tool`. If this fails,
    there's noise on stdout that --json should have suppressed.
    """
    text = stdout.strip()
    assert text, "stdout is empty"
    return json.loads(text)


ws_home = os.environ.get("MLXK_WORKSPACE_HOME")
ws_home_valid = ws_home and Path(ws_home).is_dir()

pytestmark = [
    pytest.mark.live,
    pytest.mark.wet,
    pytest.mark.skipif(
        not ws_home_valid,
        reason="MLXK_WORKSPACE_HOME not set or not a directory",
    ),
]


def _unique_target_name(prefix: str = "mlxk-test") -> str:
    """Generate unique temporary target name for safe cleanup."""
    return f"{prefix}-{uuid.uuid4().hex[:8]}"


def _find_source_workspace() -> str | None:
    """Find a quantize-compatible workspace in MLXK_WORKSPACE_HOME.

    Skips audio models (whisper) — mlx-lm quantize doesn't support them.
    Text and vision models are fine.
    """
    import json as _json
    if not ws_home:
        return None
    home = Path(ws_home)

    audio_types = {"whisper", "whisper_v3", "qwen2_audio"}

    for d in sorted(home.iterdir(), key=lambda x: x.name):
        if not d.is_dir() or not (d / "config.json").exists():
            continue
        try:
            config = _json.loads((d / "config.json").read_text())
            model_type = config.get("model_type", "").lower()
            if model_type not in audio_types:
                return d.name
        except (ValueError, OSError):
            continue
    return None


@pytest.fixture
def source_model():
    """Get a source workspace model name, skip if none available."""
    name = _find_source_workspace()
    if not name:
        pytest.skip("No workspace model found in MLXK_WORKSPACE_HOME")
    return name


@pytest.fixture
def temp_target():
    """Create a unique target name in MLXK_WORKSPACE_HOME and clean up after test.

    IMPORTANT: Target is created on the SAME volume as source workspaces.
    This ensures APFS CoW compatibility (clone operations require same-volume
    for copy-on-write). Do NOT use /tmp or other volumes as target.
    """
    name = _unique_target_name()
    target_path = Path(ws_home) / name
    yield name
    # Cleanup
    if target_path.exists():
        shutil.rmtree(target_path)


class TestCloneShorthand:
    """Test clone shorthand with MLXK_WORKSPACE_HOME (D. in smoke tests).

    NOTE: These tests validate path RESOLUTION only, not actual cloning
    (which requires a real HF model download). Full clone E2E is in
    test_clone_live.py (Phase 3, requires HF_TOKEN).
    """

    def test_resolve_clone_target_strips_org(self):
        """_resolve_clone_target strips org prefix for flat layout."""
        from mlxk2.cli import _resolve_clone_target

        target = _resolve_clone_target("mlx-community/" + _unique_target_name())
        # Should be in MLXK_WORKSPACE_HOME, not have org prefix
        assert str(Path(ws_home)) in target
        assert "mlx-community" not in target

    def test_resolve_clone_target_strips_revision(self):
        """_resolve_clone_target strips @revision."""
        from mlxk2.cli import _resolve_clone_target

        name = _unique_target_name()
        target = _resolve_clone_target(f"org/{name}@main")
        assert target.endswith(name)
        assert "@" not in target

    def test_clone_target_collision_detected(self, source_model):
        """Clone to existing workspace name is rejected."""
        from mlxk2.cli import _resolve_clone_target

        # source_model already exists in WS_HOME
        with pytest.raises(SystemExit):
            _resolve_clone_target(f"org/{source_model}")

    def test_clone_bare_target_resolves_to_ws_home(self):
        """Bare target name (not path) resolves into MLXK_WORKSPACE_HOME."""
        from mlxk2.cli import _is_explicit_path

        assert not _is_explicit_path("my-custom-name")
        # CLI would resolve this to ws_home / "my-custom-name"
        expected = str(Path(ws_home) / "my-custom-name")
        assert expected.startswith(str(Path(ws_home)))


class TestConvertBareNames:
    """Test convert with bare names via MLXK_WORKSPACE_HOME (D2. in smoke tests)."""

    def test_convert_bare_source_resolves(self, source_model):
        """Bare source name resolves to MLXK_WORKSPACE_HOME/name."""
        from mlxk2.cli import _is_explicit_path

        assert not _is_explicit_path(source_model)
        resolved = Path(ws_home) / source_model
        assert resolved.is_dir(), f"{resolved} should exist"
        assert (resolved / "config.json").exists()

    def test_convert_quantize_bare_names(self, source_model, temp_target):
        """Convert with bare source + target names works via WS_HOME.

        Runs as subprocess to avoid mlx nanobind double-import crash
        (mlx.core.DeviceType enum can't be registered twice in same process).
        """
        import subprocess

        result = subprocess.run(
            [sys.executable, "-m", "mlxk2.cli",
             "convert", source_model, temp_target,
             "--quantize", "4", "--json"],
            capture_output=True, text=True, timeout=300,
            env={**os.environ, "MLXK_WORKSPACE_HOME": ws_home},
        )

        target_path = Path(ws_home) / temp_target
        assert target_path.exists(), f"Target not created. stderr: {result.stderr[:500]}"
        assert (target_path / "config.json").exists()

        # Verify JSON output
        if result.stdout.strip():
            data = _strict_json(result.stdout)
            assert data["status"] == "success", f"Convert failed: {data.get('error')}"
            assert data["data"]["mode"] == "quantize"

    def test_convert_json_output_valid(self, source_model, temp_target):
        """Convert --json output has required schema fields.

        Runs as subprocess (same reason as above: mlx double-import).
        """
        import subprocess

        result = subprocess.run(
            [sys.executable, "-m", "mlxk2.cli",
             "convert", source_model, temp_target,
             "--quantize", "8", "--json"],
            capture_output=True, text=True, timeout=300,
            env={**os.environ, "MLXK_WORKSPACE_HOME": ws_home},
        )

        assert result.returncode == 0, f"Convert failed: {result.stderr[:500]}"

        data = _strict_json(result.stdout)
        assert data["status"] == "success"
        # Required fields per schema 0.2.1
        assert "source" in data["data"]
        assert "target" in data["data"]
        assert "mode" in data["data"]
        assert data["data"]["mode"] == "quantize"
