"""Live pipe-mode E2E checks (stdin '-', JSON error path, mlx-run wrapper).

Opt-in: pytest -m live_e2e
Requires: HF_HOME with at least one MLX chat model (uses Portfolio Discovery).
"""

from __future__ import annotations

import json
import subprocess
import sys
import os
from typing import Dict, Any, Tuple

import pytest

from .test_utils import should_skip_model, MAX_TOKENS, TEST_TEMPERATURE

pytestmark = [pytest.mark.live_e2e, pytest.mark.slow]


def _pick_first_eligible_model(portfolio_models: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    """Select the first model that passes RAM gating."""
    for key, info in portfolio_models.items():
        should_skip, _ = should_skip_model(key, portfolio_models)
        if not should_skip:
            return info
    pytest.skip("No suitable models found in portfolio (RAM gating)")


def _run_cli(args: list[str], stdin: str | None = None, timeout: int = 120) -> Tuple[str, str, int]:
    """Run mlxk CLI as subprocess."""
    result = subprocess.run(
        [sys.executable, "-m", "mlxk2.cli"] + args,
        input=stdin,
        text=True,
        capture_output=True,
        timeout=timeout,
    )
    return result.stdout, result.stderr, result.returncode


class TestPipeModeSingleModel:
    """Exercise pipe workflows against one dynamically discovered model."""

    @pytest.fixture(scope="class")
    def model_id(self, portfolio_models):
        if not os.getenv("MLXK2_ENABLE_PIPES"):
            pytest.skip("Pipe mode gated by MLXK2_ENABLE_PIPES=1")
        model = _pick_first_eligible_model(portfolio_models)
        return model["id"]

    def test_stdin_dash_appends_trailing_text(self, model_id):
        """stdin ('-') should be read and combined with trailing CLI text."""
        stdin_text = "from stdin"
        trailing = "append extra context"
        args = [
            "run",
            model_id,
            "-",
            trailing,
            "--max-tokens",
            str(MAX_TOKENS),
            "--temperature",
            str(TEST_TEMPERATURE),
        ]

        stdout, stderr, code = _run_cli(args, stdin=stdin_text, timeout=180)

        assert code == 0, f"exit={code}, stderr={stderr!r}"
        assert stdout.strip(), "Expected non-empty model output"

    def test_json_interactive_error_path(self, model_id):
        """Interactive JSON (no prompt) should emit JSON error on stdout and exit non-zero."""
        args = ["run", model_id, "--json"]
        stdout, stderr, code = _run_cli(args, timeout=60)

        assert code != 0, "Expected non-zero exit for interactive JSON"
        data = json.loads(stdout)
        assert data["status"] == "error"
        assert "interactive" in data["error"]["message"].lower()

    def test_pipe_from_list_json(self, model_id):
        """Pipe mlxk list --json into run via stdin '-'."""
        list_out, list_err, list_code = _run_cli(["list", "--json"], timeout=60)
        assert list_code == 0, f"list failed: {list_err}"
        assert list_out.strip(), "list --json returned empty output"

        args = [
            "run",
            model_id,
            "-",
            "Summarize the model list as a concise table.",
            "--max-tokens",
            str(MAX_TOKENS),
            "--temperature",
            str(TEST_TEMPERATURE),
        ]
        stdout, stderr, code = _run_cli(args, stdin=list_out, timeout=180)

        assert code == 0, f"exit={code}, stderr={stderr!r}"
        assert stdout.strip(), "Expected non-empty summary output"
