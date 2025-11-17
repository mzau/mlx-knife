"""CLI Integration E2E Tests (ADR-011).

Validates CLI commands with real models:
- `mlxk run` basic functionality
- `--json` flag output formatting
- Exit code propagation (Issue #38)
- Stop token filtering across CLI interface

Test Strategy:
- Uses Portfolio Discovery for model selection
- RAM-aware testing
- Subprocess-based execution (true E2E)
- Deterministic sampling (temperature=0.0) for reproducible code tests
- Reuses exit code patterns from test_cli_run_exit_codes.py

NOTE: Tests use temperature=0.0 (not CLI default 0.7) for deterministic
code validation. Real usage testing with default settings is covered by
ADR-013 (Community Model Quality Benchmarks).

Opt-in via: pytest -m live_e2e
Requires: HF_HOME set to model cache
"""

from __future__ import annotations

import sys
import json
import os
import subprocess
import pytest
from pathlib import Path

# Import test utilities
from .test_utils import (
    should_skip_model,
    TEST_PROMPT,
    MAX_TOKENS,
    TEST_TEMPERATURE,
)
# portfolio_models fixture is provided by conftest.py

# Opt-in markers
pytestmark = [pytest.mark.live_e2e, pytest.mark.slow]


def _run_mlxk_subprocess(args: list[str], timeout: int = 60) -> tuple[str, str, int]:
    """Run mlxk CLI in subprocess and capture output.

    Args:
        args: CLI arguments (e.g., ["run", "model-id", "prompt"])
        timeout: Timeout in seconds

    Returns:
        (stdout, stderr, exit_code) tuple
    """
    result = subprocess.run(
        [sys.executable, "-m", "mlxk2.cli"] + args,
        capture_output=True,
        text=True,
        timeout=timeout
    )
    return result.stdout, result.stderr, result.returncode


class TestRunCommandBasic:
    """Basic `mlxk run` functionality tests.

    Tests are parametrized per model via pytest_generate_tests hook.
    Each test runs independently for clean isolation.
    """

    @pytest.mark.live_e2e
    def test_run_command(self, portfolio_models, model_key, report_benchmark):
        """Validate `mlxk run` with model.

        Parametrized test (one instance per model in portfolio).

        Tests:
        - Exit code 0 on success
        - No visible stop tokens in output
        - Output is non-empty
        """
        model_info = portfolio_models[model_key]
        model_id = model_info["id"]

        # RAM gating
        should_skip, skip_reason = should_skip_model(model_key, portfolio_models)
        if should_skip:
            pytest.skip(skip_reason)

        print(f"\nTesting {model_key}: {model_id}")

        args = ["run", model_id, TEST_PROMPT, "--max-tokens", str(MAX_TOKENS), "--temperature", str(TEST_TEMPERATURE)]
        stdout, stderr, exit_code = _run_mlxk_subprocess(args, timeout=90)

        # Validate exit code
        assert exit_code == 0, (
            f"Expected exit code 0, got {exit_code}\n"
            f"stdout: {stdout}\n"
            f"stderr: {stderr}"
        )

        # Validate output is non-empty
        assert stdout.strip(), "Output is empty"

        # Validate no visible stop tokens
        stop_tokens = [
            "<|end|>", "<|eot_id|>", "<|im_end|>",
            "<|endoftext|>", "</s>", "<|end_of_text|>"
        ]
        found_tokens = [t for t in stop_tokens if t in stdout]
        assert not found_tokens, (
            f"Model {model_id} has visible stop tokens: {found_tokens}\n"
            f"Output: {stdout!r}"
        )

        print(f"✓ {model_key}: Passed (output: {len(stdout)} chars)")

        # Benchmark reporting (ADR-013 Phase 0)
        report_benchmark(stop_tokens={
            "configured": stop_tokens,
            "detected": found_tokens,
            "workaround": "none",
            "leaked": len(found_tokens) > 0
        })


class TestRunCommandJSON:
    """JSON output mode tests.

    Tests are parametrized per model via pytest_generate_tests hook.
    Use pytest -k or --maxfail to limit test count if needed.
    """

    @pytest.mark.live_e2e
    def test_run_json_output(self, portfolio_models, model_key, report_benchmark):
        """Validate `mlxk run --json` output format.

        Parametrized test (one instance per model in portfolio).

        Tests:
        - JSON envelope structure
        - status: success on successful generation
        - data.response contains output
        - No visible stop tokens
        """
        model_info = portfolio_models[model_key]
        model_id = model_info["id"]

        # RAM gating
        should_skip, skip_reason = should_skip_model(model_key, portfolio_models)
        if should_skip:
            pytest.skip(skip_reason)

        print(f"\nTesting {model_key}: {model_id}")

        args = ["run", model_id, TEST_PROMPT, "--max-tokens", str(MAX_TOKENS), "--temperature", str(TEST_TEMPERATURE), "--json"]
        stdout, stderr, exit_code = _run_mlxk_subprocess(args, timeout=90)

        # Validate exit code
        assert exit_code == 0, (
            f"Expected exit code 0, got {exit_code}\n"
            f"stderr: {stderr}"
        )

        # Parse JSON
        data = json.loads(stdout)

        # Validate envelope structure
        assert "status" in data, "Missing 'status' field"
        assert data["status"] == "success", f"Expected status=success, got {data['status']}"
        assert "data" in data, "Missing 'data' field"
        assert "response" in data["data"], "Missing 'data.response' field"

        # Extract response
        response = data["data"]["response"]
        assert response.strip(), "Response is empty"

        # Validate no stop tokens
        stop_tokens = ["<|end|>", "<|eot_id|>", "<|im_end|>"]
        found_tokens = [t for t in stop_tokens if t in response]
        assert not found_tokens, (
            f"Model {model_id} has visible stop tokens in JSON: {found_tokens}\n"
            f"Response: {response!r}"
        )

        print(f"✓ {model_key}: Passed (JSON output: {len(response)} chars)")

        # Benchmark reporting (ADR-013 Phase 0)
        report_benchmark(stop_tokens={
            "configured": stop_tokens,
            "detected": found_tokens,
            "workaround": "none",
            "leaked": len(found_tokens) > 0
        })


class TestRunCommandExitCodes:
    """Exit code propagation tests (Issue #38)."""

    @pytest.mark.live_e2e
    def test_run_invalid_model_exit_code_text(self):
        """Validate exit code 1 for invalid model (text mode).

        Tests Issue #38 fix: CLI properly propagates errors.
        """
        stdout, stderr, exit_code = _run_mlxk_subprocess(
            ["run", "nonexistent/invalid-model-12345", "test"],
            timeout=30
        )

        # Should return exit code 1 for errors
        assert exit_code == 1, (
            f"Expected exit code 1 for invalid model, got {exit_code}\n"
            f"stdout: {stdout}"
        )

        # Error message should be present
        output = stdout + stderr
        assert "error" in output.lower() or "failed" in output.lower(), (
            f"Expected error message in output, got: {output}"
        )

    @pytest.mark.live_e2e
    def test_run_invalid_model_exit_code_json(self):
        """Validate exit code 1 for invalid model (JSON mode).

        Tests Issue #38 fix with --json flag.
        """
        stdout, stderr, exit_code = _run_mlxk_subprocess(
            ["run", "nonexistent/invalid-model-12345", "test", "--json"],
            timeout=30
        )

        # Should return exit code 1
        assert exit_code == 1, (
            f"Expected exit code 1 for invalid model, got {exit_code}\n"
            f"stdout: {stdout}"
        )

        # JSON should have error status
        try:
            data = json.loads(stdout)
            assert "status" in data, "Missing 'status' field in error JSON"
            assert data["status"] == "error", (
                f"Expected status=error, got {data['status']}"
            )
            assert "error" in data, "Missing 'error' field in error JSON"
        except json.JSONDecodeError:
            # If JSON parsing fails, check stderr
            assert "error" in stderr.lower(), (
                f"Expected error in stderr, got: {stderr}"
            )


class TestRunCommandStopTokens:
    """Specific stop token filtering validation."""

    @pytest.mark.live_e2e
    def test_run_no_visible_stop_tokens_mxfp4(self, portfolio_models):
        """Validate MXFP4 model has no visible stop tokens via CLI.

        Specific regression test for Issue #32 at CLI level.
        """
        # Find MXFP4 model in portfolio (or skip)
        mxfp4_model = None
        for model_key, model_info in portfolio_models.items():
            if "mxfp4" in model_key.lower() or "gpt-oss" in model_info["id"].lower():
                # Check RAM
                should_skip, skip_reason = should_skip_model(model_key, portfolio_models)
                if not should_skip:
                    mxfp4_model = model_info["id"]
                    break

        if mxfp4_model is None:
            pytest.skip("MXFP4 model not available in portfolio or exceeds RAM")

        print(f"\nTesting MXFP4: {mxfp4_model}")

        args = ["run", mxfp4_model, TEST_PROMPT, "--max-tokens", str(MAX_TOKENS), "--temperature", str(TEST_TEMPERATURE)]
        stdout, stderr, exit_code = _run_mlxk_subprocess(args, timeout=90)

        assert exit_code == 0, f"Command failed with exit code {exit_code}"

        # MXFP4-specific stop tokens
        mxfp4_stop_tokens = ["<|end|>", "<|return|>"]
        found_tokens = [t for t in mxfp4_stop_tokens if t in stdout]

        assert not found_tokens, (
            f"MXFP4 should filter stop tokens. Found: {found_tokens}\n"
            f"Output: {stdout!r}"
        )

        print(f"✓ MXFP4: No visible stop tokens via CLI")
