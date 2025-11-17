"""Test CLI exit codes for run command error propagation.

This test suite validates that the CLI properly propagates errors from
run_model to the exit code and JSON status envelope in both text and JSON modes.

Related: GitHub Issue #38 - CLI exits with code 0 even when model fails to load

Key testing strategy:
- Mock at the MLXRunner/resolution level (not run_model_enhanced)
- This tests the actual error handling contract in run.py
- Validates that error strings are returned and detected in both modes
"""

import json
import sys
from unittest.mock import patch, MagicMock

import pytest


def _run_cli_capture_exit(argv: list[str], capsys):
    """Run CLI and capture output + exit code.

    Returns:
        tuple: (stdout, stderr, exit_code)
    """
    from mlxk2.cli import main as cli_main

    old_argv = sys.argv[:]
    sys.argv = argv[:]
    exit_code = None

    try:
        cli_main()
        exit_code = 0  # If no SystemExit raised, assume success
    except SystemExit as e:
        exit_code = e.code
    finally:
        sys.argv = old_argv

    captured = capsys.readouterr()
    return captured.out, captured.err, exit_code


class TestRunCommandExitCodes:
    """Test run command exit code propagation."""

    def test_run_nonexistent_model_text_mode_exit_code(self, capsys):
        """Test that run with invalid model returns non-zero exit code (text mode).

        This tests the real run_model error handling path:
        - resolve_model_for_operation fails (model not found)
        - run_model returns "Error: ..." string
        - CLI detects error and prints it (text mode)
        - Exit code is 1
        """
        # Mock resolve_model_for_operation to simulate nonexistent model
        with patch('mlxk2.operations.run.resolve_model_for_operation') as mock_resolve:
            # Simulate resolution failure by raising exception (typical behavior)
            mock_resolve.side_effect = RuntimeError("Failed to resolve model 'nonexistent-model'")

            stdout, stderr, exit_code = _run_cli_capture_exit(
                ["mlxk2", "run", "nonexistent-model", "hello"],
                capsys
            )

            # Should return non-zero exit code for errors
            assert exit_code == 1, (
                f"Expected exit code 1 for model error, got {exit_code}\n"
                f"stdout: {stdout}\n"
                f"stderr: {stderr}"
            )

            # In text mode, error is printed to stderr
            assert "Error:" in stderr, f"Expected error message in stderr, got: {stderr}"

    def test_run_nonexistent_model_json_mode_exit_code(self, capsys):
        """Test that run with invalid model returns non-zero exit code (JSON mode).

        This tests the real run_model error handling path in JSON mode:
        - resolve_model_for_operation fails
        - run_model returns "Error: ..." string
        - CLI wraps error in JSON envelope
        - Exit code is 1
        """
        # Mock resolve_model_for_operation to simulate nonexistent model
        with patch('mlxk2.operations.run.resolve_model_for_operation') as mock_resolve:
            mock_resolve.side_effect = RuntimeError("Failed to resolve model 'nonexistent-model'")

            stdout, stderr, exit_code = _run_cli_capture_exit(
                ["mlxk2", "run", "nonexistent-model", "hello", "--json"],
                capsys
            )

            # Should return non-zero exit code
            assert exit_code == 1, (
                f"Expected exit code 1 for model error, got {exit_code}\n"
                f"stdout: {stdout}\n"
                f"stderr: {stderr}"
            )

            # Parse JSON output from stdout (JSON mode always stdout for scripting)
            data = json.loads(stdout)

            # Should have status="error"
            assert data["status"] == "error", (
                f"Expected status='error', got '{data['status']}'\n"
                f"Full response: {json.dumps(data, indent=2)}"
            )
            assert data["error"] is not None, "Expected error field to be populated"
            assert data["error"]["message"], "Expected non-empty error message"

    def test_run_ambiguous_model_text_mode(self, capsys):
        """Test ambiguous model specification returns exit code 1 (text mode).

        This tests the ambiguous model detection path in run_model:
        - resolve_model_for_operation returns ambiguous list
        - run_model returns "Error: Ambiguous..." string
        - CLI prints error and exits with code 1
        """
        with patch('mlxk2.operations.run.resolve_model_for_operation') as mock_resolve:
            # Simulate ambiguous resolution
            mock_resolve.return_value = (None, None, ["model-a", "model-b"])

            stdout, stderr, exit_code = _run_cli_capture_exit(
                ["mlxk2", "run", "ambiguous", "hello"],
                capsys
            )

            assert exit_code == 1, (
                f"Expected exit code 1 for ambiguous model, got {exit_code}\n"
                f"stdout: {stdout}\n"
                f"stderr: {stderr}"
            )
            assert "Error:" in stderr and "Ambiguous" in stderr

    def test_run_ambiguous_model_json_mode(self, capsys):
        """Test ambiguous model specification returns exit code 1 (JSON mode)."""
        with patch('mlxk2.operations.run.resolve_model_for_operation') as mock_resolve:
            mock_resolve.return_value = (None, None, ["model-a", "model-b"])

            stdout, stderr, exit_code = _run_cli_capture_exit(
                ["mlxk2", "run", "ambiguous", "hello", "--json"],
                capsys
            )

            assert exit_code == 1
            # Parse JSON from stdout (JSON mode always stdout)
            data = json.loads(stdout)
            assert data["status"] == "error"
            assert "ambiguous" in data["error"]["message"].lower()

    def test_run_incompatible_model_text_mode(self, capsys):
        """Test incompatible model returns exit code 1 (text mode).

        This tests the runtime compatibility check in run_model:
        - Model resolves successfully
        - Compatibility check fails
        - run_model returns "Error: not compatible..." string
        - Exit code is 1
        """
        with patch('mlxk2.operations.run.resolve_model_for_operation') as mock_resolve, \
             patch('mlxk2.operations.run.get_current_model_cache') as mock_cache, \
             patch('mlxk2.operations.run.check_runtime_compatibility') as mock_compat:

            # Simulate successful resolution
            mock_resolve.return_value = ("test-model", None, None)

            # Simulate model exists in cache
            mock_cache_dir = MagicMock()
            mock_cache_dir.exists.return_value = True
            mock_snapshots_dir = MagicMock()
            mock_snapshots_dir.exists.return_value = True
            mock_snapshot_path = MagicMock()
            mock_snapshot_path.is_dir.return_value = True
            mock_snapshot_path.exists.return_value = True
            mock_snapshots_dir.iterdir.return_value = [mock_snapshot_path]
            mock_cache_dir.__truediv__ = lambda self, x: mock_snapshots_dir if x == "snapshots" else MagicMock()
            mock_cache_inst = MagicMock()
            mock_cache_inst.__truediv__ = lambda self, x: mock_cache_dir
            mock_cache.return_value = mock_cache_inst

            # Simulate incompatibility
            mock_compat.return_value = (False, "Requires mlx-lm >= 0.20.0")

            stdout, stderr, exit_code = _run_cli_capture_exit(
                ["mlxk2", "run", "incompatible-model", "hello"],
                capsys
            )

            assert exit_code == 1, (
                f"Expected exit code 1 for incompatible model, got {exit_code}\n"
                f"stdout: {stdout}\n"
                f"stderr: {stderr}"
            )
            assert "Error:" in stderr and "compatible" in stderr

    def test_run_success_text_mode_exit_code(self, capsys):
        """Test that successful run returns zero exit code (text mode).

        Mock at the MLXRunner level to simulate successful generation.
        """
        with patch('mlxk2.operations.run.MLXRunner') as mock_runner_class:
            # Setup mock runner context manager
            mock_runner = MagicMock()
            mock_runner.__enter__ = MagicMock(return_value=mock_runner)
            mock_runner.__exit__ = MagicMock(return_value=None)
            mock_runner_class.return_value = mock_runner

            # Mock successful generation
            with patch('mlxk2.operations.run.single_shot_generation') as mock_gen:
                mock_gen.return_value = "Generated response text"

                stdout, stderr, exit_code = _run_cli_capture_exit(
                    ["mlxk2", "run", "valid-model", "hello"],
                    capsys
                )

                assert exit_code == 0, (
                    f"Expected exit code 0 for success, got {exit_code}\n"
                    f"stdout: {stdout}\n"
                    f"stderr: {stderr}"
                )

    def test_run_success_json_mode_exit_code(self, capsys):
        """Test that successful run returns zero exit code (JSON mode)."""
        with patch('mlxk2.operations.run.MLXRunner') as mock_runner_class:
            mock_runner = MagicMock()
            mock_runner.__enter__ = MagicMock(return_value=mock_runner)
            mock_runner.__exit__ = MagicMock(return_value=None)
            mock_runner_class.return_value = mock_runner

            with patch('mlxk2.operations.run.single_shot_generation') as mock_gen:
                mock_gen.return_value = "Generated response text"

                stdout, stderr, exit_code = _run_cli_capture_exit(
                    ["mlxk2", "run", "valid-model", "hello", "--json"],
                    capsys
                )

                assert exit_code == 0, (
                    f"Expected exit code 0 for success, got {exit_code}\n"
                    f"stdout: {stdout}"
                )

                # Parse JSON output
                data = json.loads(stdout)
                assert data["status"] == "success"
                assert data["error"] is None
                assert data["data"]["response"] == "Generated response text"

    def test_run_runtime_exception_text_mode(self, capsys):
        """Test that runtime exceptions are caught and propagated as errors (text mode).

        This tests the exception handler in run_model (line 125-129):
        - MLXRunner raises exception during generation
        - run_model catches it and returns "Error: ..." string
        - Exit code is 1
        """
        with patch('mlxk2.operations.run.MLXRunner') as mock_runner_class:
            # Simulate MLXRunner raising exception during __enter__
            mock_runner_class.side_effect = RuntimeError("Model loading failed: Out of memory")

            stdout, stderr, exit_code = _run_cli_capture_exit(
                ["mlxk2", "run", "test-model", "hello"],
                capsys
            )

            assert exit_code == 1, (
                f"Expected exit code 1 for exception, got {exit_code}\n"
                f"stdout: {stdout}\n"
                f"stderr: {stderr}"
            )
            assert "Error:" in stderr
            assert "failed" in stderr.lower() or "memory" in stderr.lower()

    def test_run_runtime_exception_json_mode(self, capsys):
        """Test that runtime exceptions are caught and propagated as errors (JSON mode)."""
        with patch('mlxk2.operations.run.MLXRunner') as mock_runner_class:
            mock_runner_class.side_effect = RuntimeError("Model loading failed: Out of memory")

            stdout, stderr, exit_code = _run_cli_capture_exit(
                ["mlxk2", "run", "test-model", "hello", "--json"],
                capsys
            )

            assert exit_code == 1
            # Parse JSON from stdout (JSON mode always stdout)
            data = json.loads(stdout)
            assert data["status"] == "error"
            assert "failed" in data["error"]["message"].lower() or "memory" in data["error"]["message"].lower()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
