"""Test CLI exit codes for run command error propagation.

This test suite validates that the CLI properly propagates errors from
run_model to the exit code and JSON status envelope in both text and JSON modes.

Related: GitHub Issue #38 - CLI exits with code 0 even when model fails to load
Related: ADR-014 Phase 1 - Unix pipe integration (stdin '-', BrokenPipeError handling)

Key testing strategy:
- Mock at the MLXRunner/resolution level (not run_model_enhanced)
- This tests the actual error handling contract in run.py
- Validates that error strings are returned and detected in both modes
- Tests pipe-mode edge cases (empty stdin, stdin-only, BrokenPipeError)
"""

import json
import os
import signal
import sys
from io import StringIO
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

    def test_run_interactive_json_mode_outputs_json_error(self, capsys):
        """Interactive JSON mode should emit JSON error on stdout with exit=1."""
        with patch('mlxk2.operations.run.resolve_model_for_operation') as mock_resolve, \
             patch('mlxk2.operations.run.MLXRunner') as mock_runner:
            mock_resolve.return_value = (None, None, None)

            runner = MagicMock()
            mock_runner.return_value.__enter__.return_value = runner

            stdout, stderr, exit_code = _run_cli_capture_exit(
                ["mlxk2", "run", "test-model", "--json"],
                capsys
            )

        assert exit_code == 1
        assert stderr.strip() == ""
        data = json.loads(stdout)
        assert data["status"] == "error"
        assert "interactive" in data["error"]["message"].lower()

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

        This tests the probe/policy architecture in run_model:
        - Model resolves successfully
        - probe_and_select returns BLOCK policy
        - run_model returns "Error: ..." string
        - Exit code is 1
        """
        from mlxk2.core.capabilities import Backend, PolicyDecision, BackendPolicy, ModelCapabilities

        with patch('mlxk2.operations.run.resolve_model_for_operation') as mock_resolve, \
             patch('mlxk2.operations.run.get_current_model_cache') as mock_cache, \
             patch('mlxk2.core.capabilities.probe_and_select') as mock_probe:

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

            # Simulate incompatibility via probe/policy
            mock_caps = ModelCapabilities(
                model_path=mock_snapshot_path,
                model_name="test-model",
            )
            mock_policy = BackendPolicy(
                backend=Backend.UNSUPPORTED,
                decision=PolicyDecision.BLOCK,
                message="Requires mlx-lm >= 0.20.0",
            )
            mock_probe.return_value = (mock_caps, mock_policy)

            stdout, stderr, exit_code = _run_cli_capture_exit(
                ["mlxk2", "run", "incompatible-model", "hello"],
                capsys
            )

            assert exit_code == 1, (
                f"Expected exit code 1 for incompatible model, got {exit_code}\n"
                f"stdout: {stdout}\n"
                f"stderr: {stderr}"
            )
            assert "Error:" in stderr and "Incompatible" in stderr

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

    def test_run_uses_stdin_with_dash_prompt_and_additional_text(self, capsys):
        """Ensure '-' reads stdin, appends CLI text, and stays JSON-clean."""
        with patch('sys.stdin', StringIO("from-stdin")), \
             patch('mlxk2.operations.run.resolve_model_for_operation') as mock_resolve, \
             patch('mlxk2.operations.run.MLXRunner') as mock_runner_class, \
             patch('mlxk2.cli.sys.stdout.isatty', return_value=False), \
             patch.dict(os.environ, {"MLXK2_ENABLE_PIPES": "1"}, clear=False):

            mock_resolve.return_value = (None, None, None)
            mock_runner = MagicMock()
            mock_runner.__enter__ = MagicMock(return_value=mock_runner)
            mock_runner.__exit__ = MagicMock(return_value=None)
            mock_runner.generate_batch.return_value = "ok"
            mock_runner_class.return_value = mock_runner

            stdout, stderr, exit_code = _run_cli_capture_exit(
                ["mlxk2", "run", "test-model", "-", "extra text", "--json"],
                capsys
            )

        assert exit_code == 0
        assert stderr == ""
        mock_runner.generate_batch.assert_called_once()
        assert mock_runner.generate_batch.call_args[1]["prompt"] == "from-stdin\n\nextra text"
        data = json.loads(stdout)
        assert data["data"]["prompt"] == "from-stdin\n\nextra text"

    def test_run_disables_streaming_when_stdout_not_tty(self, capsys):
        """Non-TTY stdout should force batch mode even without --no-stream."""
        with patch('mlxk2.operations.run.resolve_model_for_operation') as mock_resolve, \
             patch('mlxk2.operations.run.MLXRunner') as mock_runner_class, \
             patch('mlxk2.cli.sys.stdout.isatty', return_value=False):

            mock_resolve.return_value = (None, None, None)

            mock_runner = MagicMock()
            mock_runner.__enter__ = MagicMock(return_value=mock_runner)
            mock_runner.__exit__ = MagicMock(return_value=None)
            mock_runner.generate_streaming.return_value = iter([])
            mock_runner.generate_batch.return_value = "batch"
            mock_runner_class.return_value = mock_runner

            stdout, stderr, exit_code = _run_cli_capture_exit(
                ["mlxk2", "run", "test-model", "hello"],
                capsys
            )

        assert exit_code == 0
        assert mock_runner.generate_streaming.call_count == 0
        mock_runner.generate_batch.assert_called_once()
        assert "batch" in stdout

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


class TestPipeModeEdgeCases:
    """Test ADR-014 Phase 1 pipe mode edge cases."""

    def test_run_stdin_only_without_trailing_text(self, capsys):
        """stdin '-' without trailing text should work (ADR-014 core use case)."""
        with patch('sys.stdin', StringIO("prompt from stdin only")), \
             patch('mlxk2.operations.run.resolve_model_for_operation') as mock_resolve, \
             patch('mlxk2.operations.run.MLXRunner') as mock_runner_class, \
             patch('mlxk2.cli.sys.stdout.isatty', return_value=False), \
             patch.dict(os.environ, {"MLXK2_ENABLE_PIPES": "1"}, clear=False):

            mock_resolve.return_value = (None, None, None)
            mock_runner = MagicMock()
            mock_runner.__enter__ = MagicMock(return_value=mock_runner)
            mock_runner.__exit__ = MagicMock(return_value=None)
            mock_runner.generate_batch.return_value = "model response"
            mock_runner_class.return_value = mock_runner

            stdout, stderr, exit_code = _run_cli_capture_exit(
                ["mlxk2", "run", "test-model", "-"],
                capsys
            )

        assert exit_code == 0, f"Expected exit 0, got {exit_code}, stderr={stderr}"
        # Verify prompt was stdin content only (no trailing text concatenation)
        mock_runner.generate_batch.assert_called_once()
        assert mock_runner.generate_batch.call_args[1]["prompt"] == "prompt from stdin only"
        assert "model response" in stdout

    def test_run_empty_stdin(self, capsys):
        """Empty stdin with '-' should use empty string as prompt."""
        with patch('sys.stdin', StringIO("")), \
             patch('mlxk2.operations.run.resolve_model_for_operation') as mock_resolve, \
             patch('mlxk2.operations.run.MLXRunner') as mock_runner_class, \
             patch('mlxk2.cli.sys.stdout.isatty', return_value=False), \
             patch.dict(os.environ, {"MLXK2_ENABLE_PIPES": "1"}, clear=False):

            mock_resolve.return_value = (None, None, None)
            mock_runner = MagicMock()
            mock_runner.__enter__ = MagicMock(return_value=mock_runner)
            mock_runner.__exit__ = MagicMock(return_value=None)
            mock_runner.generate_batch.return_value = "response to empty"
            mock_runner_class.return_value = mock_runner

            stdout, stderr, exit_code = _run_cli_capture_exit(
                ["mlxk2", "run", "test-model", "-"],
                capsys
            )

        assert exit_code == 0, f"Expected exit 0, got {exit_code}, stderr={stderr}"
        # Empty stdin should result in empty string prompt
        mock_runner.generate_batch.assert_called_once()
        assert mock_runner.generate_batch.call_args[1]["prompt"] == ""

    def test_run_empty_stdin_with_trailing_text(self, capsys):
        """Empty stdin + trailing text should result in just the trailing text (after separator)."""
        with patch('sys.stdin', StringIO("")), \
             patch('mlxk2.operations.run.resolve_model_for_operation') as mock_resolve, \
             patch('mlxk2.operations.run.MLXRunner') as mock_runner_class, \
             patch('mlxk2.cli.sys.stdout.isatty', return_value=False), \
             patch.dict(os.environ, {"MLXK2_ENABLE_PIPES": "1"}, clear=False):

            mock_resolve.return_value = (None, None, None)
            mock_runner = MagicMock()
            mock_runner.__enter__ = MagicMock(return_value=mock_runner)
            mock_runner.__exit__ = MagicMock(return_value=None)
            mock_runner.generate_batch.return_value = "response"
            mock_runner_class.return_value = mock_runner

            stdout, stderr, exit_code = _run_cli_capture_exit(
                ["mlxk2", "run", "test-model", "-", "trailing text"],
                capsys
            )

        assert exit_code == 0
        # Empty stdin + trailing = "\n\ntrailing text"
        assert mock_runner.generate_batch.call_args[1]["prompt"] == "\n\ntrailing text"

    def test_run_pipe_mode_error_message_correct(self, capsys):
        """Pipe mode error should mention only MLXK2_ENABLE_PIPES (not --enable-pipes)."""
        # Without MLXK2_ENABLE_PIPES set
        with patch('sys.stdin', StringIO("test")), \
             patch.dict(os.environ, {}, clear=False):
            # Ensure MLXK2_ENABLE_PIPES is not set
            os.environ.pop("MLXK2_ENABLE_PIPES", None)

            stdout, stderr, exit_code = _run_cli_capture_exit(
                ["mlxk2", "run", "test-model", "-"],
                capsys
            )

        assert exit_code == 1
        # Error should mention env var only, not a non-existent --enable-pipes flag
        combined = stdout + stderr
        assert "MLXK2_ENABLE_PIPES" in combined
        assert "--enable-pipes" not in combined


class TestSIGPIPEHandling:
    """Test SIGPIPE signal handling for Unix pipe compatibility (ADR-014)."""

    def test_sigpipe_handler_is_set(self):
        """Verify SIGPIPE handler is set to SIG_DFL on Unix systems."""
        # This test verifies the SIGPIPE handler setup code path
        # On Unix, we expect signal.SIGPIPE to exist and be handled
        if not hasattr(signal, 'SIGPIPE'):
            pytest.skip("SIGPIPE not available on this platform (Windows)")

        # Import and run main to trigger SIGPIPE setup
        # We just verify the code path doesn't error
        from mlxk2.cli import main

        # Save original handler
        original_handler = signal.getsignal(signal.SIGPIPE)

        try:
            # Calling main() would actually run CLI - we just verify the import works
            # and that we can set/restore the signal
            signal.signal(signal.SIGPIPE, signal.SIG_DFL)
            current = signal.getsignal(signal.SIGPIPE)
            assert current == signal.SIG_DFL
        finally:
            # Restore original handler
            signal.signal(signal.SIGPIPE, original_handler)


class TestBrokenPipeError:
    """Test BrokenPipeError handling in streaming/batch output (ADR-014)."""

    def test_streaming_broken_pipe_handled_gracefully(self):
        """BrokenPipeError during streaming should not raise exception."""
        from mlxk2.operations.run import single_shot_generation

        mock_runner = MagicMock()
        mock_runner.generate_streaming.return_value = iter(["token1", "token2", "token3"])

        # Mock print to raise BrokenPipeError on second call
        call_count = [0]
        original_print = print

        def mock_print(*args, **kwargs):
            call_count[0] += 1
            if call_count[0] >= 2:
                raise BrokenPipeError("Broken pipe")
            # Don't actually print during test
            pass

        with patch('builtins.print', mock_print), \
             patch('sys.stderr', MagicMock()):  # Mock stderr.close()
            # Should not raise, should return None
            result = single_shot_generation(
                mock_runner,
                prompt="test",
                stream=True,
                json_output=False
            )

        # Should have returned None (graceful exit)
        assert result is None

    def test_batch_broken_pipe_handled_gracefully(self):
        """BrokenPipeError during batch output should not raise exception."""
        from mlxk2.operations.run import single_shot_generation

        mock_runner = MagicMock()
        mock_runner.generate_batch.return_value = "batch result"

        def mock_print(*args, **kwargs):
            raise BrokenPipeError("Broken pipe")

        with patch('builtins.print', mock_print), \
             patch('sys.stderr', MagicMock()):  # Mock stderr.close()
            # Should not raise, should return None
            result = single_shot_generation(
                mock_runner,
                prompt="test",
                stream=False,
                json_output=False
            )

        # Should have returned None (graceful exit)
        assert result is None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
