"""How `mlxk run` reports the end of a generation (#66).

One runner attribute (`last_finish_reason`), three CLI surfaces: the stderr notice after
a cut answer, `data.finish_reason` in the `--json` envelope, and the typed
`context_length_exceeded` reject. Runner mocked at the MLXRunner boundary like
test_run_complete.py; the CLI envelope tests patch `run_model_enhanced` and read stdout.
"""

import json
import sys
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest

from mlxk2.core.runner.token_limits import DEFAULT_MAX_TOKENS, ContextLengthExceeded
from mlxk2.operations.run import (
    _note_finish,
    interactive_chat,
    run_model,
    single_shot_generation,
)


def _finished_runner(reason="length", generated=100, prompt=50, budget=100, window=8192):
    """A runner after one generation, with the attributes `_note_finish` reads."""
    runner = Mock()
    runner.last_finish_reason = reason
    runner.last_completion_tokens = generated
    runner.last_prompt_tokens = prompt
    runner.last_max_tokens = budget
    runner._context_length = window
    return runner


class TestNoteFinish:
    """`_note_finish`: the JSON field via the wire mapping, the notice only on a cut."""

    @pytest.mark.parametrize(
        "reason, expected",
        [("stop", "stop"), ("length", "length"), ("interrupted", "stop"), (None, None)],
    )
    def test_writes_wire_mapped_finish_reason(self, reason, expected):
        info = {}
        _note_finish(_finished_runner(reason=reason), info, None, json_output=True)
        assert info["finish_reason"] == expected

    def test_window_bound_notice(self, capsys):
        # Budget below the default ceiling means the window, not the ceiling, cut it
        _note_finish(_finished_runner(generated=100, prompt=50, budget=100, window=150), {}, None, json_output=False)
        err = capsys.readouterr().err
        assert "[Output cut at 100 tokens: prompt (50) and output fill the model's context window (150)." in err
        assert "Shorten the prompt" in err

    def test_window_bound_wins_over_explicit_max_tokens(self, capsys):
        # An explicit --max-tokens clamped by the window still points at the window
        _note_finish(_finished_runner(budget=100, window=150), {}, 500, json_output=False)
        err = capsys.readouterr().err
        assert "context window (150)" in err
        assert "--max-tokens" not in err

    def test_explicit_max_tokens_notice(self, capsys):
        _note_finish(_finished_runner(generated=100, budget=100), {}, 100, json_output=False)
        err = capsys.readouterr().err
        assert "[Output cut at 100 tokens (--max-tokens). Pass a larger --max-tokens to allow more.]" in err

    def test_default_ceiling_notice(self, capsys):
        runner = _finished_runner(generated=DEFAULT_MAX_TOKENS, budget=DEFAULT_MAX_TOKENS, window=None)
        _note_finish(runner, {}, None, json_output=False)
        err = capsys.readouterr().err
        assert f"[Output cut at {DEFAULT_MAX_TOKENS} tokens, the default max_tokens. Pass --max-tokens to allow more.]" in err

    def test_json_mode_prints_nothing_but_records_the_field(self, capsys):
        info = {}
        _note_finish(_finished_runner(reason="length"), info, None, json_output=True)
        assert info["finish_reason"] == "length"
        assert capsys.readouterr().err == ""

    def test_stop_prints_nothing(self, capsys):
        info = {}
        _note_finish(_finished_runner(reason="stop"), info, None, json_output=False)
        assert info["finish_reason"] == "stop"
        assert capsys.readouterr().err == ""

    def test_runner_without_attributes(self, capsys):
        # No generation ran (or a bare mock): no crash, nothing invented, nothing printed
        info = {}
        _note_finish(SimpleNamespace(), info, None, json_output=False)
        assert info["finish_reason"] is None

        info = {}
        _note_finish(Mock(), info, None, json_output=False)
        assert info["finish_reason"] is None
        assert capsys.readouterr().err == ""

    def test_no_result_info(self, capsys):
        _note_finish(_finished_runner(reason="length"), None, None, json_output=False)
        assert "Output cut" in capsys.readouterr().err


class TestSingleShotGeneration:
    """Both single-shot paths report the cut."""

    def test_streaming_notes_finish(self, capsys):
        runner = _finished_runner(reason="length")
        runner.generate_streaming.return_value = iter(["Hel", "lo"])
        info = {}

        single_shot_generation(runner, "prompt", stream=True, json_output=False, result_info=info)

        out, err = capsys.readouterr()
        assert "Hello" in out
        assert "Output cut" in err
        assert info["finish_reason"] == "length"

    def test_batch_notes_finish(self, capsys):
        runner = _finished_runner(reason="length")
        runner.generate_batch.return_value = "Hello"
        info = {}

        single_shot_generation(runner, "prompt", stream=False, json_output=False, result_info=info)

        out, err = capsys.readouterr()
        assert "Hello" in out
        assert "Output cut" in err
        assert info["finish_reason"] == "length"

    def test_json_batch_records_without_notice(self, capsys):
        runner = _finished_runner(reason="length")
        runner.generate_batch.return_value = "Hello"
        info = {}

        result = single_shot_generation(runner, "prompt", stream=False, json_output=True, result_info=info)

        assert result == "Hello"
        assert info["finish_reason"] == "length"
        assert capsys.readouterr().err == ""


class TestInteractiveChat:
    """The notice follows each cut answer in chat mode."""

    @pytest.mark.parametrize("stream", [True, False])
    def test_notice_after_answer(self, stream, capsys):
        runner = _finished_runner(reason="length")
        runner._format_conversation.return_value = "formatted"
        runner.generate_streaming.return_value = iter(["Hel", "lo"])
        runner.generate_batch.return_value = "Hello"

        with patch("builtins.input", side_effect=["hello", "quit"]):
            interactive_chat(runner, stream=stream, max_tokens=None)

        out, err = capsys.readouterr()
        assert "Hello" in out
        assert "[ERROR]" not in err
        assert "Output cut" in err


class TestContextLengthReject:
    """`run_model` turns the pre-execution reject into the typed error."""

    @pytest.fixture
    def mock_runner(self):
        with patch("mlxk2.operations.run.MLXRunner") as mock_runner_class:
            runner = Mock()
            mock_runner_class.return_value.__enter__.return_value = runner
            mock_runner_class.return_value.__exit__.return_value = None
            yield runner

    def test_batch_reject_json(self, mock_runner, capsys):
        mock_runner.generate_batch.side_effect = ContextLengthExceeded(5000, 4096)
        info = {}

        result = run_model("test-model", prompt="long prompt", stream=False, json_output=True, result_info=info)

        assert result.startswith("Error: Prompt is 5000 tokens")
        assert "4096" in result
        assert info["error_type"] == "context_length_exceeded"
        assert "finish_reason" not in info
        assert capsys.readouterr().err == ""

    def test_streaming_reject_text(self, mock_runner, capsys):
        mock_runner.generate_streaming.side_effect = ContextLengthExceeded(5000, 4096)
        info = {}

        result = run_model("test-model", prompt="long prompt", stream=True, json_output=False, result_info=info)

        assert result.startswith("Error: Prompt is 5000 tokens")
        assert info["error_type"] == "context_length_exceeded"
        assert "Error: Prompt is 5000 tokens" in capsys.readouterr().err

    def test_other_errors_stay_untyped(self, mock_runner):
        mock_runner.generate_batch.side_effect = RuntimeError("boom")
        info = {}

        result = run_model("test-model", prompt="p", stream=False, json_output=True, result_info=info)

        assert result == "Error: boom"
        assert "error_type" not in info


def _run_cli(argv, capsys):
    """Run the CLI like test_cli_run_exit_codes does: (stdout, stderr, exit_code)."""
    from mlxk2.cli import main as cli_main

    old_argv = sys.argv[:]
    sys.argv = argv[:]
    try:
        cli_main()
        exit_code = 0
    except SystemExit as e:
        exit_code = e.code
    finally:
        sys.argv = old_argv
    captured = capsys.readouterr()
    return captured.out, captured.err, exit_code


class TestCliJsonEnvelope:
    """`run --json`: `data.finish_reason` on success, `error.type` on a reject."""

    def test_success_envelope_carries_finish_reason(self, capsys):
        def fake_run(**kwargs):
            kwargs["result_info"]["finish_reason"] = "length"
            return "1\n2\n3"

        with patch("mlxk2.cli.run_model_enhanced", side_effect=fake_run):
            out, err, code = _run_cli(["mlxk2", "run", "test-model", "count", "--json"], capsys)

        assert code == 0
        data = json.loads(out)
        assert data["status"] == "success"
        assert data["command"] == "run"
        assert data["error"] is None
        assert data["data"]["response"] == "1\n2\n3"
        assert data["data"]["finish_reason"] == "length"

    def test_success_envelope_null_when_unknown(self, capsys):
        with patch("mlxk2.cli.run_model_enhanced", return_value="text"):
            out, err, code = _run_cli(["mlxk2", "run", "test-model", "hello", "--json"], capsys)

        assert code == 0
        data = json.loads(out)
        assert "finish_reason" in data["data"]
        assert data["data"]["finish_reason"] is None

    def test_error_envelope_typed_reject(self, capsys):
        def fake_reject(**kwargs):
            kwargs["result_info"]["error_type"] = "context_length_exceeded"
            return "Error: Prompt is 5000 tokens, but the model's context window is 4096 tokens"

        with patch("mlxk2.cli.run_model_enhanced", side_effect=fake_reject):
            out, err, code = _run_cli(["mlxk2", "run", "test-model", "hello", "--json"], capsys)

        assert code == 1
        data = json.loads(out)
        assert data["status"] == "error"
        assert data["data"] is None
        assert data["error"]["type"] == "context_length_exceeded"
        assert data["error"]["message"].startswith("Prompt is 5000 tokens")

    def test_error_envelope_default_type(self, capsys):
        with patch("mlxk2.cli.run_model_enhanced", return_value="Error: Model loading failed"):
            out, err, code = _run_cli(["mlxk2", "run", "test-model", "hello", "--json"], capsys)

        assert code == 1
        data = json.loads(out)
        assert data["status"] == "error"
        assert data["error"]["type"] == "execution_error"
        assert data["error"]["message"] == "Model loading failed"
