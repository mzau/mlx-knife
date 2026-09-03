"""
Token budget tests: G = min(X, W − P) on both generation paths, and the stop reason.

Covers mlxk2/core/runner/token_limits.py (window reader, budget rule, wire mapping)
and how the runner applies it (generate_streaming / generate_batch / generation_budget,
the last_* attributes at every exit).
"""

import json
from contextlib import contextmanager
from unittest.mock import Mock, patch

import pytest

from mlxk2.core.runner import MLXRunner
from mlxk2.core.runner.token_limits import (
    DEFAULT_MAX_TOKENS,
    DEFAULT_MAX_TOKENS_VISION,
    ContextLengthExceeded,
    get_model_context_length,
    reported_finish_reason,
    resolve_generation_budget,
)

CONTEXT_KEYS = (
    "max_position_embeddings",
    "n_positions",
    "context_length",
    "max_sequence_length",
    "seq_len",
)


def _write_config(tmp_path, payload):
    """Write config.json (dict → JSON, str → verbatim) and return the model dir."""
    text = payload if isinstance(payload, str) else json.dumps(payload)
    (tmp_path / "config.json").write_text(text)
    return str(tmp_path)


class TestContextLengthDetection:
    """get_model_context_length: the window from config.json, None when it is not stated."""

    @pytest.mark.parametrize("key", CONTEXT_KEYS)
    def test_top_level_key(self, tmp_path, key):
        assert get_model_context_length(_write_config(tmp_path, {key: 8192})) == 8192

    @pytest.mark.parametrize("key", CONTEXT_KEYS)
    def test_text_config_key(self, tmp_path, key):
        """Multimodal configs carry the text window inside text_config."""
        config = {"text_config": {key: 4096}}
        assert get_model_context_length(_write_config(tmp_path, config)) == 4096

    def test_top_level_wins_over_text_config(self, tmp_path):
        config = {"max_position_embeddings": 8192, "text_config": {"max_position_embeddings": 4096}}
        assert get_model_context_length(_write_config(tmp_path, config)) == 8192

    def test_string_digits(self, tmp_path):
        config = {"max_position_embeddings": "8192"}
        assert get_model_context_length(_write_config(tmp_path, config)) == 8192

    def test_missing_file(self, tmp_path):
        assert get_model_context_length(str(tmp_path)) is None

    def test_invalid_json(self, tmp_path):
        assert get_model_context_length(_write_config(tmp_path, "{not json")) is None

    def test_no_matching_key(self, tmp_path):
        assert get_model_context_length(_write_config(tmp_path, {"hidden_size": 4096})) is None

    def test_config_not_an_object(self, tmp_path):
        assert get_model_context_length(_write_config(tmp_path, [8192])) is None

    @pytest.mark.parametrize("value", [0, -1, "0", "-1", "8k", True, False, None])
    def test_unusable_values(self, tmp_path, value):
        """Only a positive integer (or its digit string) is a window; a bool is not an int here."""
        config = {"max_position_embeddings": value}
        assert get_model_context_length(_write_config(tmp_path, config)) is None

    def test_unusable_value_does_not_shadow_a_usable_one(self, tmp_path):
        config = {"max_position_embeddings": 0, "text_config": {"n_positions": 2048}}
        assert get_model_context_length(_write_config(tmp_path, config)) == 2048

    @pytest.mark.parametrize("text_config", [4096, "4096", [4096], None])
    def test_text_config_not_a_dict(self, tmp_path, text_config):
        config = {"text_config": text_config}
        assert get_model_context_length(_write_config(tmp_path, config)) is None


class TestResolveGenerationBudget:
    """resolve_generation_budget: min(X, W − P), reject on a full window, reject below 1."""

    def test_documented_ceilings(self):
        assert DEFAULT_MAX_TOKENS == 32768
        assert DEFAULT_MAX_TOKENS_VISION == 2048

    def test_default_ceiling_with_unknown_window(self):
        assert resolve_generation_budget(None, None, 3) == 32768

    @pytest.mark.parametrize("window", [0, -1])
    def test_non_positive_window_is_unknown(self, window):
        assert resolve_generation_budget(None, window, 3) == 32768
        assert resolve_generation_budget(500, window, 3) == 500

    def test_explicit_ceiling(self):
        assert resolve_generation_budget(500, None, 3) == 500
        assert resolve_generation_budget(500, 8192, 3) == 500
        # No window, no guard: the caller's number stands as given.
        assert resolve_generation_budget(50000, None, 3) == 50000

    def test_default_clamped_to_window(self):
        assert resolve_generation_budget(None, 8192, 3) == 8189
        assert resolve_generation_budget(None, 100000, 3) == 32768

    def test_explicit_clamped_to_window(self):
        assert resolve_generation_budget(100000, 8192, 3) == 8189
        assert resolve_generation_budget(8192, 8192, 3) == 8189

    def test_one_token_of_room(self):
        assert resolve_generation_budget(None, 4, 3) == 1

    @pytest.mark.parametrize("prompt_tokens", [3, 4])
    def test_full_window_rejects(self, prompt_tokens):
        with pytest.raises(ContextLengthExceeded) as exc:
            resolve_generation_budget(None, 3, prompt_tokens)
        assert exc.value.prompt_tokens == prompt_tokens
        assert exc.value.context_length == 3
        assert str(exc.value) == (
            f"Prompt is {prompt_tokens} tokens, but the model's context window is 3 tokens; "
            "nothing is left to generate. Shorten the prompt."
        )

    def test_explicit_ceiling_does_not_bypass_reject(self):
        with pytest.raises(ContextLengthExceeded):
            resolve_generation_budget(1, 3, 3)

    def test_reject_is_a_value_error(self):
        """Callers that only know ValueError still catch it."""
        assert issubclass(ContextLengthExceeded, ValueError)

    @pytest.mark.parametrize("requested", [0, -1])
    def test_requested_below_one(self, requested):
        with pytest.raises(ValueError, match=rf"max_tokens must be at least 1 \(got {requested}\)") as exc:
            resolve_generation_budget(requested, 8192, 3)
        assert not isinstance(exc.value, ContextLengthExceeded)


class TestReportedFinishReason:
    """reported_finish_reason: the runner's reason as every surface reports it."""

    @pytest.mark.parametrize("reason, wire", [
        ("stop", "stop"),
        ("length", "length"),
        ("interrupted", "stop"),
        (None, None),
        ("garbage", None),
        ("", None),
    ])
    def test_mapping(self, reason, wire):
        assert reported_finish_reason(reason) == wire


class MockDetokenizer:
    """Mimics the streaming detokenizer the runner decodes through (shape as in test_runner_core)."""

    def __init__(self, decode_func):
        self.decode_func = decode_func
        self.tokens = []
        self._text = ""

    def reset(self):
        self.tokens = []
        self._text = ""

    def add_token(self, token_id):
        self.tokens.append(token_id)

    def finalize(self):
        self._text = self.decode_func(self.tokens)

    @property
    def text(self):
        return self._text


def _bracket_decode(tokens):
    """One "[id]" per token: prefix-stable, so the streaming diff yields exactly one piece per token."""
    return "".join(f"[{t}]" for t in tokens)


def _stop_string_decode(stop_string, at_id=9):
    """Like _bracket_decode, but one id decodes to a stop string (it is never an EOS id)."""
    return lambda tokens: "".join(stop_string if t == at_id else f"[{t}]" for t in tokens)


def _steps(ids):
    """generate_step output: (token, logprobs) pairs, plain ints as the neighbours use."""
    return iter([(tid, 0) for tid in ids])


@contextmanager
def _runner_env(context_length=8192, prompt_ids=(1, 2, 3), decode=_bracket_decode):
    """The runner's patch seams as the neighbours patch them; window and prompt are the knobs."""
    with patch('mlxk2.core.runner.load') as mock_load, \
         patch('mlxk2.core.runner.resolve_model_for_operation') as mock_resolve, \
         patch('mlxk2.core.runner.get_current_model_cache') as mock_cache, \
         patch('mlxk2.core.runner.get_model_context_length') as mock_context, \
         patch('mlxk2.core.runner.generate_step') as mock_gen:
        mock_resolve.return_value = ("test-model", None, None)
        mock_cache.return_value = Mock()
        mock_context.return_value = context_length

        mock_tokenizer = Mock()
        mock_tokenizer.eos_token = "</s>"
        mock_tokenizer.eos_token_id = 2
        mock_tokenizer.eos_token_ids = {mock_tokenizer.eos_token_id}
        mock_tokenizer.pad_token = None
        mock_tokenizer.additional_special_tokens = []
        mock_tokenizer.added_tokens_decoder = {}
        mock_tokenizer.chat_template = None
        mock_tokenizer.name_or_path = "mock-test-model"
        mock_tokenizer.encode.return_value = list(prompt_ids)
        mock_tokenizer.detokenizer = MockDetokenizer(decode)
        mock_load.return_value = (Mock(), mock_tokenizer)
        mock_gen.return_value = iter([])  # nothing generated unless a test says otherwise

        yield {'mock_gen': mock_gen, 'mock_tokenizer': mock_tokenizer}


PATHS = ("streaming", "batch")


def _generate(runner, path, **kwargs):
    """Drive one generation path to its end and return the text it produced."""
    if path == "streaming":
        return "".join(runner.generate_streaming("test", **kwargs))
    return runner.generate_batch("test", **kwargs)


class TestRunnerBudget:
    """Both generation paths hand generate_step min(32768, W − P); generation_budget says the same."""

    @pytest.mark.parametrize("path", PATHS)
    def test_default_is_window_minus_prompt(self, path):
        with _runner_env(context_length=8192) as env, MLXRunner("test-model") as runner:
            _generate(runner, path, max_tokens=None)
            assert env['mock_gen'].call_args.kwargs['max_tokens'] == 8189
            assert runner.last_max_tokens == 8189
            assert runner.last_prompt_tokens == 3

    @pytest.mark.parametrize("path", PATHS)
    def test_default_is_capped_at_32768(self, path):
        with _runner_env(context_length=100000) as env, MLXRunner("test-model") as runner:
            _generate(runner, path, max_tokens=None)
            assert env['mock_gen'].call_args.kwargs['max_tokens'] == 32768

    @pytest.mark.parametrize("path", PATHS)
    def test_unknown_window_uses_the_ceiling_only(self, path):
        with _runner_env(context_length=None) as env, MLXRunner("test-model") as runner:
            _generate(runner, path, max_tokens=None)
            assert env['mock_gen'].call_args.kwargs['max_tokens'] == 32768
            env['mock_gen'].return_value = iter([])
            _generate(runner, path, max_tokens=50000)
            assert env['mock_gen'].call_args.kwargs['max_tokens'] == 50000

    @pytest.mark.parametrize("path", PATHS)
    def test_explicit_is_clamped_by_the_window(self, path):
        with _runner_env(context_length=8192) as env, MLXRunner("test-model") as runner:
            _generate(runner, path, max_tokens=500)
            assert env['mock_gen'].call_args.kwargs['max_tokens'] == 500
            env['mock_gen'].return_value = iter([])
            _generate(runner, path, max_tokens=10000)
            assert env['mock_gen'].call_args.kwargs['max_tokens'] == 8189
            assert runner.last_max_tokens == 8189

    @pytest.mark.parametrize("path", PATHS)
    def test_full_window_rejects_before_generate_step(self, path):
        with _runner_env(context_length=3) as env, MLXRunner("test-model") as runner:
            with pytest.raises(ContextLengthExceeded) as exc:
                _generate(runner, path, max_tokens=None)
            assert (exc.value.prompt_tokens, exc.value.context_length) == (3, 3)
            # An explicit ceiling does not open the window either.
            with pytest.raises(ContextLengthExceeded):
                _generate(runner, path, max_tokens=1)
            env['mock_gen'].assert_not_called()
            assert runner.last_finish_reason is None
            assert runner.last_max_tokens is None

    @pytest.mark.parametrize("path", PATHS)
    def test_requested_below_one_rejects_before_generate_step(self, path):
        with _runner_env() as env, MLXRunner("test-model") as runner:
            with pytest.raises(ValueError, match=r"max_tokens must be at least 1 \(got 0\)"):
                _generate(runner, path, max_tokens=0)
            env['mock_gen'].assert_not_called()

    def test_generation_budget_matches_the_generation_paths(self):
        with _runner_env(context_length=8192) as env, MLXRunner("test-model") as runner:
            assert runner.generation_budget("test") == 8189
            assert runner.generation_budget("test", max_tokens=500) == 500
            assert runner.generation_budget("test", max_tokens=10000) == 8189
            list(runner.generate_streaming("test"))
            assert env['mock_gen'].call_args.kwargs['max_tokens'] == runner.generation_budget("test")
            env['mock_gen'].assert_called_once()  # generation_budget itself never generates

    def test_generation_budget_rejects_like_the_generation_paths(self):
        with _runner_env(context_length=3) as env, MLXRunner("test-model") as runner:
            with pytest.raises(ContextLengthExceeded) as exc:
                runner.generation_budget("test")
            assert (exc.value.prompt_tokens, exc.value.context_length) == (3, 3)
            with pytest.raises(ValueError, match=r"max_tokens must be at least 1 \(got 0\)"):
                runner.generation_budget("test", max_tokens=0)
            env['mock_gen'].assert_not_called()

    def test_generation_budget_with_unknown_window(self):
        with _runner_env(context_length=None), MLXRunner("test-model") as runner:
            assert runner.generation_budget("test") == 32768

    def test_generation_budget_needs_a_loaded_model(self):
        with pytest.raises(RuntimeError, match="Model not loaded"):
            MLXRunner("test-model").generation_budget("test")


class TestRunnerFinishReason:
    """last_finish_reason and the last_* counts name every exit of both generation paths."""

    def test_nothing_before_the_first_generation(self):
        with _runner_env(), MLXRunner("test-model") as runner:
            assert runner.last_finish_reason is None
            assert runner.last_prompt_tokens is None
            assert runner.last_completion_tokens is None
            assert runner.last_max_tokens is None

    @pytest.mark.parametrize("path", PATHS)
    def test_eos_id_is_stop(self, path):
        with _runner_env() as env, MLXRunner("test-model") as runner:
            env['mock_gen'].return_value = _steps([10, 11, 2])  # 2 is the EOS id
            _generate(runner, path)
            assert runner.last_finish_reason == "stop"
            assert runner.last_completion_tokens == 3  # the EOS id itself is counted
            assert runner.last_prompt_tokens == 3
            assert runner.last_max_tokens == 8189

    @pytest.mark.parametrize("path", PATHS)
    def test_explicit_budget_exhausted_is_length(self, path):
        with _runner_env() as env, MLXRunner("test-model") as runner:
            env['mock_gen'].return_value = _steps(range(100, 105))  # exactly max_tokens, no EOS
            _generate(runner, path, max_tokens=5)
            assert runner.last_finish_reason == "length"
            assert runner.last_completion_tokens == 5
            assert runner.last_max_tokens == 5

    @pytest.mark.parametrize("path", PATHS)
    def test_window_bound_budget_exhausted_is_length(self, path):
        with _runner_env(context_length=10) as env, MLXRunner("test-model") as runner:
            env['mock_gen'].return_value = _steps(range(100, 107))  # W − P = 7 tokens, no EOS
            _generate(runner, path)
            assert runner.last_finish_reason == "length"
            assert runner.last_completion_tokens == 7
            assert runner.last_max_tokens == 7

    @pytest.mark.parametrize("path", PATHS)
    def test_generator_ending_early_is_none(self, path):
        """Neither stop nor budget nor interrupt (a mock, a closed generator): no reason is invented."""
        with _runner_env() as env, MLXRunner("test-model") as runner:
            env['mock_gen'].return_value = _steps([10, 11])
            _generate(runner, path, max_tokens=5)
            assert runner.last_finish_reason is None
            assert runner.last_completion_tokens == 2

    @pytest.mark.parametrize("path", PATHS)
    def test_empty_generator_is_none_and_resets_the_previous_reason(self, path):
        with _runner_env() as env, MLXRunner("test-model") as runner:
            env['mock_gen'].return_value = _steps([10, 2])
            _generate(runner, path)
            assert runner.last_finish_reason == "stop"
            env['mock_gen'].return_value = iter([])
            _generate(runner, path)
            assert runner.last_finish_reason is None
            assert runner.last_completion_tokens == 0
            assert runner.last_max_tokens == 8189

    @pytest.mark.parametrize("path", PATHS)
    def test_interrupt_mid_generation_is_interrupted(self, path):
        with _runner_env() as env, MLXRunner("test-model") as runner:
            def steps():
                yield (10, 0)
                runner._interrupted = True  # Ctrl-C / request_interrupt lands between two tokens
                yield (11, 0)
                yield (12, 0)

            env['mock_gen'].return_value = steps()
            out = _generate(runner, path)
            assert runner.last_finish_reason == "interrupted"
            assert runner.last_completion_tokens == 1
            assert runner.last_max_tokens == 8189
            if path == "streaming":
                assert out == "[10]\n[Generation interrupted by user]"

    def test_interrupt_between_streamed_tokens(self):
        with _runner_env() as env, MLXRunner("test-model") as runner:
            env['mock_gen'].return_value = _steps([10, 11, 12])
            stream = runner.generate_streaming("test")
            assert next(stream) == "[10]"
            runner._interrupted = True
            assert list(stream) == ["\n[Generation interrupted by user]"]
            assert runner.last_finish_reason == "interrupted"
            assert runner.last_completion_tokens == 1

    @pytest.mark.parametrize("path", PATHS)
    def test_stale_interrupt_flag_is_cleared_at_start(self, path):
        """A flag left over from an earlier Ctrl-C is not this generation's exit."""
        with _runner_env() as env, MLXRunner("test-model") as runner:
            env['mock_gen'].return_value = _steps([10, 2])
            runner._interrupted = True
            _generate(runner, path)
            assert runner.last_finish_reason == "stop"

    def test_batch_stop_string_is_stop(self):
        """Batch decodes once at the end; a stop string in that text ends the turn like an EOS id."""
        with _runner_env(decode=_stop_string_decode("</s>")) as env, MLXRunner("test-model") as runner:
            env['mock_gen'].return_value = _steps([10, 9, 11])
            assert runner.generate_batch("test") == "[10]"
            assert runner.last_finish_reason == "stop"
            assert runner.last_completion_tokens == 3  # all three ran; the cut is in the text

    def test_batch_chat_stop_token_is_stop_only_when_enabled(self):
        with _runner_env(decode=_stop_string_decode("\nHuman:")) as env, MLXRunner("test-model") as runner:
            env['mock_gen'].return_value = _steps([10, 9, 11])
            assert runner.generate_batch("test", use_chat_stop_tokens=True) == "[10]"
            assert runner.last_finish_reason == "stop"
            env['mock_gen'].return_value = _steps([10, 9, 11])
            assert runner.generate_batch("test") == "[10]\nHuman:[11]"
            assert runner.last_finish_reason is None

    def test_streaming_stop_string_is_stop(self):
        with _runner_env(decode=_stop_string_decode("</s>")) as env, MLXRunner("test-model") as runner:
            env['mock_gen'].return_value = _steps([10, 9, 11])
            assert "".join(runner.generate_streaming("test")) == "[10]"
            assert runner.last_finish_reason == "stop"
            assert runner.last_completion_tokens == 2  # stopped on the token carrying the string; 11 never ran
