"""`usage` reports what the runner counted, not a word estimate.

Every surface used to report `len(text.split()) * 1.3`: a reply the ceiling cut at five
tokens came back as `completion_tokens: 2`, contradicting the `Generation finished` line
logged beside it, which has always used the runner's real numbers.
"""

from unittest.mock import Mock, patch

from fastapi.testclient import TestClient

from mlxk2.core.server.streaming import usage_of
from mlxk2.core.server_base import app


def _estimate(text):
    return 99


class CountingRunner:
    """A runner that records what it encoded and generated, as the real ones do."""

    last_prompt_tokens = 40
    last_completion_tokens = 5
    last_finish_reason = "length"
    last_max_tokens = 5

    def _format_conversation(self, messages):
        return "prompt"

    def generate_batch(self, **kwargs):
        return "1, 2,"


def test_the_runners_own_numbers_win():
    assert usage_of(CountingRunner(), "p", "g", _estimate) == {
        "prompt_tokens": 40,
        "completion_tokens": 5,
        "total_tokens": 45,
    }


def test_a_runner_without_counts_falls_back_to_the_estimate():
    runner = CountingRunner()
    runner.last_prompt_tokens = None
    runner.last_completion_tokens = None
    assert usage_of(runner, "p", "g", _estimate)["total_tokens"] == 198


def test_a_test_double_does_not_pass_as_a_count():
    """`getattr` on a Mock answers with another Mock — presence is not a number."""
    assert usage_of(Mock(), "p", "g", _estimate)["prompt_tokens"] == 99


def test_a_bool_does_not_pass_as_a_count():
    runner = CountingRunner()
    runner.last_prompt_tokens = True
    assert usage_of(runner, "p", "g", _estimate)["prompt_tokens"] == 99


def test_the_chat_response_carries_them():
    payload = {"model": "org/model", "messages": [{"role": "user", "content": "Hi"}]}
    with patch("mlxk2.core.server_base.get_or_load_model", return_value=CountingRunner()):
        body = TestClient(app).post("/v1/chat/completions", json=payload).json()
    assert body["usage"] == {"prompt_tokens": 40, "completion_tokens": 5, "total_tokens": 45}
