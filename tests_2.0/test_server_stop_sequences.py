"""`stop` reaches the batch surfaces (audit C3a).

`generate_batch` has no `stop` parameter, so the field was accepted by the request model
and then dropped: a client got the full answer and `finish_reason` from the runner. The
sequences are now applied to the finished text — the answer ends where OpenAI says it
ends. Streams keep their own, weaker handling (C3b).
"""

import asyncio
import json
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from mlxk2.core.server.streaming import apply_stop_sequences
from mlxk2.core.server_base import app

ANSWER = "one two STOP three"


class Runner:
    last_prompt_tokens = 3
    last_completion_tokens = 6
    last_finish_reason = "length"
    last_max_tokens = 6

    def _format_conversation(self, messages):
        return "prompt"

    def generate_batch(self, **kwargs):
        return ANSWER


@pytest.mark.parametrize("stop", [None, [], ["absent"]])
def test_without_a_match_the_text_stands(stop):
    assert apply_stop_sequences(ANSWER, stop) == (ANSWER, False)


def test_the_earliest_sequence_wins():
    assert apply_stop_sequences(ANSWER, ["three", "STOP"]) == ("one two ", True)


def test_the_sequence_itself_is_removed():
    text, stopped = apply_stop_sequences(ANSWER, ["STOP"])
    assert stopped is True
    assert "STOP" not in text


SURFACES = {
    "chat": ("/v1/chat/completions", {"messages": [{"role": "user", "content": "Hi"}]}),
    "completions": ("/v1/completions", {"prompt": "Hi"}),
}


@pytest.mark.parametrize("surface", sorted(SURFACES))
def test_the_batch_answer_is_cut_and_says_stop(surface):
    path, body = SURFACES[surface]
    payload = {"model": "org/model", "stop": ["STOP"], **body}
    with patch("mlxk2.core.server_base.get_or_load_model", return_value=Runner()):
        response = TestClient(app).post(path, json=payload).json()
    choice = response["choices"][0]
    text = choice.get("text") or choice["message"]["content"]
    assert text == "one two "
    # The runner reported "length"; the stop sequence outranks it.
    assert choice["finish_reason"] == "stop"
    # The cut tokens were still generated, so the count is the runner's, not the text's.
    assert response["usage"]["completion_tokens"] == 6


@pytest.mark.parametrize("surface", sorted(SURFACES))
def test_without_stop_the_runners_reason_stands(surface):
    path, body = SURFACES[surface]
    with patch("mlxk2.core.server_base.get_or_load_model", return_value=Runner()):
        response = TestClient(app).post(path, json={"model": "org/model", **body}).json()
    choice = response["choices"][0]
    assert (choice.get("text") or choice["message"]["content"]) == ANSWER
    assert choice["finish_reason"] == "length"


class StreamingRunner(Runner):
    """Emits the answer token by token, so the stop check runs where it really runs."""

    def generate_streaming(self, **kwargs):
        yield from ["one ", "two ", "STOP", " three"]


@pytest.mark.parametrize("surface", sorted(SURFACES))
def test_a_stream_that_stops_on_a_sequence_says_so(surface):
    """Breaking out of the loop leaves the runner without an exit — the sequence is the reason."""
    path, body = SURFACES[surface]
    payload = {"model": "org/model", "stream": True, "stop": ["STOP"], **body}
    with patch("mlxk2.core.server_base.get_or_load_model", return_value=StreamingRunner()):
        with TestClient(app).stream("POST", path, json=payload) as response:
            lines = [line for line in response.iter_lines() if line.startswith("data: ")]
    reasons = [
        json.loads(line[6:])["choices"][0]["finish_reason"]
        for line in lines
        if line != "data: [DONE]"
    ]
    assert reasons[-1] == "stop"


@pytest.mark.parametrize("surface", sorted(SURFACES))
def test_a_stream_without_a_match_keeps_the_runners_reason(surface):
    path, body = SURFACES[surface]
    payload = {"model": "org/model", "stream": True, **body}
    with patch("mlxk2.core.server_base.get_or_load_model", return_value=StreamingRunner()):
        with TestClient(app).stream("POST", path, json=payload) as response:
            lines = [line for line in response.iter_lines() if line.startswith("data: ")]
    reasons = [
        json.loads(line[6:])["choices"][0]["finish_reason"]
        for line in lines
        if line != "data: [DONE]"
    ]
    assert reasons[-1] == "length"


def test_the_vision_surface_is_handed_the_sequences():
    """It had no `stop` parameter at all — a vision batch answer ignored the field."""
    from unittest.mock import AsyncMock

    from mlxk2.core.server_base import ChatCompletionRequest, _handle_vision_chat_completion

    response = {
        "id": "chatcmpl-x", "created": 0, "model": "m",
        "choices": [{"index": 0, "message": {"role": "assistant", "content": "ok"},
                     "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
    }
    request = ChatCompletionRequest(
        model="m", messages=[{"role": "user", "content": "hi"}], stop="STOP"
    )
    with patch("mlxk2.core.server_base._handle_vision_chat_completion_impl",
               new=AsyncMock(return_value=response)) as impl:
        asyncio.run(_handle_vision_chat_completion(request))
    assert impl.await_args.kwargs["stop"] == ["STOP"]
