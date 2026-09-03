"""
How a generation ended, as the server reports it (#66).

One source (the runner's last_finish_reason), reported on every text surface;
a full context window is a 400 before any token, never an SSE event; and
max_tokens below 1 never reaches a runner.
"""

import asyncio
import json
from typing import Iterator, List
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from mlxk2.core.runner.token_limits import ContextLengthExceeded
from mlxk2.core.server.handlers.chat import process_vision_chunks_server
from mlxk2.core.server.streaming import emulate_sse_stream, log_generation_end
from mlxk2.core.server_base import app

_ABSENT = object()


class Runner:
    """The runner surface the text endpoints touch; `reason` is how its last generation ended."""

    def __init__(self, reason=_ABSENT, budget=_ABSENT):
        if reason is not _ABSENT:
            self.last_finish_reason = reason
        if budget is not _ABSENT:
            self.generation_budget = budget
        self.seen = {}

    def _format_conversation(self, messages):
        return "prompt"

    def generate_batch(self, **kwargs):
        self.seen.update(kwargs)
        return "ok"

    def generate_streaming(self, **kwargs):
        self.seen.update(kwargs)
        yield "A"
        yield "B"


# surface -> (path, request body, stream flag)
SURFACES = {
    "chat-batch": ("/v1/chat/completions", {"messages": [{"role": "user", "content": "Hi"}]}, False),
    "chat-stream": ("/v1/chat/completions", {"messages": [{"role": "user", "content": "Hi"}]}, True),
    "completions-batch": ("/v1/completions", {"prompt": "Hi"}, False),
    "completions-stream": ("/v1/completions", {"prompt": "Hi"}, True),
}
ENDPOINTS = {
    "chat": ("/v1/chat/completions", {"messages": [{"role": "user", "content": "Hi"}]}),
    "completions": ("/v1/completions", {"prompt": "Hi"}),
}


def _iter_sse_lines(resp) -> Iterator[str]:
    """Iterate non-empty SSE lines as strings from a streaming response."""
    for raw in resp.iter_lines():
        if not raw:
            continue
        line = raw.decode("utf-8", errors="ignore") if isinstance(raw, bytes) else raw
        if line.strip():
            yield line


def _sse_events(resp) -> List[dict]:
    """Parse SSE events into dicts (skips [DONE])."""
    events = []
    for line in _iter_sse_lines(resp):
        if line.startswith("data: ") and line.strip() != "data: [DONE]":
            events.append(json.loads(line[len("data: "):]))
    return events


def _finish_reason_on_wire(surface: str, runner) -> object:
    """POST on one surface; return the finish_reason of the batch choice or the final SSE chunk."""
    path, body, stream = SURFACES[surface]
    payload = {"model": "org/model", "stream": stream, **body}
    client = TestClient(app)
    with patch('mlxk2.core.server_base.get_or_load_model', return_value=runner):
        if stream:
            with client.stream("POST", path, json=payload) as resp:
                assert resp.status_code == 200
                events = _sse_events(resp)
            return events[-1]["choices"][0]["finish_reason"]
        resp = client.post(path, json=payload)
        assert resp.status_code == 200
        return resp.json()["choices"][0]["finish_reason"]


# --- finish_reason on the wire -------------------------------------------------

@pytest.mark.parametrize("surface", sorted(SURFACES))
@pytest.mark.parametrize("reason,expected", [
    ("length", "length"),
    ("stop", "stop"),
    ("interrupted", "stop"),  # what the interrupt marker has always carried
])
def test_finish_reason_reported(surface, reason, expected):
    assert _finish_reason_on_wire(surface, Runner(reason=reason)) == expected


@pytest.mark.parametrize("surface", sorted(SURFACES))
def test_runner_without_attribute_reports_null(surface):
    """Absent, not invented: nothing recorded means null on the wire."""
    assert _finish_reason_on_wire(surface, Runner()) is None


@pytest.mark.parametrize("surface", ["chat-stream", "completions-stream"])
def test_stream_only_final_chunk_carries_finish_reason(surface):
    path, body, stream = SURFACES[surface]
    client = TestClient(app)
    with patch('mlxk2.core.server_base.get_or_load_model', return_value=Runner(reason="length")):
        with client.stream("POST", path, json={"model": "org/model", "stream": True, **body}) as resp:
            events = _sse_events(resp)
    reasons = [e["choices"][0]["finish_reason"] for e in events]
    assert reasons[:-1] == [None] * (len(events) - 1)
    assert reasons[-1] == "length"


# --- ContextLengthExceeded: a 400, never a stream --------------------------------

def _assert_context_length_envelope(resp):
    assert resp.status_code == 400
    body = resp.json()
    assert body["status"] == "error"
    assert body["error"]["type"] == "context_length_exceeded"
    assert body["error"]["detail"] == {"prompt_tokens": 5000, "context_length": 4096}
    assert body["error"]["retryable"] is False
    assert body["error"]["message"] == (
        "Prompt is 5000 tokens, but the model's context window is 4096 tokens; "
        "nothing is left to generate. Shorten the prompt."
    )
    assert body["request_id"]


@pytest.mark.parametrize("endpoint", sorted(ENDPOINTS))
def test_batch_reject_when_prompt_fills_window(endpoint):
    path, body = ENDPOINTS[endpoint]
    runner = Runner()
    runner.generate_batch = MagicMock(side_effect=ContextLengthExceeded(5000, 4096))
    client = TestClient(app)
    with patch('mlxk2.core.server_base.get_or_load_model', return_value=runner):
        resp = client.post(path, json={"model": "org/model", **body})
    _assert_context_length_envelope(resp)


@pytest.mark.parametrize("endpoint", sorted(ENDPOINTS))
def test_stream_reject_is_a_status_not_an_event(endpoint):
    """The pre-check asks the runner before the 200 goes out."""
    path, body = ENDPOINTS[endpoint]
    budget = MagicMock(side_effect=ContextLengthExceeded(5000, 4096))
    runner = Runner(budget=budget)
    client = TestClient(app)
    with patch('mlxk2.core.server_base.get_or_load_model', return_value=runner):
        resp = client.post(path, json={"model": "org/model", "stream": True, **body})
    _assert_context_length_envelope(resp)
    assert "text/event-stream" not in resp.headers.get("content-type", "")
    assert runner.seen == {}  # generation never started
    # Both surfaces hand the runner an already-formatted prompt
    assert budget.call_args.kwargs["use_chat_template"] is False


@pytest.mark.parametrize("endpoint", sorted(ENDPOINTS))
def test_stream_proceeds_when_budget_allows(endpoint):
    """A passing pre-check changes nothing: the runner still gets the ceiling and clamps itself."""
    path, body = ENDPOINTS[endpoint]
    runner = Runner(reason="stop", budget=MagicMock(return_value=10))
    client = TestClient(app)
    with patch('mlxk2.core.server_base.get_or_load_model', return_value=runner):
        with client.stream("POST", path, json={"model": "org/model", "stream": True, **body}) as resp:
            assert resp.status_code == 200
            events = _sse_events(resp)
    assert runner.seen["max_tokens"] == 32768
    assert events[-1]["choices"][0]["finish_reason"] == "stop"


@pytest.mark.parametrize("endpoint", sorted(ENDPOINTS))
def test_runner_without_generation_budget_still_streams(endpoint):
    path, body = ENDPOINTS[endpoint]
    client = TestClient(app)
    with patch('mlxk2.core.server_base.get_or_load_model', return_value=Runner(reason="stop")):
        with client.stream("POST", path, json={"model": "org/model", "stream": True, **body}) as resp:
            assert resp.status_code == 200
            lines = list(_iter_sse_lines(resp))
    assert lines[-1].strip() == "data: [DONE]"


# --- max_tokens below 1 never reaches a runner -----------------------------------

@pytest.mark.parametrize("endpoint", sorted(ENDPOINTS))
@pytest.mark.parametrize("max_tokens", [0, -1])
def test_max_tokens_below_one_is_a_validation_error(endpoint, max_tokens):
    path, body = ENDPOINTS[endpoint]
    client = TestClient(app)
    with patch('mlxk2.core.server_base.get_or_load_model') as load:
        resp = client.post(path, json={"model": "org/model", "max_tokens": max_tokens, **body})
    assert resp.status_code == 400
    error = resp.json()["error"]
    assert error["type"] == "validation_error"
    assert "max_tokens" in error["detail"]
    load.assert_not_called()


# --- vision helpers ------------------------------------------------------------

@pytest.mark.parametrize("reason", ["stop", "length", None])
def test_emulate_sse_stream_final_chunk_carries_finish_reason(reason):
    async def collect():
        return [chunk async for chunk in emulate_sse_stream("id", 1, "m", "text", finish_reason=reason)]

    chunks = asyncio.run(collect())
    assert chunks[-1].strip() == "data: [DONE]"
    events = [json.loads(c[len("data: "):]) for c in chunks[:-1]]
    assert [e["choices"][0]["finish_reason"] for e in events] == [None, None, reason]
    assert events[1]["choices"][0]["delta"]["content"] == "text"


@pytest.mark.parametrize("reasons,expected", [
    (["stop", "stop"], "stop"),
    (["length", "stop"], "length"),
    (["stop", "length"], "length"),
    ([None, None], None),
])
def test_process_vision_chunks_any_length_wins(reasons, expected):
    by_image = {f"img{i}.jpg": r for i, r in enumerate(reasons)}

    class ChunkRunner:
        """One fresh runner per chunk; the chunk's image says how it ended."""

        def __init__(self, model_path, model_name, verbose=False):
            self.last_finish_reason = None

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def generate(self, **kwargs):
            self.last_finish_reason = by_image[kwargs["images"][0][0]]
            return f"text for {kwargs['images'][0][0]}"

    with patch('mlxk2.core.vision_runner.VisionRunner', ChunkRunner):
        text, reason = process_vision_chunks_server(
            model_path="/mock/path", model_name="mock", prompt="p",
            images=[(name, b"") for name in by_image], chunk_size=1, image_id_map={},
            max_tokens=100, temperature=0.0, top_p=0.9, repetition_penalty=1.0,
        )
    assert reason == expected
    assert text == "text for img0.jpg\n\ntext for img1.jpg"


# --- the server log line -------------------------------------------------------

def test_log_generation_end_reports_cost_and_reason():
    runner = Runner(reason="length")
    runner.last_prompt_tokens, runner.last_completion_tokens, runner.last_max_tokens = 12, 300, 300
    logger = MagicMock()

    log_generation_end(logger, runner, "org/model", stream=True)

    logger.info.assert_called_once()
    message, fields = logger.info.call_args.args[0], logger.info.call_args.kwargs
    assert message == "Generation finished: length"
    assert fields["model"] == "org/model"
    assert fields["stream"] is True
    assert fields["finish_reason"] == "length"
    assert (fields["prompt_tokens"], fields["completion_tokens"], fields["max_tokens"]) == (12, 300, 300)
    assert "request_id" in fields


def test_log_generation_end_skips_runner_without_attributes():
    class Bare:  # no last_* attributes at all; a MagicMock would invent them
        pass

    logger = MagicMock()
    log_generation_end(logger, Bare(), "org/model", stream=False)
    logger.info.assert_not_called()


def test_completion_logs_one_line_with_request_id():
    """The middleware's request_id reaches the log line through the context variable."""
    runner = Runner(reason="stop")
    runner.last_max_tokens = 32768
    client = TestClient(app)
    with patch('mlxk2.core.server_base.get_or_load_model', return_value=runner), \
         patch('mlxk2.core.server_base.logger') as logger:
        resp = client.post("/v1/completions", json={"model": "org/model", "prompt": "Hi"})
    assert resp.status_code == 200
    logger.info.assert_called_once()
    fields = logger.info.call_args.kwargs
    assert logger.info.call_args.args[0] == "Generation finished: stop"
    assert fields["request_id"] == resp.headers["X-Request-ID"]
    assert fields["stream"] is False


class ExplodingRunner:
    """A runner whose generation fails part-way through the stream."""

    def _format_conversation(self, messages):
        return "prompt"

    def generate_streaming(self, **kwargs):
        yield "part"
        raise RuntimeError("backend exploded")


@pytest.mark.parametrize("endpoint,payload,text_of", [
    ("/v1/chat/completions",
     {"model": "org/model", "messages": [{"role": "user", "content": "Hi"}], "stream": True},
     lambda choice: choice["delta"].get("content", "")),
    ("/v1/completions",
     {"model": "org/model", "prompt": "Hi", "stream": True},
     lambda choice: choice.get("text", "")),
])
def test_failed_stream_ends_with_an_error_object_and_no_finish_reason(endpoint, payload, text_of):
    """A backend fault is not a generation outcome: no finish_reason, an ADR-004 error, no [DONE].

    `finish_reason` is an OpenAI enum with no value for a failure, and the OpenAI SDK
    raises on the `error` key without ever reading the choice — so the error object is
    the whole signal, and a trailing "stop"/[DONE] would claim the stream completed.
    """
    client = TestClient(app)
    with patch("mlxk2.core.server_base.get_or_load_model", return_value=ExplodingRunner()):
        with client.stream("POST", endpoint, json=payload) as response:
            assert response.status_code == 200  # headers are long gone; the body carries the fault
            events = [line[len("data: "):] for line in response.iter_lines() if line.startswith("data: ")]

    assert "[DONE]" not in events, "a failed stream must not claim completion"
    chunks = [json.loads(e) for e in events]
    assert "".join(text_of(c["choices"][0]) for c in chunks) == "part"  # tokens already sent stand

    last = chunks[-1]
    assert last["error"] == {"type": "internal_error", "message": "backend exploded"}
    assert all(c["choices"][0]["finish_reason"] is None for c in chunks)
