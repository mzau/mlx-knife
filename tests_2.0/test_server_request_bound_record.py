"""A request reports its own generation, whatever the model thread starts next.

The model thread takes the next waiting request the moment one returns, and that generation
resets the shared runner's `last_*` before the event loop reads them. So a batch answer takes
its record in the same model-thread call as its generation, and a stream reads the record the
runner keeps for it — for `usage`, `finish_reason` and the `Generation finished` log line.
"""

import json
from contextlib import contextmanager
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from mlxk2.core.server_base import app
from mlxk2.core.vision_runner import VisionRunner

PIXEL_PNG = (
    "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJ"
    "AAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg=="
)
TEXT = {
    "chat": ("/v1/chat/completions", {"messages": [{"role": "user", "content": "Hi"}]}),
    "completions": ("/v1/completions", {"prompt": "Hi"}),
}
# What the next request's `_begin_generation` leaves on the shared runner
NEXT_REQUEST = {
    "last_prompt_tokens": 999, "last_completion_tokens": None,
    "last_finish_reason": None, "last_max_tokens": 4096,
}


class TextRunner:
    """Records its generation as MLXRunner does: on itself, and in the record a stream hands it."""

    def _format_conversation(self, messages):
        return "prompt"

    def _begin(self, record):
        self.last_prompt_tokens, self.last_completion_tokens = 40, None
        self.last_finish_reason, self.last_max_tokens = None, 5
        if record is not None:
            record.last_prompt_tokens, record.last_completion_tokens, record.last_max_tokens = 40, 0, 5

    def _end(self, record):
        self.last_completion_tokens, self.last_finish_reason = 5, "length"
        if record is not None:
            record.last_completion_tokens, record.last_finish_reason = 5, "length"

    def generate_batch(self, **kwargs):
        self._begin(None)
        self._end(None)
        return "1, 2, 3, 4, 5"

    def generate_streaming(self, record=None, **kwargs):
        self._begin(record)
        yield from ("1,", " 2,", " 3,", " 4,", " 5")
        self._end(record)


@contextmanager
def _next_request_begins(runner):
    """Every model-thread call is followed at once by the next request's generation beginning."""
    from mlxk2.core.server import inference

    real = inference.in_worker

    async def in_worker(fn, /, *args, **kwargs):
        result = await real(fn, *args, **kwargs)
        for name, value in NEXT_REQUEST.items():
            setattr(runner, name, value)
        return result

    with patch("mlxk2.core.server.inference.in_worker", in_worker), \
         patch("mlxk2.core.server.streaming.in_worker", in_worker), \
         patch("mlxk2.core.server.handlers.chat.in_worker", in_worker), \
         patch("mlxk2.core.server_base.in_worker", in_worker), \
         patch("mlxk2.core.server_base.get_or_load_model", return_value=runner):
        yield


def _post(path, body, runner, **extra):
    with _next_request_begins(runner):
        return TestClient(app).post(path, json={"model": "org/model", **body, **extra}).json()


def _final_stream_event(path, body, runner):
    with _next_request_begins(runner):
        with TestClient(app).stream("POST", path, json={"model": "org/model", "stream": True, **body}) as resp:
            assert resp.status_code == 200
            events = [json.loads(line[len("data: "):]) for line in resp.iter_lines() if line.startswith("data: {")]
    return events[-1]


def _usage(prompt, completion):
    return {"prompt_tokens": prompt, "completion_tokens": completion, "total_tokens": prompt + completion}


@pytest.mark.parametrize("surface", sorted(TEXT))
def test_a_text_answer_reports_its_own_usage_and_finish_reason(surface):
    path, body = TEXT[surface]
    response = _post(path, body, TextRunner())
    assert response["usage"] == _usage(40, 5)
    assert response["choices"][0]["finish_reason"] == "length"


@pytest.mark.parametrize("images", [0, 1], ids=["text-on-vision-model", "one-image"])
def test_a_vision_answer_reports_its_own_usage_and_finish_reason(images):
    runner = VisionRunner("/mock/path", "mock-vision", verbose=False)

    def generate(**kwargs):
        runner.last_prompt_tokens, runner.last_completion_tokens = 7, 3
        runner.last_finish_reason, runner.last_max_tokens = "length", 3
        return "an answer"

    runner.generate = generate
    parts = [{"type": "text", "text": "Describe"}]
    parts += [{"type": "image_url", "image_url": {"url": PIXEL_PNG}}] * images
    response = _post("/v1/chat/completions", {"messages": [{"role": "user", "content": parts}]}, runner)
    assert response["usage"] == _usage(7, 3)
    assert response["choices"][0]["finish_reason"] == "length"


@pytest.mark.parametrize("surface", sorted(TEXT))
def test_a_stream_ends_with_its_own_finish_reason(surface):
    path, body = TEXT[surface]
    assert _final_stream_event(path, body, TextRunner())["choices"][0]["finish_reason"] == "length"


@pytest.mark.parametrize("stream", [False, True], ids=["batch", "stream"])
@pytest.mark.parametrize("surface", sorted(TEXT))
def test_the_log_line_names_the_requests_own_generation(surface, stream):
    path, body = TEXT[surface]
    logger = MagicMock()
    with patch("mlxk2.logging.get_logger", return_value=logger), \
         patch("mlxk2.core.server_base.logger", logger):
        if stream:
            _final_stream_event(path, body, TextRunner())
        else:
            _post(path, body, TextRunner())
    lines = [call for call in logger.info.call_args_list
             if call.args and str(call.args[0]).startswith("Generation finished")]
    assert len(lines) == 1
    fields = lines[0].kwargs
    assert lines[0].args[0] == "Generation finished: length"
    assert (fields["prompt_tokens"], fields["completion_tokens"], fields["max_tokens"]) == (40, 5, 5)
    assert fields["stream"] is stream
