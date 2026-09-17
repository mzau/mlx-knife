"""`stop` reaches the batch surfaces.

`generate_batch` has no `stop` parameter, so the field was accepted by the request model
and then dropped: a client got the full answer and `finish_reason` from the runner. The
sequences are now applied to the finished text — the answer ends where OpenAI says it
ends. Token streams keep their own, weaker handling (C3b). The vision runner applies the
cut itself, ahead of the filename header it prepends; a chunked vision stream is a batch
answer per chunk, so it is cut like one and generates no chunk past the match.
"""

import asyncio
import json
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from mlxk2.core.runner.token_limits import apply_stop_sequences
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

    def generate_streaming(self, record=None, **kwargs):
        yield from ["one ", "two ", "STOP", " three"]
        if record is not None:  # a stream reads how it ended from its record, as MLXRunner writes it
            record.last_finish_reason = self.last_finish_reason


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


# --- vision: the runner cuts the model's text, ahead of its filename header --------

# What a vision answer starts with: the runner's filename mapping, "\n\n" included
HEADER_MARKER = "<!-- mlxk:filenames -->"
PIXEL_PNG = (
    "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJ"
    "AAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg=="
)


def _backend(answer, finish_reason, generated):
    """A `load_model` that installs stand-ins for the chat template and mlx-vlm's generate.

    The real `generate` runs above them: temp files, metadata prompt, the stop cut, the
    filename header — only the model is missing.
    """
    from types import SimpleNamespace

    def load(self):
        self._apply_chat_template = lambda *args, **kwargs: "prompt"

        def generate(model, processor, prompt, image_paths, **kwargs):
            generated.append(len(image_paths))
            return SimpleNamespace(
                text=answer, finish_reason=finish_reason, prompt_tokens=3, generation_tokens=6
            )

        self._generate = generate

    return load


def _image_message(count):
    parts = [{"type": "text", "text": "Describe"}]
    parts += [{"type": "image_url", "image_url": {"url": PIXEL_PNG}}] * count
    return [{"role": "user", "content": parts}]


@pytest.mark.parametrize(
    "stop,tail,reason",
    [
        # The sequence sits in the header the runner prepends, not in the answer:
        # the answer stands, and the runner's own reason with it
        ("\n\n", ANSWER, "length"),
        ("STOP", "one two ", "stop"),
    ],
)
def test_a_vision_batch_answer_is_cut_behind_its_header(stop, tail, reason):
    from mlxk2.core.vision_runner import VisionRunner

    routed = VisionRunner("/mock/path", "mock-vision", verbose=False)
    _backend(ANSWER, "length", [])(routed)
    payload = {"model": "m", "messages": _image_message(1), "stop": stop}
    with patch("mlxk2.core.server_base.get_or_load_model", return_value=routed):
        response = TestClient(app).post("/v1/chat/completions", json=payload).json()
    choice = response["choices"][0]
    content = choice["message"]["content"]
    assert HEADER_MARKER in content
    assert content.endswith(tail)
    assert choice["finish_reason"] == reason


def _chunked_vision_stream(stop):
    """Drive `stream_vision_chunks` with a stand-in runner; returns the events and the chunks generated."""
    import threading

    from mlxk2.core.server.streaming import stream_vision_chunks

    generated = []

    class ChunkRunner:
        """Keeps the runner's contract: the cut happens inside, before the header."""

        def __init__(self, *args, **kwargs):
            self.last_finish_reason = None
            self.last_stopped_on_sequence = False

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def generate(self, *, stop=None, **kwargs):
            generated.append(kwargs["images"][0][0])
            text, self.last_stopped_on_sequence = apply_stop_sequences(ANSWER, stop)
            self.last_finish_reason = "stop"
            return "header\n\n" + text

    async def run():
        events = []
        with patch("mlxk2.core.vision_runner.VisionRunner", ChunkRunner):
            async for event in stream_vision_chunks(
                model_path="/mock", model_name="mock", prompt="p",
                images=[("first.jpg", b"1"), ("second.jpg", b"2")], chunk_size=1,
                image_id_map={}, max_tokens=10, temperature=0.0, top_p=0.9,
                repetition_penalty=1.0, completion_id="c", created=0, model="m",
                shutdown_event=threading.Event(), stop=stop,
            ):
                events.append(event)
        return events

    events = [json.loads(e[6:]) for e in asyncio.run(run()) if e != "data: [DONE]\n\n"]
    return events, generated


def _content_of(events):
    return "".join(e["choices"][0]["delta"].get("content", "") for e in events)


def test_a_chunked_vision_stream_stops_at_the_sequence():
    """The chunk's text is a batch answer: cut like one, and no later chunk is generated."""
    events, generated = _chunked_vision_stream(["STOP"])
    assert _content_of(events) == "header\n\none two "
    assert events[-1]["choices"][0]["finish_reason"] == "stop"
    assert generated == ["first.jpg"]


def test_a_chunked_vision_stream_without_a_match_streams_every_chunk():
    events, generated = _chunked_vision_stream(None)
    assert _content_of(events) == f"header\n\n{ANSWER}\n\nheader\n\n{ANSWER}"
    assert generated == ["first.jpg", "second.jpg"]


def test_the_chunked_vision_stream_is_handed_the_sequences():
    """Through the real wiring: the handler's multi-chunk branch returned before any cut."""
    from mlxk2.core.vision_runner import VisionRunner

    generated = []
    # A real VisionRunner, so the router's isinstance guard passes on its own terms
    routed = VisionRunner("/mock/path", "mock-vision", verbose=False)
    payload = {
        "model": "m", "messages": _image_message(2),
        "stream": True, "chunk": 1, "stop": "STOP",
    }
    with patch.object(VisionRunner, "load_model", _backend(ANSWER, "stop", generated)), \
         patch("mlxk2.core.server_base.get_or_load_model", return_value=routed):
        with TestClient(app).stream("POST", "/v1/chat/completions", json=payload) as response:
            lines = [line for line in response.iter_lines() if line.startswith("data: ")]
    events = [json.loads(line[6:]) for line in lines if line != "data: [DONE]"]
    content = _content_of(events)
    assert "Chunk 1/2" in content and content.endswith("one two ")
    assert "Chunk 2/2" not in content
    assert events[-1]["choices"][0]["finish_reason"] == "stop"
    # One chunk of one image was generated; the second chunk never started
    assert generated == [1]
