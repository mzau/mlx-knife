"""`stream_options.include_usage`: a stream reports its token counts in OpenAI's form.

With the option every chunk carries `"usage": null`, and one more chunk follows the last one with
a choice — empty `choices`, the counts for the whole request — before `data: [DONE]`. Without it
the stream is what it was. The counts belong to the request: a stream cut at a `stop` sequence,
or sharing its turn with another generation on the same model, still reports its own.
"""

import asyncio
import base64
import json
import re
import threading
from contextlib import contextmanager
from unittest.mock import Mock, patch

import mlx.core as mx
import pytest
from fastapi.testclient import TestClient

from mlxk2.core.runner import MLXRunner
from mlxk2.core.server_base import app
from mlxk2.core.vision_runner import VisionRunner

ASKED = {"stream_options": {"include_usage": True}}
PIXEL_PNG = (
    "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJ"
    "AAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg=="
)
# surface -> (path, request body, chunk object)
TEXT = {
    "chat": ("/v1/chat/completions", {"messages": [{"role": "user", "content": "Hi"}]}, "chat.completion.chunk"),
    "completions": ("/v1/completions", {"prompt": "Hi"}, "text_completion"),
}


class Runner:
    """The text runner surface a request touches; fills the request's record as MLXRunner does."""

    last_prompt_tokens = 11
    last_completion_tokens = 2
    last_finish_reason = "stop"

    def __init__(self, pieces=("A", "B")):
        self.pieces = pieces

    def _format_conversation(self, messages):
        return "prompt"

    def generate_batch(self, **kwargs):
        return "AB"

    def generate_streaming(self, record=None, **kwargs):
        if record is not None:
            record.last_prompt_tokens, record.last_completion_tokens = 11, 0
        for piece in self.pieces:
            if record is not None:
                record.last_completion_tokens += 1
            yield piece


def _stream(path, body, runner, **extra):
    """POST a streaming request; return its non-empty SSE lines."""
    payload = {"model": "org/model", "stream": True, **body, **extra}
    with patch("mlxk2.core.server_base.get_or_load_model", return_value=runner):
        with TestClient(app).stream("POST", path, json=payload) as response:
            assert response.status_code == 200
            return [line for line in response.iter_lines() if line]


def _events(lines):
    return [json.loads(line[len("data: "):]) for line in lines if line.startswith("data: {")]


def _assert_usage_last(lines, chunk_object, usage):
    """OpenAI's form: `usage: null` on every chunk, the usage chunk after the last choice, then [DONE]."""
    assert lines[-1] == "data: [DONE]"
    *chunks, last = _events(lines)
    first = chunks[0]
    assert last == {
        "id": first["id"], "object": chunk_object, "created": first["created"],
        "model": first["model"], "choices": [], "usage": usage,
    }
    assert all(chunk["usage"] is None and chunk["choices"] for chunk in chunks)


def _usage(prompt, completion):
    return {"prompt_tokens": prompt, "completion_tokens": completion, "total_tokens": prompt + completion}


# --- text streams ------------------------------------------------------------------

@pytest.mark.parametrize("surface", sorted(TEXT))
def test_a_text_stream_ends_with_its_usage(surface):
    path, body, chunk_object = TEXT[surface]
    _assert_usage_last(_stream(path, body, Runner(), **ASKED), chunk_object, _usage(11, 2))


def test_a_stream_cut_at_a_stop_sequence_reports_what_was_generated():
    """The emitter leaves the generator at the sequence, before the runner records an end."""
    path, body, _ = TEXT["chat"]
    lines = _stream(path, body, Runner(pieces=("A", "STOP", "C", "D")), stop="STOP", **ASKED)
    *chunks, last = _events(lines)
    assert chunks[-1]["choices"][0]["finish_reason"] == "stop"
    assert last["usage"] == _usage(11, 2)


def test_a_runner_that_counts_nothing_falls_back_to_the_estimate():
    class Silent(Runner):
        def _format_conversation(self, messages):
            return "one two three four five six seven eight nine ten"

        def generate_streaming(self, **kwargs):
            yield "a b c d e f g h i j"

    path, body, _ = TEXT["chat"]
    assert _events(_stream(path, body, Silent(), **ASKED))[-1]["usage"] == _usage(13, 13)


@pytest.mark.parametrize("surface", sorted(TEXT))
def test_without_the_option_the_stream_is_unchanged(surface):
    path, body, _ = TEXT[surface]
    plain = _stream(path, body, Runner())
    declined = _stream(path, body, Runner(), stream_options={"include_usage": False})

    def normalized(lines):
        return [re.sub(r'"(id|created)": ("[^"]*"|\d+)', r'"\1": _', line) for line in lines]

    assert not any("usage" in line for line in plain)
    assert normalized(declined) == normalized(plain)


@pytest.mark.parametrize("options,reported", [
    ({}, False),
    ({"include_usage": None}, False),
    ({"include_usage": True, "include_obfuscation": True}, True),
], ids=["empty", "null", "unknown-key"])
def test_the_option_object_is_read_leniently(options, reported):
    path, body, _ = TEXT["chat"]
    lines = _stream(path, body, Runner(), stream_options=options)
    assert lines[-1] == "data: [DONE]"
    assert (_events(lines)[-1]["choices"] == []) is reported


@pytest.mark.parametrize("surface", sorted(TEXT))
def test_a_batch_request_ignores_the_option(surface):
    path, body, _ = TEXT[surface]
    responses = []
    with patch("mlxk2.core.server_base.get_or_load_model", return_value=Runner()):
        for extra in ({}, ASKED):
            response = TestClient(app).post(path, json={"model": "org/model", **body, **extra}).json()
            responses.append({k: v for k, v in response.items() if k not in ("id", "created")})
    assert responses[1] == responses[0]
    assert responses[0]["usage"] == _usage(11, 2)


def test_a_failed_stream_sends_no_usage_chunk():
    class Exploding(Runner):
        def generate_streaming(self, record=None, **kwargs):
            yield "part"
            raise RuntimeError("backend exploded")

    path, body, _ = TEXT["chat"]
    lines = _stream(path, body, Exploding(), **ASKED)
    events = _events(lines)
    assert "data: [DONE]" not in lines
    assert events[-1]["error"]["type"] == "internal_error"
    assert all(event["choices"] and event["usage"] is None for event in events)


# --- vision and audio --------------------------------------------------------------

@pytest.mark.parametrize("images", [0, 1], ids=["text-on-vision-model", "one-image"])
def test_an_emulated_vision_stream_ends_with_its_usage(images):
    runner = VisionRunner("/mock/path", "mock-vision", verbose=False)

    def generate(**kwargs):
        runner.last_prompt_tokens, runner.last_completion_tokens = 7, 3
        return "an answer"

    runner.generate = generate
    parts = [{"type": "text", "text": "Describe"}]
    parts += [{"type": "image_url", "image_url": {"url": PIXEL_PNG}}] * images
    body = {"messages": [{"role": "user", "content": parts}]}
    _assert_usage_last(_stream("/v1/chat/completions", body, runner, **ASKED), "chat.completion.chunk", _usage(7, 3))


def test_a_chunked_vision_stream_ends_with_the_sum_of_its_chunks():
    from mlxk2.core.server.streaming import stream_vision_chunks

    class ChunkRunner:
        def __init__(self, *args, **kwargs):
            self.last_stopped_on_sequence = False

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def generate(self, **kwargs):
            self.last_finish_reason = "length"
            self.last_prompt_tokens, self.last_completion_tokens = 3, 8
            return "an answer"

    async def run():
        with patch("mlxk2.core.vision_runner.VisionRunner", ChunkRunner):
            return [event.strip() async for event in stream_vision_chunks(
                model_path="/mock", model_name="mock", prompt="p",
                images=[("first.jpg", b"1"), ("second.jpg", b"2")], chunk_size=1,
                image_id_map={}, max_tokens=8, temperature=0.0, top_p=0.9,
                repetition_penalty=1.0, completion_id="c", created=0, model="m",
                shutdown_event=threading.Event(), include_usage=True,
            )]

    _assert_usage_last(asyncio.run(run()), "chat.completion.chunk", _usage(6, 16))


def test_an_audio_chat_stream_ends_with_its_usage():
    """The transcription backend counts nothing, so the stream carries the estimate, as the batch does."""
    from mlxk2.core.server.handlers.audio import handle_audio_chat_completion
    from mlxk2.core.server.streaming import emulate_sse_stream

    class AudioRunner:
        def transcribe(self, **kwargs):
            return "a transcript"

    audio = base64.b64encode(b"RIFF....WAVEfmt ").decode()
    messages = [{"role": "user", "content": [
        {"type": "input_audio", "input_audio": {"data": audio, "format": "wav"}}]}]

    async def run():
        response = await handle_audio_chat_completion(
            "org/whisper", messages, None, 0.0, True,
            get_audio_model_fn=lambda model, verbose: AudioRunner(),
            emulate_sse_fn=emulate_sse_stream,
            count_tokens_fn=lambda text: 4,
            include_usage=True,
        )
        return [chunk.strip() async for chunk in response.body_iterator]

    _assert_usage_last(asyncio.run(run()), "chat.completion.chunk", _usage(4, 4))


# --- the counts belong to the request ------------------------------------------------

class _Detokenizer:
    """Just enough of mlx-lm's streaming detokenizer: one `t<id>.` per token."""

    def __init__(self):
        self.tokens = []

    def reset(self):
        self.tokens = []

    def add_token(self, token_id):
        self.tokens.append(int(token_id))

    def finalize(self):
        pass

    @property
    def text(self):
        return "".join(f"t{token}." for token in self.tokens)


@contextmanager
def _mlx_runner(tmp_path, tokens=10):
    """An MLXRunner over stubs: the prompt length follows the prompt's words, and every
    generation gets a token stream of its own, so two generations can interleave."""
    tokenizer = Mock()
    tokenizer.eos_token, tokenizer.eos_token_id, tokenizer.eos_token_ids = "</s>", 999, {999}
    tokenizer.pad_token = None
    tokenizer.additional_special_tokens = []
    tokenizer.added_tokens_decoder = {}
    tokenizer.chat_template = None
    tokenizer.name_or_path = "mock-counts"
    tokenizer.encode = lambda text, *args, **kwargs: list(range(1, len(text.split()) + 1))
    tokenizer.detokenizer = _Detokenizer()
    (tmp_path / "models--test-model" / "snapshots" / "abc123").mkdir(parents=True)

    def token_stream(**kwargs):
        return iter([(mx.array([i + 1]), mx.zeros(1)) for i in range(tokens)])

    with patch("mlxk2.core.runner.load", return_value=(Mock(), tokenizer)), \
         patch("mlxk2.core.runner.resolve_model_for_operation", return_value=("test-model", None, None)), \
         patch("mlxk2.core.runner.get_current_model_cache", return_value=tmp_path), \
         patch("mlxk2.core.runner.hf_to_cache_dir", return_value="models--test-model"), \
         patch("mlxk2.core.runner.get_model_context_length", return_value=8192), \
         patch("mlxk2.core.runner.generate_step", side_effect=token_stream):
        with MLXRunner("test-model") as runner:
            yield runner


def test_a_stream_left_early_keeps_its_count(tmp_path):
    from mlxk2.core.server.streaming import GenerationRecord

    record = GenerationRecord()
    with _mlx_runner(tmp_path) as runner:
        stream = runner.generate_streaming("one two three", record=record)
        for _ in range(4):
            next(stream)
    assert (record.last_prompt_tokens, record.last_completion_tokens) == (3, 4)


def test_a_generation_served_meanwhile_leaves_the_count_alone(tmp_path):
    """A request served between two steps of a stream rewrites the runner's own attributes."""
    from mlxk2.core.server.streaming import GenerationRecord

    first, second = GenerationRecord(), GenerationRecord()
    with _mlx_runner(tmp_path) as runner:
        stream = runner.generate_streaming("one two three", record=first)
        next(stream)
        next(stream)
        list(runner.generate_streaming("one two three four five", record=second))
        next(stream)
        assert runner.last_prompt_tokens == 5
    assert (first.last_prompt_tokens, first.last_completion_tokens) == (3, 3)
    assert (second.last_prompt_tokens, second.last_completion_tokens) == (5, 10)


def test_a_generation_served_meanwhile_leaves_the_budget_and_reason_alone(tmp_path):
    """The log line and the final chunk read the record, so it keeps its own budget and exit."""
    from mlxk2.core.server.streaming import GenerationRecord

    first, second = GenerationRecord(), GenerationRecord()
    with _mlx_runner(tmp_path) as runner:
        stream = runner.generate_streaming("one two three", max_tokens=10, record=first)
        next(stream)
        list(runner.generate_streaming("one two three four five", max_tokens=20, record=second))
        list(stream)
        assert runner.last_max_tokens == 20
    assert (first.last_max_tokens, first.last_completion_tokens, first.last_finish_reason) == (10, 10, "length")
    assert (second.last_max_tokens, second.last_completion_tokens, second.last_finish_reason) == (20, 10, None)


# --- what an OpenAI client reads ---------------------------------------------------

def _openai_client():
    openai = pytest.importorskip("openai")
    return openai.OpenAI(
        base_url="http://testserver/v1", api_key="unused", max_retries=0, http_client=TestClient(app)
    )


def test_the_openai_stream_helper_reads_the_usage():
    """The helper keeps only `chat.completion.chunk` objects — a usage chunk under another object is lost."""
    client = _openai_client()
    helper = getattr(client.chat.completions, "stream", None)
    if helper is None:
        pytest.skip("this openai version has no chat.completions.stream helper")
    with patch("mlxk2.core.server_base.get_or_load_model", return_value=Runner()):
        with helper(model="org/model", messages=[{"role": "user", "content": "Hi"}],
                    stream_options={"include_usage": True}) as stream:
            for _ in stream:
                pass
            usage = stream.get_final_completion().usage
    assert (usage.prompt_tokens, usage.completion_tokens, usage.total_tokens) == (11, 2, 13)


def test_the_openai_client_reads_the_completions_usage_chunk():
    client = _openai_client()
    with patch("mlxk2.core.server_base.get_or_load_model", return_value=Runner()):
        chunks = list(client.completions.create(
            model="org/model", prompt="Hi", stream=True, stream_options={"include_usage": True}
        ))
    assert chunks[-1].choices == []
    assert chunks[-1].usage.completion_tokens == 2
