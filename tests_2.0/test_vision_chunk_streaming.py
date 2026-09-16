"""
Unit tests for vision chunk streaming SSE format.

Tests the new per-chunk streaming feature where multi-image vision requests
with stream=True yield SSE events as each chunk completes, rather than
waiting for all chunks to finish.
"""

import json
from contextlib import contextmanager
from typing import Iterator
from types import SimpleNamespace
from unittest.mock import patch

from fastapi.testclient import TestClient

from mlxk2.core.server_base import app
from mlxk2.core.vision_runner import VisionRunner


def _iter_sse_lines(resp) -> Iterator[str]:
    """Iterate non-empty SSE lines as strings from a streaming response."""
    for raw in resp.iter_lines():
        if not raw:
            continue
        if isinstance(raw, bytes):
            line = raw.decode("utf-8", errors="ignore")
        else:
            line = raw
        if line.strip():
            yield line


# 1x1 PNG, small enough to inline; one image = one chunk = the emulated-SSE path
PIXEL_PNG = (
    "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJ"
    "AAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg=="
)


@contextmanager
def _wired_vision_stream(image_data_url, finish_reason):
    """Stream a vision chat request through the real server wiring with a stub runner.

    A real ``VisionRunner`` (constructing one loads nothing) so the handler's own
    ``isinstance`` guard passes on its own terms; only ``generate`` is stubbed.
    Patching ``isinstance`` wholesale instead would also make every other check in
    those modules true, which is how a broken wrapper signature stayed invisible.
    """
    runner = VisionRunner("/mock/path", "mock-vision", verbose=False)
    runner.generate = lambda **kwargs: "Single chunk response"
    runner.last_finish_reason = finish_reason

    content = [{"type": "text", "text": "Describe this image"}]
    if image_data_url:
        content.append({"type": "image_url", "image_url": {"url": image_data_url}})
    payload = {
        "model": "mock-vision-model",
        "messages": [{"role": "user", "content": content}],
        "stream": True,
    }

    with patch('mlxk2.core.server_base.get_or_load_model', return_value=runner):
        with TestClient(app).stream("POST", "/v1/chat/completions", json=payload) as resp:
            yield resp


def _parse_sse_events(resp) -> list:
    """Parse SSE events into list of dicts (skips [DONE])."""
    events = []
    for line in _iter_sse_lines(resp):
        if line.strip() == "data: [DONE]":
            continue
        if line.startswith("data: "):
            try:
                events.append(json.loads(line[len("data: "):]))
            except json.JSONDecodeError:
                pass
    return events


class TestVisionChunkStreamingSSEFormat:
    """Tests for vision per-chunk SSE streaming format (mocked endpoint)."""

    def test_single_chunk_uses_emulated_sse(self):
        """Single-chunk requests should use existing SSE emulation (batch response).

        Drives the wired path, not the implementation: the handler reaches
        ``_emulate_sse_stream`` through ``ChatHandlerContext``, so a signature that
        drifts from ``streaming.emulate_sse_stream`` fails here as a 500.
        """
        with _wired_vision_stream(PIXEL_PNG, finish_reason="length") as resp:
            assert resp.status_code == 200, f"expected 200, got {resp.status_code}: {resp.read()!r}"

            events = _parse_sse_events(resp)
            content = "".join(
                e["choices"][0]["delta"].get("content", "") for e in events
            )
            assert content == "Single chunk response"
            # The runner's stop reason has to survive the whole wiring, not just the impl
            assert events[-1]["choices"][0]["finish_reason"] == "length"

    def test_text_on_vision_model_uses_emulated_sse(self):
        """A vision model without images streams through the same wrapper (chat.py text path)."""
        with _wired_vision_stream(None, finish_reason="stop") as resp:
            assert resp.status_code == 200, f"expected 200, got {resp.status_code}: {resp.read()!r}"

            events = _parse_sse_events(resp)
            assert events[-1]["choices"][0]["finish_reason"] == "stop"

    def test_sse_format_compliance(self):
        """A chunked stream through the real server wiring: role, one content event per chunk, the
        finish reason — every event with the fields an OpenAI client reads.

        Runs a real ``VisionRunner`` per chunk with only the model stubbed, so a handler that
        rejects the request or a stream that loses a field fails here.
        """
        routed = VisionRunner("/mock/path", "mock-vision", verbose=False)

        def load(self):
            self._apply_chat_template = lambda *args, **kwargs: "prompt"
            self._generate = lambda *args, **kwargs: SimpleNamespace(
                text="an answer", finish_reason="stop", prompt_tokens=3, generation_tokens=6
            )

        image = {"type": "image_url", "image_url": {"url": PIXEL_PNG}}
        payload = {
            "model": "mock-vision-model", "stream": True, "chunk": 1,
            "messages": [{"role": "user", "content": [{"type": "text", "text": "Test"}, image, image]}],
        }
        with patch.object(VisionRunner, "load_model", load), \
             patch("mlxk2.core.server_base.get_or_load_model", return_value=routed):
            with TestClient(app).stream("POST", "/v1/chat/completions", json=payload) as resp:
                assert resp.status_code == 200, resp.read()
                events = _parse_sse_events(resp)

        assert [bool(e["choices"][0]["delta"].get("content")) for e in events] == [False, True, True, False]
        assert events[0]["choices"][0]["delta"].get("role") == "assistant"
        assert events[-1]["choices"][0]["finish_reason"] == "stop"
        for event in events:
            assert event["object"] == "chat.completion.chunk" and event["id"] == events[0]["id"]
            assert event["choices"][0]["index"] == 0 and "delta" in event["choices"][0]

def _run_stream_vision_chunks(runner_cls, images) -> list:
    """Drive stream_vision_chunks with a stand-in VisionRunner; returns the raw SSE strings."""
    import asyncio
    import threading

    from mlxk2.core.server.streaming import stream_vision_chunks

    async def run_generator():
        events = []
        # Patch at the source module where VisionRunner is defined
        with patch('mlxk2.core.vision_runner.VisionRunner', runner_cls):
            gen = stream_vision_chunks(
                model_path="/mock/path",
                model_name="mock-model",
                prompt="Test prompt",
                images=images,
                chunk_size=1,
                image_id_map={},
                max_tokens=100,
                temperature=0.0,
                top_p=0.9,
                repetition_penalty=1.0,
                completion_id="test-123",
                created=1234567890,
                model="test-model",
                shutdown_event=threading.Event(),
            )
            async for event in gen:
                events.append(event)
        return events

    return asyncio.run(run_generator())


class TestVisionChunkStreamingIntegration:
    """Integration tests that exercise the actual streaming function."""

    def test_stream_vision_chunks_generator_format(self):
        """stream_vision_chunks yields valid SSE format."""

        # Mock VisionRunner. The stream reports the runner's last_finish_reason as-is:
        # a runner that declares nothing yields null, so the mock says "stop" explicitly.
        class MockVisionRunner:
            def __init__(self, *args, **kwargs):
                self.last_finish_reason = None

            def __enter__(self):
                return self

            def __exit__(self, *args):
                pass

            def generate(self, **kwargs):
                self.last_finish_reason = "stop"
                return "Test output"

        events = _run_stream_vision_chunks(
            MockVisionRunner, images=[("img1.jpg", b"fake1"), ("img2.jpg", b"fake2")]
        )

        # Should have: role + 2 content events + final + [DONE]
        assert len(events) >= 4, f"Expected at least 4 events, got {len(events)}: {events}"

        # First event should be role
        assert events[0].startswith("data: ")
        first = json.loads(events[0][6:].strip())
        assert first["choices"][0]["delta"].get("role") == "assistant"

        # Last event should be [DONE]
        assert events[-1].strip() == "data: [DONE]"

        # Second-to-last should have finish_reason
        final = json.loads(events[-2][6:].strip())
        assert final["choices"][0]["finish_reason"] == "stop"

    def test_stream_vision_chunks_one_cut_chunk_makes_final_length(self):
        """A "length" from any chunk wins: a later "stop" does not overwrite it."""
        reasons = {"cut.jpg": "length", "whole.jpg": "stop"}

        class MockVisionRunner:
            def __init__(self, *args, **kwargs):
                self.last_finish_reason = None

            def __enter__(self):
                return self

            def __exit__(self, *args):
                pass

            def generate(self, **kwargs):
                # One fresh runner per chunk; the chunk's image says how it ended
                self.last_finish_reason = reasons[kwargs["images"][0][0]]
                return "Test output"

        events = _run_stream_vision_chunks(
            MockVisionRunner, images=[("cut.jpg", b"fake1"), ("whole.jpg", b"fake2")]
        )

        assert events[-1].strip() == "data: [DONE]"
        final = json.loads(events[-2][6:].strip())
        assert final["choices"][0]["finish_reason"] == "length"
        # Both chunks still streamed their content before the verdict
        content = [json.loads(e[6:]) for e in events[1:-2]]
        assert [c["choices"][0]["finish_reason"] for c in content] == [None, None]
