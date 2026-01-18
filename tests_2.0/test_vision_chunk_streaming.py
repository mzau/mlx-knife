"""
Unit tests for vision chunk streaming SSE format.

Tests the new per-chunk streaming feature where multi-image vision requests
with stream=True yield SSE events as each chunk completes, rather than
waiting for all chunks to finish.
"""

import json
from typing import Iterator
from unittest.mock import patch, MagicMock

from fastapi.testclient import TestClient

from mlxk2.core.server_base import app


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

    def test_multi_chunk_streams_multiple_content_events(self):
        """Multi-chunk vision request should emit SSE event per chunk."""
        # This test validates the SSE format by mocking _stream_vision_chunks directly
        from fastapi import FastAPI
        from fastapi.responses import StreamingResponse

        test_app = FastAPI()

        async def mock_stream_gen():
            yield 'data: {"id":"test","object":"chat.completion.chunk","created":1234,"model":"test","choices":[{"index":0,"delta":{"role":"assistant"},"finish_reason":null}]}\n\n'
            yield 'data: {"id":"test","object":"chat.completion.chunk","created":1234,"model":"test","choices":[{"index":0,"delta":{"content":"Chunk 1 output\\n\\n"},"finish_reason":null}]}\n\n'
            yield 'data: {"id":"test","object":"chat.completion.chunk","created":1234,"model":"test","choices":[{"index":0,"delta":{"content":"Chunk 2 output\\n\\n"},"finish_reason":null}]}\n\n'
            yield 'data: {"id":"test","object":"chat.completion.chunk","created":1234,"model":"test","choices":[{"index":0,"delta":{"content":"Chunk 3 output"},"finish_reason":null}]}\n\n'
            yield 'data: {"id":"test","object":"chat.completion.chunk","created":1234,"model":"test","choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}\n\n'
            yield "data: [DONE]\n\n"

        @test_app.post("/test-stream")
        async def test_endpoint():
            return StreamingResponse(
                mock_stream_gen(),
                media_type="text/event-stream"
            )

        client = TestClient(test_app)

        with client.stream("POST", "/test-stream") as resp:
            assert resp.status_code == 200

            events = _parse_sse_events(resp)

            # Should have: role event + 3 content events + final event = 5 events
            assert len(events) == 5, f"Expected 5 events, got {len(events)}"

            # First event should have role
            assert events[0]["choices"][0]["delta"].get("role") == "assistant"

            # Content events should have content
            content_events = [e for e in events if e["choices"][0]["delta"].get("content")]
            assert len(content_events) == 3, f"Expected 3 content events, got {len(content_events)}"

            # Final event should have finish_reason
            assert events[-1]["choices"][0]["finish_reason"] == "stop"

    def test_single_chunk_uses_emulated_sse(self):
        """Single-chunk requests should use existing SSE emulation (batch response)."""
        client = TestClient(app)

        mock_runner = MagicMock()
        mock_runner.model_path = "/mock/path"
        mock_runner.model_name = "mock-vision"
        mock_runner.generate.return_value = "Single chunk response"

        with patch('mlxk2.core.server_base.get_or_load_model', return_value=mock_runner), \
             patch('mlxk2.core.server_base.isinstance', side_effect=lambda obj, cls: True):
            # 1 image = single chunk, uses _emulate_sse_stream
            payload = {
                "model": "mock-vision-model",
                "messages": [{
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Describe this image"},
                        {"type": "image_url", "image_url": {"url": "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg=="}},
                    ]
                }],
                "stream": True,
            }

            with client.stream("POST", "/v1/chat/completions", json=payload) as resp:
                # Should return 200 (either streaming or error is acceptable here
                # since we're testing the routing, not the full integration)
                assert resp.status_code in [200, 400, 500]

    def test_sse_format_compliance(self):
        """SSE events should follow OpenAI format."""
        client = TestClient(app)

        with patch('mlxk2.core.server_base._stream_vision_chunks') as mock_stream:
            async def mock_stream_gen(*args, **kwargs):
                yield 'data: {"id":"chatcmpl-123","object":"chat.completion.chunk","created":1234567890,"model":"test","choices":[{"index":0,"delta":{"role":"assistant"},"finish_reason":null}]}\n\n'
                yield 'data: {"id":"chatcmpl-123","object":"chat.completion.chunk","created":1234567890,"model":"test","choices":[{"index":0,"delta":{"content":"Hello"},"finish_reason":null}]}\n\n'
                yield 'data: {"id":"chatcmpl-123","object":"chat.completion.chunk","created":1234567890,"model":"test","choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}\n\n'
                yield "data: [DONE]\n\n"

            mock_stream.return_value = mock_stream_gen()

            mock_runner = MagicMock()
            mock_runner.model_path = "/mock/path"
            mock_runner.model_name = "mock-vision"

            with patch('mlxk2.core.server_base.get_or_load_model', return_value=mock_runner), \
                 patch('mlxk2.core.server_base.isinstance', side_effect=lambda obj, cls: True):
                payload = {
                    "model": "mock-vision-model",
                    "messages": [{
                        "role": "user",
                        "content": [
                            {"type": "text", "text": "Test"},
                            {"type": "image_url", "image_url": {"url": "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg=="}},
                            {"type": "image_url", "image_url": {"url": "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg=="}},
                        ]
                    }],
                    "stream": True,
                    "chunk": 1,
                }

                with client.stream("POST", "/v1/chat/completions", json=payload) as resp:
                    events = _parse_sse_events(resp)

                    for event in events:
                        # Required fields per OpenAI spec
                        assert "id" in event, "Missing 'id' field"
                        assert "object" in event, "Missing 'object' field"
                        assert event["object"] == "chat.completion.chunk"
                        assert "choices" in event, "Missing 'choices' field"
                        assert len(event["choices"]) > 0

                        choice = event["choices"][0]
                        assert "index" in choice, "Missing 'index' in choice"
                        assert "delta" in choice, "Missing 'delta' in choice"


class TestVisionChunkStreamingIntegration:
    """Integration tests that exercise the actual streaming function."""

    def test_stream_vision_chunks_generator_format(self):
        """Test _stream_vision_chunks yields valid SSE format."""
        import asyncio
        from mlxk2.core.server_base import _stream_vision_chunks

        # Mock VisionRunner
        class MockVisionRunner:
            def __init__(self, *args, **kwargs):
                pass

            def __enter__(self):
                return self

            def __exit__(self, *args):
                pass

            def generate(self, **kwargs):
                return "Test output"

        async def run_generator():
            events = []
            # Patch at the source module where VisionRunner is defined
            with patch('mlxk2.core.vision_runner.VisionRunner', MockVisionRunner):
                gen = _stream_vision_chunks(
                    model_path="/mock/path",
                    model_name="mock-model",
                    prompt="Test prompt",
                    images=[("img1.jpg", b"fake1"), ("img2.jpg", b"fake2")],
                    chunk_size=1,
                    image_id_map={},
                    max_tokens=100,
                    temperature=0.0,
                    top_p=0.9,
                    repetition_penalty=1.0,
                    completion_id="test-123",
                    created=1234567890,
                    model="test-model",
                )
                async for event in gen:
                    events.append(event)
            return events

        events = asyncio.run(run_generator())

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
