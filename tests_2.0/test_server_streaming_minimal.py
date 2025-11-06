"""
Streaming SSE minimal tests for 2.0 server.

Covers:
- Happy-path SSE for /v1/completions with a few chunks
- Interrupt path yields an interrupt marker chunk
- Chat streaming passes use_chat_stop_tokens=True to the runner
"""

import json
from typing import Iterator
from unittest.mock import patch

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


def test_streaming_completions_happy_path_sse():
    client = TestClient(app)

    class DummyRunner:
        def _calculate_dynamic_max_tokens(self, server_mode: bool = True):
            return 16
        def generate_streaming(self, **kwargs):
            yield "Hello"
            yield " world"
            yield "!"

    with patch('mlxk2.core.server_base.get_or_load_model', return_value=DummyRunner()):
        payload = {"model": "org/model", "prompt": "Hi", "stream": True}
        with client.stream("POST", "/v1/completions", json=payload) as resp:
            assert resp.status_code == 200
            # Content type can vary under TestClient; just ensure header exists
            assert "content-type" in resp.headers

            lines = list(_iter_sse_lines(resp))
            # Expect at least initial data + a few chunks + final [DONE]
            assert any(l.startswith("data: ") for l in lines)
            assert any(l.strip() == "data: [DONE]" for l in lines)


def test_streaming_completions_interrupt_marker():
    client = TestClient(app)

    class InterruptingRunner:
        def _calculate_dynamic_max_tokens(self, server_mode: bool = True):
            return 16
        def generate_streaming(self, **kwargs):
            yield "Hello"
            raise KeyboardInterrupt()

    with patch('mlxk2.core.server_base.get_or_load_model', return_value=InterruptingRunner()):
        payload = {"model": "org/model", "prompt": "Hi", "stream": True}
        with client.stream("POST", "/v1/completions", json=payload) as resp:
            assert resp.status_code == 200
            lines = [l for l in _iter_sse_lines(resp) if l.startswith("data: ")]
            # Find JSON chunks (skip [DONE])
            json_chunks = []
            for l in lines:
                if l.strip() == "data: [DONE]":
                    continue
                try:
                    json_chunks.append(json.loads(l[len("data: "):]))
                except Exception:
                    pass
            # One of the chunks should contain the interrupt marker text
            assert any("interrupted" in (c.get("choices", [{}])[0].get("text", "").lower()) for c in json_chunks)


def test_chat_streaming_uses_chat_stop_tokens_flag():
    client = TestClient(app)

    captured = {}

    class CapturingRunner:
        def _calculate_dynamic_max_tokens(self, server_mode: bool = True):
            return 16
        def _format_conversation(self, messages):
            return "prompt"

        def generate_streaming(self, **kwargs):
            captured.update(kwargs)
            yield "Hi"
            yield " there"

    with patch('mlxk2.core.server_base.get_or_load_model', return_value=CapturingRunner()):
        payload = {
            "model": "org/model",
            "messages": [{"role": "user", "content": "Hi"}],
            "stream": True,
        }
        with client.stream("POST", "/v1/chat/completions", json=payload) as resp:
            assert resp.status_code == 200
            # Consume stream to ensure generator ran and captured kwargs
            for _ in _iter_sse_lines(resp):
                pass

    assert captured.get("use_chat_stop_tokens") is True
    assert captured.get("use_chat_template") is False
