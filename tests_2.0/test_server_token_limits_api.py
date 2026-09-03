"""
Server-level token ceiling tests: which number reaches the runner (#66).

The server is policy only — request max_tokens > operator ceiling > default —
and hands that number to the runner, which clamps it to the model's window.
Covered on all four text surfaces: chat/completions × batch/stream.
"""

from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from mlxk2.core.server_base import app, get_effective_max_tokens_vision


class CapturingRunner:
    """Records the kwargs the endpoint hands to generation."""

    def __init__(self):
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


def _max_tokens_seen_by_runner(surface: str, **payload_extra) -> int:
    """POST on one surface with a capturing runner; return the max_tokens it received."""
    path, body, stream = SURFACES[surface]
    payload = {"model": "org/model", "stream": stream, **body, **payload_extra}
    runner = CapturingRunner()
    client = TestClient(app)
    with patch('mlxk2.core.server_base.get_or_load_model', return_value=runner):
        if stream:
            with client.stream("POST", path, json=payload) as resp:
                assert resp.status_code == 200
                for _ in resp.iter_lines():  # the generator only runs when consumed
                    pass
        else:
            resp = client.post(path, json=payload)
            assert resp.status_code == 200
    return runner.seen["max_tokens"]


@pytest.mark.parametrize("surface", sorted(SURFACES))
def test_default_ceiling_reaches_runner(surface):
    assert _max_tokens_seen_by_runner(surface) == 32768


@pytest.mark.parametrize("surface", sorted(SURFACES))
def test_explicit_max_tokens_passes_through(surface):
    assert _max_tokens_seen_by_runner(surface, max_tokens=7) == 7


@pytest.mark.parametrize("surface", sorted(SURFACES))
def test_operator_ceiling_beats_default(monkeypatch, surface):
    monkeypatch.setattr("mlxk2.core.server_base._default_max_tokens", 500)
    assert _max_tokens_seen_by_runner(surface) == 500


@pytest.mark.parametrize("surface", sorted(SURFACES))
def test_request_beats_operator_ceiling(monkeypatch, surface):
    monkeypatch.setattr("mlxk2.core.server_base._default_max_tokens", 500)
    assert _max_tokens_seen_by_runner(surface, max_tokens=7) == 7


def test_vision_ceiling_precedence(monkeypatch):
    """Same order for the vision ceiling, with its own default."""
    monkeypatch.setattr("mlxk2.core.server_base._default_max_tokens", None)
    assert get_effective_max_tokens_vision(None) == 2048
    assert get_effective_max_tokens_vision(64) == 64
    monkeypatch.setattr("mlxk2.core.server_base._default_max_tokens", 500)
    assert get_effective_max_tokens_vision(None) == 500
    assert get_effective_max_tokens_vision(64) == 64
