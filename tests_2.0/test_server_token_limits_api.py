"""
Server-level token limit tests (edge cases without changing core behavior).

Focus: ensure endpoints pass effective max_tokens correctly:
- When request.max_tokens is None -> use runner._calculate_dynamic_max_tokens(server_mode=True)
- When request.max_tokens is set -> pass through unchanged
"""

from unittest.mock import patch

from fastapi.testclient import TestClient

from mlxk2.core.server_base import app


def test_server_completions_uses_dynamic_when_none():
    client = TestClient(app)

    class Runner:
        def _calculate_dynamic_max_tokens(self, server_mode=True):
            assert server_mode is True
            return 123

        def generate_batch(self, **kwargs):
            # Assert server passes the dynamic value
            assert kwargs.get("max_tokens") == 123
            return "ok"

    with patch('mlxk2.core.server_base.get_or_load_model', return_value=Runner()):
        payload = {"model": "org/model", "prompt": "Hi"}  # max_tokens omitted
        resp = client.post("/v1/completions", json=payload)
        assert resp.status_code == 200


def test_server_completions_respects_explicit_max_tokens():
    client = TestClient(app)

    seen = {}

    class Runner:
        def _calculate_dynamic_max_tokens(self, server_mode=True):
            return 999  # should be ignored when explicit max_tokens provided

        def generate_batch(self, **kwargs):
            seen.update(kwargs)
            return "ok"

    with patch('mlxk2.core.server_base.get_or_load_model', return_value=Runner()):
        payload = {"model": "org/model", "prompt": "Hi", "max_tokens": 7}
        resp = client.post("/v1/completions", json=payload)
        assert resp.status_code == 200
        assert seen.get("max_tokens") == 7


def test_server_chat_streaming_uses_dynamic_when_none():
    client = TestClient(app)

    captured = {}

    class Runner:
        def _calculate_dynamic_max_tokens(self, server_mode=True):
            assert server_mode is True
            return 42

        def _format_conversation(self, messages):
            return "prompt"

        def generate_streaming(self, **kwargs):
            captured.update(kwargs)
            yield "A"
            yield "B"

    with patch('mlxk2.core.server_base.get_or_load_model', return_value=Runner()):
        payload = {
            "model": "org/model",
            "messages": [{"role": "user", "content": "Hi"}],
            "stream": True,
        }
        with client.stream("POST", "/v1/chat/completions", json=payload) as resp:
            assert resp.status_code == 200
            for _ in resp.iter_lines():
                pass

    assert captured.get("max_tokens") == 42
    assert captured.get("use_chat_stop_tokens") is True
    assert captured.get("use_chat_template") is False


def test_server_chat_non_streaming_respects_explicit_max_tokens():
    client = TestClient(app)

    seen = {}

    class Runner:
        def _calculate_dynamic_max_tokens(self, server_mode=True):
            return 111

        def _format_conversation(self, messages):
            return "prompt"

        def generate_batch(self, **kwargs):
            seen.update(kwargs)
            return "ok"

    with patch('mlxk2.core.server_base.get_or_load_model', return_value=Runner()):
        payload = {
            "model": "org/model",
            "messages": [{"role": "user", "content": "Hi"}],
            "stream": False,
            "max_tokens": 5,
        }
        resp = client.post("/v1/chat/completions", json=payload)
        assert resp.status_code == 200
        assert seen.get("max_tokens") == 5

