"""
Minimal server API tests to keep suite aligned with current code.
Focus: non-streaming chat completions use chat stop tokens in batch path.
"""

from unittest.mock import Mock, patch
from fastapi.testclient import TestClient

from mlxk2.core.server_base import app


def test_chat_completions_batch_uses_chat_stop_tokens_flag():
    client = TestClient(app)

    mock_runner = Mock()
    mock_runner.generate_batch.return_value = "Assistant: Hello"
    mock_runner._format_conversation.return_value = "Human: Hi\n\nAssistant:"

    with patch('mlxk2.core.server_base.get_or_load_model', return_value=mock_runner):
        payload = {
            "model": "test/model",
            "messages": [{"role": "user", "content": "Hi"}],
            "stream": False,
        }
        resp = client.post("/v1/chat/completions", json=payload)
        assert resp.status_code == 200

        # Ensure server passed use_chat_stop_tokens=True to batch generator
        assert mock_runner.generate_batch.called
        kwargs = mock_runner.generate_batch.call_args.kwargs
        assert kwargs.get("use_chat_stop_tokens") is True

