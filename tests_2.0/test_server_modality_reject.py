"""A modality the model does not have is a reject, not a silent drop.

The server routed "text model + images" to the text path, which filters the image parts
out and answers 200 about the text alone — the client never learned its image was
dropped. The handbook promises the opposite, and that promise is what lets `/v1/models`
carry no capability label at all: the modality is answered for at request time.

422, not 501: the request is at fault, so it is a 4xx, and OpenAI answers 400 here. A 5xx
would invite clients to retry a request that cannot ever succeed.
"""

from unittest.mock import Mock, patch

import pytest
from fastapi.testclient import TestClient

from mlxk2.core.server_base import app

IMAGE = {"type": "image_url", "image_url": {"url": "data:image/png;base64,iVBORw0KGgo="}}
AUDIO = {"type": "input_audio", "input_audio": {"data": "AAAA", "format": "wav"}}


def _post(content):
    runner = Mock()  # a plain runner is not a VisionRunner
    payload = {"model": "org/text-model", "messages": [{"role": "user", "content": content}]}
    with patch("mlxk2.core.server_base.get_or_load_model", return_value=runner):
        with patch("mlxk2.core.server_base._detect_audio_backend_for_model", return_value=None):
            return TestClient(app, raise_server_exceptions=False).post(
                "/v1/chat/completions", json=payload
            )


@pytest.mark.parametrize("part,word", [(IMAGE, "images"), (AUDIO, "audio")])
def test_a_text_model_rejects_the_modality(part, word):
    response = _post([{"type": "text", "text": "What is this?"}, part])
    assert response.status_code == 422
    body = response.json()
    assert body["error"]["type"] == "capability_not_supported"
    assert word in body["error"]["message"]
    assert body["error"]["retryable"] is False


def test_a_text_only_request_is_untouched():
    """The reject must not catch the ordinary path."""
    runner = Mock()
    runner._format_conversation.return_value = "prompt"
    runner.generate_batch.return_value = "hello"
    payload = {"model": "org/text-model", "messages": [{"role": "user", "content": "Hi"}]}
    with patch("mlxk2.core.server_base.get_or_load_model", return_value=runner):
        response = TestClient(app).post("/v1/chat/completions", json=payload)
    assert response.status_code == 200
