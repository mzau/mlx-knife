"""Route-level tests for the audio upload size limit (HTTP 413).

The limit is enforced once, in the shared handler, so it fires on both audio routes —
but until now it had no route-level coverage at all: the only test that saw a 413 was
live and asserted the status alone. That is how the second instance of #62 survived,
where the envelope labelled a correct reject `internal_error`.

FastAPI TestClient against the real serve app; the capability gates are faked via
``patch.object`` on the module globals, and the limit itself is patched down so the test
does not have to build a 50 MB body to cross it.
"""

from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

import mlxk2.core.server_base as sb
from mlxk2.core.server_base import app

OVERSIZE_BODY = b"RIFF" + b"\x00" * 64
TINY_LIMIT = 16  # bytes; OVERSIZE_BODY is comfortably above it


def _post(client, route):
    return client.post(
        route,
        files={"file": ("clip.wav", OVERSIZE_BODY, "audio/wav")},
        data={"model": "mlx-community/whisper-large-v3-4bit"},
    )


@pytest.mark.parametrize(
    "route", ["/v1/audio/transcriptions", "/v1/audio/translations"]
)
def test_oversize_upload_returns_413_payload_too_large(route):
    runner_factory = MagicMock()
    with patch.object(sb, "_detect_audio_backend_for_model", lambda m: sb.Backend.MLX_AUDIO), \
         patch.object(sb, "_detect_audio_translate_capable_for_model", lambda m: True), \
         patch.object(sb, "MAX_AUDIO_SIZE_BYTES", TINY_LIMIT), \
         patch.object(sb, "get_or_load_audio_model", runner_factory):
        client = TestClient(app)
        r = _post(client, route)

    assert r.status_code == 413
    error = r.json()["error"]
    # The reject is deliberate and correct: it must not wear a server-fault label (#62).
    assert error["type"] == "payload_too_large"
    assert error["retryable"] is False
    assert "limit" in error["message"]
    # Rejected before any model work happens.
    runner_factory.assert_not_called()
