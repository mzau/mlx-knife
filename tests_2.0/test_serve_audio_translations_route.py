"""Route-level tests for serve's POST /v1/audio/translations (Issue #54).

FastAPI TestClient against the real serve app. The model resolution / capability
gates and the audio runner are faked via ``patch.object`` on the module globals —
no live backend, no model, no audio decode. Covers:
  * non-audio model           -> 400
  * audio, not translate-able  -> 422 AND the runner is never invoked
  * translate-able             -> task="translate" + no synthetic prompt threaded
  * verbose_json               -> response task == "translate"

Plus direct handler tests pinning the conditional-prompt non-regression invariant
(transcribe keeps the synthetic default; translate must not inject it).
"""

import asyncio
from unittest.mock import MagicMock, patch

from fastapi.testclient import TestClient

import mlxk2.core.server_base as sb
from mlxk2.core.server_base import MAX_AUDIO_SIZE_BYTES, app
from mlxk2.core.server.handlers.audio import handle_transcription

FAKE_WAV = b"RIFF\x00\x00\x00\x00WAVEfmt "


def _post(client, **data):
    return client.post(
        "/v1/audio/translations",
        files={"file": ("clip.wav", FAKE_WAV, "audio/wav")},
        data={"model": "mlx-community/whisper-large-v3-4bit", **data},
    )


def test_non_audio_model_returns_400():
    # Backend gate: not an audio model at all -> 400 (mirrors transcriptions).
    with patch.object(sb, "_detect_audio_backend_for_model", lambda m: None):
        client = TestClient(app)
        r = _post(client)
    assert r.status_code == 400
    assert "not an audio model" in r.json()["error"]["message"]


def test_non_translate_capable_returns_422_without_invoking_runner():
    # Audio model that cannot translate (turbo / .en / Voxtral) -> 422, never a
    # silent transcription. The runner must not be touched.
    runner_factory = MagicMock()
    with patch.object(sb, "_detect_audio_backend_for_model", lambda m: sb.Backend.MLX_AUDIO), \
         patch.object(sb, "_detect_audio_translate_capable_for_model", lambda m: False), \
         patch.object(sb, "get_or_load_audio_model", runner_factory):
        client = TestClient(app)
        r = _post(client)
    assert r.status_code == 422
    assert "does not support speech translation" in r.json()["error"]["message"]
    runner_factory.assert_not_called()


def test_translate_threads_task_and_omits_synthetic_prompt():
    runner = MagicMock()
    runner.transcribe.return_value = "ENGLISH OUTPUT"
    with patch.object(sb, "_detect_audio_backend_for_model", lambda m: sb.Backend.MLX_AUDIO), \
         patch.object(sb, "_detect_audio_translate_capable_for_model", lambda m: True), \
         patch.object(sb, "get_or_load_audio_model", lambda model, verbose: runner):
        client = TestClient(app)
        r = _post(client)
    assert r.status_code == 200
    assert r.json() == {"text": "ENGLISH OUTPUT"}
    kwargs = runner.transcribe.call_args.kwargs
    assert kwargs["task"] == "translate"
    # No prompt form field -> no synthetic "Transcribe this audio." on translate.
    assert kwargs["prompt"] is None


def test_translate_verbose_reports_task_translate():
    runner = MagicMock()
    runner.transcribe.return_value = "ENGLISH OUTPUT"
    with patch.object(sb, "_detect_audio_backend_for_model", lambda m: sb.Backend.MLX_AUDIO), \
         patch.object(sb, "_detect_audio_translate_capable_for_model", lambda m: True), \
         patch.object(sb, "get_or_load_audio_model", lambda model, verbose: runner):
        client = TestClient(app)
        r = _post(client, response_format="verbose_json")
    assert r.status_code == 200
    body = r.json()
    assert body["task"] == "translate"
    assert body["text"] == "ENGLISH OUTPUT"


# --- Direct handler invariants (independent of the route) -------------------

def _run_handler(task, prompt):
    runner = MagicMock()
    runner.transcribe.return_value = "OUT"
    result = asyncio.run(
        handle_transcription(
            content=FAKE_WAV,
            filename="clip.wav",
            model="m",
            language=None,
            prompt=prompt,
            response_format="json",
            temperature=0.0,
            get_audio_model_fn=lambda model, verbose: runner,
            max_audio_size_bytes=MAX_AUDIO_SIZE_BYTES,
            task=task,
        )
    )
    return result, runner.transcribe.call_args.kwargs


def test_handler_translate_does_not_inject_synthetic_prompt():
    result, kwargs = _run_handler(task="translate", prompt=None)
    assert result == {"text": "OUT"}
    assert kwargs["task"] == "translate"
    assert kwargs["prompt"] is None


def test_handler_transcribe_keeps_synthetic_prompt_default():
    # Non-regression: transcribe path (task=None) still gets the synthetic default.
    _, kwargs = _run_handler(task=None, prompt=None)
    assert kwargs["task"] is None
    assert kwargs["prompt"] == "Transcribe this audio."


def test_handler_user_prompt_threads_through_on_translate():
    _, kwargs = _run_handler(task="translate", prompt="medical vocabulary")
    assert kwargs["prompt"] == "medical vocabulary"
