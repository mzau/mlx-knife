"""The sampling default follows the surface, not the request model.

`temperature` used to default to 0.7 in the request model, which made "unset" and "the
caller asked for 0.7" indistinguishable — so a default chat request against Whisper or
Voxtral transcribed at 0.7, while the two file endpoints correctly used 0.0. The CLI has
had the right rule all along (`run --temperature` unset: 0.0 with audio, else 0.7).
"""

import asyncio
from unittest.mock import AsyncMock, patch

import pytest

from mlxk2.core.server_base import (
    ChatCompletionRequest,
    _handle_audio_chat_completion,
    _handle_text_chat_completion,
    get_effective_temperature,
)

RESPONSE = {
    "id": "chatcmpl-x",
    "created": 0,
    "model": "m",
    "choices": [{"index": 0, "message": {"role": "assistant", "content": "ok"}, "finish_reason": "stop"}],
    "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
}


def test_unset_falls_to_the_surface_default():
    assert get_effective_temperature(None) == 0.7
    assert get_effective_temperature(None, audio=True) == 0.0


@pytest.mark.parametrize("audio", [False, True])
def test_an_explicit_value_always_wins(audio):
    assert get_effective_temperature(0.4, audio=audio) == 0.4
    assert get_effective_temperature(0.0, audio=audio) == 0.0


def _temperature_seen(handler, impl_name, **request_extra):
    """Run one handler with the impl mocked out; report the temperature it was handed."""
    request = ChatCompletionRequest(
        model="m", messages=[{"role": "user", "content": "hi"}], **request_extra
    )
    with patch(f"mlxk2.core.server_base.{impl_name}", new=AsyncMock(return_value=RESPONSE)) as impl:
        asyncio.run(handler(request))
    return impl.await_args.kwargs["temperature"]


def test_audio_chat_transcribes_greedily_by_default():
    seen = _temperature_seen(_handle_audio_chat_completion, "_handle_audio_chat_completion_impl")
    assert seen == 0.0


def test_audio_chat_still_honours_an_explicit_temperature():
    seen = _temperature_seen(
        _handle_audio_chat_completion, "_handle_audio_chat_completion_impl", temperature=0.6
    )
    assert seen == 0.6


def test_text_chat_keeps_its_own_default():
    seen = _temperature_seen(_handle_text_chat_completion, "_handle_text_chat_completion_impl")
    assert seen == 0.7
