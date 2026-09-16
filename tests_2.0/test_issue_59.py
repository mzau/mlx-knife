"""Regression tests for issue #59: `max_tokens` reaches the transcription backend.

`AudioRunner.transcribe` took a `max_tokens` and `_transcribe_single` never put it into the
call, so `--max-tokens` had no effect on any audio model. A model that transcribes in one pass
stopped at its own default budget, whatever the caller asked for.

mlx-knife sets no audio budget of its own: a value the caller gave is handed on, and without
one the key stays out, so the model's default applies. mlx-audio drops the key for models whose
`generate` does not name it — that filtering is the library's and is not tested here.
"""

import importlib
import sys

import pytest


@pytest.fixture
def audio_runner_class(monkeypatch):
    """`AudioRunner`, imported with mlx-audio blocked.

    Importing the module applies the Whisper bridge, which pulls the real ``mlx.nn`` into a
    tree collected against the stubbed ``mlx.core``. With ``mlx_audio`` unimportable the bridge
    returns early, and nothing else in the module needs it before a model loads.
    """
    monkeypatch.setitem(sys.modules, "mlx_audio", None)
    sys.modules.pop("mlxk2.core.audio_runner", None)
    module = importlib.import_module("mlxk2.core.audio_runner")
    yield module.AudioRunner
    sys.modules.pop("mlxk2.core.audio_runner", None)


def _recording_runner(audio_runner_class):
    """An `AudioRunner` whose backend records the kwargs of each call."""
    calls = []

    def generate_transcription(**gen_kwargs):
        calls.append(gen_kwargs)
        return "a transcript"

    # A repo id, not a workspace path: the runner hands the name on and loads nothing itself.
    runner = audio_runner_class("org/stt-model", "org/stt-model")
    runner._generate_fn = generate_transcription
    return runner, calls


CLIP = [("clip.wav", b"RIFF\x00\x00\x00\x00WAVEfmt ")]


def test_explicit_max_tokens_reaches_the_backend(audio_runner_class):
    runner, calls = _recording_runner(audio_runner_class)
    assert runner.transcribe(audio=CLIP, max_tokens=32768) == "a transcript"
    assert len(calls) == 1
    assert calls[0]["max_tokens"] == 32768


def test_no_max_tokens_leaves_the_model_default(audio_runner_class):
    runner, calls = _recording_runner(audio_runner_class)
    runner.transcribe(audio=CLIP)
    assert len(calls) == 1
    assert "max_tokens" not in calls[0]


@pytest.mark.parametrize("requested", [0, -1])
def test_max_tokens_below_one_rejects_before_the_backend(audio_runner_class, requested):
    """mlx-lm's generate loop, which LLM-based ASR models run, reads a negative budget as
    unbounded and generates nothing on 0 — neither may reach it."""
    runner, calls = _recording_runner(audio_runner_class)
    with pytest.raises(ValueError, match=rf"max_tokens must be at least 1 \(got {requested}\)"):
        runner.transcribe(audio=CLIP, max_tokens=requested)
    assert calls == []
