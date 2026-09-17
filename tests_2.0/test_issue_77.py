"""Regression tests for issue #77: a transcription writes nothing into the working directory.

mlx-audio's `generate_transcription` saves every result, and its `output_path` defaults to
`"transcript"` — a file `transcript.txt` in the working directory. `_transcribe_single` passed no
path, so `mlxk run --audio` wrote there and `mlxk serve` did so on every request: a file of that
name was replaced, and a directory without write permission failed the transcription.

The fake backend saves the way the library does. Whether the library still names the parameter
`output_path` is not tested here — a live test runs the real one.
"""

import importlib
import os
import sys

import pytest


@pytest.fixture
def audio_runner_class(monkeypatch):
    """`AudioRunner`, imported with mlx-audio blocked (see `test_issue_59.py`)."""
    monkeypatch.setitem(sys.modules, "mlx_audio", None)
    sys.modules.pop("mlxk2.core.audio_runner", None)
    module = importlib.import_module("mlxk2.core.audio_runner")
    yield module.AudioRunner
    sys.modules.pop("mlxk2.core.audio_runner", None)


def _saving_runner(audio_runner_class, fail=False):
    """An `AudioRunner` whose backend saves its result like `generate_transcription` does."""
    output_paths = []

    def generate_transcription(output_path="transcript", **gen_kwargs):
        output_paths.append(output_path)
        with open(f"{output_path}.txt", "w", encoding="utf-8") as f:
            f.write("a transcript")
        if fail:
            raise ValueError("decoding failed")
        return "a transcript"

    runner = audio_runner_class("org/stt-model", "org/stt-model")
    runner._generate_fn = generate_transcription
    return runner, output_paths


CLIP = [("clip.wav", b"RIFF\x00\x00\x00\x00WAVEfmt ")]


def test_a_file_named_transcript_txt_stays_untouched(audio_runner_class, tmp_path, monkeypatch):
    notes = tmp_path / "transcript.txt"
    notes.write_text("my own notes")
    monkeypatch.chdir(tmp_path)

    runner, _ = _saving_runner(audio_runner_class)
    assert runner.transcribe(audio=CLIP) == "a transcript"

    assert notes.read_text() == "my own notes"
    assert sorted(p.name for p in tmp_path.iterdir()) == ["transcript.txt"]


def test_a_read_only_working_directory_does_not_fail(audio_runner_class, tmp_path, monkeypatch):
    read_only = tmp_path / "read-only"
    read_only.mkdir()
    read_only.chmod(0o555)
    monkeypatch.chdir(read_only)
    try:
        runner, _ = _saving_runner(audio_runner_class)
        assert runner.transcribe(audio=CLIP) == "a transcript"
    finally:
        read_only.chmod(0o755)


@pytest.mark.parametrize("fail", [False, True])
def test_the_saved_result_is_removed(audio_runner_class, tmp_path, monkeypatch, fail):
    monkeypatch.chdir(tmp_path)
    runner, output_paths = _saving_runner(audio_runner_class, fail=fail)

    if fail:
        with pytest.raises(RuntimeError, match="decoding failed"):
            runner.transcribe(audio=CLIP)
    else:
        runner.transcribe(audio=CLIP)

    assert len(output_paths) == 1
    assert not os.path.exists(os.path.dirname(os.path.abspath(output_paths[0])))
    assert list(tmp_path.iterdir()) == []
