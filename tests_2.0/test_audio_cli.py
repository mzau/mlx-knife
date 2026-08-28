"""Tests for --audio CLI argument (ADR-019 Phase 2).

Tests audio file handling in CLI without requiring actual model inference.
"""

import argparse
import json
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock
import tempfile

# Path to audio test assets
AUDIO_ASSETS = Path(__file__).parent / "assets" / "audio"


class TestAudioCLIArgument:
    """Tests for --audio CLI argument parsing and file handling."""

    def test_audio_argument_in_help(self, capsys):
        """CLI help should show --audio argument."""
        from mlxk2.cli import main
        import sys

        with pytest.raises(SystemExit) as exc_info:
            with patch.object(sys, 'argv', ['mlxk', 'run', '--help']):
                main()

        # Help exits with 0
        assert exc_info.value.code == 0
        captured = capsys.readouterr()
        assert "--audio" in captured.out

    def test_audio_help_mentions_wav(self, capsys):
        """CLI help should mention WAV format for audio."""
        from mlxk2.cli import main
        import sys

        with pytest.raises(SystemExit):
            with patch.object(sys, 'argv', ['mlxk', 'run', '--help']):
                main()

        captured = capsys.readouterr()
        assert "WAV" in captured.out or "audio" in captured.out.lower()

    def test_language_argument_in_help(self, capsys):
        """CLI help should show --language argument for audio."""
        from mlxk2.cli import main
        import sys

        with pytest.raises(SystemExit) as exc_info:
            with patch.object(sys, 'argv', ['mlxk', 'run', '--help']):
                main()

        assert exc_info.value.code == 0
        captured = capsys.readouterr()
        assert "--language" in captured.out


class TestAudioFileValidation:
    """Tests for audio file validation in CLI."""

    def test_audio_file_not_found(self, capsys):
        """Should error if audio file doesn't exist."""
        from mlxk2.cli import main
        import sys

        with pytest.raises(SystemExit) as exc_info:
            with patch.object(sys, 'argv', ['mlxk', 'run', 'test-model', '--audio', '/nonexistent/file.wav', 'prompt']):
                main()

        assert exc_info.value.code == 1
        captured = capsys.readouterr()
        assert "Audio file not found" in captured.out or "Audio file not found" in captured.err

    def test_audio_file_too_large(self, tmp_path, capsys):
        """Should error if audio file >50MB (ADR-020: limit raised for Whisper/Voxtral)."""
        from mlxk2.cli import main
        import sys

        # Create a file that's too large (just over 50MB to trigger check)
        large_file = tmp_path / "large.wav"
        # Write 51MB of zeros
        large_file.write_bytes(b'\x00' * (51 * 1024 * 1024))

        with pytest.raises(SystemExit) as exc_info:
            # Use --prompt flag to avoid argparse ambiguity with positional prompt
            with patch.object(sys, 'argv', ['mlxk', 'run', 'test-model', '--audio', str(large_file), '--prompt', 'test']):
                main()

        assert exc_info.value.code == 1
        captured = capsys.readouterr()
        assert "Audio file too large" in captured.out or "Audio file too large" in captured.err


class TestAudioCapabilityCheck:
    """Tests for audio capability detection."""

    def test_audio_without_audio_model_fails(self):
        """Should error when using --audio with non-audio model."""
        from mlxk2.operations.run import run_model

        # Pass audio to a model that doesn't exist (will fail capability check)
        result = run_model(
            model_spec="nonexistent-model-for-audio-test",
            prompt="test",
            audio=[("test.wav", b"fake audio data")],
        )

        assert result is not None
        assert "Error:" in result
        # Either "audio" capability error or model not found - both are acceptable
        assert "audio" in result.lower() or "not found" in result.lower()


class TestTranslateValidation:
    """Tests for --translate flag validation (Issue #54).

    Pre-execution rejects that do not require a real model:
    - non-English target language rejected
    - --translate without --audio rejected
    Model-dependent capability check (translate-incapable Whisper.en variant
    or non-Whisper STT) is covered by wet tests in Commit 4.
    """

    def test_translate_non_en_target_rejected(self):
        """--translate de (or any non-en target) must reject pre-execution."""
        from mlxk2.operations.run import run_model

        result = run_model(
            model_spec="nonexistent-model",
            prompt="x",
            translate="de",
        )

        assert result is not None
        assert "Error:" in result
        assert "english" in result.lower() or "'en'" in result.lower()

    def test_translate_non_en_uppercase_also_rejected(self):
        """Case insensitivity: --translate DE rejected as well."""
        from mlxk2.operations.run import run_model

        result = run_model(
            model_spec="nonexistent-model",
            prompt="x",
            translate="DE",
        )

        assert result is not None
        assert "Error:" in result
        assert "english" in result.lower() or "'en'" in result.lower()

    def test_translate_without_audio_rejected(self):
        """--translate without --audio must reject pre-execution."""
        from mlxk2.operations.run import run_model

        result = run_model(
            model_spec="nonexistent-model",
            prompt="x",
            translate="en",
        )

        assert result is not None
        assert "Error:" in result
        assert "--audio" in result or "audio" in result.lower()

    def test_translate_flag_in_help(self, capsys):
        """CLI --help should advertise --translate."""
        from mlxk2.cli import main
        import sys

        with pytest.raises(SystemExit):
            with patch.object(sys, 'argv', ['mlxk', 'run', '--help']):
                main()

        captured = capsys.readouterr()
        assert "--translate" in captured.out


class TestPromptThreading:
    """What actually reaches Whisper as `initial_prompt` (Issue #61).

    Two halves, because the bug had two. The CLI must leave the prompt slot empty
    when the user typed nothing — until #61 it filled it for every `--audio` run,
    before anything knew the backend or the task. And the run path must then apply
    the synthetic default on transcribe only, the same rule the server applies in
    core/server/handlers/audio.py. The route invariants in
    test_serve_audio_translations_route.py are the server-side twin of these.
    """

    # --- CLI half: what cli.py hands to the run path -----------------------

    def _cli_prompt(self, tmp_path, extra_argv):
        """Run `mlxk run … --audio` and return the kwargs the CLI passed on."""
        import sys
        import mlxk2.cli as cli_mod

        clip = tmp_path / "clip.wav"
        clip.write_bytes(b"RIFF\x00\x00\x00\x00WAVEfmt ")
        # Positional prompt (if any) before --audio: --audio takes one or more values.
        argv = ["mlxk", "run", "whisper-large-v3-4bit"] + extra_argv + ["--audio", str(clip)]

        with patch.object(cli_mod, "run_model_enhanced") as mock_run:
            mock_run.return_value = "TRANSCRIPT"
            with patch.object(sys, "argv", argv):
                with pytest.raises(SystemExit) as exc:
                    cli_mod.main()
        assert exc.value.code == 0
        return mock_run.call_args.kwargs

    def test_cli_leaves_prompt_empty_on_translate(self, tmp_path):
        """--translate with no positional prompt must not invent one (#61)."""
        kwargs = self._cli_prompt(tmp_path, ["--translate"])
        assert kwargs["prompt"] is None
        assert kwargs["translate"] == "en"

    def test_cli_leaves_prompt_empty_on_transcribe(self, tmp_path):
        """Plain --audio too: the default belongs to the run path, not the CLI."""
        kwargs = self._cli_prompt(tmp_path, [])
        assert kwargs["prompt"] is None
        assert kwargs["translate"] is None

    def test_cli_threads_user_prompt(self, tmp_path):
        """A positional prompt still reaches the run path untouched."""
        kwargs = self._cli_prompt(tmp_path, ["medical vocabulary"])
        assert kwargs["prompt"] == "medical vocabulary"

    # --- run path half: what run.py hands to AudioRunner -------------------

    def _transcribe_kwargs(self, tmp_path, prompt=None, translate=None):
        """Drive run_model_enhanced's MLX_AUDIO branch with a faked runner.

        `mlxk2.core.audio_runner` is injected rather than patched: importing it pulls
        mlx_audio, and the unit suite runs against the lightweight `stubs/mlx`. run.py
        imports AudioRunner inside the branch, so the injected module is what it gets.
        """
        import sys
        import types
        import mlxk2.operations.run as run_mod
        import mlxk2.operations.workspace as ws_mod
        from mlxk2.core.capabilities import Backend

        model_dir = tmp_path / "whisper-large-v3-4bit"
        model_dir.mkdir()
        (model_dir / "config.json").write_text(json.dumps({"model_type": "whisper"}))
        clip = tmp_path / "clip.wav"
        clip.write_bytes(b"RIFF\x00\x00\x00\x00WAVEfmt ")

        runner = MagicMock()
        runner.transcribe.return_value = "OUT"
        runner.__enter__.return_value = runner
        runner.__exit__.return_value = False

        fake_audio_runner = types.ModuleType("mlxk2.core.audio_runner")
        fake_audio_runner.AudioRunner = lambda *a, **k: runner

        with patch.dict(sys.modules, {"mlxk2.core.audio_runner": fake_audio_runner}), \
             patch.object(run_mod, "resolve_model_for_operation",
                          lambda spec: (str(model_dir), None, None)), \
             patch.object(ws_mod, "is_workspace_path", lambda p: True), \
             patch.object(run_mod, "detect_vision_capability", lambda *a, **k: False), \
             patch.object(run_mod, "detect_audio_capability", lambda *a, **k: True), \
             patch.object(run_mod, "detect_audio_backend", lambda *a, **k: Backend.MLX_AUDIO), \
             patch.object(run_mod, "audio_runtime_compatibility", lambda *a, **k: (True, "")), \
             patch("mlxk2.core.capabilities.detect_audio_translate_en_capability",
                   lambda *a, **k: True):
            result = run_mod.run_model_enhanced(
                model_spec="whisper-large-v3-4bit",
                prompt=prompt,
                audio=[("clip.wav", clip.read_bytes())],
                translate=translate,
                json_output=True,
            )

        assert result == "OUT", f"audio branch not reached: {result!r}"
        return runner.transcribe.call_args.kwargs

    def test_translate_drops_the_synthetic_prompt(self, tmp_path):
        """The bug itself: translate must not carry a transcription instruction."""
        kwargs = self._transcribe_kwargs(tmp_path, prompt=None, translate="en")
        assert kwargs["task"] == "translate"
        assert kwargs["prompt"] is None

    def test_transcribe_keeps_the_synthetic_prompt(self, tmp_path):
        """Non-regression, and parity with POST /v1/audio/transcriptions."""
        kwargs = self._transcribe_kwargs(tmp_path, prompt=None, translate=None)
        assert kwargs["task"] is None
        assert kwargs["prompt"] == "Transcribe this audio."

    def test_user_prompt_survives_translate(self, tmp_path):
        """An explicit prompt is an informed vocab-bias override, not a default."""
        kwargs = self._transcribe_kwargs(tmp_path, prompt="Rochefoucauld", translate="en")
        assert kwargs["prompt"] == "Rochefoucauld"

    # --- the both-at-once case: --image AND --audio -----------------------

    def _vision_kwargs(self, tmp_path, images=None, audio=None):
        """Drive run_model_enhanced's MLX_VLM branch with a faked VisionRunner."""
        import sys
        import types
        import mlxk2.operations.run as run_mod
        import mlxk2.operations.workspace as ws_mod
        from mlxk2.core.capabilities import Backend

        model_dir = tmp_path / "gemma-4-e4b-it-4bit"
        model_dir.mkdir()
        (model_dir / "config.json").write_text(json.dumps({"model_type": "gemma4"}))

        runner = MagicMock()
        runner.generate.return_value = "OUT"
        runner.__enter__.return_value = runner
        runner.__exit__.return_value = False

        fake_vision = types.ModuleType("mlxk2.core.vision_runner")
        fake_vision.VisionRunner = lambda *a, **k: runner

        with patch.dict(sys.modules, {"mlxk2.core.vision_runner": fake_vision}), \
             patch.object(run_mod, "resolve_model_for_operation",
                          lambda spec: (str(model_dir), None, None)), \
             patch.object(ws_mod, "is_workspace_path", lambda p: True), \
             patch.object(run_mod, "detect_vision_capability", lambda *a, **k: True), \
             patch.object(run_mod, "detect_audio_capability", lambda *a, **k: True), \
             patch.object(run_mod, "detect_audio_backend", lambda *a, **k: Backend.MLX_VLM), \
             patch.object(run_mod, "vision_runtime_compatibility", lambda *a, **k: (True, "")), \
             patch.object(run_mod, "check_memory_before_load", lambda *a, **k: None):
            result = run_mod.run_model_enhanced(
                model_spec="gemma-4-e4b-it-4bit",
                prompt=None,
                images=images,
                audio=audio,
                json_output=True,
            )

        assert result == "OUT", f"vision branch not reached: {result!r}"
        return runner.generate.call_args.kwargs

    def test_image_and_audio_together_keeps_the_207_default(self, tmp_path):
        """`--image X --audio Y` must still ask to transcribe, as 2.0.7 did.

        2.0.7's default lived in cli.py inside `if audios:`, so it fired whenever audio
        was present — image alongside or not. Moving it into run.py put it next to the
        images branch, where testing `images` first silently reworded this combination.
        It did, for one commit. Order is the whole content of this test.
        """
        kwargs = self._vision_kwargs(
            tmp_path,
            images=[("pic.jpg", b"\xff\xd8\xff")],
            audio=[("clip.wav", b"RIFF")],
        )
        assert kwargs["prompt"] == "Transcribe this audio."
        assert kwargs["audio"] is not None, "audio must reach the runner, not be dropped"

    def test_image_only_still_describes(self, tmp_path):
        """Non-regression on the other side of the same branch."""
        kwargs = self._vision_kwargs(tmp_path, images=[("pic.jpg", b"\xff\xd8\xff")])
        assert kwargs["prompt"] == "Describe the image."


class TestAudioTestAssets:
    """Tests to verify audio test assets are available."""

    def test_audio_assets_directory_exists(self):
        """Audio test assets directory should exist."""
        assert AUDIO_ASSETS.exists(), f"Audio assets directory not found: {AUDIO_ASSETS}"

    def test_audio_wav_files_exist(self):
        """WAV test files should be available."""
        wav_files = list(AUDIO_ASSETS.glob("*.wav"))
        assert len(wav_files) >= 1, "No WAV files found in audio assets"

    def test_sources_file_has_attribution(self):
        """sources.txt should contain license attribution."""
        sources_file = AUDIO_ASSETS / "sources.txt"
        assert sources_file.exists(), "sources.txt not found"

        content = sources_file.read_text()
        assert "CC BY 4.0" in content, "License attribution missing"
        assert "LibriSpeech" in content, "Source attribution missing"


class TestAudioBackendDetection:
    """Tests for config-based audio backend detection (ADR-020).

    Detection routes audio models to appropriate backend:
    - STT models (Voxtral, Whisper) → Backend.MLX_AUDIO
    - Multimodal models (Gemma-3n) → Backend.MLX_VLM
    """

    def test_voxtral_routes_to_mlx_audio(self, tmp_path):
        """Voxtral model_type should route to MLX_AUDIO backend."""
        from mlxk2.operations.common import detect_audio_backend
        from mlxk2.core.capabilities import Backend

        # Voxtral config (STT-focused, even with audio_config)
        config = {
            "model_type": "voxtral",
            "audio_config": {"num_mel_bins": 128},
            "vision_config": {},  # Empty (no vision)
        }

        backend = detect_audio_backend(tmp_path, config)
        assert backend == Backend.MLX_AUDIO, "Voxtral should route to MLX_AUDIO"

    def test_whisper_routes_to_mlx_audio(self, tmp_path):
        """Whisper model_type should route to MLX_AUDIO backend."""
        from mlxk2.operations.common import detect_audio_backend
        from mlxk2.core.capabilities import Backend

        config = {"model_type": "whisper"}

        backend = detect_audio_backend(tmp_path, config)
        assert backend == Backend.MLX_AUDIO, "Whisper should route to MLX_AUDIO"

    def test_gemma3n_routes_to_mlx_vlm(self, tmp_path):
        """Gemma-3n (audio + vision) should route to MLX_VLM backend."""
        from mlxk2.operations.common import detect_audio_backend
        from mlxk2.core.capabilities import Backend

        # Gemma-3n config (multimodal: vision + audio)
        config = {
            "model_type": "gemma3n",
            "audio_config": {"num_mel_bins": 80},
            "vision_config": {"image_size": 896, "patch_size": 14},  # Populated
        }

        backend = detect_audio_backend(tmp_path, config)
        assert backend == Backend.MLX_VLM, "Gemma-3n should route to MLX_VLM"

    def test_whisper_feature_extractor_routes_to_mlx_audio(self, tmp_path):
        """Models with WhisperFeatureExtractor should route to MLX_AUDIO."""
        from mlxk2.operations.common import detect_audio_backend
        from mlxk2.core.capabilities import Backend
        import json

        # Create preprocessor_config.json with WhisperFeatureExtractor
        preprocessor_config = {"feature_extractor_type": "WhisperFeatureExtractor"}
        (tmp_path / "preprocessor_config.json").write_text(json.dumps(preprocessor_config))

        # Config without explicit model_type
        config = {"hidden_size": 768}

        backend = detect_audio_backend(tmp_path, config)
        assert backend == Backend.MLX_AUDIO, "WhisperFeatureExtractor should route to MLX_AUDIO"

    def test_audio_config_only_routes_to_mlx_vlm(self, tmp_path):
        """Models with audio_config but no STT signals route to MLX_VLM (fallback)."""
        from mlxk2.operations.common import detect_audio_backend
        from mlxk2.core.capabilities import Backend

        # Unknown audio model with just audio_config
        config = {
            "model_type": "unknown_audio_model",
            "audio_config": {"sample_rate": 16000},
        }

        backend = detect_audio_backend(tmp_path, config)
        assert backend == Backend.MLX_VLM, "audio_config alone should fallback to MLX_VLM"

    def test_no_audio_config_returns_none(self, tmp_path):
        """Models without audio_config should return None."""
        from mlxk2.operations.common import detect_audio_backend

        # Pure text model
        config = {"model_type": "llama", "hidden_size": 4096}

        backend = detect_audio_backend(tmp_path, config)
        assert backend is None, "Non-audio model should return None"

    def test_name_heuristic_whisper(self, tmp_path):
        """Fallback name heuristic: 'whisper' in name routes to MLX_AUDIO."""
        from mlxk2.operations.common import detect_audio_backend
        from mlxk2.core.capabilities import Backend

        # Create probe path with "whisper" in name
        whisper_path = tmp_path / "whisper-large-v3-turbo-4bit"
        whisper_path.mkdir()

        config = {"hidden_size": 768}  # No model_type, no audio_config

        backend = detect_audio_backend(whisper_path, config)
        assert backend == Backend.MLX_AUDIO, "Name heuristic should detect whisper"

    def test_original_voxtral_no_vision_config(self, tmp_path):
        """Original Mistral Voxtral (no vision_config key) routes to MLX_AUDIO."""
        from mlxk2.operations.common import detect_audio_backend
        from mlxk2.core.capabilities import Backend

        # Original Mistral format (no vision_config key at all)
        config = {
            "model_type": "voxtral",
            "audio_config": {"encoder_config": {"num_mel_bins": 128}},
        }

        backend = detect_audio_backend(tmp_path, config)
        assert backend == Backend.MLX_AUDIO, "Original Voxtral should route to MLX_AUDIO"


class TestAudioRuntimeCompatibility:
    """Tests for audio runtime compatibility check (ADR-020)."""

    def test_mlx_audio_backend_checks_mlx_audio(self):
        """MLX_AUDIO backend should check for mlx-audio package."""
        from mlxk2.operations.common import audio_runtime_compatibility
        from mlxk2.core.capabilities import Backend
        import importlib.util

        # Skip if mlx-audio not installed (PyPI #442: [audio] extra is empty)
        if importlib.util.find_spec("mlx_audio") is None:
            pytest.skip("mlx-audio not installed (requires manual editable install)")

        # MLX_AUDIO backend (Whisper, Voxtral)
        compatible, reason = audio_runtime_compatibility(Backend.MLX_AUDIO)

        # Should be compatible when mlx-audio is installed
        assert compatible is True, f"Expected mlx-audio to be available: {reason}"
        assert reason is None

    def test_mlx_vlm_backend_checks_mlx_vlm(self):
        """MLX_VLM backend should check for mlx-vlm package."""
        from mlxk2.operations.common import audio_runtime_compatibility
        from mlxk2.core.capabilities import Backend

        # MLX_VLM backend (Gemma-3n multimodal)
        compatible, reason = audio_runtime_compatibility(Backend.MLX_VLM)

        # Should be compatible if mlx-vlm is installed
        assert compatible is True, f"Expected mlx-vlm to be available: {reason}"
        assert reason is None
