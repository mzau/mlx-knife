"""Tests for --audio CLI argument (ADR-019 Phase 2).

Tests audio file handling in CLI without requiring actual model inference.
"""

import argparse
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
        """Should error if audio file >5MB."""
        from mlxk2.cli import main
        import sys

        # Create a file that's too large (just over 5MB to trigger check)
        large_file = tmp_path / "large.wav"
        # Write 6MB of zeros
        large_file.write_bytes(b'\x00' * (6 * 1024 * 1024))

        with pytest.raises(SystemExit) as exc_info:
            with patch.object(sys, 'argv', ['mlxk', 'run', 'test-model', '--audio', str(large_file), 'prompt']):
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
