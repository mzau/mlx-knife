"""
Live E2E tests for Audio functionality (ADR-019).

Tests deterministic audio transcription with specific, verifiable content
to validate actual audio understanding (not just hallucination).

Requires:
- Python 3.10+ (mlx-vlm requirement)
- Audio model in cache (e.g., gemma-3n-E2B-it-4bit)
- Test assets in tests_2.0/assets/audio/
- HF_HOME set to model cache location

Run with:
    HF_HOME=/path/to/cache pytest -m live_e2e tests_2.0/live/test_audio_e2e_live.py -v

Known limitations (ADR-019):
- Audio duration limit: ~30 seconds (Gemma-3n architecture constraint)
- Phonetic errors on 4-bit models: expected, validate content not exact text
- Temperature 0.2 default for audio (stability vs text 0.7)

Architecture:
- Uses audio_portfolio discovery (no hardcoded models)
- Parametrized via audio_model_key fixture
- Follows Portfolio Separation pattern (like vision tests)
"""
import os
import sys
import pytest
import subprocess
from pathlib import Path

# Use the Python interpreter from the test environment
PYTHON = sys.executable

# Audio support requires Python 3.10+ (mlx-vlm requirement)
pytestmark = [
    pytest.mark.live,
    pytest.mark.live_e2e,
    pytest.mark.skipif(
        sys.version_info < (3, 10),
        reason="Audio support requires Python 3.10+ (mlx-vlm dependency)"
    )
]

# Path to audio test assets
AUDIO_ASSETS = Path(__file__).parent.parent / "assets" / "audio"


class TestAudioTranscription:
    """
    Portfolio-based audio transcription tests.

    Uses audio_model_key parametrization from conftest.py to run
    against all audio-capable models in the cache.

    Tests use deterministic audio clips with known content
    to validate actual audio understanding. Due to STT limitations
    (especially on 4-bit models), we verify key phrases rather than
    exact transcription.
    """

    def test_transcribe_short_audio_wav(self, audio_model_info, audio_model_key):
        """Test transcription of short audio clip (WAV format).

        Audio: "A man said to the universe, Sir I exist"
        Validates: Key content words present (man, universe, exist)

        Note: 4-bit models may produce phonetic errors (e.g., "Amen" for "A man")
        so we check for presence of key semantic content.
        """
        if audio_model_key == "_skipped":
            pytest.skip("Run with -m live_e2e or -m wet")
        if audio_model_key == "_no_audio_models":
            pytest.skip("No audio models found in cache")

        model_id = audio_model_info["id"]
        audio_file = AUDIO_ASSETS / "A MAN SAID TO THE UNIVERSE SIR I EXIST.wav"

        if not audio_file.exists():
            pytest.skip(f"Audio asset not found: {audio_file}")

        result = subprocess.run(
            [
                PYTHON, "-m", "mlxk2.cli", "run", model_id,
                "Transcribe this audio.",
                "--audio", str(audio_file),
                "--max-tokens", "100",
                "--temperature", "0",  # Most stable for transcription
                "--no-stream"
            ],
            capture_output=True,
            text=True,
            timeout=180,
            env=os.environ,
        )
        assert result.returncode == 0, f"Command failed for {model_id}: {result.stderr}"
        output = result.stdout.strip().lower()

        # Key semantic content must be present (allowing for phonetic errors)
        # "A man" might become "Amen", but "universe" and "exist" should be clear
        assert "universe" in output or "sir" in output or "exist" in output, \
            f"Expected 'universe', 'sir', or 'exist' in transcription for {model_id}: {result.stdout}"

    def test_transcribe_longer_audio_wav(self, audio_model_info, audio_model_key):
        """Test transcription of longer audio clip (~14 seconds, WAV).

        Audio: "Having returned to the royal cavern, Kaliko first pounded
               the gong and then sat in the throne wearing Ruggedo's
               discarded ruby crown..."

        Validates: Key content words (royal, cavern, throne, crown, gong)
        """
        if audio_model_key == "_skipped":
            pytest.skip("Run with -m live_e2e or -m wet")
        if audio_model_key == "_no_audio_models":
            pytest.skip("No audio models found in cache")

        model_id = audio_model_info["id"]
        audio_file = AUDIO_ASSETS / "HAVING RETURNED TO THE ROYAL CAVERN.wav"

        if not audio_file.exists():
            pytest.skip(f"Audio asset not found: {audio_file}")

        result = subprocess.run(
            [
                PYTHON, "-m", "mlxk2.cli", "run", model_id,
                "Transcribe this audio.",
                "--audio", str(audio_file),
                "--max-tokens", "200",
                "--temperature", "0",
                "--no-stream"
            ],
            capture_output=True,
            text=True,
            timeout=180,
            env=os.environ,
        )
        assert result.returncode == 0, f"Command failed for {model_id}: {result.stderr}"
        output = result.stdout.strip().lower()

        # Check for distinctive words from the passage
        # At least 2 of these should be present for a valid transcription
        key_words = ["royal", "cavern", "throne", "crown", "gong", "kaliko", "ruggedo"]
        found_words = [w for w in key_words if w in output]

        assert len(found_words) >= 2, \
            f"Expected at least 2 of {key_words} in transcription for {model_id}, found {found_words}: {result.stdout}"

    def test_transcribe_mp3_format(self, audio_model_info, audio_model_key):
        """Test that MP3 format is also supported.

        Same audio as WAV test but in MP3 format.
        """
        if audio_model_key == "_skipped":
            pytest.skip("Run with -m live_e2e or -m wet")
        if audio_model_key == "_no_audio_models":
            pytest.skip("No audio models found in cache")

        model_id = audio_model_info["id"]
        audio_file = AUDIO_ASSETS / "A MAN SAID TO THE UNIVERSE SIR I EXIST.mp3"

        if not audio_file.exists():
            pytest.skip(f"Audio asset not found: {audio_file}")

        result = subprocess.run(
            [
                PYTHON, "-m", "mlxk2.cli", "run", model_id,
                "Transcribe this audio.",
                "--audio", str(audio_file),
                "--max-tokens", "100",
                "--temperature", "0",
                "--no-stream"
            ],
            capture_output=True,
            text=True,
            timeout=180,
            env=os.environ,
        )
        assert result.returncode == 0, f"Command failed for {model_id}: {result.stderr}"
        output = result.stdout.strip().lower()

        # MP3 format support: same validation as WAV test
        # Simple prompt avoids multilingual drift issue with complex prompts
        assert "universe" in output or "sir" in output or "exist" in output, \
            f"Expected 'universe', 'sir', or 'exist' in MP3 transcription for {model_id}: {result.stdout}"

    def test_audio_output_not_empty(self, audio_model_info, audio_model_key):
        """Basic sanity test: audio transcription produces non-trivial output.

        Validates that the model actually processes the audio and generates
        a meaningful response (not just empty or single-word output).
        """
        if audio_model_key == "_skipped":
            pytest.skip("Run with -m live_e2e or -m wet")
        if audio_model_key == "_no_audio_models":
            pytest.skip("No audio models found in cache")

        model_id = audio_model_info["id"]
        audio_file = AUDIO_ASSETS / "A MAN SAID TO THE UNIVERSE SIR I EXIST.wav"

        if not audio_file.exists():
            pytest.skip(f"Audio asset not found: {audio_file}")

        result = subprocess.run(
            [
                PYTHON, "-m", "mlxk2.cli", "run", model_id,
                "Transcribe this audio.",
                "--audio", str(audio_file),
                "--max-tokens", "100",
                "--temperature", "0",
                "--no-stream"
            ],
            capture_output=True,
            text=True,
            timeout=180,
            env=os.environ,
        )

        assert result.returncode == 0, f"Command failed for {model_id}: {result.stderr}"

        # Basic sanity: output should have some content (more than just whitespace or a few chars)
        output = result.stdout.strip()
        assert len(output) > 10, f"Transcription too short for {model_id}: '{output}'"
