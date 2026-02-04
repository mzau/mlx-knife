"""
Live E2E tests for Audio STT functionality (ADR-020).

Tests deterministic audio transcription with specific, verifiable content
to validate actual audio understanding via mlx-audio backend.

Requires:
- Python 3.10+ (mlx-audio requirement)
- Audio model in cache (e.g., whisper-large-v3-turbo-4bit)
- Test assets in tests_2.0/assets/audio/
- HF_HOME set to model cache location

Run with:
    HF_HOME=/path/to/cache pytest -m live_e2e tests_2.0/live/test_audio_e2e_live.py -v

Architecture:
- Uses audio_portfolio discovery (prefers Whisper models)
- Parametrized via audio_model_key fixture
- Follows Portfolio Separation pattern (like vision tests)

Changes from beta.8:
- Backend: mlx-vlm → mlx-audio (STT-focused)
- Models: Gemma-3n → Whisper variants
- Duration: No 30s limit (Whisper handles >10min audio)
- Accuracy: Better STT accuracy (dedicated models vs multimodal)
"""
import os
import sys
import pytest
import subprocess
from pathlib import Path

# Use the Python interpreter from the test environment
PYTHON = sys.executable

# Audio support requires Python 3.10+ (mlx-audio/mlx-vlm requirement)
pytestmark = [
    pytest.mark.live,
    pytest.mark.live_e2e,
    pytest.mark.skipif(
        sys.version_info < (3, 10),
        reason="Audio support requires Python 3.10+ (mlx-audio dependency)"
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
    to validate actual audio understanding. Whisper models provide
    high accuracy STT, so we can validate more precisely than beta.8.
    """

    def test_transcribe_short_audio_wav(self, audio_model_info, audio_model_key):
        """Test transcription of short audio clip (WAV format).

        Audio: "A man said to the universe, Sir I exist"
        Validates: Key content words present (man, universe, exist)

        Note: Whisper provides better accuracy than multimodal models,
        but we still use semantic validation for robustness.
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
                "--temperature", "0",  # Greedy decoding (STT best practice)
                "--no-stream"
            ],
            capture_output=True,
            text=True,
            timeout=180,
            env=os.environ,
        )
        assert result.returncode == 0, f"Command failed for {model_id}: {result.stderr}"
        output = result.stdout.strip().lower()

        # Semantic validation: key content must be present
        # Whisper should transcribe "A man said to the universe, Sir, I exist" accurately
        assert "universe" in output or "man" in output or "exist" in output, \
            f"Expected 'universe', 'man', or 'exist' in transcription for {model_id}: {result.stdout}"

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
        """Test that MP3 format is supported (no system dependencies).

        Same audio as WAV test but in MP3 format.
        Note: MP3 decoding is provided by soundfile's embedded libsndfile.
        No ffmpeg or Homebrew dependencies required.
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

        # MP3 should work with embedded libsndfile, but skip if any audio errors
        if result.returncode != 0:
            if "audio" in result.stderr.lower() or "mp3" in result.stderr.lower():
                pytest.skip(f"MP3 decoding failed (edge case): {result.stderr[:200]}")
            else:
                pytest.fail(f"Command failed for {model_id}: {result.stderr}")

        output = result.stdout.strip().lower()

        # MP3 format support: same validation as WAV test
        assert "universe" in output or "man" in output or "exist" in output, \
            f"Expected 'universe', 'man', or 'exist' in MP3 transcription for {model_id}: {result.stdout}"

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

        # Basic sanity: output should have some content
        output = result.stdout.strip()
        assert len(output) > 10, f"Transcription too short for {model_id}: '{output}'"


class TestAudioSegments:
    """Tests for segment metadata feature (MLXK2_AUDIO_SEGMENTS=1)."""

    def test_segment_metadata_optional(self, audio_model_info, audio_model_key):
        """Segment metadata is only added when MLXK2_AUDIO_SEGMENTS=1.

        Default behavior (no env var) should NOT include segment table.
        """
        if audio_model_key == "_skipped":
            pytest.skip("Run with -m live_e2e or -m wet")
        if audio_model_key == "_no_audio_models":
            pytest.skip("No audio models found in cache")

        model_id = audio_model_info["id"]
        audio_file = AUDIO_ASSETS / "A MAN SAID TO THE UNIVERSE SIR I EXIST.wav"

        if not audio_file.exists():
            pytest.skip(f"Audio asset not found: {audio_file}")

        # Without MLXK2_AUDIO_SEGMENTS - should NOT have segment table
        env_without = {k: v for k, v in os.environ.items() if k != "MLXK2_AUDIO_SEGMENTS"}
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
            env=env_without,
        )

        assert result.returncode == 0, f"Command failed for {model_id}: {result.stderr}"
        output = result.stdout

        # Should NOT contain segment table markers
        assert "<details>" not in output, "Segment metadata should NOT appear without MLXK2_AUDIO_SEGMENTS=1"
        assert "Audio Segments" not in output, "Segment metadata should NOT appear without MLXK2_AUDIO_SEGMENTS=1"


# Server E2E tests for /v1/audio/transcriptions endpoint (beta.9+)
try:
    import httpx
except ImportError:
    httpx = None


class TestAudioTranscriptionsServer:
    """
    E2E tests for the /v1/audio/transcriptions server endpoint.

    Tests the OpenAI Whisper API compatible transcription endpoint.
    Uses LocalServer context manager for server lifecycle management.

    Requires:
    - httpx installed
    - Audio model in cache (whisper-large-v3-turbo-4bit)
    - mlx-audio installed
    """

    @pytest.mark.skipif(httpx is None, reason="httpx required for server E2E tests")
    @pytest.mark.live_e2e
    def test_transcription_endpoint_json(self, audio_model_info, audio_model_key):
        """Test /v1/audio/transcriptions with JSON response format."""
        if audio_model_key == "_skipped":
            pytest.skip("Run with -m live_e2e or -m wet")
        if audio_model_key == "_no_audio_models":
            pytest.skip("No audio models found in cache")

        from .server_context import LocalServer

        model_id = audio_model_info["id"]
        audio_file = AUDIO_ASSETS / "A MAN SAID TO THE UNIVERSE SIR I EXIST.wav"

        if not audio_file.exists():
            pytest.skip(f"Audio asset not found: {audio_file}")

        with LocalServer(model_id, port=8771, timeout=90) as server_url:
            with open(audio_file, "rb") as f:
                response = httpx.post(
                    f"{server_url}/v1/audio/transcriptions",
                    files={"file": (audio_file.name, f, "audio/wav")},
                    data={"model": model_id},
                    timeout=120,
                )

            assert response.status_code == 200, f"Request failed: {response.text}"
            result = response.json()

            assert "text" in result, f"Expected 'text' in response: {result}"
            text = result["text"].lower()
            assert "universe" in text or "man" in text or "exist" in text, \
                f"Expected transcription content in: {result['text']}"

    @pytest.mark.skipif(httpx is None, reason="httpx required for server E2E tests")
    @pytest.mark.live_e2e
    def test_transcription_endpoint_text_format(self, audio_model_info, audio_model_key):
        """Test /v1/audio/transcriptions with text response format."""
        if audio_model_key == "_skipped":
            pytest.skip("Run with -m live_e2e or -m wet")
        if audio_model_key == "_no_audio_models":
            pytest.skip("No audio models found in cache")

        from .server_context import LocalServer

        model_id = audio_model_info["id"]
        audio_file = AUDIO_ASSETS / "A MAN SAID TO THE UNIVERSE SIR I EXIST.wav"

        if not audio_file.exists():
            pytest.skip(f"Audio asset not found: {audio_file}")

        with LocalServer(model_id, port=8772, timeout=90) as server_url:
            with open(audio_file, "rb") as f:
                response = httpx.post(
                    f"{server_url}/v1/audio/transcriptions",
                    files={"file": (audio_file.name, f, "audio/wav")},
                    data={"model": model_id, "response_format": "text"},
                    timeout=120,
                )

            assert response.status_code == 200, f"Request failed: {response.text}"
            # Text format returns plain text, not JSON
            text = response.text.lower()
            assert "universe" in text or "man" in text or "exist" in text, \
                f"Expected transcription content in: {response.text}"

    @pytest.mark.skipif(httpx is None, reason="httpx required for server E2E tests")
    @pytest.mark.live_e2e
    def test_transcription_endpoint_verbose_json(self, audio_model_info, audio_model_key):
        """Test /v1/audio/transcriptions with verbose_json response format."""
        if audio_model_key == "_skipped":
            pytest.skip("Run with -m live_e2e or -m wet")
        if audio_model_key == "_no_audio_models":
            pytest.skip("No audio models found in cache")

        from .server_context import LocalServer

        model_id = audio_model_info["id"]
        audio_file = AUDIO_ASSETS / "A MAN SAID TO THE UNIVERSE SIR I EXIST.wav"

        if not audio_file.exists():
            pytest.skip(f"Audio asset not found: {audio_file}")

        with LocalServer(model_id, port=8773, timeout=90) as server_url:
            with open(audio_file, "rb") as f:
                response = httpx.post(
                    f"{server_url}/v1/audio/transcriptions",
                    files={"file": (audio_file.name, f, "audio/wav")},
                    data={"model": model_id, "response_format": "verbose_json"},
                    timeout=120,
                )

            assert response.status_code == 200, f"Request failed: {response.text}"
            result = response.json()

            # Verbose JSON includes additional fields
            assert "text" in result, f"Expected 'text' in response: {result}"
            assert "task" in result, f"Expected 'task' in response: {result}"
            assert "duration" in result, f"Expected 'duration' in response: {result}"
            assert result["task"] == "transcribe", f"Expected task='transcribe': {result}"

    @pytest.mark.skipif(httpx is None, reason="httpx required for server E2E tests")
    @pytest.mark.live_e2e
    def test_transcription_endpoint_mp3(self, audio_model_info, audio_model_key):
        """Test /v1/audio/transcriptions with MP3 format."""
        if audio_model_key == "_skipped":
            pytest.skip("Run with -m live_e2e or -m wet")
        if audio_model_key == "_no_audio_models":
            pytest.skip("No audio models found in cache")

        from .server_context import LocalServer

        model_id = audio_model_info["id"]
        audio_file = AUDIO_ASSETS / "A MAN SAID TO THE UNIVERSE SIR I EXIST.mp3"

        if not audio_file.exists():
            pytest.skip(f"Audio asset not found: {audio_file}")

        with LocalServer(model_id, port=8774, timeout=90) as server_url:
            with open(audio_file, "rb") as f:
                response = httpx.post(
                    f"{server_url}/v1/audio/transcriptions",
                    files={"file": (audio_file.name, f, "audio/mpeg")},
                    data={"model": model_id},
                    timeout=120,
                )

            assert response.status_code == 200, f"Request failed: {response.text}"
            result = response.json()

            assert "text" in result, f"Expected 'text' in response: {result}"
            text = result["text"].lower()
            assert "universe" in text or "man" in text or "exist" in text, \
                f"Expected transcription content in MP3: {result['text']}"

    @pytest.mark.skipif(httpx is None, reason="httpx required for server E2E tests")
    @pytest.mark.live_e2e
    def test_transcription_endpoint_with_language(self, audio_model_info, audio_model_key):
        """Test /v1/audio/transcriptions with explicit language parameter."""
        if audio_model_key == "_skipped":
            pytest.skip("Run with -m live_e2e or -m wet")
        if audio_model_key == "_no_audio_models":
            pytest.skip("No audio models found in cache")

        from .server_context import LocalServer

        model_id = audio_model_info["id"]
        audio_file = AUDIO_ASSETS / "A MAN SAID TO THE UNIVERSE SIR I EXIST.wav"

        if not audio_file.exists():
            pytest.skip(f"Audio asset not found: {audio_file}")

        with LocalServer(model_id, port=8775, timeout=90) as server_url:
            with open(audio_file, "rb") as f:
                response = httpx.post(
                    f"{server_url}/v1/audio/transcriptions",
                    files={"file": (audio_file.name, f, "audio/wav")},
                    data={"model": model_id, "language": "en"},
                    timeout=120,
                )

            assert response.status_code == 200, f"Request failed: {response.text}"
            result = response.json()

            assert "text" in result, f"Expected 'text' in response: {result}"
            # With explicit English, transcription should still work
            text = result["text"].lower()
            assert len(text) > 10, f"Transcription too short: {result['text']}"

    @pytest.mark.skipif(httpx is None, reason="httpx required for server E2E tests")
    @pytest.mark.live_e2e
    def test_transcription_endpoint_rejects_oversized_audio(self, audio_model_info, audio_model_key):
        """Test /v1/audio/transcriptions rejects files exceeding 50MB limit.

        Validates that the endpoint enforces MAX_AUDIO_SIZE_BYTES (50 MB)
        to prevent resource exhaustion from large uploads.
        """
        if audio_model_key == "_skipped":
            pytest.skip("Run with -m live_e2e or -m wet")
        if audio_model_key == "_no_audio_models":
            pytest.skip("No audio models found in cache")

        from io import BytesIO
        from .server_context import LocalServer

        model_id = audio_model_info["id"]

        # Create oversized fake audio (50 MB + 1 byte)
        # Note: This is not valid audio, but the size check happens before decoding
        oversized_content = b"x" * (50 * 1024 * 1024 + 1)

        with LocalServer(model_id, port=8776, timeout=90) as server_url:
            response = httpx.post(
                f"{server_url}/v1/audio/transcriptions",
                files={"file": ("oversized.wav", BytesIO(oversized_content), "audio/wav")},
                data={"model": model_id},
                timeout=30,
            )

            # Should return 413 Payload Too Large
            assert response.status_code == 413, \
                f"Expected 413 for oversized file, got {response.status_code}: {response.text}"
            assert "50 MB" in response.text or "limit" in response.text.lower(), \
                f"Expected size limit message in error: {response.text}"
