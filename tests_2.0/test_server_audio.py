"""
Tests for server audio API support (ADR-019 Phase 4).

Tests the OpenAI-compatible input_audio content blocks in /v1/chat/completions.
"""

import base64
import pytest

from mlxk2.core.server_base import ChatMessage, _request_has_audio
from mlxk2.tools.vision_adapter import (
    VisionHTTPAdapter,
    MAX_AUDIO_SIZE_BYTES,
    SUPPORTED_AUDIO_FORMATS,
)


class TestRequestHasAudio:
    """Tests for _request_has_audio() detection function."""

    def test_string_content_no_audio(self):
        """String content messages have no audio."""
        messages = [ChatMessage(role="user", content="Hello")]
        assert _request_has_audio(messages) is False

    def test_list_content_text_only_no_audio(self):
        """List content with only text has no audio."""
        messages = [
            ChatMessage(
                role="user",
                content=[{"type": "text", "text": "Hello"}],
            )
        ]
        assert _request_has_audio(messages) is False

    def test_list_content_with_audio(self):
        """List content with input_audio has audio."""
        messages = [
            ChatMessage(
                role="user",
                content=[
                    {"type": "text", "text": "Transcribe this"},
                    {
                        "type": "input_audio",
                        "input_audio": {"data": "base64data", "format": "wav"},
                    },
                ],
            )
        ]
        assert _request_has_audio(messages) is True

    def test_multiple_messages_audio_in_last_user_only(self):
        """Only last user message is checked for audio."""
        messages = [
            ChatMessage(
                role="user",
                content=[
                    {
                        "type": "input_audio",
                        "input_audio": {"data": "old", "format": "wav"},
                    }
                ],
            ),
            ChatMessage(role="assistant", content="Transcription..."),
            ChatMessage(role="user", content="What did you hear?"),
        ]
        # Last user message is text-only
        assert _request_has_audio(messages) is False

    def test_image_only_no_audio(self):
        """Image content is not audio."""
        messages = [
            ChatMessage(
                role="user",
                content=[
                    {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,..."}}
                ],
            )
        ]
        assert _request_has_audio(messages) is False


class TestDecodeBase64Audio:
    """Tests for VisionHTTPAdapter.decode_base64_audio()."""

    def test_decodes_wav_format(self):
        """WAV format is supported."""
        audio_bytes = b"RIFF....WAVEfmt "  # Fake WAV header
        b64_data = base64.b64encode(audio_bytes).decode()

        filename, raw_bytes = VisionHTTPAdapter.decode_base64_audio(b64_data, "wav")

        assert filename.startswith("audio_")
        assert filename.endswith(".wav")
        assert raw_bytes == audio_bytes

    def test_decodes_mp3_format(self):
        """MP3 format is supported."""
        audio_bytes = b"\xff\xfb\x90\x00"  # Fake MP3 header
        b64_data = base64.b64encode(audio_bytes).decode()

        filename, raw_bytes = VisionHTTPAdapter.decode_base64_audio(b64_data, "mp3")

        assert filename.endswith(".mp3")
        assert raw_bytes == audio_bytes

    def test_normalizes_mpeg_to_mp3(self):
        """MPEG format is normalized to mp3."""
        audio_bytes = b"\xff\xfb\x90\x00"
        b64_data = base64.b64encode(audio_bytes).decode()

        filename, _ = VisionHTTPAdapter.decode_base64_audio(b64_data, "mpeg")

        assert filename.endswith(".mp3")

    def test_rejects_unsupported_format(self):
        """Unsupported formats raise ValueError."""
        audio_bytes = b"audio data"
        b64_data = base64.b64encode(audio_bytes).decode()

        with pytest.raises(ValueError, match="Unsupported audio format"):
            VisionHTTPAdapter.decode_base64_audio(b64_data, "ogg")

    def test_rejects_invalid_base64(self):
        """Invalid base64 raises ValueError."""
        with pytest.raises(ValueError, match="Failed to decode base64"):
            VisionHTTPAdapter.decode_base64_audio("not-valid-base64!!!", "wav")

    def test_rejects_empty_data(self):
        """Empty audio data raises ValueError."""
        b64_data = base64.b64encode(b"").decode()

        with pytest.raises(ValueError, match="empty"):
            VisionHTTPAdapter.decode_base64_audio(b64_data, "wav")

    def test_rejects_oversized_audio(self):
        """Audio exceeding size limit raises ValueError."""
        # Create data larger than limit
        audio_bytes = b"x" * (MAX_AUDIO_SIZE_BYTES + 1)
        b64_data = base64.b64encode(audio_bytes).decode()

        with pytest.raises(ValueError, match="exceeds limit"):
            VisionHTTPAdapter.decode_base64_audio(b64_data, "wav")

    def test_generates_deterministic_filename(self):
        """Same content always generates same filename."""
        audio_bytes = b"test audio content"
        b64_data = base64.b64encode(audio_bytes).decode()

        filename1, _ = VisionHTTPAdapter.decode_base64_audio(b64_data, "wav")
        filename2, _ = VisionHTTPAdapter.decode_base64_audio(b64_data, "wav")

        assert filename1 == filename2


class TestParseOpenAIMessagesAudio:
    """Tests for VisionHTTPAdapter.parse_openai_messages() with audio."""

    def test_parses_audio_only_request(self):
        """Audio-only request extracts audio tuple."""
        audio_bytes = b"audio content"
        b64_data = base64.b64encode(audio_bytes).decode()

        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "input_audio",
                        "input_audio": {"data": b64_data, "format": "wav"},
                    }
                ],
            }
        ]

        prompt, images, audio = VisionHTTPAdapter.parse_openai_messages(messages)

        assert prompt == "Transcribe what is spoken in this audio."  # Default prompt
        assert images == []
        assert len(audio) == 1
        assert audio[0][1] == audio_bytes

    def test_parses_audio_with_text(self):
        """Audio with text prompt preserves text."""
        audio_bytes = b"audio content"
        b64_data = base64.b64encode(audio_bytes).decode()

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "What language is spoken?"},
                    {
                        "type": "input_audio",
                        "input_audio": {"data": b64_data, "format": "wav"},
                    },
                ],
            }
        ]

        prompt, images, audio = VisionHTTPAdapter.parse_openai_messages(messages)

        assert prompt == "What language is spoken?"
        assert len(audio) == 1

    def test_parses_image_and_audio_combined(self):
        """Both image and audio can be in same message."""
        audio_bytes = b"audio"
        audio_b64 = base64.b64encode(audio_bytes).decode()
        image_bytes = b"\xff\xd8\xff\xe0"  # JPEG magic
        image_b64 = base64.b64encode(image_bytes).decode()

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Describe both"},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"}},
                    {"type": "input_audio", "input_audio": {"data": audio_b64, "format": "wav"}},
                ],
            }
        ]

        prompt, images, audio = VisionHTTPAdapter.parse_openai_messages(messages)

        assert prompt == "Describe both"
        assert len(images) == 1
        assert len(audio) == 1

    def test_rejects_empty_audio_data(self):
        """Empty input_audio.data raises ValueError."""
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "input_audio",
                        "input_audio": {"data": "", "format": "wav"},
                    }
                ],
            }
        ]

        with pytest.raises(ValueError, match="cannot be empty"):
            VisionHTTPAdapter.parse_openai_messages(messages)

    def test_rejects_missing_input_audio_dict(self):
        """input_audio without dict raises ValueError."""
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "input_audio", "input_audio": "not a dict"},
                ],
            }
        ]

        with pytest.raises(ValueError, match="must have 'input_audio' dict"):
            VisionHTTPAdapter.parse_openai_messages(messages)


class TestSupportedAudioFormats:
    """Tests for audio format constants."""

    def test_wav_supported(self):
        assert "wav" in SUPPORTED_AUDIO_FORMATS

    def test_mp3_supported(self):
        assert "mp3" in SUPPORTED_AUDIO_FORMATS

    def test_mpeg_supported(self):
        """MPEG is normalized to mp3 in decode, but listed for validation."""
        assert "mpeg" in SUPPORTED_AUDIO_FORMATS


class TestFilterMultimodalHistoryAudio:
    """Tests for audio in multimodal history filtering."""

    def test_filters_audio_from_history(self):
        """Audio content is replaced with placeholder for text models."""
        from mlxk2.core.server_base import _filter_multimodal_history_for_text_models

        messages = [
            ChatMessage(
                role="user",
                content=[
                    {"type": "text", "text": "Transcribe this"},
                    {"type": "input_audio", "input_audio": {"data": "...", "format": "wav"}},
                ],
            ),
            ChatMessage(role="assistant", content="The audio says..."),
        ]

        filtered = _filter_multimodal_history_for_text_models(messages)

        assert len(filtered) == 2
        assert "[1 audio(s) were attached]" in filtered[0].content
        assert "Transcribe this" in filtered[0].content

    def test_filters_mixed_image_and_audio(self):
        """Both images and audio are counted in placeholder."""
        from mlxk2.core.server_base import _filter_multimodal_history_for_text_models

        messages = [
            ChatMessage(
                role="user",
                content=[
                    {"type": "text", "text": "Describe"},
                    {"type": "image_url", "image_url": {"url": "data:..."}},
                    {"type": "input_audio", "input_audio": {"data": "...", "format": "wav"}},
                ],
            ),
        ]

        filtered = _filter_multimodal_history_for_text_models(messages)

        assert "1 image(s)" in filtered[0].content
        assert "1 audio(s)" in filtered[0].content
