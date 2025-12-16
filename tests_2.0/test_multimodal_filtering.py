"""Unit tests for multimodal message filtering (Vision→Text model switch).

Tests the server-side filtering logic that enables nChat users to switch
from Vision models to Text models while preserving conversation history.

See: docs/ISSUES/VISION-MULTIMODAL-HISTORY-ISSUE.md
"""

import pytest
from mlxk2.core.server_base import _filter_multimodal_history_for_text_models, ChatMessage


class TestMultimodalMessageFiltering:
    """Test multimodal content filtering for text-only models."""

    def test_string_content_passthrough(self):
        """String content should pass through unchanged."""
        messages = [
            ChatMessage(role="user", content="Hello"),
            ChatMessage(role="assistant", content="Hi there"),
        ]

        filtered = _filter_multimodal_history_for_text_models(messages)

        assert len(filtered) == 2
        assert filtered[0].content == "Hello"
        assert filtered[1].content == "Hi there"

    def test_single_image_with_text(self):
        """Text + single image should extract text and add placeholder."""
        messages = [
            ChatMessage(
                role="user",
                content=[
                    {"type": "text", "text": "Describe this image"},
                    {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,..."}}
                ]
            ),
        ]

        filtered = _filter_multimodal_history_for_text_models(messages)

        assert len(filtered) == 1
        assert filtered[0].role == "user"
        assert "Describe this image" in filtered[0].content
        assert "[1 image(s) were attached]" in filtered[0].content

    def test_multiple_images_with_text(self):
        """Text + multiple images should count images correctly."""
        messages = [
            ChatMessage(
                role="user",
                content=[
                    {"type": "text", "text": "What do these show?"},
                    {"type": "image_url", "image_url": {"url": "data:..."}},
                    {"type": "image_url", "image_url": {"url": "data:..."}},
                    {"type": "image_url", "image_url": {"url": "data:..."}},
                ]
            ),
        ]

        filtered = _filter_multimodal_history_for_text_models(messages)

        assert len(filtered) == 1
        assert "What do these show?" in filtered[0].content
        assert "[3 image(s) were attached]" in filtered[0].content

    def test_images_only_no_text(self):
        """Images without text should only show placeholder."""
        messages = [
            ChatMessage(
                role="user",
                content=[
                    {"type": "image_url", "image_url": {"url": "data:..."}},
                    {"type": "image_url", "image_url": {"url": "data:..."}},
                ]
            ),
        ]

        filtered = _filter_multimodal_history_for_text_models(messages)

        assert len(filtered) == 1
        assert filtered[0].content == "[2 image(s) were attached]"

    def test_vision_to_text_model_switch_scenario(self):
        """Real-world scenario: Vision → Text model switch with conversation history."""
        # Simulate conversation history from nChat
        messages = [
            # User's initial prompt with images (Vision model)
            ChatMessage(
                role="user",
                content=[
                    {"type": "text", "text": "What are these pictures showing?"},
                    {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,/9j/..."}},
                    {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,iVBO..."}},
                ]
            ),
            # Vision model's response (already text-only, preserved)
            ChatMessage(
                role="assistant",
                content="The images depict various scenic locations including beaches and mountains."
            ),
            # User's follow-up (text-only, after switching to text model)
            ChatMessage(
                role="user",
                content="What island were these pictures taken on?"
            ),
        ]

        filtered = _filter_multimodal_history_for_text_models(messages)

        # Verify 3 messages preserved
        assert len(filtered) == 3

        # First message: text extracted, images placeholder added
        assert filtered[0].role == "user"
        assert "What are these pictures showing?" in filtered[0].content
        assert "[2 image(s) were attached]" in filtered[0].content

        # Second message: assistant response unchanged
        assert filtered[1].role == "assistant"
        assert filtered[1].content == "The images depict various scenic locations including beaches and mountains."

        # Third message: text-only user prompt unchanged
        assert filtered[2].role == "user"
        assert filtered[2].content == "What island were these pictures taken on?"

    def test_multiple_text_parts_combined(self):
        """Multiple text parts in multimodal content should be combined."""
        messages = [
            ChatMessage(
                role="user",
                content=[
                    {"type": "text", "text": "First part."},
                    {"type": "image_url", "image_url": {"url": "data:..."}},
                    {"type": "text", "text": "Second part."},
                ]
            ),
        ]

        filtered = _filter_multimodal_history_for_text_models(messages)

        assert "First part." in filtered[0].content
        assert "Second part." in filtered[0].content
        assert "[1 image(s) were attached]" in filtered[0].content

    def test_empty_text_parts_ignored(self):
        """Empty text parts should be ignored."""
        messages = [
            ChatMessage(
                role="user",
                content=[
                    {"type": "text", "text": ""},
                    {"type": "text", "text": "Valid text"},
                    {"type": "image_url", "image_url": {"url": "data:..."}},
                ]
            ),
        ]

        filtered = _filter_multimodal_history_for_text_models(messages)

        assert filtered[0].content == "Valid text\n\n[1 image(s) were attached]"

    def test_mixed_conversation_history(self):
        """Mixed string and array content should be handled correctly."""
        messages = [
            ChatMessage(role="user", content="Text-only question"),
            ChatMessage(role="assistant", content="Text-only answer"),
            ChatMessage(
                role="user",
                content=[
                    {"type": "text", "text": "Question with image"},
                    {"type": "image_url", "image_url": {"url": "data:..."}},
                ]
            ),
            ChatMessage(role="assistant", content="Answer to image question"),
        ]

        filtered = _filter_multimodal_history_for_text_models(messages)

        assert len(filtered) == 4
        assert filtered[0].content == "Text-only question"
        assert filtered[1].content == "Text-only answer"
        assert "Question with image" in filtered[2].content
        assert "[1 image(s) were attached]" in filtered[2].content
        assert filtered[3].content == "Answer to image question"

    def test_unknown_content_types_ignored(self):
        """Unknown content types should be ignored gracefully."""
        messages = [
            ChatMessage(
                role="user",
                content=[
                    {"type": "text", "text": "Valid text"},
                    {"type": "unknown_future_type", "data": "..."},  # Unknown type
                    {"type": "text", "text": "More text"},
                ]
            ),
        ]

        # Should not raise exception
        filtered = _filter_multimodal_history_for_text_models(messages)

        assert len(filtered) == 1
        # Unknown types ignored, only text extracted
        assert "Valid text" in filtered[0].content
        assert "More text" in filtered[0].content

    def test_sessionStorage_placeholder_removed(self):
        """nChat's [IMAGE_DATA_REMOVED] placeholder should still be filtered."""
        messages = [
            ChatMessage(
                role="user",
                content=[
                    {"type": "text", "text": "Describe"},
                    {"type": "image_url", "image_url": {"url": "[IMAGE_DATA_REMOVED]"}},
                ]
            ),
        ]

        filtered = _filter_multimodal_history_for_text_models(messages)

        # Image placeholder should still be detected and counted
        assert "[1 image(s) were attached]" in filtered[0].content
