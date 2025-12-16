"""
Server Vision API tests (ADR-012 Phase 3).

Tests for vision-specific server functionality:
- Image detection in requests
- Streaming rejection for vision
- Message format handling
- Vision request routing
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
from fastapi.testclient import TestClient
from pydantic import ValidationError

from mlxk2.core.server_base import (
    app,
    ChatMessage,
    ChatCompletionRequest,
    _request_has_images,
    _messages_to_dicts,
    _extract_text_from_messages,
)


class TestChatMessageModel:
    """Tests for ChatMessage with Vision content support."""

    def test_string_content(self):
        """ChatMessage should accept string content."""
        msg = ChatMessage(role="user", content="Hello")
        assert msg.content == "Hello"

    def test_list_content_text_only(self):
        """ChatMessage should accept list content with text."""
        content = [{"type": "text", "text": "What is this?"}]
        msg = ChatMessage(role="user", content=content)
        assert msg.content == content

    def test_list_content_with_image(self):
        """ChatMessage should accept list content with image_url."""
        content = [
            {"type": "text", "text": "What is this?"},
            {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,/9j/4A..."}}
        ]
        msg = ChatMessage(role="user", content=content)
        assert len(msg.content) == 2
        assert msg.content[1]["type"] == "image_url"

    def test_invalid_role_rejected(self):
        """ChatMessage should reject invalid roles."""
        with pytest.raises(ValidationError):
            ChatMessage(role="invalid", content="Hello")


class TestRequestHasImages:
    """Tests for _request_has_images() helper."""

    def test_string_content_no_images(self):
        """String content should not have images."""
        messages = [ChatMessage(role="user", content="Hello")]
        assert _request_has_images(messages) is False

    def test_list_content_text_only_no_images(self):
        """List content with text only should not have images."""
        messages = [
            ChatMessage(role="user", content=[{"type": "text", "text": "Hello"}])
        ]
        assert _request_has_images(messages) is False

    def test_list_content_with_image(self):
        """List content with image_url should detect images."""
        messages = [
            ChatMessage(role="user", content=[
                {"type": "text", "text": "What is this?"},
                {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,..."}}
            ])
        ]
        assert _request_has_images(messages) is True

    def test_multiple_messages_with_image_in_second(self):
        """Should detect image in any message."""
        messages = [
            ChatMessage(role="system", content="You are helpful."),
            ChatMessage(role="user", content=[
                {"type": "image_url", "image_url": {"url": "data:image/png;base64,..."}}
            ])
        ]
        assert _request_has_images(messages) is True

    def test_empty_list_content(self):
        """Empty list content should not have images."""
        messages = [ChatMessage(role="user", content=[])]
        assert _request_has_images(messages) is False


class TestMessagesToDicts:
    """Tests for _messages_to_dicts() helper."""

    def test_converts_string_content(self):
        """Should convert string content to dict format."""
        messages = [ChatMessage(role="user", content="Hello")]
        result = _messages_to_dicts(messages)
        assert result == [{"role": "user", "content": "Hello"}]

    def test_converts_list_content(self):
        """Should preserve list content in dict format."""
        content = [{"type": "text", "text": "Hello"}]
        messages = [ChatMessage(role="user", content=content)]
        result = _messages_to_dicts(messages)
        assert result == [{"role": "user", "content": content}]


class TestExtractTextFromMessages:
    """Tests for _extract_text_from_messages() helper."""

    def test_extracts_string_content(self):
        """Should extract text from string content."""
        messages = [ChatMessage(role="user", content="Hello world")]
        result = _extract_text_from_messages(messages)
        assert result == "Hello world"

    def test_extracts_from_list_content(self):
        """Should extract text items from list content."""
        messages = [
            ChatMessage(role="user", content=[
                {"type": "text", "text": "What is in this image?"},
                {"type": "image_url", "image_url": {"url": "data:..."}}
            ])
        ]
        result = _extract_text_from_messages(messages)
        assert result == "What is in this image?"

    def test_combines_multiple_messages(self):
        """Should combine text from multiple messages."""
        messages = [
            ChatMessage(role="system", content="You are helpful."),
            ChatMessage(role="user", content="What is 2+2?")
        ]
        result = _extract_text_from_messages(messages)
        assert "You are helpful." in result
        assert "What is 2+2?" in result


class TestVisionStreamingGracefulDegradation:
    """Tests for graceful degradation with vision streaming requests."""

    def test_vision_streaming_graceful_degradation(self):
        """Vision request with stream=True should gracefully degrade to batch (not 400)."""
        # Note: With graceful degradation, stream=true is silently ignored
        # The server proceeds with batch mode instead of returning an error
        client = TestClient(app)

        # Vision request format with image
        payload = {
            "model": "test/vision-model",
            "messages": [{
                "role": "user",
                "content": [
                    {"type": "text", "text": "What is this?"},
                    {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,/9j/4AAQSkZJRg=="}}
                ]
            }],
            "stream": True,
        }

        # Mock to avoid actual model loading - will fail on model not found, not streaming
        with patch('mlxk2.core.server_base._request_has_images', return_value=True):
            resp = client.post("/v1/chat/completions", json=payload)

        # Should NOT be 400 with "streaming" error - graceful degradation
        # Will be 404 (model not found) or similar, but not streaming rejection
        if resp.status_code == 400:
            assert "streaming" not in resp.text.lower(), "Should not reject due to streaming"


class TestTextPathUnaffected:
    """Tests to ensure text path is not affected by vision changes."""

    def test_text_request_uses_text_handler(self):
        """Text-only request should use text handler path."""
        client = TestClient(app)

        mock_runner = Mock()
        mock_runner.generate_batch.return_value = "Response"
        mock_runner._format_conversation.return_value = "User: Hello\n\nAssistant:"

        with patch('mlxk2.core.server_base.get_or_load_model', return_value=mock_runner):
            payload = {
                "model": "test/text-model",
                "messages": [{"role": "user", "content": "Hello"}],
                "stream": False,
            }
            resp = client.post("/v1/chat/completions", json=payload)

        assert resp.status_code == 200
        assert mock_runner.generate_batch.called

    def test_text_request_streaming_works(self):
        """Text request with streaming should work normally."""
        client = TestClient(app)

        mock_runner = Mock()
        mock_runner.generate_streaming.return_value = iter(["Hello", " world"])
        mock_runner._format_conversation.return_value = "User: Hi\n\nAssistant:"

        with patch('mlxk2.core.server_base.get_or_load_model', return_value=mock_runner):
            payload = {
                "model": "test/text-model",
                "messages": [{"role": "user", "content": "Hi"}],
                "stream": True,
            }
            resp = client.post("/v1/chat/completions", json=payload)

        # Streaming should return 200
        assert resp.status_code == 200
