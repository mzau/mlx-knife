"""
Tests for Vision HTTP Adapter (ADR-012 Phase 3).

Tests Base64 decoding, OpenAI message format parsing, and error handling.
"""

import base64
import pytest

# No MLXKError import needed - using standard ValueError
from mlxk2.tools.vision_adapter import (
    VisionHTTPAdapter,
    MAX_IMAGES_PER_REQUEST,
    MAX_IMAGE_SIZE_BYTES,
    MAX_TOTAL_IMAGE_BYTES,
)


# Test fixtures: Base64-encoded images (minimal valid images)

# 1x1 red pixel JPEG
VALID_JPEG_B64 = "/9j/4AAQSkZJRgABAQEASABIAAD/2wBDAAEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQH/2wBDAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQH/wAARCAABAAEDAREAAhEBAxEB/8QAFQABAQAAAAAAAAAAAAAAAAAAAAv/xAAUEAEAAAAAAAAAAAAAAAAAAAAA/8QAFQEBAQAAAAAAAAAAAAAAAAAAAAX/xAAUEQEAAAAAAAAAAAAAAAAAAAAA/9oADAMBAAIRAxEAPwA/wA//"

# 1x1 transparent pixel PNG
VALID_PNG_B64 = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNkYPhfDwAChwGA60e6kgAAAABJRU5ErkJggg=="


class TestDecodeBase64Image:
    """Tests for decode_base64_image()."""

    def test_decode_valid_jpeg_data_url(self):
        """Test decoding a valid JPEG data URL."""
        url = f"data:image/jpeg;base64,{VALID_JPEG_B64}"
        filename, raw_bytes = VisionHTTPAdapter.decode_base64_image(url)

        assert filename.startswith("image_")
        assert filename.endswith(".jpeg")
        assert len(raw_bytes) > 0
        assert isinstance(raw_bytes, bytes)

    def test_decode_valid_png_data_url(self):
        """Test decoding a valid PNG data URL."""
        url = f"data:image/png;base64,{VALID_PNG_B64}"
        filename, raw_bytes = VisionHTTPAdapter.decode_base64_image(url)

        assert filename.startswith("image_")
        assert filename.endswith(".png")
        assert len(raw_bytes) > 0

    def test_decode_jpg_normalized_to_jpeg(self):
        """Test that jpg MIME type is normalized to jpeg."""
        url = f"data:image/jpg;base64,{VALID_JPEG_B64}"
        filename, raw_bytes = VisionHTTPAdapter.decode_base64_image(url)

        # Should use .jpeg extension, not .jpg
        assert filename.endswith(".jpeg")

    def test_filename_is_deterministic(self):
        """Test that same image produces same filename."""
        url = f"data:image/jpeg;base64,{VALID_JPEG_B64}"
        filename1, _ = VisionHTTPAdapter.decode_base64_image(url)
        filename2, _ = VisionHTTPAdapter.decode_base64_image(url)

        assert filename1 == filename2

    def test_filename_differs_for_different_images(self):
        """Test that different images produce different filenames."""
        url1 = f"data:image/jpeg;base64,{VALID_JPEG_B64}"
        url2 = f"data:image/png;base64,{VALID_PNG_B64}"

        filename1, _ = VisionHTTPAdapter.decode_base64_image(url1)
        filename2, _ = VisionHTTPAdapter.decode_base64_image(url2)

        assert filename1 != filename2

    def test_malformed_base64_raises_error(self):
        """Test that malformed base64 data raises validation error."""
        url = "data:image/jpeg;base64,!!!invalid_base64!!!"

        with pytest.raises(ValueError) as exc:
            VisionHTTPAdapter.decode_base64_image(url)

        assert "Failed to decode base64" in str(exc.value)

    def test_empty_base64_data_raises_error(self):
        """Test that empty base64 data raises validation error."""
        url = "data:image/jpeg;base64,"

        with pytest.raises(ValueError) as exc:
            VisionHTTPAdapter.decode_base64_image(url)

        # Empty base64 data fails regex match
        assert "Invalid data URL format" in str(exc.value)

    def test_unsupported_mime_type_raises_error(self):
        """Test that unsupported MIME types raise validation error."""
        # BMP is not supported
        url = "data:image/bmp;base64,Qk0="

        with pytest.raises(ValueError) as exc:
            VisionHTTPAdapter.decode_base64_image(url)

        assert "Invalid data URL format" in str(exc.value) or "Unsupported" in str(exc.value)

    def test_external_url_raises_error(self):
        """Test that external URLs (https) are rejected."""
        url = "https://example.com/image.jpg"

        with pytest.raises(ValueError) as exc:
            VisionHTTPAdapter.decode_base64_image(url)

        assert "Only data URLs are supported" in str(exc.value)
        assert "External URLs are not supported" in str(exc.value)

    def test_non_data_url_scheme_raises_error(self):
        """Test that non-data URL schemes are rejected."""
        url = "file:///path/to/image.jpg"

        with pytest.raises(ValueError) as exc:
            VisionHTTPAdapter.decode_base64_image(url)

        assert "Only data URLs are supported" in str(exc.value)

    def test_oversized_image_raises_error(self):
        """Test that images exceeding size limit raise validation error."""
        # Create a large base64 string (> 20 MB)
        large_data = "A" * (MAX_IMAGE_SIZE_BYTES + 1000)
        large_b64 = base64.b64encode(large_data.encode()).decode()
        url = f"data:image/jpeg;base64,{large_b64}"

        with pytest.raises(ValueError) as exc:
            VisionHTTPAdapter.decode_base64_image(url)

        assert "exceeds limit" in str(exc.value).lower()


class TestParseOpenAIMessages:
    """Tests for parse_openai_messages()."""

    def test_parse_single_image_message(self):
        """Test parsing a message with one image."""
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "What's in this image?"},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{VALID_JPEG_B64}"}}
                ]
            }
        ]

        prompt, images = VisionHTTPAdapter.parse_openai_messages(messages)

        assert prompt == "What's in this image?"
        assert len(images) == 1
        assert images[0][0].startswith("image_")
        assert images[0][0].endswith(".jpeg")

    def test_parse_multiple_images_message(self):
        """Test parsing a message with multiple images."""
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Compare these images"},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{VALID_JPEG_B64}"}},
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{VALID_PNG_B64}"}}
                ]
            }
        ]

        prompt, images = VisionHTTPAdapter.parse_openai_messages(messages)

        assert prompt == "Compare these images"
        assert len(images) == 2
        assert images[0][0].endswith(".jpeg")
        assert images[1][0].endswith(".png")

    def test_parse_string_content(self):
        """Test parsing a simple string message (no vision)."""
        messages = [
            {"role": "user", "content": "Hello, how are you?"}
        ]

        prompt, images = VisionHTTPAdapter.parse_openai_messages(messages)

        assert prompt == "Hello, how are you?"
        assert len(images) == 0

    def test_parse_images_without_text_uses_default_prompt(self):
        """Test that images without text get default prompt."""
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{VALID_JPEG_B64}"}}
                ]
            }
        ]

        prompt, images = VisionHTTPAdapter.parse_openai_messages(messages)

        assert prompt == "Describe the image."
        assert len(images) == 1

    def test_parse_multiple_text_blocks_combined(self):
        """Test that multiple text blocks are combined."""
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "First part."},
                    {"type": "text", "text": "Second part."}
                ]
            }
        ]

        prompt, images = VisionHTTPAdapter.parse_openai_messages(messages)

        assert prompt == "First part. Second part."
        assert len(images) == 0

    def test_parse_unknown_content_type_skipped(self):
        """Test that unknown content types are skipped gracefully."""
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Valid text"},
                    {"type": "unknown_type", "data": "ignored"},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{VALID_JPEG_B64}"}}
                ]
            }
        ]

        prompt, images = VisionHTTPAdapter.parse_openai_messages(messages)

        assert prompt == "Valid text"
        assert len(images) == 1

    def test_empty_messages_raises_error(self):
        """Test that empty messages list raises validation error."""
        messages = []

        with pytest.raises(ValueError) as exc:
            VisionHTTPAdapter.parse_openai_messages(messages)

        assert "Messages list cannot be empty" in str(exc.value)

    def test_no_content_raises_error(self):
        """Test that messages with no text or images raise validation error."""
        messages = [
            {"role": "user", "content": []}
        ]

        with pytest.raises(ValueError) as exc:
            VisionHTTPAdapter.parse_openai_messages(messages)

        assert "at least text or images" in str(exc.value).lower()

    def test_invalid_content_type_raises_error(self):
        """Test that invalid content types raise validation error."""
        messages = [
            {"role": "user", "content": 123}  # Not string or list
        ]

        with pytest.raises(ValueError) as exc:
            VisionHTTPAdapter.parse_openai_messages(messages)

        assert "must be string or array" in str(exc.value).lower()

    def test_missing_image_url_dict_raises_error(self):
        """Test that missing image_url dict raises validation error."""
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image_url"}  # Missing image_url dict
                ]
            }
        ]

        with pytest.raises(ValueError) as exc:
            VisionHTTPAdapter.parse_openai_messages(messages)

        assert "image_url" in str(exc.value).lower()

    def test_empty_image_url_raises_error(self):
        """Test that empty image URL raises validation error."""
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": ""}}
                ]
            }
        ]

        with pytest.raises(ValueError) as exc:
            VisionHTTPAdapter.parse_openai_messages(messages)

        assert "url cannot be empty" in str(exc.value).lower()

    def test_too_many_images_raises_error(self):
        """Test that exceeding image count limit raises validation error."""
        # Create more than MAX_IMAGES_PER_REQUEST images
        content = [{"type": "text", "text": "Many images"}]
        for _ in range(MAX_IMAGES_PER_REQUEST + 1):
            content.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{VALID_JPEG_B64}"}
            })

        messages = [{"role": "user", "content": content}]

        with pytest.raises(ValueError) as exc:
            VisionHTTPAdapter.parse_openai_messages(messages)

        assert "Too many images" in str(exc.value)
        assert str(MAX_IMAGES_PER_REQUEST) in str(exc.value)

    def test_total_image_size_limit_enforced(self):
        """Test that total image size limit is enforced (critical for Metal API)."""
        # Create images that individually pass but collectively exceed total limit
        # Each image ~3 MB → 5 images would be ~15 MB (under 50 MB limit)
        # But we'll create larger images to trigger the limit
        large_data = "A" * (12 * 1024 * 1024)  # 12 MB per image
        large_b64 = base64.b64encode(large_data.encode()).decode()

        content = [{"type": "text", "text": "Many large images"}]
        # 5 images × 12 MB = 60 MB → exceeds 50 MB limit
        for _ in range(5):
            content.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{large_b64}"}
            })

        messages = [{"role": "user", "content": content}]

        with pytest.raises(ValueError) as exc:
            VisionHTTPAdapter.parse_openai_messages(messages)

        assert "Total image size" in str(exc.value)
        assert "exceeds limit" in str(exc.value)
        assert "Try fewer or smaller images" in str(exc.value)


class TestSequentialImageExtraction:
    """
    Tests for sequential image extraction logic (VISION-SEQUENTIAL-IMAGES-ISSUE).

    Images should be extracted ONLY from the most recent user message.
    Text context should be extracted from ALL messages (including assistant responses).
    """

    def test_sequential_images_extracts_only_last(self):
        """Test that sequential images in separate messages extract only the last one."""
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "describe this picture"},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{VALID_JPEG_B64}"}}
                ]
            },
            {"role": "assistant", "content": "A cat on a couch."},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "describe this picture"},
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{VALID_PNG_B64}"}}
                ]
            }
        ]

        prompt, images = VisionHTTPAdapter.parse_openai_messages(messages)

        # Should extract ONLY the PNG from the last user message
        assert len(images) == 1
        assert images[0][0].endswith(".png")  # PNG, not JPEG

        # Text context should include ALL messages
        assert "describe this picture" in prompt
        assert "A cat on a couch" in prompt

    def test_text_only_follow_up_extracts_no_images(self):
        """Test that text-only follow-up after images extracts no images."""
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "describe this picture"},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{VALID_JPEG_B64}"}}
                ]
            },
            {"role": "assistant", "content": "A tabby cat on a blue couch."},
            {"role": "user", "content": "What color was the cat?"}
        ]

        prompt, images = VisionHTTPAdapter.parse_openai_messages(messages)

        # Should extract NO images (last user message is text-only)
        assert len(images) == 0

        # Text context should include all messages
        assert "describe this picture" in prompt
        assert "A tabby cat on a blue couch" in prompt
        assert "What color was the cat?" in prompt

    def test_multiple_images_in_last_message_all_extracted(self):
        """Test that multiple images in the last user message are all extracted."""
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "first image"},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{VALID_JPEG_B64}"}}
                ]
            },
            {"role": "assistant", "content": "Described first image."},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "compare these two"},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{VALID_JPEG_B64}"}},
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{VALID_PNG_B64}"}}
                ]
            }
        ]

        prompt, images = VisionHTTPAdapter.parse_openai_messages(messages)

        # Should extract BOTH images from the last user message
        assert len(images) == 2
        assert images[0][0].endswith(".jpeg")
        assert images[1][0].endswith(".png")

        # Text context from all messages
        assert "first image" in prompt
        assert "Described first image" in prompt
        assert "compare these two" in prompt

    def test_assistant_response_preserved_in_text_context(self):
        """Test that assistant responses are preserved in text context."""
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "describe this picture"},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{VALID_JPEG_B64}"}}
                ]
            },
            {"role": "assistant", "content": "The image shows a garden with flowers."},
            {"role": "user", "content": "What country is this garden in?"}
        ]

        prompt, images = VisionHTTPAdapter.parse_openai_messages(messages)

        # No images in last message
        assert len(images) == 0

        # Assistant's description must be in text context (important for follow-up!)
        assert "The image shows a garden with flowers." in prompt
        assert "What country is this garden in?" in prompt

    def test_three_sequential_images_extracts_only_third(self):
        """Test that three sequential images extract only the third."""
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Image 1"},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{VALID_JPEG_B64}"}}
                ]
            },
            {"role": "assistant", "content": "Description 1"},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Image 2"},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{VALID_JPEG_B64}"}}
                ]
            },
            {"role": "assistant", "content": "Description 2"},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Image 3"},
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{VALID_PNG_B64}"}}
                ]
            }
        ]

        prompt, images = VisionHTTPAdapter.parse_openai_messages(messages)

        # Should extract ONLY the PNG from the third user message
        assert len(images) == 1
        assert images[0][0].endswith(".png")

        # All text context preserved
        assert "Image 1" in prompt
        assert "Description 1" in prompt
        assert "Image 2" in prompt
        assert "Description 2" in prompt
        assert "Image 3" in prompt

    def test_last_user_message_with_string_content_no_images(self):
        """Test that last user message with string content extracts no images."""
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "describe"},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{VALID_JPEG_B64}"}}
                ]
            },
            {"role": "assistant", "content": "A picture."},
            {"role": "user", "content": "Thanks!"}  # String content, not array
        ]

        prompt, images = VisionHTTPAdapter.parse_openai_messages(messages)

        # No images (last user message is string, not array)
        assert len(images) == 0

        # Text context preserved
        assert "describe" in prompt
        assert "A picture." in prompt
        assert "Thanks!" in prompt


class TestAssignImageIdsFromHistory:
    """
    Tests for history-based image ID assignment (Session 32: Option D).

    The conversation history IS the session - no server-side state needed.
    IDs are assigned chronologically based on content hash for deduplication.
    """

    def test_single_image_gets_id_1(self):
        """Test that single image in first request gets ID 1."""
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "describe"},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{VALID_JPEG_B64}"}}
                ]
            }
        ]

        id_map = VisionHTTPAdapter.assign_image_ids_from_history(messages)

        assert len(id_map) == 1
        # Should have one entry with value 1
        assert list(id_map.values()) == [1]

    def test_sequential_images_get_sequential_ids(self):
        """Test that sequential images in separate messages get sequential IDs."""
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "describe"},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{VALID_JPEG_B64}"}}
                ]
            },
            {"role": "assistant", "content": "A cat."},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "describe"},
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{VALID_PNG_B64}"}}
                ]
            }
        ]

        id_map = VisionHTTPAdapter.assign_image_ids_from_history(messages)

        # Should have 2 entries: Image 1 (JPEG) and Image 2 (PNG)
        assert len(id_map) == 2
        assert sorted(id_map.values()) == [1, 2]

    def test_deduplication_same_image_same_id(self):
        """Test that re-uploading the same image gets the same ID (deduplication)."""
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "first upload"},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{VALID_JPEG_B64}"}}
                ]
            },
            {"role": "assistant", "content": "A cat."},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "upload same image again"},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{VALID_JPEG_B64}"}}
                ]
            }
        ]

        id_map = VisionHTTPAdapter.assign_image_ids_from_history(messages)

        # Should have only 1 entry (same hash = same ID)
        assert len(id_map) == 1
        assert list(id_map.values()) == [1]

    def test_text_only_messages_skipped(self):
        """Test that text-only messages don't affect ID assignment."""
        messages = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there!"},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "describe"},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{VALID_JPEG_B64}"}}
                ]
            },
            {"role": "assistant", "content": "A cat."},
            {"role": "user", "content": "Thanks!"}
        ]

        id_map = VisionHTTPAdapter.assign_image_ids_from_history(messages)

        # Only 1 image in history
        assert len(id_map) == 1
        assert list(id_map.values()) == [1]

    def test_empty_messages_returns_empty_map(self):
        """Test that empty messages list returns empty map."""
        messages = []

        id_map = VisionHTTPAdapter.assign_image_ids_from_history(messages)

        assert id_map == {}

    def test_no_images_returns_empty_map(self):
        """Test that messages with no images return empty map."""
        messages = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi!"},
            {"role": "user", "content": "How are you?"}
        ]

        id_map = VisionHTTPAdapter.assign_image_ids_from_history(messages)

        assert id_map == {}

    def test_multiple_images_in_single_message(self):
        """Test that multiple images in single message get sequential IDs."""
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "compare these"},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{VALID_JPEG_B64}"}},
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{VALID_PNG_B64}"}}
                ]
            }
        ]

        id_map = VisionHTTPAdapter.assign_image_ids_from_history(messages)

        # Should have 2 entries: ID 1 and ID 2
        assert len(id_map) == 2
        assert sorted(id_map.values()) == [1, 2]

    def test_assistant_messages_ignored(self):
        """Test that assistant messages (even if they contain image_url) are ignored."""
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "describe"},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{VALID_JPEG_B64}"}}
                ]
            },
            {
                "role": "assistant",
                "content": [
                    {"type": "text", "text": "Here's an image:"},
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{VALID_PNG_B64}"}}
                ]
            }
        ]

        id_map = VisionHTTPAdapter.assign_image_ids_from_history(messages)

        # Only 1 image (from user message, not assistant)
        assert len(id_map) == 1
        assert list(id_map.values()) == [1]

    def test_chronological_order_preserved(self):
        """Test that IDs are assigned in chronological order."""
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{VALID_JPEG_B64}"}}
                ]
            },
            {"role": "assistant", "content": "First image."},
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{VALID_PNG_B64}"}}
                ]
            },
            {"role": "assistant", "content": "Second image."},
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{VALID_JPEG_B64}"}}
                ]
            }
        ]

        id_map = VisionHTTPAdapter.assign_image_ids_from_history(messages)

        # JPEG appeared first (ID 1), PNG appeared second (ID 2)
        # Third message is same JPEG → reuses ID 1
        assert len(id_map) == 2

        # Get hashes to verify order
        import hashlib
        jpeg_hash = hashlib.sha256(base64.b64decode(VALID_JPEG_B64)).hexdigest()[:8]
        png_hash = hashlib.sha256(base64.b64decode(VALID_PNG_B64)).hexdigest()[:8]

        assert id_map[jpeg_hash] == 1  # JPEG first
        assert id_map[png_hash] == 2   # PNG second


class TestMappingTableParsing:
    """
    Tests for parsing filename mapping tables from assistant responses.

    This enables clients to drop Base64 data from history (storage optimization)
    while preserving Image ID continuity via the server's own text output.
    """

    def test_parse_simple_mapping_table(self):
        """Test parsing a simple mapping table with two images."""
        content = """A sandy beach with blue water.

<!-- mlxk:filenames -->
| Image | Filename |
|-------|----------|
| 1 | image_5733332c.jpeg |
| 2 | image_49779094.jpeg |
"""

        parsed = VisionHTTPAdapter._parse_filename_mapping(content)

        assert len(parsed) == 2
        assert parsed["5733332c"] == 1
        assert parsed["49779094"] == 2

    def test_parse_mapping_table_with_original_filenames(self):
        """Test parsing mapping table that includes original filenames."""
        content = """Description here.

<!-- mlxk:filenames -->
| Image | Filename |
|-------|----------|
| 1 | image_5733332c.jpeg (beach.jpg) |
| 2 | image_49779094.jpeg (mountain.png) |
"""

        parsed = VisionHTTPAdapter._parse_filename_mapping(content)

        assert len(parsed) == 2
        assert parsed["5733332c"] == 1
        assert parsed["49779094"] == 2

    def test_parse_empty_content(self):
        """Test parsing content without mapping table."""
        content = "Just some regular text without any mapping."

        parsed = VisionHTTPAdapter._parse_filename_mapping(content)

        assert parsed == {}

    def test_parse_table_without_marker_still_parses(self):
        """Test that _parse_filename_mapping() is a simple parser (marker check is caller's job)."""
        content = """Some description.

| Image | Filename |
|-------|----------|
| 1 | image_5733332c.jpeg |
"""
        # Note: No <!-- mlxk:filenames --> marker
        # But _parse_filename_mapping() is just a parser - it doesn't validate markers
        # The marker check happens in assign_image_ids_from_history()

        parsed = VisionHTTPAdapter._parse_filename_mapping(content)

        # Parser extracts data regardless of marker
        assert parsed == {"5733332c": 1}

    def test_history_without_marker_ignored(self):
        """Test that assign_image_ids_from_history() ignores tables without marker."""
        messages = [
            {"role": "user", "content": "describe"},
            {
                "role": "assistant",
                "content": """Beach.

| Image | Filename |
|-------|----------|
| 1 | image_aaaaaaaa.jpeg |
"""
                # Note: No <!-- mlxk:filenames --> marker
            }
        ]

        id_map = VisionHTTPAdapter.assign_image_ids_from_history(messages)

        # Should be empty because marker check in assign_image_ids_from_history()
        assert id_map == {}

    def test_history_with_mapping_table_no_base64(self):
        """Test that Image IDs are reconstructed from mapping table without Base64 data."""
        import hashlib

        # Compute hashes for reference
        jpeg_hash = hashlib.sha256(base64.b64decode(VALID_JPEG_B64)).hexdigest()[:8]
        png_hash = hashlib.sha256(base64.b64decode(VALID_PNG_B64)).hexdigest()[:8]

        # Simulate conversation where client dropped Base64 after first request
        messages = [
            # Request 1 (Vision): User sent beach.jpg with Base64 (not in history anymore)
            {"role": "user", "content": "describe this picture"},
            {
                "role": "assistant",
                "content": f"""A sandy beach.

<!-- mlxk:filenames -->
| Image | Filename |
|-------|----------|
| 1 | image_{jpeg_hash}.jpeg |
"""
            },
            # Request 2 (Text): No images
            {"role": "user", "content": "What color?"},
            {"role": "assistant", "content": "Blue."},
            # Request 3 (Vision): User sends mountain.jpg with Base64
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "describe this new picture"},
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{VALID_PNG_B64}"}}
                ]
            }
        ]

        id_map = VisionHTTPAdapter.assign_image_ids_from_history(messages)

        # Should reconstruct beach.jpg from mapping table (ID 1)
        # And assign mountain.jpg as ID 2
        assert len(id_map) == 2
        assert id_map[jpeg_hash] == 1  # From mapping table
        assert id_map[png_hash] == 2   # From new image_url

    def test_multiple_mapping_tables_in_history(self):
        """Test handling multiple mapping tables across conversation."""
        messages = [
            {"role": "user", "content": "describe"},
            {
                "role": "assistant",
                "content": """Beach.

<!-- mlxk:filenames -->
| Image | Filename |
|-------|----------|
| 1 | image_aaaaaaaa.jpeg |
"""
            },
            {"role": "user", "content": "describe another"},
            {
                "role": "assistant",
                "content": """Mountain.

<!-- mlxk:filenames -->
| Image | Filename |
|-------|----------|
| 1 | image_aaaaaaaa.jpeg |
| 2 | image_bbbbbbbb.jpeg |
"""
            }
        ]

        id_map = VisionHTTPAdapter.assign_image_ids_from_history(messages)

        # Should use the most complete mapping (from second table)
        assert len(id_map) == 2
        assert id_map["aaaaaaaa"] == 1
        assert id_map["bbbbbbbb"] == 2
