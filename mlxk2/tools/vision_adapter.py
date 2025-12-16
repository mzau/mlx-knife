"""
Vision HTTP adapter for converting OpenAI-compatible requests to VisionRunner format.

This module handles Base64 image decoding and OpenAI message format parsing
for the server Vision API (ADR-012 Phase 3).
"""

from __future__ import annotations

import base64
import hashlib
import re
from typing import Any, Dict, List, Optional, Tuple

# No imports needed - use standard Python exceptions

# Limits for vision requests (safety and resource management)
# Conservative limits to prevent Metal OOM crashes (ADR-012 Phase 3)
# Phase 1c (future) will use batching for large collections
MAX_IMAGES_PER_REQUEST = 5  # Tested stable with 5 images
MAX_IMAGE_SIZE_BYTES = 20 * 1024 * 1024  # 20 MB per image
MAX_TOTAL_IMAGE_BYTES = 50 * 1024 * 1024  # 50 MB total (critical for Metal API)
SUPPORTED_MIME_TYPES = frozenset({"jpeg", "jpg", "png", "gif", "webp"})


class VisionHTTPAdapter:
    """Adapter for converting OpenAI Vision API format to VisionRunner format."""

    @staticmethod
    def parse_openai_messages(
        messages: List[Dict[str, Any]]
    ) -> Tuple[str, List[Tuple[str, bytes]]]:
        """
        Parse OpenAI-style messages and extract text prompt + images.

        OpenAI Vision API format:
        [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "What's in this image?"},
                    {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,..."}}
                ]
            }
        ]

        Important: Images are extracted ONLY from the most recent user message.
        Text context is extracted from ALL messages (including assistant responses).

        This follows OpenAI Vision API behavior where previous images remain in
        history for text context but are NOT sent as visual input to the model.

        Args:
            messages: List of message dicts (OpenAI format)

        Returns:
            (prompt, images) tuple where:
            - prompt: str - Combined text from all text content blocks
            - images: List[Tuple[str, bytes]] - List of (filename, raw_bytes) tuples

        Raises:
            MLXKError: If message format is invalid or images cannot be decoded

        See: docs/ISSUES/VISION-SEQUENTIAL-IMAGES-ISSUE.md
        """
        if not messages:
            raise ValueError("Messages list cannot be empty")

        text_parts = []

        # Extract text from ALL messages (for conversation context)
        for msg in messages:
            if not isinstance(msg, dict):
                raise ValueError("Each message must be a dict")

            content = msg.get("content")

            if content is None:
                continue

            # Handle string content (simple text message)
            if isinstance(content, str):
                text_parts.append(content)
                continue

            # Handle array content (Vision API format with text + images)
            if not isinstance(content, list):
                raise ValueError(
                    f"Message content must be string or array, got {type(content).__name__}"
                )

            for item in content:
                if not isinstance(item, dict):
                    raise ValueError(
                        "Each content item must be a dict with 'type' field"
                    )

                item_type = item.get("type")

                if item_type == "text":
                    text = item.get("text", "")
                    if text:
                        text_parts.append(text)

        # Extract images ONLY from the most recent user message
        images = []

        for msg in reversed(messages):
            if not isinstance(msg, dict):
                continue

            role = msg.get("role")
            if role != "user":
                continue

            content = msg.get("content")
            if content is None or not isinstance(content, list):
                # Last user message has no images (text-only follow-up)
                break

            # Process image_url items from this message only
            for item in content:
                if not isinstance(item, dict):
                    continue

                item_type = item.get("type")

                if item_type == "image_url":
                    image_url_obj = item.get("image_url")
                    if not isinstance(image_url_obj, dict):
                        raise ValueError(
                            "image_url content must have 'image_url' dict"
                        )

                    url = image_url_obj.get("url", "")
                    if not url:
                        raise ValueError(
                            "image_url.url cannot be empty"
                        )

                    # Decode base64 image
                    filename, raw_bytes = VisionHTTPAdapter.decode_base64_image(url)
                    images.append((filename, raw_bytes))

                    # Enforce image count limit
                    if len(images) > MAX_IMAGES_PER_REQUEST:
                        raise ValueError(
                            f"Too many images in request (max: {MAX_IMAGES_PER_REQUEST})"
                        )

                    # Enforce total size limit (critical for Metal API)
                    total_size = sum(len(img[1]) for img in images)
                    if total_size > MAX_TOTAL_IMAGE_BYTES:
                        size_mb = total_size / (1024 * 1024)
                        limit_mb = MAX_TOTAL_IMAGE_BYTES / (1024 * 1024)
                        raise ValueError(
                            f"Total image size ({size_mb:.1f} MB) exceeds limit ({limit_mb:.0f} MB). "
                            f"Try fewer or smaller images."
                        )

            # Stop after processing first (most recent) user message
            break

        # Combine text parts
        prompt = " ".join(text_parts).strip()

        # Validation: Must have either text or images
        if not prompt and not images:
            raise ValueError(
                "Request must contain at least text or images"
            )

        # If no text provided but images present, use default prompt
        if not prompt and images:
            prompt = "Describe the image."

        return prompt, images

    @staticmethod
    def decode_base64_image(url: str) -> Tuple[str, bytes]:
        """
        Decode a Base64-encoded image from a data URL.

        Supports data URLs like:
        - data:image/jpeg;base64,/9j/4AAQSkZJRg...
        - data:image/png;base64,iVBORw0KGgo...

        Args:
            url: Data URL string with base64-encoded image

        Returns:
            (filename, raw_bytes) tuple where:
            - filename: Generated name based on content hash (e.g., "image_a1b2c3.jpg")
            - raw_bytes: Decoded image bytes

        Raises:
            MLXKError: If URL is not a valid data URL or base64 decoding fails
        """
        # Check if it's a data URL
        if not url.startswith("data:"):
            raise ValueError(
                "Only data URLs are supported (e.g., data:image/jpeg;base64,...). "
                "External URLs are not supported."
            )

        # Parse data URL: data:image/jpeg;base64,<data>
        match = re.match(
            r"^data:image/(jpeg|jpg|png|gif|webp);base64,(.+)$", url, re.IGNORECASE
        )
        if not match:
            raise ValueError(
                "Invalid data URL format. Expected: data:image/<type>;base64,<data>. "
                f"Supported types: {', '.join(sorted(SUPPORTED_MIME_TYPES))}"
            )

        mime_type = match.group(1).lower()
        base64_data = match.group(2)

        # Normalize MIME type
        if mime_type == "jpg":
            mime_type = "jpeg"

        # Validate MIME type
        if mime_type not in SUPPORTED_MIME_TYPES:
            raise ValueError(
                f"Unsupported image type: {mime_type}. "
                f"Supported types: {', '.join(sorted(SUPPORTED_MIME_TYPES))}"
            )

        # Decode base64
        try:
            raw_bytes = base64.b64decode(base64_data, validate=True)
        except Exception as e:
            raise ValueError(
                f"Failed to decode base64 image data: {e}"
            ) from e

        # Validate that we got some data
        if not raw_bytes:
            raise ValueError("Decoded image data is empty")

        # Enforce size limit
        if len(raw_bytes) > MAX_IMAGE_SIZE_BYTES:
            size_mb = len(raw_bytes) / (1024 * 1024)
            limit_mb = MAX_IMAGE_SIZE_BYTES / (1024 * 1024)
            raise ValueError(
                f"Image size ({size_mb:.1f} MB) exceeds limit ({limit_mb:.0f} MB)"
            )

        # Generate deterministic filename from content hash
        content_hash = hashlib.sha256(raw_bytes).hexdigest()[:8]
        filename = f"image_{content_hash}.{mime_type}"

        return filename, raw_bytes

    @staticmethod
    def assign_image_ids_from_history(messages: List[Dict[str, Any]]) -> Dict[str, int]:
        """
        Assign stable image IDs from conversation history.

        Scans all messages chronologically and assigns sequential IDs based on
        content hash. This enables stable "Image 1, Image 2, ..." numbering
        across multiple requests in a conversation.

        The conversation history IS the session - no server-side state needed.
        This is 100% OpenAI API compatible.

        Strategy:
        1. Scan assistant messages for filename mapping tables (server-generated)
        2. Scan user messages for image_url content (current request)
        3. Combine both to build complete hash->ID mapping

        This allows clients to drop Base64 data from history (storage optimization)
        while preserving Image ID continuity via the server's own text output.

        Args:
            messages: List of message dicts (OpenAI format, full history)

        Returns:
            Dict mapping content_hash (8 chars) -> image_id (1-based)
            Example: {"5c691ddb": 1, "aaad16ca": 2}

        Behavior:
            - Request 1: beach.jpg → Image 1
            - Request 2: beach.jpg + mountain.jpg in history → Image 1, Image 2
            - Re-upload beach.jpg → Still Image 1 (hash match = deduplication)
        """
        seen_hashes: Dict[str, int] = {}
        next_id = 1

        for msg in messages:
            if not isinstance(msg, dict):
                continue

            role = msg.get("role")
            content = msg.get("content")

            # Scan assistant messages for filename mapping tables
            if role == "assistant" and isinstance(content, str):
                # Parse server-generated mapping tables: "| 1 | image_5733332c.jpeg |"
                # Only parse if the mlxk marker is present (avoids false positives)
                if "<!-- mlxk:filenames -->" in content:
                    parsed_hashes = VisionHTTPAdapter._parse_filename_mapping(content)
                    for hash_val, img_id in parsed_hashes.items():
                        if hash_val not in seen_hashes:
                            seen_hashes[hash_val] = img_id
                            next_id = max(next_id, img_id + 1)

            # Scan user messages for image_url content
            elif role == "user":
                if not isinstance(content, list):
                    # String content = text-only message, skip
                    continue

                # Process image_url items
                for item in content:
                    if not isinstance(item, dict):
                        continue

                    if item.get("type") != "image_url":
                        continue

                    image_url_obj = item.get("image_url")
                    if not isinstance(image_url_obj, dict):
                        continue

                    url = image_url_obj.get("url", "")
                    if not url:
                        continue

                    # Compute content hash from base64 data
                    content_hash = VisionHTTPAdapter._compute_content_hash(url)
                    if content_hash and content_hash not in seen_hashes:
                        seen_hashes[content_hash] = next_id
                        next_id += 1

        return seen_hashes

    @staticmethod
    def _parse_filename_mapping(content: str) -> Dict[str, int]:
        """
        Parse filename mapping table from assistant response text.

        Extracts hash->ID mappings from server-generated tables like:
            <!-- mlxk:filenames -->
            | Image | Filename |
            |-------|----------|
            | 1 | image_5733332c.jpeg |
            | 2 | image_49779094.jpeg |

        Args:
            content: Assistant message content (string)

        Returns:
            Dict mapping content_hash (8 chars) -> image_id (1-based)
            Example: {"5733332c": 1, "49779094": 2}
        """
        parsed: Dict[str, int] = {}

        # Pattern: "| 1 | image_5733332c.jpeg |"
        # Matches: (image_id, hash)
        pattern = r'\|\s*(\d+)\s*\|\s*image_([a-f0-9]{8})\.'

        matches = re.findall(pattern, content)
        for (img_id_str, hash_val) in matches:
            img_id = int(img_id_str)
            parsed[hash_val] = img_id

        return parsed

    @staticmethod
    def _compute_content_hash(url: str) -> Optional[str]:
        """
        Compute content hash from a data URL.

        Args:
            url: Data URL string (data:image/jpeg;base64,...)

        Returns:
            8-character hash string, or None if URL is invalid
        """
        if not url.startswith("data:"):
            return None

        # Extract base64 data after the comma
        try:
            _, base64_data = url.split(",", 1)
            raw_bytes = base64.b64decode(base64_data, validate=True)
            return hashlib.sha256(raw_bytes).hexdigest()[:8]
        except Exception:
            return None
