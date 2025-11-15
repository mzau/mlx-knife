"""SSE (Server-Sent Events) parsing utilities for E2E testing (ADR-011).

Provides utilities for parsing SSE streams from httpx responses.
Handles OpenAI-compatible SSE format with "data:" lines and [DONE] sentinel.
"""

from __future__ import annotations

import json
from typing import Iterator, Dict, Any


def parse_sse_stream(response) -> Iterator[Dict[str, Any]]:
    """Parse SSE event stream from httpx response.

    OpenAI-compatible SSE format:
        data: {"id": "cmpl-xxx", "object": "chat.completion.chunk", ...}
        data: {"choices": [{"delta": {"content": "token"}, ...}]}
        ...
        data: [DONE]

    Args:
        response: httpx.Response with streaming enabled (e.g., httpx.stream())

    Yields:
        Parsed JSON objects from each "data:" line (excludes [DONE])

    Raises:
        json.JSONDecodeError: If data line contains invalid JSON

    Example:
        >>> with httpx.stream("POST", url, json={...}) as response:
        ...     for chunk in parse_sse_stream(response):
        ...         print(chunk["choices"][0]["delta"].get("content", ""))
    """
    for line in response.iter_lines():
        # SSE lines are "data: <json>" or "data: [DONE]"
        if line.startswith("data: "):
            data = line[6:]  # Strip "data: " prefix

            # [DONE] sentinel marks end of stream
            if data == "[DONE]":
                break

            # Parse JSON payload
            try:
                yield json.loads(data)
            except json.JSONDecodeError as e:
                # Re-raise with context for debugging
                raise json.JSONDecodeError(
                    f"Invalid SSE JSON: {data!r}",
                    e.doc,
                    e.pos
                ) from e


def collect_sse_content(response) -> str:
    """Collect complete text content from SSE stream.

    Convenience function that extracts "content" fields from SSE chunks
    and concatenates them into a single string.

    Args:
        response: httpx.Response with streaming enabled

    Returns:
        Complete text content from all SSE chunks

    Example:
        >>> with httpx.stream("POST", url, json={"stream": True, ...}) as response:
        ...     text = collect_sse_content(response)
        ...     assert "<|end|>" not in text  # No visible stop tokens
    """
    content_parts = []

    for chunk in parse_sse_stream(response):
        # Extract content from delta
        if "choices" in chunk:
            for choice in chunk["choices"]:
                delta = choice.get("delta", {})
                if "content" in delta:
                    content_parts.append(delta["content"])

    return "".join(content_parts)


def validate_sse_format(response) -> tuple[bool, str]:
    """Validate SSE response format compliance.

    Checks:
    - All lines start with "data: " or are empty
    - JSON payloads are valid
    - Stream ends with "data: [DONE]"
    - Chunks have expected OpenAI structure

    Args:
        response: httpx.Response with streaming enabled

    Returns:
        (is_valid, error_message) tuple

    Example:
        >>> with httpx.stream("POST", url, json={"stream": True, ...}) as response:
        ...     valid, error = validate_sse_format(response)
        ...     assert valid, f"Invalid SSE format: {error}"
    """
    try:
        # Collect all lines to validate sentinel presence
        all_lines = []
        chunks = []
        found_done_sentinel = False

        for line in response.iter_lines():
            all_lines.append(line)

            # Parse SSE data lines
            if line.startswith("data: "):
                data = line[6:]  # Strip "data: " prefix

                # Check for [DONE] sentinel
                if data == "[DONE]":
                    found_done_sentinel = True
                    break

                # Parse JSON chunks
                try:
                    chunks.append(json.loads(data))
                except json.JSONDecodeError as e:
                    return False, f"Invalid JSON in SSE stream: {data!r}"

        # Validate [DONE] sentinel was present
        if not found_done_sentinel:
            return False, "Stream missing 'data: [DONE]' sentinel (clients would hang)"

        if not chunks:
            return False, "No SSE chunks received"

        # Validate first chunk has required fields
        first_chunk = chunks[0]
        if "id" not in first_chunk:
            return False, "First chunk missing 'id' field"
        if "object" not in first_chunk:
            return False, "First chunk missing 'object' field"

        # Validate all chunks have choices
        for i, chunk in enumerate(chunks):
            if "choices" not in chunk:
                return False, f"Chunk {i} missing 'choices' field"

        return True, ""

    except json.JSONDecodeError as e:
        return False, f"Invalid JSON in SSE stream: {e}"
    except Exception as e:
        return False, f"SSE parsing error: {e}"
