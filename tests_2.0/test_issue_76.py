"""Regression tests for issue #76: a chunked vision request reports its own token counts.

Each chunk runs on a VisionRunner of its own, which records its counts and is discarded. The
handler read `usage` afterwards from the model's shared runner — the counts of whatever that
runner generated last, or the word estimate when it had generated nothing.
"""

from types import SimpleNamespace
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from mlxk2.core.server_base import app
from mlxk2.core.vision_runner import VisionRunner

PIXEL_PNG = (
    "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJ"
    "AAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg=="
)


def _backend(prompt_tokens, generation_tokens):
    """A `load_model` whose model reports these counts for every chunk, as mlx-vlm's result does."""

    def load(self):
        self._apply_chat_template = lambda *args, **kwargs: "prompt"
        self._generate = lambda *args, **kwargs: SimpleNamespace(
            text="an answer", finish_reason="length",
            prompt_tokens=prompt_tokens, generation_tokens=generation_tokens,
        )

    return load


def _chunked_request(**extra):
    parts = [{"type": "text", "text": "Describe"}]
    parts += [{"type": "image_url", "image_url": {"url": PIXEL_PNG}}] * 2
    return {"model": "m", "messages": [{"role": "user", "content": parts}], "chunk": 1, **extra}


def _runner(prompt, completion):
    return SimpleNamespace(last_prompt_tokens=prompt, last_completion_tokens=completion)


def test_the_response_sums_its_chunks_not_the_shared_runner():
    shared = VisionRunner("/mock/path", "mock-vision", verbose=False)
    # What the shared runner kept from an earlier request of its own
    shared.last_prompt_tokens, shared.last_completion_tokens = 999, 999
    with patch.object(VisionRunner, "load_model", _backend(3, 8)), \
         patch("mlxk2.core.server_base.get_or_load_model", return_value=shared):
        body = TestClient(app).post("/v1/chat/completions", json=_chunked_request()).json()
    assert body["usage"] == {"prompt_tokens": 6, "completion_tokens": 16, "total_tokens": 22}


def test_counts_add_up_over_the_chunks():
    from mlxk2.core.server.streaming import TokenCounts

    counts = TokenCounts()
    counts.add(_runner(3, 8))
    counts.add(_runner(4, 8))
    assert (counts.last_prompt_tokens, counts.last_completion_tokens) == (7, 16)


@pytest.mark.parametrize("first", [True, False], ids=["uncounted-first", "uncounted-last"])
@pytest.mark.parametrize("uncounted", [None, True, "3"], ids=["none", "bool", "str"])
def test_one_uncounted_chunk_leaves_the_sum_unknown(first, uncounted):
    """A later counted chunk must not pass for the whole request, nor must a bool pass for a count."""
    from mlxk2.core.server.streaming import TokenCounts

    chunks = [_runner(uncounted, uncounted), _runner(3, 8)]
    counts = TokenCounts()
    for chunk in chunks if first else reversed(chunks):
        counts.add(chunk)
    assert (counts.last_prompt_tokens, counts.last_completion_tokens) == (None, None)


def test_a_chunk_runner_without_the_attributes_counts_as_uncounted():
    from mlxk2.core.server.streaming import TokenCounts

    counts = TokenCounts()
    counts.add(object())
    counts.add(_runner(3, 8))
    assert counts.last_prompt_tokens is None
