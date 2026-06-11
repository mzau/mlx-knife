"""End-to-end FIM (Fill-In-the-Middle) test with a real coder model (ADR-011).

This is the sole regression guard for FIM (the mocked unit test was dropped as
near-tautological). It verifies the mlx-knife *enabler* end-to-end: a real FIM-trained coder
model, served through `/v1/completions` — which applies **no** chat template and forwards the
raw prompt verbatim — actually fills the gap, i.e. the FIM control tokens are honored and the
model returns the correct middle. If the enabler regressed (e.g. a chat template were
applied), the FIM tokens would be mangled and this test would fail. (Background: Issue #55.)

Model selection (a FIM-trained *coder* model is required — Qwen2.5-Coder / Qwen3-Coder):
  1. `MLXK_FIM_MODEL` env var (caller-supplied: an HF id, an `mlxk list` name, or a local
     path) — the primary path, since coder models often live outside the HF cache.
  2. else: auto-discover a model with "coder" in its name from the text portfolio.
  3. else: skip.

Opt-in via: pytest -m live_e2e
Run example (also prints the real completion, useful for documenting the behavior):
    MLXK_FIM_MODEL=mlx-community/Qwen2.5-Coder-1.5B-Instruct-4bit \
    HF_HOME=/path/to/cache \
    pytest tests_2.0/live/test_fim_e2e.py -m live_e2e -v -s
"""

from __future__ import annotations

import os

import pytest

try:
    import httpx
except ImportError:
    httpx = None

from .server_context import LocalServer
from .test_utils import should_skip_model

# Match the request timeout used by the other E2E tests (server_e2e Session 22).
SERVER_REQUEST_TIMEOUT = 45.0

# Opt-in markers (same set as test_server_e2e.py).
pytestmark = [
    pytest.mark.live,
    pytest.mark.live_e2e,
    pytest.mark.slow,
    pytest.mark.skipif(httpx is None, reason="httpx required for E2E tests (pip install httpx)"),
]

# A FIM prompt in the Qwen-Coder format. The editor (not mlx-knife) assembles this; here the
# test plays the role of the editor. The model must complete the gap between prefix and suffix.
FIM_PROMPT = (
    "<|fim_prefix|>def fibonacci(n):\n    if n < 2:\n        return n\n    return "
    "<|fim_suffix|>\n\nprint(fibonacci(10))\n<|fim_middle|>"
)


def _find_coder_in_portfolio(text_portfolio) -> str | None:
    """Return the first RAM-fitting model whose name contains 'coder', else None."""
    for model_key, model_info in text_portfolio.items():
        model_id = model_info["id"]
        if "coder" in model_key.lower() or "coder" in model_id.lower():
            should_skip, _ = should_skip_model(model_key, text_portfolio)
            if not should_skip:
                return model_id
    return None


@pytest.mark.live_e2e
def test_fim_completion_end_to_end(request):
    """A real coder model served via /v1/completions fills a FIM gap correctly."""
    model = os.environ.get("MLXK_FIM_MODEL")
    if not model:
        # Lazy: only run portfolio discovery when no explicit model was supplied.
        text_portfolio = request.getfixturevalue("text_portfolio")
        model = _find_coder_in_portfolio(text_portfolio)
    if not model:
        pytest.skip(
            "No FIM-capable coder model available. Set MLXK_FIM_MODEL=<coder id/path> "
            "(e.g. mlx-community/Qwen2.5-Coder-1.5B-Instruct-4bit) or add a Qwen-Coder to the cache."
        )

    print(f"\nFIM e2e model: {model}")

    with LocalServer(model) as server_url:
        response = httpx.post(
            f"{server_url}/v1/completions",
            json={
                "model": model,
                "prompt": FIM_PROMPT,
                "max_tokens": 40,
                "temperature": 0,
                "stream": False,
            },
            timeout=SERVER_REQUEST_TIMEOUT,
        )

        assert response.status_code == 200, f"Expected 200, got {response.status_code}: {response.text}"
        data = response.json()
        assert data.get("object") == "text_completion"
        assert data.get("choices"), "Empty choices array"

        text = data["choices"][0]["text"]
        print(f"FIM completion: {text!r}")

        # End-to-end proof that FIM was honored: the gap is filled with the recursive call.
        # Whitespace-normalized so the assertion is robust to formatting differences.
        normalized = "".join(text.split())
        assert "fibonacci(n-1)" in normalized and "fibonacci(n-2)" in normalized, (
            f"FIM completion did not contain the fibonacci recursion — the model may not be "
            f"FIM-capable, or the FIM tokens were not honored.\nGot: {text!r}"
        )

        # Raw passthrough end to end: FIM control tokens must not leak into the output.
        assert "<|fim_" not in text, f"FIM control token leaked into output: {text!r}"
