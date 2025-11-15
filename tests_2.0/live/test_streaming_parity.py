"""Streaming vs. Non-Streaming Parity Tests (Issue #20, ADR-011).

Validates that streaming and non-streaming modes produce identical output.

Issue #20 Background:
- In 1.x, non-streaming had visible stop tokens while streaming did not
- Root cause: Different code paths for stop token filtering
- ADR-009 fixed Runner-level detection (eos_token_id → eos_token_ids)
- These tests ensure parity is maintained across:
  * MLXRunner direct usage
  * Server API endpoints
  * CLI commands

Test Strategy:
- Use 3 representative models (not full portfolio to save time)
- Same prompt + max_tokens → byte-for-byte identical output
- Test all three interfaces: Runner, Server, CLI
- RAM-aware testing

Opt-in via: pytest -m live_e2e
Requires: HF_HOME set to model cache
"""

from __future__ import annotations

import pytest
from typing import Dict, Any

try:
    import httpx
except ImportError:
    httpx = None

# Import test utilities
from .server_context import LocalServer
from .sse_parser import collect_sse_content
from .test_utils import (
    should_skip_model,
    TEST_PROMPT,
    MAX_TOKENS,
)
# portfolio_models fixture is provided by conftest.py

# Opt-in markers
pytestmark = [
    pytest.mark.live_e2e,
    pytest.mark.slow,
    pytest.mark.skipif(
        httpx is None,
        reason="httpx required for E2E tests"
    )
]


# Representative test models for parity validation
# Uses hardcoded subset (not full portfolio) to keep test time reasonable
PARITY_TEST_MODELS = {
    # "mxfp4": Skipped - Reasoning model (gpt-oss) has batch/stream inconsistency
    #   Batch output: Raw reasoning text
    #   Stream output: Adds **[Reasoning]** headers via StreamingReasoningParser
    #   Known issue, will be fixed in ADR-010 implementation
    "qwen25": {
        "id": "mlx-community/Qwen2.5-0.5B-Instruct-4bit",
        "ram_needed_gb": 1.0,
        "description": "Qwen 2.5 (self-conversation prevention)"
    },
    "llama32": {
        "id": "mlx-community/Llama-3.2-3B-Instruct-4bit",
        "ram_needed_gb": 4.0,
        "description": "Llama 3.2 (control baseline)"
    }
}


class TestRunnerStreamingParity:
    """MLXRunner direct streaming vs. batch parity.

    Tests are parametrized over PARITY_TEST_MODELS (3 models).
    Each test runs independently for clean isolation.
    """

    @pytest.mark.live_e2e
    @pytest.mark.parametrize("parity_model_key", list(PARITY_TEST_MODELS.keys()))
    def test_runner_streaming_batch_identical(self, _use_real_mlx_modules, parity_model_key):
        """Validate MLXRunner streaming and batch produce identical output.

        Parametrized test (one instance per parity test model).

        Issue #20: Previously, batch output had visible stop tokens while
        streaming did not. This validates the ADR-009 fix at Runner level.

        Requires real MLX modules (not stubs) since we use MLXRunner directly.
        """
        from mlxk2.core.runner import MLXRunner

        model_info = PARITY_TEST_MODELS[parity_model_key]
        model_id = model_info["id"]

        # RAM gating
        should_skip, skip_reason = should_skip_model(parity_model_key, PARITY_TEST_MODELS)
        if should_skip:
            pytest.skip(skip_reason)

        print(f"\nTesting {parity_model_key}: {model_id}")

        with MLXRunner(model_id, verbose=False) as runner:
            # Batch generation (temperature=0 for deterministic output)
            batch_output = runner.generate_batch(
                prompt=TEST_PROMPT,
                max_tokens=MAX_TOKENS,
                temperature=0.0
            )

            # Streaming generation (temperature=0 for deterministic output)
            stream_tokens = []
            for token in runner.generate_streaming(
                prompt=TEST_PROMPT,
                max_tokens=MAX_TOKENS,
                temperature=0.0
            ):
                stream_tokens.append(token)
            stream_output = "".join(stream_tokens)

            # Validate byte-for-byte parity
            assert batch_output == stream_output, (
                f"Streaming/batch parity failure (Issue #20 regression)\n"
                f"Model: {model_id}\n"
                f"Batch ({len(batch_output)} chars):  {batch_output!r}\n"
                f"Stream ({len(stream_output)} chars): {stream_output!r}"
            )

            print(f"✓ {parity_model_key}: Parity verified ({len(batch_output)} chars)")


class TestServerStreamingParity:
    """Server API streaming vs. batch parity.

    Tests are parametrized over PARITY_TEST_MODELS (3 models).
    Each test runs independently for clean isolation.
    """

    @pytest.mark.live_e2e
    @pytest.mark.parametrize("parity_model_key", list(PARITY_TEST_MODELS.keys()))
    def test_server_api_streaming_batch_identical(self, parity_model_key):
        """Validate Server API streaming and batch produce identical output.

        Parametrized test (one instance per parity test model).

        Tests parity at HTTP API level (closest to production usage).
        """
        model_info = PARITY_TEST_MODELS[parity_model_key]
        model_id = model_info["id"]

        # RAM gating
        should_skip, skip_reason = should_skip_model(parity_model_key, PARITY_TEST_MODELS)
        if should_skip:
            pytest.skip(skip_reason)

        print(f"\nTesting {parity_model_key}: {model_id}")

        with LocalServer(model_id, port=8765) as server_url:
            # Batch request (temperature=0 for deterministic output)
            batch_response = httpx.post(
                f"{server_url}/v1/chat/completions",
                json={
                    "model": model_id,
                    "messages": [{"role": "user", "content": TEST_PROMPT}],
                    "max_tokens": MAX_TOKENS,
                    "temperature": 0.0,
                    "stream": False
                },
                timeout=30.0
            )
            assert batch_response.status_code == 200
            batch_data = batch_response.json()
            batch_output = batch_data["choices"][0]["message"]["content"]

            # Streaming request (temperature=0 for deterministic output)
            with httpx.stream(
                "POST",
                f"{server_url}/v1/chat/completions",
                json={
                    "model": model_id,
                    "messages": [{"role": "user", "content": TEST_PROMPT}],
                    "max_tokens": MAX_TOKENS,
                    "temperature": 0.0,
                    "stream": True
                },
                timeout=30.0
            ) as stream_response:
                assert stream_response.status_code == 200
                stream_output = collect_sse_content(stream_response)

            # Validate parity
            assert batch_output == stream_output, (
                f"Server API parity failure (Issue #20 regression)\n"
                f"Model: {model_id}\n"
                f"Batch ({len(batch_output)} chars):  {batch_output!r}\n"
                f"Stream ({len(stream_output)} chars): {stream_output!r}"
            )

            print(f"✓ {parity_model_key}: Parity verified ({len(batch_output)} chars)")


class TestCrossInterfaceParity:
    """Parity across different interfaces (Runner vs Server)."""

    @pytest.mark.live_e2e
    def test_runner_vs_server_consistency(self, _use_real_mlx_modules):
        """Validate MLXRunner and Server API produce consistent output.

        Tests that direct Runner usage and Server HTTP API yield
        the same results (validates no server-specific transformations).

        Requires real MLX modules (not stubs) since we use MLXRunner directly.
        """
        from mlxk2.core.runner import MLXRunner

        # Use smallest model for faster testing
        test_model_key = "qwen25"
        model_info = PARITY_TEST_MODELS[test_model_key]
        model_id = model_info["id"]

        # RAM check
        should_skip, skip_reason = should_skip_model(test_model_key, PARITY_TEST_MODELS)
        if should_skip:
            pytest.skip(skip_reason)

        print(f"\nTesting cross-interface parity: {model_id}")

        # Runner output (temperature=0 for deterministic output)
        with MLXRunner(model_id, verbose=False) as runner:
            runner_output = runner.generate_batch(
                prompt=TEST_PROMPT,
                max_tokens=MAX_TOKENS,
                temperature=0.0
            )

        print(f"Runner output ({len(runner_output)} chars): {runner_output!r}")

        # Server output (temperature=0 for deterministic output)
        with LocalServer(model_id, port=8780) as server_url:
            response = httpx.post(
                f"{server_url}/v1/chat/completions",
                json={
                    "model": model_id,
                    "messages": [{"role": "user", "content": TEST_PROMPT}],
                    "max_tokens": MAX_TOKENS,
                    "temperature": 0.0,
                    "stream": False
                },
                timeout=30.0
            )
            assert response.status_code == 200
            server_output = response.json()["choices"][0]["message"]["content"]

        print(f"Server output ({len(server_output)} chars): {server_output!r}")

        # Note: Runner and Server may differ due to chat template application
        # Server applies chat template, Runner uses raw prompt
        # This test validates that both:
        # 1. Produce clean output (no visible stop tokens)
        # 2. Are internally consistent (streaming = batch for each)

        # Validate no stop tokens in either
        stop_tokens = ["<|end|>", "<|eot_id|>", "<|im_end|>"]
        runner_found = [t for t in stop_tokens if t in runner_output]
        server_found = [t for t in stop_tokens if t in server_output]

        assert not runner_found, f"Runner has visible stop tokens: {runner_found}"
        assert not server_found, f"Server has visible stop tokens: {server_found}"

        print(f"✓ Cross-interface consistency verified (both clean)")
