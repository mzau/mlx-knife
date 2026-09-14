"""Server E2E tests with real TEXT models (ADR-011 + Portfolio Separation).

Validates server/HTTP API endpoints across TEXT-ONLY model portfolio:
- Health check endpoint
- Model listing endpoint
- Chat completions (batch and streaming)
- Text completions (batch and streaming)
- Stop token filtering (Issue #20/#32)
- Error envelopes (ADR-004)

Test Strategy:
- Uses TEXT Portfolio Discovery (text_portfolio fixture)
- Vision models tested separately in test_vision_server_e2e.py
- RAM-aware testing (progressive budget: 40%-70%)
- Subprocess-based server lifecycle (true E2E)
- OpenAI-compatible API validation

Opt-in via: pytest -m live_e2e
Requires: HF_HOME set to model cache, httpx installed
"""

from __future__ import annotations

import threading
import time

import pytest
from typing import Dict, Any

try:
    import httpx
except ImportError:
    httpx = None

# Import test utilities
from .server_context import LocalServer
from .sse_parser import parse_sse_stream, collect_sse_content, validate_sse_format
from .test_utils import (
    model_timeout,
    should_skip_model,
    TEST_PROMPT,
    MAX_TOKENS,
)
# text_portfolio fixture is provided by conftest.py (Portfolio Separation)

# Server request base limit (raised from 30s to 45s).
# Accounts for: baseline (15s) + probe/policy overhead (2.7s) + generation + safety margin.
# This is the WORK term and the lower bound; model_timeout() adds what a model
# of the given size costs on top (test_utils.LOAD_ALLOWANCE_S).
SERVER_REQUEST_TIMEOUT = 45.0
# /v1/models can be slower due to cache scans + runtime checks
MODEL_LIST_TIMEOUT = 20.0
# Kubernetes' default probe timeout: a supervisor's patience, not a latency target. The
# generation below outlasts it on any model, so a probe queued behind it fails.
HEALTH_PROBE_TIMEOUT = 1.0
# Keeps a model writing until the token budget ends it, so the generation lasts
LONG_PROMPT = (
    "Write an extremely long, exhaustive, chapter-by-chapter history of sailing ships. "
    "Do not stop early."
)

# Opt-in markers
pytestmark = [
    pytest.mark.live,
    pytest.mark.live_e2e,
    pytest.mark.slow,
    pytest.mark.skipif(
        httpx is None,
        reason="httpx required for E2E tests (pip install httpx)"
    )
]


class TestServerHealthEndpoints:
    """Basic health and metadata endpoints."""

    @pytest.mark.live_e2e
    def test_health_endpoint(self, text_portfolio):
        """Validate /health endpoint returns 200 OK.

        Tests server basic liveness without model dependency.
        Uses first available TEXT model from portfolio to start server.
        """
        # Use first model that fits in RAM
        test_model = None
        for model_key, model_info in text_portfolio.items():
            should_skip, _ = should_skip_model(model_key, text_portfolio)
            if not should_skip:
                test_model = model_info["id"]
                break

        if test_model is None:
            pytest.skip("No text models available within RAM budget")

        with LocalServer(test_model) as server_url:
            response = httpx.get(f"{server_url}/health")

            assert response.status_code == 200
            data = response.json()
            # ADR-029: `healthy` is the CLI's file-integrity word and stays off the HTTP surface
            assert data.get("status") == "ok"
            assert "healthy" not in response.text

    @pytest.mark.live_e2e
    def test_health_answers_while_generating(self, text_portfolio):
        """GET /health answers while a batch and a stream share the model thread (#64).

        The loop used to run a non-streaming generation itself, so a probe got no answer
        for the whole of it. The probe timeout is the bound, not a measurement: a probe
        waiting behind the generation would outlast it. Both requests must finish too.
        """
        candidates = [
            info for key, info in text_portfolio.items()
            if not should_skip_model(key, text_portfolio)[0]
        ]
        if not candidates:
            pytest.skip("No text models available within RAM budget")
        test_model = min(candidates, key=lambda info: info.get("ram_needed_gb") or 0)["id"]
        budget = model_timeout(SERVER_REQUEST_TIMEOUT, test_model)
        payload = {
            "model": test_model,
            "messages": [{"role": "user", "content": LONG_PROMPT}],
            "max_tokens": 1000,
            "temperature": 0.0,
        }

        with LocalServer(test_model) as server_url:
            results: Dict[str, Any] = {}

            def batch():
                results["batch"] = httpx.post(
                    f"{server_url}/v1/chat/completions", json=payload, timeout=budget
                )

            def stream():
                with httpx.stream(
                    "POST", f"{server_url}/v1/chat/completions",
                    json={**payload, "stream": True}, timeout=budget,
                ) as response:
                    results["stream_status"] = response.status_code
                    results["stream"] = [line for line in response.iter_lines() if line]

            workers = [threading.Thread(target=batch), threading.Thread(target=stream)]
            for worker in workers:
                worker.start()
            probes = []
            while any(worker.is_alive() for worker in workers):
                probes.append(httpx.get(f"{server_url}/health", timeout=HEALTH_PROBE_TIMEOUT).status_code)
                time.sleep(0.25)
            for worker in workers:
                worker.join()

            assert len(probes) >= 2, "the generations ended before a probe could overlap them"
            assert set(probes) == {200}
            assert results["batch"].status_code == 200
            assert results["batch"].json()["choices"][0]["message"]["content"]
            assert results["stream_status"] == 200
            assert results["stream"][-1] == "data: [DONE]"
            assert not any('"error"' in line for line in results["stream"])

    @pytest.mark.live_e2e
    def test_v1_models_list(self, text_portfolio):
        """Validate /v1/models returns loaded model.

        Tests model metadata endpoint with pre-loaded TEXT model.
        """
        # Use first model that fits in RAM
        test_model = None
        for model_key, model_info in text_portfolio.items():
            should_skip, _ = should_skip_model(model_key, text_portfolio)
            if not should_skip:
                test_model = model_info["id"]
                break

        if test_model is None:
            pytest.skip("No text models available within RAM budget")

        with LocalServer(test_model) as server_url:
            response = httpx.get(f"{server_url}/v1/models", timeout=MODEL_LIST_TIMEOUT)

            assert response.status_code == 200
            data = response.json()

            # Validate OpenAI-compatible structure
            assert "data" in data
            assert isinstance(data["data"], list)
            assert len(data["data"]) > 0

            # Validate loaded model is listed.
            # Workspace models (absolute-path ids in the portfolio) are
            # advertised by their directory basename since #58; cache models
            # keep their org/name id.
            model_ids = [m["id"] for m in data["data"]]
            expected_id = (
                test_model.rsplit("/", 1)[-1]
                if test_model.startswith("/")
                else test_model
            )
            assert expected_id in model_ids

            # The preloaded model is the one in memory, and the only one
            assert all("loaded" in m for m in data["data"])
            assert [m["id"] for m in data["data"] if m["loaded"]] == [expected_id]


class TestChatCompletionsBatch:
    """Non-streaming chat completion tests across portfolio.

    Tests are parametrized per model via pytest_generate_tests hook.
    Each test runs with its own server instance for clean isolation.
    """

    @pytest.mark.live_e2e
    @pytest.mark.benchmark_inference
    def test_chat_completions_batch(self, text_portfolio, text_model_key, report_benchmark):
        """Validate non-streaming chat completions.

        Parametrized test (one instance per TEXT model in portfolio).

        Tests:
        - Response structure (OpenAI-compatible)
        - Stop token filtering (Issue #32)
        - Error handling
        """
        model_info = text_portfolio[text_model_key]
        model_id = model_info["id"]

        # RAM gating: skip if model exceeds budget
        should_skip, skip_reason = should_skip_model(text_model_key, text_portfolio)
        if should_skip:
            pytest.skip(skip_reason)

        print(f"\nTesting {text_model_key}: {model_id}")

        with LocalServer(model_id, port=8765) as server_url:
            # Non-streaming chat completion
            response = httpx.post(
                f"{server_url}/v1/chat/completions",
                json={
                    "model": model_id,
                    "messages": [
                        {"role": "user", "content": TEST_PROMPT}
                    ],
                    "max_tokens": MAX_TOKENS,
                    "stream": False
                },
                timeout=model_timeout(SERVER_REQUEST_TIMEOUT, model_id)
            )

            assert response.status_code == 200, f"Expected 200, got {response.status_code}"
            data = response.json()

            # Validate OpenAI structure
            assert "id" in data, "Missing 'id' field"
            assert "object" in data, "Missing 'object' field"
            assert data["object"] == "chat.completion"
            assert "choices" in data, "Missing 'choices' field"
            assert len(data["choices"]) > 0, "Empty choices array"

            # Extract response text
            choice = data["choices"][0]
            assert "message" in choice, "Missing 'message' field"
            assert "content" in choice["message"], "Missing 'content' field"
            content = choice["message"]["content"]

            # Validate stop token filtering (Issue #32)
            stop_tokens = [
                "<|end|>", "<|eot_id|>", "<|im_end|>",
                "<|endoftext|>", "</s>", "<|end_of_text|>"
            ]
            found_tokens = [t for t in stop_tokens if t in content]
            assert not found_tokens, (
                f"Model {model_id} has visible stop tokens: {found_tokens}\n"
                f"Content: {content!r}"
            )

            print(f"✓ {text_model_key}: Passed (output: {len(content)} chars)")

            # Benchmark reporting (ADR-013 Phase 0)
            # Extract usage statistics if available
            performance = {}
            if "usage" in data:
                usage = data["usage"]
                performance["prompt_tokens"] = usage.get("prompt_tokens", 0)
                performance["completion_tokens"] = usage.get("completion_tokens", 0)

            report_benchmark(
                performance=performance if performance else None,
                stop_tokens={
                    "configured": stop_tokens,
                    "detected": found_tokens,
                    "workaround": "none",
                    "leaked": len(found_tokens) > 0
                }
            )


class TestChatCompletionsStreaming:
    """SSE streaming chat completion tests across portfolio.

    Tests are parametrized per model via pytest_generate_tests hook.
    Each test runs with its own server instance for clean isolation.
    """

    @pytest.mark.live_e2e
    @pytest.mark.benchmark_inference
    def test_chat_completions_streaming(self, text_portfolio, text_model_key, report_benchmark):
        """Validate SSE streaming chat completions.

        Parametrized test (one instance per TEXT model in portfolio).

        Tests:
        - SSE format compliance (data: lines, [DONE] sentinel)
        - Chunk structure (OpenAI-compatible)
        - Stop token filtering (Issue #32)
        - Stream completion
        """
        model_info = text_portfolio[text_model_key]
        model_id = model_info["id"]

        # RAM gating
        should_skip, skip_reason = should_skip_model(text_model_key, text_portfolio)
        if should_skip:
            pytest.skip(skip_reason)

        print(f"\nTesting {text_model_key}: {model_id}")

        with LocalServer(model_id, port=8765) as server_url:
            # Streaming chat completion
            with httpx.stream(
                "POST",
                f"{server_url}/v1/chat/completions",
                json={
                    "model": model_id,
                    "messages": [
                        {"role": "user", "content": TEST_PROMPT}
                    ],
                    "max_tokens": MAX_TOKENS,
                    "stream": True
                },
                timeout=model_timeout(SERVER_REQUEST_TIMEOUT, model_id)
            ) as response:
                assert response.status_code == 200

                # Validate SSE format
                valid, error_msg = validate_sse_format(response)
                if not valid:
                    # Response consumed by validation, need to restart
                    raise AssertionError(f"SSE format invalid: {error_msg}")

            # Re-run to collect content (validation consumed stream)
            with httpx.stream(
                "POST",
                f"{server_url}/v1/chat/completions",
                json={
                    "model": model_id,
                    "messages": [
                        {"role": "user", "content": TEST_PROMPT}
                    ],
                    "max_tokens": MAX_TOKENS,
                    "stream": True
                },
                timeout=model_timeout(SERVER_REQUEST_TIMEOUT, model_id)
            ) as response:
                content = collect_sse_content(response)

            # Validate stop token filtering
            stop_tokens = [
                "<|end|>", "<|eot_id|>", "<|im_end|>",
                "<|endoftext|>", "</s>", "<|end_of_text|>"
            ]
            found_tokens = [t for t in stop_tokens if t in content]
            assert not found_tokens, (
                f"Model {model_id} has visible stop tokens in stream: {found_tokens}\n"
                f"Content: {content!r}"
            )

            print(f"✓ {text_model_key}: Passed (streamed: {len(content)} chars)")

            # Benchmark reporting (ADR-013 Phase 0)
            report_benchmark(stop_tokens={
                "configured": stop_tokens,
                "detected": found_tokens,
                "workaround": "none",
                "leaked": len(found_tokens) > 0
            })


class TestCompletionsBatch:
    """Non-streaming text completion tests."""

    @pytest.mark.live_e2e
    def test_completions_batch_basic(self, text_portfolio):
        """Validate non-streaming text completions.

        Tests basic /v1/completions endpoint with first available TEXT model.
        """
        # Use first model that fits in RAM
        test_model = None
        test_model_key = None
        for model_key, model_info in text_portfolio.items():
            should_skip, _ = should_skip_model(model_key, text_portfolio)
            if not should_skip:
                test_model = model_info["id"]
                test_model_key = model_key
                break

        if test_model is None:
            pytest.skip("No text models available within RAM budget")

        print(f"\nTesting {test_model_key}: {test_model}")

        with LocalServer(test_model) as server_url:
            response = httpx.post(
                f"{server_url}/v1/completions",
                json={
                    "model": test_model,
                    "prompt": TEST_PROMPT,
                    "max_tokens": MAX_TOKENS,
                    "stream": False
                },
                timeout=model_timeout(SERVER_REQUEST_TIMEOUT, test_model)
            )

            assert response.status_code == 200
            data = response.json()

            # Validate structure
            assert "id" in data
            assert "object" in data
            assert data["object"] == "text_completion"
            assert "choices" in data
            assert len(data["choices"]) > 0

            # Validate content
            choice = data["choices"][0]
            assert "text" in choice
            content = choice["text"]

            # Check stop tokens
            stop_tokens = ["<|end|>", "<|eot_id|>", "<|im_end|>"]
            found_tokens = [t for t in stop_tokens if t in content]
            assert not found_tokens, f"Visible stop tokens: {found_tokens}"

            print(f"✓ Passed (output: {len(content)} chars)")


class TestCompletionsStreaming:
    """SSE streaming text completion tests."""

    @pytest.mark.live_e2e
    def test_completions_streaming_basic(self, text_portfolio):
        """Validate SSE streaming text completions.

        Tests /v1/completions with stream=True (TEXT models only).
        """
        # Use first model that fits in RAM
        test_model = None
        test_model_key = None
        for model_key, model_info in text_portfolio.items():
            should_skip, _ = should_skip_model(model_key, text_portfolio)
            if not should_skip:
                test_model = model_info["id"]
                test_model_key = model_key
                break

        if test_model is None:
            pytest.skip("No text models available within RAM budget")

        print(f"\nTesting {test_model_key}: {test_model}")

        with LocalServer(test_model) as server_url:
            with httpx.stream(
                "POST",
                f"{server_url}/v1/completions",
                json={
                    "model": test_model,
                    "prompt": TEST_PROMPT,
                    "max_tokens": MAX_TOKENS,
                    "stream": True
                },
                timeout=model_timeout(SERVER_REQUEST_TIMEOUT, test_model)
            ) as response:
                assert response.status_code == 200

                # Collect content
                content_parts = []
                for chunk in parse_sse_stream(response):
                    if "choices" in chunk:
                        for choice in chunk["choices"]:
                            text = choice.get("text", "")
                            if text:
                                content_parts.append(text)

                content = "".join(content_parts)

                # Check stop tokens
                stop_tokens = ["<|end|>", "<|eot_id|>", "<|im_end|>"]
                found_tokens = [t for t in stop_tokens if t in content]
                assert not found_tokens, f"Visible stop tokens: {found_tokens}"

                print(f"✓ Passed (streamed: {len(content)} chars)")
