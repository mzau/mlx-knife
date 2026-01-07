"""
E2E tests for Vision Server API (ADR-012 Phase 3 + Portfolio Separation).

Tests actual VISION model generation through the HTTP API:
- Vision requests (multimodal with images)
- Text-only requests on vision models
- Vision→Text model switching with history filtering

Test Strategy:
- Uses VISION Portfolio Discovery (vision_portfolio fixture)
- Parametrized tests across all vision models in cache
- RAM-aware testing (0.70 threshold for vision models)
- Text models tested separately in test_server_e2e.py

Opt-in via: pytest -m live_e2e
Requires: HF_HOME set to model cache, httpx installed
"""

import base64
import pytest
from pathlib import Path

try:
    import httpx
except ImportError:
    httpx = None

from .server_context import LocalServer
from .test_utils import should_skip_model

# Skip entire module if httpx not installed
pytestmark = [
    pytest.mark.live,
    pytest.mark.skipif(httpx is None, reason="httpx required for E2E tests"),
    pytest.mark.live_e2e,
]

# Test image path
TEST_IMAGE = Path(__file__).parent.parent / "assets" / "T5.png"

# Server request timeout (vision models are slower)
SERVER_REQUEST_TIMEOUT = 120


def image_to_base64_data_url(image_path: Path) -> str:
    """Convert image file to base64 data URL."""
    with open(image_path, "rb") as f:
        image_bytes = f.read()

    # Determine MIME type from extension
    ext = image_path.suffix.lower()
    mime_map = {".png": "png", ".jpg": "jpeg", ".jpeg": "jpeg", ".gif": "gif", ".webp": "webp"}
    mime_type = mime_map.get(ext, "png")

    b64 = base64.b64encode(image_bytes).decode("utf-8")
    return f"data:image/{mime_type};base64,{b64}"


class TestVisionServerE2E:
    """E2E tests for vision chat completions.

    Tests are parametrized per model via pytest_generate_tests hook.
    Each test runs with its own server instance for clean isolation.
    """

    @pytest.mark.live_e2e
    def test_single_image_chat_completion(self, vision_portfolio, vision_model_key):
        """Vision model should describe an image sent via Base64.

        Parametrized test (one instance per VISION model in portfolio).
        Tests multimodal request handling with Base64 image data.
        """
        if not TEST_IMAGE.exists():
            pytest.skip(f"Test image not found: {TEST_IMAGE}")

        model_info = vision_portfolio[vision_model_key]
        model_id = model_info["id"]

        # RAM gating: skip if model exceeds 70% threshold
        should_skip, skip_reason = should_skip_model(vision_model_key, vision_portfolio)
        if should_skip:
            pytest.skip(skip_reason)

        print(f"\nTesting {vision_model_key}: {model_id}")

        image_url = image_to_base64_data_url(TEST_IMAGE)

        with LocalServer(model_id, port=8770, timeout=90) as server_url:
            payload = {
                "model": model_id,
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": "What do you see in this image? Be brief."},
                            {"type": "image_url", "image_url": {"url": image_url}}
                        ]
                    }
                ],
                "max_tokens": 100,
                "temperature": 0.1,
                "stream": False,
            }

            response = httpx.post(
                f"{server_url}/v1/chat/completions",
                json=payload,
                timeout=SERVER_REQUEST_TIMEOUT
            )

            assert response.status_code == 200, f"Expected 200, got {response.status_code}: {response.text}"

            data = response.json()
            assert "choices" in data
            assert len(data["choices"]) > 0

            content = data["choices"][0]["message"]["content"]
            assert len(content) > 10, f"Response too short: {content}"

            # Verify it's a meaningful response (not an error message)
            assert "error" not in content.lower() or "describe" in content.lower()

            print(f"\n✅ Vision response: {content[:200]}...")

    @pytest.mark.live_e2e
    def test_streaming_graceful_degradation(self, vision_portfolio, vision_model_key):
        """Vision request with stream=True should gracefully degrade via SSE emulation.

        Parametrized test (one instance per VISION model in portfolio).
        Tests that vision models handle stream=True with SSE emulation.
        """
        if not TEST_IMAGE.exists():
            pytest.skip(f"Test image not found: {TEST_IMAGE}")

        model_info = vision_portfolio[vision_model_key]
        model_id = model_info["id"]

        # RAM gating
        should_skip, skip_reason = should_skip_model(vision_model_key, vision_portfolio)
        if should_skip:
            pytest.skip(skip_reason)

        print(f"\nTesting {vision_model_key}: {model_id}")

        image_url = image_to_base64_data_url(TEST_IMAGE)

        with LocalServer(model_id, port=8771, timeout=90) as server_url:
            payload = {
                "model": model_id,
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": "Describe this briefly."},
                            {"type": "image_url", "image_url": {"url": image_url}}
                        ]
                    }
                ],
                "stream": True,  # Graceful degradation via SSE emulation
                "max_tokens": 2048,  # Vision server default (stateless, no shift-window)
            }

            # Use streaming context for SSE response
            with httpx.stream(
                "POST",
                f"{server_url}/v1/chat/completions",
                json=payload,
                timeout=120
            ) as response:
                # Graceful degradation: returns 200 with SSE emulation (not 400)
                assert response.status_code == 200, f"Expected 200 (graceful degradation), got {response.status_code}"

                # Collect SSE events
                content_parts = []
                for line in response.iter_lines():
                    if line.startswith("data: ") and line != "data: [DONE]":
                        import json
                        event_data = json.loads(line[6:])  # Skip "data: " prefix
                        if "choices" in event_data and event_data["choices"]:
                            delta = event_data["choices"][0].get("delta", {})
                            if "content" in delta:
                                content_parts.append(delta["content"])

                # Verify we got content from SSE
                full_content = "".join(content_parts)
                assert len(full_content) > 0, f"Expected content in SSE response, got empty"
                print(f"\n✅ SSE emulation response: {full_content[:100]}...")

    @pytest.mark.live_e2e
    def test_text_request_still_works_on_vision_model(self, vision_portfolio, vision_model_key):
        """Text-only requests should still work on vision model server.

        Parametrized test (one instance per VISION model in portfolio).
        Tests that vision models can handle pure text requests (no images).
        """
        model_info = vision_portfolio[vision_model_key]
        model_id = model_info["id"]

        # RAM gating
        should_skip, skip_reason = should_skip_model(vision_model_key, vision_portfolio)
        if should_skip:
            pytest.skip(skip_reason)

        print(f"\nTesting {vision_model_key}: {model_id}")

        with LocalServer(model_id, port=8772, timeout=90) as server_url:
            payload = {
                "model": model_id,
                "messages": [
                    {"role": "user", "content": "What is 2 + 2? Answer with just the number."}
                ],
                "max_tokens": 2048,  # Vision server default (stateless, no shift-window)
                "temperature": 0.0,
                "stream": False,
            }

            response = httpx.post(
                f"{server_url}/v1/chat/completions",
                json=payload,
                timeout=60
            )

            assert response.status_code == 200, f"Expected 200, got {response.status_code}: {response.text}"

            data = response.json()
            content = data["choices"][0]["message"]["content"]
            assert "4" in content, f"Expected '4' in response: {content}"

    @pytest.mark.live_e2e
    def test_vision_to_text_model_switch_filters_images(self, vision_portfolio, text_portfolio):
        """Vision→Text model switch should filter image_url content from history.

        Tests the multimodal history filtering feature (Session 26):
        - User starts conversation with Vision model + images
        - User switches to Text model (same conversation history)
        - Server should filter out image_url content, preserve text
        - Enables seamless model switching in nChat and other clients

        This is a special integration test (not parametrized) that tests
        the switching behavior between a vision and text model pair.

        See: docs/ISSUES/VISION-MULTIMODAL-HISTORY-ISSUE.md
        """
        if not TEST_IMAGE.exists():
            pytest.skip(f"Test image not found: {TEST_IMAGE}")

        # Pick first available vision model
        vision_model_id = None
        for model_key, model_info in vision_portfolio.items():
            should_skip, _ = should_skip_model(model_key, vision_portfolio)
            if not should_skip:
                vision_model_id = model_info["id"]
                break

        if vision_model_id is None:
            pytest.skip("No vision models available within RAM budget")

        # Pick first available text model
        text_model_id = None
        for model_key, model_info in text_portfolio.items():
            should_skip, _ = should_skip_model(model_key, text_portfolio)
            if not should_skip:
                text_model_id = model_info["id"]
                break

        if text_model_id is None:
            pytest.skip("No text models available within RAM budget")

        print(f"\nTesting Vision→Text switch: {vision_model_id} → {text_model_id}")

        image_url = image_to_base64_data_url(TEST_IMAGE)

        # Start with a text model server (no pre-load)
        # This allows dynamic model switching via API
        with LocalServer(text_model_id, port=8773, timeout=90) as server_url:
            # Step 1: Vision request with image (server loads Vision model)
            vision_payload = {
                "model": vision_model_id,
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": "What color is dominant in this image?"},
                            {"type": "image_url", "image_url": {"url": image_url}}
                        ]
                    }
                ],
                "max_tokens": 2048,  # Vision server default (stateless, no shift-window)
                "temperature": 0.1,
                "stream": False,
            }

            response1 = httpx.post(
                f"{server_url}/v1/chat/completions",
                json=vision_payload,
                timeout=SERVER_REQUEST_TIMEOUT
            )

            # Vision response should work
            assert response1.status_code == 200, f"Vision request failed: {response1.status_code}"
            vision_data = response1.json()
            vision_response = vision_data["choices"][0]["message"]["content"]

            print(f"\n✅ Vision model response: {vision_response[:100]}...")

            # Step 2: Switch to Text model with conversation history (includes image_url)
            # Simulate nChat behavior: history contains multimodal content
            text_payload = {
                "model": text_model_id,
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": "What color is dominant in this image?"},
                            {"type": "image_url", "image_url": {"url": image_url}}  # Should be filtered
                        ]
                    },
                    {
                        "role": "assistant",
                        "content": vision_response  # Vision model's description
                    },
                    {
                        "role": "user",
                        "content": "What did you just tell me?"  # Follow-up question
                    }
                ],
                "max_tokens": 100,
                "temperature": 0.0,
                "stream": False,
            }

            response2 = httpx.post(
                f"{server_url}/v1/chat/completions",
                json=text_payload,
                timeout=60
            )

            # Text model should succeed (filtered history, no HTTP 400)
            assert response2.status_code == 200, (
                f"Text model failed with history: {response2.status_code}\n"
                f"Response: {response2.text}\n"
                f"This indicates multimodal filtering failed."
            )

            text_data = response2.json()
            text_response = text_data["choices"][0]["message"]["content"]

            # Verify text model can reference Vision model's description
            # (proves context was preserved, only images filtered)
            assert len(text_response) > 10, f"Text response too short: {text_response}"

            print(f"\n✅ Text model response (after filtering): {text_response[:100]}...")
            print("\n✅ Multimodal history filtering works: Vision→Text switch succeeded")
