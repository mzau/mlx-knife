"""
SSE streaming generators for MLX Knife server.

Extracted from server_base.py as part of Phase 1 refactoring.
These functions generate OpenAI-compatible SSE events for streaming responses.
"""

from __future__ import annotations

import json
import os
import time
import uuid
from collections.abc import AsyncGenerator
from threading import Event
from typing import TYPE_CHECKING, Dict, List, Optional

if TYPE_CHECKING:
    from ..runner import MLXRunner


def _get_logger():
    """Lazy import logger to avoid circular dependencies."""
    from ...logging import get_logger
    return get_logger()


async def generate_completion_stream(
    runner: "MLXRunner",
    prompt: str,
    request_model: str,
    max_tokens: Optional[int],
    temperature: float,
    top_p: float,
    repetition_penalty: float,
    stop: Optional[List[str]],
    shutdown_event: Event,
) -> AsyncGenerator[str, None]:
    """Generate streaming completion response.

    Args:
        runner: MLXRunner instance
        prompt: User prompt
        request_model: Model name for response
        max_tokens: Max tokens to generate
        temperature: Sampling temperature
        top_p: Top-p sampling
        repetition_penalty: Repetition penalty
        stop: Stop sequences
        shutdown_event: Thread event to check for shutdown
    """
    logger = _get_logger()
    completion_id = f"cmpl-{uuid.uuid4()}"
    created = int(time.time())

    # Yield initial response
    initial_response = {
        "id": completion_id,
        "object": "text_completion",
        "created": created,
        "model": request_model,
        "choices": [
            {
                "index": 0,
                "text": "",
                "logprobs": None,
                "finish_reason": None
            }
        ]
    }

    yield f"data: {json.dumps(initial_response)}\n\n"

    # Stream tokens
    try:
        token_count = 0
        for token in runner.generate_streaming(
            prompt=prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            use_chat_template=False  # Raw completion mode
        ):
            # Stop promptly if server is shutting down
            if shutdown_event.is_set():
                raise KeyboardInterrupt()
            token_count += 1

            chunk_response = {
                "id": completion_id,
                "object": "text_completion",
                "created": created,
                "model": request_model,
                "choices": [
                    {
                        "index": 0,
                        "text": token,
                        "logprobs": None,
                        "finish_reason": None
                    }
                ]
            }

            yield f"data: {json.dumps(chunk_response)}\n\n"

            # Check for stop sequences
            if stop:
                stop_sequences = stop if isinstance(stop, list) else [stop]
                if any(s in token for s in stop_sequences):
                    break

    except KeyboardInterrupt:
        # During shutdown/disconnect avoid extra logs; best-effort cleanup
        if not shutdown_event.is_set():
            try:
                import mlx.core as mx
                mx.clear_cache()
            except (ImportError, AttributeError):
                pass  # MLX not installed or API changed
            except Exception as e:
                logger.debug(f"Metal cache cleanup failed: {e}")
            # Try to send an interrupt marker if client still connected
            try:
                interrupt_response = {
                    "id": completion_id,
                    "object": "text_completion",
                    "created": created,
                    "model": request_model,
                    "choices": [
                        {
                            "index": 0,
                            "text": "\n\n[Generation interrupted by user]",
                            "logprobs": None,
                            "finish_reason": "stop"
                        }
                    ]
                }
                yield f"data: {json.dumps(interrupt_response)}\n\n"
            except Exception:
                pass
        return

    except Exception as e:
        error_response = {
            "id": completion_id,
            "object": "text_completion",
            "created": created,
            "model": request_model,
            "choices": [
                {
                    "index": 0,
                    "text": "",
                    "logprobs": None,
                    "finish_reason": "error"
                }
            ],
            "error": str(e)
        }
        yield f"data: {json.dumps(error_response)}\n\n"

    # Final response (skip if shutting down)
    if shutdown_event.is_set():
        return
    final_response = {
        "id": completion_id,
        "object": "text_completion",
        "created": created,
        "model": request_model,
        "choices": [
            {
                "index": 0,
                "text": "",
                "logprobs": None,
                "finish_reason": "stop"
            }
        ]
    }

    yield f"data: {json.dumps(final_response)}\n\n"
    yield "data: [DONE]\n\n"


async def generate_chat_stream(
    runner: "MLXRunner",
    messages: List[Dict[str, str]],
    request_model: str,
    max_tokens: Optional[int],
    temperature: float,
    top_p: float,
    repetition_penalty: float,
    stop: Optional[List[str]],
    shutdown_event: Event,
) -> AsyncGenerator[str, None]:
    """Generate streaming chat completion response.

    Args:
        runner: MLXRunner instance
        messages: List of message dicts (already formatted for runner)
        request_model: Model name for response
        max_tokens: Max tokens to generate
        temperature: Sampling temperature
        top_p: Top-p sampling
        repetition_penalty: Repetition penalty
        stop: Stop sequences
        shutdown_event: Thread event to check for shutdown
    """
    logger = _get_logger()
    completion_id = f"chatcmpl-{uuid.uuid4()}"
    created = int(time.time())

    # Let the runner format with chat templates
    prompt = runner._format_conversation(messages)

    # Yield initial response
    initial_response = {
        "id": completion_id,
        "object": "chat.completion.chunk",
        "created": created,
        "model": request_model,
        "choices": [
            {
                "index": 0,
                "delta": {"role": "assistant", "content": ""},
                "finish_reason": None
            }
        ]
    }

    yield f"data: {json.dumps(initial_response)}\n\n"

    # Stream tokens
    try:
        for token in runner.generate_streaming(
            prompt=prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            use_chat_template=False,  # Already applied in _format_conversation
            use_chat_stop_tokens=True   # Server NEEDS chat stop tokens to prevent self-conversations
        ):
            # Stop promptly if server is shutting down
            if shutdown_event.is_set():
                raise KeyboardInterrupt()
            chunk_response = {
                "id": completion_id,
                "object": "chat.completion.chunk",
                "created": created,
                "model": request_model,
                "choices": [
                    {
                        "index": 0,
                        "delta": {"content": token},
                        "finish_reason": None
                    }
                ]
            }

            yield f"data: {json.dumps(chunk_response)}\n\n"

            # Check for stop sequences
            if stop:
                stop_sequences = stop if isinstance(stop, list) else [stop]
                if any(s in token for s in stop_sequences):
                    break

    except KeyboardInterrupt:
        if not shutdown_event.is_set():
            try:
                import mlx.core as mx
                mx.clear_cache()
            except (ImportError, AttributeError):
                pass  # MLX not installed or API changed
            except Exception as e:
                logger.debug(f"Metal cache cleanup failed: {e}")
            try:
                interrupt_response = {
                    "id": completion_id,
                    "object": "chat.completion.chunk",
                    "created": created,
                    "model": request_model,
                    "choices": [
                        {
                            "index": 0,
                            "delta": {"content": "\n\n[Generation interrupted by user]"},
                            "finish_reason": "stop"
                        }
                    ]
                }
                yield f"data: {json.dumps(interrupt_response)}\n\n"
            except Exception:
                pass
        return

    except Exception as e:
        # Optional debug logging for chat streaming errors
        if os.environ.get("MLXK2_DEBUG"):
            print(f"[DEBUG] Exception in chat streaming: {type(e).__name__}: {e}")

        # Try MLX recovery for any exception that might be interrupt-related
        if "interrupt" in str(e).lower() or "keyboard" in str(e).lower():
            if os.environ.get("MLXK2_DEBUG"):
                print("[Server] Detected interrupt-like exception, attempting MLX recovery...")
            try:
                import mlx.core as mx
                mx.clear_cache()
                if os.environ.get("MLXK2_DEBUG"):
                    print("[Server] MLX state recovered after exception")
            except Exception as recovery_error:
                if os.environ.get("MLXK2_DEBUG"):
                    print(f"[Server] MLX recovery warning: {recovery_error}")

        error_response = {
            "id": completion_id,
            "object": "chat.completion.chunk",
            "created": created,
            "model": request_model,
            "choices": [
                {
                    "index": 0,
                    "delta": {},
                    "finish_reason": "error"
                }
            ],
            "error": str(e)
        }
        yield f"data: {json.dumps(error_response)}\n\n"

    # Final response (skip if shutting down)
    if shutdown_event.is_set():
        return
    final_response = {
        "id": completion_id,
        "object": "chat.completion.chunk",
        "created": created,
        "model": request_model,
        "choices": [
            {
                "index": 0,
                "delta": {},
                "finish_reason": "stop"
            }
        ]
    }

    yield f"data: {json.dumps(final_response)}\n\n"
    yield "data: [DONE]\n\n"


async def stream_vision_chunks(
    model_path,
    model_name: str,
    prompt: str,
    images: List[tuple],
    chunk_size: int,
    image_id_map: Dict[str, int],
    max_tokens: Optional[int],
    temperature: float,
    top_p: float,
    repetition_penalty: float,
    completion_id: str,
    created: int,
    model: str,
    shutdown_event: Event,
    audio: Optional[List[tuple]] = None,
) -> AsyncGenerator[str, None]:
    """Stream SSE events per vision chunk as they complete (OpenAI-compatible).

    Unlike _process_vision_chunks_server() which waits for all chunks,
    this yields SSE events immediately after each chunk finishes.
    Uses asyncio.to_thread() to keep the event loop responsive.

    Args:
        model_path: Path to model snapshot directory
        model_name: Model name for VisionRunner
        prompt: User prompt
        images: Full list of (filename, bytes) tuples
        chunk_size: Images per chunk
        image_id_map: Pre-computed global image IDs (from conversation history)
        max_tokens, temperature, top_p, repetition_penalty: Generation params
        completion_id: Unique completion ID for SSE events
        created: Timestamp for SSE events
        model: Model name for SSE events
        shutdown_event: Thread event to check for shutdown
        audio: Optional list of (filename, bytes) tuples for audio input

    Yields:
        SSE event strings (data: {...}\n\n format)
    """
    import asyncio
    from ..vision_runner import VisionRunner

    logger = _get_logger()
    chunks = [images[i:i+chunk_size] for i in range(0, len(images), chunk_size)]
    total_images = len(images)

    # Initial role event
    initial_event = {
        "id": completion_id,
        "object": "chat.completion.chunk",
        "created": created,
        "model": model,
        "choices": [{
            "index": 0,
            "delta": {"role": "assistant"},
            "finish_reason": None
        }]
    }
    yield f"data: {json.dumps(initial_event)}\n\n"

    # Process each chunk and stream result immediately
    for chunk_idx, chunk in enumerate(chunks, start=1):
        logger.info(
            f"Vision chunk {chunk_idx}/{len(chunks)} starting",
            chunk=chunk_idx,
            total_chunks=len(chunks),
            images_in_chunk=len(chunk)
        )

        # Check shutdown before processing
        if shutdown_event.is_set():
            interrupt_event = {
                "id": completion_id,
                "object": "chat.completion.chunk",
                "created": created,
                "model": model,
                "choices": [{
                    "index": 0,
                    "delta": {"content": "\n\n[Generation interrupted]"},
                    "finish_reason": "stop"
                }]
            }
            yield f"data: {json.dumps(interrupt_event)}\n\n"
            yield "data: [DONE]\n\n"
            return

        # Process chunk in thread pool (keeps event loop responsive)
        # NOTE: Pass chunk_images as argument to avoid closure late-binding issues
        def process_chunk(chunk_images):
            with VisionRunner(model_path, model_name, verbose=False) as runner:
                return runner.generate(
                    prompt=prompt,
                    images=chunk_images,
                    audio=audio,  # Pass audio with each chunk
                    max_tokens=max_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    repetition_penalty=repetition_penalty,
                    image_id_map=image_id_map,
                    total_images=total_images,
                )

        try:
            chunk_result = await asyncio.to_thread(process_chunk, chunk)
        except Exception as e:
            logger.error(f"Vision chunk {chunk_idx}/{len(chunks)} failed: {e}")
            error_event = {
                "id": completion_id,
                "object": "chat.completion.chunk",
                "created": created,
                "model": model,
                "choices": [{
                    "index": 0,
                    "delta": {"content": f"\n\n[Error in chunk {chunk_idx}: {str(e)}]"},
                    "finish_reason": "error"
                }]
            }
            yield f"data: {json.dumps(error_event)}\n\n"
            yield "data: [DONE]\n\n"
            return

        # Content event for this chunk (with separator for multi-chunk)
        separator = "\n\n" if chunk_idx < len(chunks) else ""
        content_event = {
            "id": completion_id,
            "object": "chat.completion.chunk",
            "created": created,
            "model": model,
            "choices": [{
                "index": 0,
                "delta": {"content": chunk_result + separator},
                "finish_reason": None
            }]
        }
        yield f"data: {json.dumps(content_event)}\n\n"

        logger.info(
            f"Vision chunk {chunk_idx}/{len(chunks)} streamed",
            chunk=chunk_idx,
            total_chunks=len(chunks),
            output_length=len(chunk_result)
        )

    # Final event with finish_reason
    final_event = {
        "id": completion_id,
        "object": "chat.completion.chunk",
        "created": created,
        "model": model,
        "choices": [{
            "index": 0,
            "delta": {},
            "finish_reason": "stop"
        }]
    }
    yield f"data: {json.dumps(final_event)}\n\n"
    yield "data: [DONE]\n\n"


async def emulate_sse_stream(
    completion_id: str,
    created: int,
    model: str,
    content: str
) -> AsyncGenerator[str, None]:
    """Emulate SSE streaming for vision models (batch response as SSE events).

    Vision models don't support real streaming, so we send the complete
    response as SSE events to be compatible with streaming clients.

    Args:
        completion_id: Unique completion ID for SSE events
        created: Timestamp for SSE events
        model: Model name for SSE events
        content: Complete response content to stream
    """
    # First chunk: role
    chunk1 = {
        "id": completion_id,
        "object": "chat.completion.chunk",
        "created": created,
        "model": model,
        "choices": [{
            "index": 0,
            "delta": {"role": "assistant"},
            "finish_reason": None
        }]
    }
    yield f"data: {json.dumps(chunk1)}\n\n"

    # Second chunk: content (all at once)
    chunk2 = {
        "id": completion_id,
        "object": "chat.completion.chunk",
        "created": created,
        "model": model,
        "choices": [{
            "index": 0,
            "delta": {"content": content},
            "finish_reason": None
        }]
    }
    yield f"data: {json.dumps(chunk2)}\n\n"

    # Final chunk: finish_reason
    chunk3 = {
        "id": completion_id,
        "object": "chat.completion.chunk",
        "created": created,
        "model": model,
        "choices": [{
            "index": 0,
            "delta": {},
            "finish_reason": "stop"
        }]
    }
    yield f"data: {json.dumps(chunk3)}\n\n"

    # Done marker
    yield "data: [DONE]\n\n"
