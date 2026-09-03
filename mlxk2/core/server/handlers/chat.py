"""
Chat completion handlers for text and vision models.

Extracted from server_base.py as part of Phase 1 refactoring.
Handles both text-only and vision/audio chat completions.
"""

from __future__ import annotations

import os
import time
import uuid
from collections.abc import AsyncGenerator
from threading import Event
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple, Union

from fastapi import HTTPException
from fastapi.responses import StreamingResponse

from ..streaming import finish_reason_of, log_generation_end

if TYPE_CHECKING:
    from ...runner import MLXRunner  # noqa: F401
    from ...vision_runner import VisionRunner  # noqa: F401


def _get_logger():
    """Lazy import logger to avoid circular dependencies."""
    from ....logging import get_logger
    return get_logger()


def ensure_generation_budget(runner, prompt: str, max_tokens: Optional[int], use_chat_template: bool) -> None:
    """Raise ``ContextLengthExceeded`` now if the prompt fills the window.

    Streaming answers 200 before the runner runs, so the runner's own reject would
    land inside the SSE stream; asking first turns it into the 400 it is meant to
    be. Runners without the method (mocks) are left to their generation path.
    """
    budget = getattr(runner, "generation_budget", None)
    if callable(budget):
        budget(prompt, max_tokens=max_tokens, use_chat_template=use_chat_template)


class ChatHandlerContext:
    """Dependency container for chat handlers.

    Bundles all external dependencies to avoid passing many arguments
    to each function and to enable easy mocking in tests.
    """

    def __init__(
        self,
        get_model_fn: Callable[[str, bool], Any],
        get_effective_max_tokens_fn: Callable[[Optional[int]], int],
        get_effective_max_tokens_vision_fn: Callable[[Optional[int]], int],
        count_tokens_fn: Callable[[str], int],
        format_messages_fn: Callable[[List[Any]], List[Dict[str, str]]],
        extract_text_fn: Callable[[List[Any]], str],
        filter_multimodal_fn: Callable[[List[Any]], List[Any]],
        messages_to_dicts_fn: Callable[[List[Any]], List[Dict[str, Any]]],
        generate_chat_stream_fn: Callable,
        emulate_sse_fn: Callable[[str, int, str, str, Optional[str]], AsyncGenerator[str, None]],
        stream_vision_chunks_fn: Callable,
        process_vision_chunks_fn: Callable,
        shutdown_event: Event,
    ):
        self.get_model = get_model_fn
        self.get_effective_max_tokens = get_effective_max_tokens_fn
        self.get_effective_max_tokens_vision = get_effective_max_tokens_vision_fn
        self.count_tokens = count_tokens_fn
        self.format_messages = format_messages_fn
        self.extract_text = extract_text_fn
        self.filter_multimodal = filter_multimodal_fn
        self.messages_to_dicts = messages_to_dicts_fn
        self.generate_chat_stream = generate_chat_stream_fn
        self.emulate_sse = emulate_sse_fn
        self.stream_vision_chunks = stream_vision_chunks_fn
        self.process_vision_chunks = process_vision_chunks_fn
        self.shutdown_event = shutdown_event


async def handle_text_chat_completion(
    ctx: ChatHandlerContext,
    request_model: str,
    messages: List[Any],
    max_tokens: Optional[int],
    temperature: float,
    top_p: float,
    repetition_penalty: float,
    stream: bool,
    stop: Optional[List[str]],
    runner: Any = None,
) -> Union[Dict[str, Any], StreamingResponse]:
    """Handle text-only chat completion.

    Works with both MLXRunner (text models) and VisionRunner (vision models without images).

    Args:
        ctx: Handler context with dependencies
        request_model: Model name/path
        messages: List of ChatMessage objects
        max_tokens: Max tokens for generation
        temperature: Sampling temperature
        top_p: Top-p sampling
        repetition_penalty: Repetition penalty
        stream: Whether to stream response
        stop: Stop sequences
        runner: Pre-loaded model runner (optional, will load if not provided)

    Returns:
        ChatCompletionResponse dict or StreamingResponse
    """
    logger = _get_logger()

    if runner is None:
        runner = ctx.get_model(request_model, False)

    # Check if runner is VisionRunner (vision model loaded without images in request)
    from ...vision_runner import VisionRunner
    is_vision_runner = isinstance(runner, VisionRunner)

    # Filter multimodal history for text-only models
    # Vision models can handle multimodal content, text models cannot
    if not is_vision_runner:
        messages = ctx.filter_multimodal(messages)

    if is_vision_runner:
        # Vision model: use VisionRunner.generate() without images
        completion_id = f"chatcmpl-{uuid.uuid4()}"
        created = int(time.time())

        # Extract text prompt from messages
        prompt = ctx.extract_text(messages)

        # Vision model WITHOUT images: the vision ceiling applies (stateless, no window guard)
        generated_text = runner.generate(
            prompt=prompt,
            images=None,
            max_tokens=ctx.get_effective_max_tokens_vision(max_tokens),
            temperature=0.0,  # Greedy sampling to reduce hallucinations
            top_p=top_p or 0.9,
            repetition_penalty=repetition_penalty or 1.0,
        )
        finish_reason = finish_reason_of(runner)

        prompt_tokens = ctx.count_tokens(prompt)
        completion_tokens = ctx.count_tokens(generated_text)

        # Graceful degradation: emulate SSE for stream=true
        if stream:
            logger.info("Vision model: emulating SSE stream (batch response as single event)")
            return StreamingResponse(
                ctx.emulate_sse(completion_id, created, request_model, generated_text, finish_reason),
                media_type="text/event-stream",
                headers={"Cache-Control": "no-cache"}
            )

        return {
            "id": completion_id,
            "object": "chat.completion",
            "created": created,
            "model": request_model,
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": generated_text
                    },
                    "finish_reason": finish_reason
                }
            ],
            "usage": {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": prompt_tokens + completion_tokens
            }
        }

    # Text model: use MLXRunner
    if stream:
        # Streaming response (use filtered messages)
        message_dicts = ctx.format_messages(messages)
        effective_max_tokens = ctx.get_effective_max_tokens(max_tokens)
        # A full window must answer with a status, not inside the stream (#66)
        ensure_generation_budget(
            runner, runner._format_conversation(message_dicts), effective_max_tokens, use_chat_template=False
        )
        return StreamingResponse(
            ctx.generate_chat_stream(
                runner, message_dicts, request_model, effective_max_tokens,
                temperature, top_p, repetition_penalty, stop, ctx.shutdown_event
            ),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache"}
        )

    # Non-streaming response
    completion_id = f"chatcmpl-{uuid.uuid4()}"
    created = int(time.time())

    # Convert messages to dict format for runner
    message_dicts = ctx.format_messages(messages)

    # Let the runner format with chat templates
    prompt = runner._format_conversation(message_dicts)

    generated_text = runner.generate_batch(
        prompt=prompt,
        max_tokens=ctx.get_effective_max_tokens(max_tokens),
        temperature=temperature,
        top_p=top_p,
        repetition_penalty=repetition_penalty,
        use_chat_template=False,
        use_chat_stop_tokens=True
    )
    log_generation_end(logger, runner, request_model, stream=False)

    # Token counting
    total_prompt = ctx.extract_text(messages)
    prompt_tokens = ctx.count_tokens(total_prompt)
    completion_tokens = ctx.count_tokens(generated_text)

    return {
        "id": completion_id,
        "object": "chat.completion",
        "created": created,
        "model": request_model,
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": generated_text
                },
                "finish_reason": finish_reason_of(runner)
            }
        ],
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens
        }
    }


async def handle_vision_chat_completion(
    ctx: ChatHandlerContext,
    request_model: str,
    messages: List[Any],
    max_tokens: Optional[int],
    temperature: float,
    top_p: float,
    repetition_penalty: float,
    stream: bool,
    chunk_size_request: Optional[int],
    runner: Any = None,
) -> Union[Dict[str, Any], StreamingResponse]:
    """Handle vision/audio chat completion with images or audio (ADR-012 Phase 3, ADR-019 Phase 4).

    Supports per-chunk streaming for multi-image requests (stream=True yields
    SSE events as each chunk completes). Single-chunk requests use batch mode
    with optional SSE emulation.

    Args:
        ctx: Handler context with dependencies
        request_model: Model name/path
        messages: List of ChatMessage objects
        max_tokens: Max tokens for generation
        temperature: Sampling temperature
        top_p: Top-p sampling
        repetition_penalty: Repetition penalty
        stream: Whether to stream response
        chunk_size_request: Chunk size from request (None = use ENV/default)
        runner: Pre-loaded model runner (optional, will load if not provided)

    Returns:
        ChatCompletionResponse dict or StreamingResponse
    """
    from ....tools.vision_adapter import VisionHTTPAdapter, MAX_SAFE_CHUNK_SIZE

    logger = _get_logger()

    # Convert messages to dicts for processing
    message_dicts = ctx.messages_to_dicts(messages)

    # Find the last user message only (for stateless prompt)
    last_user_msg = None
    for msg in reversed(message_dicts):
        if msg.get("role") == "user":
            last_user_msg = msg
            break

    if last_user_msg is None:
        raise HTTPException(status_code=400, detail="No user message found")

    # Parse only the last user message (stateless prompt)
    prompt, images, audio = VisionHTTPAdapter.parse_openai_messages([last_user_msg])

    # Use FULL history for image ID assignment (consistent numbering)
    image_id_map = VisionHTTPAdapter.assign_image_ids_from_history(message_dicts)

    image_count = len(images)
    audio_count = len(audio)
    logger.info(
        f"Vision/Audio request: {image_count} image(s), {audio_count} audio(s), model={request_model}",
        model=request_model,
        image_count=image_count,
        audio_count=audio_count
    )

    # Get or load VisionRunner
    if runner is None:
        runner = ctx.get_model(request_model, False)

    # Verify we got a VisionRunner
    from ...vision_runner import VisionRunner
    if not isinstance(runner, VisionRunner):
        raise HTTPException(
            status_code=400,
            detail=f"Model '{request_model}' does not support vision inputs"
        )

    # Generate with VisionRunner
    completion_id = f"chatcmpl-{uuid.uuid4()}"
    created = int(time.time())

    # Get chunk size: request param > ENV > default
    env_chunk = int(os.environ.get("MLXK2_VISION_CHUNK_SIZE", "1"))
    chunk_size = chunk_size_request if chunk_size_request is not None else env_chunk

    # Audio+Vision: audio silently ignored when images present
    effective_audio = None if images else audio

    # Validate chunk size for Metal API stability
    if chunk_size < 1:
        raise HTTPException(
            status_code=400,
            detail=f"chunk size must be at least 1 (got: {chunk_size})."
        )
    if chunk_size > MAX_SAFE_CHUNK_SIZE:
        raise HTTPException(
            status_code=400,
            detail=(
                f"chunk size too large (max: {MAX_SAFE_CHUNK_SIZE} for Metal API stability). "
                f"This limit is based on empirically tested performance."
            )
        )

    effective_max_tokens = ctx.get_effective_max_tokens_vision(max_tokens)

    if len(images) <= chunk_size:
        # Single batch (no chunking)
        generated_text = runner.generate(
            prompt=prompt,
            images=images if images else None,
            audio=effective_audio,
            max_tokens=effective_max_tokens,
            temperature=0.0,
            top_p=top_p or 0.9,
            repetition_penalty=repetition_penalty or 1.0,
            image_id_map=image_id_map if images else None,
        )
        finish_reason = finish_reason_of(runner)
    else:
        # Multi-chunk processing
        if stream:
            # True per-chunk streaming
            logger.info(
                f"Vision request: chunk streaming ({len(images)} images, chunk_size={chunk_size})",
                model=request_model,
                image_count=len(images),
                chunk_size=chunk_size
            )
            return StreamingResponse(
                ctx.stream_vision_chunks(
                    model_path=runner.model_path,
                    model_name=runner.model_name,
                    prompt=prompt,
                    images=images,
                    chunk_size=chunk_size,
                    image_id_map=image_id_map,
                    max_tokens=effective_max_tokens,
                    temperature=0.0,
                    top_p=top_p or 0.9,
                    repetition_penalty=repetition_penalty or 1.0,
                    completion_id=completion_id,
                    created=created,
                    model=request_model,
                    shutdown_event=ctx.shutdown_event,
                    audio=effective_audio,
                ),
                media_type="text/event-stream",
                headers={"Cache-Control": "no-cache"}
            )
        # Non-streaming multi-chunk (batch mode)
        generated_text, finish_reason = ctx.process_vision_chunks(
            model_path=runner.model_path,
            model_name=runner.model_name,
            prompt=prompt,
            images=images,
            chunk_size=chunk_size,
            image_id_map=image_id_map,
            max_tokens=effective_max_tokens,
            temperature=0.0,
            top_p=top_p or 0.9,
            repetition_penalty=repetition_penalty or 1.0,
            audio=effective_audio,
        )

    logger.info(
        f"Vision generation complete: {len(generated_text)} chars",
        model=request_model,
        output_length=len(generated_text)
    )

    # Token counting
    prompt_tokens = ctx.count_tokens(prompt)
    completion_tokens = ctx.count_tokens(generated_text)

    # Graceful degradation: emulate SSE for stream=true (single-chunk only)
    if stream:
        logger.info("Vision request: emulating SSE stream (single-chunk batch response)")
        return StreamingResponse(
            ctx.emulate_sse(completion_id, created, request_model, generated_text, finish_reason),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache"}
        )

    return {
        "id": completion_id,
        "object": "chat.completion",
        "created": created,
        "model": request_model,
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": generated_text
                },
                "finish_reason": finish_reason
            }
        ],
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens
        }
    }


def process_vision_chunks_server(
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
    audio: Optional[List[tuple]] = None,
) -> Tuple[str, Optional[str]]:
    """Process vision images in batches with isolated model instances per chunk.

    Each chunk creates a fresh VisionRunner to prevent state leakage between batches.

    Args:
        model_path: Path to model snapshot directory
        model_name: Model name for VisionRunner
        prompt: User prompt
        images: Full list of (filename, bytes) tuples
        chunk_size: Images per batch
        image_id_map: Pre-computed global image IDs
        max_tokens, temperature, top_p, repetition_penalty: Generation params
        audio: Optional list of (filename, bytes) tuples for audio input

    Returns:
        Combined text with merged filename mappings, and the aggregated finish
        reason: "length" if any chunk was cut, else what the chunks reported.
    """
    from ...vision_runner import VisionRunner

    # Split into chunks
    chunks = [images[i:i+chunk_size] for i in range(0, len(images), chunk_size)]

    # Process each chunk with fresh runner
    all_results = []
    finish_reason: Optional[str] = None
    for chunk in chunks:
        with VisionRunner(model_path, model_name, verbose=False) as runner:
            chunk_result = runner.generate(
                prompt=prompt,
                images=chunk,
                audio=audio,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                repetition_penalty=repetition_penalty,
                image_id_map=image_id_map,
                total_images=len(images),
            )
            chunk_reason = finish_reason_of(runner)
        all_results.append(chunk_result)
        if chunk_reason == "length" or finish_reason is None:
            finish_reason = chunk_reason

    return "\n\n".join(all_results), finish_reason
