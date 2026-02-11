"""
Audio transcription handlers for STT models (ADR-020).

Extracted from server_base.py as part of Phase 1 refactoring.
Handles both /v1/audio/transcriptions endpoint and audio chat completion.
"""

from __future__ import annotations

import time
import uuid
from collections.abc import AsyncGenerator
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Union

from fastapi import HTTPException
from fastapi.responses import PlainTextResponse, StreamingResponse

if TYPE_CHECKING:
    from ...audio_runner import AudioRunner


def _get_logger():
    """Lazy import logger to avoid circular dependencies."""
    from ....logging import get_logger
    return get_logger()


async def handle_audio_chat_completion(
    request_model: str,
    messages: List[Dict[str, Any]],
    max_tokens: Optional[int],
    temperature: Optional[float],
    stream: bool,
    get_audio_model_fn: Callable[[str, bool], "AudioRunner"],
    emulate_sse_fn: Callable[[str, int, str, str], AsyncGenerator[str, None]],
    count_tokens_fn: Callable[[str], int],
) -> Union[Dict[str, Any], StreamingResponse]:
    """Handle audio STT chat completion with AudioRunner (ADR-020).

    Uses mlx-audio backend for Whisper and Voxtral STT models.

    Args:
        request_model: Model name/path
        messages: List of message dicts (already converted from ChatMessage)
        max_tokens: Max tokens for generation
        temperature: Sampling temperature
        stream: Whether to stream response
        get_audio_model_fn: Function to load audio model
        emulate_sse_fn: Function to create SSE stream from batch response
        count_tokens_fn: Function to count tokens

    Returns:
        ChatCompletionResponse dict or StreamingResponse
    """
    from ....tools.vision_adapter import VisionHTTPAdapter

    logger = _get_logger()

    # Find the last user message only
    last_user_msg = None
    for msg in reversed(messages):
        if msg.get("role") == "user":
            last_user_msg = msg
            break

    if last_user_msg is None:
        raise HTTPException(status_code=400, detail="No user message found")

    # Parse audio content from last user message
    prompt, _, audio = VisionHTTPAdapter.parse_openai_messages([last_user_msg])

    if not audio:
        raise HTTPException(status_code=400, detail="No audio content found in request")

    logger.info(
        f"Audio STT request: {len(audio)} audio(s), model={request_model}",
        model=request_model,
        audio_count=len(audio)
    )

    # Load AudioRunner
    runner = get_audio_model_fn(request_model, False)

    # Generate transcription
    completion_id = f"chatcmpl-{uuid.uuid4()}"
    created = int(time.time())

    generated_text = runner.transcribe(
        audio=list(audio),
        prompt=prompt or "Transcribe this audio.",
        max_tokens=max_tokens or 4096,
        temperature=temperature or 0.0,
    )

    logger.info(
        f"Audio STT complete: {len(generated_text)} chars",
        model=request_model,
        output_length=len(generated_text)
    )

    # Token counting
    prompt_tokens = count_tokens_fn(prompt or "")
    completion_tokens = count_tokens_fn(generated_text)

    # Emulate SSE for stream=true
    if stream:
        logger.info("Audio STT: emulating SSE stream (batch response as single event)")
        return StreamingResponse(
            emulate_sse_fn(completion_id, created, request_model, generated_text),
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
                "finish_reason": "stop"
            }
        ],
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens
        }
    }


async def handle_transcription(
    content: bytes,
    filename: str,
    model: str,
    language: Optional[str],
    prompt: Optional[str],
    response_format: str,
    temperature: float,
    get_audio_model_fn: Callable[[str, bool], "AudioRunner"],
    max_audio_size_bytes: int,
) -> Union[Dict[str, Any], PlainTextResponse]:
    """Create an audio transcription (OpenAI-compatible Whisper API).

    Accepts audio files and returns transcribed text.
    Supports Whisper and Voxtral STT models via mlx-audio backend.

    Args:
        content: Audio file content bytes
        filename: Original filename
        model: Model ID
        language: Optional language code
        prompt: Optional prompt to guide transcription
        response_format: Output format (json, text, verbose_json)
        temperature: Sampling temperature
        get_audio_model_fn: Function to load audio model
        max_audio_size_bytes: Maximum allowed audio file size

    Returns:
        TranscriptionResponse, VerboseTranscriptionResponse, or PlainTextResponse
    """
    logger = _get_logger()

    if not content:
        raise HTTPException(status_code=400, detail="Empty audio file")

    # Enforce audio size limit
    if len(content) > max_audio_size_bytes:
        limit_mb = max_audio_size_bytes // (1024 * 1024)
        actual_mb = len(content) / (1024 * 1024)
        raise HTTPException(
            status_code=413,
            detail=f"Audio file exceeds {limit_mb} MB limit (got {actual_mb:.1f} MB)"
        )

    try:
        # Load audio model
        runner = get_audio_model_fn(model, False)

        start_time = time.time()

        # Transcribe audio - runner.transcribe() expects List[(filename, bytes)]
        transcription = runner.transcribe(
            audio=[(filename, content)],
            prompt=prompt or "Transcribe this audio.",
            max_tokens=4096,
            temperature=temperature,
            language=language,
        )

        duration = time.time() - start_time

        logger.info(
            f"Transcription complete: {len(transcription)} chars in {duration:.2f}s",
            model=model,
            output_length=len(transcription),
            duration=duration
        )

        # Return response based on format
        if response_format == "text":
            return PlainTextResponse(content=transcription)
        elif response_format == "verbose_json":
            return {
                "task": "transcribe",
                "language": language or "auto",
                "duration": duration,
                "text": transcription
            }
        else:
            # Default: json
            return {"text": transcription}

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Transcription failed: {str(e)}", model=model)
        raise HTTPException(status_code=500, detail=f"Transcription failed: {str(e)}")
