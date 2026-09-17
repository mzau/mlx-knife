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
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple, TypeGuard

from ...errors import internal_error
from .inference import drive, in_worker
from ..runner.token_limits import reported_finish_reason


def _event(payload: dict, include_usage: bool = False) -> str:
    """One SSE event. A client that asked for usage finds ``"usage": null`` on every chunk but the last."""
    if include_usage:
        payload = {**payload, "usage": None}
    return f"data: {json.dumps(payload)}\n\n"


def _usage_event(completion_id: str, chunk_object: str, created: int, model: str, usage: Dict[str, int]) -> str:
    """The chunk OpenAI ends such a stream with, before ``[DONE]``: no choices, the request's counts."""
    return _event({
        "id": completion_id, "object": chunk_object, "created": created,
        "model": model, "choices": [], "usage": usage,
    })


def _stream_error(payload: dict, exc: Exception, include_usage: bool = False) -> str:
    """Terminal SSE event for a stream that failed part-way.

    ``finish_reason`` stays ``None``: it says why the *model* stopped, and a backend
    fault is not a generation outcome — OpenAI's enum has no value for it. The signal
    is the ADR-004 ``error`` object, which is where an OpenAI client looks (its SDK
    raises on this key and never reads the choice). The caller returns afterwards:
    no trailing chunk, no usage chunk, no ``[DONE]`` — the stream did not complete.
    """
    payload["error"] = internal_error(str(exc)).to_dict()
    return _event(payload, include_usage)

if TYPE_CHECKING:
    from ..runner import MLXRunner


def _debug_enabled() -> bool:
    """MLXK2_DEBUG, read by value: `=0` has to switch debugging off, not on.

    Same rule as the CLI's feature gates — a switch that opens on `0` is not a switch.
    """
    return os.environ.get("MLXK2_DEBUG", "").strip().lower() in ("1", "true", "yes", "on")


def _get_logger():
    """Lazy import logger to avoid circular dependencies."""
    from ...logging import get_logger
    return get_logger()


def finish_reason_of(runner) -> Optional[str]:
    """OpenAI ``finish_reason`` for the runner's last generation (None when it has none)."""
    return reported_finish_reason(getattr(runner, "last_finish_reason", None))


def stopped_on_sequence(runner) -> bool:
    """Whether the runner cut its last generation at a stop sequence it was handed."""
    return bool(getattr(runner, "last_stopped_on_sequence", False))


def _is_count(value: Any) -> TypeGuard[int]:
    """An int, not merely present: a runner that recorded nothing leaves ``None``, a test double
    answers every attribute with another mock, and a bool is an int to Python."""
    return isinstance(value, int) and not isinstance(value, bool)


def _counted(runner, attribute: str, text: str, estimate) -> int:
    """One real token count, or the estimate when the runner has none."""
    counted = getattr(runner, attribute, None)
    return counted if _is_count(counted) else estimate(text)


def count_tokens(text: str) -> int:
    """Rough token count estimation."""
    return int(len(text.split()) * 1.3)  # Approximation, convert to int


class GenerationRecord:
    """One request's generation — token counts, budget, how it ended — apart from the runner its
    model shares with other requests.

    The attributes carry the runner's names, so ``usage_of``, ``finish_reason_of`` and
    ``log_generation_end`` read either. A text stream hands it to the runner, which keeps it
    current; a batch answer copies the runner into it (``generate_recorded``); a chunked vision
    request adds up the counts of its chunks' runners.
    """

    def __init__(self) -> None:
        self.last_prompt_tokens: Optional[int] = None
        self.last_completion_tokens: Optional[int] = None
        self.last_max_tokens: Optional[int] = None
        self.last_finish_reason: Optional[str] = None
        self._chunks = 0

    def add(self, runner: Any) -> None:
        """Add one chunk's counts. A chunk that recorded none leaves the sum unknown."""
        for name in ("last_prompt_tokens", "last_completion_tokens"):
            total = getattr(self, name) if self._chunks else 0
            counted = getattr(runner, name, None)
            if _is_count(total) and _is_count(counted):
                setattr(self, name, total + counted)
            else:
                setattr(self, name, None)
        self._chunks += 1


def generate_recorded(runner, generate: Callable[..., str], /, **kwargs: Any) -> Tuple[str, GenerationRecord]:
    """One batch generation and the runner's record of it, taken in the same model-thread call.

    Once that call returns, the model thread may begin the next request, whose generation resets
    the runner before the event loop could read it.
    """
    text = generate(**kwargs)
    record = GenerationRecord()
    for name in ("last_prompt_tokens", "last_completion_tokens", "last_max_tokens", "last_finish_reason"):
        setattr(record, name, getattr(runner, name, None))
    return text, record


def usage_of(runner, prompt: str, generated: str, estimate) -> Dict[str, int]:
    """Token counts for one completion: the runner's own numbers where it has them.

    The text and vision runners record what they actually encoded and generated; the
    audio backend does not, so there the word-count estimate stands, as it does on
    ``/v1/embeddings``. Reporting the estimate everywhere made the response contradict
    the ``Generation finished`` line logged beside it, which always used the real ones.
    """
    prompt_tokens = _counted(runner, "last_prompt_tokens", prompt, estimate)
    completion_tokens = _counted(runner, "last_completion_tokens", generated, estimate)
    return {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": prompt_tokens + completion_tokens,
    }


def log_generation_end(logger, runner, model: str, stream: bool) -> None:
    """One structured line per text generation: what it cost and how it ended.

    Makes ``finish_reason: length`` visible in the plain log and filterable under
    ``--log-json``. Runners without the attributes (mocks) log nothing.
    """
    reason = getattr(runner, "last_finish_reason", None)
    if reason is None and getattr(runner, "last_max_tokens", None) is None:
        return
    from ...context import get_request_id
    logger.info(
        f"Generation finished: {reported_finish_reason(reason) or 'unknown'}",
        request_id=get_request_id(),
        model=model,
        stream=stream,
        prompt_tokens=getattr(runner, "last_prompt_tokens", None),
        completion_tokens=getattr(runner, "last_completion_tokens", None),
        max_tokens=getattr(runner, "last_max_tokens", None),
        finish_reason=reported_finish_reason(reason),
    )


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
    include_usage: bool = False,
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
        include_usage: End with a usage chunk (``stream_options.include_usage``)
    """
    logger = _get_logger()
    completion_id = f"cmpl-{uuid.uuid4()}"
    created = int(time.time())
    record = GenerationRecord()
    pieces: List[str] = []  # the streamed text, for the estimate should the runner count nothing

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

    yield _event(initial_response, include_usage)

    # Stream tokens
    stopped_on_sequence = False
    try:
        token_count = 0
        async for token in drive(runner.generate_streaming(
            prompt=prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            use_chat_template=False,  # Raw completion mode
            record=record,
        )):
            # Stop promptly if server is shutting down
            if shutdown_event.is_set():
                raise KeyboardInterrupt()
            token_count += 1
            if include_usage:
                pieces.append(token)

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

            yield _event(chunk_response, include_usage)

            # Check for stop sequences
            if stop:
                stop_sequences = stop if isinstance(stop, list) else [stop]
                if any(s in token for s in stop_sequences):
                    # Breaking abandons the generator, so the runner never records an exit
                    # and finish_reason_of() would answer null. The sequence *is* the reason.
                    stopped_on_sequence = True
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
                yield _event(interrupt_response, include_usage)
            except Exception:
                pass
        return

    except Exception as e:
        yield _stream_error({
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
            ],
        }, e, include_usage)
        return

    # Final response (skip if shutting down)
    if shutdown_event.is_set():
        return
    log_generation_end(logger, record, request_model, stream=True)
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
                "finish_reason": "stop" if stopped_on_sequence else finish_reason_of(record)
            }
        ]
    }

    yield _event(final_response, include_usage)
    if include_usage:
        usage = usage_of(record, prompt, "".join(pieces), count_tokens)
        yield _usage_event(completion_id, "text_completion", created, request_model, usage)
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
    include_usage: bool = False,
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
        include_usage: End with a usage chunk (``stream_options.include_usage``)
    """
    logger = _get_logger()
    completion_id = f"chatcmpl-{uuid.uuid4()}"
    created = int(time.time())
    record = GenerationRecord()
    pieces: List[str] = []  # the streamed text, for the estimate should the runner count nothing

    # Let the runner format with chat templates (the tokenizer belongs to the
    # model thread as much as the weights do)
    prompt = await in_worker(runner._format_conversation, messages)

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

    yield _event(initial_response, include_usage)

    # Stream tokens
    stopped_on_sequence = False
    try:
        async for token in drive(runner.generate_streaming(
            prompt=prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            use_chat_template=False,  # Already applied in _format_conversation
            use_chat_stop_tokens=True,  # Server NEEDS chat stop tokens to prevent self-conversations
            record=record,
        )):
            # Stop promptly if server is shutting down
            if shutdown_event.is_set():
                raise KeyboardInterrupt()
            if include_usage:
                pieces.append(token)
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

            yield _event(chunk_response, include_usage)

            # Check for stop sequences
            if stop:
                stop_sequences = stop if isinstance(stop, list) else [stop]
                if any(s in token for s in stop_sequences):
                    # Breaking abandons the generator, so the runner never records an exit
                    # and finish_reason_of() would answer null. The sequence *is* the reason.
                    stopped_on_sequence = True
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
                yield _event(interrupt_response, include_usage)
            except Exception:
                pass
        return

    except Exception as e:
        # Optional debug logging for chat streaming errors
        if _debug_enabled():
            print(f"[DEBUG] Exception in chat streaming: {type(e).__name__}: {e}")

        # Try MLX recovery for any exception that might be interrupt-related
        if "interrupt" in str(e).lower() or "keyboard" in str(e).lower():
            if _debug_enabled():
                print("[Server] Detected interrupt-like exception, attempting MLX recovery...")
            try:
                import mlx.core as mx
                mx.clear_cache()
                if _debug_enabled():
                    print("[Server] MLX state recovered after exception")
            except Exception as recovery_error:
                if _debug_enabled():
                    print(f"[Server] MLX recovery warning: {recovery_error}")

        yield _stream_error({
            "id": completion_id,
            "object": "chat.completion.chunk",
            "created": created,
            "model": request_model,
            "choices": [
                {
                    "index": 0,
                    "delta": {},
                    "finish_reason": None
                }
            ],
        }, e, include_usage)
        return

    # Final response (skip if shutting down)
    if shutdown_event.is_set():
        return
    log_generation_end(logger, record, request_model, stream=True)
    final_response = {
        "id": completion_id,
        "object": "chat.completion.chunk",
        "created": created,
        "model": request_model,
        "choices": [
            {
                "index": 0,
                "delta": {},
                "finish_reason": "stop" if stopped_on_sequence else finish_reason_of(record)
            }
        ]
    }

    yield _event(final_response, include_usage)
    if include_usage:
        usage = usage_of(record, prompt, "".join(pieces), count_tokens)
        yield _usage_event(completion_id, "chat.completion.chunk", created, request_model, usage)
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
    stop: Optional[List[str]] = None,
    include_usage: bool = False,
) -> AsyncGenerator[str, None]:
    """Stream SSE events per vision chunk as they complete (OpenAI-compatible).

    Unlike _process_vision_chunks_server() which waits for all chunks,
    this yields SSE events immediately after each chunk finishes.
    Uses asyncio.to_thread() to keep the event loop responsive.
    Each chunk's text is a batch answer, so the runner applies ``stop`` to it whole; a
    match ends the response there and no later chunk is generated.

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
        stop: Stop sequences, handed to the runner for each chunk
        include_usage: End with a usage chunk summed over the chunks (``stream_options.include_usage``)

    Yields:
        SSE event strings (data: {...}\n\n format)
    """
    from ..vision_runner import VisionRunner

    logger = _get_logger()
    chunks = [images[i:i+chunk_size] for i in range(0, len(images), chunk_size)]
    total_images = len(images)
    finish_reason: Optional[str] = None  # aggregated over chunks: any "length" wins
    record = GenerationRecord()
    answer: List[str] = []

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
    yield _event(initial_event, include_usage)

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
            yield _event(interrupt_event, include_usage)
            yield "data: [DONE]\n\n"
            return

        # Each chunk builds and uses its own runner inside one call on the model
        # thread, so the loop stays free and requests may interleave between chunks.
        # NOTE: Pass chunk_images as argument to avoid closure late-binding issues
        def process_chunk(chunk_images):
            with VisionRunner(model_path, model_name, verbose=False) as runner:
                text = runner.generate(
                    prompt=prompt,
                    images=chunk_images,
                    audio=audio,  # Pass audio with each chunk
                    max_tokens=max_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    repetition_penalty=repetition_penalty,
                    image_id_map=image_id_map,
                    total_images=total_images,
                    stop=stop,
                )
                record.add(runner)
                return text, finish_reason_of(runner), stopped_on_sequence(runner)

        try:
            chunk_result, chunk_reason, stopped = await in_worker(process_chunk, chunk)
        except Exception as e:
            logger.error(f"Vision chunk {chunk_idx}/{len(chunks)} failed: {e}")
            # The message goes in the error object, not into delta.content — a chat
            # client renders content as something the model said.
            yield _stream_error({
                "id": completion_id,
                "object": "chat.completion.chunk",
                "created": created,
                "model": model,
                "choices": [{
                    "index": 0,
                    "delta": {},
                    "finish_reason": None
                }]
            }, RuntimeError(f"vision chunk {chunk_idx}/{len(chunks)} failed: {e}"), include_usage)
            return

        # The runner cut the chunk's text at the sequence (before its filename header),
        # which ends the answer here and outranks the reason of any earlier chunk.
        # Otherwise one cut chunk makes the whole response a cut response.
        if stopped:
            finish_reason = "stop"
        elif chunk_reason == "length" or finish_reason is None:
            finish_reason = chunk_reason

        # Content event for this chunk (with separator for multi-chunk)
        separator = "" if stopped or chunk_idx == len(chunks) else "\n\n"
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
        yield _event(content_event, include_usage)
        if include_usage:
            answer.append(chunk_result + separator)

        logger.info(
            f"Vision chunk {chunk_idx}/{len(chunks)} streamed",
            chunk=chunk_idx,
            total_chunks=len(chunks),
            output_length=len(chunk_result)
        )
        if stopped:
            break

    # Final event with finish_reason
    final_event = {
        "id": completion_id,
        "object": "chat.completion.chunk",
        "created": created,
        "model": model,
        "choices": [{
            "index": 0,
            "delta": {},
            "finish_reason": finish_reason
        }]
    }
    yield _event(final_event, include_usage)
    if include_usage:
        usage = usage_of(record, prompt, "".join(answer), count_tokens)
        yield _usage_event(completion_id, "chat.completion.chunk", created, model, usage)
    yield "data: [DONE]\n\n"


async def emulate_sse_stream(
    completion_id: str,
    created: int,
    model: str,
    content: str,
    finish_reason: Optional[str] = None,
    usage: Optional[Dict[str, int]] = None,
) -> AsyncGenerator[str, None]:
    """Emulate SSE streaming for vision models (batch response as SSE events).

    Vision models don't support real streaming, so we send the complete
    response as SSE events to be compatible with streaming clients.

    Args:
        completion_id: Unique completion ID for SSE events
        created: Timestamp for SSE events
        model: Model name for SSE events
        content: Complete response content to stream
        finish_reason: How the batch generation ended ("stop" | "length" | None)
        usage: The request's token counts when the client asked for them
            (``stream_options.include_usage``); None sends no usage chunk
    """
    include_usage = usage is not None

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
    yield _event(chunk1, include_usage)

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
    yield _event(chunk2, include_usage)

    # Final chunk: finish_reason
    chunk3 = {
        "id": completion_id,
        "object": "chat.completion.chunk",
        "created": created,
        "model": model,
        "choices": [{
            "index": 0,
            "delta": {},
            "finish_reason": finish_reason
        }]
    }
    yield _event(chunk3, include_usage)
    if usage is not None:
        yield _usage_event(completion_id, "chat.completion.chunk", created, model, usage)

    # Done marker
    yield "data: [DONE]\n\n"
