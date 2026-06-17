"""
OpenAI-compatible API server for MLX models (2.0 implementation).
Provides REST endpoints for text generation with MLX backend.
"""

import os
import threading
import time
import uuid
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from fastapi import FastAPI, HTTPException, Request, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, PlainTextResponse
from pydantic import BaseModel, Field

from .cache import get_current_model_cache, hf_to_cache_dir
from .runner import MLXRunner
from .model_resolution import resolve_model_for_operation
from .capabilities import Backend
from ..operations.common import detect_audio_backend
from ..tools.vision_adapter import MAX_AUDIO_SIZE_BYTES
from .. import __version__
from ..logging import get_logger, set_log_level, uvicorn_log_config
from ..context import generate_request_id

# Import extracted modules (Phase 1 refactoring)
from .server.streaming import (
    generate_completion_stream as _generate_completion_stream_impl,
    generate_chat_stream as _generate_chat_stream_impl,
    stream_vision_chunks as _stream_vision_chunks_impl,
    emulate_sse_stream as _emulate_sse_stream_impl,
)
from .server.handlers.models import handle_list_models as _handle_list_models_impl
from .server.handlers.audio import (
    handle_audio_chat_completion as _handle_audio_chat_completion_impl,
    handle_transcription as _handle_transcription_impl,
)
from .server.handlers.chat import (
    ChatHandlerContext,
    handle_text_chat_completion as _handle_text_chat_completion_impl,
    handle_vision_chat_completion as _handle_vision_chat_completion_impl,
    process_vision_chunks_server as _process_vision_chunks_server_impl,
)
# Import ModelManager (Phase 2 refactoring)
from .server.model_manager import ModelManager
# Shared ADR-004 exception handlers (also used by embed-serve)
from .server.error_handlers import register_error_handlers
# ADR-015 D2: thin /v1/embeddings proxy to a separate embed-serve backend (no runner in-process)
from .server.handlers.embed_proxy import proxy_embeddings, PROXY_TIMEOUT

# Global configuration
_default_max_tokens: Optional[int] = None  # Use dynamic model-aware limits by default
# Global shutdown flag to interrupt in-flight generations promptly
_shutdown_event = threading.Event()
# Pre-load model specification (set via environment MLXK2_PRELOAD_MODEL)
_preload_model: Optional[str] = None
# ModelManager singleton (Phase 2 refactoring)
_model_manager: Optional[ModelManager] = None
# ADR-015 D2: embed-serve proxy backend URL (set via MLXK2_EMBED_BACKEND) + pooled httpx client.
# When unset, POST /v1/embeddings returns 501 (embeddings not enabled on this server).
_embed_backend: Optional[str] = None
_embed_proxy_client: Optional[Any] = None  # httpx.AsyncClient — httpx is imported lazily in lifespan

# Global logger instance (ADR-004)
logger = get_logger()


class CompletionRequest(BaseModel):
    model: str
    prompt: Union[str, List[str]]
    max_tokens: Optional[int] = None
    temperature: Optional[float] = 0.7
    top_p: Optional[float] = 0.9
    stream: Optional[bool] = False
    stop: Optional[Union[str, List[str]]] = None
    repetition_penalty: Optional[float] = 1.1


class ChatMessage(BaseModel):
    role: str = Field(..., pattern="^(system|user|assistant)$")
    # Vision API uses list content: [{"type": "text", ...}, {"type": "image_url", ...}]
    content: Union[str, List[Dict[str, Any]]]


class ChatCompletionRequest(BaseModel):
    model: str
    messages: List[ChatMessage]
    max_tokens: Optional[int] = None
    temperature: Optional[float] = 0.7
    top_p: Optional[float] = 0.9
    stream: Optional[bool] = False
    stop: Optional[Union[str, List[str]]] = None
    repetition_penalty: Optional[float] = 1.1
    chunk: Optional[int] = None  # Vision batch processing (None = use ENV/default)


class CompletionResponse(BaseModel):
    id: str
    object: str = "text_completion"
    created: int
    model: str
    choices: List[Dict[str, Any]]
    usage: Dict[str, int]


class ChatCompletionResponse(BaseModel):
    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: List[Dict[str, Any]]
    usage: Dict[str, int]


class ModelInfo(BaseModel):
    id: str
    object: str = "model"
    owned_by: str = "mlx-knife"
    permission: List = []
    context_length: Optional[int] = None


class TranscriptionResponse(BaseModel):
    """OpenAI-compatible transcription response."""
    text: str


class VerboseTranscriptionResponse(BaseModel):
    """OpenAI-compatible verbose transcription response."""
    task: str = "transcribe"
    language: str
    duration: float
    text: str


def get_or_load_model(model_spec: str, verbose: bool = False) -> Any:
    """Get model from cache or load it if not cached.

    Thread-safe model switching with proper cleanup on interruption.
    Supports both text models (MLXRunner) and vision models (VisionRunner).

    Delegates to ModelManager singleton (Phase 2 refactoring).

    Returns:
        MLXRunner for text models, VisionRunner for vision models
    """
    if _model_manager is None:
        raise HTTPException(503, "Server not ready - lifespan not initialized")
    return _model_manager.get_or_load_model(model_spec, verbose)


def get_or_load_audio_model(model_spec: str, verbose: bool = False) -> Any:
    """Get audio model from cache or load it if not cached.

    Thread-safe model switching with AudioRunner for STT models (ADR-020).
    Uses the same cache as get_or_load_model() but creates AudioRunner instances.

    Delegates to ModelManager singleton (Phase 2 refactoring).

    Returns:
        AudioRunner for STT models
    """
    if _model_manager is None:
        raise HTTPException(503, "Server not ready - lifespan not initialized")
    return _model_manager.get_or_load_audio_model(model_spec, verbose)


async def generate_completion_stream(
    runner: MLXRunner,
    prompt: str,
    request: CompletionRequest,
) -> AsyncGenerator[str, None]:
    """Generate streaming completion response.

    Delegates to extracted streaming module (Phase 1 refactoring).
    """
    max_tokens = get_effective_max_tokens(runner, request.max_tokens, server_mode=True)
    stop = request.stop if isinstance(request.stop, list) else ([request.stop] if request.stop else None)

    async for chunk in _generate_completion_stream_impl(
        runner=runner,
        prompt=prompt,
        request_model=request.model,
        max_tokens=max_tokens,
        temperature=request.temperature,
        top_p=request.top_p,
        repetition_penalty=request.repetition_penalty,
        stop=stop,
        shutdown_event=_shutdown_event,
    ):
        yield chunk
    


async def generate_chat_stream(
    runner: MLXRunner,
    messages: List[ChatMessage],
    request: ChatCompletionRequest,
) -> AsyncGenerator[str, None]:
    """Generate streaming chat completion response.

    Delegates to extracted streaming module (Phase 1 refactoring).
    """
    message_dicts = format_chat_messages_for_runner(messages)
    max_tokens = get_effective_max_tokens(runner, request.max_tokens, server_mode=True)
    stop = request.stop if isinstance(request.stop, list) else ([request.stop] if request.stop else None)

    async for chunk in _generate_chat_stream_impl(
        runner=runner,
        messages=message_dicts,
        request_model=request.model,
        max_tokens=max_tokens,
        temperature=request.temperature,
        top_p=request.top_p,
        repetition_penalty=request.repetition_penalty,
        stop=stop,
        shutdown_event=_shutdown_event,
    ):
        yield chunk
    


def format_chat_messages_for_runner(messages: List[ChatMessage]) -> List[Dict[str, str]]:
    """Convert chat messages to format expected by MLXRunner.
    
    Returns messages in dict format for the runner to apply chat templates.
    """
    return [{"role": msg.role, "content": msg.content} for msg in messages]


def get_effective_max_tokens(runner: MLXRunner, requested_max_tokens: Optional[int], server_mode: bool) -> Optional[int]:
    """Get effective max tokens for TEXT models with server DoS protection.

    Text models use shift-window context management:
    - server_mode=True: context_length / 2 (reserve half for history)
    - server_mode=False: context_length (full context for CLI)

    Priority: requested_max_tokens > _default_max_tokens (from --max-tokens CLI) > dynamic calculation
    """
    if requested_max_tokens is not None:
        return requested_max_tokens
    elif _default_max_tokens is not None:
        # Use server-wide default from CLI --max-tokens flag
        return _default_max_tokens
    else:
        # Use runner's dynamic calculation with server_mode flag
        return runner._calculate_dynamic_max_tokens(server_mode=server_mode)


def get_effective_max_tokens_vision(runner, requested_max_tokens: Optional[int]) -> int:
    """Get effective max tokens for VISION models (stateless, no shift-window).

    Vision models don't maintain conversation history in context (Metal limitations).
    Each request is stateless, so we can use a larger portion of context.

    Strategy:
    - Use 2048 as conservative default (works for all vision models)
    - Vision models typically have large context (128K+), but generation is slow
    - 2048 tokens ≈ 1500 words, enough for detailed image descriptions

    Priority: requested_max_tokens > _default_max_tokens (from --max-tokens CLI) > 2048 default
    """
    if requested_max_tokens is not None:
        return requested_max_tokens
    elif _default_max_tokens is not None:
        # Use server-wide default from CLI --max-tokens flag
        return _default_max_tokens

    # Conservative default for vision (stateless, no history to reserve)
    # Vision inference is slow, so we don't want to generate 64K tokens by default
    return 2048


def count_tokens(text: str) -> int:
    """Rough token count estimation."""
    return int(len(text.split()) * 1.3)  # Approximation, convert to int


def _request_has_images(messages: List[ChatMessage]) -> bool:
    """Check if LAST USER MESSAGE contains image content (Vision API format).

    OpenAI API semantics: Only images from the last user message are processed.
    Historical images are preserved in context but not re-processed.

    This function determines routing (vision vs text path).
    Must match actual vision processing behavior (ADR-012 Phase 3).

    Args:
        messages: List of ChatMessage objects

    Returns:
        True if the last user message contains image_url content
    """
    # Find last user message (iterate backwards for efficiency)
    for msg in reversed(messages):
        if msg.role == "user":
            # Check if THIS message has images
            if isinstance(msg.content, list):
                for item in msg.content:
                    if isinstance(item, dict) and item.get("type") == "image_url":
                        return True
            # Found last user message, no images
            return False
    # No user messages at all (shouldn't happen with validation, but be defensive)
    return False


def _request_has_audio(messages: List[ChatMessage]) -> bool:
    """Check if LAST USER MESSAGE contains audio content (input_audio format).

    OpenAI API semantics: Only audio from the last user message is processed.
    Historical audio references are preserved in context but not re-processed.

    This function determines routing (audio vs text path).
    Must match actual audio processing behavior (ADR-019 Phase 4).

    Args:
        messages: List of ChatMessage objects

    Returns:
        True if the last user message contains input_audio content
    """
    # Find last user message (iterate backwards for efficiency)
    for msg in reversed(messages):
        if msg.role == "user":
            # Check if THIS message has audio
            if isinstance(msg.content, list):
                for item in msg.content:
                    if isinstance(item, dict) and item.get("type") == "input_audio":
                        return True
            # Found last user message, no audio
            return False
    # No user messages at all
    return False


def _messages_to_dicts(messages: List[ChatMessage]) -> List[Dict[str, Any]]:
    """Convert ChatMessage objects to dict format for adapters.

    Args:
        messages: List of ChatMessage objects

    Returns:
        List of message dicts with role and content
    """
    return [{"role": msg.role, "content": msg.content} for msg in messages]


def _detect_audio_backend_for_model(model_spec: str) -> Optional[Backend]:
    """Detect audio backend for a model (STT vs multimodal).

    ADR-020: Routes audio models to appropriate backend:
    - STT models (Whisper, Voxtral) → Backend.MLX_AUDIO
    - Multimodal (Gemma-3n, Qwen3-Omni) → Backend.MLX_VLM

    Args:
        model_spec: Model name or path

    Returns:
        Backend.MLX_AUDIO for STT, Backend.MLX_VLM for multimodal, None if not audio
    """
    import json as _json
    from ..operations.workspace import is_workspace_path

    try:
        resolved_name, _, _ = resolve_model_for_operation(model_spec)
        model_path = None

        # Resolve model path
        if resolved_name and is_workspace_path(resolved_name):
            model_path = Path(resolved_name)
        elif resolved_name:
            cache_root = get_current_model_cache()
            cache_dir = cache_root / hf_to_cache_dir(resolved_name)
            snapshots_dir = cache_dir / "snapshots"
            if snapshots_dir.exists():
                snapshots = [d for d in snapshots_dir.iterdir() if d.is_dir()]
                if snapshots:
                    model_path = max(snapshots, key=lambda x: x.stat().st_mtime)
        elif is_workspace_path(model_spec):
            model_path = Path(model_spec).resolve()

        if model_path is None or not model_path.exists():
            return None

        # Load config.json
        config_path = model_path / "config.json"
        if not config_path.exists():
            return None

        config = _json.loads(config_path.read_text(encoding="utf-8", errors="ignore"))
        if not isinstance(config, dict):
            return None

        # Use shared detection function
        return detect_audio_backend(model_path, config)

    except Exception:
        return None


async def _handle_audio_chat_completion(request: ChatCompletionRequest) -> ChatCompletionResponse:
    """Handle audio STT chat completion with AudioRunner (ADR-020).

    Delegates to extracted audio handler module (Phase 1 refactoring).
    """
    message_dicts = _messages_to_dicts(request.messages)
    result = await _handle_audio_chat_completion_impl(
        request_model=request.model,
        messages=message_dicts,
        max_tokens=request.max_tokens,
        temperature=request.temperature,
        stream=request.stream,
        get_audio_model_fn=get_or_load_audio_model,
        emulate_sse_fn=_emulate_sse_stream,
        count_tokens_fn=count_tokens,
    )
    # Return StreamingResponse directly or convert dict to Pydantic model
    if isinstance(result, StreamingResponse):
        return result
    return ChatCompletionResponse(**result)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifespan."""
    global _model_manager, _preload_model, _embed_backend, _embed_proxy_client

    # Configure log level early (from environment if subprocess mode)
    import os
    env_log_level = os.environ.get("MLXK2_LOG_LEVEL", "info")
    set_log_level(env_log_level)

    logger.info("MLX Knife Server 2.0 starting up...")

    # Initialize ModelManager singleton (Phase 2 refactoring)
    _model_manager = ModelManager(_shutdown_event)

    # Pre-load model with probe/policy validation (if specified)
    preload_spec = os.environ.get("MLXK2_PRELOAD_MODEL")
    if preload_spec:
        try:
            logger.info(f"Pre-loading model with validation: {preload_spec}")

            # Detect if this is an audio-only STT model (ADR-020)
            # Audio models (Whisper, Voxtral) need AudioRunner, not MLXRunner/VisionRunner
            audio_backend = _detect_audio_backend_for_model(preload_spec)
            if audio_backend == Backend.MLX_AUDIO:
                logger.info(f"Detected audio model, using AudioRunner: {preload_spec}")
                get_or_load_audio_model(preload_spec, verbose=False)
            else:
                # Text/vision model path - uses probe/policy checks
                get_or_load_model(preload_spec, verbose=False)

            # Store resolved name for /v1/models sorting (e.g., "qwen" -> "mlx-community/Qwen2.5-0.5B-Instruct-4bit")
            from .model_resolution import resolve_model_for_operation
            resolved_name, _, _ = resolve_model_for_operation(preload_spec)
            _preload_model = resolved_name or preload_spec

            logger.info(f"Model pre-loaded successfully: {_preload_model}")
        except HTTPException as e:
            # Probe/policy validation failed (vision, memory, etc.)
            error_msg = f"Pre-load validation failed for '{preload_spec}': {e.detail}"
            logger.error(error_msg)
            # Server startup fails (this stops uvicorn)
            raise RuntimeError(error_msg) from e
        except Exception as e:
            error_msg = f"Pre-load failed for '{preload_spec}': {str(e)}"
            logger.error(error_msg)
            raise RuntimeError(error_msg) from e

    # ADR-015 D2: enable the /v1/embeddings proxy if an embed backend is configured.
    # Resilient: we do NOT probe the backend here — it may start after serve; per-request
    # failures surface as 502/504. The embed model is never loaded into this process.
    _embed_backend = os.environ.get("MLXK2_EMBED_BACKEND")
    if _embed_backend:
        import httpx
        _embed_proxy_client = httpx.AsyncClient(timeout=PROXY_TIMEOUT)
        logger.info(f"Embeddings proxy enabled -> {_embed_backend} (POST /v1/embeddings)")

    yield
    logger.info("MLX Knife Server 2.0 shutting down...")
    # ADR-015 D2: close the pooled proxy client (no-op if the proxy was not enabled)
    if _embed_proxy_client is not None:
        try:
            await _embed_proxy_client.aclose()
        except Exception:
            pass
    # Ensure shutdown flag is set so any in-flight generations stop quickly
    try:
        _request_global_interrupt()
    except Exception:
        pass
    # Clean up model cache via ModelManager
    if _model_manager:
        try:
            _model_manager.cleanup()
        except Exception:
            pass

    # Force MLX Metal memory cleanup
    try:
        import mlx.core as mx
        mx.clear_cache()
        logger.info("MLX Metal cache cleared")
    except (ImportError, AttributeError):
        pass  # MLX not installed or API changed
    except Exception as e:
        logger.warning(f"Metal cache cleanup failed: {e}")


# Create FastAPI app
app = FastAPI(
    title="MLX Knife API 2.0",
    description="OpenAI-compatible API for MLX models (2.0 implementation)",
    version=__version__,
    lifespan=lifespan
)

# Add CORS middleware for browser access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins for local development
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Request ID middleware (ADR-004)
@app.middleware("http")
async def add_request_id_middleware(request: Request, call_next):
    """Add request_id to all requests for correlation."""
    request_id = generate_request_id()
    request.state.request_id = request_id

    response = await call_next(request)
    response.headers["X-Request-ID"] = request_id
    return response


# ADR-004 exception handlers — shared with embed-serve via core/server/error_handlers.py.
register_error_handlers(app)


@app.get("/health")
async def health_check():
    """Health check endpoint (OpenAI compatible)."""
    return {"status": "healthy", "service": "mlx-knife-server-2.0"}


@app.get("/v1/models")
async def list_models():
    """List available MLX models in the cache.

    Delegates to extracted handler module (Phase 1 refactoring).
    """
    result = await _handle_list_models_impl(
        get_cache_fn=get_current_model_cache,
        preload_model=_preload_model,
    )
    # Convert dicts to ModelInfo Pydantic objects
    model_list = [
        ModelInfo(
            id=m["id"],
            object=m["object"],
            owned_by=m["owned_by"],
            permission=m.get("permission", []),
            context_length=m.get("context_length"),
        )
        for m in result["data"]
    ]
    return {"object": "list", "data": model_list}


@app.post("/v1/embeddings")
async def create_embeddings_proxy(request: Request):
    """Thin proxy to the embed-serve backend (ADR-015 Slice D2).

    serve forwards the raw request body to ``--embed-backend`` and returns the response
    verbatim — the embed model is never loaded into this process. Returns 501 when no backend
    is configured (embeddings not enabled on this server).
    """
    if _embed_backend is None or _embed_proxy_client is None:
        raise HTTPException(
            status_code=501,
            detail="Embeddings are not enabled on this server; start serve with --embed-backend URL",
        )
    if _shutdown_event.is_set():
        raise HTTPException(status_code=503, detail="Server is shutting down")
    body = await request.body()
    return await proxy_embeddings(
        client=_embed_proxy_client,
        backend_url=_embed_backend,
        body=body,
        request_id=getattr(request.state, "request_id", None),
        content_type=request.headers.get("content-type", "application/json"),
    )


@app.post("/v1/completions")
async def create_completion(request: CompletionRequest):
    """Create a text completion."""
    try:
        if _shutdown_event.is_set():
            raise HTTPException(status_code=503, detail="Server is shutting down")
        runner = get_or_load_model(request.model)

        # Handle array of prompts
        if isinstance(request.prompt, list):
            if len(request.prompt) > 1:
                raise HTTPException(status_code=400, detail="Multiple prompts not supported yet")
            prompt = request.prompt[0]
        else:
            prompt = request.prompt

        if request.stream:
            # Streaming response
            return StreamingResponse(
                generate_completion_stream(runner, prompt, request),
                media_type="text/event-stream",
                headers={"Cache-Control": "no-cache"}
            )
        else:
            # Non-streaming response
            completion_id = f"cmpl-{uuid.uuid4()}"
            created = int(time.time())

            generated_text = runner.generate_batch(
                prompt=prompt,
                max_tokens=get_effective_max_tokens(runner, request.max_tokens, server_mode=True),
                temperature=request.temperature,
                top_p=request.top_p,
                repetition_penalty=request.repetition_penalty,
                use_chat_template=False
            )

            prompt_tokens = count_tokens(prompt)
            completion_tokens = count_tokens(generated_text)

            return CompletionResponse(
                id=completion_id,
                created=created,
                model=request.model,
                choices=[
                    {
                        "index": 0,
                        "text": generated_text,
                        "logprobs": None,
                        "finish_reason": "stop"
                    }
                ],
                usage={
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": completion_tokens,
                    "total_tokens": prompt_tokens + completion_tokens
                }
            )

    except HTTPException as http_exc:
        # Preserve intended HTTP status codes from inner helpers
        raise http_exc
    except Exception as e:
        # Map unexpected errors to 500
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/v1/chat/completions")
async def create_chat_completion(request: ChatCompletionRequest):
    """Create a chat completion (text, vision, or audio)."""
    try:
        if _shutdown_event.is_set():
            raise HTTPException(status_code=503, detail="Server is shutting down")

        # Detect if request contains images or audio (Vision/Audio API format)
        has_images = _request_has_images(request.messages)
        has_audio = _request_has_audio(request.messages)

        # ADR-020: Audio-only requests need backend detection for STT vs multimodal
        # STT models (Whisper, Voxtral) → AudioRunner
        # Multimodal (Gemma-3n) → VisionRunner
        if has_audio and not has_images:
            # Check audio backend before loading model
            audio_backend = _detect_audio_backend_for_model(request.model)
            if audio_backend == Backend.MLX_AUDIO:
                # === AUDIO STT PATH (ADR-020) ===
                return await _handle_audio_chat_completion(request)

        # Load model to determine type (uses cache if already loaded)
        # This ensures we route based on MODEL type, not just request content
        runner = get_or_load_model(request.model, verbose=False)

        # Check if this is a vision model (VisionRunner handles both vision and audio)
        from .vision_runner import VisionRunner
        is_vision_model = isinstance(runner, VisionRunner)

        # Routing logic:
        # - Vision model + images/audio → Vision path (vision/audio processing)
        # - Vision model + no media → Text path (text-only on vision model)
        # - Text model + images/audio → Text path (filters multimodal history)
        # - Text model + no media → Text path (normal text processing)
        if is_vision_model and (has_images or has_audio):
            # === VISION/AUDIO PATH (ADR-012 Phase 3, ADR-019 Phase 4) ===
            # Vision model with images/audio → full vision/audio processing
            return await _handle_vision_chat_completion(request, runner=runner)
        else:
            # === TEXT PATH (existing behavior + multimodal filtering) ===
            # - Vision model without media: Use VisionRunner.generate(images=None)
            # - Text model with/without media: Filter multimodal history if needed
            return await _handle_text_chat_completion(request, runner=runner)

    except HTTPException as http_exc:
        # Preserve intended HTTP status codes from inner helpers
        raise http_exc
    except ValueError as ve:
        # Validation errors from VisionHTTPAdapter
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        # Map unexpected errors to 500
        raise HTTPException(status_code=500, detail=str(e))


def _create_chat_handler_context() -> ChatHandlerContext:
    """Create a ChatHandlerContext with all required dependencies."""
    return ChatHandlerContext(
        get_model_fn=get_or_load_model,
        get_effective_max_tokens_fn=get_effective_max_tokens,
        get_effective_max_tokens_vision_fn=get_effective_max_tokens_vision,
        count_tokens_fn=count_tokens,
        format_messages_fn=format_chat_messages_for_runner,
        extract_text_fn=_extract_text_from_messages,
        filter_multimodal_fn=_filter_multimodal_history_for_text_models,
        messages_to_dicts_fn=_messages_to_dicts,
        generate_chat_stream_fn=_generate_chat_stream_impl,
        emulate_sse_fn=_emulate_sse_stream,
        stream_vision_chunks_fn=_stream_vision_chunks_impl,
        process_vision_chunks_fn=_process_vision_chunks_server,
        shutdown_event=_shutdown_event,
    )


async def _handle_text_chat_completion(request: ChatCompletionRequest, runner: Any = None) -> ChatCompletionResponse:
    """Handle text-only chat completion.

    Delegates to extracted chat handler module (Phase 1 refactoring).
    """
    ctx = _create_chat_handler_context()
    stop = request.stop if isinstance(request.stop, list) else ([request.stop] if request.stop else None)
    result = await _handle_text_chat_completion_impl(
        ctx=ctx,
        request_model=request.model,
        messages=request.messages,
        max_tokens=request.max_tokens,
        temperature=request.temperature,
        top_p=request.top_p,
        repetition_penalty=request.repetition_penalty,
        stream=request.stream,
        stop=stop,
        runner=runner,
    )
    # Return StreamingResponse directly or convert dict to Pydantic model
    if isinstance(result, StreamingResponse):
        return result
    return ChatCompletionResponse(**result)


def _process_vision_chunks_server(
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
) -> str:
    """Process vision images in batches with isolated model instances per chunk.

    Delegates to extracted chat handler module (Phase 1 refactoring).
    """
    return _process_vision_chunks_server_impl(
        model_path=model_path,
        model_name=model_name,
        prompt=prompt,
        images=images,
        chunk_size=chunk_size,
        image_id_map=image_id_map,
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p,
        repetition_penalty=repetition_penalty,
        audio=audio,
    )


async def _stream_vision_chunks(
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
    audio: Optional[List[tuple]] = None,
) -> AsyncGenerator[str, None]:
    """Stream SSE events per vision chunk as they complete (OpenAI-compatible).

    Delegates to extracted streaming module (Phase 1 refactoring).
    """
    async for chunk in _stream_vision_chunks_impl(
        model_path=model_path,
        model_name=model_name,
        prompt=prompt,
        images=images,
        chunk_size=chunk_size,
        image_id_map=image_id_map,
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p,
        repetition_penalty=repetition_penalty,
        completion_id=completion_id,
        created=created,
        model=model,
        shutdown_event=_shutdown_event,
        audio=audio,
    ):
        yield chunk


async def _handle_vision_chat_completion(request: ChatCompletionRequest, runner: Any = None) -> ChatCompletionResponse:
    """Handle vision/audio chat completion with images or audio.

    Delegates to extracted chat handler module (Phase 1 refactoring).
    """
    ctx = _create_chat_handler_context()
    result = await _handle_vision_chat_completion_impl(
        ctx=ctx,
        request_model=request.model,
        messages=request.messages,
        max_tokens=request.max_tokens,
        temperature=request.temperature,
        top_p=request.top_p,
        repetition_penalty=request.repetition_penalty,
        stream=request.stream,
        chunk_size_request=request.chunk,
        runner=runner,
    )
    # Return StreamingResponse directly or convert dict to Pydantic model
    if isinstance(result, StreamingResponse):
        return result
    return ChatCompletionResponse(**result)


async def _emulate_sse_stream(
    completion_id: str,
    created: int,
    model: str,
    content: str
) -> AsyncGenerator[str, None]:
    """Emulate SSE streaming for vision models (batch response as SSE events).

    Delegates to extracted streaming module (Phase 1 refactoring).
    """
    async for chunk in _emulate_sse_stream_impl(
        completion_id=completion_id,
        created=created,
        model=model,
        content=content,
    ):
        yield chunk


def _extract_text_from_messages(messages: List[ChatMessage]) -> str:
    """Extract text content from messages (handles both string and list content).

    Args:
        messages: List of ChatMessage objects

    Returns:
        Combined text from all messages
    """
    text_parts = []
    for msg in messages:
        if isinstance(msg.content, str):
            text_parts.append(msg.content)
        elif isinstance(msg.content, list):
            for item in msg.content:
                if isinstance(item, dict) and item.get("type") == "text":
                    text_parts.append(item.get("text", ""))
    return "\n\n".join(text_parts)


def _filter_multimodal_history_for_text_models(messages: List[ChatMessage]) -> List[ChatMessage]:
    """Filter multimodal content from message history for text-only models.

    Handles the case where conversation history contains multimodal content
    (e.g., from previous Vision/Audio model interaction) but the current model is
    text-only and cannot process image_url or input_audio content.

    This enables model switching in clients (Vision/Audio → Text) while preserving
    the assistant's responses for context.

    Args:
        messages: List of ChatMessage objects (may contain multimodal content)

    Returns:
        List of ChatMessage objects with multimodal content filtered to text-only

    Example:
        Input:
            [
                ChatMessage(role='user', content=[
                    {'type': 'text', 'text': 'Describe this'},
                    {'type': 'image_url', 'image_url': {'url': 'data:...'}}
                ]),
                ChatMessage(role='assistant', content='The image shows...'),
                ChatMessage(role='user', content='What country?')
            ]

        Output:
            [
                ChatMessage(role='user', content='Describe this\\n\\n[1 image(s) were attached]'),
                ChatMessage(role='assistant', content='The image shows...'),
                ChatMessage(role='user', content='What country?')
            ]

    See: docs/ISSUES/VISION-MULTIMODAL-HISTORY-ISSUE.md
    """
    filtered = []

    for msg in messages:
        # String content: pass through unchanged
        if isinstance(msg.content, str):
            filtered.append(msg)
            continue

        # Array content: extract text parts and add media placeholder
        if isinstance(msg.content, list):
            text_parts = []
            image_count = 0
            audio_count = 0

            for item in msg.content:
                if not isinstance(item, dict):
                    continue

                item_type = item.get("type")

                if item_type == "text":
                    text = item.get("text", "")
                    if text:
                        text_parts.append(text)

                elif item_type == "image_url":
                    image_count += 1

                elif item_type == "input_audio":
                    audio_count += 1

            # Combine text parts
            combined_text = " ".join(text_parts)

            # Add placeholder if media were present
            placeholders = []
            if image_count > 0:
                placeholders.append(f"{image_count} image(s)")
            if audio_count > 0:
                placeholders.append(f"{audio_count} audio(s)")

            if placeholders:
                placeholder = f"[{', '.join(placeholders)} were attached]"
                if combined_text:
                    combined_text = f"{combined_text}\n\n{placeholder}"
                else:
                    combined_text = placeholder

            # Create filtered message with string content
            filtered_msg = ChatMessage(role=msg.role, content=combined_text)
            filtered.append(filtered_msg)
        else:
            # Unknown content type: pass through unchanged
            filtered.append(msg)

    return filtered


@app.post("/v1/audio/transcriptions")
async def create_transcription(
    file: UploadFile = File(...),
    model: str = Form(...),
    language: Optional[str] = Form(None),
    prompt: Optional[str] = Form(None),
    response_format: Optional[str] = Form("json"),
    temperature: Optional[float] = Form(0.0),
):
    """Create an audio transcription (OpenAI-compatible Whisper API).

    Delegates to extracted audio handler module (Phase 1 refactoring).
    """
    # Validate model is an audio STT model (routing decision stays here)
    audio_backend = _detect_audio_backend_for_model(model)
    if audio_backend != Backend.MLX_AUDIO:
        raise HTTPException(
            status_code=400,
            detail=f"Model '{model}' is not an audio transcription model. Use Whisper or Voxtral models."
        )

    # Read uploaded file
    try:
        content = await file.read()
        filename = file.filename or "audio.wav"
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to read audio file: {str(e)}")

    # Delegate to extracted handler
    result = await _handle_transcription_impl(
        content=content,
        filename=filename,
        model=model,
        language=language,
        prompt=prompt,
        response_format=response_format or "json",
        temperature=temperature or 0.0,
        get_audio_model_fn=get_or_load_audio_model,
        max_audio_size_bytes=MAX_AUDIO_SIZE_BYTES,
    )

    # Convert dict results to Pydantic models
    if isinstance(result, PlainTextResponse):
        return result
    if response_format == "verbose_json":
        return VerboseTranscriptionResponse(**result)
    return TranscriptionResponse(**result)


def cleanup_server():
    """Manual cleanup function for emergency situations."""
    logger.warning("Forcing server cleanup...")

    # Cleanup via ModelManager (Phase 2 refactoring)
    if _model_manager:
        try:
            _model_manager.cleanup()
        except Exception as e:
            logger.warning(f"Warning during ModelManager cleanup: {e}")

    # Force MLX Metal memory cleanup
    try:
        import mlx.core as mx
        mx.clear_cache()
        logger.info("MLX Metal cache cleared")
    except Exception as e:
        logger.warning(f"Warning during MLX cleanup: {e}")


def _request_global_interrupt() -> None:
    """Request all running generations to stop quickly.

    Used during server shutdown to ensure in-flight streams stop.
    """
    _shutdown_event.set()
    if _model_manager:
        try:
            _model_manager.request_interrupt()
        except Exception:
            pass




def run_server(
    host: str = "127.0.0.1",
    port: int = 8000,
    max_tokens: int = 2000,
    reload: bool = False,
    log_level: str = "info",
    preload_model: Optional[str] = None
):
    """Run the MLX Knife server 2.0."""
    import os

    # Suppress transformers/tokenizers noise (Session 89 + Session 90 fix)
    # ENV variables already set by serve.py subprocess, but set logging programmatically
    # IMPORTANT: Do NOT import transformers in global scope (breaks huggingface_hub downloads)
    try:
        from transformers import logging as transformers_logging
        import logging as python_logging
        transformers_logging.set_verbosity_error()
        python_logging.getLogger("transformers.tokenization_utils").setLevel(python_logging.ERROR)
        python_logging.getLogger("transformers.tokenization_utils_base").setLevel(python_logging.ERROR)
    except ImportError:
        pass  # transformers not installed (optional dependency for vision)

    # Import uvicorn lazily to keep module import light when server isn't used
    try:
        import uvicorn  # type: ignore
    except Exception as e:
        raise RuntimeError("uvicorn is required to run the server; install with 'pip install fastapi uvicorn'.") from e
    global _default_max_tokens
    _default_max_tokens = max_tokens

    # Check for log level from environment (subprocess mode)
    env_log_level = os.environ.get("MLXK2_LOG_LEVEL")
    if env_log_level:
        log_level = env_log_level

    # Set preload model in environment for lifespan hook
    if preload_model:
        os.environ["MLXK2_PRELOAD_MODEL"] = preload_model

    # Configure logging level for MLXKLogger and root logger (ADR-004)
    set_log_level(log_level)

    # Rely on Uvicorn's own signal handling; manage shutdown via lifespan

    logger.info(f"Starting MLX Knife Server 2.0 on http://{host}:{port}")
    logger.info(f"API docs available at http://{host}:{port}/docs")
    logger.info(f"Default max tokens: {'model-aware dynamic limits' if max_tokens is None else max_tokens}")
    logger.info("Press Ctrl-C to stop the server")

    # Enable access logs only at debug/info level (reduces noise at warning/error)
    access_log_enabled = log_level.lower() in ["debug", "info"]

    # Configure Uvicorn log format (JSON if MLXK2_LOG_JSON=1) — shared with embed-serve.
    log_config = uvicorn_log_config(log_level)

    try:
        uvicorn.run(
            "mlxk2.core.server_base:app",
            host=host,
            port=port,
            reload=reload,
            log_level=log_level,
            log_config=log_config,
            access_log=access_log_enabled,
            workers=1,
            timeout_graceful_shutdown=5,
            timeout_keep_alive=5,
            lifespan="on"
        )
    except KeyboardInterrupt:
        logger.info("Server interrupted by user")
        _request_global_interrupt()
        cleanup_server()
    except Exception as e:
        logger.error(f"Server error: {e}", error_key="server_error")
        _request_global_interrupt()
        cleanup_server()
        raise


# CLI entrypoint for supervised mode (python -m mlxk2.core.server_base)
# This allows the supervised subprocess to use the full run_server() logic
# including proper JSON log configuration via MLXK2_LOG_JSON env var
if __name__ == "__main__":
    import os

    # Read configuration from environment variables
    # (set by mlxk2/operations/serve.py in supervised mode)
    host = os.environ.get("MLXK2_HOST", "127.0.0.1")
    port = int(os.environ.get("MLXK2_PORT", "8000"))
    log_level = os.environ.get("MLXK2_LOG_LEVEL", "info")
    preload_model = os.environ.get("MLXK2_PRELOAD_MODEL")

    # Optional: max_tokens and reload (rarely used in supervised mode)
    max_tokens = None
    if max_tokens_str := os.environ.get("MLXK2_MAX_TOKENS"):
        max_tokens = int(max_tokens_str)

    reload = os.environ.get("MLXK2_RELOAD", "0") == "1"

    # Start server with full configuration support
    run_server(
        host=host,
        port=port,
        max_tokens=max_tokens,
        reload=reload,
        log_level=log_level,
        preload_model=preload_model,
    )
