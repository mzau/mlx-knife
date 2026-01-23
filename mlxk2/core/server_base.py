"""
OpenAI-compatible API server for MLX models (2.0 implementation).
Provides REST endpoints for text generation with MLX backend.
"""

import json
import os
import threading
import time
import uuid
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, JSONResponse
from pydantic import BaseModel, Field

from .cache import get_current_model_cache, hf_to_cache_dir
from .runner import MLXRunner
from .model_resolution import resolve_model_for_operation
from .capabilities import probe_and_select, PolicyDecision, Backend
from .. import __version__
from ..errors import (
    ErrorType,
    MLXKError,
    error_envelope,
)
from ..logging import get_logger, set_log_level
from ..context import generate_request_id

# Global model cache and configuration
_model_cache: Dict[str, MLXRunner] = {}
_current_model_path: Optional[str] = None
_default_max_tokens: Optional[int] = None  # Use dynamic model-aware limits by default
_model_lock = threading.Lock()  # Thread-safe model switching
# Global shutdown flag to interrupt in-flight generations promptly
_shutdown_event = threading.Event()
# Pre-load model specification (set via environment MLXK2_PRELOAD_MODEL)
_preload_model: Optional[str] = None

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


def get_or_load_model(model_spec: str, verbose: bool = False) -> Any:
    """Get model from cache or load it if not cached.

    Thread-safe model switching with proper cleanup on interruption.
    Supports both text models (MLXRunner) and vision models (VisionRunner).

    Returns:
        MLXRunner for text models, VisionRunner for vision models
    """
    global _model_cache, _current_model_path

    # Abort early if shutdown requested
    if _shutdown_event.is_set():
        raise HTTPException(status_code=503, detail="Server is shutting down")

    # Thread-safe model switching
    with _model_lock:
        if _shutdown_event.is_set():
            raise HTTPException(status_code=503, detail="Server is shutting down")
        # Simple approach like run command - let MLXRunner handle everything
        if _current_model_path != model_spec:
            # Clean up previous model
            if _model_cache:
                try:
                    for _old_runner in list(_model_cache.values()):
                        try:
                            # VisionRunner uses _cleanup_temp_files, MLXRunner uses cleanup
                            if hasattr(_old_runner, 'cleanup'):
                                _old_runner.cleanup()
                            if hasattr(_old_runner, '_cleanup_temp_files'):
                                _old_runner._cleanup_temp_files()
                        except Exception as e:
                            logger.warning(f"Warning during cleanup: {e}")
                finally:
                    _model_cache.clear()
                    _current_model_path = None

            # Load new model (disable signal handlers for server mode)
            try:
                # Unified probe/policy architecture (ARCHITECTURE.md principles)
                # Probe and select backend (fail-fast validation)
                caps = None
                policy = None
                model_path = None
                resolved_name = None
                try:
                    from ..operations.workspace import is_workspace_path

                    resolved_name, _, _ = resolve_model_for_operation(model_spec)

                    # Check if resolved_name is a workspace path (ADR-018 Phase 0c)
                    if resolved_name and is_workspace_path(resolved_name):
                        # Workspace path - use directly
                        model_path = Path(resolved_name)

                        # Debug logging for workspace path
                        logger.debug(
                            f"Preload path (workspace): resolved_name={resolved_name}, model_path={model_path}",
                            model=model_spec
                        )

                        # Check workspace exists
                        if not model_path.exists():
                            raise HTTPException(
                                status_code=404,
                                detail=f"Workspace not found: {model_spec}"
                            )
                    elif resolved_name:
                        # Cache model found - existing logic
                        cache_root = get_current_model_cache()
                        cache_dir = cache_root / hf_to_cache_dir(resolved_name)

                        # Debug logging for preload path
                        logger.debug(
                            f"Preload path (cache): resolved_name={resolved_name}, cache_dir={cache_dir}",
                            model=model_spec
                        )

                        # Get actual snapshot path
                        snapshots_dir = cache_dir / "snapshots"
                        model_path = cache_dir
                        if snapshots_dir.exists():
                            snapshots = [d for d in snapshots_dir.iterdir() if d.is_dir()]
                            if snapshots:
                                model_path = max(snapshots, key=lambda x: x.stat().st_mtime)
                    else:
                        # Resolution failed - check if local directory exists as fallback
                        if is_workspace_path(model_spec):
                            # Local workspace exists - use it
                            model_path = Path(model_spec).resolve()
                            resolved_name = str(model_path)

                            logger.debug(
                                f"Preload path (fallback workspace): model_spec={model_spec}, model_path={model_path}",
                                model=model_spec
                            )
                        else:
                            # Not found in cache and no local workspace
                            raise HTTPException(
                                status_code=404,
                                detail=f"Model not found in cache: {model_spec}"
                            )

                    logger.debug(
                        f"Preload path: model_path={model_path}",
                        model=model_spec
                    )

                    # Unified probe/policy check (ARCHITECTURE.md Principle #1: Pipeline always runs)
                    caps, policy = probe_and_select(
                        model_path,
                        resolved_name or model_spec,
                        context="server",
                        has_images=False,  # Preload doesn't have images
                    )

                    # Debug logging for probe/policy results
                    logger.debug(
                        f"Probe/policy results: is_vision={caps.is_vision}, "
                        f"backend={policy.backend.value}, decision={policy.decision.value}, "
                        f"http_status={policy.http_status}",
                        model=model_spec
                    )

                    # Handle policy decision (ARCHITECTURE.md Principle #3: Fail Fast, Fail Clearly)
                    if policy.decision == PolicyDecision.BLOCK:
                        # Use policy-defined HTTP status code (not heuristics on message)
                        # Vision: 501, Memory: 507, Dependency: 501, Framework: 400
                        status_code = policy.http_status or 400  # Default to 400 if not specified
                        logger.error(f"Model blocked by policy: {policy.message}", model=model_spec)
                        raise HTTPException(
                            status_code=status_code,
                            detail=policy.message
                        )

                    if policy.decision == PolicyDecision.WARN:
                        # Log warning but continue (e.g., text model with high memory)
                        logger.warning(f"Policy warning: {policy.message}", model=model_spec)

                except HTTPException:
                    raise  # Re-raise HTTP exceptions from policy

                # Select runner based on backend (ADR-012 Phase 3)
                if policy.backend == Backend.MLX_VLM:
                    # Vision model - use VisionRunner
                    from .vision_runner import VisionRunner
                    logger.info(f"Loading vision model: {model_spec}", model=model_spec, backend="mlx_vlm")
                    runner = VisionRunner(model_path, resolved_name or model_spec, verbose=verbose)
                else:
                    # Text model - use MLXRunner (use resolved_name for workspace support)
                    runner = MLXRunner(resolved_name or model_spec, verbose=verbose, install_signal_handlers=False)

                # If shutdown was requested, abort before expensive load
                if _shutdown_event.is_set():
                    raise KeyboardInterrupt()
                runner.load_model()
                if _shutdown_event.is_set():
                    raise KeyboardInterrupt()

                _model_cache[model_spec] = runner
                _current_model_path = model_spec

                logger.info(f"Switched to model: {model_spec}", model=model_spec)

            except HTTPException:
                # Re-raise HTTP exceptions (501, 507, etc.) from vision/memory checks
                raise
            except KeyboardInterrupt:
                # Handle interruption during model loading
                logger.warning("Model loading interrupted")
                _model_cache.clear()
                _current_model_path = None
                raise HTTPException(status_code=503, detail="Server interrupted during model load")
            except Exception as e:
                # Clean up on failed load
                logger.error(f"Model load failed: {model_spec}", error_key=f"model_load_{model_spec}", detail=str(e))
                _model_cache.clear()
                _current_model_path = None
                status_code = 404
                detail = f"Model '{model_spec}' not found or failed to load: {str(e)}"

                raise HTTPException(status_code=status_code, detail=detail)

        return _model_cache[model_spec]


async def generate_completion_stream(
    runner: MLXRunner,
    prompt: str,
    request: CompletionRequest,
) -> AsyncGenerator[str, None]:
    """Generate streaming completion response."""
    completion_id = f"cmpl-{uuid.uuid4()}"
    created = int(time.time())

    # Yield initial response
    initial_response = {
        "id": completion_id,
        "object": "text_completion",
        "created": created,
        "model": request.model,
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
            max_tokens=get_effective_max_tokens(runner, request.max_tokens, server_mode=True),
            temperature=request.temperature,
            top_p=request.top_p,
            repetition_penalty=request.repetition_penalty,
            use_chat_template=False  # Raw completion mode
        ):
            # Stop promptly if server is shutting down
            if _shutdown_event.is_set():
                raise KeyboardInterrupt()
            token_count += 1

            chunk_response = {
                "id": completion_id,
                "object": "text_completion",
                "created": created,
                "model": request.model,
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
            if request.stop:
                stop_sequences = request.stop if isinstance(request.stop, list) else [request.stop]
                if any(stop in token for stop in stop_sequences):
                    break

    except KeyboardInterrupt:
        # During shutdown/disconnect avoid extra logs; best-effort cleanup
        if not _shutdown_event.is_set():
            try:
                import mlx.core as mx
                mx.clear_cache()
            except Exception:
                pass
            # Try to send an interrupt marker if client still connected
            try:
                interrupt_response = {
                    "id": completion_id,
                    "object": "text_completion",
                    "created": created,
                    "model": request.model,
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
            "model": request.model,
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
    if _shutdown_event.is_set():
        return
    final_response = {
        "id": completion_id,
        "object": "text_completion",
        "created": created,
        "model": request.model,
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
    runner: MLXRunner,
    messages: List[ChatMessage],
    request: ChatCompletionRequest,
) -> AsyncGenerator[str, None]:
    """Generate streaming chat completion response."""
    completion_id = f"chatcmpl-{uuid.uuid4()}"
    created = int(time.time())

    # Convert messages to dict format for runner
    message_dicts = format_chat_messages_for_runner(messages)
    
    # Let the runner format with chat templates
    prompt = runner._format_conversation(message_dicts)

    # Yield initial response
    initial_response = {
        "id": completion_id,
        "object": "chat.completion.chunk",
        "created": created,
        "model": request.model,
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
            max_tokens=get_effective_max_tokens(runner, request.max_tokens, server_mode=True),
            temperature=request.temperature,
            top_p=request.top_p,
            repetition_penalty=request.repetition_penalty,
            use_chat_template=False,  # Already applied in _format_conversation
            use_chat_stop_tokens=True   # Server NEEDS chat stop tokens to prevent self-conversations
        ):
            # Stop promptly if server is shutting down
            if _shutdown_event.is_set():
                raise KeyboardInterrupt()
            chunk_response = {
                "id": completion_id,
                "object": "chat.completion.chunk",
                "created": created,
                "model": request.model,
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
            if request.stop:
                stop_sequences = request.stop if isinstance(request.stop, list) else [request.stop]
                if any(stop in token for stop in stop_sequences):
                    break

    except KeyboardInterrupt:
        if not _shutdown_event.is_set():
            try:
                import mlx.core as mx
                mx.clear_cache()
            except Exception:
                pass
            try:
                interrupt_response = {
                    "id": completion_id,
                    "object": "chat.completion.chunk",
                    "created": created,
                    "model": request.model,
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
        try:
            import os
            if os.environ.get("MLXK2_DEBUG"):
                print(f"[DEBUG] Exception in chat streaming: {type(e).__name__}: {e}")
        except Exception:
            pass
        
        # Try MLX recovery for any exception that might be interrupt-related
        if "interrupt" in str(e).lower() or "keyboard" in str(e).lower():
            try:
                import os
                if os.environ.get("MLXK2_DEBUG"):
                    print("[Server] Detected interrupt-like exception, attempting MLX recovery...")
            except Exception:
                pass
            try:
                import mlx.core as mx
                mx.clear_cache()
                try:
                    import os
                    if os.environ.get("MLXK2_DEBUG"):
                        print("[Server] MLX state recovered after exception")
                except Exception:
                    pass
            except Exception as recovery_error:
                try:
                    import os
                    if os.environ.get("MLXK2_DEBUG"):
                        print(f"[Server] MLX recovery warning: {recovery_error}")
                except Exception:
                    pass
        
        error_response = {
            "id": completion_id,
            "object": "chat.completion.chunk",
            "created": created,
            "model": request.model,
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
    if _shutdown_event.is_set():
        return
    final_response = {
        "id": completion_id,
        "object": "chat.completion.chunk",
        "created": created,
        "model": request.model,
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


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifespan."""
    # Configure log level early (from environment if subprocess mode)
    import os
    env_log_level = os.environ.get("MLXK2_LOG_LEVEL", "info")
    set_log_level(env_log_level)

    logger.info("MLX Knife Server 2.0 starting up...")

    # Pre-load model with probe/policy validation (if specified)
    global _preload_model
    preload_spec = os.environ.get("MLXK2_PRELOAD_MODEL")
    if preload_spec:
        try:
            logger.info(f"Pre-loading model with validation: {preload_spec}")
            # This will trigger probe/policy checks in get_or_load_model()
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

    yield
    logger.info("MLX Knife Server 2.0 shutting down...")
    # Ensure shutdown flag is set so any in-flight generations stop quickly
    try:
        _request_global_interrupt()
    except Exception:
        pass
    # Clean up model cache
    global _model_cache
    try:
        for _runner in list(_model_cache.values()):
            try:
                _runner.cleanup()
            except Exception:
                pass
    finally:
        _model_cache.clear()
        
        # Force MLX memory cleanup
        try:
            import mlx.core as mx
            mx.clear_cache()
            logger.info("MLX memory cleared")
        except Exception:
            pass


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


# Custom exception handler for MLXKError (ADR-004)
@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    """Convert HTTPException to error envelope."""
    request_id = getattr(request.state, "request_id", None)

    # Map HTTP status to error type
    error_type_map = {
        400: ErrorType.VALIDATION_ERROR,
        403: ErrorType.ACCESS_DENIED,
        404: ErrorType.MODEL_NOT_FOUND,
        500: ErrorType.INTERNAL_ERROR,
        501: ErrorType.NOT_IMPLEMENTED,
        503: ErrorType.SERVER_SHUTDOWN,
        507: ErrorType.INSUFFICIENT_MEMORY,
    }

    error_type = error_type_map.get(exc.status_code, ErrorType.INTERNAL_ERROR)
    error = MLXKError(
        type=error_type,
        message=exc.detail,
        retryable=(exc.status_code == 503)
    )

    envelope = error_envelope(error, request_id=request_id)
    return JSONResponse(
        status_code=exc.status_code,
        content=envelope
    )


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    """Convert FastAPI validation errors (422) to ADR-004 envelope (400).

    FastAPI returns 422 Unprocessable Entity for validation errors by default.
    We convert to 400 Bad Request with ADR-004 envelope for API consistency.
    """
    request_id = getattr(request.state, "request_id", None)

    # Format validation errors for detail field
    errors = exc.errors()
    detail = "; ".join(
        f"{'.'.join(str(loc) for loc in e['loc'])}: {e['msg']}"
        for e in errors
    )

    error = MLXKError(
        type=ErrorType.VALIDATION_ERROR,
        message="Request validation failed",
        detail=detail,
        retryable=False
    )

    envelope = error_envelope(error, request_id=request_id)
    return JSONResponse(
        status_code=400,
        content=envelope
    )


@app.get("/health")
async def health_check():
    """Health check endpoint (OpenAI compatible)."""
    return {"status": "healthy", "service": "mlx-knife-server-2.0"}


@app.get("/v1/models")
async def list_models():
    """List available MLX models in the cache.

    Returns models sorted with preloaded model first (if set), then alphabetically.
    Filters to healthy + runtime_compatible models.
    """
    from .cache import cache_dir_to_hf
    from ..operations.common import build_model_object

    model_list = []
    model_cache = get_current_model_cache()

    # Find all model directories (handle missing cache gracefully)
    if not model_cache.exists():
        # Fresh installation or custom cache location - no models yet
        models = []
    else:
        models = [d for d in model_cache.iterdir() if d.name.startswith("models--")]

    for model_dir in models:
        model_name = cache_dir_to_hf(model_dir.name)

        try:
            # Get snapshot path
            snapshots_dir = model_dir / "snapshots"
            selected_path = None
            if snapshots_dir.exists():
                snapshots = [d for d in snapshots_dir.iterdir() if d.is_dir()]
                if snapshots:
                    selected_path = snapshots[0]

            # Use shared build_model_object (single source of truth)
            model_obj = build_model_object(model_name, model_dir, selected_path)

            # Filter: healthy AND runtime_compatible
            if model_obj.get("health") != "healthy":
                continue
            if not model_obj.get("runtime_compatible"):
                continue

            # Get model context length (best effort)
            context_length = None
            try:
                if selected_path:
                    from .runner import get_model_context_length
                    context_length = get_model_context_length(str(selected_path))
            except Exception:
                pass

            model_list.append(ModelInfo(
                id=model_name,
                object="model",
                owned_by="mlx-knife-2.0",
                context_length=context_length
            ))
        except Exception as e:
            # Skip models that can't be processed
            logger.warning(f"Skipping model {model_name} from /v1/models: {e}")
            continue

    # Add preloaded workspace if present and not already in list
    global _preload_model
    if _preload_model:
        # Check if it's a workspace path
        from ..operations.workspace import is_workspace_path
        if is_workspace_path(_preload_model):
            # Check if already in list (avoid duplicates)
            if not any(m.id == _preload_model for m in model_list):
                # Get context length
                context_length = None
                try:
                    from .runner import get_model_context_length
                    context_length = get_model_context_length(_preload_model)
                except Exception:
                    pass

                model_list.append(ModelInfo(
                    id=_preload_model,  # Original path string
                    object="model",
                    owned_by="workspace",
                    context_length=context_length
                ))

    # Sort: preloaded model first, then alphabetically by id
    if _preload_model:
        def sort_key(model: ModelInfo):
            # Preloaded model gets priority (0), others sorted alphabetically
            return (0 if model.id == _preload_model else 1, model.id)
        model_list.sort(key=sort_key)
    else:
        # No preload: just alphabetical
        model_list.sort(key=lambda m: m.id)

    return {"object": "list", "data": model_list}


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


async def _handle_text_chat_completion(request: ChatCompletionRequest, runner: Any = None) -> ChatCompletionResponse:
    """Handle text-only chat completion.

    Works with both MLXRunner (text models) and VisionRunner (vision models without images).

    Args:
        request: Chat completion request
        runner: Pre-loaded model runner (optional, will load if not provided)
    """
    if runner is None:
        runner = get_or_load_model(request.model)

    # Check if runner is VisionRunner (vision model loaded without images in request)
    from .vision_runner import VisionRunner
    is_vision_runner = isinstance(runner, VisionRunner)

    # Filter multimodal history for text-only models
    # Vision models can handle multimodal content, text models cannot
    # See: docs/ISSUES/VISION-MULTIMODAL-HISTORY-ISSUE.md
    messages = request.messages
    if not is_vision_runner:
        # Text-only model: filter out image_url content from history
        messages = _filter_multimodal_history_for_text_models(messages)

    if is_vision_runner:
        # Vision model: use VisionRunner.generate() without images
        completion_id = f"chatcmpl-{uuid.uuid4()}"
        created = int(time.time())

        # Extract text prompt from messages (already filtered if needed)
        prompt = _extract_text_from_messages(messages)

        generated_text = runner.generate(
            prompt=prompt,
            images=None,  # No images for text-only request
            max_tokens=get_effective_max_tokens_vision(runner, request.max_tokens),
            temperature=0.0,  # Experiment: greedy sampling to reduce hallucinations
            top_p=request.top_p or 0.9,
            repetition_penalty=request.repetition_penalty or 1.0,
        )

        prompt_tokens = count_tokens(prompt)
        completion_tokens = count_tokens(generated_text)

        # Graceful degradation: emulate SSE for stream=true
        if request.stream:
            logger.info("Vision model: emulating SSE stream (batch response as single event)")
            return StreamingResponse(
                _emulate_sse_stream(completion_id, created, request.model, generated_text),
                media_type="text/event-stream",
                headers={"Cache-Control": "no-cache"}
            )

        return ChatCompletionResponse(
            id=completion_id,
            created=created,
            model=request.model,
            choices=[
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": generated_text
                    },
                    "finish_reason": "stop"
                }
            ],
            usage={
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": prompt_tokens + completion_tokens
            }
        )

    # Text model: use MLXRunner (existing behavior)
    if request.stream:
        # Streaming response (use filtered messages)
        return StreamingResponse(
            generate_chat_stream(runner, messages, request),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache"}
        )

    # Non-streaming response
    completion_id = f"chatcmpl-{uuid.uuid4()}"
    created = int(time.time())

    # Convert messages to dict format for runner (use filtered messages)
    message_dicts = format_chat_messages_for_runner(messages)

    # Let the runner format with chat templates
    prompt = runner._format_conversation(message_dicts)

    generated_text = runner.generate_batch(
        prompt=prompt,
        max_tokens=get_effective_max_tokens(runner, request.max_tokens, server_mode=True),
        temperature=request.temperature,
        top_p=request.top_p,
        repetition_penalty=request.repetition_penalty,
        use_chat_template=False,  # Already applied in _format_conversation
        use_chat_stop_tokens=True  # Server NEEDS chat stop tokens to prevent self-conversations
    )

    # Token counting (handle both string and list content - use filtered messages)
    total_prompt = _extract_text_from_messages(messages)
    prompt_tokens = count_tokens(total_prompt)
    completion_tokens = count_tokens(generated_text)

    return ChatCompletionResponse(
        id=completion_id,
        created=created,
        model=request.model,
        choices=[
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": generated_text
                },
                "finish_reason": "stop"
            }
        ],
        usage={
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens
        }
    )


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

    Each chunk creates a fresh VisionRunner to prevent state leakage between batches.
    Similar to CLI _process_images_in_chunks but server-optimized
    (no verbose output to stderr, uses existing image_id_map).

    Args:
        model_path: Path to model snapshot directory
        model_name: Model name for VisionRunner
        prompt: User prompt
        images: Full list of (filename, bytes) tuples
        chunk_size: Images per batch
        image_id_map: Pre-computed global image IDs (from conversation history)
        max_tokens, temperature, top_p, repetition_penalty: Generation params
        audio: Optional list of (filename, bytes) tuples for audio input

    Returns:
        Combined text with merged filename mappings
    """
    from .vision_runner import VisionRunner

    # Split into chunks
    chunks = [images[i:i+chunk_size] for i in range(0, len(images), chunk_size)]

    # Process each chunk with fresh runner (prevents state leakage)
    all_results = []
    for chunk in chunks:
        # Fresh runner per chunk to prevent KV-cache/state accumulation
        with VisionRunner(model_path, model_name, verbose=False) as runner:
            chunk_result = runner.generate(
                prompt=prompt,
                images=chunk,
                audio=audio,  # Pass audio with each chunk
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                repetition_penalty=repetition_penalty,
                image_id_map=image_id_map,  # Global numbering from conversation
                total_images=len(images),  # Enable chunk context line
            )
        all_results.append(chunk_result)

    # Concatenate results
    return "\n\n".join(all_results)


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
        audio: Optional list of (filename, bytes) tuples for audio input

    Yields:
        SSE event strings (data: {...}\n\n format)
    """
    import asyncio
    from .vision_runner import VisionRunner

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
        if _shutdown_event.is_set():
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


async def _handle_vision_chat_completion(request: ChatCompletionRequest, runner: Any = None) -> ChatCompletionResponse:
    """Handle vision/audio chat completion with images or audio (ADR-012 Phase 3, ADR-019 Phase 4).

    Supports per-chunk streaming for multi-image requests (stream=True yields
    SSE events as each chunk completes). Single-chunk requests use batch mode
    with optional SSE emulation.

    Args:
        request: Chat completion request
        runner: Pre-loaded model runner (optional, will load if not provided)
    """
    # Lazy import vision components (Python 3.9 compatibility)
    from ..tools.vision_adapter import VisionHTTPAdapter

    # Vision/Audio requests are STATELESS for the model prompt, but we track image IDs
    # across the session for consistent numbering (Image 1, 2, 3...).
    #
    # Rationale:
    # - Vision models can't "see" previous images (Metal memory limitations)
    # - History text causes pattern reproduction (model copies mappings)
    # - But we want consistent image numbering across the conversation
    # - Follow-up questions should use text models (which have history)
    message_dicts = _messages_to_dicts(request.messages)

    # Find the last user message only (for stateless prompt)
    last_user_msg = None
    for msg in reversed(message_dicts):
        if msg.get("role") == "user":
            last_user_msg = msg
            break

    if last_user_msg is None:
        raise HTTPException(status_code=400, detail="No user message found")

    # Parse only the last user message (stateless prompt)
    # Returns (prompt, images, audio) tuple
    prompt, images, audio = VisionHTTPAdapter.parse_openai_messages([last_user_msg])

    # But use FULL history for image ID assignment (consistent numbering)
    image_id_map = VisionHTTPAdapter.assign_image_ids_from_history(message_dicts)

    image_count = len(images)
    audio_count = len(audio)
    logger.info(
        f"Vision/Audio request: {image_count} image(s), {audio_count} audio(s), model={request.model}",
        model=request.model,
        image_count=image_count,
        audio_count=audio_count
    )

    # Get or load VisionRunner (uses cache if model already loaded)
    # get_or_load_model now handles Backend selection and uses VisionRunner for vision models
    if runner is None:
        runner = get_or_load_model(request.model, verbose=False)

    # Verify we got a VisionRunner (not MLXRunner) for vision request
    from .vision_runner import VisionRunner
    if not isinstance(runner, VisionRunner):
        # Model was loaded as text model but vision request received
        # This shouldn't happen if probe_and_select works correctly
        raise HTTPException(
            status_code=400,
            detail=f"Model '{request.model}' does not support vision inputs"
        )

    # Generate with VisionRunner
    completion_id = f"chatcmpl-{uuid.uuid4()}"
    created = int(time.time())

    # Get chunk size: request param > ENV > default (F-05: respect explicit chunk=1)
    env_chunk = int(os.environ.get("MLXK2_VISION_CHUNK_SIZE", "1"))
    chunk_size = request.chunk if request.chunk is not None else env_chunk

    # F-03: Audio+Vision - audio silently ignored when images present (SERVER-HANDBOOK)
    # mlx-vlm behavior is undefined for combined audio+vision, so we enforce single modality
    effective_audio = None if images else audio

    # Validate chunk size for Metal API stability
    from ..tools.vision_adapter import MAX_SAFE_CHUNK_SIZE
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

    if len(images) <= chunk_size:
        # Single batch (no chunking) - also handles audio-only requests
        generated_text = runner.generate(
            prompt=prompt,
            images=images if images else None,
            audio=effective_audio,
            max_tokens=get_effective_max_tokens_vision(runner, request.max_tokens),
            temperature=0.0,  # Experiment: greedy sampling to reduce hallucinations
            top_p=request.top_p or 0.9,
            repetition_penalty=request.repetition_penalty or 1.0,
            image_id_map=image_id_map if images else None,
        )
    else:
        # Multi-chunk processing
        if request.stream:
            # True per-chunk streaming (yields SSE events as chunks complete)
            logger.info(
                f"Vision request: chunk streaming ({len(images)} images, chunk_size={chunk_size})",
                model=request.model,
                image_count=len(images),
                chunk_size=chunk_size
            )
            return StreamingResponse(
                _stream_vision_chunks(
                    model_path=runner.model_path,
                    model_name=runner.model_name,
                    prompt=prompt,
                    images=images,
                    chunk_size=chunk_size,
                    image_id_map=image_id_map,
                    max_tokens=get_effective_max_tokens_vision(runner, request.max_tokens),
                    temperature=0.0,
                    top_p=request.top_p or 0.9,
                    repetition_penalty=request.repetition_penalty or 1.0,
                    completion_id=completion_id,
                    created=created,
                    model=request.model,
                    audio=effective_audio,
                ),
                media_type="text/event-stream",
                headers={"Cache-Control": "no-cache"}
            )
        # Non-streaming multi-chunk (batch mode)
        generated_text = _process_vision_chunks_server(
            model_path=runner.model_path,
            model_name=runner.model_name,
            prompt=prompt,
            images=images,
            chunk_size=chunk_size,
            image_id_map=image_id_map,
            max_tokens=get_effective_max_tokens_vision(runner, request.max_tokens),
            temperature=0.0,
            top_p=request.top_p or 0.9,
            repetition_penalty=request.repetition_penalty or 1.0,
            audio=effective_audio,
        )

    logger.info(
        f"Vision generation complete: {len(generated_text)} chars",
        model=request.model,
        output_length=len(generated_text)
    )

    # Token counting
    prompt_tokens = count_tokens(prompt)
    completion_tokens = count_tokens(generated_text)

    # Graceful degradation: emulate SSE for stream=true (single-chunk only)
    if request.stream:
        logger.info("Vision request: emulating SSE stream (single-chunk batch response)")
        return StreamingResponse(
            _emulate_sse_stream(completion_id, created, request.model, generated_text),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache"}
        )

    return ChatCompletionResponse(
        id=completion_id,
        created=created,
        model=request.model,
        choices=[
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": generated_text
                },
                "finish_reason": "stop"
            }
        ],
        usage={
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens
        }
    )


async def _emulate_sse_stream(
    completion_id: str,
    created: int,
    model: str,
    content: str
) -> AsyncGenerator[str, None]:
    """Emulate SSE streaming for vision models (batch response as SSE events).

    Vision models don't support real streaming, so we send the complete
    response as SSE events to be compatible with streaming clients.
    """
    import json

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


def cleanup_server():
    """Manual cleanup function for emergency situations."""
    global _model_cache, _current_model_path
    logger.warning("Forcing server cleanup...")

    # Thread-safe cleanup
    with _model_lock:
        try:
            for _runner in list(_model_cache.values()):
                try:
                    _runner.cleanup()
                except Exception as e:
                    logger.warning(f"Warning during runner cleanup: {e}")
        finally:
            _model_cache.clear()
            _current_model_path = None

            # Force MLX memory cleanup
            try:
                import mlx.core as mx
                mx.clear_cache()
                logger.info("MLX memory cleared")
            except Exception as e:
                logger.warning(f"Warning during MLX cleanup: {e}")


def _request_global_interrupt() -> None:
    """Request all running generations to stop quickly.

    Used during server shutdown to ensure in-flight streams stop.
    """
    _shutdown_event.set()
    try:
        with _model_lock:
            for _runner in list(_model_cache.values()):
                try:
                    _runner.request_interrupt()
                except Exception:
                    pass
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

    # Configure Uvicorn log format (JSON if MLXK2_LOG_JSON=1)
    json_mode = os.environ.get("MLXK2_LOG_JSON", "0") == "1"
    log_config = None
    if json_mode:
        # Use custom log config for JSON formatting
        log_config = {
            "version": 1,
            "disable_existing_loggers": False,
            "formatters": {
                "default": {
                    "()": "mlxk2.logging.JSONFormatter",
                },
                "access": {
                    "()": "mlxk2.logging.JSONFormatter",
                },
            },
            "handlers": {
                "default": {
                    "formatter": "default",
                    "class": "logging.StreamHandler",
                    "stream": "ext://sys.stderr",
                },
                "access": {
                    "formatter": "access",
                    "class": "logging.StreamHandler",
                    "stream": "ext://sys.stderr",
                },
            },
            "loggers": {
                "uvicorn": {"handlers": ["default"], "level": log_level.upper()},
                "uvicorn.error": {"handlers": ["default"], "level": log_level.upper(), "propagate": False},
                "uvicorn.access": {"handlers": ["access"], "level": "INFO", "propagate": False},
            },
        }

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
