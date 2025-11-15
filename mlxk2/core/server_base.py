"""
OpenAI-compatible API server for MLX models (2.0 implementation).
Provides REST endpoints for text generation with MLX backend.
"""

import json
import threading
import time
import uuid
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from typing import Any, Dict, List, Optional, Union

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, JSONResponse
from pydantic import BaseModel, Field

from .cache import get_current_model_cache
from .runner import MLXRunner
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
    content: str


class ChatCompletionRequest(BaseModel):
    model: str
    messages: List[ChatMessage]
    max_tokens: Optional[int] = None
    temperature: Optional[float] = 0.7
    top_p: Optional[float] = 0.9
    stream: Optional[bool] = False
    stop: Optional[Union[str, List[str]]] = None
    repetition_penalty: Optional[float] = 1.1


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


def get_or_load_model(model_spec: str, verbose: bool = False) -> MLXRunner:
    """Get model from cache or load it if not cached.
    
    Thread-safe model switching with proper cleanup on interruption.
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
            logger.info(f"Switching to model: {model_spec}", model=model_spec)

            # Clean up previous model
            if _model_cache:
                try:
                    for _old_runner in list(_model_cache.values()):
                        try:
                            _old_runner.cleanup()
                        except Exception as e:
                            logger.warning(f"Warning during cleanup: {e}")
                finally:
                    _model_cache.clear()
                    _current_model_path = None

            # Load new model (disable signal handlers for server mode)
            try:
                runner = MLXRunner(model_spec, verbose=verbose, install_signal_handlers=False)
                # If shutdown was requested, abort before expensive load
                if _shutdown_event.is_set():
                    raise KeyboardInterrupt()
                runner.load_model()
                if _shutdown_event.is_set():
                    raise KeyboardInterrupt()

                _model_cache[model_spec] = runner
                _current_model_path = model_spec

                logger.info(f"Model loaded successfully: {model_spec}", model=model_spec)

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
                raise HTTPException(status_code=404, detail=f"Model '{model_spec}' not found or failed to load: {str(e)}")

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
    """Get effective max tokens with server DoS protection."""
    if requested_max_tokens is not None:
        return requested_max_tokens
    else:
        # Use runner's dynamic calculation with server_mode flag
        return runner._calculate_dynamic_max_tokens(server_mode=server_mode)


def count_tokens(text: str) -> int:
    """Rough token count estimation."""
    return int(len(text.split()) * 1.3)  # Approximation, convert to int


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifespan."""
    # Configure log level early (from environment if subprocess mode)
    import os
    env_log_level = os.environ.get("MLXK2_LOG_LEVEL", "info")
    set_log_level(env_log_level)

    logger.info("MLX Knife Server 2.0 starting up...")
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
        403: ErrorType.ACCESS_DENIED,
        404: ErrorType.MODEL_NOT_FOUND,
        400: ErrorType.VALIDATION_ERROR,
        503: ErrorType.SERVER_SHUTDOWN,
        500: ErrorType.INTERNAL_ERROR,
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


@app.get("/health")
async def health_check():
    """Health check endpoint (OpenAI compatible)."""
    return {"status": "healthy", "service": "mlx-knife-server-2.0"}


@app.get("/v1/models")
async def list_models():
    """List available MLX models in the cache."""
    from .cache import cache_dir_to_hf
    from ..operations.common import detect_framework, read_front_matter
    from ..operations.health import is_model_healthy
    
    model_list = []
    model_cache = get_current_model_cache()
    
    # Find all model directories
    models = [d for d in model_cache.iterdir() if d.name.startswith("models--")]
    
    for model_dir in models:
        model_name = cache_dir_to_hf(model_dir.name)
        
        try:
            # Check if it's a healthy MLX model
            # Get the latest snapshot for detection
            snapshots_dir = model_dir / "snapshots"
            selected_path = None
            if snapshots_dir.exists():
                snapshots = [d for d in snapshots_dir.iterdir() if d.is_dir()]
                if snapshots:
                    selected_path = snapshots[0]

            # Read front-matter for framework detection (align with CLI behavior)
            probe = selected_path if selected_path is not None else model_dir
            fm = read_front_matter(probe)

            if detect_framework(model_name, model_dir, selected_path, fm) == "MLX" and is_model_healthy(model_name)[0]:
                # Get model context length (best effort)
                context_length = None
                try:
                    snapshots_dir = model_dir / "snapshots"
                    if snapshots_dir.exists():
                        snapshots = [d for d in snapshots_dir.iterdir() if d.is_dir()]
                        if snapshots:
                            from .runner import get_model_context_length
                            context_length = get_model_context_length(str(snapshots[0]))
                except Exception:
                    pass

                model_list.append(ModelInfo(
                    id=model_name,
                    object="model",
                    owned_by="mlx-knife-2.0",
                    context_length=context_length
                ))
        except Exception:
            # Skip models that can't be processed
            continue

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
    """Create a chat completion."""
    try:
        if _shutdown_event.is_set():
            raise HTTPException(status_code=503, detail="Server is shutting down")
        runner = get_or_load_model(request.model)

        if request.stream:
            # Streaming response
            return StreamingResponse(
                generate_chat_stream(runner, request.messages, request),
                media_type="text/event-stream",
                headers={"Cache-Control": "no-cache"}
            )
        else:
            # Non-streaming response
            completion_id = f"chatcmpl-{uuid.uuid4()}"
            created = int(time.time())

            # Convert messages to dict format for runner
            message_dicts = format_chat_messages_for_runner(request.messages)
            
            # Let the runner format with chat templates
            prompt = runner._format_conversation(message_dicts)

            generated_text = runner.generate_batch(
                prompt=prompt,
                max_tokens=get_effective_max_tokens(runner, request.max_tokens, server_mode=True),
                temperature=request.temperature,
                top_p=request.top_p,
                repetition_penalty=request.repetition_penalty,
                use_chat_template=False,  # Already applied in _format_conversation
                use_chat_stop_tokens=True   # Server NEEDS chat stop tokens to prevent self-conversations
            )

            # Token counting
            total_prompt = "\n\n".join([msg.content for msg in request.messages])
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

    except HTTPException as http_exc:
        # Preserve intended HTTP status codes from inner helpers
        raise http_exc
    except Exception as e:
        # Map unexpected errors to 500
        raise HTTPException(status_code=500, detail=str(e))


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
    log_level: str = "info"
):
    """Run the MLX Knife server 2.0."""
    import os

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
                "uvicorn.error": {"level": log_level.upper()},
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
