# mlx_knife/server.py
"""
OpenAI-compatible API server for MLX models.
Provides REST endpoints for text generation with MLX backend.
"""

import json
import time
import uuid
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from typing import Any, Dict, List, Optional, Union

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from .cache_utils import detect_framework, is_model_healthy
from .mlx_runner import MLXRunner

# Global model cache and configuration
_model_cache: Dict[str, MLXRunner] = {}
_current_model_path: Optional[str] = None
_default_max_tokens: int = 2000


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


def get_effective_max_tokens(request_max_tokens: Optional[int]) -> int:
    """Get effective max_tokens value, using global default if not specified."""
    global _default_max_tokens
    return request_max_tokens if request_max_tokens is not None else _default_max_tokens


def get_or_load_model(model_spec: str, verbose: bool = False) -> MLXRunner:
    """Get model from cache or load it if not cached."""
    global _model_cache, _current_model_path

    # Use the existing model path resolution from cache_utils
    from .cache_utils import get_model_path

    try:
        model_path, model_name, commit_hash = get_model_path(model_spec)
        if not model_path.exists():
            raise HTTPException(status_code=404, detail=f"Model {model_spec} not found in cache")
    except Exception as e:
        raise HTTPException(status_code=404, detail=f"Model {model_spec} not found: {str(e)}")

    # Check if it's an MLX model
    framework = detect_framework(model_path.parent.parent, model_name)
    if framework != "MLX":
        raise HTTPException(status_code=400, detail=f"Model {model_name} is not a valid MLX model (Framework: {framework})")

    model_path_str = str(model_path)

    # Check if we need to load a different model
    if _current_model_path != model_path_str:
        # Clear cache if switching models to avoid memory issues
        _model_cache.clear()

        # Load new model
        if verbose:
            print(f"Loading model: {model_name}")

        runner = MLXRunner(model_path_str, verbose=verbose)
        runner.load_model()

        _model_cache[model_path_str] = runner
        _current_model_path = model_path_str

    return _model_cache[model_path_str]


async def generate_completion_stream(
    runner: MLXRunner,
    prompt: str,
    request: CompletionRequest
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
            max_tokens=get_effective_max_tokens(request.max_tokens),
            temperature=request.temperature,
            top_p=request.top_p,
            repetition_penalty=request.repetition_penalty,
            use_chat_template=False  # Raw completion mode
        ):
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

    # Final response
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
    request: ChatCompletionRequest
) -> AsyncGenerator[str, None]:
    """Generate streaming chat completion response."""
    completion_id = f"chatcmpl-{uuid.uuid4()}"
    created = int(time.time())

    # Convert messages to prompt
    prompt = format_chat_messages(messages)

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
            max_tokens=get_effective_max_tokens(request.max_tokens),
            temperature=request.temperature,
            top_p=request.top_p,
            repetition_penalty=request.repetition_penalty,
            use_chat_template=True
        ):
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

    except Exception as e:
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

    # Final response
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


def format_chat_messages(messages: List[ChatMessage]) -> str:
    """Convert chat messages to a prompt string."""
    # Simple format - models with chat templates will format properly
    formatted = []
    for message in messages:
        if message.role == "system":
            formatted.append(f"System: {message.content}")
        elif message.role == "user":
            formatted.append(f"Human: {message.content}")
        elif message.role == "assistant":
            formatted.append(f"Assistant: {message.content}")

    return "\n\n".join(formatted)


def count_tokens(text: str) -> int:
    """Rough token count estimation."""
    return int(len(text.split()) * 1.3)  # Approximation, convert to int


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifespan."""
    print("MLX Knife Server starting up...")
    yield
    print("MLX Knife Server shutting down...")
    # Clean up model cache
    global _model_cache
    _model_cache.clear()


# Create FastAPI app
from . import __version__

app = FastAPI(
    title="MLX Knife API",
    description="OpenAI-compatible API for MLX models",
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


@app.get("/health")
async def health_check():
    """Health check endpoint (OpenAI compatible)."""
    return {"status": "healthy", "service": "mlx-knife-server"}




@app.get("/v1/models")
async def list_models():
    """List available models."""
    from .cache_utils import MODEL_CACHE, cache_dir_to_hf

    model_list = []
    models = [d for d in MODEL_CACHE.iterdir() if d.name.startswith("models--")]

    for model_dir in models:
        model_name = cache_dir_to_hf(model_dir.name)
        framework = detect_framework(model_dir, model_name)

        if framework == "MLX" and is_model_healthy(model_name):
            model_list.append(ModelInfo(
                id=model_name,
                object="model",
                owned_by="mlx-knife"
            ))

    return {"object": "list", "data": model_list}


@app.post("/v1/completions")
async def create_completion(request: CompletionRequest):
    """Create a text completion."""
    try:
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
                media_type="text/plain",
                headers={"Cache-Control": "no-cache"}
            )
        else:
            # Non-streaming response
            completion_id = f"cmpl-{uuid.uuid4()}"
            created = int(time.time())

            generated_text = runner.generate_batch(
                prompt=prompt,
                max_tokens=get_effective_max_tokens(request.max_tokens),
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

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/v1/chat/completions")
async def create_chat_completion(request: ChatCompletionRequest):
    """Create a chat completion."""
    try:
        runner = get_or_load_model(request.model)

        if request.stream:
            # Streaming response
            return StreamingResponse(
                generate_chat_stream(runner, request.messages, request),
                media_type="text/plain",
                headers={"Cache-Control": "no-cache"}
            )
        else:
            # Non-streaming response
            completion_id = f"chatcmpl-{uuid.uuid4()}"
            created = int(time.time())

            # Format messages to prompt
            prompt = format_chat_messages(request.messages)

            generated_text = runner.generate_batch(
                prompt=prompt,
                max_tokens=get_effective_max_tokens(request.max_tokens),
                temperature=request.temperature,
                top_p=request.top_p,
                repetition_penalty=request.repetition_penalty,
                use_chat_template=True
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

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


def run_server(
    host: str = "127.0.0.1",
    port: int = 8000,
    max_tokens: int = 2000,
    reload: bool = False,
    log_level: str = "info"
):
    """Run the MLX Knife server."""
    global _default_max_tokens
    _default_max_tokens = max_tokens

    print(f"Starting MLX Knife Server on http://{host}:{port}")
    print(f"API docs available at http://{host}:{port}/docs")
    print(f"Default max tokens: {max_tokens}")

    uvicorn.run(
        "mlx_knife.server:app",
        host=host,
        port=port,
        reload=reload,
        log_level=log_level
    )
