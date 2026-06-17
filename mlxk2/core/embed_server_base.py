"""embed-serve: single-model OpenAI-compatible embeddings backend (ADR-015 Slice D1).

A **separate process** from ``mlxk serve``. It owns exactly one :class:`EmbeddingRunner`
(loaded at startup, held for the process lifetime — *not* the one-shot ``with`` form) and
exposes ``POST /v1/embeddings`` + ``GET /health``. The main serve never loads an embedding
model in its address space (ADR-015 §Server Topology); in 2.0.7 ``serve --embed-backend URL``
proxies ``/v1/embeddings`` to this backend (Slice D2). Single model, no swapping, localhost-
internal — so there is no ModelManager here: a single ``threading.Lock`` serializes inference
(the minimal subset of what ModelManager provides, minus the inapplicable swap/memory-gate
machinery calibrated for the multi-GB main-serve process).
"""

import os
import threading
from contextlib import asynccontextmanager
from typing import List, Optional, Union

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from .. import __version__
from ..context import generate_request_id
from ..logging import get_logger, set_log_level, uvicorn_log_config
from .embedding_runner import EmbeddingRunner
from .model_resolution import model_display_identity, resolve_model_for_operation
from .server.error_handlers import register_error_handlers
from .server.handlers.embeddings import handle_embeddings

logger = get_logger()

# Single held runner + its portable identity + an inference lock (set by lifespan).
_runner: Optional[EmbeddingRunner] = None
_model_identity: str = ""
_infer_lock = threading.Lock()


# -- OpenAI-compatible request/response models ------------------------------
class EmbeddingRequest(BaseModel):
    model: str
    input: Union[str, List[str]]
    encoding_format: Optional[str] = "base64"   # OpenAI default; the SDK sends it explicitly
    dimensions: Optional[int] = None            # rejected if != native (no truncation in D1)
    user: Optional[str] = None
    # mlxk extensions (RAG) — additive; ignored by standard OpenAI clients:
    input_type: Optional[str] = "document"      # "document" | "query"
    instruct: Optional[str] = None              # query task override; implies query


class EmbeddingData(BaseModel):
    object: str = "embedding"
    index: int
    embedding: Union[List[float], str]          # float list OR base64 string


class EmbeddingUsage(BaseModel):
    prompt_tokens: int
    total_tokens: int


class EmbeddingResponse(BaseModel):
    object: str = "list"
    data: List[EmbeddingData]
    model: str
    usage: EmbeddingUsage


# -- lifespan: load + hold the single runner --------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Resolve + route + load the single embedding model; hold it for the process lifetime."""
    global _runner, _model_identity

    set_log_level(os.environ.get("MLXK2_LOG_LEVEL", "info"))

    model_spec = os.environ.get("MLXK2_EMBED_MODEL")
    if not model_spec:
        raise RuntimeError("embed-serve: MLXK2_EMBED_MODEL is not set")
    cpu = os.environ.get("MLXK2_EMBED_CPU", "0") == "1"

    logger.info(
        f"embed-serve starting: model={model_spec} device={'cpu' if cpu else 'gpu'}"
    )

    # Fail-fast pre-flight (mirrors embed_operation): resolve + route before loading.
    resolved_name, commit_hash, ambiguous = resolve_model_for_operation(model_spec)
    if ambiguous:
        raise RuntimeError(
            f"Ambiguous model specification '{model_spec}'. Could be: {ambiguous}"
        )
    if not resolved_name:
        raise RuntimeError(f"Model '{model_spec}' not found")

    from .capabilities import classify_embedder
    from ..operations.common import _load_config_json
    from .embedding_runner import resolve_model_dir

    try:
        model_dir = resolve_model_dir(resolved_name, commit_hash)
        config = _load_config_json(model_dir) or {}
    except FileNotFoundError as e:
        raise RuntimeError(str(e)) from e
    route, model_type = classify_embedder(config, resolved_name)
    if route == "encoder_declared":
        raise RuntimeError(
            f"'{resolved_name}' is a {model_type} encoder embedder, which this build does not "
            f"vendor yet. Use bge/e5 (model_type bert) or a decoder embedder such as Qwen3-Embedding."
        )
    if route is None:
        raise RuntimeError(f"Model '{resolved_name}' is not recognized as an embedding model.")

    # Load and HOLD the runner (not the one-shot context manager).
    runner = EmbeddingRunner(resolved_name, cpu=cpu, verbose=False)
    try:
        runner.load_model()
    except Exception as e:
        raise RuntimeError(f"Failed to load embedding model '{resolved_name}': {e}") from e

    _runner = runner
    _model_identity = model_display_identity(resolved_name, commit_hash)[0]
    logger.info(f"embed-serve ready: {_model_identity} ({runner.dimensions} dims)")

    yield

    logger.info("embed-serve shutting down...")
    # Hold the inference lock during cleanup so the model can't be freed while a sync-def request
    # is still inside embed() (uvicorn cannot cancel a threadpool worker). Bounded so a stuck
    # request can't hang shutdown forever.
    acquired = _infer_lock.acquire(timeout=10)
    try:
        if _runner is not None:
            _runner.cleanup()
    except Exception as e:
        logger.warning(f"embed-serve cleanup failed: {e}")
    finally:
        if acquired:
            _infer_lock.release()
    _runner = None
    _model_identity = ""


app = FastAPI(
    title="MLX Knife embed-serve",
    description="OpenAI-compatible embeddings backend (ADR-015)",
    version=__version__,
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def add_request_id_middleware(request: Request, call_next):
    """Add request_id to all requests for correlation (ADR-004)."""
    request_id = generate_request_id()
    request.state.request_id = request_id
    response = await call_next(request)
    response.headers["X-Request-ID"] = request_id
    return response


register_error_handlers(app)


@app.get("/health")
async def health():
    """Liveness check (200 once the model is loaded)."""
    if _runner is None:
        raise HTTPException(status_code=503, detail="Server not ready - model not loaded")
    return {"status": "ok", "model": _model_identity}


@app.post("/v1/embeddings")
def create_embeddings(request: EmbeddingRequest):
    """OpenAI-compatible embeddings. Sync def → FastAPI runs it in a threadpool; the inference
    lock serializes the single model so the event loop stays responsive for /health."""
    if _runner is None:
        raise HTTPException(status_code=503, detail="Server not ready - model not loaded")

    inputs = [request.input] if isinstance(request.input, str) else list(request.input)
    # instruct implies the query side (mirrors CLI `embed --instruct`).
    input_type = "query" if request.instruct else (request.input_type or "document")

    result = handle_embeddings(
        runner=_runner,
        model_identity=_model_identity or request.model,
        inputs=inputs,
        encoding_format=request.encoding_format or "base64",
        dimensions=request.dimensions,
        input_type=input_type,
        instruct=request.instruct,
        lock=_infer_lock,
    )
    return EmbeddingResponse(**result)


# CLI entrypoint for supervised mode (python -m mlxk2.core.embed_server_base).
# Reads config from env vars set by operations/embed_serve.py before Popen.
if __name__ == "__main__":
    host = os.environ.get("MLXK2_HOST", "127.0.0.1")
    port = int(os.environ.get("MLXK2_PORT", "8002"))
    log_level = os.environ.get("MLXK2_LOG_LEVEL", "info")

    try:
        import uvicorn  # type: ignore
    except Exception as e:  # noqa: BLE001
        raise RuntimeError(
            "uvicorn is required to run embed-serve; install with 'pip install fastapi uvicorn'."
        ) from e

    set_log_level(log_level)
    access_log_enabled = log_level.lower() in ("debug", "info")
    log_config = uvicorn_log_config(log_level)  # shared with serve; per-process

    logger.info(f"Starting MLX Knife embed-serve on http://{host}:{port}")

    uvicorn.run(
        "mlxk2.core.embed_server_base:app",
        host=host,
        port=port,
        log_level=log_level,
        log_config=log_config,
        access_log=access_log_enabled,
        workers=1,
        timeout_graceful_shutdown=5,
        timeout_keep_alive=5,
        lifespan="on",
    )
