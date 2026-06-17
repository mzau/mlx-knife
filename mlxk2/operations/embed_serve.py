"""``mlxk embed-serve`` operation (ADR-015 Slice D1).

Starts the single-model embeddings backend (``core/embed_server_base.py``) as a supervised
subprocess, reusing serve's signal-handling supervisor. Mirrors ``operations/serve.py``:
pre-validate the model fail-fast in the parent (clean CLI error before any boot), pass config
via env vars, then delegate to the shared ``_run_supervised_uvicorn`` with the embed module.
"""

import os
import sys

from .serve import _run_supervised_uvicorn


def start_embed_serve(
    model: str,
    port: int = 8002,
    host: str = "127.0.0.1",
    cpu: bool = False,
    log_level: str = "info",
    verbose: bool = False,
    supervise: bool = True,
) -> None:
    """Start the OpenAI-compatible embeddings backend for a single model.

    Args:
        model: Embedding model name, HF ID, or workspace path (single model, no swapping).
        port: Port to bind (default 8002; localhost-internal backend).
        host: Host to bind (default 127.0.0.1).
        cpu: Force CPU execution (recommended when co-resident with a GPU-bound serve).
        log_level: Logging level.
        verbose: Print startup details.
        supervise: Run uvicorn in a supervised subprocess for instant Ctrl-C.
    """
    # Pre-validate model before boot (fail fast with a clear CLI error — mirrors start_server).
    from ..core.model_resolution import resolve_model_for_operation
    from .embed import _route_embedding
    from .workspace import is_explicit_path

    resolved_name, commit_hash, ambiguous = resolve_model_for_operation(model)
    if ambiguous:
        raise ValueError(
            f"Ambiguous model specification '{model}'. Could be: {ambiguous}"
        )
    if not resolved_name:
        if is_explicit_path(model):
            raise ValueError(f"Workspace not found: {model}")
        raise ValueError(f"Model not found in cache: {model}")

    route, model_type = _route_embedding(resolved_name, commit_hash)
    if route == "encoder_declared":
        raise ValueError(
            f"'{resolved_name}' is a {model_type} encoder embedder, which this build does not "
            f"vendor yet. Use bge/e5 (model_type bert) or a decoder embedder such as Qwen3-Embedding."
        )
    if route is None:
        raise ValueError(
            f"Model '{resolved_name}' is not recognized as an embedding model."
        )

    # Environment for both supervised and in-process modes.
    os.environ["MLXK2_LOG_LEVEL"] = log_level
    os.environ["TQDM_DISABLE"] = "1"

    extra_env = {
        "MLXK2_EMBED_MODEL": model,
        "MLXK2_EMBED_CPU": "1" if cpu else "0",
    }

    if verbose:
        print("Starting MLX Knife embed-serve...")
        print(f"Model: {model}")
        print(f"Server will bind to: http://{host}:{port}")
        print(f"Device: {'cpu' if cpu else 'gpu'}")

    if supervise:
        # Delegate to the shared subprocess supervisor (serve.py), targeting the embed app.
        exit_code = _run_supervised_uvicorn(
            host=host,
            port=port,
            log_level=log_level,
            module="mlxk2.core.embed_server_base",
            extra_env=extra_env,
        )
        if exit_code != 0:
            sys.exit(exit_code)
        return

    # Non-supervised fallback: run uvicorn in-process (development).
    for key, value in extra_env.items():
        os.environ[key] = value
    try:
        import uvicorn  # type: ignore
    except Exception as e:  # noqa: BLE001
        raise RuntimeError(
            "uvicorn is required to run embed-serve; install with 'pip install fastapi uvicorn'."
        ) from e
    from ..logging import set_log_level, uvicorn_log_config

    set_log_level(log_level)
    uvicorn.run(
        "mlxk2.core.embed_server_base:app",
        host=host,
        port=port,
        log_level=log_level,
        log_config=uvicorn_log_config(log_level),
        lifespan="on",
    )
