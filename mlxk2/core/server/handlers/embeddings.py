"""Embeddings handler for ``POST /v1/embeddings`` (ADR-015 Slice D1).

Transport-agnostic: drives an already-loaded :class:`EmbeddingRunner` and returns the
OpenAI-compatible response dict. The handler + runner live **only** in the ``embed-serve``
process (``core/embed_server_base.py``); the main ``serve`` never imports this module — its
``/v1/embeddings`` is a thin proxy with no runner in its address space (ADR-015 §Server Topology).

Mirrors the dependency-injection style of ``handlers/audio.py`` (receives the runner + helpers,
raises :class:`fastapi.HTTPException` for client errors).
"""

from __future__ import annotations

import base64
import sys
from array import array
from threading import Lock
from typing import Any, Callable, Dict, List, Optional

from fastapi import HTTPException


def _get_logger():
    """Lazy import logger to avoid circular dependencies."""
    from ....logging import get_logger
    return get_logger()


def _to_wire(vec: List[float], encoding_format: str):
    """Encode one vector for the wire.

    ``float`` -> the raw list (the runner already returns ``List[float]``).
    ``base64`` -> base64 of little-endian float32 bytes (``<f4``) — exactly what the OpenAI
    Python SDK decodes when it requests ``encoding_format="base64"`` (its default). Uses the
    stdlib ``array`` (no numpy dependency on the request path).
    """
    if encoding_format == "base64":
        buf = array("f", vec)  # C float == float32
        if sys.byteorder != "little":
            buf.byteswap()
        return base64.b64encode(buf.tobytes()).decode("ascii")
    return vec


def _count_tokens(
    inputs: List[str], runner: Any, count_tokens_fn: Optional[Callable[[str], int]]
) -> int:
    """Best-effort prompt-token count for ``usage`` (ADR-015: precision not contractual)."""
    if count_tokens_fn is not None:
        try:
            return sum(int(count_tokens_fn(t)) for t in inputs)
        except Exception:  # noqa: BLE001 — usage is best-effort, never fail the request
            pass
    tok = getattr(runner, "tokenizer", None)
    if tok is not None and hasattr(tok, "encode"):
        try:
            return sum(len(tok.encode(t)) for t in inputs)
        except Exception:  # noqa: BLE001
            pass
    return sum(len(t.split()) for t in inputs)


def handle_embeddings(
    *,
    runner: Any,
    model_identity: str,
    inputs: List[str],
    encoding_format: str = "base64",
    dimensions: Optional[int] = None,
    input_type: str = "document",
    instruct: Optional[str] = None,
    lock: Optional[Lock] = None,
    count_tokens_fn: Optional[Callable[[str], int]] = None,
) -> Dict[str, Any]:
    """Embed ``inputs`` with the loaded runner; return an OpenAI ``/v1/embeddings`` response dict.

    Raises HTTPException(400) for empty/invalid input, an unsupported ``encoding_format`` or
    ``input_type``, or a ``dimensions`` value that doesn't match the model's native width
    (no silent Matryoshka truncation in D1). HTTPException(500) on inference failure.
    """
    logger = _get_logger()

    # -- validate ------------------------------------------------------------
    if not inputs:
        raise HTTPException(
            status_code=400,
            detail="'input' must be a non-empty string or array of strings",
        )
    if any(not isinstance(t, str) for t in inputs):
        raise HTTPException(status_code=400, detail="'input' must contain only strings")
    if any(t == "" for t in inputs):
        raise HTTPException(status_code=400, detail="'input' must not contain empty strings")
    if encoding_format not in ("float", "base64"):
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported encoding_format '{encoding_format}' (use 'float' or 'base64')",
        )
    if input_type not in ("document", "query"):
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported input_type '{input_type}' (use 'document' or 'query')",
        )

    # Reject a non-native `dimensions` request before doing any work, when the width is known.
    native_dim = getattr(runner, "dimensions", None)
    if dimensions is not None and native_dim is not None and dimensions != native_dim:
        raise HTTPException(
            status_code=400,
            detail=(
                f"This model emits {native_dim}-dimensional vectors; the 'dimensions' parameter "
                f"(requested {dimensions}) is not supported (no Matryoshka truncation)."
            ),
        )

    # -- run (serialized) -----------------------------------------------------
    # The lock must cover BOTH embed() AND the usage token count: each touches the shared
    # tokenizer, and the HuggingFace fast (Rust) tokenizer is not thread-safe — concurrent access
    # raises "Already borrowed" -> HTTP 500. The endpoint is a sync def (FastAPI runs it in a
    # threadpool), so this lock is what actually serializes the single model across requests.
    def _embed_and_count():
        vectors = runner.embed(inputs, input_type=input_type, instruct=instruct)
        return vectors, _count_tokens(inputs, runner, count_tokens_fn)

    try:
        if lock is not None:
            with lock:
                vectors, n_tokens = _embed_and_count()
        else:
            vectors, n_tokens = _embed_and_count()
    except HTTPException:
        raise
    except Exception as e:  # noqa: BLE001 — surface any inference failure as a 500 envelope
        logger.error(f"Embedding failed: {e}", model=model_identity)
        raise HTTPException(status_code=500, detail=f"Embedding failed: {e}")

    # -- shape OpenAI response (pure: vectors only — no tokenizer/model access) ----------------
    data = [
        {"object": "embedding", "index": i, "embedding": _to_wire(vec, encoding_format)}
        for i, vec in enumerate(vectors)
    ]
    return {
        "object": "list",
        "data": data,
        "model": model_identity,
        "usage": {"prompt_tokens": n_tokens, "total_tokens": n_tokens},
    }
