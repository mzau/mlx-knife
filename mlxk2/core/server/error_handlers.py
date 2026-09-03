"""Shared FastAPI exception handlers (ADR-004).

Factored out of ``server_base.py`` so both the main serve app and the embed-serve app
(``core/embed_server_base.py``) register identical error-envelope handlers without the embed
process importing serve's heavy module-level symbols (MLXRunner, ModelManager, …) just for
error formatting. Pure: depends only on FastAPI + ``mlxk2.errors``.
"""

from __future__ import annotations

from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from ...errors import ErrorType, MLXKError, error_envelope
from ..runner.token_limits import ContextLengthExceeded

# HTTP status -> ADR-004 error type (mirrors errors.ERROR_TYPE_TO_HTTP_STATUS in reverse).
_STATUS_TO_ERROR_TYPE = {
    400: ErrorType.VALIDATION_ERROR,
    403: ErrorType.ACCESS_DENIED,
    404: ErrorType.MODEL_NOT_FOUND,
    413: ErrorType.PAYLOAD_TOO_LARGE,  # Audio upload above the size limit (both audio routes)
    422: ErrorType.CAPABILITY_NOT_SUPPORTED,  # /v1/audio/translations against a non-translating model
    500: ErrorType.INTERNAL_ERROR,
    501: ErrorType.NOT_IMPLEMENTED,
    502: ErrorType.BAD_GATEWAY,  # ADR-015 D2: embed backend unreachable
    503: ErrorType.SERVER_SHUTDOWN,
    504: ErrorType.GATEWAY_TIMEOUT,  # ADR-015 D2: embed backend read-timeout
    507: ErrorType.INSUFFICIENT_MEMORY,
}


async def http_exception_handler(request: Request, exc: HTTPException):
    """Convert HTTPException to an ADR-004 error envelope."""
    request_id = getattr(request.state, "request_id", None)
    error_type = _STATUS_TO_ERROR_TYPE.get(exc.status_code, ErrorType.INTERNAL_ERROR)
    error = MLXKError(
        type=error_type,
        message=exc.detail,
        # 503 (shutting down) + 502/504 (ADR-015 D2 embed-backend transient gateway errors)
        # are all retryable: the client may re-issue the request once capacity returns.
        retryable=(exc.status_code in (502, 503, 504)),
    )
    envelope = error_envelope(error, request_id=request_id)
    return JSONResponse(status_code=exc.status_code, content=envelope)


async def validation_exception_handler(request: Request, exc: RequestValidationError):
    """Convert FastAPI validation errors (422) to an ADR-004 envelope (400).

    FastAPI returns 422 Unprocessable Entity for validation errors by default; we convert to
    400 Bad Request with the ADR-004 envelope for API consistency. A 422 raised by hand stays
    422 and goes through the handler above — a malformed request and a model that cannot serve
    a well-formed one are different answers.
    """
    request_id = getattr(request.state, "request_id", None)
    errors = exc.errors()
    detail = "; ".join(
        f"{'.'.join(str(loc) for loc in e['loc'])}: {e['msg']}" for e in errors
    )
    error = MLXKError(
        type=ErrorType.VALIDATION_ERROR,
        message="Request validation failed",
        detail=detail,
        retryable=False,
    )
    envelope = error_envelope(error, request_id=request_id)
    return JSONResponse(status_code=400, content=envelope)


async def context_length_exceeded_handler(request: Request, exc: ContextLengthExceeded):
    """The prompt fills the context window (#66): 400 with the two numbers a client needs.

    Raised by the runner before any token is produced, on both the batch path and the
    streaming pre-check, so it always arrives as a status, never inside an SSE stream.
    """
    request_id = getattr(request.state, "request_id", None)
    error = MLXKError(
        type=ErrorType.CONTEXT_LENGTH_EXCEEDED,
        message=str(exc),
        detail={"prompt_tokens": exc.prompt_tokens, "context_length": exc.context_length},
        retryable=False,
    )
    envelope = error_envelope(error, request_id=request_id)
    return JSONResponse(status_code=error.to_http_status(), content=envelope)


def register_error_handlers(app: FastAPI) -> None:
    """Register the ADR-004 exception handlers on a FastAPI app."""
    app.add_exception_handler(HTTPException, http_exception_handler)
    app.add_exception_handler(RequestValidationError, validation_exception_handler)
    app.add_exception_handler(ContextLengthExceeded, context_length_exceeded_handler)
