"""
Unified error handling for MLX Knife 2.0 (ADR-004).

Provides standardized error envelope, error type taxonomy, and HTTP status mapping.
"""

from enum import Enum
from typing import Any, Dict, Optional
from dataclasses import dataclass


class ErrorType(str, Enum):
    """Standardized error types (ADR-004 taxonomy)."""
    ACCESS_DENIED = "access_denied"
    MODEL_NOT_FOUND = "model_not_found"
    AMBIGUOUS_MATCH = "ambiguous_match"
    DOWNLOAD_FAILED = "download_failed"
    VALIDATION_ERROR = "validation_error"
    PUSH_OPERATION_FAILED = "push_operation_failed"
    SERVER_SHUTDOWN = "server_shutdown"
    INSUFFICIENT_MEMORY = "insufficient_memory"  # ADR-016: Model exceeds memory threshold
    INTERNAL_ERROR = "internal_error"


# HTTP status code mapping (ADR-004 specification)
ERROR_TYPE_TO_HTTP_STATUS: Dict[ErrorType, int] = {
    ErrorType.ACCESS_DENIED: 403,
    ErrorType.MODEL_NOT_FOUND: 404,
    ErrorType.AMBIGUOUS_MATCH: 400,
    ErrorType.DOWNLOAD_FAILED: 503,
    ErrorType.VALIDATION_ERROR: 400,
    ErrorType.PUSH_OPERATION_FAILED: 500,
    ErrorType.SERVER_SHUTDOWN: 503,
    ErrorType.INSUFFICIENT_MEMORY: 507,  # ADR-016: HTTP 507 Insufficient Storage
    ErrorType.INTERNAL_ERROR: 500,
}


@dataclass
class MLXKError:
    """Structured error information (ADR-004 error envelope).

    Attributes:
        type: Error type from ErrorType enum
        message: Human-readable error message
        detail: Optional additional error details (dict or string)
        retryable: Whether the operation can be retried (None = unknown)
    """
    type: ErrorType
    message: str
    detail: Optional[Any] = None
    retryable: Optional[bool] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        result = {
            "type": self.type.value,
            "message": self.message,
        }
        if self.detail is not None:
            result["detail"] = self.detail
        if self.retryable is not None:
            result["retryable"] = self.retryable
        return result

    def to_http_status(self) -> int:
        """Get HTTP status code for this error type."""
        return ERROR_TYPE_TO_HTTP_STATUS.get(self.type, 500)


def error_envelope(
    error: MLXKError,
    request_id: Optional[str] = None,
    data: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """Create a complete error envelope (ADR-004 specification).

    Args:
        error: MLXKError instance
        request_id: Optional request correlation ID (UUID)
        data: Optional additional response data

    Returns:
        Error envelope dict: {"status": "error", "error": {...}, "request_id": "...", "data": {...}}
    """
    envelope = {
        "status": "error",
        "error": error.to_dict()
    }
    if request_id:
        envelope["request_id"] = request_id
    if data:
        envelope["data"] = data
    return envelope


def success_envelope(
    data: Dict[str, Any],
    request_id: Optional[str] = None
) -> Dict[str, Any]:
    """Create a success response envelope (ADR-004 specification).

    Args:
        data: Response data
        request_id: Optional request correlation ID (UUID)

    Returns:
        Success envelope dict: {"status": "success", "data": {...}, "request_id": "..."}
    """
    envelope = {
        "status": "success",
        "data": data
    }
    if request_id:
        envelope["request_id"] = request_id
    return envelope


# Common error constructors for convenience
def model_not_found_error(model_name: str, detail: Optional[str] = None) -> MLXKError:
    """Create a model_not_found error."""
    return MLXKError(
        type=ErrorType.MODEL_NOT_FOUND,
        message=f"Model '{model_name}' not found or failed to load",
        detail=detail,
        retryable=False
    )


def validation_error(message: str, detail: Optional[Any] = None) -> MLXKError:
    """Create a validation_error."""
    return MLXKError(
        type=ErrorType.VALIDATION_ERROR,
        message=message,
        detail=detail,
        retryable=False
    )


def server_shutdown_error(message: str = "Server is shutting down") -> MLXKError:
    """Create a server_shutdown error."""
    return MLXKError(
        type=ErrorType.SERVER_SHUTDOWN,
        message=message,
        detail=None,
        retryable=True
    )


def internal_error(message: str, detail: Optional[Any] = None) -> MLXKError:
    """Create an internal_error."""
    return MLXKError(
        type=ErrorType.INTERNAL_ERROR,
        message=message,
        detail=detail,
        retryable=None  # Unknown if retryable
    )


def access_denied_error(message: str, detail: Optional[str] = None) -> MLXKError:
    """Create an access_denied error."""
    return MLXKError(
        type=ErrorType.ACCESS_DENIED,
        message=message,
        detail=detail,
        retryable=False
    )
