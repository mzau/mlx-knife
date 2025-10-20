"""
Request context management for MLX Knife 2.0 (ADR-004).

Provides request_id (UUID4) generation and propagation across requests.
"""

import uuid
from contextvars import ContextVar
from typing import Optional


# Context variable for request_id (thread-safe for async)
_request_id_context: ContextVar[Optional[str]] = ContextVar('request_id', default=None)


def generate_request_id() -> str:
    """Generate a new request ID (UUID4).

    Returns:
        String UUID4 (e.g., "550e8400-e29b-41d4-a716-446655440000")
    """
    return str(uuid.uuid4())


def set_request_id(request_id: str) -> None:
    """Set the current request ID in context.

    Args:
        request_id: UUID string to set as current request ID
    """
    _request_id_context.set(request_id)


def get_request_id() -> Optional[str]:
    """Get the current request ID from context.

    Returns:
        Current request ID, or None if not set
    """
    return _request_id_context.get()


def clear_request_id() -> None:
    """Clear the current request ID from context."""
    _request_id_context.set(None)


class RequestContext:
    """Context manager for request_id lifecycle.

    Usage:
        with RequestContext() as request_id:
            # request_id is available via get_request_id()
            do_work()
        # request_id is cleared on exit
    """

    def __init__(self, request_id: Optional[str] = None):
        """Initialize context manager.

        Args:
            request_id: Optional existing request ID, or generate new one
        """
        self.request_id = request_id or generate_request_id()
        self._previous_id: Optional[str] = None

    def __enter__(self) -> str:
        """Enter context and set request_id."""
        self._previous_id = get_request_id()
        set_request_id(self.request_id)
        return self.request_id

    def __exit__(self, _exc_type, _exc_val, _exc_tb):
        """Exit context and restore previous request_id."""
        set_request_id(self._previous_id)
        return False  # Don't suppress exceptions
