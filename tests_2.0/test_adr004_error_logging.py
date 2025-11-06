"""
Tests for ADR-004: Enhanced Error Handling & Logging.

Covers:
- Error envelope structure and serialization
- Error type to HTTP status mapping
- Request ID propagation
- Log redaction (HF_TOKEN, paths)
- Structured logging (plain and JSON modes)
"""

import json
import os
import re
from pathlib import Path

import pytest

from mlxk2.errors import (
    ErrorType,
    MLXKError,
    error_envelope,
    success_envelope,
    model_not_found_error,
    validation_error,
    server_shutdown_error,
    internal_error,
    access_denied_error,
    ERROR_TYPE_TO_HTTP_STATUS,
)
from mlxk2.logging import MLXKLogger
from mlxk2.context import (
    generate_request_id,
    set_request_id,
    get_request_id,
    clear_request_id,
    RequestContext,
)


# ============================================================================
# Error Envelope Tests
# ============================================================================

def test_mlxk_error_structure():
    """Error should serialize to correct structure."""
    error = MLXKError(
        type=ErrorType.MODEL_NOT_FOUND,
        message="Model not found",
        detail="Additional details",
        retryable=False
    )

    error_dict = error.to_dict()
    assert error_dict["type"] == "model_not_found"
    assert error_dict["message"] == "Model not found"
    assert error_dict["detail"] == "Additional details"
    assert error_dict["retryable"] is False


def test_mlxk_error_minimal():
    """Error should work with minimal fields."""
    error = MLXKError(
        type=ErrorType.INTERNAL_ERROR,
        message="Something went wrong"
    )

    error_dict = error.to_dict()
    assert error_dict["type"] == "internal_error"
    assert error_dict["message"] == "Something went wrong"
    assert "detail" not in error_dict
    assert "retryable" not in error_dict


def test_error_envelope_structure():
    """Error envelope should have correct structure."""
    error = MLXKError(
        type=ErrorType.VALIDATION_ERROR,
        message="Invalid request"
    )

    envelope = error_envelope(error, request_id="test-request-id")

    assert envelope["status"] == "error"
    assert envelope["error"]["type"] == "validation_error"
    assert envelope["error"]["message"] == "Invalid request"
    assert envelope["request_id"] == "test-request-id"


def test_success_envelope_structure():
    """Success envelope should have correct structure."""
    data = {"models": ["model1", "model2"]}
    envelope = success_envelope(data, request_id="test-request-id")

    assert envelope["status"] == "success"
    assert envelope["data"] == data
    assert envelope["request_id"] == "test-request-id"


def test_error_envelope_with_data():
    """Error envelope can include additional data field."""
    error = MLXKError(
        type=ErrorType.AMBIGUOUS_MATCH,
        message="Multiple matches found"
    )
    data = {"candidates": ["model1", "model2"]}

    envelope = error_envelope(error, request_id="req-123", data=data)

    assert envelope["status"] == "error"
    assert envelope["data"] == data
    assert envelope["request_id"] == "req-123"


# ============================================================================
# HTTP Status Mapping Tests (ADR-004 Specification)
# ============================================================================

def test_error_type_to_http_status_mapping():
    """All error types should map to correct HTTP status codes."""
    expected_mappings = {
        ErrorType.ACCESS_DENIED: 403,
        ErrorType.MODEL_NOT_FOUND: 404,
        ErrorType.AMBIGUOUS_MATCH: 400,
        ErrorType.DOWNLOAD_FAILED: 503,
        ErrorType.VALIDATION_ERROR: 400,
        ErrorType.PUSH_OPERATION_FAILED: 500,
        ErrorType.SERVER_SHUTDOWN: 503,
        ErrorType.INTERNAL_ERROR: 500,
    }

    for error_type, expected_status in expected_mappings.items():
        error = MLXKError(type=error_type, message="test")
        assert error.to_http_status() == expected_status
        assert ERROR_TYPE_TO_HTTP_STATUS[error_type] == expected_status


def test_common_error_constructors():
    """Common error constructors should create correct error types."""
    # model_not_found
    error = model_not_found_error("test-model")
    assert error.type == ErrorType.MODEL_NOT_FOUND
    assert "test-model" in error.message
    assert error.retryable is False

    # validation_error
    error = validation_error("Invalid input")
    assert error.type == ErrorType.VALIDATION_ERROR
    assert error.message == "Invalid input"
    assert error.retryable is False

    # server_shutdown
    error = server_shutdown_error()
    assert error.type == ErrorType.SERVER_SHUTDOWN
    assert error.retryable is True

    # internal_error
    error = internal_error("Unexpected error", detail={"stack": "..."})
    assert error.type == ErrorType.INTERNAL_ERROR
    assert error.detail == {"stack": "..."}
    assert error.retryable is None  # Unknown

    # access_denied
    error = access_denied_error("No permission")
    assert error.type == ErrorType.ACCESS_DENIED
    assert error.retryable is False


# ============================================================================
# Request ID Tests
# ============================================================================

def test_generate_request_id():
    """generate_request_id should return valid UUID4."""
    request_id = generate_request_id()
    # UUID4 format: 8-4-4-4-12 hex characters
    uuid_pattern = re.compile(
        r'^[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$',
        re.IGNORECASE
    )
    assert uuid_pattern.match(request_id), f"Invalid UUID4: {request_id}"


def test_request_id_context_propagation():
    """Request ID should propagate through context."""
    # Initially None
    assert get_request_id() is None

    # Set and retrieve
    set_request_id("test-id-123")
    assert get_request_id() == "test-id-123"

    # Clear
    clear_request_id()
    assert get_request_id() is None


def test_request_context_manager():
    """RequestContext should manage request_id lifecycle."""
    # Initially None
    assert get_request_id() is None

    with RequestContext() as request_id:
        # Inside context, request_id is set
        assert request_id is not None
        assert get_request_id() == request_id

    # After context, request_id is cleared
    assert get_request_id() is None


def test_request_context_nesting():
    """RequestContext should handle nesting correctly."""
    with RequestContext() as outer_id:
        assert get_request_id() == outer_id

        with RequestContext() as inner_id:
            assert get_request_id() == inner_id
            assert inner_id != outer_id

        # After inner context, outer_id is restored
        assert get_request_id() == outer_id

    # After outer context, None is restored
    assert get_request_id() is None


def test_request_context_with_explicit_id():
    """RequestContext should accept explicit request_id."""
    explicit_id = "my-custom-id"

    with RequestContext(request_id=explicit_id) as request_id:
        assert request_id == explicit_id
        assert get_request_id() == explicit_id


# ============================================================================
# Log Redaction Tests (ADR-004 Security Requirement)
# ============================================================================

def test_logger_redacts_hf_token():
    """Logger should redact HF tokens from messages."""
    logger = MLXKLogger("test")

    # Test token redaction
    message = "Using token hf_AbCdEfGhIjKlMnOpQrStUvWxYz123456 for auth"
    redacted = logger._redact(message)

    assert "hf_AbCdEfGhIjKlMnOpQrStUvWxYz123456" not in redacted
    assert "[REDACTED_TOKEN]" in redacted


def test_logger_redacts_home_directory():
    """Logger should redact user home directory paths."""
    logger = MLXKLogger("test")

    home_dir = str(Path.home())
    message = f"Loading model from {home_dir}/models/test"
    redacted = logger._redact(message)

    # Should replace home directory with ~
    assert home_dir not in redacted
    assert "~/models/test" in redacted


def test_logger_redacts_multiple_tokens():
    """Logger should redact multiple tokens in same message."""
    logger = MLXKLogger("test")

    message = "Token1: hf_TokenAAAAAAAAAAAAAAAAAAAAAAAAAAAA, Token2: hf_TokenBBBBBBBBBBBBBBBBBBBBBBBBBBBB"
    redacted = logger._redact(message)

    assert "hf_TokenAAAAAAAAAAAAAAAAAAAAAAAAAAAA" not in redacted
    assert "hf_TokenBBBBBBBBBBBBBBBBBBBBBBBBBBBB" not in redacted
    assert redacted.count("[REDACTED_TOKEN]") == 2


# ============================================================================
# Structured Logging Tests
# ============================================================================

def test_logger_plain_text_mode(capsys):
    """Logger should output plain text by default."""
    # Ensure JSON mode is off
    os.environ.pop("MLXK2_LOG_JSON", None)

    logger = MLXKLogger("test-plain")
    logger.info("Test message")

    captured = capsys.readouterr()
    assert "Test message" in captured.err
    # Should NOT be JSON
    assert not captured.err.strip().startswith("{")


def test_logger_json_mode(capsys):
    """Logger should output JSON when MLXK2_LOG_JSON=1."""
    # Enable JSON mode
    os.environ["MLXK2_LOG_JSON"] = "1"

    try:
        logger = MLXKLogger("test-json")
        logger.info("Test message", request_id="req-123", model="test-model")

        captured = capsys.readouterr()
        log_line = captured.err.strip()

        # Should be valid JSON
        log_entry = json.loads(log_line)
        assert log_entry["msg"] == "Test message"
        assert log_entry["level"] == "INFO"
        assert log_entry["request_id"] == "req-123"
        assert log_entry["model"] == "test-model"
        assert "ts" in log_entry  # Timestamp should be present

    finally:
        # Cleanup
        os.environ.pop("MLXK2_LOG_JSON", None)


def test_logger_levels(capsys):
    """Logger should support different log levels."""
    logger = MLXKLogger("test-levels")

    logger.info("Info message")
    logger.warning("Warning message")
    logger.error("Error message", error_key="test_error")

    captured = capsys.readouterr()
    assert "Info message" in captured.err
    assert "WARN" in captured.err
    assert "Warning message" in captured.err
    assert "ERROR" in captured.err
    assert "Error message" in captured.err


def test_logger_debug_requires_verbose(capsys):
    """DEBUG logs should only appear when verbose=True."""
    logger = MLXKLogger("test-debug")

    # Verbose off (default)
    logger.debug("Debug message 1")
    captured = capsys.readouterr()
    assert "Debug message 1" not in captured.err

    # Verbose on
    logger.set_verbose(True)
    logger.debug("Debug message 2")
    captured = capsys.readouterr()
    assert "Debug message 2" in captured.err


def test_logger_error_rate_limiting(capsys):
    """Logger should rate-limit duplicate errors (max 1/5s)."""
    logger = MLXKLogger("test-ratelimit")

    # First error should be logged
    logger.error("Repeated error", error_key="duplicate_error")
    captured = capsys.readouterr()
    assert "Repeated error" in captured.err

    # Immediate duplicate should be suppressed
    logger.error("Repeated error", error_key="duplicate_error")
    captured = capsys.readouterr()
    assert captured.err == ""  # Suppressed

    # Different error key should be logged
    logger.error("Different error", error_key="different_error")
    captured = capsys.readouterr()
    assert "Different error" in captured.err


# ============================================================================
# Integration Tests (Error Envelope + Request ID + Logging)
# ============================================================================

def test_error_envelope_includes_request_id():
    """Error envelope should include request_id when available."""
    with RequestContext() as request_id:
        error = model_not_found_error("test-model")
        envelope = error_envelope(error, request_id=get_request_id())

        assert envelope["request_id"] == request_id


def test_logger_uses_request_id(capsys):
    """Logger should include request_id in JSON logs."""
    os.environ["MLXK2_LOG_JSON"] = "1"

    try:
        logger = MLXKLogger("test-request-id")

        with RequestContext() as request_id:
            logger.info("Test message", request_id=request_id)

        captured = capsys.readouterr()
        log_entry = json.loads(captured.err.strip())

        assert log_entry["request_id"] == request_id

    finally:
        os.environ.pop("MLXK2_LOG_JSON", None)
