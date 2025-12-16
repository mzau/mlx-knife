"""
Structured logging for MLX Knife 2.0 (ADR-004).

Provides level-based logging with optional JSON output and sensitive data redaction.
"""

import json
import logging
import os
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, Optional
from collections import defaultdict


# Redaction patterns (ADR-004 specification)
TOKEN_PATTERN = re.compile(r'(hf_[a-zA-Z0-9]{30,})', re.IGNORECASE)
# Redact user-specific paths (home directories)
HOME_DIR = str(Path.home())


class MLXKLogger:
    """Structured logger with JSON support and redaction (ADR-004).

    Features:
    - Level-based logging (INFO, WARN, ERROR, DEBUG)
    - Optional JSON output via MLXK2_LOG_JSON=1
    - Automatic redaction of HF_TOKEN and user paths
    - Request correlation via request_id
    - Error flood rate limiting
    """

    def __init__(self, name: str = "mlxk2"):
        self.name = name
        self.json_mode = os.environ.get("MLXK2_LOG_JSON", "0") == "1"
        self.verbose = False  # Set by CLI --verbose flag

        # Rate limiting for duplicate errors (ADR-004: max 1/5s)
        self._error_counts: Dict[str, int] = defaultdict(int)
        self._error_last_time: Dict[str, float] = {}
        self._rate_limit_window = 5.0  # seconds

        # Setup Python logging backend
        self.logger = logging.getLogger(name)
        self.logger.setLevel(logging.DEBUG)  # Capture all, filter at handler level

        # Clear existing handlers
        self.logger.handlers.clear()

        # Add handler
        handler = logging.StreamHandler(sys.stderr)
        handler.setLevel(logging.DEBUG)

        if not self.json_mode:
            # Plain text format
            formatter = logging.Formatter('%(message)s')
        else:
            # JSON formatter handles structured output
            formatter = logging.Formatter('%(message)s')

        handler.setFormatter(formatter)
        self.logger.addHandler(handler)

        # Don't propagate to root logger
        self.logger.propagate = False

    def _redact(self, message: str) -> str:
        """Redact sensitive data from message (ADR-004).

        Redacts:
        - HF tokens (hf_...)
        - User home directory paths
        """
        # Redact HF tokens
        message = TOKEN_PATTERN.sub('[REDACTED_TOKEN]', message)

        # Redact home directory
        if HOME_DIR and HOME_DIR in message:
            message = message.replace(HOME_DIR, '~')

        return message

    def _should_log_error(self, error_key: str) -> bool:
        """Check if error should be logged (rate limiting).

        Rate limit: max 1 occurrence per 5 seconds for same error.
        """
        now = time.time()
        last_time = self._error_last_time.get(error_key, 0)

        if now - last_time >= self._rate_limit_window:
            # Reset counter
            self._error_counts[error_key] = 1
            self._error_last_time[error_key] = now
            return True
        else:
            # Increment suppressed count
            self._error_counts[error_key] += 1
            return False

    def _format_log(
        self,
        level: str,
        message: str,
        request_id: Optional[str] = None,
        **extra: Any
    ) -> str:
        """Format log message (plain or JSON)."""
        # Redact sensitive data
        message = self._redact(message)

        if self.json_mode:
            log_entry = {
                "ts": time.time(),
                "level": level,
                "msg": message,
            }
            if request_id:
                log_entry["request_id"] = request_id

            # Add extra fields (route, model, duration_ms, etc.)
            for key, value in extra.items():
                if value is not None:
                    log_entry[key] = value

            return json.dumps(log_entry)
        else:
            # Plain text format (consistent with Uvicorn style)
            # INFO messages get prefix for consistency with Uvicorn logs
            prefix = f"[{level}]"
            return f"{prefix} {message}"

    def info(self, message: str, request_id: Optional[str] = None, **extra: Any):
        """Log INFO level message."""
        formatted = self._format_log("INFO", message, request_id, **extra)
        self.logger.info(formatted)

    def warning(self, message: str, request_id: Optional[str] = None, **extra: Any):
        """Log WARN level message."""
        formatted = self._format_log("WARN", message, request_id, **extra)
        self.logger.warning(formatted)

    def error(
        self,
        message: str,
        request_id: Optional[str] = None,
        error_key: Optional[str] = None,
        **extra: Any
    ):
        """Log ERROR level message with rate limiting.

        Args:
            message: Error message
            request_id: Request correlation ID
            error_key: Key for rate limiting (default: message hash)
            **extra: Additional structured fields
        """
        # Rate limiting
        key = error_key or message
        if not self._should_log_error(key):
            return  # Suppressed

        # Add suppressed count if > 1
        count = self._error_counts.get(key, 1)
        if count > 1:
            extra["suppressed_count"] = count - 1

        formatted = self._format_log("ERROR", message, request_id, **extra)
        self.logger.error(formatted)

    def debug(self, message: str, request_id: Optional[str] = None, **extra: Any):
        """Log DEBUG level message (only if --verbose)."""
        if not self.verbose:
            return
        formatted = self._format_log("DEBUG", message, request_id, **extra)
        self.logger.debug(formatted)

    def set_verbose(self, verbose: bool):
        """Enable/disable verbose (DEBUG) logging."""
        self.verbose = verbose


# Custom JSON formatter for root logger (external libraries)
class JSONFormatter(logging.Formatter):
    """JSON formatter for root logger (captures mlx-lm, transformers, etc.)."""

    def __init__(self):
        super().__init__()
        self.json_mode = os.environ.get("MLXK2_LOG_JSON", "0") == "1"

    def format(self, record: logging.LogRecord) -> str:
        if not self.json_mode:
            # Plain text fallback
            return super().format(record)

        # Redact sensitive data
        message = record.getMessage()
        message = TOKEN_PATTERN.sub('[REDACTED_TOKEN]', message)
        if HOME_DIR and HOME_DIR in message:
            message = message.replace(HOME_DIR, '~')

        log_entry = {
            "ts": time.time(),
            "level": record.levelname,
            "msg": message,
            "logger": record.name,
        }

        return json.dumps(log_entry)


# Global logger instance
_logger: Optional[MLXKLogger] = None
_root_logger_configured = False


def get_logger() -> MLXKLogger:
    """Get global MLXKLogger instance."""
    global _logger
    if _logger is None:
        _logger = MLXKLogger()
        _configure_root_logger()
    return _logger


def _configure_root_logger():
    """Configure root logger for both JSON and plain-text modes.

    This captures logs from external libraries (mlx-lm, transformers, etc.)
    and ensures consistent output with MLXKLogger formatting.
    """
    global _root_logger_configured
    if _root_logger_configured:
        return

    json_mode = os.environ.get("MLXK2_LOG_JSON", "0") == "1"

    # Configure root logger for both JSON and plain-text modes
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.WARNING)  # Capture WARNING and above from external libs

    # Clear existing handlers to avoid duplicates
    root_logger.handlers.clear()

    # Add handler with appropriate formatter
    handler = logging.StreamHandler(sys.stderr)
    if json_mode:
        handler.setFormatter(JSONFormatter())
    else:
        # Plain text: use [LEVEL] prefix for consistency with MLXKLogger
        formatter = logging.Formatter('[%(levelname)s] %(name)s: %(message)s')
        handler.setFormatter(formatter)
    root_logger.addHandler(handler)

    _root_logger_configured = True


def set_verbose(verbose: bool):
    """Set verbose mode globally."""
    get_logger().set_verbose(verbose)


def set_log_level(level: str):
    """Set log level globally for MLXKLogger and root logger.

    Args:
        level: Log level string (debug, info, warning, error)
    """
    level_upper = level.upper()
    log_level = getattr(logging, level_upper, logging.INFO)

    # Set MLXKLogger level
    logger = get_logger()
    logger.logger.setLevel(log_level)
    for handler in logger.logger.handlers:
        handler.setLevel(log_level)

    # Set root logger level (for external libraries)
    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)
    for handler in root_logger.handlers:
        handler.setLevel(log_level)
