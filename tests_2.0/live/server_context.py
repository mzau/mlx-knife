"""LocalServer context manager for E2E testing (ADR-011).

Provides a clean subprocess-based server lifecycle for testing:
- Starts server with pre-loaded model
- Waits for health check before yielding
- Ensures graceful cleanup on exit
"""

from __future__ import annotations

import gc
import sys
import time
import subprocess
from contextlib import contextmanager
from typing import Optional

try:
    import httpx
except ImportError:
    httpx = None  # Will fail at test time with clear error

# Optional: RAM monitoring for debugging (requires psutil)
# Uncomment to enable RAM logging during test runs
# try:
#     import psutil
#     def log_ram_status(stage: str) -> None:
#         """Log current RAM status (non-blocking)."""
#         mem = psutil.virtual_memory()
#         free_gb = mem.available / (1024**3)
#         total_gb = mem.total / (1024**3)
#         print(f"[RAM-{stage}] Free: {free_gb:.1f}GB / {total_gb:.1f}GB ({mem.percent:.1f}% used)")
# except ImportError:
#     def log_ram_status(stage: str) -> None:
#         pass  # psutil not installed, skip logging


@contextmanager
def LocalServer(
    model: str,
    port: int = 8765,
    timeout: int = 60,
    log_level: str = "warning"
):
    """Start a local mlx-knife server for E2E testing.

    Context manager that:
    1. Launches server subprocess with pre-loaded model
    2. Waits for /health endpoint to respond (up to timeout)
    3. Yields server URL for testing
    4. Ensures graceful shutdown (SIGTERM → SIGKILL fallback)

    Args:
        model: Model ID to pre-load (e.g., "mlx-community/Llama-3.2-3B-Instruct-4bit")
        port: Server port (default 8765, non-standard to avoid conflicts)
        timeout: Startup timeout in seconds (default 60s for model loading)
        log_level: Server log level (default "warning" to reduce noise)

    Yields:
        server_url: str like "http://127.0.0.1:8765"

    Raises:
        TimeoutError: If server fails to start within timeout
        RuntimeError: If httpx not installed

    Example:
        >>> with LocalServer("mlx-community/Llama-3.2-3B-Instruct-4bit") as url:
        ...     response = httpx.post(f"{url}/v1/chat/completions", json={...})
        ...     assert response.status_code == 200
    """
    if httpx is None:
        raise RuntimeError("httpx required for E2E tests (pip install httpx)")

    # Start server subprocess
    proc = subprocess.Popen(
        [
            sys.executable, "-m", "mlxk2.cli",
            "serve",
            "--model", model,
            "--port", str(port),
            "--host", "127.0.0.1",
            "--log-level", log_level
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )

    server_url = f"http://127.0.0.1:{port}"

    # Wait for server health check
    start_time = time.time()
    last_error: Optional[Exception] = None

    while time.time() - start_time < timeout:
        try:
            response = httpx.get(f"{server_url}/health", timeout=2.0)
            if response.status_code == 200:
                # Server ready
                break
        except Exception as e:
            last_error = e
            time.sleep(0.5)
    else:
        # Timeout: kill server and report error
        proc.kill()
        stdout, stderr = proc.communicate()

        error_msg = (
            f"Server failed to start within {timeout}s\n"
            f"Last error: {last_error}\n"
            f"--- STDOUT ---\n{stdout}\n"
            f"--- STDERR ---\n{stderr}"
        )
        raise TimeoutError(error_msg)

    try:
        yield server_url
    finally:
        # Graceful shutdown with active polling for MLX memory cleanup
        proc.terminate()

        # Active polling: Check if process actually terminated (smart wait)
        # instead of blindly waiting 45s every time
        max_wait = 45  # Conservative timeout for very large models (>40GB)
        start = time.time()

        while time.time() - start < max_wait:
            if proc.poll() is not None:
                # Process terminated successfully
                break
            time.sleep(0.5)  # Poll every 500ms
        else:
            # Timeout: Process frozen after 45s, force kill
            # Note: This should be rare - most models cleanup in 10-20s
            proc.kill()
            proc.wait()

        # Explicit garbage collection + buffer time for Metal memory release
        # This helps prevent RAM overlap when transitioning between large models
        # (e.g., Mixtral 29GB → OpenCode 21GB back-to-back tests)
        gc.collect()
        time.sleep(2)  # Buffer for GPU memory deallocation (reduced from 5s)
