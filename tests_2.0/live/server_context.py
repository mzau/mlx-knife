"""LocalServer context manager for E2E testing (ADR-011).

Provides a clean subprocess-based server lifecycle for testing:
- Starts server with pre-loaded model
- Waits for health check before yielding
- Ensures graceful cleanup on exit
"""

from __future__ import annotations

import gc
import os
import signal
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
    # Pass environment variables (including HF_HOME) to subprocess
    env = os.environ.copy()

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
        text=True,
        env=env,
        start_new_session=True  # Create process group for robust cleanup
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
        # Robust cleanup with process group verification
        # This prevents zombie accumulation from failed cleanups
        cleanup_success = False

        try:
            # Step 1: Graceful shutdown (SIGTERM to process group)
            try:
                pgid = os.getpgid(proc.pid)
                os.killpg(pgid, signal.SIGTERM)
            except (ProcessLookupError, PermissionError):
                # Process/group already gone or not accessible - try direct terminate
                proc.terminate()

            # Step 2: Wait for graceful cleanup (most models finish in 5-10s)
            try:
                proc.wait(timeout=10)
                cleanup_success = True
            except subprocess.TimeoutExpired:
                # Graceful shutdown failed - escalate to SIGKILL
                print(f"\n⚠️  WARNING: Server cleanup timeout after 10s (PID {proc.pid}, model: {model})")
                print(f"    Escalating to SIGKILL...")

        except Exception as e:
            print(f"\n⚠️  WARNING: Error during graceful cleanup: {e}")

        # Step 3: Forceful shutdown if graceful failed
        if not cleanup_success:
            try:
                # Kill process group with SIGKILL
                try:
                    pgid = os.getpgid(proc.pid)
                    os.killpg(pgid, signal.SIGKILL)
                    print(f"    Process group {pgid} killed with SIGKILL")
                except (ProcessLookupError, PermissionError):
                    # Group gone or not accessible - try direct kill
                    proc.kill()
                    print(f"    Process {proc.pid} killed with SIGKILL")

                # Wait for death
                proc.wait(timeout=5)
                cleanup_success = True
            except subprocess.TimeoutExpired:
                # CRITICAL: Process refuses to die even with SIGKILL
                print(f"⚠️  CRITICAL: Process {proc.pid} refuses to die even with SIGKILL!")
            except (ProcessLookupError, PermissionError):
                # Process already gone - cleanup succeeded
                cleanup_success = True
            except Exception as e:
                print(f"⚠️  WARNING: Forceful cleanup failed: {e}")

        # Step 4: Drain pipes (prevents zombies from pipe backpressure)
        try:
            proc.communicate(timeout=5)
        except Exception:
            pass  # Best-effort

        # Step 5: Final verification - ensure no zombies leaked
        try:
            if proc.poll() is None:
                # Still alive - one last kill attempt
                print(f"⚠️  WARNING: Process {proc.pid} still alive after cleanup - final SIGKILL")
                proc.kill()
                proc.wait(timeout=2)
        except Exception:
            pass  # Best-effort

        # Step 6: Verify process group is dead (prevent zombie accumulation)
        try:
            # Check if any processes in the group are still alive
            result = subprocess.run(
                ["pgrep", "-g", str(os.getpgid(proc.pid))],
                capture_output=True,
                timeout=2,
            )
            if result.returncode == 0:
                # Group still has members - kill them all
                subprocess.run(["pkill", "-9", "-g", str(os.getpgid(proc.pid))], timeout=2)
                print(f"⚠️  WARNING: Killed remaining processes in group {os.getpgid(proc.pid)}")
        except (ProcessLookupError, PermissionError, subprocess.TimeoutExpired):
            pass  # Group gone or not accessible - that's fine
        except Exception:
            pass  # Best-effort

        # Step 7: Explicit garbage collection + Metal memory release buffer
        gc.collect()
        time.sleep(2)
