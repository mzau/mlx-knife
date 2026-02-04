"""LocalServer context manager for E2E testing (ADR-011).

Provides a clean subprocess-based server lifecycle for testing:
- Starts server with pre-loaded model
- Waits for health check before yielding
- Ensures graceful cleanup on exit
- Memory-aware cleanup: waits for Metal GPU cache release
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


def _get_available_memory_gb() -> float:
    """Get available system memory in GB (macOS).

    Returns available (free + speculative) memory that can be used immediately.
    Critical for robust test scheduling - ensures enough memory before next test.

    Note: macOS Tahoe caches aggressively, so "free" is often minimal.
    IMPORTANT: We do NOT count "inactive" pages because Metal/GPU cache may hold
    them even though macOS reports them as "reclaimable". This was causing false
    positives where Memory Gates reported 20+ GB available but Pixtral failed
    with "Broken pipe" due to actual memory pressure. (Session 136 fix)
    """
    try:
        result = subprocess.run(
            ["vm_stat"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0:
            lines = result.stdout.split("\n")
            page_size = 16384  # Default macOS page size (Apple Silicon)
            if "page size of" in lines[0]:
                try:
                    page_size = int(lines[0].split("page size of")[1].split()[0])
                except (ValueError, IndexError):
                    pass

            free_pages = 0
            speculative_pages = 0
            for line in lines:
                if "Pages free:" in line:
                    free_pages = int(line.split(":")[1].strip().rstrip("."))
                elif "Pages speculative:" in line:
                    speculative_pages = int(line.split(":")[1].strip().rstrip("."))

            # Available = free + speculative only (NOT inactive - may be held by GPU cache)
            return (free_pages + speculative_pages) * page_size / (1024**3)
    except Exception:
        pass
    return 0.0


def _get_memory_pressure() -> int:
    """Get macOS memory pressure level via sysctl.

    Returns:
        0 = NORMAL (system relaxed, safe to load models)
        1 = WARN (system under some pressure)
        4 = CRITICAL (system under severe pressure)
        -1 = Unable to determine
    """
    try:
        result = subprocess.run(
            ["sysctl", "-n", "vm.memory_pressure"],
            capture_output=True,
            text=True,
            timeout=2,
        )
        if result.returncode == 0:
            return int(result.stdout.strip())
    except Exception:
        pass
    return -1


def _wait_for_memory_release(
    min_free_gb: float = 20.0,
    timeout_seconds: float = 30.0,
    poll_interval: float = 1.0,
) -> bool:
    """Wait for system memory to be released after server shutdown.

    Metal GPU cache is shared across processes and released asynchronously.
    This function actively waits until enough memory is free before
    allowing the next test to start.

    Uses TWO indicators for robust detection (Session 136 finding):
    1. vm.memory_pressure == 0 (macOS kernel says system is relaxed)
    2. Available memory >= min_free_gb (enough free+speculative pages)

    Args:
        min_free_gb: Minimum free memory required (default 20 GB for vision models)
        timeout_seconds: Maximum wait time (default 30s for GPU cache release)
        poll_interval: Time between memory checks (default 1s)

    Returns:
        True if memory threshold reached, False if timeout
    """
    start_time = time.time()

    while time.time() - start_time < timeout_seconds:
        # Check memory pressure first (fast sysctl call)
        pressure = _get_memory_pressure()
        if pressure == 0:  # NORMAL - system is relaxed
            free_gb = _get_available_memory_gb()
            if free_gb >= min_free_gb:
                return True
        time.sleep(poll_interval)

    return False

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

    # Start server_base directly (NOT via CLI) to avoid double start_new_session orphan bug
    # The CLI uses start_new_session=True in serve.py, which creates a separate process group
    # that won't receive our SIGTERM. By starting server_base directly, we control the session.
    env["MLXK2_HOST"] = "127.0.0.1"
    env["MLXK2_PORT"] = str(port)
    env["MLXK2_LOG_LEVEL"] = log_level
    env["MLXK2_PRELOAD_MODEL"] = model

    proc = subprocess.Popen(
        [
            sys.executable, "-m", "mlxk2.core.server_base",
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

        # Step 7: Explicit garbage collection + Metal memory release
        gc.collect()

        # Memory Gate: Wait for memory release (robust scheduling)
        # Metal GPU cache is shared across processes - wait until enough is free
        # 8 GB threshold validated via wet-memmon (avg 10.5 GB free, Firefox running)
        if not _wait_for_memory_release(min_free_gb=8.0, timeout_seconds=10.0):
            free_gb = _get_available_memory_gb()
            print(f"⚠️  Memory release timeout: {free_gb:.1f} GB available (wanted 8 GB)")
            # Continue anyway - test may still succeed or fail with clear OOM error
