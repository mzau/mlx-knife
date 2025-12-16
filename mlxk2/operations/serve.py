"""
Server operation for 2.0 implementation.
"""

import os
import signal
import subprocess
import sys
import time
from typing import Optional

from ..core.server_base import run_server


def _run_supervised_uvicorn(host: str, port: int, log_level: str, reload: bool = False) -> int:
    """Run server as a supervised subprocess and handle Ctrl-C in parent.

    Uses the server_base __main__ entrypoint instead of uvicorn CLI directly.
    This ensures proper JSON log configuration via MLXK2_LOG_JSON env var.

    Returns the subprocess' exit code.
    """
    # Pass configuration via environment variables to subprocess
    # This allows the __main__ entrypoint to configure run_server() properly
    env = os.environ.copy()
    env["MLXK2_HOST"] = host
    env["MLXK2_PORT"] = str(port)
    env["MLXK2_LOG_LEVEL"] = log_level
    if reload:
        env["MLXK2_RELOAD"] = "1"

    # Note: MLXK2_LOG_JSON and MLXK2_PRELOAD_MODEL are already set by start_server()

    cmd = [
        sys.executable,
        "-m",
        "mlxk2.core.server_base",
    ]

    # Start in a new session so we can signal the whole process group
    proc = subprocess.Popen(
        cmd,
        env=env,
        start_new_session=True,
    )

    try:
        return proc.wait()
    except KeyboardInterrupt:
        # Suppress further SIGINT while we clean up
        previous = signal.signal(signal.SIGINT, signal.SIG_IGN)
        try:
            # First Ctrl-C: ask child to stop gracefully
            try:
                os.killpg(proc.pid, signal.SIGTERM)
            except Exception:
                pass
            # Wait briefly, then force kill if still alive
            deadline = time.time() + 5.0
            while time.time() < deadline:
                ret = proc.poll()
                if ret is not None:
                    return ret
                try:
                    time.sleep(0.1)
                except KeyboardInterrupt:
                    # Second Ctrl-C: escalate to SIGKILL immediately
                    break
            try:
                os.killpg(proc.pid, signal.SIGKILL)
            except Exception:
                pass
            # Wait for child without being interrupted
            while True:
                ret = proc.poll()
                if ret is not None:
                    return ret
                time.sleep(0.05)
        finally:
            # Restore previous handler
            try:
                signal.signal(signal.SIGINT, previous)
            except Exception:
                pass


def start_server(
    model: Optional[str] = None,
    port: int = 8000,
    host: str = "127.0.0.1",
    max_tokens: Optional[int] = None,
    reload: bool = False,
    log_level: str = "info",
    verbose: bool = False,
    supervise: bool = True,
) -> None:
    """Start OpenAI-compatible API server for MLX models.

    Args:
        model: Specific model to pre-load on startup (optional)
               If specified, validates model with probe/policy before starting.
               Server will fail-fast if model is incompatible (vision, memory, etc.)
        port: Port to bind the server to
        host: Host address to bind to
        max_tokens: Default maximum tokens for generation
        reload: Enable auto-reload for development
        log_level: Logging level
        verbose: Show detailed output
        supervise: Run uvicorn in a supervised subprocess for instant Ctrl-C
    """
    if verbose:
        print("Starting MLX Knife Server 2.0...")
        if model:
            print(f"Pre-loading model: {model}")
        print(f"Server will bind to: http://{host}:{port}")

    # Pre-load validation happens in server_base.py lifespan hook
    # via environment variable MLXK2_PRELOAD_MODEL

    if supervise:
        # Pass log_level and model via environment to subprocess (ADR-004)
        os.environ["MLXK2_LOG_LEVEL"] = log_level
        if model:
            os.environ["MLXK2_PRELOAD_MODEL"] = model
        # Delegate to subprocess-managed uvicorn
        _ = _run_supervised_uvicorn(host=host, port=port, log_level=log_level, reload=reload)
        return

    # Default: run uvicorn in-process
    run_server(
        host=host,
        port=port,
        max_tokens=max_tokens,
        reload=reload,
        log_level=log_level,
        preload_model=model,
    )
