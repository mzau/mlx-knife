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
    """Run uvicorn as a supervised subprocess and handle Ctrl-C in parent.

    Returns the subprocess' exit code.
    """
    cmd = [
        sys.executable,
        "-m",
        "uvicorn",
        "mlxk2.core.server_base:app",
        "--host",
        host,
        "--port",
        str(port),
        "--log-level",
        log_level,
        "--workers",
        "1",
        "--timeout-keep-alive",
        "5",
        "--timeout-graceful-shutdown",
        "5",
        "--lifespan",
        "on",
    ]
    if reload:
        cmd.append("--reload")

    # Start in a new session so we can signal the whole process group
    proc = subprocess.Popen(
        cmd,
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
        model: Specific model to load on startup (optional)
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

    if supervise:
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
    )
