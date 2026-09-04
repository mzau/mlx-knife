"""
Server operation for 2.0 implementation.
"""

import os
import signal
import subprocess
import sys
import time
from typing import Optional

from ..core.parent_watch import PARENT_ALIVE_FD_ENV
from ..core.server_base import _operator_ceiling_from_env, run_server

# Every stop signal must reach the same teardown. Before issue #60 only SIGINT did, so
# `kill`, a shell trap or launchd killed the supervisor and left the child holding the port.
_STOP_SIGNALS = tuple(
    getattr(signal, name)
    for name in ("SIGINT", "SIGTERM", "SIGHUP")
    if hasattr(signal, name)
)


class _StopRequested(BaseException):
    """Raised by the stop-signal handler so the teardown runs in the main flow.

    Deliberately not a KeyboardInterrupt subclass: Popen.wait() special-cases that one
    and waits another 0.25s for a child that never sees SIGINT (it has its own session).
    """


def _cli_exit_code(rc: int) -> int:
    """Turn Popen's exit code into one a shell or a supervisor understands.

    Popen reports a signal death as a negative number, and sys.exit(-15) surfaces as 241 -
    which means nothing to anyone. 128+signo is what a shell reports for the same event.
    """
    return 128 - rc if rc < 0 else rc


def _run_supervised_uvicorn(
    host: str,
    port: int,
    log_level: str,
    reload: bool = False,
    *,
    module: str = "mlxk2.core.server_base",
    extra_env: Optional[dict] = None,
) -> int:
    """Run a server as a supervised subprocess and own its shutdown.

    Ctrl-C, SIGTERM and SIGHUP all end in the same teardown: SIGTERM to the child's
    process group, five seconds of grace, then SIGKILL. A second stop signal skips the
    rest of the grace. If we die anyway, the child notices and stops itself.

    Uses the given module's __main__ entrypoint (default ``mlxk2.core.server_base``;
    embed-serve passes ``mlxk2.core.embed_server_base``) instead of the uvicorn CLI.
    This ensures proper JSON log configuration via MLXK2_LOG_JSON env var. ``extra_env``
    carries caller-specific subprocess config (e.g. embed-serve's MLXK2_EMBED_MODEL).

    Returns the subprocess' exit code.
    """
    # Pass configuration via environment variables to subprocess
    # This allows the __main__ entrypoint to configure run_server() properly
    env = os.environ.copy()
    env["MLXK2_HOST"] = host
    env["MLXK2_PORT"] = str(port)
    env["MLXK2_LOG_LEVEL"] = log_level

    # Suppress transformers/tokenizers noise in server subprocess (Session 89 + Session 90 fix)
    # IMPORTANT: Set in subprocess ENV, NOT in global __init__.py (breaks huggingface_hub downloads)
    env["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "1"
    env["TOKENIZERS_PARALLELISM"] = "false"  # Prevent fork warning in uvicorn/multiprocessing

    if reload:
        env["MLXK2_RELOAD"] = "1"

    # Note: MLXK2_LOG_JSON and MLXK2_PRELOAD_MODEL are already set by start_server()

    # Caller-supplied subprocess env (e.g. embed-serve's MLXK2_EMBED_MODEL / MLXK2_EMBED_CPU).
    if extra_env:
        env.update(extra_env)

    cmd = [
        sys.executable,
        "-m",
        module,
    ]

    # Signals cannot cover SIGKILL on us, or our own crash. The child watches this pipe's
    # read end and stops itself when our write end dies with us (issue #60, second half).
    read_fd, write_fd = os.pipe()
    env[PARENT_ALIVE_FD_ENV] = str(read_fd)

    # The first stop signal enters the teardown below; a second one skips the rest of the
    # grace period instead of killing us mid-cleanup.
    state = {"stopping": False, "escalate": False}

    def _on_stop(signum, frame):
        if state["stopping"]:
            state["escalate"] = True
            return
        state["stopping"] = True
        raise _StopRequested(signum)

    # Installed before Popen: a signal arriving between spawn and wait used to orphan the
    # child outright, which is exactly the `mlxk serve & kill $!` case from scripts.
    previous = {}
    for sig in _STOP_SIGNALS:
        try:
            previous[sig] = signal.signal(sig, _on_stop)
        except (ValueError, OSError):
            pass  # not the main thread, or not supported here

    proc = None
    try:
        # Start in a new session so we can signal the whole process group
        proc = subprocess.Popen(
            cmd,
            env=env,
            start_new_session=True,
            pass_fds=(read_fd,),
        )
        os.close(read_fd)
        read_fd = -1
        return proc.wait()
    except (_StopRequested, KeyboardInterrupt) as stop:
        if proc is None:
            # Signalled before the child existed - there is nothing to tear down.
            return 128 + int(stop.args[0] if stop.args else signal.SIGINT)
        try:
            os.killpg(proc.pid, signal.SIGTERM)
        except Exception:
            pass
        deadline = time.time() + 5.0
        while time.time() < deadline and not state["escalate"]:
            ret = proc.poll()
            if ret is not None:
                return ret
            time.sleep(0.1)
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except Exception:
            pass
        while True:
            ret = proc.poll()
            if ret is not None:
                return ret
            time.sleep(0.05)
    finally:
        # A signal arriving here has nothing left to stop; let the handler fall through to
        # its second-signal branch rather than raise out of a finally.
        state["stopping"] = True
        for sig, handler in previous.items():
            try:
                signal.signal(sig, handler)
            except (ValueError, OSError):
                pass
        for fd in (read_fd, write_fd):
            if fd >= 0:
                try:
                    os.close(fd)
                except OSError:
                    pass


def validate_serve_options(
    max_tokens: Optional[int] = None,
    chunk: int = 1,
    embed_backend: Optional[str] = None,
) -> Optional[int]:
    """Check what `serve` was asked for, and report the ceiling that will apply.

    Side-effect free on purpose: the CLI runs this *before* it prints anything, so a
    rejected option cannot be preceded by a "starting" envelope. `start_server` runs it
    too, for callers that skip the CLI.

    Returns the operator ceiling in force — the flag if given, else MLXK2_MAX_TOKENS.
    """
    from ..tools.vision_adapter import MAX_SAFE_CHUNK_SIZE
    if chunk < 1:
        raise ValueError(
            f"chunk size must be at least 1 (got: {chunk})."
        )
    if chunk > MAX_SAFE_CHUNK_SIZE:
        raise ValueError(
            f"chunk size too large (max: {MAX_SAFE_CHUNK_SIZE} for Metal API stability). "
            f"This limit is based on empirically tested performance."
        )

    # The child would otherwise die on this, and it only knows the environment variable's
    # wording, never the flag's. An exported MLXK2_MAX_TOKENS is the operator's other way in.
    if max_tokens is not None and max_tokens < 1:
        raise ValueError(f"--max-tokens must be at least 1 (got {max_tokens})")
    ceiling = max_tokens if max_tokens is not None else _operator_ceiling_from_env()

    if embed_backend is not None:
        from urllib.parse import urlparse
        parsed = urlparse(embed_backend)
        if parsed.scheme not in ("http", "https") or not parsed.netloc:
            raise ValueError(
                f"--embed-backend must be an http(s) URL with a host (got: {embed_backend!r})"
            )
    return ceiling


def start_server(
    model: Optional[str] = None,
    port: int = 8000,
    host: str = "127.0.0.1",
    max_tokens: Optional[int] = None,
    reload: bool = False,
    log_level: str = "info",
    chunk: int = 1,
    verbose: bool = False,
    supervise: bool = True,
    embed_backend: Optional[str] = None,
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
        chunk: Default batch size for vision requests (default: 1, max: 5)
        verbose: Show detailed output
        supervise: Run uvicorn in a supervised subprocess for deterministic shutdown
        embed_backend: ADR-015 D2 — URL of a separate embed-serve backend. When set,
               serve proxies POST /v1/embeddings to it (the embed model is never loaded
               here). The URL is validated fail-fast; the backend is NOT probed at startup.
    """
    ceiling = validate_serve_options(max_tokens=max_tokens, chunk=chunk, embed_backend=embed_backend)
    del ceiling  # start_server exports the flag itself; the value is for the caller's report

    # ADR-015 D2: --embed-backend is the single source of truth for the proxy. Validate the URL
    # fail-fast, then bridge it to the server subprocess via env (MLXK2_EMBED_BACKEND) —
    # _run_supervised_uvicorn copies os.environ, so the server_base lifespan reads it directly
    # (no run_server() signature change needed). When the flag is absent, clear any ambient value
    # so an exported env var can't silently enable (and thereby un-gate) the proxy.
    if embed_backend is not None:
        os.environ["MLXK2_EMBED_BACKEND"] = embed_backend
    else:
        os.environ.pop("MLXK2_EMBED_BACKEND", None)

    # Set environment variables for server configuration
    # These apply to both supervised and non-supervised modes
    os.environ["MLXK2_LOG_LEVEL"] = log_level
    # Suppress tqdm progress bars in server mode (must be set before tqdm import)
    os.environ["TQDM_DISABLE"] = "1"
    if model:
        # Pre-validate model specification before starting server (consistency with run.py)
        from ..core.model_resolution import resolve_model_for_operation
        from .workspace import is_explicit_path
        resolved_name, _, ambiguous = resolve_model_for_operation(model)
        if ambiguous:
            raise ValueError(
                f"Ambiguous model specification '{model}'. Could be: {ambiguous}"
            )
        if not resolved_name:
            # Model not found - give appropriate error message
            if is_explicit_path(model):
                raise ValueError(f"Workspace not found: {model}")
            else:
                raise ValueError(f"Model not found in cache: {model}")
        os.environ["MLXK2_PRELOAD_MODEL"] = model
    if max_tokens is not None:
        os.environ["MLXK2_MAX_TOKENS"] = str(max_tokens)
    if chunk != 1:
        os.environ["MLXK2_VISION_CHUNK_SIZE"] = str(chunk)

    if verbose:
        print("Starting MLX Knife Server 2.0...")
        if model:
            print(f"Pre-loading model: {model}")
        print(f"Server will bind to: http://{host}:{port}")
        if chunk != 1:
            print(f"Vision batch size: {chunk}")

    # Pre-load validation happens in server_base.py lifespan hook
    # via environment variable MLXK2_PRELOAD_MODEL

    if supervise:
        # Delegate to subprocess-managed uvicorn (env vars already set above)
        exit_code = _cli_exit_code(
            _run_supervised_uvicorn(host=host, port=port, log_level=log_level, reload=reload)
        )
        # Propagate failure exit codes to caller (for CI/CD)
        if exit_code != 0:
            sys.exit(exit_code)
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
