"""
Process guard for pytest integration tests.

Tracks spawned server subprocesses and ensures they are terminated on
Ctrl-C (SIGINT), SIGTERM, normal test teardown, and at interpreter exit.

Usage:
- Call `register_popen(proc, label)` after starting a subprocess.
- Optionally `unregister(pid)` after clean termination.
- Handlers are installed automatically when importing this module, but
  can also be installed explicitly via `install_signal_handlers()`.
"""
from __future__ import annotations

import atexit
import os
import signal
import threading
import time
from typing import Dict, Optional

import psutil

_registry_lock = threading.RLock()
_registry: Dict[int, Dict[str, Optional[int]]] = {}
_handlers_installed = False


def _safe_get_pgid(pid: int) -> Optional[int]:
    try:
        return os.getpgid(pid)
    except Exception:
        return None


def register_popen(proc, label: str = "tracked-proc") -> None:
    """Register a subprocess.Popen for guarded cleanup."""
    if proc is None:
        return
    pid = getattr(proc, "pid", None)
    if not pid:
        return
    pgid = _safe_get_pgid(pid)
    with _registry_lock:
        _registry[pid] = {"label": label, "pgid": pgid}


def unregister(pid: int) -> None:
    with _registry_lock:
        _registry.pop(pid, None)


def _kill_pid_tree(pid: int, timeout: float = 1.0) -> None:
    """Terminate a process and its children, escalating if needed."""
    try:
        proc = psutil.Process(pid)
    except psutil.NoSuchProcess:
        return

    # Try to terminate children first
    children = proc.children(recursive=True)
    for ch in children:
        try:
            ch.terminate()
        except psutil.NoSuchProcess:
            pass

    # Terminate main process
    try:
        proc.terminate()
    except psutil.NoSuchProcess:
        return

    t0 = time.time()
    while time.time() - t0 < timeout:
        if not proc.is_running():
            return
        time.sleep(0.1)

    # Escalate to kill
    for ch in children:
        try:
            ch.kill()
        except psutil.NoSuchProcess:
            pass
    try:
        proc.kill()
    except psutil.NoSuchProcess:
        pass


def kill_all(label_filter: Optional[str] = None) -> None:
    """Kill all tracked processes (optionally filtered by label)."""
    with _registry_lock:
        items = list(_registry.items())

    for pid, meta in items:
        label = (meta or {}).get("label")
        pgid = (meta or {}).get("pgid")
        if label_filter and label != label_filter:
            continue
        # Try process group termination first (POSIX)
        if pgid and pgid > 0 and hasattr(os, "killpg"):
            try:
                os.killpg(pgid, signal.SIGTERM)
                # Give the group a moment
                time.sleep(0.2)
            except Exception:
                pass
        # Fallback to individual tree kill with short timeout
        _kill_pid_tree(pid, timeout=0.8)
        # Final escalation: SIGKILL the group if still around
        if pgid and pgid > 0 and hasattr(os, "killpg"):
            try:
                os.killpg(pgid, signal.SIGKILL)
            except Exception:
                pass
        unregister(pid)


def _signal_handler_factory(prev_handler):
    def _handler(signum, frame):
        # Best-effort kill of tracked server processes
        try:
            kill_all()
        finally:
            # Chain to previous handler behavior
            if callable(prev_handler):
                try:
                    prev_handler(signum, frame)
                    return
                except Exception:
                    # If previous handler was Python's default raising KeyboardInterrupt,
                    # re-raise to allow pytest to handle interruption.
                    raise
            # If default/ignore, re-send signal to self to honor semantics
            try:
                signal.signal(signum, signal.SIG_DFL)
                os.kill(os.getpid(), signum)
            except Exception:
                pass
    return _handler


def install_signal_handlers() -> None:
    global _handlers_installed
    if _handlers_installed:
        return
    if os.environ.get("MLXK_TEST_DISABLE_PROCESS_GUARD"):
        _handlers_installed = True
        return
    # Chain SIGINT and SIGTERM
    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            prev = signal.getsignal(sig)
            signal.signal(sig, _signal_handler_factory(prev))
        except Exception:
            pass
    atexit.register(lambda: kill_all())
    _handlers_installed = True


# Note: Do NOT auto-install on import. Tests that need the guard should call
# install_signal_handlers() explicitly to avoid interfering with non-server runs.
