"""Supervisor liveness watch for the supervised server subprocesses (issue #60).

``mlxk serve`` and ``mlxk embed-serve`` hand their child the read end of a pipe whose
write end only the supervisor holds. The supervisor's signal handlers cover every stop a
user or a script can send; this covers the two they cannot: SIGKILL on the supervisor, and
a supervisor crash. macOS has no PR_SET_PDEATHSIG, so the pipe is the portable equivalent.

No-op unless MLXK2_PARENT_ALIVE_FD is set, so a directly started server - the live-test
harness, or ``start_server(supervise=False)`` - keeps the semantics it has always had.
"""

import os
import signal
import threading
import time

from ..logging import get_logger

logger = get_logger()

PARENT_ALIVE_FD_ENV = "MLXK2_PARENT_ALIVE_FD"


def watch_supervisor(grace: float = 5.0) -> bool:
    """Start a daemon thread that ends this process once the supervisor is gone.

    Returns True when the watch is active, False when there is no supervisor to watch.
    """
    raw = os.environ.get(PARENT_ALIVE_FD_ENV)
    if not raw:
        return False
    try:
        fd = int(raw)
    except ValueError:
        logger.warning(f"Ignoring malformed {PARENT_ALIVE_FD_ENV}={raw!r}")
        return False

    def _watch() -> None:
        _wait_for_eof(fd)
        logger.warning("Supervisor is gone - shutting down")
        _stop_self(grace)

    threading.Thread(target=_watch, name="mlxk-supervisor-watch", daemon=True).start()
    return True


def _wait_for_eof(fd: int) -> None:
    """Block until the pipe's last write end closes. The supervisor never writes."""
    while True:
        try:
            if not os.read(fd, 1):
                return
        except InterruptedError:
            continue
        except OSError:
            return  # fd went away - treat that like the supervisor being gone


def _stop_self(grace: float) -> None:
    """SIGTERM ourselves, then SIGKILL if that was not enough.

    The escalation is not optional: with the supervisor gone, nobody is left to force the
    issue if the graceful shutdown stalls.
    """
    # Started by the supervisor we lead our own group, so signalling the group takes
    # uvicorn's --reload children along. Otherwise signal ourselves, never strangers.
    pgid = os.getpgrp()
    leader = pgid == os.getpid()

    def _send(sig: int) -> None:
        try:
            if leader:
                os.killpg(pgid, sig)
            else:
                os.kill(os.getpid(), sig)
        except OSError:
            os._exit(1)

    _send(signal.SIGTERM)
    time.sleep(grace)
    _send(signal.SIGKILL)
