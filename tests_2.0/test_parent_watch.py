"""Supervisor-death watch in the server subprocess (issue #60, second half).

Signals cover every stop a user or a script can send. They cannot cover SIGKILL on the
supervisor, or a supervisor crash - after those, only the child itself can notice. It does
so by watching a pipe whose write end dies with the supervisor.
"""

import os
import subprocess
import sys
import threading
import time
from pathlib import Path

import pytest

from mlxk2.core import parent_watch
from mlxk2.core.parent_watch import PARENT_ALIVE_FD_ENV, watch_supervisor

_REPO_ROOT = Path(__file__).resolve().parents[1]

# Spawned by _ORPHAN_MAKER below. Dies when its supervisor does - or refuses to take the
# graceful signal, to prove the watch escalates on its own.
_WATCHED_CHILD = (
    "import signal, sys, time\n"
    "from mlxk2.core.parent_watch import watch_supervisor\n"
    "if 'stubborn' in sys.argv:\n"
    "    signal.signal(signal.SIGTERM, signal.SIG_IGN)\n"
    "assert watch_supervisor(grace=1.0)\n"
    "time.sleep(60)\n"
)

# Stands in for a supervisor that is SIGKILLed or crashes: it spawns the child the way
# serve.py does, reports the pid, and vanishes without any cleanup of its own.
_ORPHAN_MAKER = (
    "import os, subprocess, sys\n"
    "read_fd, write_fd = os.pipe()\n"
    "env = dict(os.environ, MLXK2_PARENT_ALIVE_FD=str(read_fd))\n"
    "proc = subprocess.Popen([sys.executable, '-c'] + sys.argv[1:], env=env,\n"
    "                        pass_fds=(read_fd,), start_new_session=True)\n"
    "print(proc.pid, flush=True)\n"
    "os._exit(0)\n"
)


def _still_running(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    return True


def test_no_supervisor_means_no_watch(monkeypatch):
    """A directly started server - the live-test harness - must be unaffected."""
    monkeypatch.delenv(PARENT_ALIVE_FD_ENV, raising=False)
    assert watch_supervisor() is False


def test_malformed_fd_is_ignored(monkeypatch):
    monkeypatch.setenv(PARENT_ALIVE_FD_ENV, "not-a-number")
    assert watch_supervisor() is False


def test_eof_arrives_only_once_the_write_end_closes():
    read_fd, write_fd = os.pipe()
    seen = threading.Event()
    threading.Thread(
        target=lambda: (parent_watch._wait_for_eof(read_fd), seen.set()),
        daemon=True,
    ).start()
    try:
        time.sleep(0.2)
        assert not seen.is_set(), "reported the supervisor dead while it was alive"
        os.close(write_fd)
        assert seen.wait(2.0), "did not notice the write end closing"
    finally:
        os.close(read_fd)


@pytest.mark.parametrize("stubborn", [False, True], ids=["graceful", "stubborn"])
def test_child_stops_itself_when_the_supervisor_vanishes(stubborn):
    argv = [sys.executable, "-c", _ORPHAN_MAKER, _WATCHED_CHILD]
    if stubborn:
        argv.append("stubborn")
    maker = subprocess.run(
        argv, capture_output=True, text=True, timeout=30, cwd=str(_REPO_ROOT)
    )
    assert maker.returncode == 0, maker.stderr
    pid = int(maker.stdout.strip())

    deadline = time.time() + 10.0
    try:
        while time.time() < deadline:
            if not _still_running(pid):
                return
            time.sleep(0.1)
        pytest.fail(f"child {pid} outlived its supervisor")
    finally:
        if _still_running(pid):
            os.kill(pid, 9)
