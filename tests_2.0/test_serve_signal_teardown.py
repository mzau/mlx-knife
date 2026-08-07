"""Stop-signal teardown of the supervised server (issue #60).

The teardown used to hang off `except KeyboardInterrupt` alone, so `kill`, a shell trap or
launchd killed the supervisor before it ran and the child kept the model and the port.

These tests drive the real `_run_supervised_uvicorn` against a real (but trivial) child,
because the mocked-Popen tests in test_serve_supervisor.py cannot observe process state at
all - which is why the bug survived two releases unnoticed.
"""

import os
import signal
import subprocess
import sys
import threading
import time

import pytest
from unittest.mock import patch

import mlxk2.operations.serve as serve_mod
from mlxk2.core.parent_watch import PARENT_ALIVE_FD_ENV

# serve.py imports the subprocess module, so patching Popen there patches it everywhere.
# Keep the real one to spawn our stand-in child with.
_REAL_POPEN = subprocess.Popen

# Stands in for the server: sleeps, and dies on SIGTERM unless asked to ignore it.
_SLEEPER = (
    "import signal, sys, time\n"
    "if 'stubborn' in sys.argv:\n"
    "    signal.signal(signal.SIGTERM, signal.SIG_IGN)\n"
    "time.sleep(60)\n"
)


def _group_is_gone(pid: int) -> bool:
    try:
        os.killpg(pid, 0)
    except ProcessLookupError:
        return True
    return False


def _supervise(signals, *, stubborn=False, spacing=0.4):
    """Run the real supervisor against a sleeper child, signalling ourselves as we go.

    Returns (exit_code, child_proc, elapsed_seconds, popen_env).
    """
    spawned = {}

    def fake_popen(cmd, env=None, **kw):
        argv = [sys.executable, "-c", _SLEEPER] + (["stubborn"] if stubborn else [])
        proc = _REAL_POPEN(argv, env=env, **kw)
        spawned["proc"] = proc
        spawned["env"] = env
        return proc

    timers = [
        threading.Timer(spacing * (i + 1), os.kill, (os.getpid(), sig))
        for i, sig in enumerate(signals)
    ]
    started = time.time()
    with patch.object(serve_mod.subprocess, "Popen", side_effect=fake_popen):
        for timer in timers:
            timer.start()
        try:
            rc = serve_mod._run_supervised_uvicorn("127.0.0.1", 8000, "warning")
        finally:
            for timer in timers:
                timer.cancel()
    return rc, spawned["proc"], time.time() - started, spawned["env"]


@pytest.mark.parametrize("sig", [signal.SIGTERM, signal.SIGHUP, signal.SIGINT])
def test_stop_signal_tears_the_child_down(sig):
    """The bug: only SIGINT did this. SIGTERM/SIGHUP left the child running."""
    rc, proc, _, _ = _supervise([sig])
    assert _group_is_gone(proc.pid), f"{sig.name} left the child group alive"
    assert rc == -signal.SIGTERM  # the child got the graceful signal, not SIGKILL


def test_second_stop_signal_skips_the_remaining_grace():
    """A stubborn child normally costs the full 5s grace; a second signal cuts it short."""
    rc, proc, elapsed, _ = _supervise([signal.SIGTERM, signal.SIGTERM], stubborn=True)
    assert _group_is_gone(proc.pid)
    assert rc == -signal.SIGKILL
    assert elapsed < 3.0, f"escalation waited {elapsed:.1f}s - the grace was not skipped"


def test_stubborn_child_is_killed_after_the_grace():
    rc, proc, elapsed, _ = _supervise([signal.SIGTERM], stubborn=True)
    assert _group_is_gone(proc.pid)
    assert rc == -signal.SIGKILL
    assert elapsed >= 5.0


def test_handlers_are_restored():
    before = {sig: signal.getsignal(sig) for sig in serve_mod._STOP_SIGNALS}
    _supervise([signal.SIGTERM])
    after = {sig: signal.getsignal(sig) for sig in serve_mod._STOP_SIGNALS}
    assert after == before


@pytest.mark.parametrize(
    "rc, expected",
    [(0, 0), (1, 1), (-signal.SIGTERM, 143), (-signal.SIGKILL, 137)],
)
def test_signal_deaths_are_reported_the_way_a_shell_reports_them(rc, expected):
    """sys.exit(-15) surfaces as 241, which no supervisor can read as 'stopped on request'."""
    assert serve_mod._cli_exit_code(rc) == expected


def test_child_is_handed_a_supervisor_pipe():
    """Second half of the fix: the child needs a live fd to notice a dead supervisor."""
    _, _, _, env = _supervise([signal.SIGTERM])
    assert int(env[PARENT_ALIVE_FD_ENV]) >= 0
