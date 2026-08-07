"""`mlxk serve` as a user runs it: started from the CLI, stopped with a signal (issue #60).

Every other live test boots ``mlxk2.core.server_base`` directly through server_context.py,
which routes around the supervisor entirely - that is precisely why the orphan bug survived
two releases. This file is the only one that exercises the CLI's own process handling.

No model is preloaded: the contract under test is about processes and the port, and both
hold whatever is loaded.
"""

import os
import signal
import socket
import subprocess
import sys
import time

import pytest

try:
    import httpx
except ImportError:  # pragma: no cover
    httpx = None

pytestmark = pytest.mark.live_e2e

STARTUP_TIMEOUT = 60.0   # imports plus app construction, no model load
SHUTDOWN_TIMEOUT = 20.0  # 5s grace + SIGKILL, plus room for a loaded runtime


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def _port_is_free(port: int) -> bool:
    with socket.socket() as sock:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            sock.bind(("127.0.0.1", port))
        except OSError:
            return False
    return True


def _server_child(supervisor_pid: int):
    """The uvicorn process the supervisor spawned, or None."""
    found = subprocess.run(
        ["pgrep", "-P", str(supervisor_pid)], capture_output=True, text=True
    )
    pids = [int(line) for line in found.stdout.split() if line.strip()]
    return pids[0] if pids else None


def _alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    return True


def _wait_until(predicate, timeout: float, what: str):
    deadline = time.time() + timeout
    while time.time() < deadline:
        if predicate():
            return
        time.sleep(0.2)
    pytest.fail(f"timed out after {timeout:.0f}s waiting for {what}")


def _start_cli_server(port: int):
    """Start `mlxk serve` through the CLI and wait until it answers. Returns (proc, child)."""
    proc = subprocess.Popen(
        [sys.executable, "-m", "mlxk2.cli", "serve",
         "--host", "127.0.0.1", "--port", str(port), "--log-level", "warning"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    deadline = time.time() + STARTUP_TIMEOUT
    while time.time() < deadline:
        if proc.poll() is not None:
            pytest.fail(f"mlxk serve exited during startup (rc={proc.returncode})")
        try:
            if httpx.get(f"http://127.0.0.1:{port}/health", timeout=1.0).status_code == 200:
                child = _server_child(proc.pid)
                assert child is not None, "supervisor answered /health but has no child"
                return proc, child
        except Exception:
            time.sleep(0.5)
    proc.kill()
    pytest.fail(f"mlxk serve did not answer /health within {STARTUP_TIMEOUT:.0f}s")


@pytest.mark.skipif(httpx is None, reason="httpx required")
@pytest.mark.parametrize(
    "sig",
    [signal.SIGTERM, signal.SIGHUP, signal.SIGKILL],
    ids=["sigterm", "sighup", "sigkill"],
)
def test_signalling_the_supervisor_leaves_nothing_behind(sig):
    """SIGTERM/SIGHUP are handled; SIGKILL cannot be - there the child stops itself."""
    port = _free_port()
    proc, child = _start_cli_server(port)
    try:
        os.kill(proc.pid, sig)
        proc.wait(timeout=SHUTDOWN_TIMEOUT)
        # The stranded model is the visible half; the bound port is the worse one, because
        # a restart then talks to the old server with the old model and says nothing.
        _wait_until(lambda: not _alive(child), SHUTDOWN_TIMEOUT, f"child {child} to exit")
        _wait_until(lambda: _port_is_free(port), SHUTDOWN_TIMEOUT, f"port {port} to be free")
    finally:
        if _alive(child):
            os.kill(child, signal.SIGKILL)
        if proc.poll() is None:
            proc.kill()
