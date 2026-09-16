"""No mlxk process imports Python from the directory the command was started in.

Two ways in, both measured. `mlxk serve` started its worker with `-m`, which puts that
directory first on the worker's module search path, so a `mlxk2/` or a `fastapi.py` lying next
to the models ran instead of the installed package. And every command starts interpreters
underneath itself: multiprocessing runs its resource tracker as `python -c …` in the same
directory, and one progress bar is enough to start one - which is how a transcription reached a
planted `multiprocessing/` package. `mlxk clone` copies every file of a model repository, so
such a directory is not hypothetical.

These tests drive the real supervisor against real children: the mocked-Popen tests in
test_serve_supervisor.py cannot observe what a child imports, which is the whole question here.
"""

import os
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path

import pytest

import mlxk2
import mlxk2.operations.serve as serve_mod

# Cheap stand-in for the server: it pulls in mlxk2 and mlxk2.logging (stdlib only from there)
# and has no __main__ block, so the child is done as soon as the imports are.
TARGET = "mlxk2.core.parent_watch"

CHILD_TIMEOUT = 60.0


def _bait(marker: Path) -> str:
    """A planted module that records which process ran it, then ends that process at once.

    It writes more than a flag because the answer "something imported it" is not actionable:
    the interesting part is whether that something was the worker, an interpreter the worker
    started, or a process of this test run.
    """
    return (
        "import os, sys\n"
        f"with open({str(marker)!r}, 'a') as fh:\n"
        "    fh.write(f'pid={os.getpid()} ppid={os.getppid()} argv={sys.argv[:2]} "
        "safe={sys.flags.safe_path} env={os.environ.get(\"PYTHONSAFEPATH\")} "
        "path0={sys.path[0]!r}\\n')\n"
        "os._exit(0)\n"
    )


@pytest.fixture(scope="module", autouse=True)
def tracker_started_outside_the_bait():
    """Start this process's own resource tracker before any test enters a planted directory.

    It is launched the way every helper interpreter is - `-c`, inheriting the directory the
    starting process is in - and it does its imports a moment later, so a tracker started
    inside a planted directory reports a defect of this test run rather than of the server.
    Started here, it holds the repository as its directory for the rest of the module.
    """
    from multiprocessing import resource_tracker

    resource_tracker.ensure_running()


@pytest.fixture
def start_directory(tmp_path, monkeypatch):
    """A working directory the test controls, entered with a predictable environment.

    An inherited PYTHONSAFEPATH would make the pre-fix behaviour look fixed, and an empty
    PYTHONPATH component puts the working directory back on the path no matter what the
    worker does - neither belongs in the measurement.
    """
    monkeypatch.delenv("PYTHONSAFEPATH", raising=False)
    inherited = os.environ.get("PYTHONPATH")
    if inherited is not None:
        kept = [part for part in inherited.split(os.pathsep) if part]
        if kept:
            monkeypatch.setenv("PYTHONPATH", os.pathsep.join(kept))
        else:
            monkeypatch.delenv("PYTHONPATH")
    monkeypatch.chdir(tmp_path)
    return tmp_path


def _supervise(module=TARGET, **kwargs):
    """Run the real supervisor against a real child, with a rescue for a child that hangs.

    The supervisor waits without a timeout; SIGTERM to ourselves enters its own teardown,
    which is the same path a user's Ctrl-C takes.
    """
    rescue = threading.Timer(CHILD_TIMEOUT, os.kill, (os.getpid(), signal.SIGTERM))
    rescue.start()
    try:
        return serve_mod._run_supervised_uvicorn("127.0.0.1", 0, "warning", module=module, **kwargs)
    finally:
        rescue.cancel()


@pytest.mark.parametrize("planted", ["mlxk2/__init__.py", "json.py"])
def test_the_worker_runs_no_module_from_the_start_directory(start_directory, planted):
    """mlxk2's own name and a plain import of the worker, planted where the server starts."""
    marker = start_directory / f"marker-{planted.replace('/', '-')}"
    bait = start_directory / planted
    bait.parent.mkdir(parents=True, exist_ok=True)
    bait.write_text(_bait(marker))

    exit_code = _supervise()

    # Order matters: a checkout without an install exits non-zero for a different reason, and
    # asserting the code first would report that as the security property holding.
    assert not marker.exists(), f"the worker executed the planted {planted}"
    assert exit_code == 0, f"worker exited {exit_code}"


@pytest.mark.parametrize(
    "on_pythonpath, untouched",
    [
        (("elsewhere",), False),                 # the root is not on the path at all
        (("elsewhere", "root"), False),          # it is, behind the other copy
        (("elsewhere", "alias"), False),         # ... under another name
        (("root", "elsewhere"), True),           # it comes first anyway
        (("bystander", "root", "elsewhere"), True),  # it wins anyway, behind a stranger
    ],
    ids=["absent", "behind", "behind-as-alias", "already-first", "already-winning"],
)
def test_the_worker_imports_the_package_the_supervisor_imported(tmp_path, on_pythonpath, untouched):
    """`python -m mlxk2.cli serve` inside a checkout must not hand the worker another copy.

    The bootstrap runs against a fake package here, so the three copies can be told apart:
    the supervisor's root, one on PYTHONPATH, one in the working directory. Being on the path
    is not the same as coming first - the root may sit on PYTHONPATH behind the other copy,
    also under another name. And where the root wins on its own, the path has to stay as it
    is: in an ordinary install the root is site-packages, which belongs behind the stdlib.
    """
    root = tmp_path / "check out ü"          # spaces and non-ASCII survive as argv
    elsewhere = tmp_path / "elsewhere"
    here = tmp_path / "here"
    for where, tag in ((root, "root"), (elsewhere, "pythonpath"), (here, "cwd")):
        (where / "pkg").mkdir(parents=True)
        (where / "pkg" / "__init__.py").write_text(f"TAG = {tag!r}\n")
        (where / "pkg" / "mod.py").write_text(
            "import sys\n"
            "from . import TAG\n"                      # a relative import needs __package__
            "main = sys.modules['__main__']\n"
            "print(TAG, __name__, main.__spec__.name, sys.argv == [__file__])\n"
            "print(sys.path)\n"
            "sys.exit(7)\n"
        )
    places = {"root": root, "elsewhere": elsewhere, "alias": tmp_path / "alias",
              "bystander": tmp_path / "bystander"}
    places["alias"].symlink_to(root)
    places["bystander"].mkdir()
    env = {k: v for k, v in os.environ.items() if k not in ("PYTHONSAFEPATH", "PYTHONPATH")}
    env["PYTHONPATH"] = os.pathsep.join(str(places[name]) for name in on_pythonpath)
    worker_env = {**env, "PYTHONSAFEPATH": "1"}

    done = subprocess.run(
        [sys.executable, "-c", serve_mod._WORKER_BOOTSTRAP, "pkg.mod", str(root)],
        cwd=here, env=worker_env, capture_output=True, text=True, timeout=CHILD_TIMEOUT,
    )

    # Same shape as `python -m pkg.mod`: run as __main__, argv[0] the module file, the spec
    # naming the module, and the module's own exit code passed through.
    shape, path = (done.stdout.splitlines() + ["", ""])[:2]
    assert shape.split() == ["root", "__main__", "pkg.mod", "True"], done.stderr
    assert done.returncode == 7

    if untouched:
        plain = subprocess.run(
            [sys.executable, "-c", "import sys; print(sys.path)"],
            cwd=here, env=worker_env, capture_output=True, text=True, timeout=CHILD_TIMEOUT,
        )
        assert path == plain.stdout.strip(), "the bootstrap moved a path that already won"

    # The control: `-m` is what the fix replaced, and it takes the copy lying in the cwd.
    control = subprocess.run(
        [sys.executable, "-m", "pkg.mod"],
        cwd=here, env=env, capture_output=True, text=True, timeout=CHILD_TIMEOUT,
    )
    assert control.stdout.split()[0] == "cwd", control.stderr


def _plant_multiprocessing(start_directory):
    """`multiprocessing` in the start directory: the name every helper interpreter imports."""
    marker = start_directory / "marker-multiprocessing"
    bait = start_directory / "multiprocessing"
    bait.mkdir()
    (bait / "__init__.py").write_text(_bait(marker))
    return marker


def _run_helper(source, start_directory, tmp_path, name):
    """Run a module of our own as the worker; it starts the helper interpreter in question.

    It lies on PYTHONPATH rather than in the start directory, so the worker has to find it
    the way it finds mlxk2 - anything reachable only through the start directory would make
    the test pass for the wrong reason. Returns (exit code, receipt path).
    """
    receipt = tmp_path / f"receipt-{name}"
    helper = tmp_path / f"helper-{name}"
    helper.mkdir()
    (helper / f"{name}.py").write_text(source)
    exit_code = _supervise(
        module=name,
        extra_env={"PYTHONPATH": str(helper), "SEARCH_PATH_TEST_RECEIPT": str(receipt)},
    )
    return exit_code, receipt


def test_multiprocessing_helpers_keep_off_the_start_directory(start_directory, tmp_path):
    """uvicorn's --reload worker and the resource tracker behind it are fresh interpreters.

    Both are started with `-c` and inherit the working directory, so they look for
    `multiprocessing` itself where the server was started.
    """
    marker = _plant_multiprocessing(start_directory)

    exit_code, receipt = _run_helper(
        "import multiprocessing, os, pathlib\n"
        "\n"
        "def _receipt():\n"
        "    pathlib.Path(os.environ['SEARCH_PATH_TEST_RECEIPT']).write_text('ran\\n')\n"
        "\n"
        "if __name__ == '__main__':\n"
        "    child = multiprocessing.get_context('spawn').Process(target=_receipt)\n"
        "    child.start()\n"
        "    child.join()\n",
        start_directory, tmp_path, "spawner",
    )

    assert not marker.exists(), "an interpreter started by the worker ran the planted package"
    # Without this the test would also pass if no second interpreter had been started at all.
    assert receipt.exists(), "the spawned interpreter never ran"
    assert exit_code == 0, f"worker exited {exit_code}"


def test_an_interpreter_started_directly_keeps_off_the_start_directory(start_directory, tmp_path):
    """A library that starts `sys.executable -c …` itself gets no interpreter flags from us.

    multiprocessing reproduces them; a plain subprocess call does not, and reads the
    environment instead.
    """
    marker = _plant_multiprocessing(start_directory)

    exit_code, receipt = _run_helper(
        "import os, subprocess, sys\n"
        "\n"
        "if __name__ == '__main__':\n"
        "    code = ('import multiprocessing, os, pathlib; '\n"
        "            \"pathlib.Path(os.environ['SEARCH_PATH_TEST_RECEIPT']).write_text('ran')\")\n"
        "    raise SystemExit(subprocess.run([sys.executable, '-c', code]).returncode)\n",
        start_directory, tmp_path, "starter",
    )

    assert not marker.exists(), "an interpreter started by the worker ran the planted package"
    assert receipt.exists(), "the started interpreter never ran"
    assert exit_code == 0, f"worker exited {exit_code}"


def test_every_command_keeps_its_own_helper_interpreters_off_it(start_directory, tmp_path):
    """`mlxk run --audio` starts one: Whisper draws a progress bar, tqdm takes a
    multiprocessing lock for it, and that starts the resource tracker - a `python -c …` child
    in the directory the command was started in. The command's own path is fine, a console
    script has its own directory first, so what is tested here is what it hands down.

    The stand-in is a script, like the console script it stands in for: its directory is on the
    path, the start directory is not.
    """
    marker = _plant_multiprocessing(start_directory)
    receipt = tmp_path / "receipt-cli"
    driver_dir = tmp_path / "driver"
    driver_dir.mkdir()
    driver = driver_dir / "driver.py"
    driver.write_text(
        "import os, pathlib\n"
        "import mlxk2.cli                             # the bootstrap under test\n"
        "import multiprocessing\n"
        "from multiprocessing import resource_tracker\n"
        "multiprocessing.RLock()                      # this is what starts the tracker\n"
        "print(resource_tracker._resource_tracker._pid)\n"
        "pathlib.Path(os.environ['SEARCH_PATH_TEST_RECEIPT']).write_text('ran')\n"
    )
    env = {k: v for k, v in os.environ.items() if k != "PYTHONSAFEPATH"}
    env["SEARCH_PATH_TEST_RECEIPT"] = str(receipt)

    done = subprocess.run(
        [sys.executable, str(driver)],
        cwd=start_directory, env=env, capture_output=True, text=True, timeout=CHILD_TIMEOUT,
    )
    assert done.returncode == 0, done.stderr
    assert receipt.exists(), "the driver never got as far as the lock"

    # The tracker does its imports after it is forked off, so read the marker only once that
    # process is gone - otherwise a clean run is indistinguishable from a late one.
    tracker = int(done.stdout.split()[0])
    deadline = time.time() + 30
    while time.time() < deadline:
        try:
            os.kill(tracker, 0)
        except ProcessLookupError:
            break
        time.sleep(0.05)
    else:
        pytest.fail(f"resource tracker {tracker} outlived its parent")

    assert not marker.exists(), "an interpreter the command started ran the planted package"


def test_the_root_handed_over_is_the_running_package(tmp_path, monkeypatch):
    """The path the worker is told to trust is where this mlxk2 was imported from."""
    captured = {}

    def fake_popen(cmd, env=None, **kw):
        captured["cmd"] = cmd
        raise RuntimeError("not started on purpose")

    monkeypatch.setattr(serve_mod.subprocess, "Popen", fake_popen)
    with pytest.raises(RuntimeError):
        serve_mod._run_supervised_uvicorn("127.0.0.1", 8000, "warning")

    handed_over = Path(captured["cmd"][-1])
    assert handed_over.samefile(Path(mlxk2.__file__).parent.parent)
