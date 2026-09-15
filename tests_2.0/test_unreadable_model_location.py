"""An unreadable model location ends in the error that stopped the command.

A name is resolved against two roots, and either can refuse to be read. With the model cache
unreadable, `run` reported one of its own variables: the pre-flight swallowed the resolver's
error, and the lines after it read names it had never bound — or, with an image attached, it
claimed the model was not found. With the workspace home unreadable, the HF_HOME bootstrap
raised a traceback — it runs when the CLI is imported, before any error handling exists.

The real entry point runs in a subprocess, because only a fresh import exercises the bootstrap.
No model is needed: resolution fails before anything would load.
"""

import os
import subprocess
import sys
from pathlib import Path

import pytest

_IMAGE = Path(__file__).parent / "assets" / "T1.png"


@pytest.mark.parametrize(
    "root, media",
    [("cache", []), ("workspace-home", []), ("cache", ["--image", str(_IMAGE)])],
    ids=["cache", "workspace-home", "cache-image"],
)
def test_run_reports_the_unreadable_location(tmp_path, root, media):
    env = {k: v for k, v in os.environ.items() if not k.startswith(("MLXK", "HF_"))}
    env["HF_HOME"] = str(tmp_path / "hf")
    if root == "cache":
        locked = tmp_path / "hf" / "hub"
    else:
        locked = tmp_path / "workspaces"
        env["MLXK_WORKSPACE_HOME"] = str(locked)
    locked.mkdir(parents=True)
    locked.chmod(0)

    try:
        if os.access(locked, os.R_OK):
            pytest.skip("a mode of 000 does not stop this user from reading (root?)")
        done = subprocess.run(
            [sys.executable, "-m", "mlxk2.cli", "run", "nosuchmodel", "hi", *media],
            text=True, capture_output=True, env=env, timeout=120,
        )
    finally:
        locked.chmod(0o700)
    output = done.stdout + done.stderr

    assert done.returncode != 0, output
    assert "Traceback" not in output, output
    assert "local variable" not in output, output
    assert "Permission denied" in output, output
