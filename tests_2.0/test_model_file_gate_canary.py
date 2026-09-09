"""Canary for the CVE-2026-5843 bridge (ADR-023).

A failure here is **not** a defect in mlx-knife. It means upstream moved, and the bridge in
`mlxk2/core/remote_code.py` needs re-reading — not deleting on reflex.

⚠ The bridge has two halves and they retire separately:

- **mlx-lm.** Upstream gated the `model_file` exec behind `trust_remote_code` on `main`
  (2026-06-11) and has released nothing since 2026-04-22. When a release finally carries the
  parameter, mlx-knife's guard becomes redundant *for this backend* — mlx-lm's own default is
  then the same refusal.
- **mlx-vlm.** The same exec lives in `get_model_and_args` with **no gate at all** — no
  parameter, no check, on every version that has the branch. Nothing upstream retires this
  half, so the guard stays as long as mlx-knife can route through mlx-vlm.

Deliberately source-based, like `test_audio_bridge_canary.py`: importing `mlx_lm` would pull
in the real `mlx.core`, which TESTING-DETAILS forbids inside the stub-collecting tree. Located
through the installed distribution rather than `find_spec`, because this tree puts its own
`mlx_lm` / `mlx_vlm` stubs on `sys.path` and a canary that reads a stub measures nothing.
"""

import ast
from importlib.metadata import PackageNotFoundError, distribution
from pathlib import Path

import pytest


def _upstream_source(dist_name: str, relative_path: str) -> str:
    """Read a file out of an installed distribution without importing it."""
    try:
        dist = distribution(dist_name)
    except PackageNotFoundError:
        pytest.skip(f"{dist_name} not installed")
    path = Path(dist.locate_file(relative_path))
    if not path.exists():
        pytest.fail(f"{dist_name} no longer ships {relative_path} — re-read the bridge")
    return path.read_text(encoding="utf-8")


def _function(source: str, name: str) -> ast.FunctionDef:
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return node
    pytest.fail(f"upstream no longer defines {name}() — re-read the bridge")


def _arg_names(fn: ast.FunctionDef) -> set:
    args = fn.args
    return {a.arg for a in (*args.posonlyargs, *args.args, *args.kwonlyargs)}


def test_mlx_lm_load_model_still_has_no_trust_gate():
    """Green means the shipped mlx-lm still executes `model_file` unconditionally.

    When this fails, the installed mlx-lm finally carries the CVE fix: its own default
    refuses, so the mlx-lm half of the guard is redundant. **The mlx-vlm half is not** — see
    the module docstring before removing anything.
    """
    fn = _function(_upstream_source("mlx-lm", "mlx_lm/utils.py"), "load_model")

    assert "trust_remote_code" not in _arg_names(fn), (
        "mlx_lm.utils.load_model now takes trust_remote_code — a release with the "
        "CVE-2026-5843 fix has landed. The mlx-lm half of the guard in "
        "mlxk2/core/remote_code.py is redundant; the mlx-vlm half still is not."
    )


def test_mlx_lm_still_reads_the_key_the_guard_checks():
    """The guard keys on `model_file`. If upstream renames it, the guard stops guarding."""
    source = _upstream_source("mlx-lm", "mlx_lm/utils.py")

    assert '"model_file"' in source or "'model_file'" in source, (
        "mlx_lm no longer mentions model_file — the key the guard checks may have been "
        "renamed or the mechanism removed; re-read mlxk2/core/remote_code.py"
    )


def test_mlx_vlm_half_is_still_ungated():
    """The half no upstream release retires.

    0.6.10 (the current pin) has no `model_file` branch at all, so this skips — that skip is
    the +1 in the tree's skip count. Once a bump brings the branch, this states what
    matters: it arrives without a gate.
    """
    source = _upstream_source("mlx-vlm", "mlx_vlm/utils.py")

    if "model_file" not in source:
        pytest.skip("pinned mlx-vlm has no model_file branch yet")

    fn = _function(source, "get_model_and_args")
    assert "trust_remote_code" not in _arg_names(fn), (
        "mlx_vlm.utils.get_model_and_args gained trust_remote_code — upstream grew its own "
        "gate; re-read whether mlx-knife's guard is still the only one"
    )
