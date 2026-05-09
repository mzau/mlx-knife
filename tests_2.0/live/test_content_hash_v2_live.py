"""Live tests for content_hash v2 (ADR-025) — replaces SMOKE-TEST-2.0.6.md §A1–A10 + §B1–B6.

Migrates manual smoke tests for the content_hash v2 algorithm + sentinel format
into automated pytest. Opt-in via MLXK2_LIVE_CHV2=1.

Required env (for opt-in):
- MLXK2_LIVE_CHV2=1                     enable
- HF_TOKEN                              for HF access (or models pre-cached)
- MLXK2_LIVE_CHV2_TEXT_MODEL            default: mlx-community/Llama-3.2-1B-Instruct-4bit
- MLXK2_LIVE_CHV2_VISION_MODEL          default: mlx-community/Llama-3.2-11B-Vision-Instruct-4bit
                                        (only for §A5/§B; quantized 4bit ~6 GB, multi-shard)

The vision fixture is gated against full-precision (bf16/fp16/fp32) targets to
avoid accidental multi-tens-of-GB downloads — those models are smoke-test-only
material, not routine CI fodder. The lightweight A1–A4 / A8–A10 set runs without
the vision model.

Cleanup-on-Ctrl-C / re-runnability:
- All test workspaces live inside a per-session tmp dir (pytest's auto-rotated
  tmp_path_factory). Pytest rotates these across sessions, so an aborted run
  is cleaned on the next run automatically.
- TEST_PREFIX="mlxk-test-chv2-" guards every workspace name.
- _safe_rmtree() refuses to delete dirs that don't carry the prefix — defensive
  guard against pointing at a real model by accident.
- MLXK_WORKSPACE_HOME is monkeypatched to the per-session tmp dir; the user's
  real workspace home is never touched.

CLI invocation: subprocess against the installed `mlxk` entry point. This
avoids capsys-vs-fixture-scope friction (capsys is function-scoped only),
keeps env-isolation clean, and exercises the real binary path.

Run:
    MLXK2_LIVE_CHV2=1 HF_TOKEN=... pytest tests_2.0/live/test_content_hash_v2_live.py -v -m live_chv2
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
from pathlib import Path
from typing import Any, Dict, Tuple

import pytest


TEST_PREFIX = "mlxk-test-chv2-"

V2_PATTERN = re.compile(r"^sha256:[0-9a-f]{64}$")
V2_OR_V1_PATTERN = re.compile(r"^(sha256:[0-9a-f]{64}|[0-9a-f]{64})$")


# Module-level skip: test file is skipped entirely unless explicitly opted in.
live_enabled = os.environ.get("MLXK2_LIVE_CHV2") == "1"
hf_token_present = bool(os.environ.get("HF_TOKEN"))

TEXT_MODEL = os.environ.get(
    "MLXK2_LIVE_CHV2_TEXT_MODEL",
    "mlx-community/Llama-3.2-1B-Instruct-4bit",
)
VISION_MODEL = os.environ.get(
    "MLXK2_LIVE_CHV2_VISION_MODEL",
    "mlx-community/Llama-3.2-11B-Vision-Instruct-4bit",
)

# Hard-gate against accidental full-precision downloads.
# bf16/fp16/fp32 vision models are 20+ GB; not appropriate for routine CI.
_FULL_PRECISION_PATTERN = re.compile(r"-(bf16|fp16|fp32)\b", re.IGNORECASE)

pytestmark = [
    pytest.mark.live,
    pytest.mark.live_chv2,
    pytest.mark.skipif(
        not (live_enabled and hf_token_present),
        reason="Live chv2 tests disabled. Set MLXK2_LIVE_CHV2=1 + HF_TOKEN.",
    ),
]


# =========================================================================
# Helpers
# =========================================================================


def _safe_rmtree(path: Path) -> None:
    """Remove a directory only if its name carries TEST_PREFIX.

    Defensive guard: refuses to touch real models, even if a caller passes a
    path outside the test sandbox by mistake.
    """
    if path.exists() and path.name.startswith(TEST_PREFIX):
        shutil.rmtree(path)


def _run_mlxk(argv: list[str], env_overrides: Dict[str, str] = None) -> Tuple[str, str, int]:
    """Run the installed `mlxk` CLI in a subprocess; return (stdout, stderr, exit_code).

    Inherits the current process env (including MLXK_WORKSPACE_HOME set by the
    auto-applied fixture). Optional env_overrides extend that.
    """
    env = os.environ.copy()
    if env_overrides:
        env.update(env_overrides)
    proc = subprocess.run(
        ["mlxk"] + argv,
        capture_output=True,
        text=True,
        env=env,
    )
    return proc.stdout, proc.stderr, proc.returncode


def _load_sentinel(workspace: Path) -> Dict[str, Any]:
    return json.loads((workspace / ".mlxk_workspace.json").read_text())


def _ensure_clean(workspace: Path) -> None:
    """Pre-test cleanup: remove the named workspace if a previous run left it."""
    _safe_rmtree(workspace)


# =========================================================================
# Fixtures
# =========================================================================


@pytest.fixture(scope="session")
def chv2_workspace_home(tmp_path_factory) -> Path:
    """Session-scoped MLXK_WORKSPACE_HOME under pytest's auto-rotated tmp.

    All test workspaces created in this file land here. Pytest rotates
    /tmp/pytest-of-USER/ across sessions, so an aborted run gets cleaned
    on the next session automatically.
    """
    return tmp_path_factory.mktemp("chv2_workspace_home")


@pytest.fixture(autouse=True)
def _set_workspace_home(chv2_workspace_home, monkeypatch):
    """Auto-applied: every test in this file sees MLXK_WORKSPACE_HOME → tmp."""
    monkeypatch.setenv("MLXK_WORKSPACE_HOME", str(chv2_workspace_home))


@pytest.fixture(scope="session")
def fresh_v2_workspace(chv2_workspace_home) -> Path:
    """Clone TEXT_MODEL once per session into a v2 workspace.

    Reused across A1–A4 + A8–A10 to avoid re-cloning. Tests treat this
    workspace as read-only; mutating tests must operate on copies.
    """
    workspace_name = f"{TEST_PREFIX}fresh-text"
    workspace_path = chv2_workspace_home / workspace_name

    _ensure_clean(workspace_path)

    # subprocess inherits MLXK_WORKSPACE_HOME from the test process env
    env = {"MLXK_WORKSPACE_HOME": str(chv2_workspace_home)}
    stdout, stderr, code = _run_mlxk(
        ["clone", TEXT_MODEL, workspace_name, "--json"],
        env_overrides=env,
    )
    if code != 0:
        pytest.skip(f"Clone of {TEXT_MODEL} failed: {stderr or stdout}")

    if not (workspace_path / ".mlxk_workspace.json").exists():
        pytest.skip(f"Workspace not created at {workspace_path}")

    yield workspace_path

    _safe_rmtree(workspace_path)


@pytest.fixture(scope="session")
def broken_multishard_workspace(chv2_workspace_home) -> Path:
    """Clone VISION_MODEL, copy to *-broken, remove index → §B-Prep equivalent.

    Heavy fixture (vision model clone). Tests using it skip cleanly if the
    clone fails or the model turns out to be single-shard (no index.json).
    """
    # Gate: refuse to clone full-precision vision models (20+ GB territory).
    if _FULL_PRECISION_PATTERN.search(VISION_MODEL):
        pytest.skip(
            f"Refusing to clone full-precision vision model {VISION_MODEL!r}. "
            f"Set MLXK2_LIVE_CHV2_VISION_MODEL to a quantized variant "
            f"(e.g. -4bit, -8bit) to keep CI fast."
        )

    source_name = f"{TEST_PREFIX}vision-source"
    broken_name = f"{TEST_PREFIX}vision-broken"
    source_path = chv2_workspace_home / source_name
    broken_path = chv2_workspace_home / broken_name

    _ensure_clean(source_path)
    _ensure_clean(broken_path)

    env = {"MLXK_WORKSPACE_HOME": str(chv2_workspace_home)}
    stdout, stderr, code = _run_mlxk(
        ["clone", VISION_MODEL, source_name, "--json"],
        env_overrides=env,
    )
    if code != 0:
        pytest.skip(f"Vision clone of {VISION_MODEL} failed: {stderr or stdout}")

    index_file = source_path / "model.safetensors.index.json"
    if not index_file.exists():
        pytest.skip(
            f"Vision model {VISION_MODEL} is single-shard (no index.json) "
            f"— not a valid --repair-index target. "
            f"Set MLXK2_LIVE_CHV2_VISION_MODEL to a multi-shard model."
        )

    # Build broken twin via copytree + rm index (cp -cR analogue)
    shutil.copytree(source_path, broken_path, copy_function=shutil.copy2)
    (broken_path / "model.safetensors.index.json").unlink()

    yield broken_path

    _safe_rmtree(source_path)
    _safe_rmtree(broken_path)


# =========================================================================
# §A — Fresh Clones & Converts
# =========================================================================


def test_a1_fresh_clone_writes_v2_sentinel(fresh_v2_workspace):
    """A1: fresh clone writes hash_algorithm=v2 in sentinel."""
    sentinel = _load_sentinel(fresh_v2_workspace)
    assert sentinel.get("hash_algorithm") == "v2", (
        f"expected hash_algorithm=v2, got {sentinel.get('hash_algorithm')!r}"
    )


def test_a2_file_index_covers_files(fresh_v2_workspace):
    """A2: file_index has > 5 entries (config + tokenizer + safetensors + …)."""
    sentinel = _load_sentinel(fresh_v2_workspace)
    assert len(sentinel.get("file_index", [])) > 5


def test_a3_file_index_includes_safetensors_header(fresh_v2_workspace):
    """A3: at least one file_index entry is *.safetensors with sha/size/mtime_ns."""
    sentinel = _load_sentinel(fresh_v2_workspace)
    safetensors_entries = [
        e for e in sentinel.get("file_index", [])
        if e.get("path", "").endswith(".safetensors")
    ]
    assert safetensors_entries, "no .safetensors entry in file_index"
    entry = safetensors_entries[0]
    for field in ("sha", "size", "mtime_ns"):
        assert field in entry, f"missing {field} on safetensors entry"
    assert V2_PATTERN.match(entry["sha"]), (
        f"safetensors sha not in v2 format: {entry['sha']!r}"
    )


def test_a4_clean_after_fresh_clone(fresh_v2_workspace):
    """A4: mlxk show reports Clean: ✓ on a freshly cloned workspace."""
    stdout, stderr, code = _run_mlxk(["show", str(fresh_v2_workspace)])
    assert code == 0, f"show failed: {stderr or stdout}"
    assert "Clean: ✓" in stdout, f"expected 'Clean: ✓' in show output:\n{stdout}"


def test_a5_convert_repair_index_writes_v2(broken_multishard_workspace, chv2_workspace_home):
    """A5: mlxk convert --repair-index writes a v2 sentinel into the target."""
    target_name = f"{TEST_PREFIX}a5-repaired"
    target_path = chv2_workspace_home / target_name
    _ensure_clean(target_path)

    try:
        stdout, stderr, code = _run_mlxk(
            ["convert", str(broken_multishard_workspace), target_name,
             "--repair-index", "--json"],
        )
        assert code == 0, f"convert --repair-index failed: {stderr or stdout}"

        sentinel = _load_sentinel(target_path)
        assert sentinel.get("hash_algorithm") == "v2"
        assert V2_PATTERN.match(sentinel.get("content_hash", "")), (
            f"target content_hash not v2-format: {sentinel.get('content_hash')!r}"
        )
    finally:
        _safe_rmtree(target_path)


def test_a8_list_hash_column_strips_prefix(fresh_v2_workspace):
    """A8: mlxk list hash column shows 7-hex digest, not literal 'sha256:'."""
    stdout, stderr, code = _run_mlxk(["list", str(fresh_v2_workspace)])
    assert code == 0, f"list failed: {stderr or stdout}"

    workspace_line = next(
        (line for line in stdout.splitlines() if fresh_v2_workspace.name in line),
        None,
    )
    assert workspace_line is not None, f"workspace not found in list output:\n{stdout}"
    assert "sha256:" not in workspace_line, (
        f"raw 'sha256:' prefix leaked into list output:\n{workspace_line}"
    )
    assert re.search(r"\b[0-9a-f]{7}\b", workspace_line), (
        f"no 7-hex digest in list line:\n{workspace_line}"
    )


def test_a9_list_json_content_hash_pattern(fresh_v2_workspace):
    """A9: mlxk list --json content_hash matches v2 pattern (sha256:<64-hex>)."""
    stdout, stderr, code = _run_mlxk(["list", str(fresh_v2_workspace), "--json"])
    assert code == 0, f"list --json failed: {stderr or stdout}"

    payload = json.loads(stdout)
    models = payload.get("data", {}).get("models", [])
    assert models, f"no models in list output:\n{stdout}"

    content_hash = models[0].get("content_hash")
    assert content_hash, "content_hash missing from list --json output"
    assert V2_PATTERN.match(content_hash), (
        f"content_hash does not match v2 pattern: {content_hash!r}"
    )


def test_a10_show_long_form_prefix_plus_16_hex(fresh_v2_workspace):
    """A10: mlxk show Content-Hash line is 'sha256:<16-hex>...' (prefix + 16 hex)."""
    stdout, stderr, code = _run_mlxk(["show", str(fresh_v2_workspace)])
    assert code == 0, f"show failed: {stderr or stdout}"

    hash_line = next(
        (line for line in stdout.splitlines() if "Content Hash:" in line),
        None,
    )
    assert hash_line is not None, f"'Content Hash:' line missing in show output:\n{stdout}"
    assert re.search(r"Content Hash:\s+sha256:[0-9a-f]{16}\.\.\.", hash_line), (
        f"show hash line doesn't match 'sha256:<16-hex>...': {hash_line!r}"
    )


# =========================================================================
# §B — Coverage Gap Regression (Issue #52)
# =========================================================================


def test_b1_to_b6_repair_creates_different_hash(
    broken_multishard_workspace, chv2_workspace_home
):
    """§B1–§B6 condensed: --repair-index produces a target whose content_hash
    differs from the source's, and the target is itself clean.

    This is the closing condition for Issue #52: pre-fix, source and target
    had identical content_hash because v1 didn't include the index file.
    """
    target_name = f"{TEST_PREFIX}b-repaired"
    target_path = chv2_workspace_home / target_name
    _ensure_clean(target_path)

    try:
        # B1: repair-index convert
        stdout, stderr, code = _run_mlxk(
            ["convert", str(broken_multishard_workspace), target_name,
             "--repair-index", "--json"],
        )
        assert code == 0, f"convert --repair-index failed: {stderr or stdout}"

        # B2/B3: capture both sentinels
        source_hash = _load_sentinel(broken_multishard_workspace).get("content_hash")
        target_hash = _load_sentinel(target_path).get("content_hash")

        assert V2_OR_V1_PATTERN.match(source_hash or ""), (
            f"source content_hash invalid format: {source_hash!r}"
        )
        assert V2_PATTERN.match(target_hash or ""), (
            f"target content_hash not v2-format: {target_hash!r}"
        )

        # B4: hashes must differ (the regression assertion)
        assert source_hash != target_hash, (
            f"source and target hashes identical — Issue #52 regression!\n"
            f"  source: {source_hash}\n  target: {target_hash}"
        )

        # B5: repaired workspace shows Clean: ✓
        stdout, stderr, code = _run_mlxk(["show", str(target_path)])
        assert code == 0, f"show on repaired ws failed: {stderr or stdout}"
        assert "Clean: ✓" in stdout, f"repaired ws not clean:\n{stdout}"

        # B6: manual file edit on a copy → Clean: ✗ (isolated, no source pollution)
        edit_target_name = f"{TEST_PREFIX}b6-edited"
        edit_target_path = chv2_workspace_home / edit_target_name
        _ensure_clean(edit_target_path)
        try:
            shutil.copytree(target_path, edit_target_path, copy_function=shutil.copy2)
            with (edit_target_path / "config.json").open("a") as f:
                f.write('{"extra":1}')
            stdout, stderr, code = _run_mlxk(["show", str(edit_target_path)])
            assert code == 0, f"show on edited ws failed: {stderr or stdout}"
            assert "Clean: ✗" in stdout, f"edited ws should be dirty, got:\n{stdout}"
        finally:
            _safe_rmtree(edit_target_path)

    finally:
        _safe_rmtree(target_path)
