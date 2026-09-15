"""Regression tests for issue #70: an empty model name is not a search pattern.

`resolve_model_for_operation` matched workspace directories and cached models by
case-insensitive substring. `"" in name` holds for every name, so an empty spec
resolved to whatever `iterdir()` yielded first — in filesystem order, and without
telling the caller which model answered.

The environments below are the ones in which the defect bit; the existing edge-case
test could not catch it because it deliberately clears MLXK_WORKSPACE_HOME. Each test
therefore SETS the environment instead of removing it.
"""

import pytest

from mlxk2.cli import _resolve_workspace_for_bootstrap
from mlxk2.core.model_resolution import resolve_model_for_operation
from mlxk2.operations.rm import rm_operation
from mlxk2.operations.workspace import is_workspace_path

# Blank specs resolve to nothing at all; they never carry a revision, so the whole
# tuple is pinned.
BLANK_SPECS = ["", " ", "  ", "\t", "\n", "   \t\n   "]

# The other door: a revision with no name in front of it. The pattern that reaches the
# cache is empty here too. The tuple echoes the revision back, as it does for any @hash
# that fails to resolve, so these pin "no model was selected" rather than the tuple.
HASH_ONLY_SPECS = ["@abc", "@a", "@", "@a@b"]

EMPTY_NAME_SPECS = BLANK_SPECS + HASH_ONLY_SPECS


def assert_nothing_resolved(spec):
    resolved, _, ambiguous = resolve_model_for_operation(spec)
    assert resolved is None, f"{spec!r} selected {resolved!r}"
    assert ambiguous == [], f"{spec!r} came back as ambiguous: {ambiguous!r}"


def _make_model_dir(path):
    """Minimal on-disk shape that is_workspace_path() accepts as a model."""
    path.mkdir(parents=True, exist_ok=True)
    (path / "config.json").write_text('{"model_type": "llama"}')
    (path / "model.safetensors").write_bytes(b"\x00" * 64)
    return path


def _make_cached_model(hub, hf_name, commit_hash="abc123def456"):
    """Minimal HF cache entry: models--org--name/snapshots/<hash>/."""
    from mlxk2.core.cache import hf_to_cache_dir

    snapshot = hub / hf_to_cache_dir(hf_name) / "snapshots" / commit_hash
    return _make_model_dir(snapshot)


@pytest.fixture
def ws_home(tmp_path, monkeypatch):
    """Workspace home with three models, one of them named with a space.

    The space matters: whitespace-only specs look safe only as long as no directory
    name contains whitespace, so a space-free fixture would not pin the behaviour.
    """
    home = tmp_path / "workspaces"
    for name in ("zebra-model", "alpha-model", "my model"):
        _make_model_dir(home / name)
    monkeypatch.setenv("MLXK_WORKSPACE_HOME", str(home))
    monkeypatch.setenv("HF_HOME", str(tmp_path / "empty-cache"))
    return home


@pytest.fixture
def cache_home(tmp_path, monkeypatch):
    """HF cache factory with no workspace home at all.

    #70 was filed as conditional on MLXK_WORKSPACE_HOME. It is not: the cache branch
    reports ambiguity only for two or more matches, so a cache holding exactly ONE
    model resolved the empty spec to that model.
    """
    monkeypatch.delenv("MLXK_WORKSPACE_HOME", raising=False)
    monkeypatch.setenv("HF_HOME", str(tmp_path / "cache"))
    hub = tmp_path / "cache" / "hub"

    def _build(*hf_names):
        for hf_name in hf_names:
            _make_cached_model(hub, hf_name)
        return hub

    return _build


class TestEmptyNameIsRefused:
    """The guard: every empty-name spec resolves to nothing, in every environment."""

    @pytest.mark.parametrize("spec", EMPTY_NAME_SPECS)
    def test_workspace_home_with_models(self, spec, ws_home):
        assert_nothing_resolved(spec)

    @pytest.mark.parametrize("spec", EMPTY_NAME_SPECS)
    def test_workspace_home_is_itself_a_model(self, spec, tmp_path, monkeypatch):
        # `workspace_home / ""` is workspace_home itself, so this reaches the exact-match
        # branch rather than the substring loop — a guard placed after it would miss this.
        home = _make_model_dir(tmp_path / "solo-workspace")
        monkeypatch.setenv("MLXK_WORKSPACE_HOME", str(home))
        monkeypatch.setenv("HF_HOME", str(tmp_path / "empty-cache"))
        assert_nothing_resolved(spec)

    @pytest.mark.parametrize("spec", EMPTY_NAME_SPECS)
    def test_cache_holding_exactly_one_model(self, spec, cache_home):
        cache_home("mlx-community/Solo-4bit")
        assert_nothing_resolved(spec)

    @pytest.mark.parametrize("spec", EMPTY_NAME_SPECS)
    def test_cache_holding_several_models(self, spec, cache_home):
        cache_home("mlx-community/Alpha-4bit", "mlx-community/Beta-4bit")
        # Not merely "no model": the whole cache must not come back as an ambiguity list.
        assert_nothing_resolved(spec)

    @pytest.mark.parametrize("spec", BLANK_SPECS)
    def test_blank_specs_carry_no_revision(self, spec, ws_home):
        assert resolve_model_for_operation(spec) == (None, None, [])

    def test_hash_prefix_does_not_pick_a_model(self, cache_home):
        # "@abc" reached find_model_by_hash("", "abc"), which patterns over every cached
        # model and returns the first whose snapshot starts with the prefix.
        cache_home("mlx-community/Alpha-4bit")
        assert_nothing_resolved("@abc123def456")

    def test_non_string_specs_do_not_raise(self, ws_home):
        # These used to raise TypeError out of the workspace join or the "@" membership test.
        for spec in (None, 0, [], b""):
            assert resolve_model_for_operation(spec) == (None, None, [])


class TestGuardDoesNotOverreach:
    """What must keep working: the guard rejects empty names, nothing else."""

    def test_exact_workspace_name_resolves(self, ws_home):
        resolved, _, _ = resolve_model_for_operation("zebra-model")
        assert resolved == str(ws_home / "zebra-model")

    def test_partial_workspace_name_still_resolves(self, ws_home):
        resolved, _, _ = resolve_model_for_operation("zebra")
        assert resolved == str(ws_home / "zebra-model")

    def test_workspace_name_starting_with_at_still_resolves(self, tmp_path, monkeypatch):
        # The guard must not read "@" in a workspace name as a revision separator: the
        # workspace branch matches directory names literally and never split on "@".
        home = tmp_path / "workspaces"
        _make_model_dir(home / "@local-qwen")
        monkeypatch.setenv("MLXK_WORKSPACE_HOME", str(home))
        monkeypatch.setenv("HF_HOME", str(tmp_path / "empty-cache"))
        resolved, _, _ = resolve_model_for_operation("@local-qwen")
        assert resolved == str(home / "@local-qwen")

    def test_unknown_name_is_unchanged(self, ws_home):
        assert resolve_model_for_operation("nonexistent") == (None, None, [])

    def test_cached_model_resolves(self, cache_home):
        cache_home("mlx-community/Solo-4bit")
        resolved, _, _ = resolve_model_for_operation("mlx-community/Solo-4bit")
        assert resolved == "mlx-community/Solo-4bit"

    def test_explicit_path_resolves(self, tmp_path, monkeypatch):
        model = _make_model_dir(tmp_path / "explicit-model")
        monkeypatch.delenv("MLXK_WORKSPACE_HOME", raising=False)
        monkeypatch.chdir(model)
        resolved, _, _ = resolve_model_for_operation(".")
        assert resolved == str(model.resolve())


class TestEmptyNameOutsideTheResolver:
    """Two paths that reach the same defect without going through the resolver."""

    def test_empty_path_is_not_the_working_directory(self, tmp_path, monkeypatch):
        # Path("") is Path("."), so callers falling back to the raw spec asked whether the
        # *working directory* was a model — `mlxk run ""` inside a model dir loaded it.
        model = _make_model_dir(tmp_path / "cwd-model")
        # A directory really named "   ", otherwise the whitespace assertion below holds
        # simply because the path does not exist, with or without the guard.
        _make_model_dir(model / "   ")
        monkeypatch.chdir(model)
        assert is_workspace_path("") is False
        assert is_workspace_path("   ") is False
        # The explicit spellings of "the working directory" keep working.
        assert is_workspace_path(".") is True
        assert is_workspace_path("./") is True

    def test_bootstrap_does_not_redirect_hf_home(self, ws_home):
        # cli.py carries its own copy of the substring loop, running before any import so
        # it can point HF_HOME at a workspace's .hf_cache. It matched on "" as well.
        assert _resolve_workspace_for_bootstrap("") is None
        # One space, not three: only a single space is a substring of "my model", so the
        # three-space case alone would pass with the whitespace half of the guard removed.
        assert _resolve_workspace_for_bootstrap(" ") is None
        assert _resolve_workspace_for_bootstrap("   ") is None
        assert _resolve_workspace_for_bootstrap("zebra") == ws_home / "zebra-model"


class TestEmptyNameDoesNotDelete:
    """`rm` is where reading an empty name as a pattern cost data, not just honesty."""

    def test_rm_refuses_empty_spec_with_one_cached_model(self, cache_home):
        hub = cache_home("mlx-community/Solo-4bit")
        result = rm_operation("", force=True)
        assert result["status"] == "error"
        assert result["error"]["type"] == "model_not_found"
        assert list(hub.iterdir()), "rm deleted a model for an empty spec"

    def test_rm_refuses_empty_spec_in_workspace_home(self, ws_home, tmp_path, monkeypatch):
        # The cache has to EXIST: rm_operation returns cache_not_found before it reaches
        # the resolver at all, which would make every assertion here hold vacuously.
        hub = tmp_path / "cache" / "hub"
        _make_cached_model(hub, "mlx-community/Solo-4bit")
        monkeypatch.setenv("HF_HOME", str(tmp_path / "cache"))

        result = rm_operation("", force=True)
        assert result["status"] == "error"
        assert result["error"]["type"] == "model_not_found"
        assert sorted(p.name for p in ws_home.iterdir()) == [
            "alpha-model",
            "my model",
            "zebra-model",
        ]
        assert list(hub.iterdir()), "rm deleted a cached model for an empty spec"


def _working_directory_with_files(root, blank):
    """A working directory that holds files — and, for a blank spec, a directory really named
    after it, so the spec fails on the guard rather than on a path that does not exist."""
    cwd = root / "cwd"
    cwd.mkdir()
    (cwd / "notes.md").write_text("private")
    if blank.strip() == "" and blank:
        (cwd / blank).mkdir()
        (cwd / blank / "notes.md").write_text("private")
    return cwd


class TestEmptyPathDoesNotPushOrConvert:
    """`push` and `convert` never call the resolver, so the guard above does not reach them:
    they turn the path into `Path("")`, which is the working directory."""

    @pytest.mark.parametrize("spec", ["", "   "])
    def test_push_does_not_take_the_working_directory(self, spec, tmp_path, monkeypatch):
        from mlxk2.operations.push import push_operation

        monkeypatch.chdir(_working_directory_with_files(tmp_path, spec))
        # Check-only never contacts the Hub, and reports the folder it would have uploaded.
        result = push_operation(spec, "org/repo", check_only=True)
        assert result["status"] == "error"
        assert result["error"]["type"] == "ValidationError"
        assert result["data"]["local_files_count"] is None, "push counted the working directory"

    def test_push_never_reaches_the_hub(self, tmp_path, monkeypatch):
        import sys
        import types
        from unittest.mock import MagicMock

        from mlxk2.operations.push import push_operation

        monkeypatch.chdir(_working_directory_with_files(tmp_path, ""))
        monkeypatch.setenv("HF_TOKEN", "not-a-token")
        # The hub is replaced wholesale, so nothing can be uploaded even with the guard gone.
        hub = types.ModuleType("huggingface_hub")
        hub.HfApi, hub.upload_folder = MagicMock(), MagicMock()
        errors = types.ModuleType("huggingface_hub.errors")
        errors.HfHubHTTPError = errors.RepositoryNotFoundError = errors.RevisionNotFoundError = type(
            "HubError", (Exception,), {})
        monkeypatch.setitem(sys.modules, "huggingface_hub", hub)
        monkeypatch.setitem(sys.modules, "huggingface_hub.errors", errors)

        result = push_operation("", "org/repo")
        # Unguarded, this called upload_folder(folder_path=".") and reported success.
        hub.upload_folder.assert_not_called()
        hub.HfApi.assert_not_called()
        assert result["error"]["type"] == "ValidationError"

    @pytest.mark.parametrize("spec", ["", "   "])
    def test_convert_does_not_read_the_working_directory(self, spec, tmp_path, monkeypatch):
        from mlxk2.operations.convert import convert_operation

        monkeypatch.chdir(_working_directory_with_files(tmp_path, spec))
        target = tmp_path / "out"
        result = convert_operation(spec, str(target), "repair-index")
        assert result["error"]["type"] == "ValidationError"
        assert not target.exists(), "convert wrote a workspace from the working directory"

    @pytest.mark.parametrize("spec", ["", "   "])
    def test_convert_does_not_write_into_the_working_directory(self, spec, tmp_path, monkeypatch):
        from mlxk2.operations.convert import convert_operation

        source = _make_model_dir(tmp_path / "src")
        cwd = tmp_path / "empty-cwd"
        cwd.mkdir()  # repair-index accepts an empty target, so this is the case that wrote
        monkeypatch.chdir(cwd)
        result = convert_operation(str(source), spec, "repair-index")
        assert result["error"]["type"] == "ValidationError"
        assert list(cwd.iterdir()) == [], "convert wrote into the working directory"

    def test_convert_cli_does_not_read_an_empty_name_as_the_workspace_home(self, tmp_path):
        # The CLI resolves bare names into MLXK_WORKSPACE_HOME before the operation runs, and
        # `home / ""` is the home itself — a guard in the operation alone would never see "".
        import json
        import os
        import subprocess
        import sys
        from pathlib import Path

        home = tmp_path / "workspaces"
        _make_model_dir(home / "zebra-model")
        target = tmp_path / "out"
        env = {k: v for k, v in os.environ.items() if not k.startswith(("MLXK", "HF_"))}
        env.update(MLXK_WORKSPACE_HOME=str(home), HF_HOME=str(tmp_path / "hf"),
                   PYTHONPATH=str(Path(__file__).resolve().parents[1]))
        done = subprocess.run(
            [sys.executable, "-m", "mlxk2.cli", "convert", "", str(target), "--repair-index", "--json"],
            text=True, capture_output=True, env=env, cwd=tmp_path, timeout=120,
        )
        result = json.loads(done.stdout)
        assert result["error"]["type"] == "ValidationError", done.stdout + done.stderr
        assert not target.exists(), "convert took the workspace home as its source"
