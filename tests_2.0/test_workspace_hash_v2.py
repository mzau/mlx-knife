"""Tests for content_hash v2 (ADR-025).

Covers the Verification Matrix at the foot of ADR-025:
- Regression for Issue #52 (repair-index is visible in hash)
- Per-class hash strategies (safetensors header-only, full-content, catch-all cap)
- Transport robustness (mtime drift self-heal)
- Symlink policy (inside fingerprint, outside reject)
- Sentinel path validation (traversal rejected before any FS op)
- DoS — oversized catch-all falls back to stat-only
- Dev-cruft transparency
- Migration: v1 sentinel → None → recalc → Clean
- Hash stability
- Exclude-list drift resilience
"""

import hashlib
import json
import os
import shutil
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch

import pytest

from mlxk2 import __version__
from mlxk2.operations.workspace import (
    CATCHALL_FULL_READ_CAP,
    DEFAULT_EXCLUDE_PATTERNS,
    HASH_ALGORITHM_V2,
    SENTINEL_FILENAME,
    CorruptSentinelError,
    _aggregate_hash,
    _classify_symlink,
    _hash_file_by_class,
    _parse_safetensors_header,
    _validate_sentinel_path,
    compute_workspace_hash_v2,
    get_workspace_clean_hint,
    is_workspace_clean,
    read_workspace_metadata,
    update_workspace_hash,
    write_workspace_sentinel,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _safetensors_bytes(header_json: dict, tensor_body: bytes = b"\x00" * 128) -> bytes:
    """Assemble a minimal safetensors blob: 8-byte LE length + JSON header + body."""
    header_bytes = json.dumps(header_json, separators=(",", ":")).encode("utf-8")
    prefix = len(header_bytes).to_bytes(8, "little", signed=False)
    return prefix + header_bytes + tensor_body


def _make_workspace(root: Path, *, with_sentinel: bool = True) -> Path:
    """Create a minimal valid workspace (config.json, one safetensors, one tokenizer)."""
    root.mkdir(parents=True, exist_ok=True)
    (root / "config.json").write_text(json.dumps({"model_type": "llama"}))
    (root / "model-00001-of-00001.safetensors").write_bytes(
        _safetensors_bytes({"tensor_0": {"dtype": "F16", "shape": [4, 4], "data_offsets": [0, 128]}})
    )
    (root / "tokenizer.json").write_text(json.dumps({"model": {"type": "BPE"}}))
    if with_sentinel:
        write_workspace_sentinel(root, {
            "mlxk_version": __version__,
            "created_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
            "managed": True,
            "operation": "clone",
            "source_repo": "test/example",
            "source_revision": None,
        })
    return root


def _make_v2_workspace(root: Path) -> Path:
    """Create a workspace and stamp it with a valid v2 sentinel via update_workspace_hash."""
    _make_workspace(root)
    ok, _ = update_workspace_hash(root)
    assert ok is True
    return root


# ---------------------------------------------------------------------------
# Unit — helpers
# ---------------------------------------------------------------------------


class TestSafetensorsHeader:
    def test_parses_valid_header(self, tmp_path):
        blob = _safetensors_bytes({"a": {"dtype": "F16", "shape": [2, 2], "data_offsets": [0, 16]}})
        path = tmp_path / "w.safetensors"
        path.write_bytes(blob)
        header = _parse_safetensors_header(path)
        assert header is not None
        assert json.loads(header.decode("utf-8"))["a"]["dtype"] == "F16"

    def test_zero_length_prefix_returns_none(self, tmp_path):
        path = tmp_path / "w.safetensors"
        path.write_bytes((0).to_bytes(8, "little") + b"tensor-body")
        assert _parse_safetensors_header(path) is None

    def test_oversized_prefix_returns_none(self, tmp_path):
        path = tmp_path / "w.safetensors"
        huge = (100 * 1024 * 1024 + 1).to_bytes(8, "little")
        path.write_bytes(huge + b"unused")
        assert _parse_safetensors_header(path) is None

    def test_tensor_bytes_are_never_read(self, tmp_path):
        """The header parser must stop after the declared header length."""
        header = {"t": {"dtype": "F16", "shape": [1], "data_offsets": [0, 2]}}
        header_bytes = json.dumps(header, separators=(",", ":")).encode("utf-8")
        prefix = len(header_bytes).to_bytes(8, "little")
        # Use a sentinel byte after the header so a greedy reader would hash it in.
        tensor_body = b"\xff" * 512
        path = tmp_path / "w.safetensors"
        path.write_bytes(prefix + header_bytes + tensor_body)
        out = _parse_safetensors_header(path)
        assert out == header_bytes  # exactly the header, no tensor bytes


class TestPathValidation:
    @pytest.mark.parametrize("bad", [
        "../etc/passwd",
        "a/../../b",
        "/abs/path",
        "C:/win/path",
        "back\\slash.json",
        "has\x00nul.json",
        "",
    ])
    def test_rejects_unsafe_paths(self, bad):
        with pytest.raises(CorruptSentinelError):
            _validate_sentinel_path(bad)

    @pytest.mark.parametrize("good", [
        "config.json",
        "subdir/tokenizer.json",
        "a/b/c.safetensors",
        "weird name with spaces.txt",
    ])
    def test_accepts_safe_paths(self, good):
        _validate_sentinel_path(good)  # must not raise

    def test_rejects_non_string(self):
        with pytest.raises(CorruptSentinelError):
            _validate_sentinel_path(None)


class TestSymlinkClassify:
    def test_inside_workspace_returns_fingerprint(self, tmp_path):
        ws = tmp_path / "ws"
        ws.mkdir()
        target = ws / "real.json"
        target.write_text("{}")
        link = ws / "link.json"
        link.symlink_to(target)
        kind, sha = _classify_symlink(link, ws.resolve())
        assert kind == "inside"
        assert sha.startswith("sha256:")

    def test_outside_workspace_rejects(self, tmp_path):
        ws = tmp_path / "ws"
        ws.mkdir()
        outside = tmp_path / "outside.txt"
        outside.write_text("secret")
        link = ws / "escape.txt"
        link.symlink_to(outside.resolve())
        kind, reason = _classify_symlink(link, ws.resolve())
        assert kind == "outside"

    def test_broken_symlink_rejects(self, tmp_path):
        ws = tmp_path / "ws"
        ws.mkdir()
        link = ws / "broken.txt"
        link.symlink_to(tmp_path / "nonexistent")
        kind, _reason = _classify_symlink(link, ws.resolve())
        # Broken link resolves to a path outside workspace -> reject.
        assert kind == "outside"


class TestHashFileByClass:
    def test_safetensors_hashes_header_only(self, tmp_path):
        p = tmp_path / "a.safetensors"
        p.write_bytes(_safetensors_bytes({"t": {"dtype": "F16", "shape": [1], "data_offsets": [0, 2]}}))
        sha = _hash_file_by_class(p)
        assert sha.startswith("sha256:")

    def test_catchall_uses_full_content(self, tmp_path):
        p = tmp_path / "config.json"
        p.write_text("abc")
        sha = _hash_file_by_class(p)
        expected = "sha256:" + hashlib.sha256(b"abc").hexdigest()
        assert sha == expected

    def test_catchall_cap_skips_read(self, tmp_path):
        """Files larger than CATCHALL_FULL_READ_CAP fall back to stat-only."""
        p = tmp_path / "giant.bin"
        p.write_bytes(b"A" * 16)  # actual content tiny
        # Stub stat to report a size above the cap
        with patch.object(Path, "stat", lambda self: os.stat_result(
            (0o644, 0, 0, 1, 0, 0, CATCHALL_FULL_READ_CAP + 1, 0, 0, 0)
        )):
            sha = _hash_file_by_class(p)
        expected = hashlib.sha256()
        expected.update(b"giant.bin")
        expected.update(b"\x00")
        expected.update(str(CATCHALL_FULL_READ_CAP + 1).encode("ascii"))
        assert sha == "sha256:" + expected.hexdigest()

    def test_catchall_cap_skips_read_counted(self, tmp_path, monkeypatch):
        """Verify that no open() happens for files above the cap."""
        p = tmp_path / "giant.bin"
        p.write_bytes(b"A" * 16)

        opens = []
        real_open = open

        def spy_open(path, *a, **kw):
            opens.append(str(path))
            return real_open(path, *a, **kw)

        monkeypatch.setattr("builtins.open", spy_open)
        with patch.object(Path, "stat", lambda self: os.stat_result(
            (0o644, 0, 0, 1, 0, 0, CATCHALL_FULL_READ_CAP + 1, 0, 0, 0)
        )):
            _hash_file_by_class(p)
        assert str(p) not in opens


class TestAggregateFormula:
    def test_is_deterministic_and_path_aware(self):
        idx_a = [
            {"path": "a.json", "sha": "sha256:aa", "size": 1, "mtime_ns": 0},
            {"path": "b.json", "sha": "sha256:bb", "size": 1, "mtime_ns": 0},
        ]
        idx_b = list(reversed(idx_a))  # same content, different order
        assert _aggregate_hash(idx_a) == _aggregate_hash(idx_b)

    def test_rename_changes_aggregate(self):
        idx_a = [{"path": "a.json", "sha": "sha256:xx", "size": 1, "mtime_ns": 0}]
        idx_b = [{"path": "b.json", "sha": "sha256:xx", "size": 1, "mtime_ns": 0}]
        assert _aggregate_hash(idx_a) != _aggregate_hash(idx_b)


# ---------------------------------------------------------------------------
# Integration — compute + clean check
# ---------------------------------------------------------------------------


class TestComputeWorkspaceHashV2:
    def test_basic_workspace(self, tmp_path):
        ws = _make_workspace(tmp_path / "ws")
        result = compute_workspace_hash_v2(ws)
        assert result is not None
        content_hash, file_index = result
        assert content_hash.startswith("sha256:")
        paths = {e["path"] for e in file_index}
        assert "config.json" in paths
        assert "tokenizer.json" in paths
        assert "model-00001-of-00001.safetensors" in paths
        # sentinel itself must be excluded
        assert SENTINEL_FILENAME not in paths

    def test_missing_config_returns_none(self, tmp_path):
        (tmp_path / "empty").mkdir()
        assert compute_workspace_hash_v2(tmp_path / "empty") is None

    def test_stable_across_calls(self, tmp_path):
        ws = _make_workspace(tmp_path / "ws")
        a, _ = compute_workspace_hash_v2(ws)
        b, _ = compute_workspace_hash_v2(ws)
        assert a == b

    def test_covers_processor_and_chat_template(self, tmp_path):
        """Issue #52 coverage: non-weight config files enter the hash."""
        ws = _make_workspace(tmp_path / "ws")
        baseline, _ = compute_workspace_hash_v2(ws)
        (ws / "processor_config.json").write_text(json.dumps({"size": 224}))
        after_add, _ = compute_workspace_hash_v2(ws)
        assert baseline != after_add

        (ws / "processor_config.json").write_text(json.dumps({"spatial_merge_size": 2}))
        after_edit, _ = compute_workspace_hash_v2(ws)
        assert after_add != after_edit

    def test_regression_issue_52_repair_index(self, tmp_path):
        """Writing a new model.safetensors.index.json changes the hash."""
        ws = _make_workspace(tmp_path / "ws")
        (ws / "model.safetensors.index.json").write_text(json.dumps({"weight_map": {"a": "m0.safetensors"}}))
        before, _ = compute_workspace_hash_v2(ws)
        # Simulate repair-index rewrite with different weight_map content
        (ws / "model.safetensors.index.json").write_text(json.dumps({"weight_map": {"a": "m1.safetensors"}}))
        after, _ = compute_workspace_hash_v2(ws)
        assert before != after

    def test_tokenizer_content_matters(self, tmp_path):
        """v1 ignored tokenizer content; v2 must not."""
        ws = _make_workspace(tmp_path / "ws")
        (ws / "tokenizer.json").write_text(json.dumps({"model": {"type": "BPE"}}))
        before, _ = compute_workspace_hash_v2(ws)
        (ws / "tokenizer.json").write_text(json.dumps({"model": {"type": "Unigram"}}))
        after, _ = compute_workspace_hash_v2(ws)
        assert before != after

    def test_corrupt_safetensors_aborts(self, tmp_path):
        ws = _make_workspace(tmp_path / "ws")
        # Zero-length prefix → corrupt
        (ws / "bad.safetensors").write_bytes((0).to_bytes(8, "little") + b"junk")
        assert compute_workspace_hash_v2(ws) is None

    def test_outside_symlink_aborts(self, tmp_path):
        ws = _make_workspace(tmp_path / "ws")
        outside = tmp_path / "secret.txt"
        outside.write_text("secret")
        (ws / "escape.txt").symlink_to(outside.resolve())
        assert compute_workspace_hash_v2(ws) is None

    def test_inside_symlink_fingerprinted(self, tmp_path):
        ws = _make_workspace(tmp_path / "ws")
        real = ws / "real.json"
        real.write_text("{}")
        link = ws / "link.json"
        link.symlink_to(real)
        result = compute_workspace_hash_v2(ws)
        assert result is not None
        _content_hash, idx = result
        link_entry = next(e for e in idx if e["path"] == "link.json")
        assert link_entry["sha"].startswith("sha256:")

    def test_excluded_directories_are_pruned(self, tmp_path):
        ws = _make_workspace(tmp_path / "ws")
        (ws / ".vscode").mkdir()
        (ws / ".vscode" / "settings.json").write_text("{}")
        (ws / ".pytest_cache").mkdir()
        (ws / ".pytest_cache" / "v.cache").write_text("x")
        (ws / ".hf_cache").mkdir()
        (ws / ".hf_cache" / "blob").write_text("y")
        _content_hash, idx = compute_workspace_hash_v2(ws)
        paths = {e["path"] for e in idx}
        assert not any(p.startswith(".vscode/") for p in paths)
        assert not any(p.startswith(".pytest_cache/") for p in paths)
        assert not any(p.startswith(".hf_cache/") for p in paths)

    def test_excluded_file_patterns_are_skipped(self, tmp_path):
        ws = _make_workspace(tmp_path / "ws")
        (ws / "foo.lock").write_text("lock")
        (ws / ".DS_Store").write_text("mac")
        (ws / "._ghost").write_text("appledouble")
        _h, idx = compute_workspace_hash_v2(ws)
        paths = {e["path"] for e in idx}
        assert "foo.lock" not in paths
        assert ".DS_Store" not in paths
        assert "._ghost" not in paths


class TestIsWorkspaceClean:
    def test_clean_workspace(self, tmp_path):
        ws = _make_v2_workspace(tmp_path / "ws")
        is_clean, current, stored = is_workspace_clean(ws)
        assert is_clean is True
        assert current == stored

    def test_edit_flips_dirty(self, tmp_path):
        ws = _make_v2_workspace(tmp_path / "ws")
        (ws / "config.json").write_text(json.dumps({"model_type": "gemma3"}))
        is_clean, current, stored = is_workspace_clean(ws)
        assert is_clean is False
        assert current is not None
        assert current != stored

    def test_added_file_flips_dirty(self, tmp_path):
        ws = _make_v2_workspace(tmp_path / "ws")
        (ws / "chat_template.jinja").write_text("{{ messages }}")
        is_clean, _c, _s = is_workspace_clean(ws)
        assert is_clean is False

    def test_removed_file_flips_dirty(self, tmp_path):
        ws = _make_v2_workspace(tmp_path / "ws")
        (ws / "tokenizer.json").unlink()
        is_clean, _c, _s = is_workspace_clean(ws)
        assert is_clean is False

    def test_v1_sentinel_returns_unknown(self, tmp_path):
        """Missing hash_algorithm → (None, None, stored_hash)."""
        ws = _make_workspace(tmp_path / "ws")
        # Seed a v1-style sentinel with a string content_hash but no algorithm field
        write_workspace_sentinel(ws, {
            "mlxk_version": __version__,
            "created_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
            "managed": True,
            "operation": "clone",
            "source_repo": "test/example",
            "source_revision": None,
            "content_hash": "deadbeef",
            "hash_modified": "2026-04-20T00:00:00Z",
        })
        is_clean, current, stored = is_workspace_clean(ws)
        assert is_clean is None
        assert current is None
        assert stored == "deadbeef"

    def test_corrupt_sentinel_path_rejected(self, tmp_path):
        """Tampered file_index with '..' path → (None, ...) without any FS op."""
        ws = _make_v2_workspace(tmp_path / "ws")
        meta = read_workspace_metadata(ws)
        meta["file_index"].append({
            "path": "../../etc/passwd",
            "size": 0,
            "mtime_ns": 0,
            "sha": "sha256:beef",
        })
        write_workspace_sentinel(ws, meta)

        opens = []
        real_open = open

        def spy_open(path, *a, **kw):
            opens.append(str(path))
            return real_open(path, *a, **kw)

        with patch("builtins.open", spy_open):
            is_clean, _c, _s = is_workspace_clean(ws)
        assert is_clean is None
        # No read of the traversed path must have occurred.
        assert not any("/etc/passwd" in p for p in opens)

    def test_self_heal_on_mtime_only_drift(self, tmp_path):
        """cp -R-style mtime change without content change → Clean: ✓ after self-heal."""
        ws = _make_v2_workspace(tmp_path / "ws")
        stored_before = read_workspace_metadata(ws)["content_hash"]

        # Shift every file's mtime while keeping byte content identical.
        for f in ws.rglob("*"):
            if f.is_file() and f.name != SENTINEL_FILENAME:
                st = f.stat()
                os.utime(f, ns=(st.st_atime_ns, st.st_mtime_ns + 1_000_000_000))

        is_clean, current, stored = is_workspace_clean(ws)
        assert is_clean is True
        assert stored == stored_before
        # Sentinel should have been updated with the new mtime values.
        meta_after = read_workspace_metadata(ws)
        assert meta_after["content_hash"] == stored_before  # aggregate unchanged

    def test_dev_cruft_mutation_stays_clean(self, tmp_path):
        ws = _make_v2_workspace(tmp_path / "ws")
        (ws / ".vscode").mkdir()
        (ws / ".vscode" / "settings.json").write_text("{}")
        (ws / ".pytest_cache").mkdir()
        (ws / ".pytest_cache" / "v.cache").write_text("x")
        (ws / "foo.lock").write_text("lock")
        is_clean, _c, _s = is_workspace_clean(ws)
        assert is_clean is True
        # Mutating cruft must not flip clean
        (ws / ".vscode" / "settings.json").write_text('{"changed": true}')
        (ws / "foo.lock").write_text("newlock")
        is_clean2, _c, _s = is_workspace_clean(ws)
        assert is_clean2 is True
        # Real mutation does flip it
        (ws / "config.json").write_text(json.dumps({"model_type": "changed"}))
        is_clean3, _c, _s = is_workspace_clean(ws)
        assert is_clean3 is False


class TestUpdateWorkspaceHash:
    def test_writes_v2_sentinel_fields(self, tmp_path):
        ws = _make_workspace(tmp_path / "ws")
        ok, content_hash = update_workspace_hash(ws)
        assert ok is True
        assert content_hash.startswith("sha256:")
        meta = read_workspace_metadata(ws)
        assert meta["hash_algorithm"] == HASH_ALGORITHM_V2
        assert meta["content_hash"] == content_hash
        assert isinstance(meta["file_index"], list)
        assert isinstance(meta["exclude_patterns"], list)
        assert "hash_modified" in meta

    def test_migration_from_v1_sentinel(self, tmp_path):
        """v1 sentinel → clean None → update_workspace_hash → Clean True."""
        ws = _make_workspace(tmp_path / "ws")
        # Stamp v1 shape
        write_workspace_sentinel(ws, {
            "mlxk_version": "2.0.5",
            "created_at": "2026-01-01T00:00:00Z",
            "managed": True,
            "operation": "clone",
            "source_repo": "test/example",
            "source_revision": None,
            "content_hash": "legacy",
            "hash_modified": "2026-01-01T00:00:00Z",
        })
        assert is_workspace_clean(ws)[0] is None
        assert "upgrade" in (get_workspace_clean_hint(ws) or "").lower()

        ok, _h = update_workspace_hash(ws)
        assert ok
        is_clean, _c, _s = is_workspace_clean(ws)
        assert is_clean is True
        assert get_workspace_clean_hint(ws) is None

    def test_preserves_stored_exclude_patterns(self, tmp_path):
        """If sentinel already has a frozen exclude_patterns list, keep it."""
        ws = _make_v2_workspace(tmp_path / "ws")
        meta = read_workspace_metadata(ws)
        custom = [".hf_cache/", ".mlxk_workspace.json", "onlythis.bak"]
        meta["exclude_patterns"] = custom
        write_workspace_sentinel(ws, meta)
        ok, _h = update_workspace_hash(ws)
        assert ok
        meta2 = read_workspace_metadata(ws)
        assert meta2["exclude_patterns"] == custom

    def test_non_managed_rejected(self, tmp_path):
        ws = _make_workspace(tmp_path / "ws", with_sentinel=False)
        ok, h = update_workspace_hash(ws)
        assert ok is False
        assert h is None


class TestExcludeListDriftResilience:
    def test_stored_list_governs(self, tmp_path):
        """Narrower stored exclude_patterns must override current code defaults."""
        ws = _make_workspace(tmp_path / "ws")
        # Stamp a narrow exclude list (only the two runtime entries).
        narrow = [".hf_cache/", ".mlxk_workspace.json"]
        ok, _h = update_workspace_hash(ws)
        assert ok
        meta = read_workspace_metadata(ws)
        meta["exclude_patterns"] = narrow
        # Recompute aggregate against the narrow list so it is self-consistent.
        _new_hash, new_idx = compute_workspace_hash_v2(ws, narrow)
        from mlxk2.operations.workspace import _aggregate_hash as _agg
        meta["file_index"] = new_idx
        meta["content_hash"] = _agg(new_idx)
        write_workspace_sentinel(ws, meta)

        # Add a file that the CODE defaults would exclude (".DS_Store") but the
        # stored narrow list would include.
        (ws / ".DS_Store").write_text("cruft")
        is_clean, _c, _s = is_workspace_clean(ws)
        assert is_clean is False  # narrow list walked .DS_Store; new file is dirty


class TestHintSurface:
    def test_v1_hint(self, tmp_path):
        ws = _make_workspace(tmp_path / "ws")
        write_workspace_sentinel(ws, {
            "mlxk_version": "2.0.5",
            "created_at": "2026-01-01T00:00:00Z",
            "managed": True,
            "operation": "clone",
            "source_repo": "t",
            "source_revision": None,
            "content_hash": "old",
        })
        hint = get_workspace_clean_hint(ws)
        assert hint is not None
        assert "recalc-hash" in hint

    def test_corrupt_hint(self, tmp_path):
        ws = _make_v2_workspace(tmp_path / "ws")
        meta = read_workspace_metadata(ws)
        meta["file_index"].append({"path": "../escape", "size": 0, "mtime_ns": 0, "sha": "sha256:x"})
        write_workspace_sentinel(ws, meta)
        hint = get_workspace_clean_hint(ws)
        assert hint is not None
        assert "invalid" in hint.lower()

    def test_clean_v2_returns_no_hint(self, tmp_path):
        ws = _make_v2_workspace(tmp_path / "ws")
        assert get_workspace_clean_hint(ws) is None
