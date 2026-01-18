"""Offline tests for push --check-only (workspace health)."""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

# Skip all tests if push is not enabled
# Push tests now run by default (alpha features included in standard test suite)

from mlxk2.operations.push import push_operation, DEFAULT_PUSH_BRANCH


def test_check_only_minimal_invalid_config(tmp_path):
    ws: Path = tmp_path / "ws"
    ws.mkdir()
    # Invalid JSON config
    (ws / "config.json").write_text("{")
    # A dummy weight file
    (ws / "model.safetensors").write_text("data")

    res = push_operation(str(ws), "org/model", branch=DEFAULT_PUSH_BRANCH, check_only=True)
    assert res["status"] == "success"
    diag = res["data"]["workspace_health"]
    assert diag["config"]["exists"] is True
    assert diag["config"]["valid_json"] is False
    assert diag["healthy"] is False
    assert any(a["code"] == "config_invalid_json" for a in diag["anomalies"])


def test_check_only_index_missing_shard(tmp_path):
    ws = tmp_path / "ws"
    ws.mkdir()
    (ws / "config.json").write_text('{"model_type": "base"}')
    # Index references a missing shard
    idx = {"weight_map": {"w0": "model-00001-of-00002.safetensors", "w1": "model-00002-of-00002.safetensors"}}
    (ws / "model.safetensors.index.json").write_text(json.dumps(idx))
    # Create only one shard
    (ws / "model-00001-of-00002.safetensors").write_text("x")

    res = push_operation(str(ws), "org/model", branch=DEFAULT_PUSH_BRANCH, check_only=True)
    diag = res["data"]["workspace_health"]
    assert diag["healthy"] is False
    assert any(a["code"] == "index_missing_shard" for a in diag["anomalies"])


def test_check_only_gguf_single_file_ok(tmp_path):
    ws = tmp_path / "ws"
    ws.mkdir()
    (ws / "config.json").write_text('{"model_type": "base"}')
    # Single GGUF file
    (ws / "model.gguf").write_bytes(b"\x00\x01\x02")

    res = push_operation(str(ws), "org/model", branch=DEFAULT_PUSH_BRANCH, check_only=True)
    diag = res["data"]["workspace_health"]
    assert diag["healthy"] is True
    assert diag["weights"]["count"] == 1
    assert "gguf" in diag["weights"]["formats"]


def test_check_only_lfs_pointer_detected(tmp_path):
    ws = tmp_path / "ws"
    ws.mkdir()
    (ws / "config.json").write_text("{}")
    # Create a small LFS pointer file
    lfs = (ws / "pytorch_model.bin")
    lfs.write_text("version https://git-lfs.github.com/spec/v1\nOID sha256:abc\nsize 123\n")

    res = push_operation(str(ws), "org/model", branch=DEFAULT_PUSH_BRANCH, check_only=True)
    diag = res["data"]["workspace_health"]
    assert diag["healthy"] is False
    assert any(a["code"] == "lfs_pointer_detected" for a in diag["anomalies"])


class TestPushAmbiguousWorkspace:
    """Tests for ambiguous workspace pattern handling (Session 103)."""

    def test_push_ambiguous_prefix_pattern(self, tmp_path):
        """Ambiguous prefix pattern should return clear error with matches list."""
        import os

        # Create multiple workspaces with common prefix
        for name in ["gemma-3n-4bit", "gemma-3n-8bit", "gemma-3n-FIXED-4bit"]:
            ws = tmp_path / name
            ws.mkdir()
            (ws / "config.json").write_text('{"model_type": "gemma"}')
            (ws / "model.safetensors").write_text("data")

        old_cwd = os.getcwd()
        try:
            os.chdir(tmp_path)
            res = push_operation("./gemma-", "org/model", branch=DEFAULT_PUSH_BRANCH, check_only=True)

            assert res["status"] == "error"
            assert res["error"]["type"] == "ambiguous_workspace"
            assert "3 workspaces" in res["error"]["message"]
            assert "matches" in res["error"]
            assert len(res["error"]["matches"]) == 3
        finally:
            os.chdir(old_cwd)

    def test_push_prefix_single_match_succeeds(self, tmp_path):
        """Prefix pattern with single match should work."""
        import os

        ws = tmp_path / "unique-model-4bit"
        ws.mkdir()
        (ws / "config.json").write_text('{"model_type": "llama"}')
        (ws / "model.safetensors").write_text("data")

        old_cwd = os.getcwd()
        try:
            os.chdir(tmp_path)
            res = push_operation("./unique-", "org/model", branch=DEFAULT_PUSH_BRANCH, check_only=True)

            # Should succeed (check_only doesn't need HF_TOKEN)
            assert res["status"] == "success"
            assert res["data"]["workspace_health"]["healthy"] is True
        finally:
            os.chdir(old_cwd)

    def test_push_prefix_no_match(self, tmp_path):
        """Prefix pattern with no matches should return workspace_not_found."""
        import os

        old_cwd = os.getcwd()
        try:
            os.chdir(tmp_path)
            res = push_operation("./nonexistent-", "org/model", branch=DEFAULT_PUSH_BRANCH, check_only=True)

            assert res["status"] == "error"
            assert res["error"]["type"] == "workspace_not_found"
            assert "No workspace found" in res["error"]["message"]
        finally:
            os.chdir(old_cwd)

    def test_push_exact_path_still_works(self, tmp_path):
        """Exact workspace path should still work as before."""
        ws = tmp_path / "my-model"
        ws.mkdir()
        (ws / "config.json").write_text('{"model_type": "llama"}')
        (ws / "model.safetensors").write_text("data")

        res = push_operation(str(ws), "org/model", branch=DEFAULT_PUSH_BRANCH, check_only=True)

        assert res["status"] == "success"
        assert res["data"]["workspace_health"]["healthy"] is True

    def test_push_dot_pushes_current_directory(self, tmp_path):
        """'push .' pushes the current directory directly (not directory scan)."""
        import os

        # tmp_path itself as a workspace
        (tmp_path / "config.json").write_text('{"model_type": "llama"}')
        (tmp_path / "model.safetensors").write_text("data" * 100)  # Needs some size

        old_cwd = os.getcwd()
        try:
            os.chdir(tmp_path)
            res = push_operation(".", "org/model", branch=DEFAULT_PUSH_BRANCH, check_only=True)

            # Should succeed - pushes current directory directly (not ambiguous error)
            assert res["status"] == "success"
            # workspace_health should exist (check_only performs health analysis)
            assert "workspace_health" in res["data"]
        finally:
            os.chdir(old_cwd)
