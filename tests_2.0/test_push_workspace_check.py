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
