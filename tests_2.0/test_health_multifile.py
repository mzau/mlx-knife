"""Deterministic tests for multi-file safetensors health (Issue #27 parity)."""

import json
from pathlib import Path


def _write_idx(dir: Path, shards: list[str]):
    idx = {
        "metadata": {},
        "weight_map": {f"layer{i}": shard for i, shard in enumerate(shards)}
    }
    (dir / "model.safetensors.index.json").write_text(json.dumps(idx))


def test_multifile_index_missing_shard_is_unhealthy(isolated_cache):
    snap = isolated_cache / "models--test--mf" / "snapshots" / "main"
    snap.mkdir(parents=True)
    shards = ["model-00001-of-00002.safetensors", "model-00002-of-00002.safetensors"]
    _write_idx(snap, shards)
    # Create only one shard (subset)
    (snap / shards[0]).write_bytes(b"ok")

    from mlxk2.operations.health import health_check_operation
    result = health_check_operation("test/mf")
    assert result["status"] == "success"
    assert any(m["name"] == "test/mf" and m["status"] == "unhealthy" for m in result["data"]["unhealthy"])


def test_multifile_index_empty_shard_is_unhealthy(isolated_cache):
    snap = isolated_cache / "models--test--mf2" / "snapshots" / "main"
    snap.mkdir(parents=True)
    shards = ["model-00001-of-00002.safetensors", "model-00002-of-00002.safetensors"]
    _write_idx(snap, shards)
    # Create both, but one empty
    (snap / shards[0]).write_bytes(b"ok")
    (snap / shards[1]).write_bytes(b"")

    from mlxk2.operations.health import health_check_operation
    result = health_check_operation("test/mf2")
    assert result["status"] == "success"
    assert any(m["name"] == "test/mf2" and m["status"] == "unhealthy" for m in result["data"]["unhealthy"])


def test_multifile_index_complete_is_healthy(isolated_cache):
    snap = isolated_cache / "models--test--mf3" / "snapshots" / "main"
    snap.mkdir(parents=True)
    shards = ["model-00001-of-00002.safetensors", "model-00002-of-00002.safetensors"]
    _write_idx(snap, shards)
    for s in shards:
        (snap / s).write_bytes(b"ok")
    # Minimal valid config
    (snap / "config.json").write_text(json.dumps({"model_type": "test"}))


def test_multifile_pattern_missing_shard_is_unhealthy(isolated_cache):
    snap = isolated_cache / "models--test--mf4" / "snapshots" / "main"
    snap.mkdir(parents=True)
    # No index file; only pattern shards
    shards = [
        "model-00001-of-00003.safetensors",
        # missing 00002
        "model-00003-of-00003.safetensors",
    ]
    for s in shards:
        (snap / s).write_bytes(b"ok")
    (snap / "config.json").write_text(json.dumps({"model_type": "test"}))

    from mlxk2.operations.health import health_check_operation
    result = health_check_operation("test/mf4")
    assert any(m["name"] == "test/mf4" and m["status"] == "unhealthy" for m in result["data"]["unhealthy"])


def test_multifile_pattern_complete_is_unhealthy_without_index(isolated_cache):
    snap = isolated_cache / "models--test--mf5" / "snapshots" / "main"
    snap.mkdir(parents=True)
    shards = [
        "model-00001-of-00003.safetensors",
        "model-00002-of-00003.safetensors",
        "model-00003-of-00003.safetensors",
    ]
    for s in shards:
        (snap / s).write_bytes(b"ok")
    (snap / "config.json").write_text(json.dumps({"model_type": "test"}))

    from mlxk2.operations.health import health_check_operation
    result = health_check_operation("test/mf5")
    # Robust policy: without index, sharded safetensors are unhealthy even if complete
    assert any(m["name"] == "test/mf5" and m["status"] == "unhealthy" for m in result["data"]["unhealthy"])


def test_partial_tmp_marker_is_unhealthy(isolated_cache):
    snap = isolated_cache / "models--test--partial" / "snapshots" / "main"
    snap.mkdir(parents=True)
    # Single-file weight but with partial marker
    (snap / "model.safetensors").write_bytes(b"ok")
    (snap / ".partial.tmp").write_bytes(b"downloading")
    (snap / "config.json").write_text(json.dumps({"model_type": "test"}))

    from mlxk2.operations.health import health_check_operation
    result = health_check_operation("test/partial")
    assert any(m["name"] == "test/partial" and m["status"] == "unhealthy" for m in result["data"]["unhealthy"])
