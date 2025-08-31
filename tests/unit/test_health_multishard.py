"""
Strict health checks for multi-shard/index models (Issue #27 backport).
"""
import json
from pathlib import Path

from mlx_knife.cache_utils import is_model_healthy


def _write_json(p: Path, data: dict) -> None:
    p.write_text(json.dumps(data))


def _mk_snapshot(tmp_dir: Path, name: str = "models--org--model", snap: str = "main") -> Path:
    d = tmp_dir / "hub" / name / "snapshots" / snap
    d.mkdir(parents=True, exist_ok=True)
    return d


def test_index_complete_healthy(temp_cache_dir: Path):
    d = _mk_snapshot(temp_cache_dir)
    # Valid config
    _write_json(d / "config.json", {"model_type": "test"})
    # Two shards + index
    (d / "model-00001-of-00002.safetensors").write_bytes(b"a" * 1024)
    (d / "model-00002-of-00002.safetensors").write_bytes(b"b" * 1024)
    _write_json(
        d / "model.safetensors.index.json",
        {"weight_map": {"w1": "model-00001-of-00002.safetensors", "w2": "model-00002-of-00002.safetensors"}},
    )
    assert is_model_healthy(str(d)) is True


def test_index_missing_shard_unhealthy(temp_cache_dir: Path):
    d = _mk_snapshot(temp_cache_dir)
    _write_json(d / "config.json", {"model_type": "test"})
    # Only one shard present
    (d / "model-00001-of-00002.safetensors").write_bytes(b"a" * 1024)
    _write_json(
        d / "model.safetensors.index.json",
        {"weight_map": {"w1": "model-00001-of-00002.safetensors", "w2": "model-00002-of-00002.safetensors"}},
    )
    assert is_model_healthy(str(d)) is False


def test_index_empty_shard_unhealthy(temp_cache_dir: Path):
    d = _mk_snapshot(temp_cache_dir)
    _write_json(d / "config.json", {"model_type": "test"})
    (d / "model-00001-of-00002.safetensors").write_bytes(b"")  # empty
    (d / "model-00002-of-00002.safetensors").write_bytes(b"b" * 1024)
    _write_json(
        d / "model.safetensors.index.json",
        {"weight_map": {"w1": "model-00001-of-00002.safetensors", "w2": "model-00002-of-00002.safetensors"}},
    )
    assert is_model_healthy(str(d)) is False


def test_index_lfs_pointer_unhealthy(temp_cache_dir: Path):
    d = _mk_snapshot(temp_cache_dir)
    _write_json(d / "config.json", {"model_type": "test"})
    # One shard becomes an LFS pointer (small text with signature)
    (d / "model-00001-of-00002.safetensors").write_text(
        "version https://git-lfs.github.com/spec/v1\n"
        "oid sha256:deadbeef\n"
        "size 100\n"
    )
    (d / "model-00002-of-00002.safetensors").write_bytes(b"b" * 1024)
    _write_json(
        d / "model.safetensors.index.json",
        {"weight_map": {"w1": "model-00001-of-00002.safetensors", "w2": "model-00002-of-00002.safetensors"}},
    )
    assert is_model_healthy(str(d)) is False


def test_pattern_complete_without_index_unhealthy(temp_cache_dir: Path):
    d = _mk_snapshot(temp_cache_dir)
    _write_json(d / "config.json", {"model_type": "test"})
    # Two shards with pattern but no index
    (d / "model-00001-of-00002.safetensors").write_bytes(b"a" * 1024)
    (d / "model-00002-of-00002.safetensors").write_bytes(b"b" * 1024)
    assert is_model_healthy(str(d)) is False


def test_pattern_incomplete_unhealthy(temp_cache_dir: Path):
    d = _mk_snapshot(temp_cache_dir)
    _write_json(d / "config.json", {"model_type": "test"})
    # Only one pattern shard present, no index
    (d / "model-00001-of-00002.safetensors").write_bytes(b"a" * 1024)
    assert is_model_healthy(str(d)) is False


def test_partial_marker_unhealthy(temp_cache_dir: Path):
    d = _mk_snapshot(temp_cache_dir)
    _write_json(d / "config.json", {"model_type": "test"})
    (d / "model.safetensors.partial").write_bytes(b"x")
    assert is_model_healthy(str(d)) is False


def test_single_file_safetensors_ok(temp_cache_dir: Path):
    d = _mk_snapshot(temp_cache_dir)
    _write_json(d / "config.json", {"model_type": "test"})
    (d / "model.safetensors").write_bytes(b"weights" * 256)
    assert is_model_healthy(str(d)) is True


def test_single_file_gguf_ok(temp_cache_dir: Path):
    d = _mk_snapshot(temp_cache_dir)
    _write_json(d / "config.json", {"model_type": "test"})
    (d / "model.q4_0.gguf").write_bytes(b"gguf-weights" * 256)
    assert is_model_healthy(str(d)) is True


def test_pytorch_index_complete_ok(temp_cache_dir: Path):
    d = _mk_snapshot(temp_cache_dir)
    _write_json(d / "config.json", {"model_type": "test"})
    (d / "pytorch_model-00001-of-00002.bin").write_bytes(b"a" * 1024)
    (d / "pytorch_model-00002-of-00002.bin").write_bytes(b"b" * 1024)
    _write_json(
        d / "pytorch_model.bin.index.json",
        {
            "weight_map": {
                "w1": "pytorch_model-00001-of-00002.bin",
                "w2": "pytorch_model-00002-of-00002.bin",
            }
        },
    )
    assert is_model_healthy(str(d)) is True

