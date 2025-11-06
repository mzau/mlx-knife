"""Tests for JSON API spec v0.1.2: list operation minimal model object.

Covers: size_bytes, last_modified (ISO-8601 Z), framework, model_type,
capabilities, health, hash selection, cached.
"""

from datetime import datetime
from typing import Set
import pytest

from mlxk2.operations.list import list_models


def _is_iso_utc_z(ts: str) -> bool:
    try:
        # Must end with 'Z' and be parseable
        if not ts.endswith("Z"):
            return False
        # Strip Z, attempt parsing
        datetime.fromisoformat(ts.replace("Z", ""))
        return True
    except Exception:
        return False


@pytest.mark.spec
def test_list_minimal_model_object_fields(mock_models, isolated_cache):
    """Each model entry returns the minimal model object with health."""
    result = list_models()
    assert result["status"] == "success"
    assert result["command"] == "list"

    models = result["data"]["models"]
    assert isinstance(models, list)
    assert result["data"]["count"] == len(models)

    # Allowed enums
    allowed_framework: Set[str] = {"MLX", "GGUF", "PyTorch", "Unknown"}
    allowed_model_types: Set[str] = {"chat", "embedding", "base", "unknown"}

    # Verify minimal fields and types
    for m in models:
        # Required fields
        assert set([
            "name", "hash", "size_bytes", "last_modified", "framework",
            "model_type", "capabilities", "health", "cached"
        ]).issubset(m.keys())

        assert isinstance(m["name"], str) and "/" in m["name"]

        # hash: 40-char or None
        h = m["hash"]
        assert (h is None) or (isinstance(h, str) and len(h) == 40)

        # size_bytes integer >= 0
        assert isinstance(m["size_bytes"], int)
        assert m["size_bytes"] >= 0

        # last_modified as ISO-8601 UTC Z
        assert isinstance(m["last_modified"], str)
        assert _is_iso_utc_z(m["last_modified"]) is True

        # framework
        assert m["framework"] in allowed_framework

        # model_type + capabilities
        assert m["model_type"] in allowed_model_types
        assert isinstance(m["capabilities"], list)

        # health
        assert m["health"] in {"healthy", "unhealthy"}

        # cached flag
        assert m["cached"] is True

        # Spec 0.1.2: no human-readable size; ensure we do not expose 'size' or internal paths
        assert "size" not in m
        assert "hashes" not in m


@pytest.mark.spec
def test_list_pattern_filter_case_insensitive(mock_models, isolated_cache):
    """Pattern filters case-insensitively on model name."""
    result = list_models(pattern="llama")
    models = result["data"]["models"]
    assert all("llama" in m["name"].lower() for m in models)

    # A different pattern should yield different subset
    result_q = list_models(pattern="Qwen")
    models_q = result_q["data"]["models"]
    assert all("qwen" in m["name"].lower() for m in models_q)
    # Ensure partition is non-trivial in our fixture
    assert set(m["name"].lower() for m in models).isdisjoint(
        set(m["name"].lower() for m in models_q)
    ) is True


@pytest.mark.spec
def test_list_empty_cache(isolated_cache):
    """Empty cache yields empty list and count 0."""
    # Remove all models (keep canary)
    for d in isolated_cache.iterdir():
        if d.is_dir() and d.name.startswith("models--") and "TEST-CACHE-SENTINEL" not in d.name:
            # Safe in tests; strict delete is enforced by fixture env var
            from shutil import rmtree
            rmtree(d)

    result = list_models()
    assert result["status"] == "success"
    assert result["data"]["models"] == []
    assert result["data"]["count"] == 0
