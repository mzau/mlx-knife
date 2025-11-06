"""Tests for JSON API spec v0.1.2: show operation variants.

Validates minimal model object and that --files and --config yield different
optional data sections.
"""

from datetime import datetime
import pytest

from mlxk2.operations.show import show_model_operation


def _is_iso_utc_z(ts: str) -> bool:
    try:
        if not ts.endswith("Z"):
            return False
        datetime.fromisoformat(ts.replace("Z", ""))
        return True
    except Exception:
        return False


@pytest.mark.spec
def test_show_minimal_model_object(mock_models, isolated_cache):
    name = "mlx-community/Phi-3-mini-4k-instruct-4bit"
    res = show_model_operation(name)
    assert res["status"] == "success"
    assert res["command"] == "show"

    model = res["data"]["model"]
    assert set([
        "name", "hash", "size_bytes", "last_modified", "framework",
        "model_type", "capabilities", "health", "cached"
    ]).issubset(model.keys())
    assert model["name"] == name
    assert (model["hash"] is None) or (isinstance(model["hash"], str) and len(model["hash"]) == 40)
    assert isinstance(model["size_bytes"], int) and model["size_bytes"] > 0
    assert _is_iso_utc_z(model["last_modified"]) is True
    assert model["cached"] is True
    # Ensure show does not expose human-readable size
    assert "size" not in model

    # Default branch returns metadata when available
    assert "metadata" in res["data"]


@pytest.mark.spec
def test_show_with_files_and_config_are_different(mock_models, isolated_cache):
    name = "mlx-community/Phi-3-mini-4k-instruct-4bit"

    res_files = show_model_operation(name, include_files=True, include_config=False)
    assert res_files["status"] == "success"
    assert "files" in res_files["data"]
    assert res_files["data"].get("metadata") is None
    assert "config" not in res_files["data"]

    files = res_files["data"]["files"]
    assert isinstance(files, list) and len(files) > 0
    # Validate file entry shape
    first = files[0]
    assert set(["name", "size", "type"]).issubset(first.keys())

    res_config = show_model_operation(name, include_files=False, include_config=True)
    assert res_config["status"] == "success"
    assert "config" in res_config["data"]
    assert res_config["data"].get("metadata") is None
    assert "files" not in res_config["data"]

    cfg = res_config["data"]["config"]
    assert isinstance(cfg, dict) and len(cfg) > 0

    # Compare that the two payloads differ in optional sections
    assert ("files" in res_files["data"]) != ("files" in res_config["data"])  # XOR presence
    assert ("config" in res_files["data"]) != ("config" in res_config["data"])  # XOR presence
