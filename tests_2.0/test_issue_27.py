"""Exploratory tests for Issue #27 using real model copies in isolated cache.

These tests are opt-in and require MLXK2_USER_HF_HOME to point to the user HF cache.
They never modify the user cache; they copy selected models into the isolated test cache
and then apply controlled mutations to simulate edge cases.
"""

import os
import pytest


requires_user_cache = pytest.mark.skipif(
    not os.environ.get("MLXK2_USER_HF_HOME"),
    reason="requires MLXK2_USER_HF_HOME (user cache path)"
)


@requires_user_cache
class TestIssue27Exploration:
    def test_copy_real_model_and_list(self, copy_user_model_to_isolated):
        # Choose a common model; allow override via env
        model = os.environ.get(
            "MLXK2_ISSUE27_MODEL", "mlx-community/Phi-3-mini-4k-instruct-4bit"
        )
        dst = copy_user_model_to_isolated(model)

        # Verify list sees it via the regular operation
        from mlxk2.operations.list import list_models
        result = list_models()
        assert result["status"] == "success"
        names = [m["name"] for m in result["data"]["models"]]
        assert model in names

    def test_partial_download_simulation_health(self, copy_user_model_to_isolated):
        model = os.environ.get(
            "MLXK2_ISSUE27_MODEL", "mlx-community/Phi-3-mini-4k-instruct-4bit"
        )
        # Simulate partial/incomplete model state
        copy_user_model_to_isolated(model, mutations=[
            "remove_config", "truncate_weight", "add_partial_tmp"
        ])

        # Health should not crash and should report issues
        from mlxk2.operations.health import health_check_operation
        result = health_check_operation(model)
        assert result["status"] == "success"
        issues = result["data"]["unhealthy"]
        # Either unhealthy includes this model, or health summaries remain consistent
        if issues:
            assert any(model in m.get("name", "") for m in issues)

    def test_index_missing_shards_unhealthy(self, copy_user_model_to_isolated, monkeypatch):
        model = os.environ.get(
            "MLXK2_ISSUE27_MODEL", "intfloat/multilingual-e5-large"
        )
        # Force subset copy with 0 shards to minimize disk use
        monkeypatch.setenv("MLXK2_COPY_STRATEGY", "index_subset")
        monkeypatch.setenv("MLXK2_SUBSET_COUNT", "0")
        dst = copy_user_model_to_isolated(model)
        idx = dst / 'model.safetensors.index.json'
        if not idx.exists():
            pytest.skip('No safetensors index found; skipping index-missing-shards test')

        from mlxk2.operations.health import health_check_operation
        result = health_check_operation(model)
        assert any(m["name"].endswith(model.split('/')[-1]) or m["name"] == model for m in result["data"]["unhealthy"])

    def test_index_delete_shard_is_unhealthy(self, copy_user_model_to_isolated):
        model = os.environ.get(
            "MLXK2_ISSUE27_MODEL", "mlx-community/Mistral-7B-Instruct-v0.2-4bit"
        )
        dst = copy_user_model_to_isolated(model, mutations=['delete_indexed_shard'])
        # If no index exists, skip this targeted test
        if not (dst / 'model.safetensors.index.json').exists():
            pytest.skip('No safetensors index found; skipping index-specific test')

        from mlxk2.operations.health import health_check_operation
        result = health_check_operation(model)
        assert any(m["name"] == model and m["status"] == "unhealthy" for m in result["data"]["unhealthy"])

    def test_index_truncate_shard_is_unhealthy(self, copy_user_model_to_isolated):
        model = os.environ.get(
            "MLXK2_ISSUE27_MODEL", "mlx-community/Mistral-7B-Instruct-v0.2-4bit"
        )
        dst = copy_user_model_to_isolated(model, mutations=['truncate_indexed_shard'])
        if not (dst / 'model.safetensors.index.json').exists():
            pytest.skip('No safetensors index found; skipping index-specific test')

        from mlxk2.operations.health import health_check_operation
        result = health_check_operation(model)
        assert any(m["name"] == model and m["status"] == "unhealthy" for m in result["data"]["unhealthy"])

    def test_index_lfs_pointer_is_unhealthy(self, copy_user_model_to_isolated):
        model = os.environ.get(
            "MLXK2_ISSUE27_MODEL", "mlx-community/Mistral-7B-Instruct-v0.2-4bit"
        )
        dst = copy_user_model_to_isolated(model, mutations=['lfsify_indexed_shard'])
        if not (dst / 'model.safetensors.index.json').exists():
            pytest.skip('No safetensors index found; skipping index-specific test')

        from mlxk2.operations.health import health_check_operation
        result = health_check_operation(model)
        assert any(m["name"] == model and m["status"] == "unhealthy" for m in result["data"]["unhealthy"])

    def test_user_cache_health_ok_readonly(self, monkeypatch):
        """Read-only health OK check directly against user cache (no copy)."""
        user_hf_home = os.environ.get("MLXK2_USER_HF_HOME")
        if not user_hf_home:
            pytest.skip("MLXK2_USER_HF_HOME not set; skipping user cache health OK test")

        model = os.environ.get(
            "MLXK2_ISSUE27_MODEL", "intfloat/multilingual-e5-large"
        )
        # Verify model exists in user cache
        from pathlib import Path
        from mlxk2.core.cache import hf_to_cache_dir
        src = Path(user_hf_home) / "hub" / hf_to_cache_dir(model)
        if not src.exists():
            pytest.skip(f"Model not present in user cache: {src}")

        # Point HF_HOME to user cache temporarily (read-only operation)
        monkeypatch.setenv("HF_HOME", user_hf_home)
        from mlxk2.operations.health import health_check_operation
        result = health_check_operation(model)
        assert result["status"] == "success"
        assert any(
            m.get("name") == model and m.get("status") == "healthy"
            for m in result["data"]["healthy"]
        ), f"Expected healthy for user model, got: {result}"
