"""Unit tests for portfolio discovery functions (Phase 2: Test Portfolio Separation).

Tests the new discover_text_models() and discover_vision_models() functions
that enable separate Text and Vision test portfolios.
"""

import json
import sys
from pathlib import Path
from unittest.mock import patch, MagicMock
import pytest

# Add tests_2.0 to path to import live.test_utils
_tests_dir = Path(__file__).parent
if str(_tests_dir) not in sys.path:
    sys.path.insert(0, str(_tests_dir))


class TestTextModelsDiscovery:
    """Tests for discover_text_models() function."""

    def test_discover_text_models_filters_out_vision(self, monkeypatch):
        """Verify that discover_text_models() filters out vision models."""
        # Mock discover_mlx_models_in_user_cache to return mixed portfolio
        mock_all_models = [
            {"model_id": "mlx-community/Qwen2.5-0.5B-Instruct-4bit", "ram_needed_gb": 1.0, "snapshot_path": None, "weight_count": None},
            {"model_id": "mlx-community/Llama-3.2-11B-Vision-Instruct-4bit", "ram_needed_gb": 24.0, "snapshot_path": None, "weight_count": None},
            {"model_id": "mlx-community/Phi-3-mini-4k-instruct-4bit", "ram_needed_gb": 3.0, "snapshot_path": None, "weight_count": None},
        ]

        # Mock mlxk list --json output with capabilities
        mock_list_output = {
            "status": "success",
            "command": "list",
            "data": {
                "models": [
                    {"name": "mlx-community/Qwen2.5-0.5B-Instruct-4bit", "capabilities": ["text-generation", "chat"]},
                    {"name": "mlx-community/Llama-3.2-11B-Vision-Instruct-4bit", "capabilities": ["text-generation", "chat", "vision"]},
                    {"name": "mlx-community/Phi-3-mini-4k-instruct-4bit", "capabilities": ["text-generation", "chat"]},
                ],
                "count": 3
            },
            "error": None
        }

        with patch("live.test_utils.discover_mlx_models_in_user_cache", return_value=mock_all_models):
            with patch("subprocess.run") as mock_run:
                mock_run.return_value = MagicMock(
                    returncode=0,
                    stdout=json.dumps(mock_list_output)
                )

                # Set HF_HOME to enable filtering
                monkeypatch.setenv("HF_HOME", "/fake/cache")

                from live.test_utils import discover_text_models
                result = discover_text_models()

                # Should return only text models (no vision)
                assert len(result) == 2
                model_ids = [m["model_id"] for m in result]
                assert "mlx-community/Qwen2.5-0.5B-Instruct-4bit" in model_ids
                assert "mlx-community/Phi-3-mini-4k-instruct-4bit" in model_ids
                assert "mlx-community/Llama-3.2-11B-Vision-Instruct-4bit" not in model_ids

    def test_discover_text_models_returns_empty_when_no_hf_home(self, monkeypatch):
        """Verify fallback behavior when HF_HOME not set.

        Without HF_HOME, discover_mlx_models_in_user_cache returns [] (by design).
        This ensures tests fall back to TEST_MODELS hardcoded models.
        See TESTING.md for Portfolio Discovery requirements.
        """
        # Mock discover_mlx_models_in_user_cache to return [] (simulates no HF_HOME)
        with patch("live.test_utils.discover_mlx_models_in_user_cache", return_value=[]):
            # Unset HF_HOME
            monkeypatch.delenv("HF_HOME", raising=False)

            from live.test_utils import discover_text_models
            result = discover_text_models()

            # Should return empty (triggers fallback to TEST_MODELS in portfolio fixture)
            assert result == []

    def test_discover_text_models_handles_empty_portfolio(self):
        """Verify behavior when no models discovered."""
        with patch("live.test_utils.discover_mlx_models_in_user_cache", return_value=[]):
            from live.test_utils import discover_text_models
            result = discover_text_models()

            assert result == []

    def test_discover_text_models_handles_subprocess_error(self, monkeypatch):
        """Verify fallback when mlxk list --json fails."""
        mock_all_models = [
            {"model_id": "mlx-community/Qwen2.5-0.5B-Instruct-4bit", "ram_needed_gb": 1.0, "snapshot_path": None, "weight_count": None},
        ]

        with patch("live.test_utils.discover_mlx_models_in_user_cache", return_value=mock_all_models):
            with patch("subprocess.run") as mock_run:
                mock_run.return_value = MagicMock(returncode=1)
                monkeypatch.setenv("HF_HOME", "/fake/cache")

                from live.test_utils import discover_text_models
                result = discover_text_models()

                # Should return all models (fallback on error)
                assert result == mock_all_models


class TestVisionModelsDiscovery:
    """Tests for discover_vision_models() function."""

    def test_discover_vision_models_filters_only_vision(self, monkeypatch):
        """Verify that discover_vision_models() returns only vision models."""
        mock_all_models = [
            {"model_id": "mlx-community/Qwen2.5-0.5B-Instruct-4bit", "ram_needed_gb": 1.0, "snapshot_path": None, "weight_count": None},
            {"model_id": "mlx-community/Llama-3.2-11B-Vision-Instruct-4bit", "ram_needed_gb": 24.0, "snapshot_path": None, "weight_count": None},
            {"model_id": "mlx-community/pixtral-12b-8bit", "ram_needed_gb": 18.0, "snapshot_path": None, "weight_count": None},
        ]

        mock_list_output = {
            "status": "success",
            "command": "list",
            "data": {
                "models": [
                    {"name": "mlx-community/Qwen2.5-0.5B-Instruct-4bit", "capabilities": ["text-generation", "chat"]},
                    {"name": "mlx-community/Llama-3.2-11B-Vision-Instruct-4bit", "capabilities": ["text-generation", "chat", "vision"]},
                    {"name": "mlx-community/pixtral-12b-8bit", "capabilities": ["text-generation", "chat", "vision"]},
                ],
                "count": 3
            },
            "error": None
        }

        with patch("live.test_utils.discover_mlx_models_in_user_cache", return_value=mock_all_models):
            with patch("subprocess.run") as mock_run:
                mock_run.return_value = MagicMock(
                    returncode=0,
                    stdout=json.dumps(mock_list_output)
                )
                monkeypatch.setenv("HF_HOME", "/fake/cache")

                from live.test_utils import discover_vision_models
                result = discover_vision_models()

                # Should return only vision models
                assert len(result) == 2
                model_ids = [m["model_id"] for m in result]
                assert "mlx-community/Llama-3.2-11B-Vision-Instruct-4bit" in model_ids
                assert "mlx-community/pixtral-12b-8bit" in model_ids
                assert "mlx-community/Qwen2.5-0.5B-Instruct-4bit" not in model_ids

    def test_discover_vision_models_uses_default_cache_when_no_hf_home(self, monkeypatch):
        """Verify that vision models use default cache when HF_HOME not set."""
        mock_all_models = [
            {"model_id": "mlx-community/Llama-3.2-11B-Vision-Instruct-4bit", "ram_needed_gb": 24.0, "snapshot_path": None, "weight_count": None},
        ]

        mock_list_output = {
            "status": "success",
            "command": "list",
            "data": {
                "models": [
                    {"name": "mlx-community/Llama-3.2-11B-Vision-Instruct-4bit", "capabilities": ["text-generation", "chat", "vision"], "size_bytes": 12000000000},
                ],
                "count": 1
            },
            "error": None
        }

        with patch("live.test_utils.discover_mlx_models_in_user_cache", return_value=mock_all_models):
            with patch("subprocess.run") as mock_run:
                mock_run.return_value = MagicMock(
                    returncode=0,
                    stdout=json.dumps(mock_list_output)
                )
                # No HF_HOME set - should still work with default cache
                monkeypatch.delenv("HF_HOME", raising=False)

                from live.test_utils import discover_vision_models
                result = discover_vision_models()

                # Should return vision models (using default cache)
                assert len(result) == 1
                assert result[0]["model_id"] == "mlx-community/Llama-3.2-11B-Vision-Instruct-4bit"

    def test_discover_vision_models_handles_empty_portfolio(self):
        """Verify behavior when no models discovered."""
        with patch("live.test_utils.discover_mlx_models_in_user_cache", return_value=[]):
            from live.test_utils import discover_vision_models
            result = discover_vision_models()

            assert result == []

    def test_discover_vision_models_handles_subprocess_error(self, monkeypatch):
        """Verify fallback when mlxk list --json fails."""
        mock_all_models = [
            {"model_id": "mlx-community/Llama-3.2-11B-Vision-Instruct-4bit", "ram_needed_gb": 24.0, "snapshot_path": None, "weight_count": None},
        ]

        with patch("live.test_utils.discover_mlx_models_in_user_cache", return_value=mock_all_models):
            with patch("subprocess.run") as mock_run:
                mock_run.return_value = MagicMock(returncode=1)
                monkeypatch.setenv("HF_HOME", "/fake/cache")

                from live.test_utils import discover_vision_models
                result = discover_vision_models()

                # Should return empty (not fallback to all on error)
                assert result == []


class TestPortfolioStructure:
    """Verify that new functions return same structure as discover_mlx_models_in_user_cache."""

    def test_text_models_return_same_structure(self, monkeypatch):
        """Verify discover_text_models() returns same dict structure."""
        expected_structure = [
            {"model_id": "test-model", "ram_needed_gb": 5.0, "snapshot_path": None, "weight_count": None}
        ]

        mock_list_output = {
            "status": "success",
            "command": "list",
            "data": {"models": [{"name": "test-model", "capabilities": ["text-generation"]}], "count": 1},
            "error": None
        }

        with patch("live.test_utils.discover_mlx_models_in_user_cache", return_value=expected_structure):
            with patch("subprocess.run") as mock_run:
                mock_run.return_value = MagicMock(returncode=0, stdout=json.dumps(mock_list_output))
                monkeypatch.setenv("HF_HOME", "/fake/cache")

                from live.test_utils import discover_text_models
                result = discover_text_models()

                # Verify structure matches
                assert len(result) == 1
                assert "model_id" in result[0]
                assert "ram_needed_gb" in result[0]
                assert "snapshot_path" in result[0]
                assert "weight_count" in result[0]

    def test_vision_models_return_same_structure(self, monkeypatch):
        """Verify discover_vision_models() returns same dict structure."""
        expected_structure = [
            {"model_id": "test-vision-model", "ram_needed_gb": 24.0, "snapshot_path": None, "weight_count": None}
        ]

        mock_list_output = {
            "status": "success",
            "command": "list",
            "data": {"models": [{"name": "test-vision-model", "capabilities": ["text-generation", "vision"]}], "count": 1},
            "error": None
        }

        with patch("live.test_utils.discover_mlx_models_in_user_cache", return_value=expected_structure):
            with patch("subprocess.run") as mock_run:
                mock_run.return_value = MagicMock(returncode=0, stdout=json.dumps(mock_list_output))
                monkeypatch.setenv("HF_HOME", "/fake/cache")

                from live.test_utils import discover_vision_models
                result = discover_vision_models()

                # Verify structure matches
                assert len(result) == 1
                assert "model_id" in result[0]
                assert "ram_needed_gb" in result[0]
                assert "snapshot_path" in result[0]
                assert "weight_count" in result[0]


class TestKnownBrokenIsCapabilityScoped:
    """A break on one capability must not cost coverage on another.

    Regression guard for the defect these tests could not see before: the
    exclusion used to be applied inside the shared base discovery, which feeds
    both the text and the vision axis, so a model with a broken text loader
    silently lost its (working) vision coverage too.
    """

    WORKSPACE = "/ws/some-multimodal-4bit"          # absolute path -> basename arm
    CACHE = "mlx-community/some-text-model-4bit"    # org/name       -> exact arm

    POLICY = {
        "some-multimodal-4bit": {
            "breaks": {"chat"},
            "condition": "unit fixture: text loader fails, vision path works",
        },
        CACHE: {
            "breaks": {"chat"},
            "condition": "unit fixture: pure text model, load fails",
        },
    }

    ALL_MODELS = [
        {"model_id": WORKSPACE, "ram_needed_gb": 6.0, "snapshot_path": None, "weight_count": None},
        {"model_id": CACHE, "ram_needed_gb": 9.0, "snapshot_path": None, "weight_count": None},
        {"model_id": "mlx-community/healthy-vision-4bit", "ram_needed_gb": 8.0, "snapshot_path": None, "weight_count": None},
    ]

    LIST_JSON = {
        "status": "success",
        "command": "list",
        "data": {
            "models": [
                {"name": WORKSPACE, "capabilities": ["text-generation", "chat", "vision"], "size_bytes": 5_000_000_000},
                {"name": CACHE, "capabilities": ["text-generation", "chat"], "size_bytes": 9_000_000_000},
                {"name": "mlx-community/healthy-vision-4bit", "capabilities": ["text-generation", "chat", "vision"], "size_bytes": 7_000_000_000},
            ],
            "count": 3,
        },
        "error": None,
    }

    @pytest.fixture(autouse=True)
    def _policy(self, monkeypatch):
        """Swap in a synthetic policy so the tests do not depend on the real list."""
        import live.test_utils as tu
        monkeypatch.setattr(tu, "KNOWN_BROKEN_MODELS", self.POLICY)

    def _axis(self, fn_name, monkeypatch):
        monkeypatch.setenv("HF_HOME", "/fake/cache")
        with patch("live.test_utils.discover_mlx_models_in_user_cache", return_value=self.ALL_MODELS):
            with patch("subprocess.run") as mock_run:
                mock_run.return_value = MagicMock(returncode=0, stdout=json.dumps(self.LIST_JSON))
                import live.test_utils as tu
                return [m["model_id"] for m in getattr(tu, fn_name)()]

    def test_chat_break_does_not_cost_vision_coverage(self, monkeypatch):
        """The whole point: chat-broken but vision-capable stays on the vision axis."""
        assert self.WORKSPACE not in self._axis("discover_text_models", monkeypatch)
        assert self.WORKSPACE in self._axis("discover_vision_models", monkeypatch)

    def test_pure_text_chat_break_is_excluded_from_text(self, monkeypatch):
        """An exact org/name entry still excludes on its own axis."""
        assert self.CACHE not in self._axis("discover_text_models", monkeypatch)

    def test_unlisted_model_is_untouched(self, monkeypatch):
        assert "mlx-community/healthy-vision-4bit" in self._axis("discover_vision_models", monkeypatch)

    def test_text_fallback_paths_still_apply_the_policy(self, monkeypatch):
        """A failing `mlxk list` must not silently restore an unfiltered portfolio."""
        monkeypatch.setenv("HF_HOME", "/fake/cache")
        with patch("live.test_utils.discover_mlx_models_in_user_cache", return_value=self.ALL_MODELS):
            with patch("subprocess.run") as mock_run:
                mock_run.return_value = MagicMock(returncode=1, stdout="")
                import live.test_utils as tu
                assert self.CACHE not in [m["model_id"] for m in tu.discover_text_models()]

    def test_base_discovery_carries_no_policy(self, monkeypatch):
        """The shared base returns mlxk's raw verdict — no mock of it here."""
        monkeypatch.setenv("HF_HOME", "/fake/cache")
        payload = {
            "status": "success",
            "command": "list",
            "data": {"models": [{
                "name": self.CACHE, "framework": "MLX", "health": "healthy",
                "runtime_compatible": True, "model_type": "chat",
                "size_bytes": 9_000_000_000,
            }], "count": 1},
            "error": None,
        }
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=0, stdout=json.dumps(payload))
            import live.test_utils as tu
            discovered = [m["model_id"] for m in tu.discover_mlx_models_in_user_cache()]
        assert discovered == [self.CACHE]


class TestKnownBrokenPolicyShape:
    """Validate the REAL list (no synthetic policy patched in)."""

    def test_every_entry_names_capabilities_and_a_condition(self):
        from live.test_utils import BROKEN_CAPABILITIES, KNOWN_BROKEN_MODELS

        for model_id, entry in KNOWN_BROKEN_MODELS.items():
            assert set(entry["breaks"]) <= BROKEN_CAPABILITIES, model_id
            assert entry["breaks"], f"{model_id}: an entry breaking nothing excludes nothing"
            assert entry["condition"].strip(), f"{model_id}: measured condition required"

    def test_unknown_capability_raises_instead_of_silently_missing(self):
        from live.test_utils import is_known_broken

        with pytest.raises(ValueError, match="unknown capability"):
            is_known_broken("mlx-community/anything", "visual")

    def test_org_prefixed_entry_never_matches_a_workspace_path(self):
        """The basename is stripped off the query, never off the entry."""
        import live.test_utils as tu

        policy = {"mlx-community/foo-4bit": {"breaks": {"vision"}, "condition": "fixture"}}
        original = tu.KNOWN_BROKEN_MODELS
        tu.KNOWN_BROKEN_MODELS = policy
        try:
            assert tu.is_known_broken("mlx-community/foo-4bit", "vision") is True
            assert tu.is_known_broken("/ws/foo-4bit", "vision") is False
        finally:
            tu.KNOWN_BROKEN_MODELS = original


class TestTimeLimitStaging:
    """Tests for the size-staged time limits (test_utils.model_timeout()).

    Deterministic, no models: the size index is faked so the staging itself is
    what is under test. Why this staging exists at all: limits used to hang off
    the MODULE, so the file with the 90 s limit held models up to 29.7 GB while
    the file with 180 s held models up to 8.9 GB. The biggest model the RAM gate
    admits was structurally always the first to hit a wall.
    """

    def test_tier_boundaries_are_inclusive_upper_bounds(self):
        from live.test_utils import size_allowance_s

        assert size_allowance_s(0.18) == 40
        assert size_allowance_s(8.0) == 40      # boundary belongs to the tier below
        assert size_allowance_s(8.01) == 90
        assert size_allowance_s(20.0) == 90
        assert size_allowance_s(20.1) == 140
        assert size_allowance_s(29.65) == 140   # GLM-4.7-Flash-8bit, the measured case

    def test_unknown_size_gets_the_widest_tier(self):
        """What cannot be bounded must not be bounded tightly."""
        from live.test_utils import (
            LOAD_ALLOWANCE_S,
            size_allowance_s,
            UNKNOWN_SIZE_ALLOWANCE_S,
        )

        assert size_allowance_s(None) == UNKNOWN_SIZE_ALLOWANCE_S
        assert UNKNOWN_SIZE_ALLOWANCE_S == max(tier for _, tier in LOAD_ALLOWANCE_S)

    def test_rate_takes_over_above_the_table(self):
        """The table stops at the text RAM gate; the vision gate sits higher.

        Below the last bound the tier always wins, so a bigger model can never
        end up with a tighter limit than a smaller one.
        """
        from live.test_utils import size_allowance_s

        assert size_allowance_s(32.0) == 140            # still the tier
        assert size_allowance_s(44.8) > 140             # 0.70 x 64 GB vision gate
        sizes = [0.5, 4.0, 8.0, 12.0, 20.0, 26.0, 32.0, 40.0, 44.8, 60.0]
        allowances = [size_allowance_s(s) for s in sizes]
        assert allowances == sorted(allowances), allowances

    def test_base_is_the_lower_bound_never_the_limit(self):
        """The base encodes the work, so nothing ever gets tighter."""
        from live.test_utils import model_timeout, size_allowance_s

        for base in (30.0, 45.0, 60, 90, 180, 600):
            for size in (None, 0.8, 12.0, 29.65):
                assert model_timeout(base, "x") >= base
        # And the widening is exactly the tier, not a multiplier.
        assert model_timeout(90, None) == 90 + size_allowance_s(None)

    def test_size_resolves_by_full_name_and_by_basename(self):
        """Workspace models arrive as an absolute path, cache models as org/name."""
        import live.test_utils as tu

        tu._model_size_index.cache_clear()
        fake = {"mlx-community/Big-8bit": 29.65, "Big-8bit": 29.65}
        with patch.object(tu, "_model_size_index", lambda: fake):
            assert tu.model_size_gb("mlx-community/Big-8bit") == 29.65
            assert tu.model_size_gb("/workspace/models/Big-8bit") == 29.65
            assert tu.model_size_gb("mlx-community/Unknown-4bit") is None
            assert tu.model_size_gb(None) is None
        tu._model_size_index.cache_clear()

    def test_a_broken_index_widens_rather_than_narrows(self):
        """`mlxk list --json` failing must not produce tight limits."""
        import live.test_utils as tu

        tu._model_size_index.cache_clear()
        with patch.object(tu, "_model_size_index", dict):
            assert tu.model_timeout(90, "mlx-community/Anything") == 90 + tu.UNKNOWN_SIZE_ALLOWANCE_S
        tu._model_size_index.cache_clear()

    def test_index_survives_a_failing_list_call(self):
        import subprocess

        import live.test_utils as tu

        tu._model_size_index.cache_clear()
        with patch.object(subprocess, "run", side_effect=OSError("boom")):
            assert tu._model_size_index() == {}
        tu._model_size_index.cache_clear()

    def test_index_keeps_the_larger_model_on_a_basename_collision(self):
        """A cache copy and a workspace clone can share a basename."""
        import subprocess

        import live.test_utils as tu

        payload = {"data": {"models": [
            {"name": "mlx-community/twin-4bit", "size_bytes": 1 * 1024**3},
            {"name": "/ws/twin-4bit", "size_bytes": 9 * 1024**3},
        ]}}
        completed = MagicMock(returncode=0, stdout=json.dumps(payload))

        tu._model_size_index.cache_clear()
        with patch.object(subprocess, "run", return_value=completed):
            index = tu._model_size_index()
        tu._model_size_index.cache_clear()

        assert index["twin-4bit"] == pytest.approx(9.0)
        assert index["mlx-community/twin-4bit"] == pytest.approx(1.0)
