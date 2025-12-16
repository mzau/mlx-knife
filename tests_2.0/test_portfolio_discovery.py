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

    def test_discover_text_models_returns_all_when_no_hf_home(self, monkeypatch):
        """Verify fallback behavior when HF_HOME not set."""
        mock_all_models = [
            {"model_id": "mlx-community/Qwen2.5-0.5B-Instruct-4bit", "ram_needed_gb": 1.0, "snapshot_path": None, "weight_count": None},
        ]

        with patch("live.test_utils.discover_mlx_models_in_user_cache", return_value=mock_all_models):
            # Unset HF_HOME
            monkeypatch.delenv("HF_HOME", raising=False)

            from live.test_utils import discover_text_models
            result = discover_text_models()

            # Should return all models (fallback)
            assert len(result) == 1
            assert result == mock_all_models

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

    def test_discover_vision_models_returns_empty_when_no_hf_home(self, monkeypatch):
        """Verify that vision models require HF_HOME."""
        mock_all_models = [
            {"model_id": "mlx-community/Llama-3.2-11B-Vision-Instruct-4bit", "ram_needed_gb": 24.0, "snapshot_path": None, "weight_count": None},
        ]

        with patch("live.test_utils.discover_mlx_models_in_user_cache", return_value=mock_all_models):
            monkeypatch.delenv("HF_HOME", raising=False)

            from live.test_utils import discover_vision_models
            result = discover_vision_models()

            # Should return empty (vision needs HF_HOME)
            assert result == []

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
