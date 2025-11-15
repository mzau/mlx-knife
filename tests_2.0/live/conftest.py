"""Shared fixtures for live E2E tests (ADR-011).

This conftest.py provides pytest fixtures for the live/ test package.
For utility functions and constants, see test_utils.py.
"""

from __future__ import annotations

import os
import sys
import pytest

# Prevent tokenizer fork warnings and potential deadlocks
# See: https://github.com/huggingface/tokenizers/issues/1047
os.environ["TOKENIZERS_PARALLELISM"] = "false"
from pathlib import Path
from typing import Dict, Any

# Import utilities from test_utils
from .test_utils import (
    discover_mlx_models_in_user_cache,
    TEST_MODELS,
)

# Import the real MLX modules fixture from parent test module
# This is needed for tests that use MLXRunner directly (e.g., streaming parity)
# The fixture is already decorated with @pytest.fixture in test_stop_tokens_live.py
# We just import and re-export it here so it's available to tests in this package
_parent_dir = Path(__file__).parent.parent
sys.path.insert(0, str(_parent_dir))
try:
    from test_stop_tokens_live import _use_real_mlx_modules
finally:
    sys.path.remove(str(_parent_dir))

# The imported fixture is now available to all tests in this package


@pytest.fixture(scope="function", autouse=True)
def _skip_unless_live_e2e_marker(request):
    """Auto-skip E2E tests unless -m live_e2e is explicitly used.

    E2E tests are marker-required (🔒) - they require real models and httpx.
    This fixture ensures they are skipped in the default pytest run.

    Exception: show_model_portfolio marker is allowed (convenience diagnostics).
    """
    # Check if test has live_e2e marker
    if request.node.get_closest_marker("live_e2e"):
        # Check if -m live_e2e or -m show_model_portfolio was specified
        selected_markers = request.config.getoption("-m") or ""
        if "live_e2e" not in selected_markers and "show_model_portfolio" not in selected_markers:
            pytest.skip("Run with -m live_e2e to enable E2E tests with real models")


def pytest_generate_tests(metafunc):
    """Generate parametrized tests for model_key parameter.

    If a test function has 'model_key' in its signature, this hook
    automatically parametrizes it over all models in the portfolio.
    This replaces the old loop-based approach (which caused RAM leaks)
    with pytest-native parametrization for proper test isolation.

    Each parametrized test gets its own server instance lifecycle,
    preventing accumulated RAM leaks from improper cleanup.

    IMPORTANT: This hook runs during COLLECTION phase. We check for
    live_e2e marker BEFORE doing portfolio discovery to avoid slow
    collection when marker is not requested (maintains marker-required 🔒).
    """
    if "model_key" in metafunc.fixturenames:
        # Check if live_e2e marker is requested (COLLECTION-TIME check)
        selected_markers = metafunc.config.getoption("-m") or ""
        if "live_e2e" not in selected_markers:
            # Parametrize with dummy value to allow collection
            # Tests will be skipped by _skip_unless_live_e2e_marker fixture
            # This prevents "fixture 'model_key' not found" errors
            metafunc.parametrize("model_key", ["_skipped"])
            return

        # Portfolio Discovery at collection time (uses subprocess mlxk list)
        discovered = discover_mlx_models_in_user_cache()

        if discovered:
            # Use discovered models - generate keys matching portfolio_models fixture
            model_keys = [f"discovered_{i:02d}" for i in range(len(discovered))]
        else:
            # Fallback to hardcoded test models
            model_keys = list(TEST_MODELS.keys())

        # Parametrize the test over all model keys
        metafunc.parametrize("model_key", model_keys)


@pytest.fixture(scope="module")
def portfolio_models():
    """Dynamic model portfolio: discovered models OR hardcoded fallback.

    Reuses Portfolio Discovery from ADR-009 (test_stop_tokens_live.py).
    Enables portfolio testing when HF_HOME is set, falls back to
    3 hardcoded test models otherwise (backward compatibility).

    Returns:
        Dict[str, Dict[str, Any]]: Model portfolio keyed by model_key
            {
                "discovered_00": {
                    "id": "mlx-community/Llama-3.2-3B-Instruct-4bit",
                    "ram_needed_gb": 4.0,
                    "expected_issue": None,
                    "description": "Discovered: ..."
                },
                ...
            }
    """
    discovered = discover_mlx_models_in_user_cache()

    if discovered:
        # Convert discovered models to TEST_MODELS format
        result = {}
        for i, model in enumerate(discovered):
            key = f"discovered_{i:02d}"
            result[key] = {
                "id": model["model_id"],
                "ram_needed_gb": model["ram_needed_gb"],
                "expected_issue": None,  # Unknown for discovered models
                "description": f"Discovered: {model['model_id']} ({model['weight_count']} weights)"
            }

        print(f"\n🔍 Portfolio Discovery: Found {len(result)} MLX models in cache")
        return result
    else:
        # Fallback to hardcoded test models
        print(f"\n📋 Using hardcoded TEST_MODELS (3 models)")
        return TEST_MODELS


@pytest.fixture
def model_info(portfolio_models, model_key):
    """Get model info for the current parametrized model_key.

    This fixture provides convenient access to model metadata in
    parametrized tests. It automatically looks up the model_key
    in the portfolio and returns the model info dict.

    Usage:
        def test_something(model_info):
            model_id = model_info["id"]
            ram_needed = model_info["ram_needed_gb"]
            ...

    Returns:
        Dict[str, Any]: Model metadata with keys:
            - id: Model ID (e.g., "mlx-community/Llama-3.2-3B-Instruct-4bit")
            - ram_needed_gb: Estimated RAM requirement
            - expected_issue: Known issue or None
            - description: Human-readable description
    """
    return portfolio_models[model_key]
