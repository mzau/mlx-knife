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


def _parse_model_family(model_id: str) -> tuple[str, str]:
    """Extract model family and variant from HuggingFace model ID.

    Examples:
        "mlx-community/Llama-3.2-3B-Instruct-4bit" → ("llama", "3.2-3b-instruct")
        "mlx-community/Qwen2.5-7B-Instruct-4bit" → ("qwen", "2.5-7b-instruct")
        "mlx-community/phi-3-mini-4k-instruct" → ("phi-3", "mini-4k-instruct")

    Args:
        model_id: HuggingFace model ID (org/name format)

    Returns:
        (family, variant) tuple. Returns ("unknown", model_name) if parsing fails.
    """
    # Extract model name from org/name
    model_name = model_id.split("/")[-1].lower()

    # Common patterns
    if "llama" in model_name:
        family = "llama"
        # Extract variant (everything after "llama-")
        variant = model_name.split("llama-", 1)[1] if "llama-" in model_name else model_name
        # Remove quantization suffix (-4bit, -8bit, etc.)
        variant = variant.replace("-4bit", "").replace("-8bit", "").replace("-fp16", "")
        return family, variant

    if "qwen" in model_name:
        family = "qwen"
        variant = model_name.split("qwen", 1)[1] if "qwen" in model_name else model_name
        variant = variant.replace("-4bit", "").replace("-8bit", "").replace("-fp16", "")
        return family, variant

    if "phi" in model_name:
        # Phi models: phi-3.5, phi-3, phi-2, etc.
        # Check most specific version first
        if "phi-3.5" in model_name:
            family = "phi-3.5"
            variant = model_name.split("phi-3.5-", 1)[1] if "phi-3.5-" in model_name else "base"
        elif "phi-3" in model_name:
            family = "phi-3"
            variant = model_name.split("phi-3-", 1)[1] if "phi-3-" in model_name else "base"
        elif "phi-2" in model_name:
            family = "phi-2"
            variant = model_name.split("phi-2-", 1)[1] if "phi-2-" in model_name else "base"
        else:
            family = "phi"
            variant = model_name
        variant = variant.replace("-4bit", "").replace("-8bit", "")
        return family, variant

    if "deepseek" in model_name:
        family = "deepseek"
        variant = model_name.replace("deepseek-", "")
        variant = variant.replace("-4bit", "").replace("-8bit", "")
        return family, variant

    if "mistral" in model_name or "mixtral" in model_name:
        family = "mistral" if "mistral" in model_name else "mixtral"
        variant = model_name.replace(f"{family}-", "")
        variant = variant.replace("-4bit", "").replace("-8bit", "")
        return family, variant

    # Fallback: unknown family
    return "unknown", model_name.replace("-4bit", "").replace("-8bit", "")


@pytest.fixture
def report_benchmark(request, model_info):
    """Helper for writing benchmark data to test reports (ADR-013 Phase 0).

    Simplifies adding model metadata and performance metrics to E2E test reports.
    Reports are written as JSONL via pytest_runtest_makereport hook.

    Usage:
        def test_something(report_benchmark, model_info):
            # ... test logic ...

            # Report model info only
            report_benchmark()

            # Report with performance metrics
            report_benchmark(performance={
                "tokens_per_sec": 45.2,
                "ram_peak_mb": 3200,
                "prompt_tokens": 15,
                "completion_tokens": 42
            })

            # Report with stop token data
            report_benchmark(stop_tokens={
                "configured": ["<|end|>"],
                "detected": ["<|end|>"],
                "workaround": "none",
                "leaked": False
            })

    Args:
        performance: Optional performance metrics dict
        stop_tokens: Optional stop token validation data
        **extra: Additional metadata (goes to metadata section)
    """
    def _report(performance: Dict[str, Any] = None, stop_tokens: Dict[str, Any] = None, **extra):
        # Extract model family/variant from model_id
        model_id = model_info["id"]
        family, variant = _parse_model_family(model_id)

        # Build model section (convert RAM estimate to disk size)
        # ram_needed_gb includes 1.2x overhead, so disk size = ram_needed_gb / 1.2
        disk_size_gb = model_info["ram_needed_gb"] / 1.2

        request.node.user_properties.append(("model", {
            "id": model_id,
            "size_gb": round(disk_size_gb, 2),
            "family": family,
            "variant": variant,
        }))

        # Add performance if provided
        if performance:
            request.node.user_properties.append(("performance", performance))

        # Add stop_tokens if provided
        if stop_tokens:
            request.node.user_properties.append(("stop_tokens", stop_tokens))

        # Add any extra metadata
        for key, value in extra.items():
            request.node.user_properties.append((key, value))

    return _report


# ============================================================================
# Benchmark Reporting (ADR-013 Phase 0)
# ============================================================================

def pytest_addoption(parser):
    """Add --report-output option for benchmark reporting."""
    parser.addoption(
        "--report-output",
        action="store",
        default=None,
        metavar="PATH",
        help="Generate benchmark reports to JSONL file (ADR-013 Phase 0)"
    )


def pytest_configure(config):
    """Initialize report file if --report-output is specified."""
    config.report_file = None
    if report_path := config.getoption("--report-output"):
        config.report_file = Path(report_path).open("a", encoding="utf-8")
        print(f"\n📊 Benchmark reporting enabled: {report_path}")


def pytest_unconfigure(config):
    """Close report file at end of session."""
    if config.report_file:
        config.report_file.close()


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_makereport(item, call):
    """Generate benchmark report for each test (if --report-output enabled).

    Reports are written as JSONL (one JSON object per line) to allow
    streaming and easy appending across test runs.

    Schema version: 0.1.0 (Phase 0 - Experimental)
    See: benchmarks/schemas/report-v0.1.schema.json
    """
    import json
    from datetime import datetime

    outcome = yield
    report = outcome.get_result()

    # Only report on test call phase (not setup/teardown)
    if call.when == "call" and item.config.report_file:
        try:
            # Import version here to avoid circular imports
            from mlxk2 import __version__
        except ImportError:
            __version__ = "unknown"

        # Build report data (required fields)
        data = {
            "schema_version": "0.1.0",
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "mlx_knife_version": __version__,
            "test": item.nodeid,
            "outcome": report.outcome,
        }

        # Add duration if available
        if hasattr(report, "duration"):
            data["duration"] = report.duration

        # Add skip reason for skipped tests
        if report.outcome == "skipped" and hasattr(report, "longrepr"):
            # Extract skip reason from longrepr tuple
            if isinstance(report.longrepr, tuple) and len(report.longrepr) >= 3:
                skip_reason = report.longrepr[2]
                data.setdefault("metadata", {})["skip_reason"] = skip_reason

        # Extract structured data from user_properties
        # Tests can add data via: request.node.user_properties.append(("key", value))
        for key, value in item.user_properties:
            if key in ("model", "performance", "stop_tokens", "system"):
                # Structured sections (top-level keys)
                data[key] = value
            else:
                # Everything else goes to metadata
                data.setdefault("metadata", {})[key] = value

        # Write JSONL (one line per report)
        try:
            item.config.report_file.write(json.dumps(data) + "\n")
            item.config.report_file.flush()
        except Exception as e:
            # Don't fail tests if reporting fails
            print(f"\n⚠️  Benchmark report write failed: {e}")

