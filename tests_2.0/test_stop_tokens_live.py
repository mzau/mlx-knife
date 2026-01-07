"""Real-model stop token detection tests for Issue #32 (ADR-009).

This test suite validates stop token handling with real TEXT models that exhibit
known issues:
- MXFP4: Visible `<|end|>` tokens in output
- Qwen 2.5: Self-conversation (chat template role markers)
- Llama 3.2: Control baseline (should work correctly)

Test Strategy (ADR-009):
1. Phase 1: Baseline measurement (document broken behavior)
2. Phase 2: Fix validation (verify 2-LOC fix works)
3. Phase 3: Empirical mapping (document tokenizer configs)

Portfolio Discovery:
- Auto-discovers MLX TEXT chat models only (excludes Vision "chat+vision")
- Uses MLXRunner (mlx-lm) which cannot load Vision models (mllama etc.)
- See Portfolio Separation: live/test_utils.py for separated text/vision portfolios

Opt-in via: pytest -m live_stop_tokens
Requires: HF_HOME set to SSD cache (CoW same-volume requirement, ADR-007)

RAM Safety:
- Tests automatically skip models that exceed available RAM
- Progressive budget scaling: 40% (16GB), 50% (32GB), 60% (64GB), 70% (96GB+)
- Larger systems have lower relative overhead, enabling better RAM utilization
- See TESTING-DETAILS.md: "RAM-Aware Model Selection Strategy"
"""

from __future__ import annotations

import os
import sys
import pytest
import json
import subprocess
from pathlib import Path
from typing import Dict, Any, Optional
import importlib
import importlib.util

# Portfolio Separation reference: live/test_utils.py provides discover_text_models()
# but we can't use it here due to circular import (test_utils imports from this file)
# Instead, we fix discover_mlx_models_in_user_cache() to exclude Vision models directly

# Opt-in marker for live tests
pytestmark = [pytest.mark.live_stop_tokens, pytest.mark.slow]


@pytest.fixture(scope="module", autouse=True)
def _use_real_mlx_modules():
    """Ensure live tests use real mlx / mlx-lm without polluting the rest of the suite."""
    stub_path = Path(__file__).parent / "stubs"
    stub_path_str = str(stub_path)

    # Remove stub path from sys.path (if present) and remember to restore it later
    path_removed = False
    if stub_path_str in sys.path:
        sys.path = [p for p in sys.path if p != stub_path_str]
        path_removed = True

    # Remove stub modules from sys.modules so real modules can be imported
    removed_modules: Dict[str, Any] = {}
    for module_name, module in list(sys.modules.items()):
        module_file = getattr(module, "__file__", "") or ""
        if module_file and stub_path_str in module_file:
            removed_modules[module_name] = module
            sys.modules.pop(module_name, None)
    # Also clear any previously installed huggingface_hub shims
    removed_hf_modules: Dict[str, Any] = {}
    for module_name, module in list(sys.modules.items()):
        if module_name == "huggingface_hub" or module_name.startswith("huggingface_hub."):
            removed_hf_modules[module_name] = module
            sys.modules.pop(module_name, None)

    # Require real mlx / mlx-lm; skip entire module if not available
    missing_runtime = False
    if (
        importlib.util.find_spec("mlx.core") is None
        or importlib.util.find_spec("mlx_lm") is None
    ):
        missing_runtime = True
    else:
        try:
            huggingface_hub = importlib.import_module("huggingface_hub")
        except ImportError:
            missing_runtime = True
        else:
            if not hasattr(huggingface_hub, "snapshot_download"):
                for name, mod in removed_modules.items():
                    sys.modules[name] = mod
                for name, mod in removed_hf_modules.items():
                    sys.modules[name] = mod
                if path_removed and stub_path_str not in sys.path:
                    sys.path.insert(0, stub_path_str)
                pytest.skip(
                    "requires huggingface_hub.snapshot_download (install latest huggingface-hub)",
                    allow_module_level=True,
                )
    if missing_runtime:
        # Restore previous state before skipping so rest of suite still uses stubs
        sys.modules.update({name: mod for name, mod in removed_modules.items()
                            if name not in sys.modules})
        sys.modules.update({name: mod for name, mod in removed_hf_modules.items()
                            if name not in sys.modules})
        if path_removed and stub_path_str not in sys.path:
            sys.path.insert(0, stub_path_str)
        pytest.skip(
            "requires mlx / mlx-lm native runtime (Apple Silicon)",
            allow_module_level=True,
        )

    try:
        yield
    finally:
        # Restore stub modules for the remainder of the test run
        for name, module in removed_modules.items():
            sys.modules[name] = module
        for name, module in removed_hf_modules.items():
            sys.modules[name] = module

        # Ensure stub path is back at the front for unit tests
        if path_removed and stub_path_str not in sys.path:
            sys.path.insert(0, stub_path_str)

# HF_HOME is optional: Portfolio Discovery uses it if set, falls back to hardcoded TEST_MODELS
_HF_HOME = os.environ.get("HF_HOME")


def get_system_ram_gb() -> float:
    """Detect system RAM in GB (macOS portable)."""
    try:
        result = subprocess.run(
            ["sysctl", "hw.memsize"],
            capture_output=True,
            text=True,
            check=True
        )
        # Output: "hw.memsize: 68719476736"
        memsize_bytes = int(result.stdout.strip().split(":")[1].strip())
        return memsize_bytes / (1024**3)  # Convert to GB
    except Exception:
        # Fallback: assume minimum safe config (16GB)
        return 16.0


def get_safe_ram_budget_gb() -> float:
    """Get safe RAM budget for model loading (progressive scaling).

    Progressive budget strategy (relative overhead decreases with larger systems):
    - 16GB System: 40% budget (6.4GB) - high relative OS overhead
    - 32GB System: 50% budget (16GB) - moderate overhead
    - 64GB System: 60% budget (38.4GB) - low overhead
    - 96GB+ System: 70% budget (67GB+) - minimal overhead

    Rationale:
    - OS/System baseline overhead is ~4-6GB (relatively constant)
    - Larger systems have more headroom after OS overhead
    - Progressive scaling allows better utilization of high-RAM systems
    """
    system_ram = get_system_ram_gb()

    # Progressive budget scaling
    if system_ram >= 96:
        budget_ratio = 0.70  # 70% for 96GB+ systems
    elif system_ram >= 64:
        budget_ratio = 0.60  # 60% for 64GB systems
    elif system_ram >= 32:
        budget_ratio = 0.50  # 50% for 32GB systems
    else:
        budget_ratio = 0.40  # 40% for 16GB systems (conservative)

    safe_budget = system_ram * budget_ratio
    return safe_budget


def discover_mlx_models_in_user_cache() -> List[Dict[str, Any]]:
    """Discover MLX chat models via mlxk list --json (production command).

    Uses production CLI instead of duplicating cache scanning logic.
    Leverages official JSON API (docs/json-api-schema.json modelObject).

    Filters for:
    - Framework: MLX only (not GGUF/PyTorch)
    - Health: healthy only (static file integrity)
    - Runtime: runtime_compatible only (mlx-lm/mlx-vlm can load)
    - Type: chat models (TEXT + VISION, includes all model_type="chat")
    - Exclusions: KNOWN_BROKEN_MODELS (upstream runtime bugs)

    Note: Returns BOTH text and vision models. Caller must filter by capabilities
    if needed (e.g., portfolio_models fixture filters to TEXT-only).

    Returns:
        List of dicts with keys: model_id, ram_needed_gb, snapshot_path, weight_count
        Note: snapshot_path and weight_count set to None (not needed for tests)
    """
    import subprocess
    import json
    from mlxk2.core.model_resolution import resolve_model_for_operation
    from mlxk2.core.cache import get_current_model_cache, hf_to_cache_dir

    # Import blacklist (local import to avoid circular dependency)
    # KNOWN_BROKEN_MODELS is defined in tests_2.0/live/test_utils.py
    try:
        sys.path.insert(0, str(Path(__file__).parent / "live"))
        from test_utils import KNOWN_BROKEN_MODELS
        sys.path.pop(0)
    except ImportError:
        KNOWN_BROKEN_MODELS = set()  # Fallback if import fails

    # Check HF_HOME is set (required for mlxk list)
    env = os.environ.copy()
    if not env.get("HF_HOME"):
        return []

    try:
        # Call production mlxk list command
        result = subprocess.run(
            [sys.executable, "-m", "mlxk2.cli", "list", "--json"],
            capture_output=True,
            text=True,
            timeout=30,
            env=env  # Pass environment with HF_HOME
        )

        if result.returncode != 0:
            return []

        # Parse JSON response (docs/json-api-schema.json)
        data = json.loads(result.stdout)

        # Extract models array from response
        models = data.get("data", {}).get("models", [])

        # Filter per schema modelObject fields
        discovered = []
        for model in models:
            # Filter: MLX + healthy + runtime_compatible + chat (TEXT + VISION)
            model_type = model.get("model_type")
            is_chat = (
                isinstance(model_type, str) and
                model_type == "chat"  # Includes both text and vision chat models
            )
            if (model.get("framework") == "MLX" and
                model.get("health") == "healthy" and
                model.get("runtime_compatible") is True and
                is_chat):

                # RAM estimation: size_bytes * 1.2 overhead
                size_bytes = model.get("size_bytes", 0)
                ram_gb = (size_bytes / (1024**3)) * 1.2 if size_bytes else 0

                # Resolve to canonical cache name to avoid 404 during preload
                model_name = model["name"]
                try:
                    resolved_name, _, _ = resolve_model_for_operation(model_name)
                    if resolved_name:
                        model_name = resolved_name
                except Exception:
                    pass

                # FILTER: Exclude known broken models (upstream runtime bugs)
                if model_name in KNOWN_BROKEN_MODELS:
                    continue

                # Ensure cache directory exists (defensive against stale listings)
                try:
                    cache_dir = get_current_model_cache() / hf_to_cache_dir(model_name)
                    if not cache_dir.exists():
                        continue
                except Exception:
                    continue

                discovered.append({
                    "model_id": model_name,  # Canonical model ID
                    "ram_needed_gb": ram_gb,
                    "snapshot_path": None,      # Not provided by list, not needed
                    "weight_count": None        # Not provided by list, not needed
                })

        return discovered

    except Exception:
        # Robust: return empty list on any error (keeps tests runnable)
        return []


# Test models from ADR-009 with RAM requirements
# RAM estimates from TESTING-DETAILS.md: "RAM-Aware Model Selection Strategy"
TEST_MODELS = {
    "mxfp4": {
        "id": "mlx-community/gpt-oss-20b-MXFP4-Q8",
        "expected_issue": "visible_end_token",
        "description": "MXFP4 format with visible <|end|> in output",
        "ram_needed_gb": 12.0  # 20B MXFP4 (~12GB empirical)
    },
    "qwen25": {
        "id": "mlx-community/Qwen2.5-0.5B-Instruct-4bit",
        "expected_issue": "self_conversation",
        "description": "Qwen 2.5 generates chat template markers",
        "ram_needed_gb": 1.0  # 0.5B 4-bit (~1GB)
    },
    "llama32": {
        "id": "mlx-community/Llama-3.2-3B-Instruct-4bit",
        "expected_issue": None,
        "description": "Control baseline (should work correctly)",
        "ram_needed_gb": 4.0  # 3B 4-bit (~4GB)
    }
}


@pytest.fixture(scope="module")
def portfolio_models():
    """Dynamic TEXT model portfolio: discovered models OR hardcoded fallback.

    Discovers MLX TEXT chat models only (excludes Vision "chat+vision").
    Uses MLXRunner (mlx-lm) which cannot load Vision models.

    Enables portfolio testing when HF_HOME is set, falls back to
    3 hardcoded test models otherwise (backward compatibility).
    """
    all_models = discover_mlx_models_in_user_cache()  # Returns TEXT + VISION

    if all_models:
        # Filter to TEXT-only models (exclude Vision)
        # Vision models have "vision" in capabilities array (from mlxk list --json)
        import subprocess
        import json
        import os

        env = os.environ.copy()
        if env.get("HF_HOME"):
            try:
                result_data = subprocess.run(
                    [sys.executable, "-m", "mlxk2.cli", "list", "--json"],
                    capture_output=True,
                    text=True,
                    timeout=30,
                    env=env
                )
                if result_data.returncode == 0:
                    data = json.loads(result_data.stdout)
                    models_list = data.get("data", {}).get("models", [])
                    # Build set of vision model IDs
                    vision_ids = {m["name"] for m in models_list if "vision" in m.get("capabilities", [])}
                    # Filter out vision models
                    text_models = [m for m in all_models if m["model_id"] not in vision_ids]
                else:
                    text_models = all_models  # Fallback: include all
            except Exception:
                text_models = all_models  # Fallback: include all
        else:
            text_models = all_models  # No HF_HOME, use all

        # Convert discovered TEXT models to TEST_MODELS format
        result = {}
        for i, model in enumerate(text_models):
            key = f"discovered_{i:02d}"
            result[key] = {
                "id": model["model_id"],
                "ram_needed_gb": model["ram_needed_gb"],
                "expected_issue": None,  # Unknown for discovered models
                "description": f"Discovered: {model['model_id']} ({model.get('weight_count', 'unknown')} weights)"
            }

        print(f"\n🔍 Portfolio Discovery: Found {len(result)} MLX TEXT models in cache")
        return result
    else:
        # Fallback to hardcoded test models
        print(f"\n📋 Using hardcoded TEST_MODELS (3 models)")
        return TEST_MODELS


def should_skip_model(model_key: str, models_dict: Dict[str, Any] = None) -> tuple[bool, str]:
    """Check if model should be skipped due to insufficient RAM.

    Args:
        model_key: Key in models dictionary
        models_dict: Optional models dict (defaults to TEST_MODELS)

    Returns:
        (should_skip, reason)
    """
    if models_dict is None:
        models_dict = TEST_MODELS

    model_info = models_dict[model_key]
    ram_needed = model_info["ram_needed_gb"]
    ram_budget = get_safe_ram_budget_gb()
    system_ram = get_system_ram_gb()

    if ram_needed > ram_budget:
        budget_pct = int((ram_budget / system_ram * 100) if system_ram > 0 else 40)
        return (
            True,
            f"Model requires {ram_needed}GB but only {ram_budget:.1f}GB available "
            f"({budget_pct}% of {system_ram:.0f}GB system RAM). See TESTING-DETAILS.md RAM-Aware Model Selection."
        )
    return (False, "")

# Standard test prompt (simple, predictable)
TEST_PROMPT = "Write one sentence about cats."
MAX_TOKENS = 50


def pytest_generate_tests(metafunc):
    """Dynamically parametrize empirical mapping test with discovered model keys.

    This hook runs during test collection (before test execution).
    Enables process-per-model isolation: each model runs in separate pytest process.

    Architecture Decision (Session 56):
    - Prevents memory leak accumulation (71GB swap with 20 models in one process)
    - OS-level cleanup between models (process exit guarantees full cleanup)
    - Reflects real-world usage (users never load 20+ models sequentially)
    """
    if metafunc.function.__name__ == "test_empirical_mapping_single_model":
        # Lightweight discovery for parametrization (same logic as portfolio_models fixture)
        from live.test_utils import discover_mlx_models_in_user_cache

        all_models = discover_mlx_models_in_user_cache()

        if all_models:
            # Filter to TEXT-only models (exclude Vision) - same as portfolio_models fixture
            import json
            env = os.environ.copy()
            if env.get("HF_HOME"):
                try:
                    result_data = subprocess.run(
                        [sys.executable, "-m", "mlxk2.cli", "list", "--json"],
                        capture_output=True,
                        text=True,
                        timeout=30,
                        env=env
                    )
                    if result_data.returncode == 0:
                        data = json.loads(result_data.stdout)
                        models_list = data.get("data", {}).get("models", [])
                        vision_ids = {m["name"] for m in models_list if "vision" in m.get("capabilities", [])}
                        text_models = [m for m in all_models if m["model_id"] not in vision_ids]
                    else:
                        text_models = all_models
                except Exception:
                    text_models = all_models
            else:
                text_models = all_models

            # Generate model keys (discovered_00, discovered_01, ...)
            model_keys = [f"discovered_{i:02d}" for i in range(len(text_models))]
        else:
            # Fallback to hardcoded test models
            model_keys = list(TEST_MODELS.keys())

        # Parametrize with model keys (each key becomes a separate test)
        # ids= makes test names readable: test_empirical_mapping_single_model[discovered_00]
        metafunc.parametrize("model_key_param", model_keys, ids=lambda x: x)


class TestStopTokensValidation:
    """Validation: Verify stop token handling works correctly (Issue #32, ADR-009)."""

    @pytest.mark.live_stop_tokens
    def test_mxfp4_stop_token_filtering(self, request):
        """MXFP4: Stop tokens should be filtered correctly.

        After ADR-009 2-LOC fix (eos_token_id → eos_token_ids):
        - Model should stop cleanly without visible stop tokens
        - No `<|end|>` or `<|return|>` in output

        Background (Issue #32):
        - MXFP4 previously showed visible `<|end|>` tokens
        - Root cause: Runner only checked singular eos_token_id
        - Fix: Use eos_token_ids Set to handle multiple EOS tokens
        """
        # Only run when explicitly selected with -m live_stop_tokens or -m wet
        selected = request.config.getoption("-m") or ""
        if "live_stop_tokens" not in selected and "wet" not in selected:
            pytest.skip("Run with -m live_stop_tokens or -m wet to enable live model tests")

        # RAM Safety Check
        should_skip, reason = should_skip_model("mxfp4")
        if should_skip:
            pytest.skip(reason)

        from mlxk2.core.runner import MLXRunner

        model_id = TEST_MODELS["mxfp4"]["id"]

        # Run inference
        with MLXRunner(model_id) as runner:
            output = runner.generate_batch(
                prompt=TEST_PROMPT,
                max_tokens=MAX_TOKENS
            )

        # Validate clean output
        print(f"\n{'='*60}")
        print(f"VALIDATION: MXFP4")
        print(f"{'='*60}")
        print(f"Model: {model_id}")
        print(f"Prompt: {TEST_PROMPT}")
        print(f"Output: {output!r}")

        # Assert no visible stop tokens
        assert "<|end|>" not in output, "MXFP4 should filter <|end|> token"
        assert "<|return|>" not in output, "MXFP4 should filter <|return|> token"

        print("✓ MXFP4: Stop tokens correctly filtered")

    @pytest.mark.live_stop_tokens
    def test_qwen25_no_self_conversation(self, request):
        """Qwen 2.5: Should not generate chat template role markers (self-conversation).

        Self-Conversation Definition (ADR-009):
        - Model generates chat template role markers (User:, Assistant:, etc.)
        - Common patterns: '\nUser:', '\nAssistant:', '<|im_start|>user', '<|im_start|>assistant'
        - Specific to Qwen: '<|im_start|>', '<|im_end|>' markers

        Expected Behavior:
        - Model stops cleanly after its response
        - No chat template markers in output
        """
        # Only run when explicitly selected with -m live_stop_tokens or -m wet
        selected = request.config.getoption("-m") or ""
        if "live_stop_tokens" not in selected and "wet" not in selected:
            pytest.skip("Run with -m live_stop_tokens or -m wet to enable live model tests")

        # RAM Safety Check
        should_skip, reason = should_skip_model("qwen25")
        if should_skip:
            pytest.skip(reason)

        from mlxk2.core.runner import MLXRunner

        model_id = TEST_MODELS["qwen25"]["id"]

        # Run inference
        with MLXRunner(model_id) as runner:
            output = runner.generate_batch(
                prompt=TEST_PROMPT,
                max_tokens=MAX_TOKENS
            )

        # Validate clean output
        print(f"\n{'='*60}")
        print(f"VALIDATION: Qwen 2.5")
        print(f"{'='*60}")
        print(f"Model: {model_id}")
        print(f"Prompt: {TEST_PROMPT}")
        print(f"Output: {output!r}")

        # Check for self-conversation patterns
        generic_markers = ["\nUser:", "\nAssistant:", "\nHuman:", "\nAI:"]
        qwen_markers = ["<|im_start|>user", "<|im_start|>assistant", "<|im_start|>", "<|im_end|>"]

        found_generic = [m for m in generic_markers if m in output]
        found_qwen = [m for m in qwen_markers if m in output]

        print(f"Generic markers found: {found_generic}")
        print(f"Qwen markers found: {found_qwen}")

        # Assert no self-conversation
        assert not found_generic, f"Qwen 2.5 should not generate generic chat markers. Found: {found_generic}"
        assert not found_qwen, f"Qwen 2.5 should not generate Qwen-specific markers. Found: {found_qwen}"

        print("✓ Qwen 2.5: No self-conversation")

    @pytest.mark.live_stop_tokens
    def test_llama32_regression_control(self, request):
        """Llama 3.2: Regression control (should work correctly).

        Llama 3.2 has 3 eos_token_ids: [128008, 128001, 128009]
        This validates that the 2-LOC fix correctly handles multi-EOS models.

        Expected Behavior:
        - Clean output without visible stop tokens
        - No self-conversation
        - Serves as regression baseline
        """
        # Only run when explicitly selected with -m live_stop_tokens or -m wet
        selected = request.config.getoption("-m") or ""
        if "live_stop_tokens" not in selected and "wet" not in selected:
            pytest.skip("Run with -m live_stop_tokens or -m wet to enable live model tests")

        # RAM Safety Check
        should_skip, reason = should_skip_model("llama32")
        if should_skip:
            pytest.skip(reason)

        from mlxk2.core.runner import MLXRunner
        from mlxk2.core.cache import get_current_model_cache, hf_to_cache_dir
        from pathlib import Path

        model_id = TEST_MODELS["llama32"]["id"]

        # Check if model exists in cache
        cache = get_current_model_cache()
        model_dir = cache / hf_to_cache_dir(model_id)
        snapshots_dir = model_dir / "snapshots"
        if not snapshots_dir.exists() or not any(snapshots_dir.iterdir()):
            pytest.skip(f"Model not in cache: {model_id}")

        # Run inference
        with MLXRunner(model_id) as runner:
            output = runner.generate_batch(
                prompt=TEST_PROMPT,
                max_tokens=MAX_TOKENS
            )

        # Validate clean output
        print(f"\n{'='*60}")
        print(f"VALIDATION: Llama 3.2 (Regression Control)")
        print(f"{'='*60}")
        print(f"Model: {model_id}")
        print(f"Prompt: {TEST_PROMPT}")
        print(f"Output: {output!r}")

        # Llama 3.2 stop tokens
        llama_stop_tokens = ["<|eot_id|>", "</s>", "<|end_of_text|>"]
        found_stop = [t for t in llama_stop_tokens if t in output]
        assert not found_stop, f"Llama 3.2 should filter stop tokens. Found: {found_stop}"

        # No generic chat markers
        generic_markers = ["\nUser:", "\nAssistant:", "\nHuman:", "\nAI:"]
        found_markers = [m for m in generic_markers if m in output]
        assert not found_markers, f"Llama 3.2 should not self-converse. Found: {found_markers}"

        print("✓ Llama 3.2: Clean output (regression control passed)")


class TestStopTokensEmpiricalMapping:
    """Phase 3: Empirical mapping - document tokenizer configs and observed tokens."""

    @pytest.mark.live_stop_tokens
    def test_empirical_mapping_single_model(self, model_key_param, portfolio_models, request):
        """Document tokenizer configs and empirically observed stop tokens (ONE model per test).

        ARCHITECTURE DECISION (Session 56):
        - Each model runs in SEPARATE pytest process (process isolation)
        - OS guarantees complete memory cleanup between models
        - Prevents memory leak accumulation (71GB swap with 20 models in one process)
        - Reflects real-world usage (users never load 20 models sequentially)

        Uses portfolio_models fixture for dynamic model discovery.
        Each test writes JSONL fragment, final report generated by finalize test.

        Report Format (ADR-009):
        {
          "model": "gpt-oss",
          "configured_eos": ["<|return|>"],     # From tokenizer.eos_token
          "configured_eos_ids": [50256, ...],   # From tokenizer.eos_token_ids
          "generated_tokens": ["<|end|>", ...], # Empirically observed
          "workaround_needed": True/False
        }
        """
        # Only run when explicitly selected with -m live_stop_tokens or -m wet
        selected = request.config.getoption("-m") or ""
        if "live_stop_tokens" not in selected and "wet" not in selected:
            pytest.skip("Run with -m live_stop_tokens or -m wet to enable portfolio discovery")

        from mlxk2.core.runner import MLXRunner

        # Get model_key from pytest parametrize
        model_key = model_key_param
        model_info = portfolio_models[model_key]
        model_id = model_info["id"]

        system_ram = get_system_ram_gb()
        ram_budget = get_safe_ram_budget_gb()
        budget_ratio = ram_budget / system_ram if system_ram > 0 else 0.40

        # Skip models that exceed RAM budget
        should_skip, skip_reason = should_skip_model(model_key, portfolio_models)
        if should_skip:
            print(f"\nSkipping {model_key}: {skip_reason}")
            result = {
                "model_key": model_key,
                "model_id": model_id,
                "skipped": True,
                "skip_reason": skip_reason,
                "system_ram_gb": round(system_ram, 1),
                "ram_budget_gb": round(ram_budget, 1),
                "budget_ratio": round(budget_ratio, 2)
            }
        else:
            with MLXRunner(model_id) as runner:
                # Get tokenizer config
                tokenizer = runner.tokenizer

                # Extract configured stop tokens
                eos_token = getattr(tokenizer, "eos_token", None)
                eos_token_id = getattr(tokenizer, "eos_token_id", None)

                # Try to get eos_token_ids (Set or List)
                eos_token_ids = None
                if hasattr(tokenizer, "eos_token_ids"):
                    eos_token_ids = tokenizer.eos_token_ids
                    if hasattr(eos_token_ids, "__iter__"):
                        eos_token_ids = list(eos_token_ids)

                # Run inference to observe actual behavior
                output = runner.generate_batch(
                    prompt=TEST_PROMPT,
                    max_tokens=MAX_TOKENS
                )

                # Detect visible stop tokens
                potential_stop_tokens = ["<|end|>", "<|eot_id|>", "<|im_end|>", "<|endoftext|>"]
                found_stop_tokens = [t for t in potential_stop_tokens if t in output]

                result = {
                    "model_key": model_key,
                    "model_id": model_id,
                    "configured_eos_token": eos_token,
                    "configured_eos_token_id": eos_token_id,
                    "configured_eos_token_ids": eos_token_ids,
                    "generated_output": output[:100],  # First 100 chars for reference
                    "visible_stop_tokens": found_stop_tokens,
                    "workaround_needed": bool(found_stop_tokens),
                    "system_ram_gb": round(system_ram, 1),
                    "ram_budget_gb": round(ram_budget, 1),
                    "budget_ratio": round(budget_ratio, 2)
                }

        # Write JSONL fragment (append mode - each test writes one line)
        fragments_path = Path("stop_token_config_fragments.jsonl")
        with open(fragments_path, "a") as f:
            f.write(json.dumps(result) + "\n")

        print(f"\n{'='*60}")
        print(f"EMPIRICAL MAPPING: {model_key}")
        print(f"{'='*60}")
        print(json.dumps(result, indent=2))

    @pytest.mark.live_stop_tokens
    def test_empirical_mapping_generate_report(self, request):
        """Finalize: Aggregate JSONL fragments into final JSON report.

        Runs AFTER all single-model tests complete.
        Reads stop_token_config_fragments.jsonl and generates stop_token_config_report.json.
        """
        # Only run when explicitly selected
        selected = request.config.getoption("-m") or ""
        if "live_stop_tokens" not in selected and "wet" not in selected:
            pytest.skip("Run with -m live_stop_tokens or -m wet to enable portfolio discovery")

        fragments_path = Path("stop_token_config_fragments.jsonl")
        report_path = Path("stop_token_config_report.json")

        if not fragments_path.exists():
            pytest.skip("No fragments found - single-model tests may not have run")

        # Read all JSONL fragments
        fragments = []
        with open(fragments_path, "r") as f:
            for line in f:
                if line.strip():
                    fragments.append(json.loads(line))

        # Build final report
        report = {}

        # Extract system info from first fragment
        if fragments:
            first = fragments[0]
            report["_system_info"] = {
                "system_ram_gb": first.get("system_ram_gb", 0),
                "ram_budget_gb": first.get("ram_budget_gb", 0),
                "budget_ratio": first.get("budget_ratio", 0)
            }

        # Add all model results
        for fragment in fragments:
            model_key = fragment.pop("model_key")
            # Remove system_info fields from individual entries
            fragment.pop("system_ram_gb", None)
            fragment.pop("ram_budget_gb", None)
            fragment.pop("budget_ratio", None)
            report[model_key] = fragment

        # Write final JSON report
        report_path.write_text(json.dumps(report, indent=2))

        print(f"\n{'='*60}")
        print(f"EMPIRICAL MAPPING REPORT")
        print(f"{'='*60}")
        print(json.dumps(report, indent=2))
        print(f"\nReport saved to: {report_path.absolute()}")

        # Summary
        models_needing_fix = [
            k for k, v in report.items()
            if isinstance(v, dict) and v.get("workaround_needed")
        ]
        print(f"\nModels needing fix: {models_needing_fix}")

        # Cleanup fragments
        fragments_path.unlink()
        print(f"Cleaned up: {fragments_path}")
