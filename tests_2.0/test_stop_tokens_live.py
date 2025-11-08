"""Real-model stop token detection tests for Issue #32 (ADR-009).

This test suite validates stop token handling with real models that exhibit
known issues:
- MXFP4: Visible `<|end|>` tokens in output
- Qwen 2.5: Self-conversation (chat template role markers)
- Llama 3.2: Control baseline (should work correctly)

Test Strategy (ADR-009):
1. Phase 1: Baseline measurement (document broken behavior)
2. Phase 2: Fix validation (verify 2-LOC fix works)
3. Phase 3: Empirical mapping (document tokenizer configs)

Opt-in via: pytest -m live_stop_tokens
Requires: HF_HOME set to SSD cache (CoW same-volume requirement, ADR-007)

RAM Safety:
- Tests automatically skip models that exceed available RAM
- Progressive budget scaling: 40% (16GB), 50% (32GB), 60% (64GB), 70% (96GB+)
- Larger systems have lower relative overhead, enabling better RAM utilization
- See TESTING.md: "RAM-Aware Model Selection Strategy"
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
    """Discover MLX chat models in user HF cache (Category 2: read-only).

    Uses production infrastructure (mlxk2.operations.common) to filter:
    - Framework: MLX only (not GGUF/PyTorch)
    - Health: healthy only (not broken/incomplete)
    - Runtime: runtime_compatible only (mlx-lm can load)
    - Type: chat models only (for stop token testing)
    - RAM: estimated from file sizes (filtering in should_skip_model)

    Returns:
        List of dicts with keys: model_id, ram_needed_gb, snapshot_path, weight_count
    """
    hf_home = os.environ.get("HF_HOME")
    if not hf_home:
        return []

    hub_path = Path(hf_home) / "hub"
    if not hub_path.exists():
        return []

    # Use production infrastructure for filtering
    from mlxk2.operations.common import build_model_object
    from mlxk2.core.cache import cache_dir_to_hf

    discovered = []

    # Scan models--org--name directories
    for model_dir in hub_path.glob("models--*--*"):
        try:
            # Parse model_id from directory name
            model_id = cache_dir_to_hf(model_dir.name)

            # Find latest snapshot
            snapshots_dir = model_dir / "snapshots"
            if not snapshots_dir.exists():
                continue

            snapshot_dirs = [d for d in snapshots_dir.iterdir() if d.is_dir()]
            if not snapshot_dirs:
                continue

            # Most recent snapshot (by mtime)
            latest_snapshot = max(snapshot_dirs, key=lambda p: p.stat().st_mtime)

            # Use production filter logic (health + runtime + framework + type)
            model_obj = build_model_object(model_id, model_dir, latest_snapshot)

            # Filter: MLX + healthy + runtime_compatible + chat only
            if (model_obj.get("framework") != "MLX" or
                model_obj.get("health") != "healthy" or
                model_obj.get("runtime_compatible") is not True or
                model_obj.get("model_type") != "chat"):
                continue  # Skip non-chat/unhealthy/incompatible models

            # Estimate RAM (safetensors file sizes + 20% overhead)
            weight_files = list(latest_snapshot.glob("*.safetensors"))
            if not weight_files:
                continue

            total_bytes = sum(f.stat().st_size for f in weight_files if f.is_file())
            ram_gb = (total_bytes / (1024**3)) * 1.2

            discovered.append({
                "model_id": model_id,
                "ram_needed_gb": ram_gb,
                "snapshot_path": latest_snapshot,
                "weight_count": len(weight_files)
            })

        except Exception:
            # Skip broken models silently (keep portfolio discovery robust)
            continue

    return discovered


# Test models from ADR-009 with RAM requirements
# RAM estimates from TESTING.md: "RAM-Aware Model Selection Strategy"
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
    """Dynamic model portfolio: discovered models OR hardcoded fallback.

    Enables portfolio testing when HF_HOME is set, falls back to
    3 hardcoded test models otherwise (backward compatibility).
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
            f"({budget_pct}% of {system_ram:.0f}GB system RAM). See TESTING.md RAM-Aware Model Selection."
        )
    return (False, "")

# Standard test prompt (simple, predictable)
TEST_PROMPT = "Write one sentence about cats."
MAX_TOKENS = 50


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
        # Only run when explicitly selected with -m live_stop_tokens
        selected = request.config.getoption("-m") or ""
        if "live_stop_tokens" not in selected:
            pytest.skip("Run with -m live_stop_tokens to enable live model tests")

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
        # Only run when explicitly selected with -m live_stop_tokens
        selected = request.config.getoption("-m") or ""
        if "live_stop_tokens" not in selected:
            pytest.skip("Run with -m live_stop_tokens to enable live model tests")

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
        # Only run when explicitly selected with -m live_stop_tokens
        selected = request.config.getoption("-m") or ""
        if "live_stop_tokens" not in selected:
            pytest.skip("Run with -m live_stop_tokens to enable live model tests")

        # RAM Safety Check
        should_skip, reason = should_skip_model("llama32")
        if should_skip:
            pytest.skip(reason)

        from mlxk2.core.runner import MLXRunner

        model_id = TEST_MODELS["llama32"]["id"]

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
    def test_empirical_mapping_all_models(self, portfolio_models, request):
        """Document tokenizer configs and empirically observed stop tokens.

        Uses portfolio_models fixture for dynamic model discovery.
        Generates report: stop_token_config_report.json

        Report Format (ADR-009):
        {
          "model": "gpt-oss",
          "configured_eos": ["<|return|>"],     # From tokenizer.eos_token
          "configured_eos_ids": [50256, ...],   # From tokenizer.eos_token_ids
          "generated_tokens": ["<|end|>", ...], # Empirically observed
          "workaround_needed": True/False
        }
        """
        # Only run when explicitly selected with -m live_stop_tokens
        selected = request.config.getoption("-m") or ""
        if "live_stop_tokens" not in selected:
            pytest.skip("Run with -m live_stop_tokens to enable portfolio discovery")

        from mlxk2.core.runner import MLXRunner

        report = {}
        system_ram = get_system_ram_gb()
        ram_budget = get_safe_ram_budget_gb()

        # Calculate actual budget ratio used
        budget_ratio = ram_budget / system_ram if system_ram > 0 else 0.40

        # Add system info to report
        report["_system_info"] = {
            "system_ram_gb": round(system_ram, 1),
            "ram_budget_gb": round(ram_budget, 1),
            "budget_ratio": round(budget_ratio, 2)
        }

        for model_key, model_info in portfolio_models.items():
            model_id = model_info["id"]

            # Skip models that exceed RAM budget
            should_skip, skip_reason = should_skip_model(model_key, portfolio_models)
            if should_skip:
                print(f"\nSkipping {model_key}: {skip_reason}")
                report[model_key] = {
                    "model_id": model_id,
                    "skipped": True,
                    "skip_reason": skip_reason
                }
                continue

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

                report[model_key] = {
                    "model_id": model_id,
                    "configured_eos_token": eos_token,
                    "configured_eos_token_id": eos_token_id,
                    "configured_eos_token_ids": eos_token_ids,
                    "generated_output": output[:100],  # First 100 chars for reference
                    "visible_stop_tokens": found_stop_tokens,
                    "workaround_needed": bool(found_stop_tokens)
                }

        # Write report
        report_path = Path("stop_token_config_report.json")
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
