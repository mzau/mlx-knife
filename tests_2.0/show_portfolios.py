#!/usr/bin/env python3
"""Show text and vision portfolios for inspection.

Usage:
    HF_HOME=/path/to/cache python tests_2.0/show_portfolios.py
"""

import sys
from pathlib import Path

# Add tests_2.0 to path
_tests_dir = Path(__file__).parent
if str(_tests_dir) not in sys.path:
    sys.path.insert(0, str(_tests_dir))

from live.test_utils import (
    discover_text_models,
    discover_vision_models,
    discover_mlx_models_in_user_cache,
    get_system_memory_bytes,
)


def format_size_gb(gb: float) -> str:
    """Format GB value for display."""
    if gb == float('inf'):
        return "∞ (SKIP)"
    return f"{gb:.1f} GB"


def main():
    import os

    # Check HF_HOME
    hf_home = os.environ.get("HF_HOME")
    if not hf_home:
        print("❌ HF_HOME not set. Please set HF_HOME to your model cache.")
        print("   Example: HF_HOME=/path/to/cache python tests_2.0/show_portfolios.py")
        sys.exit(1)

    print(f"🗂️  Model Cache: {hf_home}\n")

    # Get system memory
    system_memory_bytes = get_system_memory_bytes()
    system_memory_gb = system_memory_bytes / (1024**3) if system_memory_bytes else 0
    print(f"💾 System Memory: {system_memory_gb:.1f} GB")
    print(f"   70% Threshold: {system_memory_gb * 0.70:.1f} GB (Vision models above this will be skipped)\n")

    # Discover all models (legacy)
    print("=" * 80)
    print("📦 ALL MODELS (Legacy discover_mlx_models_in_user_cache)")
    print("=" * 80)
    all_models = discover_mlx_models_in_user_cache()

    if not all_models:
        print("   No models found in cache (or HF_HOME not set correctly)")
    else:
        print(f"   Total: {len(all_models)} models (Text + Vision mixed)\n")
        for i, model in enumerate(all_models):
            ram = format_size_gb(model["ram_needed_gb"])
            print(f"   {i:02d}. {model['model_id']}")
            print(f"       RAM (1.2x): {ram}")

    print("\n")

    # Discover text models
    print("=" * 80)
    print("📝 TEXT PORTFOLIO (discover_text_models)")
    print("=" * 80)
    text_models = discover_text_models()

    if not text_models:
        print("   No text models found")
    else:
        print(f"   Total: {len(text_models)} models (Vision filtered out)\n")
        for i, model in enumerate(text_models):
            ram = format_size_gb(model["ram_needed_gb"])
            print(f"   text_{i:02d}. {model['model_id']}")
            print(f"       RAM (1.2x text formula): {ram}")

    print("\n")

    # Discover vision models
    print("=" * 80)
    print("👁️  VISION PORTFOLIO (discover_vision_models)")
    print("=" * 80)
    vision_models = discover_vision_models()

    if not vision_models:
        print("   No vision models found")
    else:
        print(f"   Total: {len(vision_models)} models (Text filtered out)\n")
        for i, model in enumerate(vision_models):
            ram = format_size_gb(model["ram_needed_gb"])
            skip_marker = " ⚠️ WILL BE SKIPPED" if model["ram_needed_gb"] == float('inf') else ""
            print(f"   vision_{i:02d}. {model['model_id']}")
            print(f"       RAM (0.70 threshold vision formula): {ram}{skip_marker}")

    print("\n")

    # Summary
    print("=" * 80)
    print("📊 SUMMARY")
    print("=" * 80)
    print(f"   All Models (unfiltered):  {len(all_models)}")
    print(f"   Text Portfolio:           {len(text_models)}")
    print(f"   Vision Portfolio:         {len(vision_models)}")

    # "All" is mlxk's raw verdict and carries no test policy, so the axes do NOT
    # have to add up to it: a model can be excluded from one axis and kept on
    # another. Name every such model instead of printing a bare mismatch — an
    # unexplained warning is how a silent exclusion goes unnoticed.
    if all_models:
        from live.test_utils import KNOWN_BROKEN_MODELS, is_known_broken

        all_ids = {m["model_id"] for m in all_models}
        placed = {m["model_id"] for m in text_models} | {m["model_id"] for m in vision_models}
        unplaced = sorted(all_ids - placed)
        if unplaced:
            print(f"   ℹ️  {len(unplaced)} model(s) in All but on no axis:")
            for model_id in unplaced:
                entry = (KNOWN_BROKEN_MODELS.get(model_id)
                         or KNOWN_BROKEN_MODELS.get(model_id.rsplit("/", 1)[-1]))
                if entry:
                    print(f"        {model_id}")
                    print(f"          broken on {sorted(entry['breaks'])}: {entry['condition']}")
                else:
                    print(f"        {model_id}  (no known-broken entry — check capabilities)")

        cross = [m["model_id"] for m in vision_models
                 if is_known_broken(m["model_id"], "chat")]
        if cross:
            print(f"   ✅ {len(cross)} chat-broken model(s) kept on the vision axis:")
            for model_id in cross:
                print(f"        {model_id}")

    # Count skippable vision models
    if vision_models:
        skippable = sum(1 for m in vision_models if m["ram_needed_gb"] == float('inf'))
        if skippable > 0:
            print(f"   ⚠️  {skippable} vision model(s) exceed 70% threshold (will be skipped)")

    print()


if __name__ == "__main__":
    main()
