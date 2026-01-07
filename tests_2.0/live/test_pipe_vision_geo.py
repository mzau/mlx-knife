"""Vision→Geo pipe integration test (Session 72-75 validation).

Simple smoke test for the complete pipeline:
- Vision model with chunking (--chunk 1) for geo-test images
- Pipe to text model for geo-location inference

PASSED criteria (minimal):
- Both phases exit 0 (no crash)
- Output not empty
- Output contains geo-related terms (heuristic)

FAILED criteria:
- Process crash (non-zero exit)
- Empty output
- Import/model errors

Opt-in: pytest -m live_vision_pipe -v
Requires: HF_HOME with vision+text models, MLXK2_ENABLE_PIPES=1

See: TESTING-DETAILS.md for test strategy
"""

from __future__ import annotations

import subprocess
import sys
import os
from pathlib import Path
from typing import Dict, Any

import pytest

from .test_utils import should_skip_model

pytestmark = [pytest.mark.live, pytest.mark.live_vision_pipe, pytest.mark.slow]

# Test images (9 JPEGs in geo-test collection)
GEO_TEST_DIR = Path(__file__).parent.parent / "assets" / "geo-test"
GEO_IMAGES = sorted(GEO_TEST_DIR.glob("coll2_*.jpeg"))


def _pick_best_eligible_text_model(text_portfolio: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    """Select the best text model for geo-inference (RAM-aware, task-appropriate).

    Rationale: After vision model unloads (~12GB), we want the best available
    text model for geo-inference. Prefer general-purpose chat models over
    specialized models (coder, math) which lack geographic knowledge.

    Portfolio structure: dict of dicts with keys like 'text_00', 'text_01', etc.
    Each value has keys: id, ram_needed_gb, description, expected_issue
    """
    # Blacklist patterns for specialized models (not good for geo-inference)
    SPECIALIZED_PATTERNS = ["coder", "code", "math", "medical", "legal"]

    eligible = []
    # Portfolio is a dict - iterate over items
    for key, info in text_portfolio.items():
        should_skip, _ = should_skip_model(key, text_portfolio)
        if should_skip:
            continue

        model_id = info.get("id", "").lower()
        # Skip specialized models (coder/math/etc - poor geographic knowledge)
        if any(pattern in model_id for pattern in SPECIALIZED_PATTERNS):
            continue

        eligible.append((key, info))

    if not eligible:
        pytest.skip("No suitable general-purpose text models found in portfolio (RAM gating)")

    # Sort by RAM requirement (DESC) - larger general-purpose models = better geo knowledge
    # Use ram_needed_gb (from portfolio, not ram_mb!)
    eligible.sort(key=lambda x: x[1].get("ram_needed_gb", 0), reverse=True)

    return eligible[0][1]  # Return largest general-purpose model info dict


def _run_cli(args: list[str], stdin: str | None = None, timeout: int = 600) -> tuple[str, str, int]:
    """Run mlxk CLI as subprocess."""
    result = subprocess.run(
        [sys.executable, "-m", "mlxk2.cli"] + args,
        input=stdin,
        text=True,
        capture_output=True,
        timeout=timeout,
        env={**os.environ, "MLXK2_ENABLE_PIPES": "1"},
    )
    return result.stdout, result.stderr, result.returncode


class TestVisionGeoPipeline:
    """Integration test for Vision→Geo pipeline (Sessions 72-75)."""

    @pytest.fixture(scope="class")
    def vision_model_id(self):
        """Get vision model (hardcoded for now - pixtral only viable model)."""
        # TODO: Use vision_portfolio when more vision models are viable
        # Currently only pixtral works reliably (blacklist filters others)
        return "pixtral"

    @pytest.fixture(scope="class")
    def text_model_id(self, text_portfolio):
        """Get best (largest) eligible text model from portfolio (RAM-aware).

        Sequential loading strategy (Session 73): Vision model unloads first
        (~12GB freed), then text model loads. Pick largest available for quality.
        """
        model = _pick_best_eligible_text_model(text_portfolio)
        model_id = model.get("id")  # Portfolio uses 'id', not 'model_id'
        ram_gb = model.get("ram_needed_gb", "unknown")

        # Standard print (works with -s flag like all other tests)
        print(f"\n🌍 Vision→Geo Pipe: Selected text model: {model_id} (~{ram_gb:.1f}GB)")

        return model_id

    @pytest.fixture(scope="class")
    def check_prerequisites(self):
        """Check if pipe mode is enabled and images exist."""
        if not os.getenv("MLXK2_ENABLE_PIPES"):
            pytest.skip("Pipe mode gated by MLXK2_ENABLE_PIPES=1")

        if not GEO_IMAGES:
            pytest.skip(f"No geo-test images found in {GEO_TEST_DIR}")

        assert len(GEO_IMAGES) == 9, f"Expected 9 images, found {len(GEO_IMAGES)}"

    def test_vision_batch_processing_chunk_1(self, check_prerequisites, vision_model_id):
        """Test vision batch processing with chunk=1 (incremental output).

        Validates: ADR-012 Phase 1c, Sessions 73-75 fixes
        PASSED: Process succeeds, output not empty, multiple images mentioned
        """
        image_paths = [str(p) for p in GEO_IMAGES]

        args = [
            "run",
            vision_model_id,
            "--image", *image_paths,
            "--chunk", "1",
            "--max-tokens", "12000",
            "--prompt", (
                "Describe each image in best possible detail. "
                "Don't repeat unimportant camera information. "
                "Number images according to metadata image number."
            ),
        ]

        stdout, stderr, code = _run_cli(args, timeout=600)

        # Minimal criteria: Process succeeds and produces output
        assert code == 0, f"Vision phase failed: exit={code}\nstderr={stderr}"
        assert stdout.strip(), "Vision output is empty"

        # Heuristic: Output should mention multiple images (smoke test)
        image_mentions = sum(1 for i in range(1, 10) if f"Image {i}" in stdout or f"image {i}" in stdout.lower())
        assert image_mentions >= 5, f"Only {image_mentions}/9 images mentioned (expected most/all)"

    def test_vision_to_geo_pipe(self, check_prerequisites, vision_model_id, text_model_id):
        """Test complete Vision→Geo pipeline.

        Validates: Session 73 pipe stdin + --prompt, complete integration
        PASSED: Both phases succeed, geo output mentions location concepts
        """
        image_paths = [str(p) for p in GEO_IMAGES]

        # Phase 1: Vision descriptions
        vision_args = [
            "run",
            vision_model_id,
            "--image", *image_paths,
            "--chunk", "1",
            "--max-tokens", "12000",
            "--prompt", (
                "Describe each image in best possible detail. "
                "Don't repeat unimportant camera information. "
                "Number images according to metadata image number."
            ),
        ]

        vision_stdout, vision_stderr, vision_code = _run_cli(vision_args, timeout=600)

        assert vision_code == 0, f"Vision phase failed: {vision_stderr}"
        assert vision_stdout.strip(), "Vision output is empty"

        # Phase 2: Geo inference via pipe
        geo_args = [
            "run",
            text_model_id,
            "-",
            "--prompt", (
                "According to the location information - "
                "tell me the area where all the images have been made."
            ),
            "--max-tokens", "500",
        ]

        geo_stdout, geo_stderr, geo_code = _run_cli(geo_args, stdin=vision_stdout, timeout=300)

        assert geo_code == 0, f"Geo phase failed: exit={geo_code}\nstderr={geo_stderr}"
        assert geo_stdout.strip(), "Geo output is empty"

        # Heuristic: Output should mention location-related concepts (smoke test)
        # NOTE: We don't verify accuracy (no GOLD), just that pipe workflow functions
        geo_lower = geo_stdout.lower()
        has_location_terms = any(term in geo_lower for term in [
            "location", "area", "region", "place", "city", "country",
            "latitude", "longitude", "coordinates", "gps"
        ])

        assert has_location_terms, f"Geo output lacks location terms (pipe may have failed):\n{geo_stdout[:300]}"

    def test_vision_chunk_isolation_no_hallucination(self, check_prerequisites, vision_model_id):
        """Test chunk isolation with chunk=1 (Session 73 regression test).

        Validates: Fresh VisionRunner per chunk, no state leakage
        PASSED: Process succeeds, both images mentioned separately
        """
        # Test with only 2 images, chunk=1 (minimal isolation test)
        image_paths = [str(p) for p in GEO_IMAGES[:2]]

        args = [
            "run",
            vision_model_id,
            "--image", *image_paths,
            "--chunk", "1",
            "--max-tokens", "800",
            "--prompt", "Describe this image briefly.",
        ]

        stdout, stderr, code = _run_cli(args, timeout=240)

        # Minimal criteria: Process succeeds, output not empty, both batches present
        assert code == 0, f"exit={code}\nstderr={stderr}"
        assert stdout.strip(), "Output is empty"

        # Smoke test: Both batches should be visible (chunk workflow functioning)
        # NOTE: We don't verify isolation quality - just that 2 batches were processed
        assert "batch 1/2" in stdout.lower(), "Batch 1/2 not found (chunking failed?)"
        assert "batch 2/2" in stdout.lower(), "Batch 2/2 not found (chunking failed?)"
