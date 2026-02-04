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
    def vision_model_id(self, vision_portfolio):
        """Get vision model from portfolio (pixtral preferred)."""
        # TODO: Use vision_portfolio when more vision models are viable
        # Currently only pixtral works reliably (blacklist filters others)
        # Session 133: Support any available pixtral variant (4bit, 8bit, etc.)
        for key, info in vision_portfolio.items():
            model_id = info.get("id", "")
            if "pixtral" in model_id.lower():
                return model_id
        # Fallback if no pixtral found
        pytest.skip("No pixtral model found in vision portfolio")

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

    def test_vision_batch_processing_chunk_1(self, check_prerequisites, vision_model_id, request):
        """Test vision batch processing with chunk=1 (incremental output).

        Validates: ADR-012 Phase 1c, Sessions 73-75 fixes, Session 93 chunk streaming
        PASSED: Process succeeds, output not empty, all chunks processed
        """
        image_paths = [str(p) for p in GEO_IMAGES]

        args = [
            "run",
            vision_model_id,
            "--image", *image_paths,
            "--chunk", "1",
            "--max-tokens", "12000",
            "--prompt", "Describe each image in best possible detail.",
        ]

        stdout, stderr, code = _run_cli(args, timeout=600)

        # Minimal criteria: Process succeeds and produces output
        assert code == 0, f"Vision phase failed: exit={code}\nstderr={stderr}"
        assert stdout.strip(), "Vision output is empty"

        # Session 93: With chunk=1, no image numbers in metadata (hallucination fix)
        # Instead, verify all chunks were processed by checking chunk markers
        chunk_markers = sum(1 for i in range(1, 10) if f"Chunk {i}/9" in stdout)
        assert chunk_markers == 9, f"Only {chunk_markers}/9 chunks found (expected all chunks processed)"

    def test_vision_to_geo_pipe(self, check_prerequisites, vision_model_id, text_model_id, request):
        """Test complete Vision→Geo pipeline.

        Validates: Session 73 pipe stdin + --prompt, complete integration
        PASSED: Both phases succeed, geo output mentions location concepts
        """
        import time
        import json
        from datetime import datetime, timezone

        image_paths = [str(p) for p in GEO_IMAGES]

        # Phase 1: Vision descriptions
        vision_start = time.time()
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
        vision_end = time.time()

        # Log Vision phase as sub-test
        if request.config.report_file:
            # Import schema version helper
            from conftest import _get_current_report_schema_version

            vision_entry = {
                "schema_version": _get_current_report_schema_version(),
                "timestamp": datetime.fromtimestamp(vision_end, timezone.utc).isoformat(),
                "mlx_knife_version": __import__("mlxk2").__version__,
                "test": f"{request.node.nodeid}[vision_phase]",
                "outcome": "passed" if vision_code == 0 else "failed",
                "duration": vision_end - vision_start,
                "model": {"id": vision_model_id, "size_gb": 12.6, "family": "pixtral"},
                "metadata": {"inference_modality": "vision"},
            }
            request.config.report_file.write(json.dumps(vision_entry) + "\n")
            request.config.report_file.flush()

        assert vision_code == 0, f"Vision phase failed: {vision_stderr}"
        assert vision_stdout.strip(), "Vision output is empty"

        # Phase 2: Geo inference via pipe
        text_start = time.time()
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
        text_end = time.time()

        # Log Text phase as sub-test
        # Note: size_gb lookup from portfolio would be ideal, but hardcoded for Mixtral-8x7B as fallback
        # TODO: Extract size_gb from portfolio when available (Session 80 follow-up)
        if request.config.report_file:
            # Best-effort size_gb lookup (Mixtral-8x7B is 24.5GB, but might vary by quantization)
            text_size_gb = 24.5 if "mixtral" in text_model_id.lower() else 0

            text_entry = {
                "schema_version": _get_current_report_schema_version(),
                "timestamp": datetime.fromtimestamp(text_end, timezone.utc).isoformat(),
                "mlx_knife_version": __import__("mlxk2").__version__,
                "test": f"{request.node.nodeid}[text_phase]",
                "outcome": "passed" if geo_code == 0 else "failed",
                "duration": text_end - text_start,
                "model": {"id": text_model_id, "size_gb": text_size_gb},
                "metadata": {"inference_modality": "text"},
            }
            request.config.report_file.write(json.dumps(text_entry) + "\n")
            request.config.report_file.flush()

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

    def test_vision_chunk_isolation_no_hallucination(self, check_prerequisites, vision_model_id, request):
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

        # Smoke test: Both chunks should be visible (chunk workflow functioning)
        # NOTE: We don't verify isolation quality - just that 2 chunks were processed
        assert "chunk 1/2" in stdout.lower(), "Chunk 1/2 not found (chunking failed?)"
        assert "chunk 2/2" in stdout.lower(), "Chunk 2/2 not found (chunking failed?)"
