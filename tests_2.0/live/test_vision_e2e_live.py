"""
Live E2E tests for Vision functionality (ADR-012).

Tests deterministic vision queries with specific, verifiable answers
to validate actual image understanding (not just hallucination).

Requires:
- Python 3.10+ (mlx-vlm requirement)
- Vision model in cache (default: pixtral-12b-4bit, see VISION_TEST_MODELS)
- Test assets in tests_2.0/assets/
- HF_HOME optional (uses default cache if not set)

Run with:
    HF_HOME=/path/to/cache pytest -m live_e2e tests_2.0/live/test_vision_e2e_live.py
"""
import os
import sys
import pytest
import subprocess
from pathlib import Path

# Must match VISION_TEST_MODELS fallback (see tests_2.0/live/test_utils.py)
VISION_MODEL = "pixtral-12b-4bit"

# Vision support requires Python 3.10+ (mlx-vlm requirement)
pytestmark = [
    pytest.mark.live,
    pytest.mark.live_e2e,
    pytest.mark.skipif(
        sys.version_info < (3, 10),
        reason="Vision support requires Python 3.10+ (mlx-vlm dependency)"
    )
]


class TestVisionDeterministicQueries:
    """
    Test vision functionality with specific, verifiable queries.

    These tests use deterministic questions that have specific, expected answers
    to validate actual image understanding rather than hallucination.
    """

    @pytest.mark.benchmark_inference
    def test_chess_position_e6(self):
        """Test reading specific chess position (e6 = black king)."""
        result = subprocess.run(
            [
                "mlxk", "run", VISION_MODEL,
                "What is on field e6? Answer briefly.",
                "--image", "tests_2.0/assets/T2.png",
                "--max-tokens", "50",  # Increased to ensure full answer
                "--temperature", "0",  # Deterministic output to reduce hallucination variance
                "--no-stream"
            ],
            capture_output=True,
            text=True,
            timeout=180,
            env=os.environ,
        )
        assert result.returncode == 0, f"Command failed: {result.stderr}"
        output = result.stdout.strip().lower()
        # Expected: "black king" on e6 - either "black king" or just "black" (if truncated)
        assert "black" in output or "king" in output, f"Expected 'black' or 'king' in output: {result.stdout}"

    @pytest.mark.benchmark_inference
    def test_contract_name_extraction(self):
        """Test OCR: extract name from contract document."""
        result = subprocess.run(
            [
                "mlxk", "run", VISION_MODEL,
                "What name is on the contract?",
                "--image", "tests_2.0/assets/T4.png",
                "--max-tokens", "30",
                "--no-stream"
            ],
            capture_output=True,
            text=True,
            timeout=180,
            env=os.environ,
        )
        assert result.returncode == 0, f"Command failed: {result.stderr}"
        output = result.stdout.strip()
        # Expected: "John A. Smith" (exact text from contract)
        assert "John" in output, f"Expected 'John' in output: {result.stdout}"
        assert "Smith" in output, f"Expected 'Smith' in output: {result.stdout}"

    @pytest.mark.benchmark_inference
    def test_mug_color_identification(self):
        """Test color recognition: blue mug."""
        result = subprocess.run(
            [
                "mlxk", "run", VISION_MODEL,
                "What color is the mug?",
                "--image", "tests_2.0/assets/T1.png",
                "--max-tokens", "20",
                "--no-stream"
            ],
            capture_output=True,
            text=True,
            timeout=180,
            env=os.environ,
        )
        assert result.returncode == 0, f"Command failed: {result.stderr}"
        output = result.stdout.strip().lower()
        # Expected: "blue"
        assert "blue" in output, f"Expected 'blue' in output: {result.stdout}"

    @pytest.mark.benchmark_inference
    def test_chart_axis_label_reading(self):
        """Test chart OCR: read Y-axis label."""
        result = subprocess.run(
            [
                "mlxk", "run", VISION_MODEL,
                "What is the Y-axis label?",
                "--image", "tests_2.0/assets/T6.png",
                "--max-tokens", "30",
                "--no-stream"
            ],
            capture_output=True,
            text=True,
            timeout=180,
            env=os.environ,
        )
        assert result.returncode == 0, f"Command failed: {result.stderr}"
        output = result.stdout.strip().lower()
        # Expected: "tokens/s" or "tokens per second"
        assert "token" in output, f"Expected 'token' in output: {result.stdout}"

    @pytest.mark.benchmark_inference
    def test_large_image_support(self):
        """Test that 2.7MB image (T2.png) is accepted (10MB limit)."""
        image_path = Path("tests_2.0/assets/T2.png")
        assert image_path.exists(), f"Test asset not found: {image_path}"

        # Verify image is indeed >2MB (old limit would have rejected it)
        size_mb = image_path.stat().st_size / (1024 * 1024)
        assert size_mb > 2.0, f"T2.png should be >2MB, got {size_mb:.1f}MB"
        assert size_mb < 10.0, f"T2.png should be <10MB, got {size_mb:.1f}MB"

        # Test that it's accepted and processed
        result = subprocess.run(
            [
                "mlxk", "run", VISION_MODEL,
                "What game is this?",
                "--image", str(image_path),
                "--max-tokens", "20",
                "--no-stream"
            ],
            capture_output=True,
            text=True,
            timeout=180,
            env=os.environ,
        )
        assert result.returncode == 0, f"Large image rejected: {result.stderr}"
        output = result.stdout.strip().lower()
        assert "chess" in output, f"Expected 'chess' in output: {result.stdout}"
