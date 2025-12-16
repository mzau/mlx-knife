"""Unit tests for RAM calculation utilities (Test Portfolio Separation).

Tests the modularized RAM calculation functions for text and vision models.
"""

import sys
from pathlib import Path

# Add tests_2.0 to path for imports
_tests_dir = Path(__file__).parent
if str(_tests_dir) not in sys.path:
    sys.path.insert(0, str(_tests_dir))

from live.test_utils import (
    calculate_text_model_ram_gb,
    calculate_vision_model_ram_gb,
    get_system_memory_bytes,
)


class TestTextModelRAMCalculation:
    """Tests for text model RAM calculation."""

    def test_calculate_text_model_ram_gb_uses_1point2_multiplier(self):
        """Verify text models use 1.2x multiplier."""
        # 10 GB model
        size_bytes = 10 * (1024**3)
        ram_gb = calculate_text_model_ram_gb(size_bytes)

        # Should be 10 * 1.2 = 12 GB
        assert ram_gb == 12.0

    def test_calculate_text_model_ram_gb_small_model(self):
        """Test RAM calculation for small model (0.5GB)."""
        size_bytes = int(0.5 * (1024**3))
        ram_gb = calculate_text_model_ram_gb(size_bytes)

        # Should be 0.5 * 1.2 = 0.6 GB
        assert ram_gb == 0.6

    def test_calculate_text_model_ram_gb_zero_size(self):
        """Test RAM calculation for zero size model."""
        ram_gb = calculate_text_model_ram_gb(0)
        assert ram_gb == 0.0


class TestVisionModelRAMCalculation:
    """Tests for vision model RAM calculation."""

    def test_calculate_vision_model_ram_gb_below_threshold(self):
        """Verify vision models below 70% threshold return actual size."""
        # 64 GB system, 32 GB model (50% ratio, below 70%)
        system_memory_bytes = 64 * (1024**3)
        model_size_bytes = 32 * (1024**3)

        ram_gb = calculate_vision_model_ram_gb(model_size_bytes, system_memory_bytes)

        # Should return actual size (no 1.2x multiplier for vision)
        assert ram_gb == 32.0

    def test_calculate_vision_model_ram_gb_above_threshold(self):
        """Verify vision models above 70% threshold return infinity."""
        # 64 GB system, 48 GB model (75% ratio, above 70%)
        system_memory_bytes = 64 * (1024**3)
        model_size_bytes = 48 * (1024**3)

        ram_gb = calculate_vision_model_ram_gb(model_size_bytes, system_memory_bytes)

        # Should return infinity (signal to skip)
        assert ram_gb == float('inf')

    def test_calculate_vision_model_ram_gb_exactly_at_threshold(self):
        """Test vision model exactly at 70% threshold."""
        # 64 GB system, 44.8 GB model (exactly 70%)
        system_memory_bytes = 64 * (1024**3)
        model_size_bytes = int(64 * 0.70 * (1024**3))

        ram_gb = calculate_vision_model_ram_gb(model_size_bytes, system_memory_bytes)

        # Should return actual size (at threshold, not above)
        # Note: Due to integer conversion, might be slightly below 70%
        assert ram_gb < float('inf')

    def test_calculate_vision_model_ram_gb_zero_system_memory(self):
        """Test vision model calculation with zero system memory."""
        model_size_bytes = 10 * (1024**3)
        ram_gb = calculate_vision_model_ram_gb(model_size_bytes, 0)

        # Should return infinity (cannot determine)
        assert ram_gb == float('inf')

    def test_calculate_vision_model_ram_gb_small_model(self):
        """Test vision model calculation for small model."""
        # 64 GB system, 5 GB model (~8% ratio)
        system_memory_bytes = 64 * (1024**3)
        model_size_bytes = 5 * (1024**3)

        ram_gb = calculate_vision_model_ram_gb(model_size_bytes, system_memory_bytes)

        # Should return actual size (well below threshold)
        assert ram_gb == 5.0


class TestSystemMemoryRetrieval:
    """Tests for system memory retrieval."""

    def test_get_system_memory_bytes_returns_positive(self):
        """Verify get_system_memory_bytes() returns positive value on macOS."""
        memory_bytes = get_system_memory_bytes()

        # On macOS should return positive value (at least 4GB for modern systems)
        # On non-macOS or error, returns 0
        assert memory_bytes >= 0

        # If not zero, should be reasonable value (>= 4GB, <= 1TB)
        if memory_bytes > 0:
            assert memory_bytes >= 4 * (1024**3)  # At least 4GB
            assert memory_bytes <= 1024 * (1024**3)  # At most 1TB


class TestRAMCalculationDifferences:
    """Tests demonstrating the difference between text and vision RAM calculations."""

    def test_same_model_size_different_calculations(self):
        """Verify text and vision use different formulas for same model size."""
        # 10 GB model on 64 GB system
        size_bytes = 10 * (1024**3)
        system_memory_bytes = 64 * (1024**3)

        text_ram = calculate_text_model_ram_gb(size_bytes)
        vision_ram = calculate_vision_model_ram_gb(size_bytes, system_memory_bytes)

        # Text: 10 * 1.2 = 12 GB
        assert text_ram == 12.0

        # Vision: 10 GB (no multiplier, below 70% threshold)
        assert vision_ram == 10.0

        # Different results for same model size!
        assert text_ram != vision_ram

    def test_vision_blocks_where_text_allows(self):
        """Demonstrate vision blocks large models that text would allow."""
        # 50 GB model on 64 GB system
        size_bytes = 50 * (1024**3)
        system_memory_bytes = 64 * (1024**3)

        text_ram = calculate_text_model_ram_gb(size_bytes)
        vision_ram = calculate_vision_model_ram_gb(size_bytes, system_memory_bytes)

        # Text: 50 * 1.2 = 60 GB (would be skipped by RAM budget, but not infinity)
        assert text_ram == 60.0

        # Vision: 50/64 = 78% > 70% threshold → infinity (blocked)
        assert vision_ram == float('inf')

        # Vision is more conservative!
        assert vision_ram > text_ram
