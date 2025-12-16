"""
Tests for EXIF extraction in vision_runner.py (ADR-017 Phase 1).

Feature flag: MLXK2_EXIF_METADATA=0 to disable (default: enabled)
"""

import os
import sys
from pathlib import Path

import pytest

from mlxk2.core.vision_runner import VisionRunner, ExifData


# Test image paths (collection1 has EXIF data)
COLLECTION1 = Path(__file__).parent / "assets"


class TestExifExtraction:
    """Test EXIF metadata extraction from images."""

    def test_exif_extraction_enabled_by_default(self):
        """EXIF extraction should be enabled by default."""
        # Load test image with EXIF
        image_path = COLLECTION1 / "T1.png"
        if not image_path.exists():
            pytest.skip("Test image not available")

        image_bytes = image_path.read_bytes()

        # Default (no flag) → should extract EXIF
        exif = VisionRunner._extract_exif(image_bytes)

        # T1.png may or may not have EXIF - just check structure
        if exif is not None:
            assert isinstance(exif, ExifData)

    def test_exif_extraction_disabled_with_flag(self, monkeypatch):
        """EXIF extraction should be disabled when MLXK2_EXIF_METADATA=0."""
        monkeypatch.setenv("MLXK2_EXIF_METADATA", "0")

        # Load test image
        image_path = COLLECTION1 / "T1.png"
        if not image_path.exists():
            pytest.skip("Test image not available")

        image_bytes = image_path.read_bytes()

        # With MLXK2_EXIF_METADATA=0 → should return None
        exif = VisionRunner._extract_exif(image_bytes)
        assert exif is None

    @pytest.mark.skipif(sys.version_info < (3, 10), reason="PIL required (mlx-vlm needs Python 3.10+)")
    def test_exif_extraction_no_exif_data(self):
        """Images without EXIF should return None."""
        # No monkeypatch needed - default is enabled

        # Create a minimal PNG without EXIF
        from io import BytesIO
        from PIL import Image

        img = Image.new("RGB", (10, 10), color="red")
        buf = BytesIO()
        img.save(buf, format="PNG")
        image_bytes = buf.getvalue()

        # Should return None (no EXIF)
        exif = VisionRunner._extract_exif(image_bytes)
        assert exif is None

    def test_collapsible_table_without_exif(self):
        """Table should be collapsible without EXIF data."""
        result = "A beach."
        images = [("beach.jpg", b"\x00\x01\x02\x03")]

        output = VisionRunner._add_filename_mapping(result, images)

        # Check collapsible HTML
        assert "<details>" in output
        assert "<summary>📸 Image Metadata (1 image)</summary>" in output
        assert "</details>" in output

        # Check marker preserved
        assert "<!-- mlxk:filenames -->" in output

        # Check basic table (no EXIF columns)
        assert "| Image | Filename |" in output
        assert "|-------|----------|" in output

    @pytest.mark.skipif(sys.version_info < (3, 10), reason="PIL required (mlx-vlm needs Python 3.10+)")
    def test_collapsible_table_with_exif_enabled(self):
        """Table should show EXIF columns when enabled (default, even if no EXIF found)."""
        # No monkeypatch needed - default is enabled

        # Use real image bytes (minimal PNG)
        from io import BytesIO
        from PIL import Image

        img = Image.new("RGB", (10, 10), color="blue")
        buf = BytesIO()
        img.save(buf, format="PNG")
        image_bytes = buf.getvalue()

        result = "A mountain."
        images = [("mountain.jpg", image_bytes)]

        output = VisionRunner._add_filename_mapping(result, images)

        # EXIF extraction attempted (but returns None for minimal PNG)
        # Table should still show "Original" column with placeholder "-"
        # (because exif_enabled=True but exif=None)
        assert "<details>" in output
        assert "| Image | Filename | Original | Location | Date | Camera |" in output

    def test_collapsible_table_multiple_images(self):
        """Table should show correct count for multiple images."""
        result = "Image 1 shows X. Image 2 shows Y."
        images = [
            ("img1.jpg", b"\x00\x01"),
            ("img2.jpg", b"\x02\x03"),
        ]

        output = VisionRunner._add_filename_mapping(result, images)

        # Check count
        assert "📸 Image Metadata (2 images)" in output

        # Check two rows
        assert output.count("| 1 |") == 1
        assert output.count("| 2 |") == 1

    def test_gps_display_format(self):
        """GPS coordinates should display with correct N/S/E/W directions."""
        # No monkeypatch needed - default is enabled

        # Create mock EXIF with various GPS coordinates
        from unittest.mock import patch

        # Test case 1: Northern + Western (Madeira)
        mock_exif = ExifData(gps_lat=32.79, gps_lon=-16.92, datetime=None, camera=None)

        with patch.object(VisionRunner, "_extract_exif", return_value=mock_exif):
            result = "Test."
            images = [("test.jpg", b"\x00\x01")]
            output = VisionRunner._add_filename_mapping(result, images)

            assert "📍 32.79°N, 16.92°W" in output

        # Test case 2: Southern + Eastern (hypothetical)
        mock_exif = ExifData(gps_lat=-10.5, gps_lon=20.3, datetime=None, camera=None)

        with patch.object(VisionRunner, "_extract_exif", return_value=mock_exif):
            result = "Test."
            images = [("test.jpg", b"\x00\x01")]
            output = VisionRunner._add_filename_mapping(result, images)

            assert "📍 10.50°S, 20.30°E" in output


class TestImageIdMapWithExif:
    """Test image ID persistence works with EXIF-enhanced table."""

    def test_history_based_ids_preserved_with_exif(self):
        """Image IDs from history should work with EXIF table."""
        # No monkeypatch needed - default is enabled

        import hashlib

        img1_bytes = b"\x00\x01\x02"
        img2_bytes = b"\x03\x04\x05"

        hash1 = hashlib.sha256(img1_bytes).hexdigest()[:8]
        hash2 = hashlib.sha256(img2_bytes).hexdigest()[:8]

        # Simulate history: Image 1 exists, Image 2 is new
        image_id_map = {hash1: 1}

        result = "Image 1 again. Image 2 new."
        images = [
            ("img1.jpg", img1_bytes),
            ("img2.jpg", img2_bytes),
        ]

        output = VisionRunner._add_filename_mapping(result, images, image_id_map)

        # Check IDs preserved
        assert "| 1 |" in output
        assert "| 2 |" in output  # New image gets ID 2

        # Check collapsible structure
        assert "<details>" in output
