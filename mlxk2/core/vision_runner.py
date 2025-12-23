"""
Vision runner wrapping mlx-vlm for Phase 1b (ADR-012).

Minimal, non-streaming implementation that mirrors the MLXRunner contract
well enough for CLI usage. Streaming is not guaranteed by mlx-vlm, so we
force batch mode and return the generated string.
"""

from __future__ import annotations

import hashlib
import os
import tempfile
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, Optional, Sequence, Tuple


@dataclass
class ExifData:
    """EXIF metadata extracted from image (optional, privacy-controlled)."""

    gps_lat: Optional[float] = None
    gps_lon: Optional[float] = None
    datetime: Optional[str] = None  # ISO 8601 format
    camera: Optional[str] = None


class VisionRunner:
    """Simple wrapper around mlx-vlm generate API."""

    def __init__(self, model_path: Path, model_name: str, verbose: bool = False):
        self.model_path = Path(model_path)
        self.model_name = model_name  # HF repo_id for mlx-vlm
        self.verbose = verbose
        self.model = None
        self.processor = None
        self.config = None
        self._generate = None
        self._load = None
        self._load_config = None
        self._apply_chat_template = None
        self._temp_files = []  # Track created temp files for cleanup

    def __enter__(self):
        self.load_model()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._cleanup_temp_files()
        return False

    def _cleanup_temp_files(self):
        """Remove all temporary image files created during generation."""
        import os

        for path in self._temp_files:
            try:
                if os.path.exists(path):
                    os.unlink(path)
            except Exception:
                # Ignore cleanup errors (best effort)
                pass
        self._temp_files.clear()

    def load_model(self):
        import os

        # Suppress HF progress bars during vision model loading (pull shows them)
        # Scoped suppression: restore previous state after loading
        prev_pbar = os.environ.get("HF_HUB_DISABLE_PROGRESS_BARS")
        os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
        try:
            self._load_model_impl()
        finally:
            if prev_pbar is None:
                os.environ.pop("HF_HUB_DISABLE_PROGRESS_BARS", None)
            else:
                os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = prev_pbar

    def _load_model_impl(self):
        """Internal model loading - called with progress bars suppressed."""
        try:
            import mlx_vlm  # type: ignore
            from mlx_vlm.utils import load_config  # type: ignore
            from mlx_vlm.prompt_utils import apply_chat_template  # type: ignore
        except Exception as e:  # pragma: no cover - exercised in integration runs
            raise RuntimeError(f"Failed to import mlx-vlm (vision backend): {e}") from e

        self._load = getattr(mlx_vlm, "load", None)
        self._generate = getattr(mlx_vlm, "generate", None)
        self._load_config = load_config
        self._apply_chat_template = apply_chat_template

        if self._load is None or self._generate is None:
            raise RuntimeError("mlx-vlm is missing load()/generate() API")

        # mlx-vlm expects HF repo_id, not local path
        # local_files_only=True: Use mlx-knife's cache only, never download (pull's responsibility)
        loaded = self._load(self.model_name, local_files_only=True)
        if isinstance(loaded, tuple):
            # Common pattern: (model, processor)
            self.model = loaded[0] if len(loaded) > 0 else None
            self.processor = loaded[1] if len(loaded) > 1 else None
        elif isinstance(loaded, dict):
            self.model = loaded.get("model") or loaded.get("vlm")
            self.processor = loaded.get("processor")
        else:
            self.model = loaded

        if self.model is None:
            raise RuntimeError("mlx-vlm load() returned no model")

        # Load config for chat template (local cache only)
        self.config = self._load_config(self.model_name, local_files_only=True)

    def _prepare_images(self, images: Sequence[Tuple[str, bytes]] | None):
        """
        Convert (filename, bytes) tuples to temporary file paths.

        mlx-vlm expects file paths as strings, not PIL objects.
        We write the image bytes to temporary files and return the paths.
        """
        if not images:
            return None

        image_paths = []
        for filename, raw in images:
            # Create a temporary file with appropriate extension
            suffix = Path(filename).suffix or ".jpg"
            tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
            tmp.write(raw)
            tmp.flush()
            tmp.close()
            image_paths.append(tmp.name)
            # Track temp file for cleanup
            self._temp_files.append(tmp.name)

        return image_paths

    def generate(
        self,
        prompt: str,
        images: Sequence[Tuple[str, bytes]] | None,
        max_tokens: Optional[int] = None,
        temperature: float = 0.4,
        top_p: float = 0.9,
        repetition_penalty: float = 1.0,
        image_id_map: Optional[Dict[str, int]] = None,
    ) -> str:
        """Generate a response with optional images. Non-streaming.

        Args:
            prompt: Text prompt for generation
            images: List of (filename, bytes) tuples
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_p: Top-p sampling
            repetition_penalty: Repetition penalty
            image_id_map: Optional mapping of content_hash -> image_id for stable
                         numbering across requests. If None, uses request-scoped IDs.
        """
        # Prepare image file paths
        image_paths = self._prepare_images(images)

        try:
            # Apply chat template (required for vision models)
            num_images = len(image_paths) if image_paths else 0
            formatted_prompt = self._apply_chat_template(
                self.processor, self.config, prompt, num_images=num_images
            )

            # Build generation kwargs
            gen_kwargs = {
                "verbose": self.verbose,
            }
            if max_tokens is not None:
                gen_kwargs["max_tokens"] = max_tokens
            if temperature is not None:
                gen_kwargs["temperature"] = temperature
            if top_p is not None:
                gen_kwargs["top_p"] = top_p
            if repetition_penalty is not None:
                gen_kwargs["repetition_penalty"] = repetition_penalty

            # Call mlx-vlm generate with correct API
            result = self._generate(
                self.model,
                self.processor,
                formatted_prompt,
                image_paths,  # List of file paths
                **gen_kwargs,
            )
            normalized = self._normalize_result(result)

            # Add filename mapping (even for single images - enables cross-model workflows)
            if images:
                normalized = self._add_filename_mapping(normalized, images, image_id_map)

            return normalized
        except Exception as e:
            raise RuntimeError(f"mlx-vlm generate() failed: {e}") from e
        finally:
            # Clean up temp files after generation (success or error)
            self._cleanup_temp_files()

    @staticmethod
    def _extract_exif(image_bytes: bytes) -> Optional[ExifData]:
        """
        Extract EXIF metadata from image bytes (optional, privacy-controlled).

        Feature flag: MLXK2_EXIF_METADATA=0 to disable (default: enabled)

        Returns:
            ExifData with GPS, DateTime, Camera info, or None if extraction disabled/failed
        """
        # Privacy: Can be disabled via MLXK2_EXIF_METADATA=0
        if os.environ.get("MLXK2_EXIF_METADATA") == "0":
            return None

        try:
            from PIL import Image
            from PIL.ExifTags import GPSTAGS
            import io

            img = Image.open(io.BytesIO(image_bytes))
            exif_data = img.getexif()

            if not exif_data:
                return None

            exif = ExifData()

            # Extract GPS coordinates (use get_ifd for GPS IFD, not get)
            try:
                gps_info = exif_data.get_ifd(34853)  # GPSInfo IFD
            except (KeyError, AttributeError):
                gps_info = None

            if gps_info:
                gps_dict = {}
                for key, val in gps_info.items():
                    tag = GPSTAGS.get(key, key)
                    gps_dict[tag] = val

                # Convert GPS coordinates to decimal degrees
                def convert_to_degrees(value):
                    """Convert GPS coordinate to decimal degrees."""
                    if not value or len(value) != 3:
                        return None
                    d, m, s = value
                    return float(d) + float(m) / 60.0 + float(s) / 3600.0

                lat = convert_to_degrees(gps_dict.get("GPSLatitude"))
                lon = convert_to_degrees(gps_dict.get("GPSLongitude"))

                if lat is not None and gps_dict.get("GPSLatitudeRef") == "S":
                    lat = -lat
                if lon is not None and gps_dict.get("GPSLongitudeRef") == "W":
                    lon = -lon

                exif.gps_lat = lat
                exif.gps_lon = lon

            # Extract DateTime (tag 36867 = DateTimeOriginal, 306 = DateTime)
            dt_original = exif_data.get(36867) or exif_data.get(306)
            if dt_original:
                try:
                    # EXIF format: "2023:12:06 12:19:21"
                    dt = datetime.strptime(str(dt_original), "%Y:%m:%d %H:%M:%S")
                    exif.datetime = dt.isoformat()  # Convert to ISO 8601
                except Exception:
                    pass

            # Extract Camera model (tag 272 = Model)
            camera = exif_data.get(272)
            if camera:
                exif.camera = str(camera).strip()

            # Return None if no useful EXIF found
            if all(x is None for x in [exif.gps_lat, exif.gps_lon, exif.datetime, exif.camera]):
                return None

            return exif

        except Exception:
            # Silently fail (EXIF extraction is optional)
            return None

    @staticmethod
    def _normalize_result(result) -> str:
        if result is None:
            return ""
        if isinstance(result, str):
            return result
        for attr in ("text", "response", "generated_text", "output"):
            try:
                val = getattr(result, attr)
                if isinstance(val, str):
                    return val
            except Exception:
                pass
        if isinstance(result, dict):
            for key in ("text", "response", "generated_text", "output"):
                val = result.get(key)
                if isinstance(val, str):
                    return val
        if isinstance(result, Iterable):
            return "".join(str(tok) for tok in result)
        return str(result)

    @staticmethod
    def _add_filename_mapping(
        result: str,
        images: Sequence[Tuple[str, bytes]],
        image_id_map: Optional[Dict[str, int]] = None,
    ) -> str:
        """Add filename mapping footer for multiple images (deterministic).

        Vision models reference images by position (Image 1, Image 2, etc.).
        This footer helps users map positions back to original filenames.

        The mapping is formatted as a Markdown table with an HTML comment marker
        '<!-- mlxk:filenames -->'. This makes it:
        1. Renders as a proper table in markdown-aware clients
        2. Easy to identify as server-generated (marker invisible but detectable)
        3. Parseable by clients that want to extract the mapping
        4. Unlikely to be reproduced by the model (specific HTML comment syntax)

        Enhanced in ADR-017 Phase 1:
        - Collapsible <details> wrapper (collapsed by default)
        - Optional EXIF metadata columns (GPS, DateTime, Camera)
        - Feature flag: MLXK2_EXIF_METADATA=1 enables EXIF extraction

        Args:
            result: Model output text
            images: List of (filename, bytes) tuples
            image_id_map: Optional mapping of content_hash -> image_id for stable
                         numbering. If None, uses request-scoped sequential IDs.

        Returns:
            Result with appended filename mapping
        """
        # Extract EXIF data (optional, controlled by feature flag)
        exif_enabled = os.environ.get("MLXK2_EXIF_METADATA") != "0"
        exif_list = []
        if exif_enabled:
            for _, raw_bytes in images:
                exif_list.append(VisionRunner._extract_exif(raw_bytes))

        # Build table rows
        rows = []
        for i, (filename, raw_bytes) in enumerate(images, 1):
            if image_id_map:
                # Use history-based stable IDs
                content_hash = hashlib.sha256(raw_bytes).hexdigest()[:8]
                img_id = image_id_map.get(content_hash, i)  # Fallback to sequential
            else:
                # CLI mode: request-scoped sequential IDs
                img_id = i

            # Compute hashed filename for display
            content_hash = hashlib.sha256(raw_bytes).hexdigest()[:8]
            hashed_name = f"image_{content_hash}.jpeg"

            # Build row with optional EXIF columns
            row = f"| {img_id} | {hashed_name}"

            if exif_enabled:
                # EXIF mode enabled: Always show Original + metadata columns
                exif = exif_list[i - 1] if i <= len(exif_list) else None

                # Original filename (always show when exif_enabled)
                row += f" | {Path(filename).name}"

                if exif:
                    # GPS Location
                    if exif.gps_lat is not None and exif.gps_lon is not None:
                        lat_dir = "N" if exif.gps_lat >= 0 else "S"
                        lon_dir = "E" if exif.gps_lon >= 0 else "W"
                        row += f" | 📍 {abs(exif.gps_lat):.2f}°{lat_dir}, {abs(exif.gps_lon):.2f}°{lon_dir}"
                    else:
                        row += " | -"

                    # DateTime
                    if exif.datetime:
                        # Format: "2023-12-06T12:19:21" → "📅 2023-12-06"
                        date_only = exif.datetime.split("T")[0]
                        row += f" | 📅 {date_only}"
                    else:
                        row += " | -"

                    # Camera
                    if exif.camera:
                        row += f" | {exif.camera}"
                    else:
                        row += " | -"
                else:
                    # EXIF enabled but none found: show placeholders
                    row += " | - | - | -"

            row += " |"
            rows.append(row)

        # Format header based on EXIF mode
        if exif_enabled:
            header = "| Image | Filename | Original | Location | Date | Camera |"
            separator = "|-------|----------|----------|----------|------|--------|"
        else:
            header = "| Image | Filename |"
            separator = "|-------|----------|"

        # Build collapsible HTML details (collapsed by default)
        # The marker comment is preserved for backwards compatibility
        count = len(images)
        mapping = "\n\n<details>\n"
        mapping += f"<summary>📸 Image Metadata ({count} image{'s' if count != 1 else ''})</summary>\n\n"
        mapping += "<!-- mlxk:filenames -->\n"
        mapping += f"{header}\n"
        mapping += f"{separator}\n"
        mapping += "\n".join(rows) + "\n"
        mapping += "\n</details>\n"

        return result + mapping
