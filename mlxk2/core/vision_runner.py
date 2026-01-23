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
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from ..operations.workspace import is_workspace_path


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

        # Check if model_path is a workspace directory
        if is_workspace_path(self.model_path):
            # Workspace path - use model_path directly
            model_ref = str(self.model_path)
            loaded = self._load(model_ref)  # No local_files_only needed for direct path
        else:
            # HF repo_id - use model_name with local_files_only
            # local_files_only=True: Use mlx-knife's cache only, never download (pull's responsibility)
            model_ref = self.model_name
            loaded = self._load(model_ref, local_files_only=True)

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

        # Load config for chat template (use same model_ref)
        self.config = self._load_config(model_ref, local_files_only=(model_ref == self.model_name))

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

    def _prepare_audio(self, audio: Sequence[Tuple[str, bytes]] | None):
        """
        Convert (filename, bytes) tuples to temporary file paths.

        mlx-vlm expects audio file paths as strings.
        We write the audio bytes to temporary files and return the paths.
        """
        if not audio:
            return None

        audio_paths = []
        for filename, raw in audio:
            # Create a temporary file with appropriate extension
            suffix = Path(filename).suffix or ".wav"
            tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
            tmp.write(raw)
            tmp.flush()
            tmp.close()
            audio_paths.append(tmp.name)
            # Track temp file for cleanup
            self._temp_files.append(tmp.name)

        return audio_paths

    def generate(
        self,
        prompt: str,
        images: Sequence[Tuple[str, bytes]] | None,
        audio: Sequence[Tuple[str, bytes]] | None = None,
        max_tokens: Optional[int] = None,
        temperature: float = 0.4,
        top_p: float = 0.9,
        repetition_penalty: float = 1.0,
        image_id_map: Optional[Dict[str, int]] = None,
        total_images: Optional[int] = None,
    ) -> str:
        """Generate a response with optional images and audio. Non-streaming.

        Args:
            prompt: Text prompt for generation
            images: List of (filename, bytes) tuples for images
            audio: List of (filename, bytes) tuples for audio files
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_p: Top-p sampling
            repetition_penalty: Repetition penalty
            image_id_map: Optional mapping of content_hash -> image_id for stable
                         numbering across requests. If None, uses request-scoped IDs.
            total_images: Total number of images in full batch (for chunking context)
        """
        # Prepare image and audio file paths
        image_paths = self._prepare_images(images)
        audio_paths = self._prepare_audio(audio)

        try:
            # Augment prompt with metadata context (GPS, datetime, camera, chunk info)
            augmented_prompt = self._augment_prompt_with_metadata(
                prompt, images, image_id_map, total_images
            )

            # Apply chat template (required for vision/audio models)
            num_images = len(image_paths) if image_paths else 0
            num_audios = len(audio_paths) if audio_paths else 0
            formatted_prompt = self._apply_chat_template(
                self.processor, self.config, augmented_prompt,
                num_images=num_images, num_audios=num_audios
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
                audio=audio_paths,  # List of audio file paths (None if no audio)
                **gen_kwargs,
            )
            normalized = self._normalize_result(result)

            # Add filename mapping (even for single images - enables cross-model workflows)
            if images:
                normalized = self._add_filename_mapping(normalized, images, image_id_map, total_images)

            return normalized
        except Exception as e:
            raise RuntimeError(f"mlx-vlm generate() failed: {e}") from e
        finally:
            # Clean up temp files after generation (success or error)
            self._cleanup_temp_files()

    @staticmethod
    def _augment_prompt_with_metadata(
        prompt: str,
        images: Sequence[Tuple[str, bytes]],
        image_id_map: Optional[Dict[str, int]],
        total_images: Optional[int],
    ) -> str:
        """Augment user prompt with image metadata context for better responses.

        Prepends metadata (GPS coordinates, datetime, camera) to the user prompt
        so the model can use this information in its response.

        Uses _extract_all_image_metadata() for Single Source of Truth.

        Feature flag: MLXK2_VISION_METADATA_CONTEXT=0 to disable (default: enabled)

        Args:
            prompt: User prompt
            images: List of (filename, bytes) tuples
            image_id_map: Mapping of content_hash -> image_id for stable numbering
            total_images: Total images in full batch (for chunking context)

        Returns:
            Augmented prompt with metadata context prepended
        """
        # Feature flag: Can be disabled
        if os.environ.get("MLXK2_VISION_METADATA_CONTEXT") == "0":
            return prompt

        if not images:
            return prompt

        # Single Source of Truth: Extract metadata once
        metadata_list = VisionRunner._extract_all_image_metadata(images, image_id_map)

        metadata_lines = []

        # DO NOT add chunk context - causes models to hallucinate missing images
        # Problem: "chunk 2/5" tells model 5 total exist → hallucinates others

        # Per-image metadata in bracket format
        # Strategy: Use LOCAL numbering within chunk to prevent hallucinations
        # - Single image (chunk_size=1): No number, just EXIF
        # - Multiple images: Local numbers (1, 2, 3...) not global
        for local_idx, meta in enumerate(metadata_list, 1):
            exif = meta['exif']

            # Build metadata string for this image
            meta_parts = []

            # Only add image reference if multiple images in this chunk
            if len(metadata_list) > 1:
                meta_parts.append(f"Image {local_idx}")  # Local numbering

            if exif:
                if exif.gps_lat is not None and exif.gps_lon is not None:
                    # Format GPS coordinates (4 decimals = ~11m precision for street-level accuracy)
                    lat_dir = "N" if exif.gps_lat >= 0 else "S"
                    lon_dir = "E" if exif.gps_lon >= 0 else "W"
                    meta_parts.append(
                        f"GPS: {abs(exif.gps_lat):.4f}°{lat_dir}, {abs(exif.gps_lon):.4f}°{lon_dir}"
                    )

                if exif.datetime:
                    # Format datetime (just date for brevity)
                    meta_parts.append(f"Date: {exif.datetime[:10]}")

                if exif.camera:
                    # Shorten camera name for token efficiency
                    camera = exif.camera.replace("(", "").replace(")", "").replace("generation", "gen")
                    meta_parts.append(f"Camera: {camera}")

            # Add metadata line if we have any metadata
            # (either image number for multi-image, or EXIF data, or both)
            if meta_parts:
                metadata_lines.append("[" + " | ".join(meta_parts) + "]")

        # Prepend metadata to prompt
        if metadata_lines:
            metadata_context = "\n".join(metadata_lines)
            return f"{metadata_context}\n\n{prompt}"
        else:
            return prompt

    @staticmethod
    def _extract_all_image_metadata(
        images: Sequence[Tuple[str, bytes]],
        image_id_map: Optional[Dict[str, int]] = None,
    ) -> List[Dict[str, Any]]:
        """Extract metadata for all images (Single Source of Truth).

        Central function that extracts all metadata once, used by both:
        - Bracket format (_augment_prompt_with_metadata)
        - HTML table format (_add_filename_mapping)

        Args:
            images: List of (filename, bytes) tuples
            image_id_map: Optional mapping of content_hash -> image_id for stable numbering

        Returns:
            List of metadata dicts, one per image:
            {
                'image_id': int,
                'filename': str,
                'content_hash': str,
                'exif': ExifData or None,
            }
        """
        metadata_list = []

        for idx, (filename, img_bytes) in enumerate(images, 1):
            # Calculate content hash
            content_hash = hashlib.sha256(img_bytes).hexdigest()[:8]

            # Determine image ID (stable across requests if image_id_map provided)
            if image_id_map:
                img_id = image_id_map.get(content_hash, idx)
            else:
                img_id = idx

            # Extract EXIF (respects MLXK2_EXIF_METADATA flag)
            exif = VisionRunner._extract_exif(img_bytes)

            metadata_list.append({
                'image_id': img_id,
                'filename': filename,
                'content_hash': content_hash,
                'exif': exif,
            })

        return metadata_list

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

            # Extract DateTime
            # Priority:
            # 1. GPS-Timestamp (Tag 7+29) - precise UTC, cannot be misconfigured
            # 2. DateTimeOriginal (Tag 36867) - fallback if no GPS
            # 3. Tag 306 (DateTime) - NEVER use, gets modified by image editors
            #
            # Phase 1: Date only (time component not displayed in metadata table)
            # Always UTC for consistency
            dt = None

            # Try GPS timestamp first (Tag 29 = GPSDateStamp, Tag 7 = GPSTimeStamp)
            if gps_info:
                gps_date = gps_dict.get("GPSDateStamp")  # "2025:05:11"
                gps_time = gps_dict.get("GPSTimeStamp")  # (12, 40, 22)

                if gps_date and gps_time:
                    try:
                        # Combine GPS date + time (always UTC)
                        gps_dt_str = f"{gps_date} {int(gps_time[0]):02d}:{int(gps_time[1]):02d}:{int(gps_time[2]):02d}"
                        dt = datetime.strptime(gps_dt_str, "%Y:%m:%d %H:%M:%S")
                    except Exception:
                        pass  # Fall through to DateTimeOriginal

            # Fallback: DateTimeOriginal (Tag 36867)
            if not dt:
                dt_original = exif_data.get(36867)
                if dt_original:
                    try:
                        # EXIF format: "2023:12:06 12:19:21"
                        dt = datetime.strptime(str(dt_original), "%Y:%m:%d %H:%M:%S")
                    except Exception:
                        pass

            if dt:
                exif.datetime = dt.isoformat()  # Convert to ISO 8601

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
        total_images: Optional[int] = None,
    ) -> str:
        """Add filename mapping header for multiple images (deterministic).

        Vision models reference images by position (Image 1, Image 2, etc.).
        This header helps users map positions back to original filenames.

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
            Result with prepended filename mapping (metadata before model output)
        """
        # Single Source of Truth: Extract metadata once
        metadata_list = VisionRunner._extract_all_image_metadata(images, image_id_map)

        # Check if EXIF is enabled
        exif_enabled = os.environ.get("MLXK2_EXIF_METADATA") != "0"

        # Build table rows from metadata
        rows = []
        for meta in metadata_list:
            img_id = meta['image_id']
            content_hash = meta['content_hash']
            filename = meta['filename']
            exif = meta['exif']

            hashed_name = f"image_{content_hash}.jpeg"

            # Build row with optional EXIF columns
            row = f"| {img_id} | {hashed_name}"

            if exif_enabled:
                # EXIF mode enabled: Always show Original + metadata columns
                # Original filename (always show when exif_enabled)
                row += f" | {Path(filename).name}"

                if exif:
                    # GPS Location
                    if exif.gps_lat is not None and exif.gps_lon is not None:
                        lat_dir = "N" if exif.gps_lat >= 0 else "S"
                        lon_dir = "E" if exif.gps_lon >= 0 else "W"
                        row += f" | 📍 {abs(exif.gps_lat):.4f}°{lat_dir}, {abs(exif.gps_lon):.4f}°{lon_dir}"
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
        mapping = "<details>\n"  # No leading newlines - this goes at beginning of output

        # Determine summary text (with chunk info if chunking is active)
        if total_images and total_images > count and image_id_map:
            # Chunking is active - show batch info
            chunk_ids = []
            for _, raw_bytes in images:
                content_hash = hashlib.sha256(raw_bytes).hexdigest()[:8]
                if content_hash in image_id_map:
                    chunk_ids.append(image_id_map[content_hash])

            if chunk_ids:
                start_id = min(chunk_ids)
                end_id = max(chunk_ids)
                chunk_size = len(images)
                batch_num = (start_id - 1) // chunk_size + 1
                total_batches = (total_images + chunk_size - 1) // chunk_size
                mapping += f"<summary>📸 Chunk {batch_num}/{total_batches}: Images {start_id}-{end_id}</summary>\n\n"
            else:
                # Fallback if chunk_ids calculation fails
                mapping += f"<summary>📸 Image Metadata ({count} image{'s' if count != 1 else ''})</summary>\n\n"
        else:
            # No chunking - standard summary
            mapping += f"<summary>📸 Image Metadata ({count} image{'s' if count != 1 else ''})</summary>\n\n"
        mapping += "<!-- mlxk:filenames -->\n"
        mapping += f"{header}\n"
        mapping += f"{separator}\n"
        mapping += "\n".join(rows) + "\n"
        mapping += "\n</details>\n\n"  # Spacing after metadata table

        # Metadata goes BEFORE model output (matches input order, clearer UX in chunking)
        return mapping + result
