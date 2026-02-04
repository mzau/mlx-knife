"""
Audio runner wrapping mlx-audio for STT transcription (ADR-020).

Dedicated AudioRunner for speech-to-text models (Whisper, Voxtral, VibeVoice).
Multimodal audio models (Gemma-3n, Qwen3-Omni) use VisionRunner instead.

Backend routing: config-based detection determines MLX_AUDIO vs MLX_VLM.
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

from ..operations.workspace import is_workspace_path


# ============================================================================
# CRITICAL: Monkey-patch mlx-audio tokenizer BEFORE any imports
# ============================================================================
# Workaround for mlx-audio Issue #479: tiktoken assets were removed in commit
# f7328a4 (Jan 29, 2026), but code still tries to load them.
#
# We bundle the assets from commit 9349644 in mlxk2/assets/whisper/ and patch
# get_encoding() to use them instead.
#
# This MUST happen at module import time, before any mlx-audio code runs!
# ============================================================================

def _apply_tiktoken_patch():
    """Apply tiktoken asset patch globally at module import time."""
    try:
        import base64
        import tiktoken
        from functools import lru_cache

        # Import tokenizer module first (but don't trigger get_encoding yet)
        import mlx_audio.stt.models.whisper.tokenizer as whisper_tokenizer

        # Get path to our bundled tiktoken assets
        assets_dir = Path(__file__).parent.parent / "assets" / "whisper"
        if not assets_dir.exists():
            # Assets not found - skip patching (will fall back to HF WhisperProcessor)
            return

        @lru_cache(maxsize=None)
        def patched_get_encoding(name: str = "gpt2", num_languages: int = 99):
            """Patched get_encoding using mlxk2's bundled tiktoken files."""
            vocab_path = assets_dir / f"{name}.tiktoken"

            if not vocab_path.exists():
                raise FileNotFoundError(
                    f"Tiktoken vocabulary file not found: {vocab_path}\n"
                    f"mlx-audio Issue #479: assets were removed in f7328a4"
                )

            with open(vocab_path) as fid:
                ranks = {
                    base64.b64decode(token): int(rank)
                    for token, rank in (line.split() for line in fid if line)
                }

            n_vocab = len(ranks)
            special_tokens = {}

            # Build special tokens (from mlx-audio tokenizer.py:343-358)
            from mlx_audio.stt.models.whisper.tokenizer import LANGUAGES

            specials = [
                "<|endoftext|>",
                "<|startoftranscript|>",
                *[f"<|{lang}|>" for lang in list(LANGUAGES.keys())[:num_languages]],
                "<|translate|>",
                "<|transcribe|>",
                "<|startoflm|>",
                "<|startofprev|>",
                "<|nospeech|>",
                "<|notimestamps|>",
                *[f"<|{i * 0.02:.2f}|>" for i in range(1501)],
            ]

            for token in specials:
                special_tokens[token] = n_vocab
                n_vocab += 1

            return tiktoken.Encoding(
                name=vocab_path.name,
                explicit_n_vocab=n_vocab,
                pat_str=r"""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""",
                mergeable_ranks=ranks,
                special_tokens=special_tokens,
            )

        # Patch the module globally
        whisper_tokenizer.get_encoding = patched_get_encoding

    except ImportError:
        # mlx-audio not installed - skip patching
        pass


# Apply patch immediately at module import
_apply_tiktoken_patch()


class AudioRunner:
    """Wrapper around mlx-audio STT API for dedicated transcription models.

    Supports:
    - Whisper variants (large-v3-turbo, base, small, etc.)
    - Voxtral (mini, small)
    - VibeVoice-ASR

    Usage:
        with AudioRunner(model_path, model_name, verbose) as runner:
            result = runner.transcribe(audio=[("file.wav", audio_bytes)])
    """

    def __init__(self, model_path: Path, model_name: str, verbose: bool = False):
        self.model_path = Path(model_path)
        self.model_name = model_name  # HF repo_id or workspace path
        self.verbose = verbose
        self.model = None
        self.processor = None
        self._generate_fn = None
        self._load_fn = None
        self._temp_files: List[str] = []  # Track created temp files for cleanup

    def __enter__(self):
        self.load_model()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._cleanup_temp_files()
        return False

    def _cleanup_temp_files(self):
        """Remove all temporary audio files created during transcription."""
        for path in self._temp_files:
            try:
                if os.path.exists(path):
                    os.unlink(path)
            except Exception:
                # Ignore cleanup errors (best effort)
                pass
        self._temp_files.clear()

    def load_model(self):
        """Load the audio model and processor.

        Supports both HF cache models and workspace paths.
        """
        # Suppress HF progress bars during loading (pull shows them)
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
            # Import mlx-audio STT module (0.3.0 API)
            # Note: tiktoken patch was already applied at module import time
            from mlx_audio.stt import load_model
            from mlx_audio.stt.generate import generate_transcription
        except ImportError as e:
            raise RuntimeError(
                f"Failed to import mlx-audio (audio backend): {e}\n"
                "Install with: pip install mlx-knife[audio]"
            ) from e

        self._generate_fn = generate_transcription
        self._load_fn = load_model

        # Check if model_path is a workspace directory
        if is_workspace_path(self.model_path):
            # Workspace path - load model directly
            model_ref = str(self.model_path)
            try:
                self.model = self._load_fn(model_ref)
                self.processor = None  # Processor handled internally
            except Exception as e:
                # Extract error details (some exceptions have empty messages)
                error_type = type(e).__name__
                error_msg = str(e) if str(e) else f"{error_type} (no details)"
                raise RuntimeError(f"Failed to load audio model from workspace: {error_msg}") from e
        else:
            # HF repo_id - defer loading to transcribe() (high-level API)
            self.model = None
            self.processor = None

    def _write_temp_audio(self, filename: str, audio_bytes: bytes) -> str:
        """Write audio bytes to a temporary file.

        mlx-audio expects file paths, not bytes. We write to temp files
        and track them for cleanup.

        Args:
            filename: Original filename (for extension detection)
            audio_bytes: Raw audio data

        Returns:
            Path to temporary file
        """
        suffix = Path(filename).suffix or ".wav"
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
        tmp.write(audio_bytes)
        tmp.flush()
        tmp.close()
        self._temp_files.append(tmp.name)
        return tmp.name

    def transcribe(
        self,
        audio: Sequence[Tuple[str, bytes]],
        prompt: Optional[str] = None,
        max_tokens: int = 4096,  # Ignored (Whisper generates full transcription)
        temperature: float = 0.0,
        language: Optional[str] = None,
    ) -> str:
        """Transcribe audio files to text.

        Args:
            audio: List of (filename, bytes) tuples for audio files
            prompt: Optional context for transcription (improves domain-specific accuracy)
            max_tokens: Ignored (Whisper generates full transcription automatically)
            temperature: Sampling temperature (0.0 = deterministic, best for accuracy)
            language: Language code (e.g., 'en', 'de'). Auto-detect if None.

        Returns:
            Transcription text. If MLXK2_AUDIO_SEGMENTS=1, includes segment table.
        """
        if not audio:
            return ""

        # Prepare audio file paths
        audio_paths = []
        for filename, audio_bytes in audio:
            path = self._write_temp_audio(filename, audio_bytes)
            audio_paths.append(path)

        try:
            all_transcriptions = []

            for audio_path in audio_paths:
                result = self._transcribe_single(
                    audio_path=audio_path,
                    prompt=prompt,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    language=language,
                )
                all_transcriptions.append(result)

            # Combine results (newline-separated for multiple files)
            combined = "\n\n".join(all_transcriptions)
            return combined.strip()

        except Exception as e:
            error_type = type(e).__name__
            error_msg = str(e) if str(e) else f"{error_type} (no details)"
            raise RuntimeError(f"mlx-audio transcribe() failed: {error_msg}") from e
        finally:
            # Clean up temp files after transcription
            self._cleanup_temp_files()

    def _transcribe_single(
        self,
        audio_path: str,
        prompt: Optional[str] = None,
        max_tokens: int = 4096,  # Ignored
        temperature: float = 0.0,
        language: Optional[str] = None,
    ) -> str:
        """Transcribe a single audio file.

        Uses generate_transcription() with either pre-loaded model (workspace)
        or model name (HF cache).
        """
        try:
            # Build kwargs for generate_transcription
            gen_kwargs = {
                "audio": audio_path,
                "verbose": self.verbose,
            }

            if is_workspace_path(self.model_path):
                # Workspace path - use pre-loaded model
                if self.model is None:
                    raise RuntimeError("Model not loaded. Call load_model() first.")
                gen_kwargs["model"] = self.model
            else:
                # HF repo_id - pass model name (handles loading internally)
                gen_kwargs["model"] = self.model_name

            # Add Whisper generation parameters (via **kwargs → model.generate())
            # These are filtered by generate_transcription() to match model.generate() signature
            if prompt:
                gen_kwargs["initial_prompt"] = prompt
            if temperature is not None:
                gen_kwargs["temperature"] = temperature
            if language:
                gen_kwargs["language"] = language

            # Optimize for batch STT (not streaming)
            # chunk_duration=30.0: Process 30s chunks (Whisper's max context window)
            # For 60min podcasts, this provides best accuracy vs latency balance
            gen_kwargs["chunk_duration"] = 30.0

            # Call generate_transcription
            result = self._generate_fn(**gen_kwargs)

            # Extract transcription text
            text = self._extract_text(result)

            # Optionally add segment metadata (MLXK2_AUDIO_SEGMENTS=1)
            if os.environ.get("MLXK2_AUDIO_SEGMENTS") == "1":
                segments = self._extract_segments(result)
                if segments:
                    text = self._add_segment_metadata(text, segments)

            return text

        except Exception as e:
            error_type = type(e).__name__
            error_msg = str(e) if str(e) else f"{error_type} (no details)"
            raise RuntimeError(f"Transcription failed for {audio_path}: {error_msg}") from e

    def _extract_text(self, result) -> str:
        """Extract transcription text from result object.

        mlx-audio returns various formats depending on model/version.
        """
        if result is None:
            return ""

        # String result
        if isinstance(result, str):
            return result

        # Dict with 'text' key
        if isinstance(result, dict):
            text = result.get("text", "")
            if isinstance(text, str):
                return text

        # Object with 'text' attribute
        if hasattr(result, "text"):
            text = result.text
            if isinstance(text, str):
                return text

        # Fallback: string conversion
        return str(result)

    def _extract_segments(self, result) -> Optional[List[Dict]]:
        """Extract segment data from result (if available).

        Whisper models provide segments with timestamps:
        [{"start": 0.0, "end": 2.34, "text": "..."}, ...]

        VibeVoice-ASR provides speaker diarization:
        [{"start_time": 0.0, "end_time": 2.5, "text": "...", "speaker_id": 0}, ...]
        """
        if result is None:
            return None

        segments = None

        # Dict with 'segments' key
        if isinstance(result, dict):
            segments = result.get("segments")

        # Object with 'segments' attribute
        elif hasattr(result, "segments"):
            segments = result.segments

        # Validate segments format
        if segments and isinstance(segments, list) and len(segments) > 0:
            # Check if first segment has expected keys
            first = segments[0]
            if isinstance(first, dict) and ("start" in first or "start_time" in first):
                return segments

        return None

    def _add_segment_metadata(self, text: str, segments: List[Dict]) -> str:
        """Add segment timestamps as collapsible HTML table.

        Format matches VisionRunner's image metadata table (collapsible).
        """
        count = len(segments)
        lines = [
            "<details>",
            f"<summary>Audio Segments ({count} segment{'s' if count != 1 else ''})</summary>",
            "",
            "| Start | End | Text |",
            "|-------|-----|------|",
        ]

        for seg in segments:
            # Handle both Whisper format (start/end) and VibeVoice format (start_time/end_time)
            start = seg.get("start") or seg.get("start_time", 0)
            end = seg.get("end") or seg.get("end_time", 0)
            seg_text = seg.get("text", "").strip()

            # Escape pipe characters in text
            seg_text = seg_text.replace("|", "\\|")

            lines.append(f"| {start:.2f}s | {end:.2f}s | {seg_text} |")

        lines.append("")
        lines.append("</details>")
        lines.append("")

        # Segments go after the transcription (metadata is supplementary)
        return text + "\n\n" + "\n".join(lines)
