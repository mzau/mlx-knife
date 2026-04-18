"""Structural invariants on the model_type frozensets in capabilities.py (ADR-023).

These tests fix the shape of the registry so silent drift can't reintroduce the
bug class that motivated ADR-023 (silent multimodal -> text downgrade).

Runtime lists (VISION_MODEL_TYPES, AUDIO_MODEL_TYPES, STT_MODEL_TYPES) and the
quantize list (VISION_QUANTIZE_TYPES) are deliberately curated independently:
runtime-capable is not the same as quantize-verified. These tests guard the
properties we DO want to hold across edits, without forcing a subset/superset
relationship that the curation reality does not actually support.
"""

import pytest

from mlxk2.core.capabilities import (
    AUDIO_MODEL_TYPES,
    STT_MODEL_TYPES,
    VISION_MODEL_TYPES,
    VISION_QUANTIZE_TYPES,
    classify_convert_target,
)


class TestFrozensetHygiene:
    """Basic structural properties every model_type list must satisfy."""

    @pytest.mark.parametrize("name,frozen", [
        ("VISION_MODEL_TYPES", VISION_MODEL_TYPES),
        ("VISION_QUANTIZE_TYPES", VISION_QUANTIZE_TYPES),
        ("STT_MODEL_TYPES", STT_MODEL_TYPES),
        ("AUDIO_MODEL_TYPES", AUDIO_MODEL_TYPES),
    ])
    def test_all_entries_are_lowercase_ascii(self, name, frozen):
        for entry in frozen:
            assert entry == entry.lower(), f"{name} contains non-lowercase: {entry!r}"
            assert entry.isascii(), f"{name} contains non-ASCII: {entry!r}"
            assert entry == entry.strip(), f"{name} contains whitespace: {entry!r}"

    def test_vision_quantize_and_stt_are_disjoint(self):
        """A model_type cannot be both a quantize-vision and an STT type."""
        overlap = VISION_QUANTIZE_TYPES & STT_MODEL_TYPES
        assert not overlap, f"VISION_QUANTIZE_TYPES overlaps STT_MODEL_TYPES: {overlap}"


class TestClassifyConvertTargetContract:
    """classify_convert_target must honor the four outcomes in the defined order."""

    def test_every_vision_quantize_type_routes_to_vision(self):
        """Every entry on the quantize whitelist must classify as vision."""
        for model_type in VISION_QUANTIZE_TYPES:
            assert classify_convert_target({"model_type": model_type}) == "vision", (
                f"{model_type!r} should route to vision"
            )

    def test_every_stt_type_routes_to_stt_unsupported(self):
        for model_type in STT_MODEL_TYPES:
            assert classify_convert_target({"model_type": model_type}) == "stt_unsupported"

    def test_is_case_insensitive(self):
        assert classify_convert_target({"model_type": "GEMMA3"}) == "vision"
        assert classify_convert_target({"model_type": "Whisper"}) == "stt_unsupported"
        assert classify_convert_target({"model_type": "LLaMA"}) == "text"

    def test_missing_config_is_text(self):
        """A config without model_type and without multimodal markers is text."""
        assert classify_convert_target({}) == "text"

    def test_malformed_model_type_is_text(self):
        """Non-string model_type degrades gracefully to text."""
        assert classify_convert_target({"model_type": None}) == "text"
        assert classify_convert_target({"model_type": 42}) == "text"

    def test_vision_marker_without_whitelist_rejects(self):
        """An unknown model_type with vision_config must reject, not route to text."""
        assert classify_convert_target({
            "model_type": "qwen3_vl",
            "vision_config": {"foo": "bar"},
        }) == "unsupported_multimodal"

    def test_audio_marker_without_whitelist_rejects(self):
        """An unknown model_type with audio_config must reject."""
        assert classify_convert_target({
            "model_type": "unknown_audio_mm",
            "audio_config": {"foo": "bar"},
        }) == "unsupported_multimodal"

    def test_empty_vision_config_dict_still_triggers_reject(self):
        """An empty dict for vision_config is still a multimodal marker."""
        assert classify_convert_target({
            "model_type": "unknown_vlm",
            "vision_config": {},
        }) == "unsupported_multimodal"

    def test_non_dict_vision_config_does_not_trigger_reject(self):
        """A non-dict vision_config (bogus config) should NOT classify as multimodal.

        Protects against false positives where someone writes vision_config: null
        or vision_config: "text" in a config.json.
        """
        assert classify_convert_target({
            "model_type": "llama",
            "vision_config": None,
        }) == "text"
        assert classify_convert_target({
            "model_type": "llama",
            "vision_config": "not-a-dict",
        }) == "text"

    def test_verified_vision_type_with_vision_config_still_routes_vision(self):
        """Order matters: verified wins over multimodal-marker reject.

        gemma3 is in VISION_QUANTIZE_TYPES AND carries vision_config -> vision."""
        assert classify_convert_target({
            "model_type": "gemma3",
            "vision_config": {"something": "here"},
        }) == "vision"
