"""Tests for probe/policy architecture (capabilities.py).

Tests the unified capability detection and backend selection logic.
"""

import json
import pytest
import sys
from pathlib import Path
from unittest.mock import patch, MagicMock

from mlxk2.core.capabilities import (
    Backend,
    PolicyDecision,
    MEMORY_THRESHOLD_PERCENT,
    VISION_MODEL_TYPES,
    AUDIO_MODEL_TYPES,
    ModelCapabilities,
    BackendPolicy,
    probe_model_capabilities,
    select_backend_policy,
    probe_and_select,
    _get_system_memory_bytes,
    _get_model_size_bytes,
    _format_bytes_gb,
    _detect_vision_from_config,
    _detect_vision_from_files,
)
from mlxk2.operations.common import (
    detect_audio_capability,
    detect_model_type,
    detect_vision_capability,
)


class TestVisionModelTypes:
    """Tests for VISION_MODEL_TYPES constant."""

    def test_contains_known_types(self):
        """Should contain all known vision model types."""
        expected = {"llava", "llava_next", "pixtral", "qwen2_vl", "phi3_v", "mllama", "paligemma", "idefics", "smolvlm"}
        assert expected.issubset(VISION_MODEL_TYPES)

    def test_is_frozenset(self):
        """Should be immutable."""
        assert isinstance(VISION_MODEL_TYPES, frozenset)


class TestAudioModelTypes:
    """Tests for AUDIO_MODEL_TYPES constant (ADR-019)."""

    def test_contains_gemma3n(self):
        """Should contain Gemma-3n model types."""
        assert "gemma3n" in AUDIO_MODEL_TYPES
        assert "gemma3n_audio" in AUDIO_MODEL_TYPES

    def test_is_frozenset(self):
        """Should be immutable."""
        assert isinstance(AUDIO_MODEL_TYPES, frozenset)


class TestDetectAudioCapability:
    """Tests for detect_audio_capability (ADR-019)."""

    def test_detects_audio_config_in_config_json(self, tmp_path):
        """Should detect audio capability via audio_config key."""
        config = {"model_type": "gemma3n", "audio_config": {"some": "config"}}
        (tmp_path / "config.json").write_text(json.dumps(config))
        assert detect_audio_capability(tmp_path, config) is True

    def test_detects_gemma3n_model_type(self, tmp_path):
        """Should detect audio capability via model_type."""
        config = {"model_type": "gemma3n"}
        (tmp_path / "config.json").write_text(json.dumps(config))
        assert detect_audio_capability(tmp_path, config) is True

    def test_processor_config_audio_seq_length_is_not_a_standalone_signal(self, tmp_path):
        """Class B (2.0.6): processor_config.json:audio_seq_length is no longer trusted alone.

        The Gemma4 processor template ships audio_seq_length=750 for every variant,
        including text-only/vision-only ones with `audio_config: null` (gemma-4-26b-a4b-it-4bit,
        gemma-4-31b-bf16). Real audio models are caught by audio_config or model_type
        signals; processor_config.json is no longer consulted here.
        """
        config = {"model_type": "llama"}  # Not an audio type
        (tmp_path / "config.json").write_text(json.dumps(config))
        processor_config = {"audio_seq_length": 188, "processor_class": "Gemma3nProcessor"}
        (tmp_path / "processor_config.json").write_text(json.dumps(processor_config))
        assert detect_audio_capability(tmp_path, config) is False

    def test_non_audio_model(self, tmp_path):
        """Should return False for non-audio models."""
        config = {"model_type": "llama"}
        (tmp_path / "config.json").write_text(json.dumps(config))
        assert detect_audio_capability(tmp_path, config) is False

    def test_none_config(self, tmp_path):
        """Should return False for None config (checks processor_config.json)."""
        assert detect_audio_capability(tmp_path, None) is False

    def test_empty_config(self, tmp_path):
        """Should return False for empty config."""
        (tmp_path / "config.json").write_text("{}")
        assert detect_audio_capability(tmp_path, {}) is False

    # Class B (2.0.6): audio_config key-existence false-positive fix.
    # `audio_config: null` and `audio_config: {}` are stub markers that do
    # not denote a real audio tower (Gemma4-31b-{6bit,bf16} carry the null
    # form). Detection must require a truthy dict.

    def test_audio_config_null_is_not_audio(self, tmp_path):
        """Gemma4-31b/26b stub: audio_config: null (key present, value null)."""
        config = {"model_type": "gemma4", "audio_config": None}
        (tmp_path / "config.json").write_text(json.dumps(config))
        assert detect_audio_capability(tmp_path, config) is False

    def test_audio_config_empty_dict_is_not_audio(self, tmp_path):
        """audio_config: {} stub does not denote a real audio tower."""
        config = {"model_type": "llama", "audio_config": {}}
        (tmp_path / "config.json").write_text(json.dumps(config))
        assert detect_audio_capability(tmp_path, config) is False

    def test_audio_config_truthy_dict_is_audio(self, tmp_path):
        """Regression anchor: Gemma-4-e4b-it-4bit (truthy audio_config dict)."""
        config = {"model_type": "gemma4", "audio_config": {"hidden_size": 1024}}
        (tmp_path / "config.json").write_text(json.dumps(config))
        assert detect_audio_capability(tmp_path, config) is True

    def test_gemma4_template_stub_is_not_audio(self, tmp_path):
        """Real-world anchor: gemma-4-26b-a4b-it-4bit / gemma-4-31b-bf16.

        These ship `audio_config: null` in config.json plus `audio_seq_length: 750`
        in the Gemma4 processor template. Neither signal alone (per Class B) nor
        the template stub may tag them as audio.
        """
        config = {"model_type": "gemma4", "audio_config": None, "vision_config": {"hidden_size": 1152}}
        (tmp_path / "config.json").write_text(json.dumps(config))
        (tmp_path / "processor_config.json").write_text(
            json.dumps({"audio_seq_length": 750, "processor_class": "Gemma4Processor"})
        )
        assert detect_audio_capability(tmp_path, config) is False


class TestDetectModelTypeSTT:
    """Class A (2.0.6): STT model_type substring matching.

    detect_model_type Step 4 must match STT_MODEL_TYPES tokens as substrings,
    consistent with detect_audio_capability. Narrow exact-set membership left
    e.g. `vibevoice_asr` misclassified as `base+audio` instead of `audio`.
    """

    def test_whisper_returns_audio(self):
        """Regression anchor: exact whisper model_type still classifies as audio."""
        config = {"model_type": "whisper"}
        assert detect_model_type("openai/whisper-tiny", config, {}) == "audio"

    def test_vibevoice_asr_returns_audio(self):
        """VibeVoice-ASR substring match: 'vibevoice' in 'vibevoice_asr'."""
        config = {"model_type": "vibevoice_asr"}
        assert detect_model_type("microsoft/VibeVoice-ASR-4bit", config, {}) == "audio"

    def test_voxtral_returns_audio(self):
        """Voxtral classified as audio (substring or exact match)."""
        config = {"model_type": "voxtral"}
        assert detect_model_type("mistralai/Voxtral-Mini", config, {}) == "audio"

    def test_text_model_unaffected(self):
        """Regression anchor: plain llama still classifies as base."""
        config = {"model_type": "llama"}
        assert detect_model_type("meta-llama/Llama-3", config, {}) == "base"

    def test_chat_model_takes_precedence(self):
        """Chat models (Instruct/Chat name hint) still classify as chat over base."""
        config = {"model_type": "llama"}
        assert detect_model_type("meta-llama/Llama-3-Instruct", config, {}) == "chat"


class TestDetectVisionCapabilityTruthyDict:
    """Class B (2.0.6) symmetric vision fix: empty dict stub is not vision."""

    def test_vision_config_null_is_not_vision(self, tmp_path):
        """vision_config: null (isinstance check already filters this)."""
        config = {"model_type": "llama", "vision_config": None}
        (tmp_path / "config.json").write_text(json.dumps(config))
        assert detect_vision_capability(tmp_path, config) is False

    def test_vision_config_empty_dict_is_not_vision(self, tmp_path):
        """vision_config: {} stub does not denote a real vision tower."""
        config = {"model_type": "llama", "vision_config": {}}
        (tmp_path / "config.json").write_text(json.dumps(config))
        assert detect_vision_capability(tmp_path, config) is False

    def test_vision_config_truthy_dict_is_vision(self, tmp_path):
        """Regression anchor: Mistral-Small-3.1 / Gemma4 with vision_config dict."""
        config = {"model_type": "llama", "vision_config": {"hidden_size": 1024}}
        (tmp_path / "config.json").write_text(json.dumps(config))
        assert detect_vision_capability(tmp_path, config) is True

    def test_vision_model_type_still_detected(self, tmp_path):
        """Regression anchor: VISION_MODEL_TYPES match independently of vision_config."""
        config = {"model_type": "pixtral"}
        (tmp_path / "config.json").write_text(json.dumps(config))
        assert detect_vision_capability(tmp_path, config) is True


class TestHelperFunctions:
    """Tests for helper functions."""

    def test_format_bytes_gb(self):
        """Should format bytes as GB correctly."""
        assert _format_bytes_gb(64 * 1024**3) == "64.0 GB"
        assert _format_bytes_gb(1 * 1024**3) == "1.0 GB"
        assert _format_bytes_gb(int(24.5 * 1024**3)) == "24.5 GB"

    def test_get_system_memory_bytes_returns_integer(self):
        """On macOS, should return positive integer."""
        result = _get_system_memory_bytes()
        if result is not None:
            assert isinstance(result, int)
            assert result > 0

    def test_get_system_memory_bytes_handles_error(self):
        """Should return None on error."""
        with patch("mlxk2.core.capabilities.subprocess.run") as mock_run:
            mock_run.side_effect = FileNotFoundError()
            result = _get_system_memory_bytes()
            assert result is None

    def test_get_model_size_bytes(self, tmp_path):
        """Should calculate total model size."""
        # Create some files
        (tmp_path / "model.safetensors").write_bytes(b"x" * 1000)
        (tmp_path / "config.json").write_text("{}")

        result = _get_model_size_bytes(tmp_path)
        assert result >= 1000  # At least the model file

    def test_get_model_size_bytes_empty_dir(self, tmp_path):
        """Should return 0 for empty directory."""
        result = _get_model_size_bytes(tmp_path)
        assert result == 0


class TestDetectVisionFromConfig:
    """Tests for _detect_vision_from_config."""

    def test_detects_mllama_type(self):
        """Should detect mllama as vision model."""
        config = {"model_type": "mllama"}
        assert _detect_vision_from_config(config) is True

    def test_detects_llava_type(self):
        """Should detect llava as vision model."""
        config = {"model_type": "llava"}
        assert _detect_vision_from_config(config) is True

    def test_detects_pixtral_type(self):
        """Should detect pixtral as vision model."""
        config = {"model_type": "pixtral"}
        assert _detect_vision_from_config(config) is True

    def test_detects_image_processor(self):
        """Should detect vision via image_processor field."""
        config = {"model_type": "llama", "image_processor": "CLIPImageProcessor"}
        assert _detect_vision_from_config(config) is True

    def test_detects_preprocessor_config(self):
        """Should detect vision via preprocessor_config field."""
        config = {"model_type": "llama", "preprocessor_config": {"image_size": 224}}
        assert _detect_vision_from_config(config) is True

    def test_non_vision_model(self):
        """Should not detect plain llama as vision."""
        config = {"model_type": "llama"}
        assert _detect_vision_from_config(config) is False

    def test_none_config(self):
        """Should return False for None config."""
        assert _detect_vision_from_config(None) is False

    def test_empty_config(self):
        """Should return False for empty config."""
        assert _detect_vision_from_config({}) is False


class TestDetectVisionFromFiles:
    """Tests for _detect_vision_from_files."""

    def test_detects_preprocessor_config_json(self, tmp_path):
        """Should detect vision via preprocessor_config.json presence."""
        (tmp_path / "preprocessor_config.json").write_text("{}")
        assert _detect_vision_from_files(tmp_path) is True

    def test_detects_processor_config_json(self, tmp_path):
        """Should detect vision via processor_config.json presence."""
        (tmp_path / "processor_config.json").write_text("{}")
        assert _detect_vision_from_files(tmp_path) is True

    def test_detects_image_processor_config_json(self, tmp_path):
        """Should detect vision via image_processor_config.json presence."""
        (tmp_path / "image_processor_config.json").write_text("{}")
        assert _detect_vision_from_files(tmp_path) is True

    def test_no_vision_files(self, tmp_path):
        """Should return False when no vision files present."""
        (tmp_path / "config.json").write_text("{}")
        assert _detect_vision_from_files(tmp_path) is False


class TestProbeModelCapabilities:
    """Tests for probe_model_capabilities."""

    def test_probes_text_model(self, tmp_path):
        """Should correctly probe a text-only model."""
        config = {"model_type": "llama"}
        (tmp_path / "config.json").write_text(json.dumps(config))
        (tmp_path / "model.safetensors").write_bytes(b"x" * 1000)

        caps = probe_model_capabilities(tmp_path, "test/model")

        assert caps.model_path == tmp_path
        assert caps.model_name == "test/model"
        assert caps.is_vision is False
        assert caps.config_valid is True
        assert caps.model_type == "llama"
        assert "text-generation" in caps.capabilities_list
        assert "vision" not in caps.capabilities_list

    def test_probes_vision_model_by_config(self, tmp_path):
        """Should detect vision model from config.json."""
        config = {"model_type": "mllama"}
        (tmp_path / "config.json").write_text(json.dumps(config))

        caps = probe_model_capabilities(tmp_path, "test/vision-model")

        assert caps.is_vision is True
        assert "vision" in caps.capabilities_list

    def test_probes_vision_model_by_files(self, tmp_path):
        """Should detect vision model from file presence."""
        config = {"model_type": "llama"}  # Not a vision type
        (tmp_path / "config.json").write_text(json.dumps(config))
        (tmp_path / "preprocessor_config.json").write_text("{}")

        caps = probe_model_capabilities(tmp_path, "test/vision-model")

        assert caps.is_vision is True
        assert "vision" in caps.capabilities_list

    def test_probes_chat_model_by_type(self, tmp_path):
        """Should detect chat capability from model_type."""
        config = {"model_type": "chat"}
        (tmp_path / "config.json").write_text(json.dumps(config))

        caps = probe_model_capabilities(tmp_path, "test/chat-model")

        assert caps.is_chat is True
        assert "chat" in caps.capabilities_list

    def test_probes_chat_model_by_template(self, tmp_path):
        """Should detect chat capability from chat_template."""
        config = {"model_type": "llama"}
        (tmp_path / "config.json").write_text(json.dumps(config))
        tokenizer = {"chat_template": "{% for message in messages %}...{% endfor %}"}
        (tmp_path / "tokenizer_config.json").write_text(json.dumps(tokenizer))

        caps = probe_model_capabilities(tmp_path, "test/chat-model")

        assert caps.is_chat is True
        assert "chat" in caps.capabilities_list

    def test_probes_chat_model_by_name(self, tmp_path):
        """Should detect chat capability from model name."""
        config = {"model_type": "llama"}
        (tmp_path / "config.json").write_text(json.dumps(config))

        caps = probe_model_capabilities(tmp_path, "test/Llama-3-Instruct")

        assert caps.is_chat is True
        assert "chat" in caps.capabilities_list

    def test_probes_embedding_model(self, tmp_path):
        """Should detect embedding model from name."""
        config = {"model_type": "bert"}
        (tmp_path / "config.json").write_text(json.dumps(config))

        caps = probe_model_capabilities(tmp_path, "test/text-embedding-model")

        assert caps.is_embedding is True
        assert "embeddings" in caps.capabilities_list
        assert "text-generation" not in caps.capabilities_list

    def test_probes_memory_information(self, tmp_path):
        """Should probe memory information."""
        config = {"model_type": "llama"}
        (tmp_path / "config.json").write_text(json.dumps(config))
        (tmp_path / "model.safetensors").write_bytes(b"x" * 10000)

        with patch("mlxk2.core.capabilities._get_system_memory_bytes") as mock_mem:
            mock_mem.return_value = 64 * 1024**3  # 64GB

            caps = probe_model_capabilities(tmp_path, "test/model")

            assert caps.system_memory_bytes == 64 * 1024**3
            assert caps.model_size_bytes >= 10000
            assert caps.memory_ratio > 0

    def test_probes_python_version(self, tmp_path):
        """Should record Python version."""
        config = {"model_type": "llama"}
        (tmp_path / "config.json").write_text(json.dumps(config))

        caps = probe_model_capabilities(tmp_path, "test/model")

        assert caps.python_version == sys.version_info[:3]

    def test_handles_missing_config(self, tmp_path):
        """Should handle missing config.json gracefully."""
        caps = probe_model_capabilities(tmp_path, "test/model")

        assert caps.config_valid is False
        assert caps.config is None

    def test_handles_invalid_config(self, tmp_path):
        """Should handle invalid JSON in config.json."""
        (tmp_path / "config.json").write_text("not valid json")

        caps = probe_model_capabilities(tmp_path, "test/model")

        assert caps.config_valid is False

    def test_accepts_preloaded_config(self, tmp_path):
        """Should use provided config instead of loading from file."""
        config = {"model_type": "pixtral"}

        caps = probe_model_capabilities(tmp_path, "test/model", config=config)

        assert caps.config_valid is True
        assert caps.model_type == "pixtral"
        assert caps.is_vision is True


class TestSelectBackendPolicy:
    """Tests for select_backend_policy."""

    def test_text_model_cli_allows(self, tmp_path):
        """Text model in CLI context should be allowed with MLX_LM backend."""
        caps = ModelCapabilities(
            model_path=tmp_path,
            model_name="test/model",
            is_vision=False,
            mlx_lm_available=True,
        )

        policy = select_backend_policy(caps, context="cli")

        assert policy.backend == Backend.MLX_LM
        assert policy.decision == PolicyDecision.ALLOW
        assert policy.message is None

    def test_vision_model_cli_allows(self, tmp_path):
        """Vision model in CLI context should be allowed with MLX_VLM backend."""
        caps = ModelCapabilities(
            model_path=tmp_path,
            model_name="test/vision-model",
            is_vision=True,
            python_version=(3, 10, 0),
            mlx_vlm_available=True,
            memory_ratio=0.5,  # Under threshold
        )

        policy = select_backend_policy(caps, context="cli")

        assert policy.backend == Backend.MLX_VLM
        assert policy.decision == PolicyDecision.ALLOW

    def test_vision_model_server_allows(self, tmp_path):
        """Vision model in server context should be allowed (ADR-012 Phase 3)."""
        caps = ModelCapabilities(
            model_path=tmp_path,
            model_name="test/vision-model",
            is_vision=True,
            python_version=(3, 10, 0),
            mlx_vlm_available=True,
        )

        policy = select_backend_policy(caps, context="server")

        # Vision is now supported in server (2.0.4-beta.1)
        assert policy.backend == Backend.MLX_VLM
        assert policy.decision == PolicyDecision.ALLOW

    def test_vision_model_python_39_blocks(self, tmp_path):
        """Vision model with Python 3.9 should be blocked."""
        caps = ModelCapabilities(
            model_path=tmp_path,
            model_name="test/vision-model",
            is_vision=True,
            python_version=(3, 9, 0),
            mlx_vlm_available=True,
        )

        policy = select_backend_policy(caps, context="cli")

        assert policy.backend == Backend.UNSUPPORTED
        assert policy.decision == PolicyDecision.BLOCK
        assert policy.http_status == 501
        assert "3.10" in policy.message
        assert policy.error_type == "python_version_error"

    def test_vision_model_no_mlx_vlm_blocks(self, tmp_path):
        """Vision model without mlx-vlm should be blocked."""
        caps = ModelCapabilities(
            model_path=tmp_path,
            model_name="test/vision-model",
            is_vision=True,
            python_version=(3, 10, 0),
            mlx_vlm_available=False,
        )

        policy = select_backend_policy(caps, context="cli")

        assert policy.backend == Backend.UNSUPPORTED
        assert policy.decision == PolicyDecision.BLOCK
        assert "mlx-vlm" in policy.message
        assert policy.error_type == "missing_dependency"

    def test_vision_model_memory_over_threshold_blocks(self, tmp_path):
        """Vision model over 70% memory should be blocked."""
        caps = ModelCapabilities(
            model_path=tmp_path,
            model_name="test/vision-model",
            is_vision=True,
            python_version=(3, 10, 0),
            mlx_vlm_available=True,
            system_memory_bytes=64 * 1024**3,
            model_size_bytes=50 * 1024**3,  # 78% of 64GB
            memory_ratio=0.78,
        )

        policy = select_backend_policy(caps, context="cli")

        assert policy.backend == Backend.UNSUPPORTED
        assert policy.decision == PolicyDecision.BLOCK
        assert policy.http_status == 507
        assert "70%" in policy.message
        assert "Metal OOM" in policy.message
        assert policy.error_type == "insufficient_memory"

    def test_text_model_memory_over_threshold_warns(self, tmp_path):
        """Text model over 70% memory should warn but allow."""
        caps = ModelCapabilities(
            model_path=tmp_path,
            model_name="test/text-model",
            is_vision=False,
            mlx_lm_available=True,
            system_memory_bytes=64 * 1024**3,
            model_size_bytes=50 * 1024**3,
            memory_ratio=0.78,
        )

        policy = select_backend_policy(caps, context="cli")

        assert policy.backend == Backend.MLX_LM
        assert policy.decision == PolicyDecision.WARN
        assert "70%" in policy.message
        assert "swapping" in policy.message

    def test_images_on_non_vision_model_blocks(self, tmp_path):
        """Passing images to non-vision model should be blocked."""
        caps = ModelCapabilities(
            model_path=tmp_path,
            model_name="test/text-model",
            is_vision=False,
            python_version=(3, 10, 0),
            mlx_vlm_available=True,
        )

        policy = select_backend_policy(caps, context="cli", has_images=True)

        assert policy.backend == Backend.UNSUPPORTED
        assert policy.decision == PolicyDecision.BLOCK
        assert policy.http_status == 400
        assert "vision" in policy.message.lower()
        assert policy.error_type == "capability_mismatch"

    def test_text_model_no_mlx_lm_blocks(self, tmp_path):
        """Text model without mlx-lm should be blocked."""
        caps = ModelCapabilities(
            model_path=tmp_path,
            model_name="test/text-model",
            is_vision=False,
            mlx_lm_available=False,
        )

        policy = select_backend_policy(caps, context="cli")

        assert policy.backend == Backend.UNSUPPORTED
        assert policy.decision == PolicyDecision.BLOCK
        assert "mlx-lm" in policy.message
        assert policy.error_type == "missing_dependency"


class TestProbeAndSelect:
    """Tests for probe_and_select convenience function."""

    def test_combines_probe_and_select(self, tmp_path):
        """Should run both probe and select in one call."""
        config = {"model_type": "llama"}
        (tmp_path / "config.json").write_text(json.dumps(config))
        # Create valid MLX weight file for runtime compatibility
        (tmp_path / "model.safetensors").write_bytes(b"weights")

        with patch("mlxk2.core.capabilities._check_mlx_lm_available") as mock_lm, \
             patch("mlxk2.core.capabilities._check_text_runtime_compatibility") as mock_runtime:
            mock_lm.return_value = True
            mock_runtime.return_value = (True, None)  # Compatible

            caps, policy = probe_and_select(tmp_path, "mlx-community/test-model", context="cli")

            assert caps.model_type == "llama"
            assert caps.is_vision is False
            assert policy.backend == Backend.MLX_LM
            assert policy.decision == PolicyDecision.ALLOW

    def test_context_parameter_forwarded_to_policy(self, tmp_path):
        """Context parameter should be forwarded to select_backend_policy (text model)."""
        config = {"model_type": "llama"}
        (tmp_path / "config.json").write_text(json.dumps(config))

        with patch("mlxk2.core.capabilities._check_mlx_lm_available") as mock_lm, \
             patch("mlxk2.core.capabilities._check_text_runtime_compatibility") as mock_runtime:
            mock_lm.return_value = True
            mock_runtime.return_value = (True, None)

            _, policy = probe_and_select(tmp_path, "test/model", context="server")

            # Text model should work in server context
            assert policy.decision == PolicyDecision.ALLOW
            assert policy.backend == Backend.MLX_LM

    @pytest.mark.skipif(sys.version_info < (3, 10), reason="Vision requires Python 3.10+")
    def test_vision_server_context_allowed_on_python_310_plus(self, tmp_path):
        """Vision models should be allowed in server context on Python 3.10+ (ADR-012 Phase 3)."""
        config = {"model_type": "mllama"}
        (tmp_path / "config.json").write_text(json.dumps(config))

        with patch("mlxk2.core.capabilities._check_mlx_vlm_available") as mock_vlm:
            mock_vlm.return_value = True

            _, policy = probe_and_select(tmp_path, "test/vision", context="server")

            # Server allows vision on Python 3.10+ (ADR-012 Phase 3)
            assert policy.decision == PolicyDecision.ALLOW
            assert policy.backend == Backend.MLX_VLM

    def test_respects_has_images_parameter(self, tmp_path):
        """Should pass has_images through to select_backend_policy."""
        config = {"model_type": "llama"}  # Not vision
        (tmp_path / "config.json").write_text(json.dumps(config))

        with patch("mlxk2.core.capabilities._check_mlx_vlm_available") as mock_vlm:
            mock_vlm.return_value = True

            _, policy = probe_and_select(tmp_path, "test/model", has_images=True)

            # Images on non-vision should block
            assert policy.decision == PolicyDecision.BLOCK
            assert policy.error_type == "capability_mismatch"


class TestMemoryThreshold:
    """Tests for MEMORY_THRESHOLD_PERCENT constant."""

    def test_threshold_is_70_percent(self):
        """Threshold should be 70% as per ADR-016."""
        assert MEMORY_THRESHOLD_PERCENT == 0.70

    def test_threshold_calculation(self):
        """Verify threshold calculation for 64GB system."""
        system_memory = 64 * 1024**3
        threshold = int(system_memory * MEMORY_THRESHOLD_PERCENT)
        expected = int(44.8 * 1024**3)  # 70% of 64GB

        # Allow small rounding difference
        assert abs(threshold - expected) < 1024**2  # Within 1MB


class TestEmbeddingGate:
    """Tests for embedding model runtime compatibility gate (common.py:636-639).

    Embedding models should be detected but blocked from `mlxk run` with a
    helpful message pointing to `mlxk embed`.
    """

    def test_detect_capabilities_embedding_model_type(self, tmp_path):
        """model_type='embedding' should return only EMBEDDINGS capability."""
        from mlxk2.operations.common import detect_capabilities

        caps = detect_capabilities(
            model_type="embedding",
            hf_name="test/embedding-model",
            tok_hints={},
            config={},
            probe=tmp_path,
        )

        assert caps == ["embeddings"]
        assert "text-generation" not in caps

    def test_embedding_model_runtime_incompatible(self, tmp_path):
        """Embedding models should have runtime_compatible=False with helpful reason."""
        from mlxk2.operations.common import build_model_object

        # Create minimal embedding model structure with workspace sentinel
        config = {"model_type": "embedding"}
        (tmp_path / "config.json").write_text(json.dumps(config))
        (tmp_path / "model.safetensors").write_bytes(b"x" * 100)
        # Add workspace sentinel so it's treated as workspace path (ADR-018)
        sentinel = {"managed_by": "mlxk", "source": "test"}
        (tmp_path / ".mlxk-workspace.json").write_text(json.dumps(sentinel))
        # Add README with MLX library tag so framework is detected as MLX
        readme = """---
library_name: mlx
---
# Test Embedding Model
"""
        (tmp_path / "README.md").write_text(readme)

        # Use absolute path so it's treated as workspace path
        result = build_model_object(
            hf_name=str(tmp_path),  # Absolute path = workspace
            model_root=tmp_path,
            selected_path=tmp_path,
        )

        assert result["runtime_compatible"] is False
        assert "mlxk embed" in result["reason"]
        assert "embeddings" in result["capabilities"]

    def test_embedding_model_detected_by_name_heuristic(self, tmp_path):
        """Models with 'embedding' in name should be detected as embedding models."""
        config = {"model_type": "bert"}  # Not explicitly "embedding" type
        (tmp_path / "config.json").write_text(json.dumps(config))

        caps = probe_model_capabilities(tmp_path, "test/text-embedding-3-small")

        assert caps.is_embedding is True
        assert "embeddings" in caps.capabilities_list
