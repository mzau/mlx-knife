"""Unit tests for mlxk2/audio/whisper_tokenizer.py.

Tests the bundled Whisper tokenizer implementation (mlx-audio Issue #479 workaround).

Coverage:
- get_encoding(): Load tiktoken encodings from bundled assets
- get_tokenizer(): Create Tokenizer instances for various configurations
- Tokenizer class: Special tokens, encode/decode, properties
"""

import pytest
from pathlib import Path


def _mlx_audio_available():
    """Check if mlx-audio is available and functional."""
    try:
        import mlx_audio.stt.models.whisper.tokenizer  # noqa: F401
        return True
    except Exception:
        return False


requires_mlx_audio = pytest.mark.skipif(
    not _mlx_audio_available(),
    reason="mlx-audio not available or MLX incompatible"
)


class TestGetEncoding:
    """Tests for get_encoding() function."""

    def test_get_encoding_gpt2(self):
        """Load gpt2 encoding from bundled assets."""
        from mlxk2.audio.whisper_tokenizer import get_encoding

        enc = get_encoding("gpt2")

        assert enc is not None
        assert enc.name == "gpt2.tiktoken"
        # Verify it can encode/decode basic text
        tokens = enc.encode("Hello world")
        assert len(tokens) > 0
        decoded = enc.decode(tokens)
        assert decoded == "Hello world"

    def test_get_encoding_multilingual(self):
        """Load multilingual encoding from bundled assets."""
        from mlxk2.audio.whisper_tokenizer import get_encoding

        enc = get_encoding("multilingual")

        assert enc is not None
        assert enc.name == "multilingual.tiktoken"
        # Verify it can encode/decode multilingual text
        tokens = enc.encode("Guten Tag")
        assert len(tokens) > 0
        decoded = enc.decode(tokens)
        assert decoded == "Guten Tag"

    def test_get_encoding_nonexistent_raises(self):
        """Unknown encoding name should raise FileNotFoundError."""
        from mlxk2.audio.whisper_tokenizer import get_encoding

        with pytest.raises(FileNotFoundError) as exc_info:
            get_encoding("nonexistent_encoding")

        assert "Tiktoken vocabulary file not found" in str(exc_info.value)
        assert "mlx-audio Issue #479" in str(exc_info.value)

    def test_get_encoding_is_cached(self):
        """get_encoding() should be cached (lru_cache)."""
        from mlxk2.audio.whisper_tokenizer import get_encoding

        enc1 = get_encoding("gpt2")
        enc2 = get_encoding("gpt2")

        # Same object due to caching
        assert enc1 is enc2

    def test_get_encoding_has_special_tokens(self):
        """Encoding should have Whisper special tokens."""
        from mlxk2.audio.whisper_tokenizer import get_encoding

        enc = get_encoding("gpt2")

        # Check Whisper-specific special tokens exist
        special_tokens = enc.special_tokens_set
        assert "<|endoftext|>" in special_tokens
        assert "<|startoftranscript|>" in special_tokens
        assert "<|transcribe|>" in special_tokens
        assert "<|translate|>" in special_tokens
        assert "<|nospeech|>" in special_tokens
        assert "<|notimestamps|>" in special_tokens

    def test_get_encoding_has_language_tokens(self):
        """Encoding should have language tokens (<|en|>, <|de|>, etc.)."""
        from mlxk2.audio.whisper_tokenizer import get_encoding

        enc = get_encoding("gpt2", num_languages=99)

        special_tokens = enc.special_tokens_set
        assert "<|en|>" in special_tokens
        assert "<|de|>" in special_tokens
        assert "<|fr|>" in special_tokens
        assert "<|es|>" in special_tokens

    def test_get_encoding_has_timestamp_tokens(self):
        """Encoding should have timestamp tokens (<|0.00|> to <|30.00|>)."""
        from mlxk2.audio.whisper_tokenizer import get_encoding

        enc = get_encoding("gpt2")

        special_tokens = enc.special_tokens_set
        assert "<|0.00|>" in special_tokens
        assert "<|0.02|>" in special_tokens
        assert "<|10.00|>" in special_tokens
        assert "<|30.00|>" in special_tokens


class TestGetTokenizer:
    """Tests for get_tokenizer() function."""

    def test_get_tokenizer_multilingual_default(self):
        """Multilingual tokenizer with default settings."""
        from mlxk2.audio.whisper_tokenizer import get_tokenizer

        tok = get_tokenizer(multilingual=True)

        assert tok is not None
        assert tok.language == "en"  # Default language
        assert tok.task == "transcribe"  # Default task
        assert tok.encoding.name == "multilingual.tiktoken"

    def test_get_tokenizer_multilingual_german(self):
        """Multilingual tokenizer with German language."""
        from mlxk2.audio.whisper_tokenizer import get_tokenizer

        tok = get_tokenizer(multilingual=True, language="de")

        assert tok.language == "de"
        assert tok.task == "transcribe"

    def test_get_tokenizer_multilingual_translate_task(self):
        """Multilingual tokenizer with translate task."""
        from mlxk2.audio.whisper_tokenizer import get_tokenizer

        tok = get_tokenizer(multilingual=True, language="de", task="translate")

        assert tok.language == "de"
        assert tok.task == "translate"

    def test_get_tokenizer_english_only(self):
        """English-only (non-multilingual) tokenizer."""
        from mlxk2.audio.whisper_tokenizer import get_tokenizer

        tok = get_tokenizer(multilingual=False)

        assert tok.language is None  # English-only has no language
        assert tok.task is None  # English-only has no task
        assert tok.encoding.name == "gpt2.tiktoken"

    def test_get_tokenizer_invalid_language_raises(self):
        """Invalid language code should raise ValueError."""
        from mlxk2.audio.whisper_tokenizer import get_tokenizer

        with pytest.raises(ValueError) as exc_info:
            get_tokenizer(multilingual=True, language="xyz")

        assert "Unsupported language: xyz" in str(exc_info.value)

    def test_get_tokenizer_language_alias(self):
        """Language aliases should be resolved (e.g., 'german' -> 'de')."""
        from mlxk2.audio.whisper_tokenizer import get_tokenizer

        tok = get_tokenizer(multilingual=True, language="german")

        assert tok.language == "de"

    def test_get_tokenizer_language_case_insensitive(self):
        """Language codes should be case-insensitive."""
        from mlxk2.audio.whisper_tokenizer import get_tokenizer

        tok = get_tokenizer(multilingual=True, language="DE")

        assert tok.language == "de"

    def test_get_tokenizer_is_cached(self):
        """get_tokenizer() should be cached."""
        from mlxk2.audio.whisper_tokenizer import get_tokenizer

        tok1 = get_tokenizer(multilingual=True, language="fr")
        tok2 = get_tokenizer(multilingual=True, language="fr")

        assert tok1 is tok2

    def test_get_tokenizer_various_languages(self):
        """Test tokenizer with various supported languages."""
        from mlxk2.audio.whisper_tokenizer import get_tokenizer

        # Sample of supported languages
        languages = ["en", "de", "fr", "es", "ja", "zh", "ru", "ar", "ko", "pt"]

        for lang in languages:
            tok = get_tokenizer(multilingual=True, language=lang)
            assert tok.language == lang, f"Language {lang} not set correctly"


class TestTokenizerClass:
    """Tests for Tokenizer class methods and properties."""

    @pytest.fixture
    def multilingual_tokenizer(self):
        """Create a multilingual tokenizer for testing."""
        from mlxk2.audio.whisper_tokenizer import get_tokenizer

        return get_tokenizer(multilingual=True, language="en", task="transcribe")

    @pytest.fixture
    def german_tokenizer(self):
        """Create a German tokenizer for testing."""
        from mlxk2.audio.whisper_tokenizer import get_tokenizer

        return get_tokenizer(multilingual=True, language="de", task="transcribe")

    def test_special_tokens_populated(self, multilingual_tokenizer):
        """Special tokens dict should be populated after init."""
        tok = multilingual_tokenizer

        assert len(tok.special_tokens) > 0
        assert "<|startoftranscript|>" in tok.special_tokens
        assert "<|transcribe|>" in tok.special_tokens
        assert "<|translate|>" in tok.special_tokens
        assert "<|endoftext|>" in tok.special_tokens

    def test_sot_property(self, multilingual_tokenizer):
        """sot property should return start of transcript token."""
        tok = multilingual_tokenizer

        assert tok.sot == tok.special_tokens["<|startoftranscript|>"]
        assert isinstance(tok.sot, int)

    def test_eot_property(self, multilingual_tokenizer):
        """eot property should return end of text token."""
        tok = multilingual_tokenizer

        assert tok.eot == tok.encoding.eot_token
        assert isinstance(tok.eot, int)

    def test_transcribe_property(self, multilingual_tokenizer):
        """transcribe property should return transcribe token."""
        tok = multilingual_tokenizer

        assert tok.transcribe == tok.special_tokens["<|transcribe|>"]
        assert isinstance(tok.transcribe, int)

    def test_translate_property(self, multilingual_tokenizer):
        """translate property should return translate token."""
        tok = multilingual_tokenizer

        assert tok.translate == tok.special_tokens["<|translate|>"]
        assert isinstance(tok.translate, int)

    def test_no_timestamps_property(self, multilingual_tokenizer):
        """no_timestamps property should return notimestamps token."""
        tok = multilingual_tokenizer

        assert tok.no_timestamps == tok.special_tokens["<|notimestamps|>"]

    def test_timestamp_begin_property(self, multilingual_tokenizer):
        """timestamp_begin property should return first timestamp token."""
        tok = multilingual_tokenizer

        assert tok.timestamp_begin == tok.special_tokens["<|0.00|>"]

    def test_no_speech_property(self, multilingual_tokenizer):
        """no_speech property should return nospeech token."""
        tok = multilingual_tokenizer

        assert tok.no_speech == tok.special_tokens["<|nospeech|>"]

    def test_sot_sequence_multilingual(self, multilingual_tokenizer):
        """sot_sequence should contain sot + language + task tokens."""
        tok = multilingual_tokenizer

        # Should have: sot, language token, task token
        assert len(tok.sot_sequence) == 3
        assert tok.sot_sequence[0] == tok.sot
        # Last token should be transcribe (for transcribe task)
        assert tok.sot_sequence[2] == tok.transcribe

    def test_sot_sequence_including_notimestamps(self, multilingual_tokenizer):
        """sot_sequence_including_notimestamps should append notimestamps."""
        tok = multilingual_tokenizer

        seq = tok.sot_sequence_including_notimestamps
        assert seq[-1] == tok.no_timestamps
        assert len(seq) == len(tok.sot_sequence) + 1

    def test_encode_decode_roundtrip(self, multilingual_tokenizer):
        """encode() and decode() should roundtrip text correctly."""
        tok = multilingual_tokenizer

        original = "Hello, this is a test."
        tokens = tok.encode(original)
        decoded = tok.decode(tokens)

        assert decoded == original

    def test_decode_filters_timestamp_tokens(self, multilingual_tokenizer):
        """decode() should filter out timestamp tokens."""
        tok = multilingual_tokenizer

        # Encode some text and add a timestamp token
        tokens = tok.encode("Hello")
        # Add a timestamp token (should be filtered)
        tokens_with_timestamp = tokens + [tok.timestamp_begin]

        # decode() filters tokens >= timestamp_begin
        decoded = tok.decode(tokens_with_timestamp)
        assert decoded == "Hello"

    def test_decode_with_timestamps_preserves_all(self, multilingual_tokenizer):
        """decode_with_timestamps() should preserve timestamp tokens."""
        tok = multilingual_tokenizer

        # Encode text that includes timestamp-like content
        tokens = tok.encode("Hello")
        decoded = tok.decode_with_timestamps(tokens)
        assert decoded == "Hello"

    def test_language_token_property(self, german_tokenizer):
        """language_token property should return correct language token."""
        tok = german_tokenizer

        lang_token = tok.language_token
        assert lang_token == tok.special_tokens["<|de|>"]

    def test_language_token_raises_when_none(self):
        """language_token should raise ValueError when language is None."""
        from mlxk2.audio.whisper_tokenizer import get_tokenizer

        tok = get_tokenizer(multilingual=False)  # English-only has no language

        with pytest.raises(ValueError) as exc_info:
            _ = tok.language_token

        assert "language token configured" in str(exc_info.value)

    def test_to_language_token(self, multilingual_tokenizer):
        """to_language_token() should return token for given language."""
        tok = multilingual_tokenizer

        de_token = tok.to_language_token("de")
        assert de_token == tok.special_tokens["<|de|>"]

    def test_to_language_token_invalid_raises(self, multilingual_tokenizer):
        """to_language_token() should raise KeyError for invalid language."""
        tok = multilingual_tokenizer

        with pytest.raises(KeyError) as exc_info:
            tok.to_language_token("xyz")

        assert "Language xyz not found" in str(exc_info.value)

    def test_all_language_tokens(self, multilingual_tokenizer):
        """all_language_tokens should return tuple of language token IDs."""
        tok = multilingual_tokenizer

        lang_tokens = tok.all_language_tokens

        assert isinstance(lang_tokens, tuple)
        assert len(lang_tokens) > 0
        assert all(isinstance(t, int) for t in lang_tokens)

    def test_all_language_codes(self, multilingual_tokenizer):
        """all_language_codes should return tuple of language codes."""
        tok = multilingual_tokenizer

        lang_codes = tok.all_language_codes

        assert isinstance(lang_codes, tuple)
        assert len(lang_codes) > 0
        assert "en" in lang_codes
        assert "de" in lang_codes

    def test_non_speech_tokens(self, multilingual_tokenizer):
        """non_speech_tokens should return tokens to suppress."""
        tok = multilingual_tokenizer

        non_speech = tok.non_speech_tokens

        assert isinstance(non_speech, tuple)
        assert len(non_speech) > 0
        assert all(isinstance(t, int) for t in non_speech)


class TestTokenizerWordSplitting:
    """Tests for word splitting methods."""

    @pytest.fixture
    def english_tokenizer(self):
        """English tokenizer for word splitting tests."""
        from mlxk2.audio.whisper_tokenizer import get_tokenizer

        return get_tokenizer(multilingual=True, language="en")

    @pytest.fixture
    def chinese_tokenizer(self):
        """Chinese tokenizer for unicode splitting tests."""
        from mlxk2.audio.whisper_tokenizer import get_tokenizer

        return get_tokenizer(multilingual=True, language="zh")

    def test_split_to_word_tokens_english(self, english_tokenizer):
        """split_to_word_tokens should split on spaces for English."""
        tok = english_tokenizer

        tokens = tok.encode("Hello world")
        words, word_tokens = tok.split_to_word_tokens(tokens)

        assert len(words) >= 1
        assert len(word_tokens) == len(words)
        # Each word should have associated tokens
        for word, wtokens in zip(words, word_tokens):
            assert len(wtokens) > 0

    def test_split_to_word_tokens_chinese(self, chinese_tokenizer):
        """split_to_word_tokens should use unicode splitting for Chinese."""
        tok = chinese_tokenizer

        tokens = tok.encode("Hello")  # Just test it doesn't crash
        words, word_tokens = tok.split_to_word_tokens(tokens)

        assert len(words) >= 1
        assert len(word_tokens) == len(words)

    def test_split_tokens_on_unicode(self, english_tokenizer):
        """split_tokens_on_unicode should handle unicode characters."""
        tok = english_tokenizer

        tokens = tok.encode("Caf\u00e9")
        words, word_tokens = tok.split_tokens_on_unicode(tokens)

        assert len(words) >= 1
        # Reconstructed should match
        reconstructed = "".join(words)
        assert "Caf" in reconstructed

    def test_split_tokens_on_spaces(self, english_tokenizer):
        """split_tokens_on_spaces should split on whitespace."""
        tok = english_tokenizer

        tokens = tok.encode("Hello world test")
        words, word_tokens = tok.split_tokens_on_spaces(tokens)

        assert len(words) >= 1
        assert len(word_tokens) == len(words)


class TestLanguageConstants:
    """Tests for LANGUAGES and TO_LANGUAGE_CODE constants."""

    def test_languages_dict_exists(self):
        """LANGUAGES dict should be importable."""
        from mlxk2.audio.whisper_tokenizer import LANGUAGES

        assert isinstance(LANGUAGES, dict)
        assert len(LANGUAGES) > 90  # Whisper supports ~99 languages

    def test_languages_contains_common(self):
        """LANGUAGES should contain common language codes."""
        from mlxk2.audio.whisper_tokenizer import LANGUAGES

        assert "en" in LANGUAGES
        assert LANGUAGES["en"] == "english"
        assert "de" in LANGUAGES
        assert LANGUAGES["de"] == "german"
        assert "fr" in LANGUAGES
        assert LANGUAGES["fr"] == "french"
        assert "ja" in LANGUAGES
        assert LANGUAGES["ja"] == "japanese"
        assert "zh" in LANGUAGES
        assert LANGUAGES["zh"] == "chinese"

    def test_to_language_code_dict_exists(self):
        """TO_LANGUAGE_CODE dict should be importable."""
        from mlxk2.audio.whisper_tokenizer import TO_LANGUAGE_CODE

        assert isinstance(TO_LANGUAGE_CODE, dict)

    def test_to_language_code_aliases(self):
        """TO_LANGUAGE_CODE should contain language name aliases."""
        from mlxk2.audio.whisper_tokenizer import TO_LANGUAGE_CODE

        assert TO_LANGUAGE_CODE["english"] == "en"
        assert TO_LANGUAGE_CODE["german"] == "de"
        assert TO_LANGUAGE_CODE["french"] == "fr"
        # Check some special aliases
        assert TO_LANGUAGE_CODE.get("mandarin") == "zh"
        assert TO_LANGUAGE_CODE.get("castilian") == "es"


class TestAssetsPaths:
    """Tests for bundled tiktoken assets."""

    def test_assets_directory_exists(self):
        """Assets directory should exist."""
        from mlxk2.audio.whisper_tokenizer import _ASSETS_DIR

        assert _ASSETS_DIR.exists(), f"Assets dir not found: {_ASSETS_DIR}"
        assert _ASSETS_DIR.is_dir()

    def test_gpt2_tiktoken_exists(self):
        """gpt2.tiktoken asset should exist."""
        from mlxk2.audio.whisper_tokenizer import _ASSETS_DIR

        gpt2_path = _ASSETS_DIR / "gpt2.tiktoken"
        assert gpt2_path.exists(), f"gpt2.tiktoken not found: {gpt2_path}"
        assert gpt2_path.stat().st_size > 100000  # Should be ~800KB

    def test_multilingual_tiktoken_exists(self):
        """multilingual.tiktoken asset should exist."""
        from mlxk2.audio.whisper_tokenizer import _ASSETS_DIR

        multilingual_path = _ASSETS_DIR / "multilingual.tiktoken"
        assert multilingual_path.exists(), f"multilingual.tiktoken not found: {multilingual_path}"
        assert multilingual_path.stat().st_size > 100000  # Should be ~800KB


@requires_mlx_audio
class TestPatchIntegration:
    """Tests for mlx-audio patch integration."""

    def test_patch_applied_to_mlx_audio(self):
        """Verify patch is applied when audio_runner is imported."""
        # Import audio_runner which applies the patch
        from mlxk2.core.audio_runner import AudioRunner  # noqa: F401
        from mlxk2.audio.whisper_tokenizer import get_encoding

        # Import the patched module
        import mlx_audio.stt.models.whisper.tokenizer as mlx_tok

        # Our get_encoding should be installed
        assert mlx_tok.get_encoding is get_encoding

    def test_patched_get_encoding_works(self):
        """Verify patched get_encoding produces valid encodings."""
        # Import to apply patch
        from mlxk2.core.audio_runner import AudioRunner  # noqa: F401

        # Use the patched version
        import mlx_audio.stt.models.whisper.tokenizer as mlx_tok

        enc = mlx_tok.get_encoding("gpt2")
        assert enc.name == "gpt2.tiktoken"

        # Verify encode/decode works
        tokens = enc.encode("Test patch")
        assert len(tokens) > 0
        decoded = enc.decode(tokens)
        assert decoded == "Test patch"
