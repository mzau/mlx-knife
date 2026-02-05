# Copyright 2023 Apple Inc.
# SPDX-License-Identifier: MIT
#
# Whisper tokenizer implementation from mlx-audio commit 9349644 (2026-01-26).
# This code was removed in PyPI release 0.3.1 (2026-01-29), breaking Whisper
# transcription for models without HuggingFace processor.
#
# Bundled here as workaround for mlx-audio Issue #479.
# See: https://github.com/Blaizzy/mlx-audio/issues/479
#
# Original source:
# https://github.com/Blaizzy/mlx-audio/blob/9349644/mlx_audio/stt/models/whisper/tokenizer.py

from __future__ import annotations

import base64
import string
from dataclasses import dataclass, field
from functools import cached_property, lru_cache
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import tiktoken

# Try to import LANGUAGES from mlx-audio (still present in 0.3.1)
# Fall back to our own definition if import fails
try:
    from mlx_audio.stt.models.whisper.tokenizer import LANGUAGES, TO_LANGUAGE_CODE
except ImportError:
    # Fallback: Full LANGUAGES dict from mlx-audio
    LANGUAGES = {
        "en": "english",
        "zh": "chinese",
        "de": "german",
        "es": "spanish",
        "ru": "russian",
        "ko": "korean",
        "fr": "french",
        "ja": "japanese",
        "pt": "portuguese",
        "tr": "turkish",
        "pl": "polish",
        "ca": "catalan",
        "nl": "dutch",
        "ar": "arabic",
        "sv": "swedish",
        "it": "italian",
        "id": "indonesian",
        "hi": "hindi",
        "fi": "finnish",
        "vi": "vietnamese",
        "he": "hebrew",
        "uk": "ukrainian",
        "el": "greek",
        "ms": "malay",
        "cs": "czech",
        "ro": "romanian",
        "da": "danish",
        "hu": "hungarian",
        "ta": "tamil",
        "no": "norwegian",
        "th": "thai",
        "ur": "urdu",
        "hr": "croatian",
        "bg": "bulgarian",
        "lt": "lithuanian",
        "la": "latin",
        "mi": "maori",
        "ml": "malayalam",
        "cy": "welsh",
        "sk": "slovak",
        "te": "telugu",
        "fa": "persian",
        "lv": "latvian",
        "bn": "bengali",
        "sr": "serbian",
        "az": "azerbaijani",
        "sl": "slovenian",
        "kn": "kannada",
        "et": "estonian",
        "mk": "macedonian",
        "br": "breton",
        "eu": "basque",
        "is": "icelandic",
        "hy": "armenian",
        "ne": "nepali",
        "mn": "mongolian",
        "bs": "bosnian",
        "kk": "kazakh",
        "sq": "albanian",
        "sw": "swahili",
        "gl": "galician",
        "mr": "marathi",
        "pa": "punjabi",
        "si": "sinhala",
        "km": "khmer",
        "sn": "shona",
        "yo": "yoruba",
        "so": "somali",
        "af": "afrikaans",
        "oc": "occitan",
        "ka": "georgian",
        "be": "belarusian",
        "tg": "tajik",
        "sd": "sindhi",
        "gu": "gujarati",
        "am": "amharic",
        "yi": "yiddish",
        "lo": "lao",
        "uz": "uzbek",
        "fo": "faroese",
        "ht": "haitian creole",
        "ps": "pashto",
        "tk": "turkmen",
        "nn": "nynorsk",
        "mt": "maltese",
        "sa": "sanskrit",
        "lb": "luxembourgish",
        "my": "myanmar",
        "bo": "tibetan",
        "tl": "tagalog",
        "mg": "malagasy",
        "as": "assamese",
        "tt": "tatar",
        "haw": "hawaiian",
        "ln": "lingala",
        "ha": "hausa",
        "ba": "bashkir",
        "jw": "javanese",
        "su": "sundanese",
        "yue": "cantonese",
    }

    TO_LANGUAGE_CODE = {
        **{language: code for code, language in LANGUAGES.items()},
        "burmese": "my",
        "valencian": "ca",
        "flemish": "nl",
        "haitian": "ht",
        "letzeburgesch": "lb",
        "pushto": "ps",
        "panjabi": "pa",
        "moldavian": "ro",
        "moldovan": "ro",
        "sinhalese": "si",
        "castilian": "es",
        "mandarin": "zh",
    }


# Path to bundled tiktoken assets in mlxk2
_ASSETS_DIR = Path(__file__).parent.parent / "assets" / "whisper"


@dataclass
class Tokenizer:
    """A thin wrapper around `tiktoken` providing quick access to special tokens."""

    encoding: tiktoken.Encoding
    num_languages: int
    language: Optional[str] = None
    task: Optional[str] = None
    sot_sequence: Tuple[int, ...] = ()
    special_tokens: Dict[str, int] = field(default_factory=dict)

    def __post_init__(self):
        for special in self.encoding.special_tokens_set:
            special_token = self.encoding.encode_single_token(special)
            self.special_tokens[special] = special_token

        sot: int = self.special_tokens["<|startoftranscript|>"]
        translate: int = self.special_tokens["<|translate|>"]
        transcribe: int = self.special_tokens["<|transcribe|>"]

        langs = tuple(LANGUAGES.keys())[: self.num_languages]
        sot_sequence = [sot]
        if self.language is not None:
            sot_sequence.append(sot + 1 + langs.index(self.language))
        if self.task is not None:
            task_token: int = transcribe if self.task == "transcribe" else translate
            sot_sequence.append(task_token)

        self.sot_sequence = tuple(sot_sequence)

    def encode(self, text, **kwargs):
        return self.encoding.encode(text, **kwargs)

    def decode(self, token_ids: List[int], **kwargs) -> str:
        token_ids = [t for t in token_ids if t < self.timestamp_begin]
        return self.encoding.decode(token_ids, **kwargs)

    def decode_with_timestamps(self, token_ids: List[int], **kwargs) -> str:
        """Timestamp tokens are above other special tokens' id range and are ignored by decode().
        This method decodes given tokens with timestamps tokens annotated, e.g. "<|1.08|>".
        """
        return self.encoding.decode(token_ids, **kwargs)

    @cached_property
    def eot(self) -> int:
        return self.encoding.eot_token

    @cached_property
    def transcribe(self) -> int:
        return self.special_tokens["<|transcribe|>"]

    @cached_property
    def translate(self) -> int:
        return self.special_tokens["<|translate|>"]

    @cached_property
    def sot(self) -> int:
        return self.special_tokens["<|startoftranscript|>"]

    @cached_property
    def sot_lm(self) -> int:
        return self.special_tokens["<|startoflm|>"]

    @cached_property
    def sot_prev(self) -> int:
        return self.special_tokens["<|startofprev|>"]

    @cached_property
    def no_speech(self) -> int:
        return self.special_tokens["<|nospeech|>"]

    @cached_property
    def no_timestamps(self) -> int:
        return self.special_tokens["<|notimestamps|>"]

    @cached_property
    def timestamp_begin(self) -> int:
        return self.special_tokens["<|0.00|>"]

    @cached_property
    def language_token(self) -> int:
        """Returns the token id corresponding to the value of the `language` field."""
        if self.language is None:
            raise ValueError("This tokenizer does not have language token configured")

        return self.to_language_token(self.language)

    def to_language_token(self, language):
        if token := self.special_tokens.get(f"<|{language}|>", None):
            return token

        raise KeyError(f"Language {language} not found in tokenizer.")

    @cached_property
    def all_language_tokens(self) -> Tuple[int, ...]:
        result = []
        for token, token_id in self.special_tokens.items():
            if token.strip("<|>") in LANGUAGES:
                result.append(token_id)
        return tuple(result)[: self.num_languages]

    @cached_property
    def all_language_codes(self) -> Tuple[str, ...]:
        return tuple(self.decode([_l]).strip("<|>") for _l in self.all_language_tokens)

    @cached_property
    def sot_sequence_including_notimestamps(self) -> Tuple[int, ...]:
        return tuple(list(self.sot_sequence) + [self.no_timestamps])

    @cached_property
    def non_speech_tokens(self) -> Tuple[int, ...]:
        """Returns the list of tokens to suppress in order to avoid any speaker tags or
        non-speech annotations, to prevent sampling texts that are not actually spoken
        in the audio, e.g.

        - (SPEAKING FOREIGN LANGUAGE)
        - [DAVID] Hey there,

        keeping basic punctuations like commas, periods, question marks, etc.
        """
        symbols = list('"#()*+/:;<=>@[\\]^_`{|}~')
        symbols += (
            "<< >> <<< >>> -- --- -( -[ (' (\" (( )) ((( ))) [[ ]] {{ }} ".split()
        )

        # symbols that may be a single token or multiple tokens depending on the tokenizer.
        # In case they're multiple tokens, suppress the first token, which is safe because:
        # These are between U+2640 and U+267F miscellaneous symbols that are okay to suppress
        # in generations, and in the 3-byte UTF-8 representation they share the first two bytes.
        miscellaneous = set("\u2669\u266a\u266b\u266c\u266d\u266e\u266f")
        assert all(0x2640 <= ord(c) <= 0x267F for c in miscellaneous)

        # allow hyphens "-" and single quotes "'" between words, but not at the beginning
        result = {self.encoding.encode(" -")[0], self.encoding.encode(" '")[0]}
        for symbol in symbols + list(miscellaneous):
            for tokens in [
                self.encoding.encode(symbol),
                self.encoding.encode(" " + symbol),
            ]:
                if len(tokens) == 1 or symbol in miscellaneous:
                    result.add(tokens[0])

        return tuple(sorted(result))

    def split_to_word_tokens(self, tokens: List[int]):
        if self.language in {"zh", "ja", "th", "lo", "my", "yue"}:
            # These languages don't typically use spaces, so it is difficult to split words
            # without morpheme analysis. Here, we instead split words at any
            # position where the tokens are decoded as valid unicode points
            return self.split_tokens_on_unicode(tokens)

        return self.split_tokens_on_spaces(tokens)

    def split_tokens_on_unicode(self, tokens: List[int]):
        decoded_full = self.decode_with_timestamps(tokens)
        replacement_char = "\ufffd"

        words = []
        word_tokens = []
        current_tokens = []
        unicode_offset = 0

        for token in tokens:
            current_tokens.append(token)
            decoded = self.decode_with_timestamps(current_tokens)

            if (
                replacement_char not in decoded
                or decoded_full[unicode_offset + decoded.index(replacement_char)]
                == replacement_char
            ):
                words.append(decoded)
                word_tokens.append(current_tokens)
                current_tokens = []
                unicode_offset += len(decoded)

        return words, word_tokens

    def split_tokens_on_spaces(self, tokens: List[int]):
        subwords, subword_tokens_list = self.split_tokens_on_unicode(tokens)
        words = []
        word_tokens = []

        for subword, subword_tokens in zip(subwords, subword_tokens_list):
            special = subword_tokens[0] >= self.eot
            with_space = subword.startswith(" ")
            punctuation = subword.strip() in string.punctuation
            if special or with_space or punctuation or len(words) == 0:
                words.append(subword)
                word_tokens.append(subword_tokens)
            else:
                words[-1] = words[-1] + subword
                word_tokens[-1].extend(subword_tokens)

        return words, word_tokens


@lru_cache(maxsize=None)
def get_encoding(name: str = "gpt2", num_languages: int = 99) -> tiktoken.Encoding:
    """Load tiktoken encoding from bundled assets.

    Uses assets from mlxk2/assets/whisper/ instead of mlx-audio's removed assets.
    """
    vocab_path = _ASSETS_DIR / f"{name}.tiktoken"

    if not vocab_path.exists():
        raise FileNotFoundError(
            f"Tiktoken vocabulary file not found: {vocab_path}\n"
            f"This is an mlx-audio Issue #479 workaround.\n"
            f"Expected assets in: {_ASSETS_DIR}"
        )

    with open(vocab_path) as fid:
        ranks = {
            base64.b64decode(token): int(rank)
            for token, rank in (line.split() for line in fid if line)
        }

    n_vocab = len(ranks)
    special_tokens = {}

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


@lru_cache(maxsize=None)
def get_tokenizer(
    multilingual: bool,
    *,
    num_languages: int = 99,
    language: Optional[str] = None,
    task: Optional[str] = None,  # Literal["transcribe", "translate", None]
) -> Tokenizer:
    """Get a Whisper tokenizer.

    Args:
        multilingual: Whether to use multilingual tokenizer (True for most models)
        num_languages: Number of languages supported (default 99)
        language: Language code (e.g., 'en', 'de') or None for auto-detect
        task: 'transcribe' or 'translate' or None

    Returns:
        Tokenizer instance
    """
    if language is not None:
        language = language.lower()
        if language not in LANGUAGES:
            if language in TO_LANGUAGE_CODE:
                language = TO_LANGUAGE_CODE[language]
            else:
                raise ValueError(f"Unsupported language: {language}")

    if multilingual:
        encoding_name = "multilingual"
        language = language or "en"
        task = task or "transcribe"
    else:
        encoding_name = "gpt2"
        language = None
        task = None

    encoding = get_encoding(name=encoding_name, num_languages=num_languages)

    return Tokenizer(
        encoding=encoding, num_languages=num_languages, language=language, task=task
    )
