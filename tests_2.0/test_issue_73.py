"""Regression tests for issue #73: one detokenizer per generation, not per decode.

`TokenizerWrapper.detokenizer` is a factory — reading it constructs a new instance over
the whole vocabulary. The streaming loop read it up to three times per token, which cost
~61 ms each on a 151k BPE vocabulary and made streaming 65x slower than the same
generation unstreamed.

The mocks elsewhere in this suite assign `detokenizer` as a plain attribute, so every read
returns the same object and the defect is invisible. The tokenizer here counts reads and
returns a fresh instance each time, the way the real one does.
"""

from contextlib import contextmanager
from unittest.mock import Mock, patch

import mlx.core as mx
from mlxk2.core.runner import MLXRunner


# A token id that is not in eos_token_ids, so generation runs to max_tokens. The count is
# above the runner's 10-token sliding window, where each token used to pay three reads.
TOKEN_COUNT = 15


class RecordingDetokenizer:
    """Mimics BPEStreamingDetokenizer, and records that reset() was called."""

    def __init__(self, decode_func):
        self.decode_func = decode_func
        self.tokens = []
        self.resets = 0
        self._text = ""

    def reset(self):
        self.tokens = []
        self.resets += 1
        self._text = ""

    def add_token(self, token_id):
        self.tokens.append(token_id)

    def finalize(self):
        self._text = self.decode_func(self.tokens)

    @property
    def text(self):
        return self._text


def _decode(tokens):
    """Deterministic decode whose output grows monotonically with the token list."""
    return "".join(f"t{int(t)}." for t in tokens)


def _factory_tokenizer(created):
    """A tokenizer whose `detokenizer` is a property, appending each instance to `created`."""

    class FactoryTokenizer(Mock):
        @property
        def detokenizer(self):
            detok = RecordingDetokenizer(_decode)
            created.append(detok)
            return detok

    tok = FactoryTokenizer()
    tok.eos_token = "</s>"
    tok.eos_token_id = 999
    tok.eos_token_ids = {999}
    tok.pad_token = None
    tok.additional_special_tokens = []
    tok.added_tokens_decoder = {}
    tok.chat_template = None
    tok.name_or_path = "mock-issue73"
    tok.encode = lambda *args, **kwargs: [1, 2, 3]
    return tok


@contextmanager
def _runner_with_factory_tokenizer(tmp_path, created, token_count=TOKEN_COUNT):
    """Bring up an MLXRunner over a stub model and the counting tokenizer."""
    model_name = "test-model"
    with patch('mlxk2.core.runner.load') as mock_load, \
         patch('mlxk2.core.runner.resolve_model_for_operation') as mock_resolve, \
         patch('mlxk2.core.runner.get_current_model_cache') as mock_cache, \
         patch('mlxk2.core.runner.hf_to_cache_dir') as mock_hf_to_cache, \
         patch('mlxk2.core.runner.get_model_context_length') as mock_context, \
         patch('mlxk2.core.runner.generate_step') as mock_gen:
        mock_resolve.return_value = (model_name, None, None)
        mock_cache.return_value = tmp_path
        mock_hf_to_cache.return_value = f"models--{model_name}"
        mock_context.return_value = 8192
        (tmp_path / f"models--{model_name}" / "snapshots" / "abc123").mkdir(parents=True)
        mock_load.return_value = (Mock(), _factory_tokenizer(created))
        mock_gen.return_value = [
            (mx.array([i + 1]), mx.zeros(1)) for i in range(token_count)
        ]
        with MLXRunner(model_name) as runner:
            yield runner


class TestOneDetokenizerPerGeneration:
    def test_streaming_builds_exactly_one(self, tmp_path):
        created = []
        with _runner_with_factory_tokenizer(tmp_path, created) as runner:
            list(runner.generate_streaming("test prompt", max_tokens=TOKEN_COUNT))
        assert len(created) == 1, (
            f"expected one detokenizer for the whole generation, got {len(created)} "
            f"for {TOKEN_COUNT} tokens — the factory is being read inside the loop again"
        )

    def test_batch_builds_exactly_one(self, tmp_path):
        created = []
        with _runner_with_factory_tokenizer(tmp_path, created) as runner:
            runner.generate_batch("test prompt", max_tokens=TOKEN_COUNT)
        assert len(created) == 1, f"expected one detokenizer, got {len(created)}"

    def test_the_single_instance_is_reset_per_decode(self, tmp_path):
        """Reuse is only safe because each decode resets. Without it text accumulates."""
        created = []
        with _runner_with_factory_tokenizer(tmp_path, created) as runner:
            list(runner.generate_streaming("test prompt", max_tokens=TOKEN_COUNT))
        assert created[0].resets >= TOKEN_COUNT, (
            f"the reused detokenizer was reset {created[0].resets} times for "
            f"{TOKEN_COUNT} tokens; a missed reset makes decoded text accumulate"
        )

    def test_streamed_text_covers_every_token(self, tmp_path):
        """Guards the reuse against dropping or duplicating output."""
        created = []
        with _runner_with_factory_tokenizer(tmp_path, created) as runner:
            chunks = list(runner.generate_streaming("test prompt", max_tokens=TOKEN_COUNT))
        streamed = "".join(chunks)
        for token_id in range(1, TOKEN_COUNT + 1):
            assert f"t{token_id}." in streamed, (
                f"token {token_id} is missing from the streamed text: {streamed!r}"
            )
