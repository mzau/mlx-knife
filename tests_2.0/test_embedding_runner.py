"""Unit tests for EmbeddingRunner pooling/normalize/prefix math (ADR-015 Slice A).

These exercise embed() directly with a fake model/tokenizer and a numpy-backed array shim
for ``self._mx`` — no disk model, no mlx_lm.load. The in-process pytest run forces the
lightweight ``mlx.core`` stub (conftest) whose arrays aren't subscriptable, so we use numpy
as a faithful stand-in for the array ops (subscript, mean, linalg.norm, tolist). Real mlx is
covered end-to-end by the subprocess live smoke (which does not load the stub).
"""

import pytest

np = pytest.importorskip("numpy")

from mlxk2.core.embedding_runner import EmbeddingRunner


class _NpLinalg:
    @staticmethod
    def norm(v):
        return float(np.linalg.norm(np.asarray(v)))


class _NpMx:
    """Minimal numpy-backed stand-in for the mlx.core surface embed() uses."""

    linalg = _NpLinalg()
    float32 = np.float32

    @staticmethod
    def array(x):
        return np.array(x)

    @staticmethod
    def eval(x):  # noqa: D401 - lazy-eval no-op for numpy
        return None


class _FakeBody:
    """Stands in for ``model.model`` (the transformer body). Returns a fixed hidden state
    and records the inputs it was called with (so EOS-append can be asserted)."""

    def __init__(self, hidden):
        self._hidden = hidden
        self.calls = []

    def __call__(self, inputs):
        self.calls.append(inputs)
        return self._hidden


class _FakeModel:
    def __init__(self, hidden, model_type="qwen3", hidden_size=4):
        self.model = _FakeBody(hidden)
        self.model_type = model_type

        class _Args:
            pass

        self.args = _Args()
        self.args.hidden_size = hidden_size


class _FakeTokenizer:
    def __init__(self, eos_token_id=None):
        self.eos_token_id = eos_token_id
        self.encoded = []  # list of (text, ids) as returned by encode (pre-EOS-append)

    def encode(self, text):
        ids = [ord(c) % 100 + 1 for c in text] or [1]
        self.encoded.append((text, ids))
        return ids


def _make_runner(hidden, *, model_type="qwen3", eos=None, hidden_size=4, pool="last_token"):
    r = EmbeddingRunner("fake")
    r.model = _FakeModel(np.array(hidden), model_type=model_type, hidden_size=hidden_size)
    r.tokenizer = _FakeTokenizer(eos_token_id=eos)
    r._mx = _NpMx
    r._model_type = model_type
    r._dimensions = hidden_size
    r._pool_strategy = pool
    r._model_loaded = True
    return r


def test_last_token_pooling_and_l2_normalization():
    # last row [3,4,0,0] -> normalized [0.6,0.8,0,0]
    hidden = [[[1.0, 0.0, 0.0, 0.0],
               [0.0, 1.0, 0.0, 0.0],
               [3.0, 4.0, 0.0, 0.0]]]
    r = _make_runner(hidden)
    (vec,) = r.embed(["abc"])
    assert len(vec) == 4
    assert vec == pytest.approx([0.6, 0.8, 0.0, 0.0], abs=1e-5)
    assert sum(x * x for x in vec) == pytest.approx(1.0, abs=1e-5)


def test_mean_pooling_branch():
    # mean of rows = [1.5,2.0,0,0] -> normalized [0.6,0.8,0,0]
    hidden = [[[0.0, 0.0, 0.0, 0.0],
               [3.0, 4.0, 0.0, 0.0]]]
    r = _make_runner(hidden, pool="mean")
    (vec,) = r.embed(["xy"])
    assert vec == pytest.approx([0.6, 0.8, 0.0, 0.0], abs=1e-5)


def test_eos_appended_to_model_input():
    hidden = [[[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]]]
    r = _make_runner(hidden, eos=999)
    r.embed(["a"])
    last_inputs = r.model.model.calls[-1].tolist()[0]
    assert last_inputs[-1] == 999


def test_no_eos_when_tokenizer_lacks_one():
    hidden = [[[1.0, 0.0, 0.0, 0.0]]]
    r = _make_runner(hidden, eos=None)
    r.embed(["a"])
    last_inputs = r.model.model.calls[-1].tolist()[0]
    # "a" -> [ord('a')%100+1] == [98]; nothing appended.
    assert last_inputs == [98]


def test_query_prefix_applied_only_for_query():
    hidden = [[[1.0, 0.0, 0.0, 0.0]]]
    r = _make_runner(hidden, model_type="qwen3")
    r.embed(["weather"], input_type="document")
    assert r.tokenizer.encoded[-1][0] == "weather"
    r.embed(["weather"], input_type="query")
    q_text = r.tokenizer.encoded[-1][0]
    assert q_text.startswith("Instruct:") and q_text.endswith("Query:weather")


def test_instruct_override_used_in_query_prefix():
    hidden = [[[1.0, 0.0, 0.0, 0.0]]]
    r = _make_runner(hidden)
    r.embed(["q"], input_type="query", instruct="Find relevant code")
    assert "Find relevant code" in r.tokenizer.encoded[-1][0]


def test_non_qwen3_decoder_does_not_get_qwen3_prefix():
    hidden = [[[1.0, 0.0, 0.0, 0.0]]]
    r = _make_runner(hidden, model_type="mistral")
    r.embed(["q"], input_type="query")
    assert r.tokenizer.encoded[-1][0] == "q"  # no qwen3 instruction prefix


def test_batch_returns_one_vector_per_text():
    hidden = [[[1.0, 0.0, 0.0, 0.0]]]
    r = _make_runner(hidden)
    out = r.embed(["a", "b", "c"])
    assert len(out) == 3


def test_dimensions_property():
    hidden = [[[1.0, 0.0, 0.0, 0.0]]]
    r = _make_runner(hidden, hidden_size=4)
    assert r.dimensions == 4


def test_embed_encoder_stub_raises():
    r = EmbeddingRunner("fake")
    with pytest.raises(NotImplementedError):
        r.embed_encoder(["x"])


def test_embed_before_load_raises():
    r = EmbeddingRunner("fake")
    with pytest.raises(RuntimeError):
        r.embed(["x"])


def test_double_context_enter_raises():
    r = EmbeddingRunner("fake")
    r._context_entered = True
    with pytest.raises(RuntimeError):
        r.__enter__()
