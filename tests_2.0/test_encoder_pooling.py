"""Unit tests for the encoder (BERT) path: pooling/family/prefix inference + the encoder embed
loop (ADR-015 Slice B).

Pure logic + the ``_embed_encoder`` loop are exercised with a numpy-backed ``mlx.core`` shim (the
in-process conftest forces a non-subscriptable mlx stub, and there is no ``mlx.nn`` stub — so the
real BERT forward is covered only by the subprocess live smoke). Here we use a fake BERT model +
fake HF tokenizer to assert pooling, prefixing, and the no-EOS-append contract.
"""

import pytest

np = pytest.importorskip("numpy")

from mlxk2.core.embedding_runner import EmbeddingRunner
from mlxk2.core.encoders.pooling import infer_pool_and_family, encoder_prefix


# --- infer_pool_and_family --------------------------------------------------

@pytest.mark.parametrize("name,expected", [
    ("mlx-community/bge-small-en-v1.5-4bit", ("cls", "bge")),
    ("bge-base-en", ("cls", "bge")),
    ("mxbai-embed-large-v1", ("cls", "bge")),       # mxbai shares CLS + the bge query instruction
    ("multilingual-e5-small-mlx", ("mean", "e5")),
    ("intfloat/e5-base-v2", ("mean", "e5")),
    ("sentence-transformers/all-MiniLM-L6-v2", ("mean", "generic")),
    ("some/unknown-encoder", ("mean", "generic")),
])
def test_infer_pool_and_family(name, expected):
    assert infer_pool_and_family(name) == expected


# --- encoder_prefix ---------------------------------------------------------

def test_bge_prefix_query_only():
    assert encoder_prefix("bge", "query").startswith("Represent this sentence")
    assert encoder_prefix("bge", "document") == ""  # bge documents are embedded raw


def test_e5_prefix_both_sides():
    assert encoder_prefix("e5", "query") == "query: "
    assert encoder_prefix("e5", "document") == "passage: "


def test_generic_prefix_none():
    assert encoder_prefix("generic", "query") == ""
    assert encoder_prefix("generic", "document") == ""


# --- _pool: CLS vs mean (numpy shim) ----------------------------------------

class _NpLinalg:
    @staticmethod
    def norm(v):
        return float(np.linalg.norm(np.asarray(v)))


class _NpMx:
    linalg = _NpLinalg()
    float32 = np.float32

    @staticmethod
    def array(x):
        return np.array(x)

    @staticmethod
    def eval(x):
        return None


def test_cls_pool_picks_first_token_and_normalizes():
    r = EmbeddingRunner("fake")
    r._mx = _NpMx
    r._pool_strategy = "cls"
    # CLS row [3,4,0,0] -> normalized [0.6,0.8,0,0]; other rows must be ignored.
    hidden = np.array([[[3.0, 4.0, 0.0, 0.0],
                        [9.0, 9.0, 9.0, 9.0],
                        [1.0, 1.0, 1.0, 1.0]]])
    vec = r._finalize(hidden)
    assert vec == pytest.approx([0.6, 0.8, 0.0, 0.0], abs=1e-5)
    assert sum(x * x for x in vec) == pytest.approx(1.0, abs=1e-5)


# --- _embed_encoder loop: no EOS-append, prefix applied ---------------------

class _FakeBert:
    """Stands in for the vendored BertModel call; records the token ids it was given."""

    def __init__(self, hidden):
        self._hidden = hidden
        self.calls = []

    def __call__(self, inputs):
        self.calls.append(inputs)
        return self._hidden


class _FakeHFTokenizer:
    """Mimics a HF AutoTokenizer: callable returning {'input_ids': ...} with [CLS]/[SEP] markers.
    Has an eos_token_id to prove the encoder path never appends it (decoder-only behavior)."""

    eos_token_id = 102  # [SEP] — must NOT be appended by the encoder path

    def __init__(self):
        self.seen_texts = []

    def __call__(self, text, truncation=False, max_length=None):
        self.seen_texts.append(text)
        ids = [101] + [ord(c) % 50 + 1 for c in text] + [102]  # [CLS] ... [SEP]
        return {"input_ids": ids}


def _encoder_runner(hidden, *, family="generic", pool="mean"):
    r = EmbeddingRunner("fake")
    r._mx = _NpMx
    r._is_encoder = True
    r._encoder_family = family
    r._pool_strategy = pool
    r._max_seq = 512
    r._model_loaded = True
    r.model = _FakeBert(np.array(hidden))
    r.tokenizer = _FakeHFTokenizer()
    return r


def test_encoder_does_not_append_eos():
    hidden = [[[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]]
    r = _encoder_runner(hidden)
    r.embed(["hi"])
    fed = r.model.calls[-1].tolist()[0]
    expected = r.tokenizer.seen_texts and [101] + [ord(c) % 50 + 1 for c in "hi"] + [102]
    assert fed == expected            # exactly the tokenizer ids — no extra trailing EOS
    assert fed[-1] == 102             # ends on [SEP], not a re-appended token


def test_encoder_applies_e5_passage_prefix_on_document_side():
    hidden = [[[1.0, 0.0], [0.0, 1.0]]]
    r = _encoder_runner(hidden, family="e5")
    r.embed(["hello"], input_type="document")
    assert r.tokenizer.seen_texts[-1] == "passage: hello"
    r.embed(["hello"], input_type="query")
    assert r.tokenizer.seen_texts[-1] == "query: hello"


def test_encoder_embed_returns_normalized_vector_per_text():
    hidden = [[[3.0, 4.0], [0.0, 0.0]]]
    r = _encoder_runner(hidden, pool="cls")
    out = r.embed(["a", "b"])
    assert len(out) == 2
    for vec in out:
        assert sum(x * x for x in vec) == pytest.approx(1.0, abs=1e-5)
