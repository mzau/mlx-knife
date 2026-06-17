"""Unit tests for the embeddings HTTP handler (ADR-015 Slice D1).

The EmbeddingRunner is mocked — these validate handle_embeddings()'s request shaping,
base64/float wire encoding, usage counting, and error cases without loading a model.
"""

import base64
import struct
import threading

import pytest
from fastapi import HTTPException

from mlxk2.core.server.handlers.embeddings import handle_embeddings, _to_wire


class _FakeTok:
    def encode(self, t):
        # 1 token per whitespace-separated word (deterministic, model-free).
        return list(range(len(t.split())))


class _FakeRunner:
    dimensions = 3

    def __init__(self):
        self.tokenizer = _FakeTok()
        self.calls = []

    def embed(self, texts, *, input_type="document", instruct=None):
        self.calls.append((list(texts), input_type, instruct))
        return [[0.1, 0.2, 0.3] for _ in texts]


def _decode_b64(s):
    raw = base64.b64decode(s)
    return list(struct.unpack(f"<{len(raw) // 4}f", raw))


def test_single_string_float():
    r = _FakeRunner()
    res = handle_embeddings(runner=r, model_identity="org/m", inputs=["hello"], encoding_format="float")
    assert res["object"] == "list"
    assert res["model"] == "org/m"
    assert len(res["data"]) == 1
    d = res["data"][0]
    assert d["object"] == "embedding" and d["index"] == 0
    assert d["embedding"] == [0.1, 0.2, 0.3]


def test_list_input_indices_and_order():
    r = _FakeRunner()
    res = handle_embeddings(runner=r, model_identity="org/m", inputs=["a", "b", "c"], encoding_format="float")
    assert [d["index"] for d in res["data"]] == [0, 1, 2]
    assert len(res["data"]) == 3


def test_base64_roundtrip_matches_float():
    r = _FakeRunner()
    res = handle_embeddings(runner=r, model_identity="org/m", inputs=["hello"], encoding_format="base64")
    emb = res["data"][0]["embedding"]
    assert isinstance(emb, str)
    assert _decode_b64(emb) == pytest.approx([0.1, 0.2, 0.3], abs=1e-6)


def test_usage_counts_tokens():
    r = _FakeRunner()
    res = handle_embeddings(runner=r, model_identity="org/m",
                            inputs=["one two three", "four"], encoding_format="float")
    assert res["usage"]["prompt_tokens"] == 4         # 3 words + 1 word via _FakeTok
    assert res["usage"]["total_tokens"] == 4          # embeddings have no completion


def test_input_type_and_instruct_forwarded():
    r = _FakeRunner()
    handle_embeddings(runner=r, model_identity="org/m", inputs=["q"], encoding_format="float",
                      input_type="query", instruct="Find docs")
    texts, input_type, instruct = r.calls[-1]
    assert input_type == "query" and instruct == "Find docs"


def test_empty_input_400():
    r = _FakeRunner()
    with pytest.raises(HTTPException) as ei:
        handle_embeddings(runner=r, model_identity="org/m", inputs=[], encoding_format="float")
    assert ei.value.status_code == 400
    assert r.calls == []


def test_empty_string_input_400():
    r = _FakeRunner()
    with pytest.raises(HTTPException) as ei:
        handle_embeddings(runner=r, model_identity="org/m", inputs=[""], encoding_format="float")
    assert ei.value.status_code == 400
    assert r.calls == []                               # rejected before calling embed()


def test_dimensions_mismatch_400_before_inference():
    r = _FakeRunner()
    with pytest.raises(HTTPException) as ei:
        handle_embeddings(runner=r, model_identity="org/m", inputs=["x"], dimensions=999)
    assert ei.value.status_code == 400
    assert r.calls == []                               # rejected before calling embed()


def test_dimensions_match_ok():
    r = _FakeRunner()
    res = handle_embeddings(runner=r, model_identity="org/m", inputs=["x"], dimensions=3,
                            encoding_format="float")
    assert len(res["data"]) == 1


def test_bad_encoding_format_400():
    r = _FakeRunner()
    with pytest.raises(HTTPException) as ei:
        handle_embeddings(runner=r, model_identity="org/m", inputs=["x"], encoding_format="hex")
    assert ei.value.status_code == 400
    assert r.calls == []


def test_bad_input_type_400():
    r = _FakeRunner()
    with pytest.raises(HTTPException) as ei:
        handle_embeddings(runner=r, model_identity="org/m", inputs=["x"], input_type="banana")
    assert ei.value.status_code == 400


def test_non_string_input_400():
    r = _FakeRunner()
    with pytest.raises(HTTPException) as ei:
        handle_embeddings(runner=r, model_identity="org/m", inputs=[123], encoding_format="float")
    assert ei.value.status_code == 400


def test_lock_used_and_released():
    r = _FakeRunner()
    lock = threading.Lock()
    res = handle_embeddings(runner=r, model_identity="org/m", inputs=["x"], encoding_format="float",
                            lock=lock)
    assert len(res["data"]) == 1
    assert not lock.locked()                           # released after use


def test_inference_failure_500():
    class _BoomRunner(_FakeRunner):
        def embed(self, texts, **k):
            raise RuntimeError("kaboom")

    with pytest.raises(HTTPException) as ei:
        handle_embeddings(runner=_BoomRunner(), model_identity="org/m", inputs=["x"],
                          encoding_format="float")
    assert ei.value.status_code == 500


def test_to_wire_base64_is_little_endian_float32():
    assert base64.b64decode(_to_wire([1.0], "base64")) == struct.pack("<f", 1.0)


def test_to_wire_float_passthrough():
    assert _to_wire([1.0, 2.0], "float") == [1.0, 2.0]


def test_system_fingerprint_echoed():
    r = _FakeRunner()
    res = handle_embeddings(runner=r, model_identity="org/m", system_fingerprint="a1b2c3d4.gpu",
                            inputs=["x"], encoding_format="float")
    assert res["model"] == "org/m"                       # clean selector
    assert res["system_fingerprint"] == "a1b2c3d4.gpu"   # realization token, opaque passthrough


def test_system_fingerprint_defaults_none():
    r = _FakeRunner()
    res = handle_embeddings(runner=r, model_identity="org/m", inputs=["x"], encoding_format="float")
    assert res["system_fingerprint"] is None             # absent unless the backend supplies it


def test_usage_fallback_without_tokenizer():
    # No tokenizer + no count_tokens_fn -> whitespace estimate, never raises.
    class _NoTok(_FakeRunner):
        tokenizer = None

    res = handle_embeddings(runner=_NoTok(), model_identity="org/m",
                            inputs=["a b c"], encoding_format="float")
    assert res["usage"]["prompt_tokens"] == 3
