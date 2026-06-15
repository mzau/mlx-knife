"""Unit tests for embed_operation envelopes + routing (ADR-015 Slices A + B).

EmbeddingRunner is mocked — these validate the operation's control flow, error envelopes,
batch passthrough, and metadata stamping without loading any model.
"""

import json
from pathlib import Path

import pytest
from unittest.mock import patch

import mlxk2.operations.embed as embed_mod
import mlxk2.core.model_resolution as mr
from mlxk2.operations.embed import embed_operation, _route_embedding


class _FakeRunner:
    dimensions = 3

    def __init__(self, *a, **k):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def embed(self, texts, **k):
        return [[0.1, 0.2, 0.3] for _ in texts]


def _patches(resolved=("mlx-community/Qwen3-Embedding-0.6B", None, None), route=("decoder", "qwen3"),
             identity=("mlx-community/Qwen3-Embedding-0.6B", "sha256:deadbeef")):
    # route is the (route, model_type) tuple returned by _route_embedding.
    return (
        patch.object(embed_mod, "resolve_model_for_operation", return_value=resolved),
        patch.object(embed_mod, "_route_embedding", return_value=route),
        patch.object(embed_mod, "EmbeddingRunner", _FakeRunner),
        patch.object(embed_mod, "model_display_identity", return_value=identity),
    )


def test_single_envelope():
    p1, p2, p3, p4 = _patches()
    with p1, p2, p3, p4:
        res = embed_operation("Qwen3-Embedding-0.6B", "hello")
    assert res["status"] == "success"
    recs = res["data"]["records"]
    assert len(recs) == 1
    assert recs[0]["text"] == "hello"
    assert recs[0]["embedding"] == [0.1, 0.2, 0.3]
    assert recs[0]["metadata"] == {
        "model": "mlx-community/Qwen3-Embedding-0.6B",
        "dimensions": 3,
        "content_hash": "sha256:deadbeef",
        "device": "gpu",
    }


def test_cpu_flag_stamps_device():
    p1, p2, p3, p4 = _patches()
    with p1, p2, p3, p4:
        res = embed_operation("M", "hello", cpu=True)
    assert res["data"]["records"][0]["metadata"]["device"] == "cpu"


def test_batch_passthrough_fields_preserved():
    lines = [json.dumps({"text": "a", "id": 1}), json.dumps({"text": "b", "id": 2})]
    p1, p2, p3, p4 = _patches()
    with p1, p2, p3, p4:
        res = embed_operation("M", batch=True, batch_lines=lines)
    recs = res["data"]["records"]
    assert [r["id"] for r in recs] == [1, 2]
    assert all("embedding" in r and "metadata" in r for r in recs)
    assert all(r["metadata"]["content_hash"] == "sha256:deadbeef" for r in recs)


def test_ambiguous_match_error():
    with patch.object(embed_mod, "resolve_model_for_operation", return_value=(None, None, ["a", "b"])):
        res = embed_operation("amb", "x")
    assert res["status"] == "error"
    assert res["error"]["type"] == "AmbiguousMatch"


def test_model_not_found_error():
    with patch.object(embed_mod, "resolve_model_for_operation", return_value=(None, None, None)):
        res = embed_operation("nope", "x")
    assert res["error"]["type"] == "ModelNotFound"


def test_bert_encoder_runs_via_runner():
    # Slice B: model_type bert is runnable — the operation must produce a success envelope,
    # NOT a reject (the runner self-selects the encoder path).
    p1, p2, p3, p4 = _patches(resolved=("mlx-community/bge-small-en-v1.5-4bit", None, None),
                              route=("encoder", "bert"),
                              identity=("mlx-community/bge-small-en-v1.5-4bit", "abc123"))
    with p1, p2, p3, p4:
        res = embed_operation("bge-small-en-v1.5-4bit", "x")
    assert res["status"] == "success"
    assert res["data"]["records"][0]["metadata"]["model"] == "mlx-community/bge-small-en-v1.5-4bit"


def test_declared_encoder_route_not_implemented():
    # A declared-but-not-vendored encoder (xlm-roberta/modernbert/nomic_bert) rejects honestly,
    # naming its model_type — never a silent failure.
    p1, p2, p3, p4 = _patches(resolved=("intfloat/multilingual-e5-large", None, None),
                              route=("encoder_declared", "xlm-roberta"))
    with p1, p2, p3, p4:
        res = embed_operation("multilingual-e5-large", "x")
    assert res["error"]["type"] == "NotImplemented"
    assert "xlm-roberta" in res["error"]["message"]


def test_unrecognized_model_not_an_embedder():
    p1, p2, p3, p4 = _patches(resolved=("Llama-3", None, None), route=(None, "llama"))
    with p1, p2, p3, p4:
        res = embed_operation("Llama-3", "x")
    assert res["error"]["type"] == "NotAnEmbedder"


def test_malformed_jsonl_line():
    p1, p2, p3, p4 = _patches()
    with p1, p2, p3, p4:
        res = embed_operation("M", batch=True, batch_lines=["{not valid json}"])
    assert res["error"]["type"] == "ValidationError"


def test_batch_line_missing_text_field():
    p1, p2, p3, p4 = _patches()
    with p1, p2, p3, p4:
        res = embed_operation("M", batch=True, batch_lines=[json.dumps({"foo": "bar"})])
    assert res["error"]["type"] == "ValidationError"


def test_missing_text_non_batch():
    p1, p2, p3, p4 = _patches()
    with p1, p2, p3, p4:
        res = embed_operation("M", text=None)
    assert res["error"]["type"] == "ValidationError"


# --- _route_embedding: exercise the REAL routing logic against synthetic configs ---

def _route(name, config):
    with patch.object(embed_mod, "resolve_model_dir", return_value=Path("/fake")), \
         patch.object(embed_mod, "_load_config_json", return_value=config):
        return _route_embedding(name, None)


def test_route_qwen3_embedding_is_decoder():
    cfg = {"model_type": "qwen3", "architectures": ["Qwen3ForCausalLM"]}
    assert _route("mlx-community/Qwen3-Embedding-0.6B-4bit-DWQ", cfg) == ("decoder", "qwen3")


def test_route_plain_qwen3_chat_is_not_an_embedder():
    # A normal qwen3 chat model (no embed name) must not be treated as an embedder.
    cfg = {"model_type": "qwen3", "architectures": ["Qwen3ForCausalLM"]}
    assert _route("mlx-community/Qwen3-4B-Instruct", cfg)[0] is None


def test_route_bert_encoder_is_runnable():
    cfg = {"model_type": "bert", "architectures": ["BertModel"]}
    assert _route("bge-small-en-v1.5", cfg) == ("encoder", "bert")


@pytest.mark.parametrize("mt", ["xlm-roberta", "modernbert", "nomic_bert"])
def test_route_declared_but_not_runnable_encoders(mt):
    # Declared encoder embedders that Slice B does not vendor -> "encoder_declared" (honest reject).
    cfg = {"model_type": mt, "architectures": ["SomeModel"]}
    assert _route(f"some/{mt}-model", cfg) == ("encoder_declared", mt)


def test_route_gemma3_text_causal_lm_is_not_an_embedder():
    # ADR-015 review Finding 3: a plain causal Gemma-3 text LM (model_type gemma3_text)
    # must NOT be misclassified as an (unvendored) encoder embedder.
    cfg = {"model_type": "gemma3_text", "architectures": ["Gemma3ForCausalLM"]}
    assert _route("google/gemma-3-1b-it", cfg)[0] is None


def test_route_embeddinggemma_deferred_to_slice_c():
    # gemma3_text is shared between causal Gemma-3 text LMs and EmbeddingGemma. Slice B does
    # NOT attempt the principled distinction (sentence-transformers sidecar / bidirectional
    # attention) — that is Slice C's config-first detection. So EmbeddingGemma is simply not
    # routed as a runnable embedder here; it falls through to "not recognized as an embedder".
    cfg = {"model_type": "gemma3_text", "architectures": ["Gemma3TextModel"]}
    assert _route("google/embeddinggemma-300m", cfg)[0] is None


def test_route_unknown_text_model_is_none():
    cfg = {"model_type": "llama", "architectures": ["LlamaForCausalLM"]}
    assert _route("meta-llama/Llama-3.1-8B-Instruct", cfg)[0] is None


# --- model_display_identity: portable org/name + content_hash, never an absolute path ---

def test_identity_workspace_uses_source_repo_and_content_hash():
    meta = {"source_repo": "mlx-community/Qwen3-Embedding-0.6B-4bit-DWQ",
            "content_hash": "sha256:2827abc"}
    with patch.object(mr, "is_workspace_path", return_value=True), \
         patch.object(mr, "read_workspace_metadata", return_value=meta):
        disp, ch = mr.model_display_identity("/Volumes/mz/mlx-models/Qwen3-Embedding-0.6B-4bit-DWQ", None)
    assert disp == "mlx-community/Qwen3-Embedding-0.6B-4bit-DWQ"  # org/name, NOT the path
    assert ch == "sha256:2827abc"


def test_identity_unmanaged_workspace_falls_back_to_basename():
    with patch.object(mr, "is_workspace_path", return_value=True), \
         patch.object(mr, "read_workspace_metadata", return_value={}):
        disp, ch = mr.model_display_identity("/abs/path/My-Local-Model", None)
    assert disp == "My-Local-Model"  # basename, never the absolute path
    assert ch is None


def test_identity_cache_uses_name_and_explicit_revision():
    with patch.object(mr, "is_workspace_path", return_value=False):
        disp, ch = mr.model_display_identity("mlx-community/Qwen3-Embedding-4B-4bit-DWQ", "b5d88f1fe49b")
    assert disp == "mlx-community/Qwen3-Embedding-4B-4bit-DWQ"
    assert ch == "b5d88f1fe49b"
