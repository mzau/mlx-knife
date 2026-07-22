"""Live pipe tests for the `mlxk embed` ENCODER path (ADR-015 Slice B).

Opt-in / env-gated — skipped unless MLXK2_ENABLE_ALPHA_FEATURES=1. Drives the CLI as a subprocess
(real mlx + vendored BERT, no conftest stub) over BOTH verified encoder branches when present:

  - bge-small-en-v1.5-4bit  — cache, CLS pooling, 4-bit quantized, WordPiece tokenizer
  - multilingual-e5-small-mlx — workspace, mean pooling, float16, XLM-R sentencepiece tokenizer

Each model self-skips if it is not resolvable, so the suite runs whatever subset is available.
Requires HF_HOME / MLXK_WORKSPACE_HOME pointing at the models.
"""

import json
import math
import os
import sys
import subprocess

import pytest

from .test_utils import EMBED_TEST_MODELS

pytestmark = pytest.mark.live_e2e

if not os.getenv("MLXK2_ENABLE_ALPHA_FEATURES"):
    pytest.skip("embed is alpha-gated; set MLXK2_ENABLE_ALPHA_FEATURES=1", allow_module_level=True)

# Encoder fixtures (CLS + mean branches) from the shared verified-representatives set —
# single source of truth in test_utils.EMBED_TEST_MODELS, no per-test hardcoding.
ENCODER_MODELS = [
    (m["id"], m["dims"]) for m in EMBED_TEST_MODELS.values() if m["kind"] == "encoder"
]


def _run(model, args, stdin=None, timeout=300):
    env = dict(os.environ)
    env["MLXK2_ENABLE_ALPHA_FEATURES"] = "1"
    return subprocess.run(
        [sys.executable, "-m", "mlxk2.cli", "embed", model] + args,
        input=stdin, text=True, capture_output=True, env=env, timeout=timeout,
    )


def _cos(u, v):
    dot = sum(a * b for a, b in zip(u, v))
    nu = math.sqrt(sum(a * a for a in u))
    nv = math.sqrt(sum(b * b for b in v))
    return dot / (nu * nv)


def _require(model):
    """Skip this model if it can't be embedded (not pulled / wrong env)."""
    p = _run(model, ["ping"], timeout=300)
    if p.returncode != 0:
        pytest.skip(f"encoder model '{model}' not available: {p.stderr.strip()[:200]}")


@pytest.mark.parametrize("model,dims", ENCODER_MODELS)
def test_single_line_jsonl_normalized(model, dims):
    _require(model)
    p = _run(model, ["-"], stdin="machine learning tutorial")
    assert p.returncode == 0, p.stderr
    lines = [l for l in p.stdout.splitlines() if l.strip()]
    assert len(lines) == 1
    rec = json.loads(lines[0])
    emb, meta = rec["embedding"], rec["metadata"]
    assert rec["text"] == "machine learning tutorial"
    assert len(emb) == meta["dimensions"] == dims
    assert abs(math.sqrt(sum(x * x for x in emb)) - 1.0) < 1e-3      # L2-normalized
    assert meta["model"] and not meta["model"].startswith("/")      # org/name, never a path
    assert "content_hash" in meta
    assert meta["device"] == "gpu"


@pytest.mark.parametrize("model,dims", ENCODER_MODELS)
def test_batch_passthrough_and_cosine(model, dims):
    _require(model)
    stdin = "\n".join([
        json.dumps({"id": "a", "text": "machine learning tutorial"}),
        json.dumps({"id": "b", "text": "a beginners guide to machine learning"}),
        json.dumps({"id": "c", "text": "the price of bananas in Ecuador this year"}),
    ])
    p = _run(model, ["--batch"], stdin=stdin)
    assert p.returncode == 0, p.stderr
    recs = [json.loads(l) for l in p.stdout.splitlines() if l.strip()]
    assert [r["id"] for r in recs] == ["a", "b", "c"]               # passthrough + order
    assert all("embedding" in r and "metadata" in r for r in recs)
    e = {r["id"]: r["embedding"] for r in recs}
    # Vectors must be non-degenerate (catches a dead forward) and semantically sane.
    assert any(abs(x) > 1e-6 for x in e["a"])
    assert _cos(e["a"], e["b"]) > _cos(e["a"], e["c"])             # related > unrelated


@pytest.mark.parametrize("model,dims", ENCODER_MODELS)
def test_query_differs_from_document(model, dims):
    _require(model)
    doc = _run(model, ["-"], stdin="what is machine learning")
    qry = _run(model, ["-", "--query"], stdin="what is machine learning")
    assert doc.returncode == 0 and qry.returncode == 0, (doc.stderr, qry.stderr)
    dv = json.loads(doc.stdout.splitlines()[0])["embedding"]
    qv = json.loads(qry.stdout.splitlines()[0])["embedding"]
    # bge prefixes only the query side; e5 uses query:/passage: — either way the vectors differ.
    assert _cos(dv, qv) < 0.99  # prefix must shift the query vector (real: bge ~0.965, e5 ~0.945; 0.9999 was too loose to catch a dead prefix)
