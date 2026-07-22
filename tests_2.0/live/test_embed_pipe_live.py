"""Live pipe tests for `mlxk embed` (ADR-015 Slice A, decoder vertical).

Opt-in / env-gated — skipped unless MLXK2_ENABLE_ALPHA_FEATURES=1 and an embedding model
is resolvable. Drives the CLI as a subprocess (real mlx, no conftest stub) and asserts the
JSONL contract, L2-normalization, dimensions, batch passthrough, the --json envelope, and the
end-to-end semantic sanity (related > unrelated cosine; query ≠ document).

Configure the model via MLXK_EMBED_MODEL (default: the ADR showcase
Qwen3-Embedding-0.6B-4bit-DWQ). Requires HF_HOME / MLXK_WORKSPACE_HOME pointing at it.
"""

import json
import math
import os
import sys
import subprocess

import pytest

pytestmark = pytest.mark.live_e2e

EMBED_MODEL = os.getenv("MLXK_EMBED_MODEL", "Qwen3-Embedding-0.6B-4bit-DWQ")

if not os.getenv("MLXK2_ENABLE_ALPHA_FEATURES"):
    pytest.skip("embed is alpha-gated; set MLXK2_ENABLE_ALPHA_FEATURES=1", allow_module_level=True)


def _run(args, stdin=None, timeout=300):
    env = dict(os.environ)
    env["MLXK2_ENABLE_ALPHA_FEATURES"] = "1"
    return subprocess.run(
        [sys.executable, "-m", "mlxk2.cli", "embed", EMBED_MODEL] + args,
        input=stdin, text=True, capture_output=True, env=env, timeout=timeout,
    )


def _cos(u, v):
    dot = sum(a * b for a, b in zip(u, v))
    nu = math.sqrt(sum(a * a for a in u))
    nv = math.sqrt(sum(b * b for b in v))
    return dot / (nu * nv)


@pytest.fixture(scope="module")
def model_available():
    """Skip the module if the embedding model can't be embedded (not pulled / wrong env)."""
    p = _run(["ping"], timeout=300)
    if p.returncode != 0:
        pytest.skip(f"embed model '{EMBED_MODEL}' not available: {p.stderr.strip()[:200]}")
    return EMBED_MODEL


def test_single_line_jsonl_normalized(model_available):
    p = _run(["-"], stdin="machine learning tutorial")
    assert p.returncode == 0, p.stderr
    lines = [l for l in p.stdout.splitlines() if l.strip()]
    assert len(lines) == 1
    rec = json.loads(lines[0])
    emb, meta = rec["embedding"], rec["metadata"]
    assert rec["text"] == "machine learning tutorial"
    assert len(emb) == meta["dimensions"]
    assert abs(math.sqrt(sum(x * x for x in emb)) - 1.0) < 1e-3  # L2-normalized
    assert meta["model"] and "/" not in meta["model"][:1]  # org/name, never an absolute path
    assert not meta["model"].startswith("/")               # no leaked filesystem path
    assert "content_hash" in meta
    assert meta["device"] == "gpu"


def test_batch_passthrough_and_cosine(model_available):
    stdin = "\n".join([
        json.dumps({"id": "a", "text": "machine learning tutorial"}),
        json.dumps({"id": "b", "text": "a beginners guide to machine learning"}),
        json.dumps({"id": "c", "text": "the price of bananas in Ecuador this year"}),
    ])
    p = _run(["--batch"], stdin=stdin)
    assert p.returncode == 0, p.stderr
    recs = [json.loads(l) for l in p.stdout.splitlines() if l.strip()]
    assert [r["id"] for r in recs] == ["a", "b", "c"]  # passthrough + order
    assert all("embedding" in r and "metadata" in r for r in recs)
    e = {r["id"]: r["embedding"] for r in recs}
    assert _cos(e["a"], e["b"]) > _cos(e["a"], e["c"])  # related > unrelated


def test_json_envelope(model_available):
    # --json renders ONE standard envelope (not JSONL); records ride under data.records.
    p = _run(["-", "--json"], stdin="machine learning tutorial")
    assert p.returncode == 0, p.stderr
    env = json.loads(p.stdout)  # single envelope object, parseable as a whole
    assert env["status"] == "success"
    assert env["command"] == "embed"
    assert env["error"] is None
    recs = env["data"]["records"]
    assert len(recs) == 1
    rec = recs[0]
    assert rec["text"] == "machine learning tutorial"
    meta = rec["metadata"]
    assert len(rec["embedding"]) == meta["dimensions"]
    assert {"model", "dimensions", "content_hash", "device"} <= set(meta)


def test_query_differs_from_document(model_available):
    doc = _run(["-"], stdin="what is machine learning")
    qry = _run(["-", "--query"], stdin="what is machine learning")
    assert doc.returncode == 0 and qry.returncode == 0
    dv = json.loads(doc.stdout.splitlines()[0])["embedding"]
    qv = json.loads(qry.stdout.splitlines()[0])["embedding"]
    assert _cos(dv, qv) < 0.98  # instruction prefix must shift the query vector (real ~0.94; 0.9999 was too loose to catch a dead prefix)


def test_cpu_device_stamp_and_determinism(model_available):
    """--cpu stamps device=cpu, is bit-deterministic, and agrees closely with (but is
    not identical to) the gpu backend — the README 'don't mix cpu/gpu vectors' caveat."""
    cpu1 = _run(["-", "--cpu"], stdin="what is machine learning")
    cpu2 = _run(["-", "--cpu"], stdin="what is machine learning")
    gpu = _run(["-"], stdin="what is machine learning")
    assert cpu1.returncode == 0 and cpu2.returncode == 0 and gpu.returncode == 0
    r1 = json.loads(cpu1.stdout.splitlines()[0])
    r2 = json.loads(cpu2.stdout.splitlines()[0])
    rg = json.loads(gpu.stdout.splitlines()[0])
    assert r1["metadata"]["device"] == "cpu"   # --cpu stamps the device
    assert rg["metadata"]["device"] == "gpu"   # default stays gpu
    v1, v2, vg = r1["embedding"], r2["embedding"], rg["embedding"]
    assert _cos(v1, v2) > 0.99999              # cpu is deterministic (real: bit-identical)
    assert 0.95 < _cos(v1, vg) < 0.999         # cpu≈gpu but distinct backends (real ~0.986; don't mix)
