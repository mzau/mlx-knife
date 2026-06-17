"""Live E2E tests for `mlxk embed-serve` (ADR-015 Slice D1).

Opt-in / env-gated — skipped unless MLXK2_ENABLE_ALPHA_FEATURES=1. Boots the real embed-serve
backend (separate process, real mlx) over the verified representatives (decoder + encoders) and
exercises the OpenAI /v1/embeddings contract:

  - httpx float + base64 round-trip (the SDK's default-base64 path, proven without `openai`)
  - batch ordering + semantic cosine sanity
  - the real OpenAI Python client (Use Case 4) — guarded by importorskip("openai")
  - parity: served vector == `mlxk embed` CLI vector (same model + device)

Each model self-skips if it is not resolvable/runnable in this env (HF_HOME / MLXK_WORKSPACE_HOME).
"""

import base64
import json
import math
import os
import signal
import struct
import subprocess
import sys
import time
from contextlib import contextmanager
from typing import Optional

import pytest

from .test_utils import EMBED_TEST_MODELS

try:
    import httpx
except ImportError:  # pragma: no cover
    httpx = None

pytestmark = pytest.mark.live_e2e

if not os.getenv("MLXK2_ENABLE_ALPHA_FEATURES"):
    pytest.skip("embed-serve is alpha-gated; set MLXK2_ENABLE_ALPHA_FEATURES=1",
                allow_module_level=True)

EMBED_PORT = 8766  # non-standard, distinct from LocalServer's 8765


def _runnable_or_skip(model_id: str):
    """Skip a model that isn't resolvable / runnable as an embedder in this env (no model load)."""
    from mlxk2.core.model_resolution import resolve_model_for_operation
    from mlxk2.operations.embed import _route_embedding

    resolved, commit, ambiguous = resolve_model_for_operation(model_id)
    if not resolved or ambiguous:
        pytest.skip(f"embed model '{model_id}' not resolvable in this env")
    route, _ = _route_embedding(resolved, commit)
    if route not in ("decoder", "encoder"):
        pytest.skip(f"embed model '{model_id}' not a runnable embedder here (route={route})")


@contextmanager
def EmbedServer(model: str, port: int = EMBED_PORT, timeout: int = 120, log_level: str = "warning"):
    """Boot a real embed-serve backend subprocess; yield its base URL; ensure cleanup."""
    if httpx is None:
        raise RuntimeError("httpx required for E2E tests (pip install httpx)")

    env = os.environ.copy()
    env["MLXK2_HOST"] = "127.0.0.1"
    env["MLXK2_PORT"] = str(port)
    env["MLXK2_LOG_LEVEL"] = log_level
    env["MLXK2_EMBED_MODEL"] = model
    env["MLXK2_EMBED_CPU"] = "0"

    proc = subprocess.Popen(
        [sys.executable, "-m", "mlxk2.core.embed_server_base"],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
        env=env, start_new_session=True,
    )
    url = f"http://127.0.0.1:{port}"

    start = time.time()
    last_err: Optional[Exception] = None
    while time.time() - start < timeout:
        if proc.poll() is not None:  # died during startup
            out, err = proc.communicate()
            raise RuntimeError(f"embed-serve exited early (rc={proc.returncode})\nSTDERR:\n{err}")
        try:
            if httpx.get(f"{url}/health", timeout=2.0).status_code == 200:
                break
        except Exception as e:
            last_err = e
            time.sleep(0.5)
    else:
        proc.kill()
        out, err = proc.communicate()
        raise TimeoutError(f"embed-serve not ready in {timeout}s (last={last_err})\nSTDERR:\n{err}")

    try:
        yield url
    finally:
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
        except (ProcessLookupError, PermissionError):
            proc.terminate()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
            except (ProcessLookupError, PermissionError):
                proc.kill()
            proc.wait(timeout=5)


@pytest.fixture(scope="module", params=list(EMBED_TEST_MODELS.values()),
                ids=lambda m: m["id"])
def embed_backend(request):
    m = request.param
    _runnable_or_skip(m["id"])
    with EmbedServer(m["id"]) as url:
        yield url, m["id"], m["dims"]


def _decode_b64(s):
    raw = base64.b64decode(s)
    return list(struct.unpack(f"<{len(raw) // 4}f", raw))


def _cos(u, v):
    dot = sum(a * b for a, b in zip(u, v))
    nu = math.sqrt(sum(a * a for a in u))
    nv = math.sqrt(sum(b * b for b in v))
    return dot / (nu * nv)


def _post(url, payload):
    return httpx.post(f"{url}/v1/embeddings", json=payload, timeout=120.0)


def test_health(embed_backend):
    url, model_id, _ = embed_backend
    r = httpx.get(f"{url}/health", timeout=5.0)
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_float_single(embed_backend):
    url, model_id, dims = embed_backend
    r = _post(url, {"model": model_id, "input": "machine learning tutorial",
                    "encoding_format": "float"})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["object"] == "list"
    assert body["model"] and not body["model"].startswith("/")     # org/name, never a path
    assert len(body["data"]) == 1
    emb = body["data"][0]["embedding"]
    assert isinstance(emb, list) and len(emb) == dims
    assert abs(math.sqrt(sum(x * x for x in emb)) - 1.0) < 1e-3     # L2-normalized
    assert body["usage"]["prompt_tokens"] > 0
    assert body["usage"]["total_tokens"] == body["usage"]["prompt_tokens"]


def test_base64_matches_float(embed_backend):
    url, model_id, _ = embed_backend
    f = _post(url, {"model": model_id, "input": "hello world", "encoding_format": "float"})
    b = _post(url, {"model": model_id, "input": "hello world", "encoding_format": "base64"})
    assert f.status_code == 200 and b.status_code == 200
    fv = f.json()["data"][0]["embedding"]
    bv = b.json()["data"][0]["embedding"]
    assert isinstance(bv, str)
    assert _decode_b64(bv) == pytest.approx(fv, abs=1e-5)


def test_batch_order_and_cosine(embed_backend):
    url, model_id, _ = embed_backend
    r = _post(url, {"model": model_id, "encoding_format": "float", "input": [
        "machine learning tutorial",
        "a beginners guide to machine learning",
        "the price of bananas in Ecuador this year",
    ]})
    assert r.status_code == 200, r.text
    data = r.json()["data"]
    assert [d["index"] for d in data] == [0, 1, 2]
    e = [d["embedding"] for d in data]
    assert any(abs(x) > 1e-6 for x in e[0])                        # non-degenerate
    assert _cos(e[0], e[1]) > _cos(e[0], e[2])                     # related > unrelated


def test_dimensions_mismatch_422_or_400(embed_backend):
    url, model_id, dims = embed_backend
    r = _post(url, {"model": model_id, "input": "x", "dimensions": dims + 7})
    assert r.status_code == 400, r.text
    assert r.json()["status"] == "error"


def test_concurrent_requests_serialized(embed_backend):
    # Regression (review finding): tokenizer access (embed + usage count) must be fully serialized
    # under the inference lock — otherwise the HF fast tokenizer raises "Already borrowed" -> HTTP
    # 500 under concurrent load. Fire many simultaneous requests; all must return 200.
    import concurrent.futures as cf

    url, model_id, _ = embed_backend

    def one(i):
        return _post(url, {"model": model_id, "input": f"concurrent request {i}",
                           "encoding_format": "float"}).status_code

    with cf.ThreadPoolExecutor(max_workers=12) as ex:
        codes = list(ex.map(one, range(24)))
    assert all(c == 200 for c in codes), f"non-200 under concurrency: {codes}"


def test_openai_sdk_use_case_4(embed_backend):
    # The canonical ADR Use Case 4: a real OpenAI client against the backend (default base64 path).
    openai = pytest.importorskip("openai")
    url, model_id, dims = embed_backend
    client = openai.OpenAI(base_url=f"{url}/v1", api_key="-")
    single = client.embeddings.create(model=model_id, input="how does stop-token detection work?")
    assert len(single.data) == 1
    assert len(single.data[0].embedding) == dims
    batch = client.embeddings.create(model=model_id, input=["alpha", "beta", "gamma"])
    assert len(batch.data) == 3
    assert [d.index for d in batch.data] == [0, 1, 2]


def test_cli_parity(embed_backend):
    # Served (float) vector must equal the `mlxk embed` CLI vector (same model + device=gpu).
    url, model_id, _ = embed_backend
    text = "vector store consistency requires one model and one device"
    served = _post(url, {"model": model_id, "input": text, "encoding_format": "float"})
    assert served.status_code == 200, served.text
    sv = served.json()["data"][0]["embedding"]

    env = dict(os.environ)
    env["MLXK2_ENABLE_ALPHA_FEATURES"] = "1"
    cli = subprocess.run(
        [sys.executable, "-m", "mlxk2.cli", "embed", model_id, "-"],
        input=text, text=True, capture_output=True, env=env, timeout=300,
    )
    assert cli.returncode == 0, cli.stderr
    cv = json.loads([ln for ln in cli.stdout.splitlines() if ln.strip()][0])["embedding"]
    assert len(cv) == len(sv)
    assert cv == pytest.approx(sv, abs=1e-5)
