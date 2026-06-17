"""Live E2E tests for serve's POST /v1/embeddings thin proxy (ADR-015 Slice D2).

Opt-in / env-gated — skipped unless MLXK2_ENABLE_ALPHA_FEATURES=1. Boots a real embed-serve
backend (separate process, real mlx) AND a real `mlxk serve` configured with
``--embed-backend`` (via MLXK2_EMBED_BACKEND), then exercises the D2 contract end-to-end:

  - parity: a vector served through serve's proxy == the same vector from the backend directly
  - base64-default + batch ordering survive the byte-for-byte proxy
  - the real OpenAI Python client against serve's port (one base URL for chat + embeddings)
  - resilience: serve with a dead backend -> 502; serve with no --embed-backend -> 501

The proxy is model-agnostic, so the parity tests use one runnable representative; the 502/501
paths need no model at all.
"""

import os

import pytest

try:
    import httpx
except ImportError:  # pragma: no cover
    httpx = None

pytestmark = pytest.mark.live_e2e

# Skip BEFORE importing the D1 live module (which also module-skips when alpha is unset).
if not os.getenv("MLXK2_ENABLE_ALPHA_FEATURES"):
    pytest.skip("serve --embed-backend is alpha-gated; set MLXK2_ENABLE_ALPHA_FEATURES=1",
                allow_module_level=True)

from .server_context import LocalServer  # noqa: E402
from .test_embed_serve_live import EmbedServer, _runnable_or_skip  # noqa: E402
from .test_utils import EMBED_TEST_MODELS  # noqa: E402

SERVE_PORT = 8767   # distinct from EmbedServer(8766) and LocalServer default(8765)


def _first_runnable():
    """Pick the first embed model runnable as an embedder in this env (no model load)."""
    from mlxk2.core.model_resolution import resolve_model_for_operation
    from mlxk2.operations.embed import _route_embedding

    for m in EMBED_TEST_MODELS.values():
        resolved, commit, ambiguous = resolve_model_for_operation(m["id"])
        if not resolved or ambiguous:
            continue
        route, _ = _route_embedding(resolved, commit)
        if route in ("decoder", "encoder"):
            return m
    return None


@pytest.fixture(scope="module")
def proxy_pair():
    """Boot embed-serve backend + serve --embed-backend; yield (serve_url, backend_url, id, dims)."""
    m = _first_runnable()
    if m is None:
        pytest.skip("no runnable embed model available in this env")
    _runnable_or_skip(m["id"])
    with EmbedServer(m["id"]) as backend_url:
        with LocalServer(model=None, port=SERVE_PORT,
                         extra_env={"MLXK2_EMBED_BACKEND": backend_url}) as serve_url:
            yield serve_url, backend_url, m["id"], m["dims"]


def _post(url, payload):
    return httpx.post(f"{url}/v1/embeddings", json=payload, timeout=120.0)


def test_proxy_parity_with_direct_backend(proxy_pair):
    serve_url, backend_url, model_id, dims = proxy_pair
    payload = {"model": model_id, "input": "machine learning tutorial", "encoding_format": "float"}
    via_serve = _post(serve_url, payload)
    via_backend = _post(backend_url, payload)
    assert via_serve.status_code == 200, via_serve.text
    assert via_backend.status_code == 200, via_backend.text
    sv = via_serve.json()["data"][0]["embedding"]
    bv = via_backend.json()["data"][0]["embedding"]
    assert len(sv) == dims
    assert sv == pytest.approx(bv, abs=1e-6)  # thin proxy -> identical bytes


def test_proxy_base64_default(proxy_pair):
    serve_url, _, model_id, _ = proxy_pair
    r = _post(serve_url, {"model": model_id, "input": "hello world"})
    assert r.status_code == 200, r.text
    emb = r.json()["data"][0]["embedding"]
    assert isinstance(emb, str)  # base64 default survives the proxy


def test_proxy_batch_order(proxy_pair):
    serve_url, _, model_id, _ = proxy_pair
    r = _post(serve_url, {"model": model_id, "encoding_format": "float",
                          "input": ["alpha", "beta", "gamma"]})
    assert r.status_code == 200, r.text
    assert [d["index"] for d in r.json()["data"]] == [0, 1, 2]


def test_proxy_openai_sdk_one_base_url(proxy_pair):
    # The D2 promise: a RAG client points at serve for BOTH chat and embeddings.
    openai = pytest.importorskip("openai")
    serve_url, _, model_id, dims = proxy_pair
    client = openai.OpenAI(base_url=f"{serve_url}/v1", api_key="-")
    out = client.embeddings.create(model=model_id, input="how does the proxy forward bytes?")
    assert len(out.data) == 1
    assert len(out.data[0].embedding) == dims


def test_proxy_backend_down_returns_502():
    # Resilient (no startup probe): serve points at a closed port -> per-request 502 gateway error.
    with LocalServer(model=None, port=8768,
                     extra_env={"MLXK2_EMBED_BACKEND": "http://127.0.0.1:9"}) as serve_url:
        r = httpx.post(f"{serve_url}/v1/embeddings",
                       json={"model": "x", "input": "hi"}, timeout=15.0)
    assert r.status_code == 502, r.text
    assert r.json()["error"]["type"] == "bad_gateway"


def test_proxy_unconfigured_returns_501():
    # serve started WITHOUT --embed-backend: /v1/embeddings is not enabled here.
    with LocalServer(model=None, port=8769) as serve_url:
        r = httpx.post(f"{serve_url}/v1/embeddings",
                       json={"model": "x", "input": "hi"}, timeout=10.0)
    assert r.status_code == 501, r.text
    assert r.json()["error"]["type"] == "not_implemented"
