"""Route-level tests for serve's POST /v1/embeddings thin proxy (ADR-015 Slice D2).

FastAPI TestClient against the real serve app. The backend is faked via an injected
``httpx.AsyncClient(MockTransport)`` patched onto the module globals — no live backend, no model.
Covers the route-only branches (501 when unconfigured) and the route->handler wiring
(body + content-type forwarded, backend response returned).
"""

from unittest.mock import patch

import httpx
from fastapi.testclient import TestClient

import mlxk2.core.server_base as sb
from mlxk2.core.server_base import app


def test_unconfigured_returns_501():
    # No --embed-backend configured -> _embed_backend is None -> 501 not_implemented envelope.
    # (No `with`: lifespan is not run, so the globals stay at their None defaults.)
    client = TestClient(app)
    r = client.post("/v1/embeddings", json={"model": "m", "input": "hi"})
    assert r.status_code == 501
    body = r.json()
    assert body["status"] == "error"
    assert body["error"]["type"] == "not_implemented"


def test_configured_proxies_to_backend():
    captured = {}

    def handler(request):
        captured["url"] = str(request.url)
        captured["body"] = request.content
        captured["ct"] = request.headers.get("content-type")
        return httpx.Response(200, content=b'{"object":"list","data":[1]}',
                              headers={"content-type": "application/json"})

    fake_client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    with patch.object(sb, "_embed_backend", "http://be:8002"), \
         patch.object(sb, "_embed_proxy_client", fake_client):
        client = TestClient(app)
        r = client.post("/v1/embeddings", content=b'{"model":"m","input":"hi"}',
                        headers={"content-type": "application/json"})

    assert r.status_code == 200
    assert r.json() == {"object": "list", "data": [1]}
    assert captured["url"] == "http://be:8002/v1/embeddings"
    assert captured["body"] == b'{"model":"m","input":"hi"}'
    assert captured["ct"] == "application/json"


def test_configured_backend_error_passes_through():
    def handler(request):
        return httpx.Response(400, content=b'{"status":"error","error":{"type":"validation_error"}}',
                              headers={"content-type": "application/json"})

    fake_client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    with patch.object(sb, "_embed_backend", "http://be:8002"), \
         patch.object(sb, "_embed_proxy_client", fake_client):
        client = TestClient(app)
        r = client.post("/v1/embeddings", json={"model": "m", "input": "x"})

    assert r.status_code == 400  # backend's status preserved
    assert r.json()["error"]["type"] == "validation_error"
