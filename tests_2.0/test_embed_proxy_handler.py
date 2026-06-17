"""Unit tests for the /v1/embeddings thin-proxy handler (ADR-015 Slice D2).

``httpx.MockTransport`` stands in for the embed-serve backend — no live server, no model. These
validate verbatim forwarding (status/body/content-type), header + URL handling, and the
transport-error -> 502/504 mapping. The handler is async; tests drive it via ``asyncio.run`` so
no pytest-asyncio dependency is needed.
"""

import asyncio

import httpx
import pytest
from fastapi import HTTPException

from mlxk2.core.server.handlers.embed_proxy import proxy_embeddings, PROXY_TIMEOUT


def _client(handler):
    return httpx.AsyncClient(transport=httpx.MockTransport(handler))


def _run(handler, *, backend_url="http://be:8002", body=b"{}", request_id=None,
         content_type="application/json"):
    async def go():
        async with _client(handler) as c:
            return await proxy_embeddings(client=c, backend_url=backend_url, body=body,
                                          request_id=request_id, content_type=content_type)
    return asyncio.run(go())


def test_timeout_constant_values():
    # The live test imports these; pin the contract.
    assert PROXY_TIMEOUT.connect == 3.0
    assert PROXY_TIMEOUT.read == 120.0


def test_forwards_body_and_returns_verbatim():
    seen = {}

    def handler(request):
        seen["url"] = str(request.url)
        seen["method"] = request.method
        seen["body"] = request.content
        seen["ct"] = request.headers.get("content-type")
        seen["xrid"] = request.headers.get("x-request-id")
        return httpx.Response(200, content=b'{"object":"list","data":[]}',
                              headers={"content-type": "application/json"})

    resp = _run(handler, body=b'{"model":"m","input":"hi"}', request_id="req-123")
    assert resp.status_code == 200
    assert resp.body == b'{"object":"list","data":[]}'
    assert resp.media_type == "application/json"
    assert seen["url"] == "http://be:8002/v1/embeddings"
    assert seen["method"] == "POST"
    assert seen["body"] == b'{"model":"m","input":"hi"}'
    assert seen["ct"] == "application/json"
    assert seen["xrid"] == "req-123"


def test_no_request_id_omits_header():
    seen = {}

    def handler(request):
        seen["xrid"] = request.headers.get("x-request-id")
        return httpx.Response(200, content=b"{}")

    _run(handler, request_id=None)
    assert seen["xrid"] is None


def test_extension_fields_pass_through_byte_for_byte():
    raw = b'{"model":"m","input":"hi","input_type":"query","instruct":"Find","future":42}'
    seen = {}

    def handler(request):
        seen["body"] = request.content
        return httpx.Response(200, content=b"{}", headers={"content-type": "application/json"})

    _run(handler, body=raw)
    assert seen["body"] == raw  # forwarded untouched — the THIN-proxy contract


def test_trailing_slash_in_backend_url_normalized():
    seen = {}

    def handler(request):
        seen["url"] = str(request.url)
        return httpx.Response(200, content=b"{}")

    _run(handler, backend_url="http://be:8002/")
    assert seen["url"] == "http://be:8002/v1/embeddings"


def test_backend_4xx_passed_through_verbatim():
    body = b'{"status":"error","error":{"type":"validation_error"}}'

    def handler(request):
        return httpx.Response(400, content=body, headers={"content-type": "application/json"})

    resp = _run(handler)
    assert resp.status_code == 400
    assert resp.body == body  # not re-wrapped by serve


def test_backend_5xx_passed_through_verbatim():
    def handler(request):
        return httpx.Response(503, content=b'{"status":"error"}',
                              headers={"content-type": "application/json"})

    resp = _run(handler)
    assert resp.status_code == 503


def test_connect_error_maps_to_502():
    def handler(request):
        raise httpx.ConnectError("connection refused")

    with pytest.raises(HTTPException) as ei:
        _run(handler)
    assert ei.value.status_code == 502


def test_connect_timeout_maps_to_502():
    def handler(request):
        raise httpx.ConnectTimeout("timed out connecting")

    with pytest.raises(HTTPException) as ei:
        _run(handler)
    assert ei.value.status_code == 502


def test_read_timeout_maps_to_504():
    def handler(request):
        raise httpx.ReadTimeout("timed out reading")

    with pytest.raises(HTTPException) as ei:
        _run(handler)
    assert ei.value.status_code == 504


def test_write_timeout_maps_to_504():
    # Large request body stalls mid-upload -> connected-but-too-slow -> 504, not 502.
    def handler(request):
        raise httpx.WriteTimeout("timed out writing")

    with pytest.raises(HTTPException) as ei:
        _run(handler)
    assert ei.value.status_code == 504


def test_pool_timeout_maps_to_504():
    # Pool timeout is a TimeoutException -> 504 (gateway timeout), not the 502 catch-all.
    def handler(request):
        raise httpx.PoolTimeout("no connection available")

    with pytest.raises(HTTPException) as ei:
        _run(handler)
    assert ei.value.status_code == 504


def test_non_timeout_transport_error_maps_to_502():
    # A protocol/transport error that is NOT a timeout falls to the httpx.HTTPError catch-all -> 502.
    def handler(request):
        raise httpx.RemoteProtocolError("server disconnected without sending a response")

    with pytest.raises(HTTPException) as ei:
        _run(handler)
    assert ei.value.status_code == 502
