"""Thin proxy for ``POST /v1/embeddings`` (ADR-015 Slice D2).

The main ``serve`` process **forwards bytes only** — the embed model is never loaded into
serve's address space (ADR-015 §Server Topology). The runner + handler live exclusively in the
``embed-serve`` backend (``core/embed_server_base.py`` + ``handlers/embeddings.py``); this module
just relays the raw request body to that backend and returns its response verbatim, so the
client sees one OpenAI surface on ``serve``.

Transport-agnostic: receives an already-constructed :class:`httpx.AsyncClient` (created once in
serve's lifespan for connection pooling), which makes the forwarder unit-testable via
:class:`httpx.MockTransport` without a live backend. Transport failures map to ADR-004 gateway
errors (502 unreachable / 504 read-timeout); backend ``4xx/5xx`` envelopes pass through unchanged.
"""

from __future__ import annotations

import httpx
from fastapi import HTTPException, Response

# Single source of truth for proxy timeouts (the live test imports these).
#   connect=3s  -> fail fast when the backend is down (prompt 502, not a hang)
#   read=120s   -> matches the D1 live-test budget for large batches
#   write=30s   -> bound large request-body uploads
#   pool=5s     -> bound waiting for a free pooled connection
PROXY_TIMEOUT = httpx.Timeout(connect=3.0, read=120.0, write=30.0, pool=5.0)


async def proxy_embeddings(
    *,
    client: httpx.AsyncClient,
    backend_url: str,
    body: bytes,
    request_id: str | None = None,
    content_type: str = "application/json",
) -> Response:
    """Forward the raw body to ``{backend_url}/v1/embeddings`` and return the response verbatim.

    Args:
        client: shared ``httpx.AsyncClient`` (pooled; owned by serve's lifespan).
        backend_url: validated embed-serve base, e.g. ``http://127.0.0.1:8002`` (trailing
            slash tolerated).
        body: the raw request bytes (``await request.body()``) — forwarded byte-for-byte so
            extension fields (``input_type``, ``instruct``, …) and base64 vectors survive.
        request_id: serve's correlation id, forwarded as ``X-Request-ID`` for cross-process logs.
        content_type: forwarded ``Content-Type`` (defaults to JSON).

    Returns:
        A :class:`fastapi.Response` carrying the backend's status, body and content-type
        unchanged. Backend ``4xx/5xx`` errors (already ADR-004 envelopes) pass through as-is.

    Raises:
        HTTPException: 502 if the backend is unreachable / connection fails / connect-times-out;
            504 if the backend read-times-out.
    """
    # accept-encoding: identity keeps the proxy byte-for-byte — without it httpx would negotiate
    # gzip/br and transparently decompress, so resp.content would no longer be the backend's bytes.
    headers = {"content-type": content_type, "accept-encoding": "identity"}
    if request_id:
        headers["x-request-id"] = request_id
    target = backend_url.rstrip("/") + "/v1/embeddings"
    try:
        # Timeout passed explicitly so it holds even if the shared client was built without one.
        resp = await client.post(target, content=body, headers=headers, timeout=PROXY_TIMEOUT)
    except httpx.ConnectTimeout as e:  # could not establish the connection in time -> unreachable
        raise HTTPException(status_code=502, detail=f"Embed backend connect timeout: {e}")
    except httpx.ConnectError as e:  # connection refused / DNS failure / no route
        raise HTTPException(status_code=502, detail=f"Embed backend unreachable: {e}")
    except httpx.TimeoutException as e:  # read/write/pool timeout — connected but too slow
        raise HTTPException(status_code=504, detail=f"Embed backend timed out: {e}")
    except httpx.HTTPError as e:  # catch-all transport/protocol error
        raise HTTPException(status_code=502, detail=f"Embed backend error: {e}")
    return Response(
        content=resp.content,
        status_code=resp.status_code,
        media_type=resp.headers.get("content-type", "application/json"),
    )
