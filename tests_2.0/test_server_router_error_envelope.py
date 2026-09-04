"""Router rejects carry the ADR-004 envelope (audit C2).

404 (no route matches) and 405 (wrong method) are raised by Starlette's router before
any endpoint runs, as the *base* HTTPException. FastAPI's subclass — which every one of
our own raises uses — is not in that class's MRO, so the two used to leave the server as
``{"detail": ...}``: no error type, no request_id, nothing to correlate with a log line.
"""

import pytest
from fastapi import FastAPI, HTTPException
from fastapi.testclient import TestClient

from mlxk2.core.server.error_handlers import register_error_handlers
from mlxk2.core.server_base import app


@pytest.fixture
def client():
    return TestClient(app, raise_server_exceptions=False)


def test_unmatched_path_is_not_found(client):
    body = client.get("/v1/v1/models").json()
    assert body["error"]["type"] == "not_found"
    assert body["error"]["retryable"] is False
    assert body["request_id"]


def test_wrong_method_is_method_not_allowed(client):
    response = client.get("/v1/chat/completions")
    assert response.status_code == 405
    assert response.json()["error"]["type"] == "method_not_allowed"
    # The Allow header is the only useful part of a 405 — the envelope must not drop it.
    assert response.headers["allow"] == "POST"


def test_our_own_404_keeps_its_own_type():
    """The router handler must not swallow the raises that already had an envelope."""
    other = FastAPI()
    register_error_handlers(other)

    @other.get("/probe")
    async def probe():
        raise HTTPException(status_code=404, detail="Model not found in cache: nope")

    body = TestClient(other, raise_server_exceptions=False).get("/probe").json()
    assert body["error"]["type"] == "model_not_found"
    assert body["error"]["message"] == "Model not found in cache: nope"
