"""Tests for API key authentication (CWE-862, CWE-208).

Validates that when API_KEYS is configured, only authenticated
requests reach the /query endpoint, and public paths remain open.
"""
from unittest.mock import patch, AsyncMock

import pytest
from fastapi.testclient import TestClient

VALID_KEY = "test-key-abc123"
SECOND_KEY = "test-key-def456"

MOCK_SEARCH = [
    {"doc_id": "doc_001", "snippet": "RAG combines retrieval.", "score": 0.95, "source": "notes.txt"},
]
MOCK_SYNTH = {"answer": "RAG [doc_001].", "citations_used": [MOCK_SEARCH[0]], "confidence": 0.8}


@pytest.fixture(autouse=True)
def _skip_env_validation():
    with patch("src.api.main.os.getenv", return_value="fake-key"):
        yield


@pytest.fixture()
def _mock_llm():
    """Patch LLM calls so query endpoint doesn't hit OpenAI."""
    with patch("src.api.main.perform_search", return_value=MOCK_SEARCH), \
         patch("src.api.main.synthesize_answer_async", new_callable=AsyncMock, return_value=MOCK_SYNTH), \
         patch("src.api.main.classify_query_async", new_callable=AsyncMock, return_value="factual"):
        yield


def _make_client_with_keys(keys_csv: str):
    """Set API_KEYS and return a TestClient, restoring original keys on teardown."""
    import src.api.main as main_mod
    import src.api.middleware as mw_mod
    original_keys = mw_mod._API_KEYS.copy()
    mw_mod._API_KEYS.clear()
    if keys_csv:
        mw_mod._API_KEYS.update(k.strip() for k in keys_csv.split(",") if k.strip())
    client = TestClient(main_mod.app, raise_server_exceptions=False)
    yield client
    mw_mod._API_KEYS.clear()
    mw_mod._API_KEYS.update(original_keys)


@pytest.fixture()
def authed_client():
    yield from _make_client_with_keys(f"{VALID_KEY},{SECOND_KEY}")


@pytest.fixture()
def open_client():
    yield from _make_client_with_keys("")


class TestAuthEnabled:
    """When API_KEYS is set, POST /query requires a valid key."""

    @pytest.mark.usefixtures("_mock_llm")
    def test_bearer_token_accepted(self, authed_client):
        resp = authed_client.post(
            "/query",
            json={"query": "What is RAG?"},
            headers={"Authorization": f"Bearer {VALID_KEY}"},
        )
        assert resp.status_code == 200

    @pytest.mark.usefixtures("_mock_llm")
    def test_x_api_key_accepted(self, authed_client):
        resp = authed_client.post(
            "/query",
            json={"query": "What is RAG?"},
            headers={"X-API-Key": SECOND_KEY},
        )
        assert resp.status_code == 200

    def test_missing_key_rejected(self, authed_client):
        resp = authed_client.post("/query", json={"query": "What is RAG?"})
        assert resp.status_code == 401
        assert "API key" in resp.json()["detail"]

    def test_wrong_key_rejected(self, authed_client):
        resp = authed_client.post(
            "/query",
            json={"query": "What is RAG?"},
            headers={"Authorization": "Bearer wrong-key-999"},
        )
        assert resp.status_code == 401

    def test_empty_bearer_rejected(self, authed_client):
        resp = authed_client.post(
            "/query",
            json={"query": "What is RAG?"},
            headers={"Authorization": "Bearer "},
        )
        assert resp.status_code == 401

    def test_health_endpoint_public(self, authed_client):
        resp = authed_client.get("/health")
        assert resp.status_code == 200

    def test_get_requests_skip_auth(self, authed_client):
        resp = authed_client.get("/health")
        assert resp.status_code == 200


class TestAuthDisabled:
    """When API_KEYS is empty, all requests pass through."""

    @pytest.mark.usefixtures("_mock_llm")
    def test_query_without_key_succeeds(self, open_client):
        resp = open_client.post("/query", json={"query": "What is RAG?"})
        assert resp.status_code == 200


class TestTimingSafety:
    """Verify constant-time comparison is used (structural check)."""

    def test_hmac_import_present(self):
        import src.api.middleware as mod
        import inspect
        source = inspect.getsource(mod.authenticate)
        assert "compare_digest" in source

    def test_authenticate_rejects_partial_match(self, authed_client):
        """A key that shares a prefix with a valid key must still be rejected."""
        resp = authed_client.post(
            "/query",
            json={"query": "test"},
            headers={"Authorization": f"Bearer {VALID_KEY[:-1]}"},
        )
        assert resp.status_code == 401
