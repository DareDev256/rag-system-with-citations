"""Tests for FastAPI endpoints using TestClient.

Mocks all external deps (LLM, search) — no API keys or FAISS needed.
"""
from unittest.mock import patch, AsyncMock

import pytest
from fastapi.testclient import TestClient


@pytest.fixture(autouse=True)
def _skip_env_validation():
    with patch("src.api.main.os.getenv", return_value="fake-key"):
        yield


@pytest.fixture()
def client():
    from src.api.main import app
    return TestClient(app, raise_server_exceptions=False)


MOCK_SEARCH = [
    {"doc_id": "doc_001", "snippet": "RAG combines retrieval.", "score": 0.95, "source": "notes.txt"},
    {"doc_id": "doc_002", "snippet": "FAISS is fast.", "score": 0.88, "source": "guide.txt"},
]


class TestHealthEndpoint:
    def test_returns_ok(self, client):
        resp = client.get("/health")
        assert resp.status_code == 200
        assert resp.json() == {"status": "ok"}

    def test_post_not_allowed(self, client):
        assert client.post("/health").status_code == 405


class TestQueryEndpoint:
    @patch("src.api.main.perform_search", return_value=MOCK_SEARCH)
    @patch("src.api.main.synthesize_answer_async", new_callable=AsyncMock)
    @patch("src.api.main.classify_query_async", new_callable=AsyncMock)
    def test_successful_query(self, mock_classify, mock_synth, mock_search, client):
        mock_classify.return_value = "factual"
        mock_synth.return_value = {
            "answer": "RAG is great [doc_001].",
            "citations_used": [MOCK_SEARCH[0]],
            "confidence": 0.8,
        }
        resp = client.post("/query", json={"query": "What is RAG?"})
        data = resp.json()
        assert resp.status_code == 200
        assert data["category"] == "factual"
        assert data["confidence"] == 0.8
        assert len(data["citations"]) == 1
        assert data["citations"][0]["doc_id"] == "doc_001"
        assert data["latency_ms"] > 0

    @patch("src.api.main.perform_search", return_value=MOCK_SEARCH)
    @patch("src.api.main.synthesize_answer_async", new_callable=AsyncMock)
    @patch("src.api.main.classify_query_async", new_callable=AsyncMock)
    def test_custom_k_forwarded(self, mock_classify, mock_synth, mock_search, client):
        mock_classify.return_value = "factual"
        mock_synth.return_value = {"answer": "x", "citations_used": [], "confidence": 0.3}
        client.post("/query", json={"query": "test", "k": 10})
        mock_search.assert_called_once_with("test", k=10)

    @patch("src.api.main.perform_search", return_value=MOCK_SEARCH)
    @patch("src.api.main.synthesize_answer_async", new_callable=AsyncMock)
    @patch("src.api.main.classify_query_async", new_callable=AsyncMock)
    def test_default_k_is_five(self, mock_classify, mock_synth, mock_search, client):
        mock_classify.return_value = "factual"
        mock_synth.return_value = {"answer": "x", "citations_used": [], "confidence": 0.3}
        client.post("/query", json={"query": "test"})
        mock_search.assert_called_once_with("test", k=5)

    @patch("src.api.main.perform_search", return_value=[])
    @patch("src.api.main.synthesize_answer_async", new_callable=AsyncMock)
    @patch("src.api.main.classify_query_async", new_callable=AsyncMock)
    def test_empty_results(self, mock_classify, mock_synth, mock_search, client):
        mock_classify.return_value = "ambiguous"
        mock_synth.return_value = {"answer": "No info.", "citations_used": [], "confidence": 0.0}
        data = client.post("/query", json={"query": "unknown"}).json()
        assert data["confidence"] == 0.0
        assert data["citations"] == []

    def test_empty_query_rejected(self, client):
        assert client.post("/query", json={"query": ""}).status_code == 422

    def test_missing_query_rejected(self, client):
        assert client.post("/query", json={}).status_code == 422

    def test_query_too_long_rejected(self, client):
        assert client.post("/query", json={"query": "x" * 1001}).status_code == 422

    def test_k_out_of_range_rejected(self, client):
        assert client.post("/query", json={"query": "t", "k": 0}).status_code == 422
