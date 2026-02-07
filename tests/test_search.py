"""Tests for the retrieval orchestration layer.

Covers perform_search, get_search_engine, and Embedder singleton
by mocking FAISS and SentenceTransformer — no model downloads or
index files required.
"""
from unittest.mock import patch, MagicMock, PropertyMock

import numpy as np
import pytest

import src.retrieval.search as search_mod
from src.retrieval.search import perform_search, get_search_engine
from src.retrieval.embed import Embedder, get_embedder


# ── Fixtures ────────────────────────────────────────────────────

@pytest.fixture(autouse=True)
def reset_search_globals():
    """Reset module-level singletons between tests."""
    search_mod._vector_store = None
    search_mod._embedder = None
    yield
    search_mod._vector_store = None
    search_mod._embedder = None


@pytest.fixture(autouse=True)
def reset_embedder_singleton():
    """Reset Embedder singleton between tests."""
    Embedder._instance = None
    yield
    Embedder._instance = None


# ── Embedder Singleton ──────────────────────────────────────────

class TestEmbedderSingleton:
    @patch("src.retrieval.embed.SentenceTransformer")
    def test_returns_same_instance(self, mock_st):
        a = Embedder()
        b = Embedder()
        assert a is b
        # Model loaded only once despite two instantiations
        mock_st.assert_called_once_with("all-MiniLM-L6-v2")

    @patch("src.retrieval.embed.SentenceTransformer")
    def test_encode_delegates_to_model(self, mock_st):
        mock_model = MagicMock()
        fake_embeddings = np.array([[0.1, 0.2], [0.3, 0.4]])
        mock_model.encode.return_value = fake_embeddings
        mock_st.return_value = mock_model

        emb = Embedder()
        result = emb.encode(["hello", "world"])

        mock_model.encode.assert_called_once_with(
            ["hello", "world"], convert_to_numpy=True
        )
        assert result == [[0.1, 0.2], [0.3, 0.4]]

    @patch("src.retrieval.embed.SentenceTransformer")
    def test_encode_single_text(self, mock_st):
        mock_model = MagicMock()
        mock_model.encode.return_value = np.array([[0.5, 0.6, 0.7]])
        mock_st.return_value = mock_model

        result = Embedder().encode(["single query"])
        assert len(result) == 1
        assert len(result[0]) == 3

    @patch("src.retrieval.embed.SentenceTransformer")
    def test_get_embedder_returns_singleton(self, mock_st):
        a = get_embedder()
        b = get_embedder()
        assert a is b


# ── get_search_engine ───────────────────────────────────────────

class TestGetSearchEngine:
    @patch("src.retrieval.search.get_embedder")
    @patch("src.retrieval.search.VectorStore")
    @patch("src.retrieval.search.os.path.exists", return_value=True)
    def test_returns_store_and_embedder(self, mock_exists, mock_vs_cls, mock_get_emb):
        mock_store = MagicMock()
        mock_vs_cls.return_value = mock_store
        mock_emb = MagicMock()
        mock_get_emb.return_value = mock_emb

        store, embedder = get_search_engine()
        assert store is mock_store
        assert embedder is mock_emb
        mock_store.load_index.assert_called_once()

    @patch("src.retrieval.search.get_embedder")
    @patch("src.retrieval.search.VectorStore")
    @patch("src.retrieval.search.os.path.exists", return_value=True)
    def test_caches_across_calls(self, mock_exists, mock_vs_cls, mock_get_emb):
        mock_vs_cls.return_value = MagicMock()
        mock_get_emb.return_value = MagicMock()

        first = get_search_engine()
        second = get_search_engine()
        assert first[0] is second[0]
        assert first[1] is second[1]
        # Constructor called only once
        mock_vs_cls.assert_called_once()

    @patch("src.retrieval.search.get_embedder")
    @patch("src.retrieval.search.VectorStore")
    @patch("src.retrieval.search.os.makedirs")
    @patch("src.retrieval.search.os.path.exists", return_value=False)
    def test_creates_data_dir_if_missing(self, mock_exists, mock_makedirs, mock_vs_cls, mock_get_emb):
        mock_vs_cls.return_value = MagicMock()
        mock_get_emb.return_value = MagicMock()

        get_search_engine()
        mock_makedirs.assert_called_once_with("data_store")

    @patch("src.retrieval.search.get_embedder")
    @patch("src.retrieval.search.VectorStore")
    @patch("src.retrieval.search.os.path.exists", return_value=True)
    def test_passes_correct_paths(self, mock_exists, mock_vs_cls, mock_get_emb):
        mock_vs_cls.return_value = MagicMock()
        mock_get_emb.return_value = MagicMock()

        get_search_engine()
        call_kwargs = mock_vs_cls.call_args[1]
        assert call_kwargs["index_path"] == "data_store/faiss.index"
        assert call_kwargs["metadata_path"] == "data_store/meta.pkl"


# ── perform_search ──────────────────────────────────────────────

class TestPerformSearch:
    @patch("src.retrieval.search.get_search_engine")
    def test_returns_search_results(self, mock_engine):
        mock_store = MagicMock()
        mock_embedder = MagicMock()
        mock_embedder.encode.return_value = [[0.1, 0.2, 0.3]]
        mock_store.search.return_value = [
            {"doc_id": "d1", "snippet": "hello", "score": 0.95, "source": "a.txt"},
        ]
        mock_engine.return_value = (mock_store, mock_embedder)

        results = perform_search("test query")
        assert len(results) == 1
        assert results[0]["doc_id"] == "d1"
        assert results[0]["score"] == 0.95

    @patch("src.retrieval.search.get_search_engine")
    def test_encodes_query_and_uses_first_vector(self, mock_engine):
        mock_store = MagicMock()
        mock_embedder = MagicMock()
        mock_embedder.encode.return_value = [[0.1, 0.2], [0.9, 0.9]]
        mock_store.search.return_value = []
        mock_engine.return_value = (mock_store, mock_embedder)

        perform_search("my query")
        mock_embedder.encode.assert_called_once_with(["my query"])
        # Should pass the first (and only relevant) vector
        mock_store.search.assert_called_once_with([0.1, 0.2], k=3)

    @patch("src.retrieval.search.get_search_engine")
    def test_forwards_custom_k(self, mock_engine):
        mock_store = MagicMock()
        mock_embedder = MagicMock()
        mock_embedder.encode.return_value = [[0.5, 0.5]]
        mock_store.search.return_value = []
        mock_engine.return_value = (mock_store, mock_embedder)

        perform_search("query", k=10)
        mock_store.search.assert_called_once_with([0.5, 0.5], k=10)

    @patch("src.retrieval.search.get_search_engine")
    def test_empty_index_returns_empty(self, mock_engine):
        mock_store = MagicMock()
        mock_embedder = MagicMock()
        mock_embedder.encode.return_value = [[0.1]]
        mock_store.search.return_value = []
        mock_engine.return_value = (mock_store, mock_embedder)

        assert perform_search("anything") == []

    @patch("src.retrieval.search.get_search_engine")
    def test_multiple_results_preserved_order(self, mock_engine):
        mock_store = MagicMock()
        mock_embedder = MagicMock()
        mock_embedder.encode.return_value = [[0.1, 0.2]]
        mock_store.search.return_value = [
            {"doc_id": "d1", "snippet": "first", "score": 0.99, "source": "a.txt"},
            {"doc_id": "d2", "snippet": "second", "score": 0.85, "source": "b.txt"},
            {"doc_id": "d3", "snippet": "third", "score": 0.70, "source": "c.txt"},
        ]
        mock_engine.return_value = (mock_store, mock_embedder)

        results = perform_search("query", k=3)
        assert [r["doc_id"] for r in results] == ["d1", "d2", "d3"]
        assert results[0]["score"] > results[1]["score"] > results[2]["score"]

    @patch("src.retrieval.search.get_search_engine")
    def test_default_k_is_three(self, mock_engine):
        mock_store = MagicMock()
        mock_embedder = MagicMock()
        mock_embedder.encode.return_value = [[0.1]]
        mock_store.search.return_value = []
        mock_engine.return_value = (mock_store, mock_embedder)

        perform_search("q")
        mock_store.search.assert_called_once_with([0.1], k=3)
