"""Tests for VectorStore by mocking FAISS — no index files needed."""
from unittest.mock import patch, MagicMock, mock_open

import numpy as np
import pytest

from src.retrieval.vector_store import VectorStore


class TestVectorStoreInit:
    def test_defaults(self):
        vs = VectorStore()
        assert vs.index_path == "faiss_index.bin"
        assert vs.index is None
        assert vs.metadata == []

    def test_custom_paths(self):
        vs = VectorStore(index_path="/tmp/idx", metadata_path="/tmp/meta")
        assert vs.index_path == "/tmp/idx"


class TestLoadIndex:
    @patch("src.retrieval.vector_store.os.path.exists", return_value=False)
    def test_missing_files_starts_fresh(self, mock_exists):
        vs = VectorStore()
        vs.load_index()
        assert vs.index is None

    @patch("builtins.open", mock_open(read_data=b""))
    @patch("src.retrieval.vector_store.pickle.load", return_value=[{"doc_id": "d1"}])
    @patch("src.retrieval.vector_store.faiss.read_index")
    @patch("src.retrieval.vector_store.os.path.exists", return_value=True)
    def test_loads_existing_index(self, mock_exists, mock_read, mock_pickle):
        mock_read.return_value = MagicMock()
        vs = VectorStore()
        vs.load_index()
        assert vs.index is not None
        assert vs.metadata == [{"doc_id": "d1"}]


class TestAddDocuments:
    def test_empty_embeddings_noop(self):
        vs = VectorStore()
        vs.add_documents([], [])
        assert vs.index is None

    @patch("src.retrieval.vector_store.faiss.normalize_L2")
    @patch("src.retrieval.vector_store.faiss.IndexFlatIP")
    def test_adds_and_appends(self, mock_ip, mock_norm):
        mock_ip.return_value = MagicMock()
        vs = VectorStore()
        vs.add_documents([[0.1, 0.2]], [{"doc_id": "a"}])
        vs.add_documents([[0.3, 0.4]], [{"doc_id": "b"}])
        assert vs.metadata == [{"doc_id": "a"}, {"doc_id": "b"}]


class TestSearch:
    def test_no_index_returns_empty(self):
        assert VectorStore().search([0.1, 0.2]) == []

    @patch("src.retrieval.vector_store.faiss.normalize_L2")
    def test_returns_matched_docs(self, mock_norm):
        vs = VectorStore()
        vs.metadata = [
            {"doc_id": "d1", "text": "first", "source": "a.txt"},
            {"doc_id": "d2", "text": "second", "source": "b.txt"},
        ]
        fake_index = MagicMock()
        fake_index.search.return_value = (
            np.array([[0.95, 0.80]]), np.array([[0, 1]]),
        )
        vs.index = fake_index
        results = vs.search([0.1, 0.2], k=2)
        assert len(results) == 2
        assert results[0]["doc_id"] == "d1"
        assert results[0]["score"] == pytest.approx(0.95)

    @patch("src.retrieval.vector_store.faiss.normalize_L2")
    def test_skips_invalid_and_out_of_range_indices(self, mock_norm):
        vs = VectorStore()
        vs.metadata = [{"doc_id": "d1", "text": "only", "source": "a.txt"}]
        fake_index = MagicMock()
        fake_index.search.return_value = (
            np.array([[0.9, 0.5, 0.3]]), np.array([[0, -1, 99]]),
        )
        vs.index = fake_index
        results = vs.search([0.1], k=3)
        assert len(results) == 1
        assert results[0]["doc_id"] == "d1"
