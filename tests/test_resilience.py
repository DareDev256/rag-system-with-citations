"""Resilience tests — error paths that crash production but pass in dev.

Targets file I/O failures, corrupted state recovery, metadata edge cases,
and singleton poisoning resistance. Every test here represents a real
production incident pattern (killed process → corrupted JSON, disk full
→ partial write, concurrent startup → race condition).
"""
from unittest.mock import patch, MagicMock, mock_open
import json

import numpy as np
import pytest

from src.retrieval.vector_store import VectorStore


# ─── VectorStore.load_index() corruption paths ─────────────────────

class TestLoadIndexCorruption:
    """Killed processes leave behind truncated JSON and partial FAISS files."""

    @patch("builtins.open", mock_open(read_data="{truncated"))
    @patch("src.retrieval.vector_store.json.load", side_effect=json.JSONDecodeError("x", "", 0))
    @patch("src.retrieval.vector_store.faiss.read_index")
    @patch("src.retrieval.vector_store.os.path.exists", return_value=True)
    def test_corrupted_json_metadata_propagates(self, _exists, _read, _json):
        """Corrupted metadata.json must raise — silent fallback hides data loss."""
        _read.return_value = MagicMock()
        vs = VectorStore()
        with pytest.raises(json.JSONDecodeError):
            vs.load_index()

    @patch("src.retrieval.vector_store.faiss.read_index", side_effect=RuntimeError("file truncated"))
    @patch("src.retrieval.vector_store.os.path.exists", return_value=True)
    def test_corrupted_faiss_index_propagates(self, _exists, _read):
        """Corrupted FAISS binary must raise — not silently start with None index."""
        vs = VectorStore()
        with pytest.raises(RuntimeError, match="file truncated"):
            vs.load_index()

    @patch("builtins.open", side_effect=PermissionError("permission denied"))
    @patch("src.retrieval.vector_store.faiss.read_index")
    @patch("src.retrieval.vector_store.os.path.exists", return_value=True)
    def test_metadata_permission_denied_propagates(self, _exists, _read, _open):
        """Permission errors on metadata must propagate — not silently drop docs."""
        _read.return_value = MagicMock()
        vs = VectorStore()
        with pytest.raises(PermissionError):
            vs.load_index()

    @patch("src.retrieval.vector_store.faiss.read_index", side_effect=OSError("disk I/O error"))
    @patch("src.retrieval.vector_store.os.path.exists", return_value=True)
    def test_faiss_oserror_propagates(self, _exists, _read):
        """OS-level I/O errors on FAISS file must not be swallowed."""
        vs = VectorStore()
        with pytest.raises(OSError, match="disk I/O error"):
            vs.load_index()

    @patch("builtins.open", mock_open(read_data="null"))
    @patch("src.retrieval.vector_store.json.load", return_value=None)
    @patch("src.retrieval.vector_store.faiss.read_index")
    @patch("src.retrieval.vector_store.os.path.exists", return_value=True)
    def test_metadata_null_json_loaded_as_none(self, _exists, _read, _json):
        """metadata.json containing `null` → metadata becomes None (caller beware)."""
        _read.return_value = MagicMock()
        vs = VectorStore()
        vs.load_index()
        assert vs.metadata is None  # Documents this footgun


# ─── VectorStore.save_index() failure paths ─────────────────────────

class TestSaveIndexFailures:
    """Disk full and permission errors during save must not corrupt state."""

    @patch("src.retrieval.vector_store.faiss.write_index", side_effect=OSError("No space left"))
    def test_faiss_write_oserror_propagates(self, _write):
        """Disk-full on FAISS write must raise — not leave orphaned metadata."""
        vs = VectorStore()
        vs.index = MagicMock()
        with pytest.raises(OSError, match="No space left"):
            vs.save_index()

    @patch("src.retrieval.vector_store.faiss.write_index")
    def test_metadata_write_permission_error_propagates(self, _write):
        """Permission denied on metadata write must raise after FAISS write succeeds."""
        vs = VectorStore(metadata_path="/root/no_access.json")
        vs.index = MagicMock()
        vs.metadata = [{"doc_id": "d1"}]
        with patch("builtins.open", side_effect=PermissionError("denied")):
            with pytest.raises(PermissionError):
                vs.save_index()

    @patch("src.retrieval.vector_store.faiss.write_index")
    def test_metadata_with_non_serializable_raises(self, _write):
        """Non-JSON-serializable metadata must fail fast on save."""
        vs = VectorStore()
        vs.index = MagicMock()
        vs.metadata = [{"doc_id": "d1", "bad": {1, 2, 3}}]  # set is not JSON serializable
        with patch("builtins.open", mock_open()):
            with pytest.raises(TypeError):
                vs.save_index()


# ─── VectorStore.search() metadata edge cases ──────────────────────

class TestSearchMetadataEdgeCases:
    """Metadata drift from index — partial deletes, missing fields."""

    @patch("src.retrieval.vector_store.faiss.normalize_L2")
    def test_metadata_missing_all_keys_returns_nones(self, _norm):
        """Metadata entry with no doc_id/text/source → None fields in result."""
        vs = VectorStore()
        vs.metadata = [{}]  # empty dict — all .get() return None
        fake_index = MagicMock(d=2)
        fake_index.search.return_value = (np.array([[0.9]]), np.array([[0]]))
        vs.index = fake_index
        results = vs.search([0.1, 0.2], k=1)
        assert len(results) == 1
        assert results[0]["doc_id"] is None
        assert results[0]["snippet"] is None
        assert results[0]["source"] is None

    @patch("src.retrieval.vector_store.faiss.normalize_L2")
    def test_k_larger_than_index_size(self, _norm):
        """k > index size → FAISS pads with -1 indices, all filtered out."""
        vs = VectorStore()
        vs.metadata = [{"doc_id": "only", "text": "one", "source": "a.txt"}]
        fake_index = MagicMock(d=1)
        fake_index.search.return_value = (
            np.array([[0.9, 0.0, 0.0]]), np.array([[0, -1, -1]])
        )
        vs.index = fake_index
        results = vs.search([0.5], k=3)
        assert len(results) == 1

    @patch("src.retrieval.vector_store.faiss.normalize_L2")
    def test_zero_score_result_still_returned(self, _norm):
        """Score=0.0 is valid (orthogonal vectors) — must not be filtered."""
        vs = VectorStore()
        vs.metadata = [{"doc_id": "d1", "text": "t", "source": "s"}]
        fake_index = MagicMock(d=1)
        fake_index.search.return_value = (np.array([[0.0]]), np.array([[0]]))
        vs.index = fake_index
        results = vs.search([0.1], k=1)
        assert len(results) == 1
        assert results[0]["score"] == 0.0

    @patch("src.retrieval.vector_store.faiss.normalize_L2")
    def test_negative_score_from_faiss(self, _norm):
        """FAISS can return negative scores with some index types."""
        vs = VectorStore()
        vs.metadata = [{"doc_id": "d1", "text": "t", "source": "s"}]
        fake_index = MagicMock(d=1)
        fake_index.search.return_value = (np.array([[-0.5]]), np.array([[0]]))
        vs.index = fake_index
        results = vs.search([0.1], k=1)
        assert results[0]["score"] == pytest.approx(-0.5)


# ─── VectorStore.add_documents() edge cases ─────────────────────────

class TestAddDocumentsEdgeCases:

    @patch("src.retrieval.vector_store.faiss.normalize_L2")
    @patch("src.retrieval.vector_store.faiss.IndexFlatIP")
    def test_metadata_count_tracks_embeddings(self, mock_ip, _norm):
        """Metadata length must always equal index size after sequential adds."""
        mock_ip.return_value = MagicMock(d=2)
        vs = VectorStore()
        batch1 = [[0.1, 0.2]] * 5
        meta1 = [{"doc_id": f"d{i}"} for i in range(5)]
        vs.add_documents(batch1, meta1)
        assert len(vs.metadata) == 5

        batch2 = [[0.3, 0.4]] * 3
        meta2 = [{"doc_id": f"e{i}"} for i in range(3)]
        vs.add_documents(batch2, meta2)
        assert len(vs.metadata) == 8

    @patch("src.retrieval.vector_store.faiss.normalize_L2")
    @patch("src.retrieval.vector_store.faiss.IndexFlatIP")
    def test_single_embedding_creates_index(self, mock_ip, _norm):
        """Single-document add must auto-create index."""
        mock_ip.return_value = MagicMock(d=3)
        vs = VectorStore()
        assert vs.index is None
        vs.add_documents([[1.0, 2.0, 3.0]], [{"doc_id": "solo"}])
        assert vs.index is not None


# ─── Search engine singleton recovery ───────────────────────────────

class TestSearchEngineSingleton:
    """get_search_engine() singleton must not cache a failed VectorStore."""

    @patch("src.retrieval.search._embedder", None)
    @patch("src.retrieval.search._vector_store", None)
    @patch("src.retrieval.search.create_default_store")
    @patch("src.retrieval.search.get_embedder")
    def test_successful_init_caches(self, mock_emb, mock_factory):
        """Happy path: both singletons assigned after successful load."""
        from src.retrieval import search
        mock_store = MagicMock()
        mock_factory.return_value = mock_store
        mock_emb.return_value = MagicMock()

        store, emb = search.get_search_engine()
        mock_store.load_index.assert_called_once()
        assert store is mock_store

        # Reset module-level singletons for test isolation
        search._vector_store = None
        search._embedder = None

    @patch("src.retrieval.search._embedder", None)
    @patch("src.retrieval.search._vector_store", None)
    @patch("src.retrieval.search.create_default_store")
    def test_failed_load_does_not_poison_singleton(self, mock_factory):
        """If load_index() raises, _vector_store must stay None for retry."""
        from src.retrieval import search
        mock_store = MagicMock()
        mock_store.load_index.side_effect = RuntimeError("corrupted")
        mock_factory.return_value = mock_store

        with pytest.raises(RuntimeError, match="corrupted"):
            search.get_search_engine()

        # Singleton must NOT be poisoned
        assert search._vector_store is None

        # Reset for isolation
        search._vector_store = None
        search._embedder = None
