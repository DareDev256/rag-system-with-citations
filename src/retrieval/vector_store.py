"""FAISS index wrapper with JSON metadata storage.

Provides a clean Python interface over FAISS IndexFlatIP (inner product /
cosine similarity on L2-normalized vectors). Metadata is stored as JSON
instead of pickle to eliminate deserialization attacks (CWE-502).

Typical lifecycle:
    store = VectorStore()
    store.load_index()              # Load existing or start fresh
    store.add_documents(embeds, meta)  # Add vectors + metadata
    store.save_index()              # Persist to disk
    results = store.search(vec, k=5)   # Retrieve top-k similar docs
"""

import faiss
import json
import logging
import os

import numpy as np
from typing import List, Dict, Any

logger = logging.getLogger(__name__)

# ─── Default storage layout ─────────────────────────────────────────
# Centralizes path construction so search.py and ingest.py don't
# independently hardcode the same directory/filenames.
DEFAULT_STORE_DIR = "data_store"
DEFAULT_INDEX_FILE = "faiss.index"
DEFAULT_META_FILE = "meta.json"


def default_store_paths(base_dir: str = DEFAULT_STORE_DIR) -> tuple:
    """Return (index_path, meta_path) for the given base directory.

    Creates the directory if it doesn't exist (exist_ok avoids TOCTOU).
    """
    os.makedirs(base_dir, exist_ok=True)
    return (
        os.path.join(base_dir, DEFAULT_INDEX_FILE),
        os.path.join(base_dir, DEFAULT_META_FILE),
    )


def create_default_store(base_dir: str = DEFAULT_STORE_DIR) -> "VectorStore":
    """Factory that creates a VectorStore with standard paths.

    Single source of truth for the data_store layout — callers don't
    need to know the internal file naming convention.
    """
    index_path, meta_path = default_store_paths(base_dir)
    return VectorStore(index_path=index_path, metadata_path=meta_path)


class VectorStore:
    """FAISS-backed vector store with parallel JSON metadata.

    Each vector in the FAISS index has a corresponding metadata dict
    (doc_id, text, source) stored at the same positional index in
    self.metadata. Vectors are L2-normalized before insertion for
    cosine similarity via IndexFlatIP.

    Args:
        index_path: Filesystem path for the FAISS binary index.
        metadata_path: Filesystem path for the JSON metadata file.
    """

    def __init__(self, index_path="faiss_index.bin", metadata_path="metadata.json"):
        self.index_path = index_path
        self.metadata_path = metadata_path
        self.index = None
        self.metadata = []  # List of dicts corresponding to index IDs

    def load_index(self) -> None:
        """Load FAISS index and metadata from disk, or start with empty state."""
        if os.path.exists(self.index_path) and os.path.exists(self.metadata_path):
            logger.info("Loading index from %s", self.index_path)
            self.index = faiss.read_index(self.index_path)
            with open(self.metadata_path, "r", encoding="utf-8") as f:
                self.metadata = json.load(f)
        else:
            logger.info("No existing index found. Starting fresh.")
            self.index = None
            self.metadata = []

    def create_index(self, dimension: int) -> None:
        """Create a fresh FAISS IndexFlatIP for the given embedding dimension.

        Uses inner product (IP) on L2-normalized vectors, which is equivalent
        to cosine similarity. Vectors must be normalized before add().
        """
        # L2 Distance (Euclidean). For inner product (cosine), use faiss.IndexFlatIP
        # and normalize vectors beforehand. We'll stick to L2 for simplicity or 
        # assume normalized if we want cosine similarity.
        # Sentence-transformers usually work well with Cosine Similarity.
        # Let's use Inner Product (IP) and ensuring normalization in embedder or ingest.
        
        self.index = faiss.IndexFlatIP(dimension)
        self.metadata = []

    def add_documents(self, embeddings: List[List[float]], docs_metadata: List[Dict[str, Any]]) -> None:
        """Add document vectors and metadata to the index.

        Auto-creates the index on first call. Validates embedding dimensions
        match the existing index to prevent cryptic FAISS segfaults.

        Raises:
            ValueError: If embedding dimension doesn't match the index.
        """
        if not embeddings:
            return
        
        dim = len(embeddings[0])
        if self.index is None:
            self.create_index(dim)
        elif self.index.d != dim:
            raise ValueError(
                f"Embedding dimension {dim} does not match index dimension {self.index.d}"
            )

        vectors = np.array(embeddings).astype('float32')
        # Normalize for Cosine Similarity (IndexFlatIP)
        faiss.normalize_L2(vectors)
        
        self.index.add(vectors)
        self.metadata.extend(docs_metadata)
        
    def save_index(self) -> None:
        """Persist the FAISS index and metadata to disk."""
        if self.index:
            faiss.write_index(self.index, self.index_path)
            with open(self.metadata_path, "w", encoding="utf-8") as f:
                json.dump(self.metadata, f, ensure_ascii=False)
            logger.info("Index saved.")

    def search(self, query_vector: List[float], k: int = 3) -> List[Dict[str, Any]]:
        """Find the k most similar documents to the query vector.

        Args:
            query_vector: Dense embedding of the search query.
            k: Number of results to return.

        Returns:
            List of dicts with keys: doc_id, snippet, score, source.
            Empty list if no index is loaded.

        Raises:
            ValueError: If query dimension doesn't match the index.
        """
        if not self.index:
            return []

        if len(query_vector) != self.index.d:
            raise ValueError(
                f"Query dimension {len(query_vector)} does not match index dimension {self.index.d}"
            )

        vector = np.array([query_vector]).astype('float32')
        faiss.normalize_L2(vector)
        
        distances, indices = self.index.search(vector, k)
        
        results = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx != -1 and idx < len(self.metadata):
                doc = self.metadata[idx]
                results.append({
                    "doc_id": doc.get("doc_id"),
                    "snippet": doc.get("text"),
                    "score": float(dist),
                    "source": doc.get("source")
                })
        return results
