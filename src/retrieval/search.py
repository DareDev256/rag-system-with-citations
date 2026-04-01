"""Search orchestration layer — connects the embedder to the FAISS index.

Manages singleton initialization of the VectorStore and Embedder with
thread-safe lazy loading. The singleton is only assigned after a successful
load, preventing poisoned state from propagating on transient failures.
"""

from src.retrieval.embed import get_embedder
from src.retrieval.vector_store import create_default_store
import threading

_vector_store = None
_embedder = None
_search_engine_lock = threading.Lock()


def get_search_engine():
    """Return the (VectorStore, Embedder) singleton pair, initializing on first call.

    Thread-safe via double-checked locking. On load failure (corrupted index,
    permission denied), the singleton stays None so the next call retries
    instead of serving a broken instance.
    """
    global _vector_store, _embedder
    if _vector_store is not None and _embedder is not None:
        return _vector_store, _embedder
    with _search_engine_lock:
        if _vector_store is None:
            store = create_default_store()
            store.load_index()
            # Only assign singleton AFTER successful load — prevents poisoning on failure
            _vector_store = store

        if _embedder is None:
            _embedder = get_embedder()

    return _vector_store, _embedder

def perform_search(query: str, k: int = 3):
    """Encode a query and retrieve the top-k most similar documents.

    Args:
        query: Natural-language search query.
        k: Number of results to return (default 3, max bounded by FAISS index size).

    Returns:
        List of dicts with keys: doc_id, snippet, score, source.
    """
    store, embedder = get_search_engine()
    query_emb = embedder.encode([query])[0]
    return store.search(query_emb, k=k)
