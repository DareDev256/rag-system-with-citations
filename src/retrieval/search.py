from src.retrieval.embed import get_embedder
from src.retrieval.vector_store import VectorStore
import os
import threading

# Initialize components globally or per request?
# For a simple app, global initialization is fine to keep index in memory.

_vector_store = None
_embedder = None
_search_engine_lock = threading.Lock()

def get_search_engine():
    global _vector_store, _embedder
    if _vector_store is not None and _embedder is not None:
        return _vector_store, _embedder
    with _search_engine_lock:
        if _vector_store is None:
            # Default paths — exist_ok avoids TOCTOU race between exists() and makedirs()
            base_path = "data_store"
            os.makedirs(base_path, exist_ok=True)

            index_path = os.path.join(base_path, "faiss.index")
            meta_path = os.path.join(base_path, "meta.json")

            store = VectorStore(index_path=index_path, metadata_path=meta_path)
            store.load_index()
            # Only assign singleton AFTER successful load — prevents poisoning on failure
            _vector_store = store

        if _embedder is None:
            _embedder = get_embedder()

    return _vector_store, _embedder

def perform_search(query: str, k: int = 3):
    store, embedder = get_search_engine()
    query_emb = embedder.encode([query])[0]
    return store.search(query_emb, k=k)
