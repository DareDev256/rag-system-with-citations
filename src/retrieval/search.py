from src.retrieval.embed import get_embedder
from src.retrieval.vector_store import VectorStore
import os

# Initialize components globally or per request?
# For a simple app, global initialization is fine to keep index in memory.

_vector_store = None
_embedder = None

def get_search_engine():
    global _vector_store, _embedder
    if _vector_store is None:
        # Default paths
        base_path = "data_store"
        if not os.path.exists(base_path):
            os.makedirs(base_path)
            
        index_path = os.path.join(base_path, "faiss.index")
        meta_path = os.path.join(base_path, "meta.pkl")
        
        _vector_store = VectorStore(index_path=index_path, metadata_path=meta_path)
        _vector_store.load_index()
        
    if _embedder is None:
        _embedder = get_embedder()
        
    return _vector_store, _embedder

def perform_search(query: str, k: int = 3):
    store, embedder = get_search_engine()
    query_emb = embedder.encode([query])[0]
    return store.search(query_emb, k=k)
