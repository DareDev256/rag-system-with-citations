import faiss
import numpy as np
import pickle
import os
from typing import List, Dict, Any

class VectorStore:
    def __init__(self, index_path="faiss_index.bin", metadata_path="metadata.pkl"):
        self.index_path = index_path
        self.metadata_path = metadata_path
        self.index = None
        self.metadata = []  # List of dicts corresponding to index IDs
    
    def load_index(self):
        if os.path.exists(self.index_path) and os.path.exists(self.metadata_path):
            print(f"Loading index from {self.index_path}")
            self.index = faiss.read_index(self.index_path)
            with open(self.metadata_path, "rb") as f:
                self.metadata = pickle.load(f)
        else:
            print("No existing index found. Starting fresh.")
            self.index = None
            self.metadata = []

    def create_index(self, dimension: int):
        # L2 Distance (Euclidean). For inner product (cosine), use faiss.IndexFlatIP
        # and normalize vectors beforehand. We'll stick to L2 for simplicity or 
        # assume normalized if we want cosine similarity.
        # Sentence-transformers usually work well with Cosine Similarity.
        # Let's use Inner Product (IP) and ensuring normalization in embedder or ingest.
        
        self.index = faiss.IndexFlatIP(dimension)
        self.metadata = []

    def add_documents(self, embeddings: List[List[float]], docs_metadata: List[Dict[str, Any]]):
        if not embeddings:
            return
        
        dim = len(embeddings[0])
        if self.index is None:
            self.create_index(dim)
        
        vectors = np.array(embeddings).astype('float32')
        # Normalize for Cosine Similarity (IndexFlatIP)
        faiss.normalize_L2(vectors)
        
        self.index.add(vectors)
        self.metadata.extend(docs_metadata)
        
    def save_index(self):
        if self.index:
            faiss.write_index(self.index, self.index_path)
            with open(self.metadata_path, "wb") as f:
                pickle.dump(self.metadata, f)
            print("Index saved.")

    def search(self, query_vector: List[float], k: int = 3):
        if not self.index:
            return []
        
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
