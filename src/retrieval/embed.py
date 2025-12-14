from sentence_transformers import SentenceTransformer
from typing import List
import os

class Embedder:
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(Embedder, cls).__new__(cls)
            # Load model once. Using the lightweight all-MiniLM-L6-v2
            # Set cache folder if needed, or rely on default
            print("Loading embedding model...")
            cls._instance.model = SentenceTransformer('all-MiniLM-L6-v2')
            print("Embedding model loaded.")
        return cls._instance

    def encode(self, texts: List[str]) -> List[List[float]]:
        # Returns a list of vectors (list of floats)
        embeddings = self.model.encode(texts, convert_to_numpy=True)
        return embeddings.tolist()

# Global accessor
def get_embedder():
    return Embedder()
