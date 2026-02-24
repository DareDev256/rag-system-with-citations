from sentence_transformers import SentenceTransformer
from typing import List
import logging
import os

logger = logging.getLogger(__name__)

EMBEDDING_MODEL = os.environ.get("EMBEDDING_MODEL", "all-MiniLM-L6-v2")


class Embedder:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            instance = super(Embedder, cls).__new__(cls)
            logger.info("Loading embedding model: %s", EMBEDDING_MODEL)
            try:
                instance.model = SentenceTransformer(EMBEDDING_MODEL)
            except Exception:
                logger.exception("Failed to load embedding model: %s", EMBEDDING_MODEL)
                raise
            cls._instance = instance
            logger.info("Embedding model loaded.")
        return cls._instance

    def encode(self, texts: List[str]) -> List[List[float]]:
        embeddings = self.model.encode(texts, convert_to_numpy=True)
        return embeddings.tolist()


# Global accessor
def get_embedder():
    return Embedder()
