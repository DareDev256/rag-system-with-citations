from sentence_transformers import SentenceTransformer
from typing import List
import logging
import os
import re

logger = logging.getLogger(__name__)

# ─── Model Name Validation (CWE-73) ─────────────────────────────
# SentenceTransformer() accepts HuggingFace model IDs AND local filesystem
# paths / URLs. Without validation, a poisoned EMBEDDING_MODEL env var
# (e.g. "../../etc/passwd", "http://evil.com/backdoor") triggers path
# traversal or SSRF at model load time. Allowlist safe characters and
# reject anything that looks like a path or URL.
_MODEL_NAME_RE = re.compile(r'^[a-zA-Z0-9][a-zA-Z0-9_\-\.\/]{0,127}$')


def _validate_model_name(name: str) -> str:
    """Reject model names that look like filesystem paths or URLs."""
    if name.startswith(("/", ".", "~", "http://", "https://", "file://")):
        raise ValueError(
            f"EMBEDDING_MODEL must be a HuggingFace model ID, not a path or URL: '{name}'"
        )
    if not _MODEL_NAME_RE.match(name):
        raise ValueError(
            f"Invalid EMBEDDING_MODEL: '{name}' contains disallowed characters"
        )
    return name


EMBEDDING_MODEL = _validate_model_name(
    os.environ.get("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
)


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
