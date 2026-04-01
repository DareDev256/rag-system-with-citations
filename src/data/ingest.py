"""Document ingestion pipeline — load, chunk, embed, and index.

Reads ``.txt`` files from the corpus directory, splits them into
paragraph-level chunks, generates embeddings via Sentence Transformers,
and writes a FAISS index + JSON metadata to ``data_store/``.

Usage::

    python -m src.data.ingest
"""

import glob
import logging
import os
from typing import List, Dict
from src.retrieval.embed import get_embedder
from src.retrieval.vector_store import DEFAULT_STORE_DIR, create_default_store

logger = logging.getLogger(__name__)

CORPUS_DIR = "src/data/corpus"
INDEX_DIR = DEFAULT_STORE_DIR


def load_documents(corpus_dir: str) -> List[Dict[str, str]]:
    """Load and chunk ``.txt`` files from *corpus_dir* into document dicts.

    Each paragraph (split on double newlines) becomes a separate document
    with keys ``doc_id`` (``<filename>_<index>``), ``text``, and ``source``.
    Applies a path-traversal guard via ``os.path.realpath`` and skips files
    that fail to read (permission denied, encoding errors).

    Returns an empty list if *corpus_dir* does not exist.
    """
    docs = []
    if not os.path.exists(corpus_dir):
        logger.warning("Corpus directory %s does not exist.", corpus_dir)
        return docs

    # Path traversal guard — resolve symlinks and verify canonical path
    real_corpus = os.path.realpath(corpus_dir)
    for filepath in glob.glob(os.path.join(corpus_dir, "*.txt")):
        real_file = os.path.realpath(filepath)
        if not real_file.startswith(real_corpus + os.sep) and real_file != real_corpus:
            logger.warning("Skipping %s: path traversal detected", filepath)
            continue
        filename = os.path.basename(real_file)
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                text = f.read()
        except (OSError, UnicodeDecodeError) as e:
            logger.warning("Skipping %s: %s", filename, e)
            continue
        # Simple chunking by paragraph or fixed size could go here.
        # For this demo, treating each file or paragraph as a doc.
        # Let's split by double newline to get paragraphs.
        paragraphs = text.split("\n\n")
        for i, p in enumerate(paragraphs):
            p = p.strip()
            if not p:
                continue
            docs.append({
                "doc_id": f"{filename}_{i}",
                "text": p,
                "source": filename
            })
    return docs

def ingest():
    """Run the full ingestion pipeline: load → embed → index → save.

    Creates ``data_store/`` if it doesn't exist, embeds all document
    chunks with the configured Sentence Transformers model, builds a
    FAISS index, and persists it alongside JSON metadata.
    """
    logger.info("Starting ingestion...")
    
    docs = load_documents(CORPUS_DIR)
    logger.info("Loaded %d documents/chunks.", len(docs))

    if not docs:
        logger.warning("No documents found to ingest.")
        return

    embedder = get_embedder()
    texts = [d["text"] for d in docs]

    logger.info("Embedding documents...")
    embeddings = embedder.encode(texts)

    vector_store = create_default_store(INDEX_DIR)

    # Initialize index with correct dimension
    vector_store.create_index(dimension=len(embeddings[0]))
    vector_store.add_documents(embeddings, docs)
    vector_store.save_index()
    
    logger.info("Ingestion complete.")

if __name__ == "__main__":
    ingest()
