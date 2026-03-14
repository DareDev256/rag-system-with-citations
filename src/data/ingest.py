import glob
import logging
import os
from typing import List, Dict
from src.retrieval.embed import get_embedder
from src.retrieval.vector_store import VectorStore

logger = logging.getLogger(__name__)

CORPUS_DIR = "src/data/corpus"
INDEX_DIR = "data_store"

def load_documents(corpus_dir: str) -> List[Dict[str, str]]:
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
    logger.info("Starting ingestion...")
    
    # Ensure index dir exists
    os.makedirs(INDEX_DIR, exist_ok=True)
        
    docs = load_documents(CORPUS_DIR)
    logger.info("Loaded %d documents/chunks.", len(docs))
    
    if not docs:
        logger.warning("No documents found to ingest.")
        return

    embedder = get_embedder()
    texts = [d["text"] for d in docs]
    
    logger.info("Embedding documents...")
    embeddings = embedder.encode(texts)
    
    vector_store = VectorStore(
        index_path=os.path.join(INDEX_DIR, "faiss.index"),
        metadata_path=os.path.join(INDEX_DIR, "meta.json")
    )
    
    # Initialize index with correct dimension
    vector_store.create_index(dimension=len(embeddings[0]))
    vector_store.add_documents(embeddings, docs)
    vector_store.save_index()
    
    logger.info("Ingestion complete.")

if __name__ == "__main__":
    ingest()
