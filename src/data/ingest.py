import os
import glob
from typing import List, Dict
from src.retrieval.embed import get_embedder
from src.retrieval.vector_store import VectorStore

CORPUS_DIR = "src/data/corpus"
INDEX_DIR = "data_store"

def load_documents(corpus_dir: str) -> List[Dict[str, str]]:
    docs = []
    if not os.path.exists(corpus_dir):
        print(f"Corpus directory {corpus_dir} does not exist.")
        return docs
        
    for filepath in glob.glob(os.path.join(corpus_dir, "*.txt")):
        filename = os.path.basename(filepath)
        with open(filepath, "r", encoding="utf-8") as f:
            text = f.read()
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
    print("Starting ingestion...")
    
    # Ensure index dir exists
    if not os.path.exists(INDEX_DIR):
        os.makedirs(INDEX_DIR)
        
    docs = load_documents(CORPUS_DIR)
    print(f"Loaded {len(docs)} documents/chunks.")
    
    if not docs:
        print("No documents found to ingest.")
        return

    embedder = get_embedder()
    texts = [d["text"] for d in docs]
    
    print("Embedding documents...")
    embeddings = embedder.encode(texts)
    
    vector_store = VectorStore(
        index_path=os.path.join(INDEX_DIR, "faiss.index"),
        metadata_path=os.path.join(INDEX_DIR, "meta.pkl")
    )
    
    # Initialize index with correct dimension
    vector_store.create_index(dimension=len(embeddings[0]))
    vector_store.add_documents(embeddings, docs)
    vector_store.save_index()
    
    print("Ingestion complete.")

if __name__ == "__main__":
    ingest()
