# RAG System with Citations

A production-ready Retrieval-Augmented Generation (RAG) API that provides grounded answers with explicit citations and confidence scoring.

## Features
- **FastAPI Backend**: Fully async, non-blocking endpoints for high concurrency.
- **Vector Retrieval**: FAISS-based vector store with Sentence Transformers embeddings.
- **Query Processing**: Distinguishes between Factual, Exploratory, and Ambiguous queries.
- **Grounded Synthesis**: LLM prompts designed to minimize hallucinations and enforce citation usage.
- **Smart Citation Filtering**: Only returns citations actually referenced in the answer.
- **Real Confidence Scoring**: Calculates confidence based on citation coverage and grounding.
- **Configurable Models**: Choose your OpenAI model via environment variables.
- **Evaluation Framework**: Offline metrics for hallucination and citation coverage.

## Architecture

1.  **Ingest**: Text documents -> Chunking -> Embedding (`all-MiniLM-L6-v2`) -> FAISS Index.
2.  **Query**:
    *   **Classify**: Determine query intent (Factual vs Exploratory).
    *   **Retrieve**: Fetch Top-K relevant chunks.
    *   **Synthesize**: LLM generates answer using *only* retrieved context.
3.  **Response**: Returns Answer + Citations + Latency + Confidence.

## Setup

1.  **Clone & Install**
    ```bash
    git clone <repo_url>
    cd rag-system-with-citations
    python3 -m venv venv
    source venv/bin/activate
    pip install -r requirements.txt
    ```

2.  **Configure**
    Copy `.env.example` to `.env` and add your OpenAI API Key:
    ```bash
    cp .env.example .env
    # Edit .env and set OPENAI_API_KEY=sk-...
    ```

3.  **Ingest Data**
    Load the dummy corpus (or add your own `.txt` files to `src/data/corpus/`):
    ```bash
    python -m src.data.ingest
    ```

4.  **Run API**
    ```bash
    uvicorn src.api.main:app --reload
    ```
    Access docs at `http://localhost:8000/docs`.

## Evaluation

Run the offline evaluation suite:
```bash
python -m src.eval.evaluate
```
Results are saved to `reports/eval_results.csv`.

## Confidence Scoring

The API returns a `confidence` score (0.0-1.0) based on:

| Score | Meaning |
|-------|---------|
| 0.0 | No search results or error |
| 0.1 | LLM refused to answer (appropriate when context insufficient) |
| 0.3 | Answer given but no citations used (potential hallucination) |
| 0.6-1.0 | Answer with citations (scales with citation coverage) |

## Model Configuration

Set these environment variables to customize models:

```bash
SYNTHESIS_MODEL=gpt-4o-mini    # For answer generation (default: gpt-4o-mini)
CLASSIFICATION_MODEL=gpt-4o-mini  # For query classification (default: gpt-4o-mini)
```

## Trade-offs & Limitations
- **Vector DB**: Uses FAISS In-Memory for simplicity. For scale, switch to `pgvector` or Pinecone.
- **LLM**: Uses OpenAI for quality. Local LLMs (Llama 3) support is possible via `ollama`.
- **Latency**: Embedding is fast (local), but Synthesis depends on external API.
- **Async**: API is fully async for better concurrency under load.

## Requirements
- Python 3.11+
- Docker (optional)
