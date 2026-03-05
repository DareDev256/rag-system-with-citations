# RAG System with Citations

A production-ready Retrieval-Augmented Generation API that delivers grounded answers with explicit source citations and real-time confidence scoring.

Built with **FastAPI** + **FAISS** + **OpenAI**, designed to minimize hallucinations through citation-enforced prompting and smart confidence metrics.

## Why This Exists

Standard LLM APIs hallucinate. This system forces every answer to cite its sources — and scores confidence based on how well the answer is grounded in retrieved documents. If the model can't back up its claims, the confidence score reflects that.

## Architecture

```
                    ┌─────────────┐
                    │  POST /query │
                    └──────┬──────┘
                           │
                    ┌──────▼──────┐
                    │  Classify   │  ← GPT-4o-mini determines query type
                    │  Query      │    (factual / exploratory / ambiguous)
                    └──────┬──────┘
                           │
                    ┌──────▼──────┐
                    │  Retrieve   │  ← FAISS similarity search (Top-K)
                    │  Context    │    Sentence Transformers embeddings
                    └──────┬──────┘
                           │
                    ┌──────▼──────┐
                    │  Synthesize │  ← GPT-4o-mini generates answer
                    │  Answer     │    with enforced [doc_id] citations
                    └──────┬──────┘
                           │
                    ┌──────▼──────┐
                    │  Score &    │  ← Real confidence based on
                    │  Validate   │    citation coverage + grounding
                    └─────────────┘
```

**Data Flow:** Documents → Chunk → Embed (all-MiniLM-L6-v2) → FAISS Index → Query-time retrieval → LLM synthesis with citations → Confidence scoring

## Project Structure

```
src/
├── api/
│   ├── main.py          # FastAPI app, /query and /health endpoints
│   └── schemas.py       # Pydantic request/response models
├── llm/
│   ├── prompt.py        # RAG + classification prompt templates
│   └── synthesize.py    # LLM calls, citation extraction, confidence scoring
├── retrieval/
│   ├── embed.py         # Sentence Transformers embedder (singleton)
│   ├── search.py        # Search orchestration layer
│   └── vector_store.py  # FAISS index wrapper (load/save/search)
├── eval/
│   ├── evaluate.py      # Offline evaluation pipeline
│   └── metrics.py       # Citation coverage + hallucination metrics
├── data/
│   ├── corpus/          # Source documents (.txt)
│   └── ingest.py        # Document loading + chunking + indexing
└── utils/
    └── timing.py        # Latency measurement decorator
```

## Quick Start

```bash
git clone https://github.com/DareDev256/rag-system-with-citations.git
cd rag-system-with-citations
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

```bash
cp .env.example .env
# Edit .env → set OPENAI_API_KEY=sk-...
```

```bash
python -m src.data.ingest     # Build the FAISS index
uvicorn src.api.main:app --reload  # Start the API
```

API docs available at `http://localhost:8000/docs`

## API Usage

### Query

```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"query": "What is retrieval-augmented generation?"}'
```

Control retrieval depth with `k` (1–20, default 5):

```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"query": "How does FAISS indexing work?", "k": 3}'
```

### Response Format

```json
{
  "query": "What is retrieval-augmented generation?",
  "category": "factual",
  "answer": "Retrieval-augmented generation (RAG) is ... [doc_001]",
  "citations": [
    {
      "doc_id": "doc_001",
      "snippet": "RAG combines retrieval with generation...",
      "score": 0.92,
      "source": "rag_overview.txt"
    }
  ],
  "confidence": 0.85,
  "latency_ms": 1234.56
}
```

### Health Check

```bash
curl http://localhost:8000/health
```

## Confidence Scoring

The API returns a real `confidence` score (0.0–1.0) — not a random number, but a calculated metric based on how well the answer is grounded:

| Score | Meaning |
|-------|---------|
| **0.0** | No search results or error |
| **0.1** | LLM refused to answer (appropriate when context is insufficient) |
| **0.3** | Answer given but no citations used (potential hallucination) |
| **0.6–1.0** | Answer with citations — scales with citation coverage ratio |

Formula: `confidence = 0.6 + 0.4 × (cited_docs / retrieved_docs)`

## Testing

219 tests across 10 test suites — pure function tests, mock-based LLM tests, async tests, API endpoint tests, edge cases, hardening, and integration tests:

```bash
pytest tests/ -v
```

| Suite | Tests | Coverage |
|-------|-------|----------|
| `test_core.py` | 50 | Context formatting, citation extraction, confidence scoring, schema validation, evaluation metrics, document loading, latency measurement |
| `test_edge_cases.py` | 44 | Unicode, missing keys, boundary values, exception propagation, type coercion, hallucination rate |
| `test_hardening.py` | 29 | Security headers (CSP directives, HSTS, cache-control, DNS prefetch), prompt injection resistance, file I/O error resilience, `_client_kwargs` config |
| `test_llm.py` | 14 | Mock-based `classify_query` and `synthesize_answer` — category fallback, citation parsing, error handling, hallucination filtering |
| `test_llm_async.py` | 14 | Async variants of all LLM functions using `AsyncMock` — validates parity with sync implementations |
| `test_api.py` | 10 | FastAPI `TestClient` — health endpoint, query happy path, validation errors (422), `k` parameter forwarding, latency headers |
| `test_vector_store.py` | 14 | FAISS `VectorStore` wrapper — init, load/save index, add documents, create index, similarity search |
| `test_search.py` | 17 | `Embedder` singleton behavior, model load failure recovery, configurable model, `get_search_engine` factory, `perform_search` orchestration |
| `test_integration_gaps.py` | 14 | Environment validation, ingest pipeline, LLM client factories (sync + async), proxy base_url forwarding |
| `test_evaluation.py` | 13 | `run_evaluation` pipeline, keyword matching, CSV output, latency measurement, coverage forwarding |

All tests use mocks — no FAISS index, no OpenAI API calls, no external dependencies required to run.

## Configuration

Set via environment variables or `.env`:

| Variable | Default | Description |
|----------|---------|-------------|
| `OPENAI_API_KEY` | — | Required. Your OpenAI API key (or proxy key) |
| `OPENAI_BASE_URL` | — | Optional. Route LLM calls through any OpenAI-compatible proxy |
| `SYNTHESIS_MODEL` | `gpt-4o-mini` | Model for answer generation |
| `CLASSIFICATION_MODEL` | `gpt-4o-mini` | Model for query classification |
| `EMBEDDING_MODEL` | `all-MiniLM-L6-v2` | Sentence Transformers model for embeddings |
| `CSP_POLICY` | (strict default) | Override Content-Security-Policy header |
| `HSTS_MAX_AGE` | `63072000` | HSTS max-age in seconds (default 2 years) |

### Using Alternative LLM Providers

Any OpenAI-compatible proxy works as a drop-in backend — Anthropic (via LiteLLM), Qwen (via vLLM), local models (via Ollama):

```bash
# Example: route through LiteLLM to use Claude
export OPENAI_BASE_URL=http://localhost:4000/v1
export SYNTHESIS_MODEL=claude-sonnet-4-20250514
```

See **[docs/proxy-integration.md](docs/proxy-integration.md)** for full setup guides (LiteLLM, OpenRouter, Ollama, vLLM).

## Docker

```bash
docker build -t rag-citations .
docker run -p 8000:8000 --env-file .env rag-citations
```

## Evaluation

Run the offline evaluation suite:

```bash
python -m src.eval.evaluate
```

Results saved to `reports/eval_results.csv`.

## Trade-offs & Design Decisions

- **FAISS in-memory** — Simple and fast for prototyping. For production scale, swap to pgvector or Pinecone.
- **OpenAI dependency** — Chosen for output quality. Could swap to local models via Ollama for cost/latency trade-offs.
- **Sync FAISS + Async LLM** — FAISS search is CPU-bound and fast enough synchronously. LLM calls are async to avoid blocking the event loop.
- **Citation-first design** — The system prioritizes verifiable answers over fluent ones. If the LLM doesn't cite sources, confidence drops to 0.3.

## Requirements

- Python 3.9+
- OpenAI API key
- Docker (optional)
