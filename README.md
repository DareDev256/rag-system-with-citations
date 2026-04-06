# RAG System with Citations

![Python 3.9+](https://img.shields.io/badge/python-3.9%2B-3776ab?style=flat-square&logo=python&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-009688?style=flat-square&logo=fastapi&logoColor=white)
![FAISS](https://img.shields.io/badge/FAISS-vector_search-4A154B?style=flat-square)
![Version](https://img.shields.io/badge/version-1.20.7-blue?style=flat-square)
![Tests](https://img.shields.io/badge/tests-488_passing-2ea44f?style=flat-square)
![Security](https://img.shields.io/badge/defense_layers-22-e05d44?style=flat-square&logo=shield)

A production-hardened Retrieval-Augmented Generation API that delivers grounded answers with explicit source citations, real-time confidence scoring, and 18-layer defense-in-depth security.

**Every answer must cite its sources. If it can't, the confidence score reflects that.**

---

## The Problem

Standard LLM APIs hallucinate freely and report high confidence regardless. This system forces citation-backed answers and scores confidence based on actual grounding — not vibes.

## Key Features

- **Citation-enforced answers** — LLM must reference `[doc_id]` from retrieved context or confidence drops to 0.3
- **Real confidence scoring** — calculated from citation coverage ratio, not model self-assessment
- **Hallucination detection** — flags citations that reference documents not in the retrieval set, plus a word-overlap `hallucination_rate` and composite `answer_quality_score` (0–1) in diagnostics
- **22-layer security** — rate limiting (thread-safe), API auth, input validation, output sanitization, HSTS, CSP (validated), request tracing, SSRF prevention (base URL + embeddings + LLM proxy), webhook HMAC verification, header injection defense, auth failure logging, pinned dependencies, CI security scanning, indirect prompt injection defense (snippet truncation + injection pattern neutralization + XML delimiter isolation), **metadata schema validation** (CWE-20 — type/length/format enforcement on ingest + storage load)
- **Provider-agnostic** — swap OpenAI for Claude, Ollama, or any OpenAI-compatible proxy with one env var
- **Webhook-triggered reindexing** — `POST /webhook/reindex` with HMAC-SHA256 signature verification for CMS/CI pipeline integration
- **488 tests, zero external deps** — full mock coverage, runs without API keys or FAISS indexes

## Architecture

```
  Request                           Security Perimeter
    │                    ┌──────────────────────────────────────────┐
    ▼                    │                                          │
┌────────┐  ┌───────────┤──────┐  ┌──────────┐  ┌──────────────┐  │
│ Client ├─►│ Rate Limit │ Auth │─►│ Body     │─►│ Request ID   │  │
└────────┘  │ (per-IP)   │(key) │  │ Size Cap │  │ (X-Request-ID│) │
            └────────────┴──────┘  └────┬─────┘  └──────┬───────┘  │
                                        │               │          │
                                   ┌────▼───────────────▼──┐       │
                                   │    POST /query         │       │
                                   └───────────┬────────────┘       │
                                               │                    │
                    ┌──────────────────────────┼────────────────┐   │
                    │  Pipeline                │                │   │
                    │  ┌───────────┐  ┌────────▼───────┐       │   │
                    │  │ Classify  │◄─┤ LLM Provider   │       │   │
                    │  │ Query     │  │ (configurable) │       │   │
                    │  └─────┬─────┘  └────────────────┘       │   │
                    │        │                                  │   │
                    │  ┌─────▼─────┐                            │   │
                    │  │ FAISS     │  ← all-MiniLM-L6-v2       │   │
                    │  │ Retrieve  │    similarity search       │   │
                    │  └─────┬─────┘                            │   │
                    │        │                                  │   │
                    │  ┌─────▼─────┐  ┌────────────────┐       │   │
                    │  │ Synthesize│─►│ Citation Check  │       │   │
                    │  │ Answer    │  │ + Hallucination │       │   │
                    │  └───────────┘  │   Detection     │       │   │
                    │                 └────────┬────────┘       │   │
                    └──────────────────────────┼────────────────┘   │
                                               │                    │
                                   ┌───────────▼────────────┐      │
                                   │ Output Sanitization    │      │
                                   │ + Security Headers     │      │
                                   └───────────┬────────────┘      │
                                               │                    │
                                               ▼                    │
                                        JSON Response               │
                                   (answer + citations +            │
                                    confidence + diagnostics)       │
                                        └───────────────────────────┘
```

**Data flow:** Documents → Chunk → Embed (all-MiniLM-L6-v2) → FAISS Index → Query-time retrieval → LLM synthesis with enforced citations → Confidence scoring → Sanitized response

## Quick Start

```bash
git clone https://github.com/DareDev256/rag-system-with-citations.git
cd rag-system-with-citations
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt
```

```bash
cp .env.example .env
# Set OPENAI_API_KEY=sk-...
```

```bash
python -m src.data.ingest          # Build the FAISS index
uvicorn src.api.main:app --reload  # Start the API → http://localhost:8000/docs
```

## API

### `POST /query`

```bash
# Basic query
curl -s -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"query": "What is retrieval-augmented generation?"}'

# With auth + retrieval depth + diagnostics
curl -s -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer YOUR_API_KEY" \
  -d '{"query": "How does FAISS indexing work?", "k": 3, "include_diagnostics": true}'
```

### Response

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
  "latency_ms": 1234.56,
  "diagnostics": {
    "retrieval_ms": 12.34,
    "synthesis_ms": 1220.45,
    "documents_searched": 5,
    "citation_coverage": 0.2,
    "hallucinated_citations": [],
    "hallucination_rate": 0.15,
    "answer_quality_score": 0.72
  }
}
```

### `POST /webhook/reindex`

Trigger document re-indexing via signed webhook (e.g., from a CMS or CI pipeline):

```bash
# Generate signature and send
SECRET="your-webhook-secret"
PAYLOAD='{"event": "document_updated"}'
SIG="sha256=$(echo -n "$PAYLOAD" | openssl dgst -sha256 -hmac "$SECRET" | cut -d' ' -f2)"
curl -X POST http://localhost:8000/webhook/reindex \
  -H "X-Hub-Signature-256: $SIG" \
  -d "$PAYLOAD"
```

### `GET /health`

```bash
curl http://localhost:8000/health
```

## Confidence Scoring

Real metric, not model self-assessment:

| Score | Signal |
|-------|--------|
| `0.0` | No search results or pipeline error |
| `0.1` | LLM refused to answer (context insufficient) |
| `0.3` | Answer generated but zero citations (hallucination risk) |
| `0.6–1.0` | Cited answer — scales with `0.6 + 0.4 × (cited_docs / retrieved_docs)` |

## Security

22-layer defense-in-depth. Each layer maps to a specific CWE threat class:

| Layer | Defense | CWE |
|-------|---------|-----|
| 1 | Rate limiting (per-IP sliding window) | CWE-770 |
| 2 | Trusted proxy IP resolution | CWE-348 |
| 3 | API key auth (constant-time comparison) | CWE-862 |
| 4 | Request body size cap (Content-Length + chunked) | CWE-400 |
| 5 | Input validation (Pydantic field constraints) | CWE-20 |
| 6 | Request ID tracing | CWE-778 |
| 7 | LLM timeout enforcement | CWE-400 |
| 8 | FAISS path traversal guard | CWE-22 |
| 9 | Output sanitization (answer + citations) | CWE-116 |
| 10 | Security headers (CSP, HSTS, X-Frame-Options) | CWE-693 |
| 11 | Error message sanitization (no stack traces) | CWE-209 |
| 12 | Log injection prevention | CWE-117 |
| 13 | Embedding model name validation | CWE-94 |
| 14 | Non-root Docker container | CWE-250 |
| 15 | Webhook HMAC-SHA256 signature verification | CWE-345 |
| 16 | SSRF prevention on proxy URL (scheme validation) | CWE-918 |
| 17 | Indirect prompt injection — snippet truncation | CWE-74 |
| 18 | Indirect prompt injection — pattern neutralization | CWE-74 |
| 19 | Indirect prompt injection — XML delimiter isolation | CWE-74 |
| 20 | Auth failure logging (brute-force visibility) | CWE-778 |
| 21 | Metadata schema validation (type/length/format) | CWE-20 |
| 22 | Pinned dependencies (no floating versions) | CWE-829 |

Full architecture and threat model: **[docs/security.md](docs/security.md)**

### Deep Dives

- **[Security Architecture](docs/security.md)** — full threat model, 22-layer CWE mapping, configuration reference, known gaps
- **[Proxy Integration](docs/proxy-integration.md)** — LiteLLM, OpenRouter, Ollama, vLLM setup guides
- **[SocialBu MCP Integration](docs/socialbu-mcp-integration.md)** — connect AI agents to SocialBu for automated social media posting via MCP + OpenAPI proxy

## Testing

488 tests across 18 suites. All mocked — runs without API keys, FAISS indexes, or network access:

```bash
pytest tests/ -v
```

| Suite | Tests | Scope |
|-------|------:|-------|
| `test_hardening.py` | 92 | Security headers, prompt injection, rate limiting, SSRF prevention, error sanitization |
| `test_edge_cases.py` | 62 | Unicode, boundary values, type coercion, hallucination edge cases |
| `test_core.py` | 52 | Citation extraction, confidence scoring, schema validation, metrics |
| `test_response_builders.py` | 40 | Citation assembly, diagnostics math, LLM output parsing, None-guard defense, type coercion |
| `test_contracts.py` | 35 | Behavioral interface tests — prompt safety, client factories, citation patterns |
| `test_prompt_injection.py` | 29 | Indirect injection defense — truncation, pattern blocking, false-positive avoidance, XML delimiters |
| `test_search.py` | 19 | Embedder singleton, search orchestration, model failure recovery |
| `test_integration_gaps.py` | 18 | Lifespan, ingest pipeline, client factories, race conditions |
| `test_ip_resolution.py` | 18 | Proxy extraction, spoofing resistance, IPv6, fallback |
| `test_vector_store.py` | 17 | FAISS wrapper — load/save/search, dimension validation |
| `test_middleware.py` | 17 | Body size, request ID, global exception handler |
| `test_resilience.py` | 16 | File I/O failures, corrupted state, singleton poisoning |
| `test_llm.py` | 14 | Sync LLM mocks — classification, synthesis, error handling |
| `test_llm_async.py` | 14 | Async LLM mocks — parity with sync implementations |
| `test_api.py` | 14 | FastAPI endpoint tests — happy path, validation, diagnostics |
| `test_evaluation.py` | 13 | Evaluation pipeline, keyword matching, CSV output |
| `test_auth.py` | 10 | API key auth — Bearer/X-API-Key, rejection, constant-time |
| `test_chunked_bypass.py` | 8 | Chunked encoding enforcement, early abort, multi-chunk accumulation |

All tests use mocks — no FAISS index, no OpenAI API calls, no external dependencies required to run.

## Configuration

All settings via environment variables or `.env`:

| Variable | Default | Description |
|----------|---------|-------------|
| `OPENAI_API_KEY` | — | **Required.** OpenAI API key (or proxy key) |
| `API_KEYS` | — | Comma-separated keys for `/query` auth |
| `OPENAI_BASE_URL` | — | Route through any OpenAI-compatible proxy (HTTPS/HTTP only, SSRF-validated) |
| `SYNTHESIS_MODEL` | `gpt-4o-mini` | Answer generation model |
| `CLASSIFICATION_MODEL` | `gpt-4o-mini` | Query classification model |
| `EMBEDDING_MODEL` | `all-MiniLM-L6-v2` | Sentence Transformers embedding model |
| `LLM_TIMEOUT` | `30` | OpenAI request timeout (1–300 seconds) |
| `WEBHOOK_SECRET` | — | HMAC-SHA256 secret for `/webhook/reindex` (disabled when unset) |
| `RATE_LIMIT_RPM` | `30` | Max requests/min per IP |
| `MAX_BODY_BYTES` | `65536` | Request body size limit (413 on exceed) |
| `TRUSTED_PROXY_COUNT` | `0` | Reverse proxies for X-Forwarded-For extraction |
| `CSP_POLICY` | strict default | Content-Security-Policy override |
| `CORS_ORIGINS` | — | Comma-separated allowed origins (CORS disabled when unset) |
| `DISABLE_DOCS` | — | Set to any value to hide `/docs` and `/redoc` in production |
| `HSTS_MAX_AGE` | `63072000` | HSTS max-age (seconds) |

Every response includes `X-Request-ID` for incident traceability (auto-generated or pass your own, max 64 chars).

### Alternative LLM Providers

Any OpenAI-compatible backend works as a drop-in replacement:

```bash
# Claude via LiteLLM
export OPENAI_BASE_URL=http://localhost:4000/v1
export SYNTHESIS_MODEL=claude-sonnet-4-20250514

# Local models via Ollama
export OPENAI_BASE_URL=http://localhost:11434/v1
export SYNTHESIS_MODEL=llama3
```

See **[docs/proxy-integration.md](docs/proxy-integration.md)** for full guides (LiteLLM, OpenRouter, Ollama, vLLM).

## Project Structure

```
src/
├── api/
│   ├── main.py          # FastAPI app, endpoint orchestration, /query + /health
│   ├── middleware.py     # Rate limiter, auth, security headers, body size, request ID
│   ├── response.py      # Citation assembly + diagnostics builders
│   └── schemas.py       # Pydantic request/response models
├── llm/
│   ├── prompt.py        # RAG + classification prompt templates
│   └── synthesize.py    # LLM calls (sync + async), prompt constants, safe-call helpers, CitationAnalysis, confidence scoring
├── retrieval/
│   ├── embed.py         # Sentence Transformers embedder (singleton)
│   ├── search.py        # Search orchestration layer
│   └── vector_store.py  # FAISS index wrapper (JSON metadata, not pickle)
├── eval/
│   ├── evaluate.py      # Offline evaluation pipeline
│   └── metrics.py       # Citation coverage + hallucination metrics
├── data/
│   ├── corpus/          # Source documents (.txt)
│   └── ingest.py        # Document loading + chunking + indexing
└── utils/
    ├── env.py           # Safe env var parsing (bounded int)
    ├── ip.py            # Trusted proxy IP resolution
    └── timing.py        # Latency decorator + TimingContext context manager
```
22-layer defense-in-depth covering webhook signature verification, API key authentication, rate limiting, input validation, output sanitization, security headers, request tracing, LLM timeout enforcement, indirect prompt injection defense, and more. See **[docs/security.md](docs/security.md)** for the full security architecture, threat model, and known gaps.

## Docker

```bash
docker build -t rag-citations .
docker run -p 8000:8000 --env-file .env rag-citations
```

Runs as non-root `appuser`. Healthcheck built in.

## Evaluation

```bash
python -m src.eval.evaluate
```

Results saved to `reports/eval_results.csv`.

## Design Decisions

| Decision | Rationale | Trade-off |
|----------|-----------|-----------|
| FAISS in-memory | Fast, zero-config for prototyping | Swap to pgvector/Pinecone for scale |
| OpenAI default | Best output quality for citations | Configurable — swap via env var |
| Sync FAISS + Async LLM | FAISS is CPU-bound and fast; LLM is I/O-bound | Async FAISS unnecessary at this scale |
| Citation-first | Verifiable > fluent — no citation = low confidence | May reject valid answers that paraphrase |
| JSON metadata (not pickle) | Eliminates deserialization attacks (CWE-502) | Slightly more verbose storage |

## Requirements

- Python 3.9+
- OpenAI API key (or compatible proxy)
- Docker (optional)
