# Security Architecture

Defense-in-depth model for the RAG System with Citations API. Every layer addresses a specific threat class — skip one and you leave a gap.

## Threat Surface

```
Client → [Rate Limiter] → [Input Validation] → [Security Headers]
                                                        │
    ┌───────────────────────────────────────────────────┘
    ▼
[Request ID] → [Classify LLM] → [FAISS Retrieval] → [Synthesize LLM]
                     │                  │                    │
              Timeout + Error     Path Traversal       Output Sanitization
                Masking             Guard              (answer + citations)
                     │                                       │
                     └───────────────────┬───────────────────┘
                                         ▼
                              [Response to Client]
```

## Defense Layers

### 1. Rate Limiting (CWE-770)

**Threat:** Unbounded `/query` requests drain the OpenAI API budget or DoS the service.

**Implementation:** In-memory per-IP sliding window (`src/api/main.py`).

| Parameter | Default | Env Var | Description |
|-----------|---------|---------|-------------|
| Window | 60s | — | Fixed 1-minute sliding window |
| Max requests | 30 | `RATE_LIMIT_RPM` | Requests per IP per window |
| Max tracked IPs | 10,000 | — | Hard cap, stale IPs evicted (CWE-400) |

**Why in-memory?** Single-process deployment. For multi-worker scaling, swap to Redis-backed middleware.

### 2. Input Validation (CWE-20)

**Threat:** Oversized, malformed, or injection-laden payloads.

**Implementation:** Pydantic models with constraints (`src/api/schemas.py`).

```python
query: str           # 1–1000 chars (constr)
k: int               # 1–20, default 5
include_diagnostics: bool  # opt-in only
```

Pydantic rejects invalid types/lengths before any business logic runs. FastAPI returns 422 with field-level errors.

### 3. Security Headers

**Threat:** Clickjacking, MIME sniffing, XSS, data exfiltration.

**Implementation:** `SecurityHeadersMiddleware` + `_apply_security_headers()` shared helper (`src/api/main.py`).

| Header | Value | Purpose |
|--------|-------|---------|
| `Content-Security-Policy` | `default-src 'self'; frame-ancestors 'none'; ...` | Block inline scripts, restrict resource origins |
| `Strict-Transport-Security` | `max-age=63072000; includeSubDomains; preload` | Force HTTPS for 2 years |
| `X-Content-Type-Options` | `nosniff` | Prevent MIME type sniffing |
| `X-Frame-Options` | `DENY` | Block framing (clickjacking) |
| `Referrer-Policy` | `strict-origin-when-cross-origin` | Limit referrer leakage |
| `Permissions-Policy` | `camera=(), microphone=(), geolocation=()` | Disable browser APIs |
| `X-DNS-Prefetch-Control` | `off` | Prevent DNS prefetch information leakage |
| `X-Permitted-Cross-Domain-Policies` | `none` | Block Flash/PDF cross-domain access |
| `Cache-Control` | `no-store` | Prevent caching of API responses |

**Key detail:** Exception handler responses bypass `BaseHTTPMiddleware`. The shared `_apply_security_headers()` function is called from both the middleware and the global exception handler to prevent header gaps on 500 responses.

**Overrides:**

```bash
CSP_POLICY="default-src 'self' 'unsafe-inline'"   # Custom CSP (e.g., for Swagger UI)
HSTS_MAX_AGE=0                                      # Disable HSTS (dev only)
```

### 4. Request ID Tracing (CWE-778)

**Threat:** Security incidents can't be correlated across logs without request-level identifiers.

**Implementation:** `RequestIDMiddleware` generates or validates `X-Request-ID` on every request.

- Client-provided IDs: accepted if ≤64 chars and no control characters
- Missing/invalid IDs: replaced with `uuid4().hex`
- Returned in response header and used in all log entries

### 5. Output Sanitization (CWE-116)

**Threat:** LLM responses and corpus-sourced fields may contain C0 control characters (terminal injection, ANSI escape attacks, log poisoning).

**Implementation:** `_sanitize_output()` strips chars in `U+0000–U+001F` range, preserving `\n`, `\r`, `\t`.

**Sanitized fields:**
- `answer` — LLM-generated, fully untrusted
- `query` — reflected back to client, untrusted
- `doc_id`, `snippet`, `source` — corpus-sourced, loaded from disk metadata

### 6. LLM Timeout & Error Masking (CWE-209)

**Threat:** Hung upstream LLM calls exhaust workers. Error messages leak internal state.

| Control | Value | Env Var |
|---------|-------|---------|
| Request timeout | 30s | `LLM_TIMEOUT` (min: 1s) |
| Error response | `"Error generating answer."` | — |
| Classification fallback | `"exploratory"` | — |

On failure, only `type(e).__name__` is logged — no stack traces, no API keys, no model names reach the client.

### 7. Global Exception Handler (CWE-209)

**Threat:** Unhandled exceptions leak stack traces, file paths, and module names in default FastAPI 500 responses.

**Response format:**
```json
{"detail": "Internal server error.", "request_id": "abc123"}
```

The request ID enables incident tracing in server logs without exposing internals to the client.

### 8. Path Traversal Guard (CWE-22)

**Threat:** Malicious filenames in the corpus directory could escape the intended path.

**Implementation:** `load_documents()` in `src/data/ingest.py` calls `os.path.realpath()` on every file path and verifies it starts with the canonical corpus directory.

### 9. Serialization Safety (CWE-502)

**Threat:** Pickle deserialization allows arbitrary code execution.

**Implementation:** All persistence uses JSON — FAISS metadata (`meta.json`), index files (FAISS binary format). No pickle anywhere in the codebase. Migrated in v1.2.0.

### 10. CORS (CWE-942)

**Threat:** Unauthorized cross-origin requests to the API.

**Implementation:** Disabled by default. Opt-in via `CORS_ORIGINS` env var.

```bash
CORS_ORIGINS="https://app.example.com,https://staging.example.com"
```

Only `GET` and `POST` methods allowed. Only `Content-Type` and `Authorization` headers permitted.

### 11. Concurrency Safety (CWE-362)

**Threat:** Race conditions in singleton initialization under concurrent access — duplicate clients, leaked connections.

**Implementation:** Double-checked locking on all singleton factories:
- `get_llm_client()` — `threading.Lock`
- `get_async_llm_client()` — `asyncio.Lock`
- `get_search_engine()` — `threading.Lock`

### 12. Swagger/ReDoc Suppression

**Threat:** Auto-generated API docs expose endpoint details in production.

```bash
DISABLE_DOCS=1   # Removes /docs and /redoc routes
```

## Log Injection Prevention

User input in log messages is sanitized: control characters stripped, query truncated to 200 chars. Format:

```
Query: <sanitized> | Category: <category> | request_id=<uuid>
```

## Container Hardening

The `Dockerfile` runs the application as a non-root user (`appuser`) with a health check that uses Python's `urllib` instead of `curl` (reducing attack surface by avoiding shell utilities in the container).

## What's Not Covered (Known Gaps)

| Gap | Risk | Mitigation Path |
|-----|------|-----------------|
| No API authentication | Any client can query | Add API key or JWT middleware |
| No request body size limit | Large payloads consume memory | Add `--limit-request-body` to uvicorn |
| No dependency pinning | Supply chain drift | Generate `requirements.lock` |
| In-memory rate limiter | Resets on restart, per-worker only | Redis-backed rate limiter |
| No audit log | Compliance gaps | Structured JSON logging to external sink |
