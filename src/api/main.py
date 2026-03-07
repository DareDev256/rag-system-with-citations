from collections import defaultdict
from contextlib import asynccontextmanager

import logging
import os
import re
import time

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import Response
from src.api.schemas import QueryRequest, QueryResponse, Citation, Diagnostics
from src.retrieval.search import perform_search
from src.llm.synthesize import synthesize_answer_async, classify_query_async, extract_cited_doc_ids

# Logs
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("rag_api")


# ─── Rate Limiter (in-memory, per-IP) ────────────────────────────────
# Each /query call costs real money (OpenAI API). Without rate limiting,
# any client can drain the budget or DoS the service. CWE-770.
_RATE_LIMIT = int(os.getenv("RATE_LIMIT_RPM", "30"))  # requests per minute
_rate_store: dict = defaultdict(list)  # IP -> list of timestamps


def _check_rate_limit(client_ip: str) -> bool:
    """Return True if request is allowed, False if rate-limited."""
    now = time.monotonic()
    window = now - 60  # 1-minute sliding window
    # Prune expired entries
    _rate_store[client_ip] = [t for t in _rate_store[client_ip] if t > window]
    if len(_rate_store[client_ip]) >= _RATE_LIMIT:
        return False
    _rate_store[client_ip].append(now)
    return True


# ─── Output sanitization ─────────────────────────────────────────────
# LLM responses are untrusted — strip control chars before returning to client.
# Input is already sanitized for logging, but output was trusted blindly.
_CONTROL_CHAR_RE = re.compile(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]')


def _sanitize_output(text: str) -> str:
    """Strip C0 control chars from LLM output (preserves \\n, \\r, \\t)."""
    return _CONTROL_CHAR_RE.sub('', text)


@asynccontextmanager
async def lifespan(app):
    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY environment variable is not set")
    yield


app = FastAPI(
    title="RAG System with Citations",
    description="Production-ready RAG API with source attribution and confidence scoring",
    version="1.6.0",
    docs_url=None if os.getenv("DISABLE_DOCS") else "/docs",
    redoc_url=None if os.getenv("DISABLE_DOCS") else "/redoc",
    lifespan=lifespan,
)


# Default CSP — override via CSP_POLICY env var for proxied/Swagger environments
_DEFAULT_CSP = (
    "default-src 'self'; "
    "script-src 'self'; "
    "style-src 'self'; "
    "img-src 'self' data:; "
    "font-src 'self'; "
    "connect-src 'self'; "
    "object-src 'none'; "
    "base-uri 'self'; "
    "form-action 'self'; "
    "frame-ancestors 'none'; "
    "upgrade-insecure-requests"
)

# HSTS max-age in seconds (default 2 years), override via HSTS_MAX_AGE env var
def _parse_hsts_max_age() -> int:
    raw = os.getenv("HSTS_MAX_AGE", "63072000")
    try:
        val = int(raw)
        if val < 0:
            raise ValueError
        return val
    except (ValueError, TypeError):
        logger.warning("Invalid HSTS_MAX_AGE '%s', using default 63072000", raw)
        return 63072000

_HSTS_MAX_AGE = _parse_hsts_max_age()


class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        response: Response = await call_next(request)
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["X-DNS-Prefetch-Control"] = "off"
        response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
        response.headers["Permissions-Policy"] = "camera=(), microphone=(), geolocation=()"
        response.headers["X-Permitted-Cross-Domain-Policies"] = "none"
        response.headers["Cache-Control"] = "no-store"
        response.headers["Content-Security-Policy"] = os.getenv("CSP_POLICY", _DEFAULT_CSP)
        response.headers["Strict-Transport-Security"] = (
            f"max-age={_HSTS_MAX_AGE}; includeSubDomains; preload"
        )
        return response

app.add_middleware(SecurityHeadersMiddleware)

# CORS — restrict to explicit origins in production
allowed_origins = os.getenv("CORS_ORIGINS", "").split(",")
allowed_origins = [o.strip() for o in allowed_origins if o.strip()]
if allowed_origins:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=allowed_origins,
        allow_methods=["GET", "POST"],
        allow_headers=["Content-Type", "Authorization"],
    )


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.post("/query", response_model=QueryResponse)
async def query_endpoint(request: QueryRequest, req: Request):
    # Rate limit — protect OpenAI budget and prevent abuse
    client_ip = req.client.host if req.client else "unknown"
    if not _check_rate_limit(client_ip):
        raise HTTPException(status_code=429, detail="Rate limit exceeded. Try again later.")

    start_total = time.perf_counter()
    original_query = request.query

    # 1. Classify (async - non-blocking)
    category = await classify_query_async(original_query)
    # Strip all C0 control chars (U+0000–U+001F) to prevent log injection
    safe_query = re.sub(r'[\x00-\x1f\x7f]', ' ', original_query)[:200]
    logger.info("Query: %s | Category: %s", safe_query, category)

    # 2. Rewrite if ambiguous (placeholder for future enhancement)
    final_query = original_query
    if category == "ambiguous":
        # Could add query expansion or clarification here
        pass

    # 3. Retrieve (sync - FAISS is CPU-bound, fast enough)
    start_retrieval = time.perf_counter()
    search_results = perform_search(final_query, k=request.k)
    retrieval_ms = (time.perf_counter() - start_retrieval) * 1000

    # 4. Synthesize (async - non-blocking LLM call)
    start_synthesis = time.perf_counter()
    synthesis_result = await synthesize_answer_async(final_query, search_results)
    synthesis_ms = (time.perf_counter() - start_synthesis) * 1000

    # 5. Format Response
    citations = [
        Citation(
            doc_id=res["doc_id"],
            snippet=res["snippet"],
            score=res.get("score"),
            source=res.get("source")
        )
        for res in synthesis_result.get("citations_used", [])
    ]

    end_total = time.perf_counter()
    latency = (end_total - start_total) * 1000

    # 6. Build diagnostics (opt-in)
    diagnostics = None
    if request.include_diagnostics:
        available_ids = {res["doc_id"] for res in search_results}
        all_cited = extract_cited_doc_ids(synthesis_result["answer"])
        hallucinated = sorted(all_cited - available_ids)
        valid_cited = all_cited & available_ids
        coverage = len(valid_cited) / len(search_results) if search_results else 0.0

        diagnostics = Diagnostics(
            retrieval_ms=round(retrieval_ms, 2),
            synthesis_ms=round(synthesis_ms, 2),
            documents_searched=len(search_results),
            citation_coverage=round(coverage, 2),
            hallucinated_citations=hallucinated,
        )

    # Sanitize outputs — LLM responses and reflected query are untrusted
    safe_answer = _sanitize_output(synthesis_result["answer"])
    safe_reflected_query = _sanitize_output(original_query)

    return QueryResponse(
        query=safe_reflected_query,
        category=category,
        answer=safe_answer,
        citations=citations,
        confidence=synthesis_result["confidence"],
        latency_ms=round(latency, 2),
        diagnostics=diagnostics,
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
