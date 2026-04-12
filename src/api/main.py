"""FastAPI application — endpoint orchestration for the RAG pipeline.

This module is the thin entry point: it wires up middleware, defines
the /query and /health endpoints, and delegates all heavy lifting to
the retrieval, LLM, and response-builder modules. Security enforcement
(rate limiting, auth, headers, body size) lives in middleware.py.
"""

import logging
import os
import re

from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from src.api.middleware import (
    MaxBodySizeMiddleware,
    RequestIDMiddleware,
    SecurityHeadersMiddleware,
    apply_security_headers,
    authenticate,
    check_rate_limit,
    sanitize_output,
)
from src.api.schemas import QueryRequest, QueryResponse
from src.api.response import build_citations, build_diagnostics
from src.retrieval.search import perform_search
from src.llm.synthesize import classify_query_async, synthesize_answer_async
from src.utils.ip import resolve_client_ip
from src.utils.timing import TimingContext

# Logs
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("rag_api")

_CONTROL_RE = re.compile(r'[\x00-\x1f\x7f]')


def _safe_log_query(query: str, max_len: int = 200) -> str:
    """Strip control chars and truncate for safe log output."""
    return _CONTROL_RE.sub(' ', query)[:max_len]


@asynccontextmanager
async def lifespan(app):
    """Validate required env vars at startup; fail fast if missing."""
    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY environment variable is not set")
    yield


app = FastAPI(
    title="RAG System with Citations",
    description="Production-ready RAG API with source attribution and confidence scoring",
    version="1.21.6",
    docs_url=None if os.getenv("DISABLE_DOCS") else "/docs",
    redoc_url=None if os.getenv("DISABLE_DOCS") else "/redoc",
    lifespan=lifespan,
)


# ─── Middleware Stack ─────────────────────────────────────────────────
# Registration order is reversed at runtime (last registered = outermost).
# Execution order: CORS → MaxBody → RequestID → SecurityHeaders
app.add_middleware(SecurityHeadersMiddleware)
app.add_middleware(RequestIDMiddleware)
app.add_middleware(MaxBodySizeMiddleware)

allowed_origins = os.getenv("CORS_ORIGINS", "").split(",")
allowed_origins = [o.strip() for o in allowed_origins if o.strip()]
if allowed_origins:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=allowed_origins,
        allow_methods=["GET", "POST"],
        allow_headers=["Content-Type", "Authorization"],
    )


# ─── Global Exception Handler (CWE-209) ─────────────────────────────
@app.exception_handler(Exception)
async def _global_exception_handler(request: Request, exc: Exception):
    request_id = request.state.request_id if hasattr(request.state, "request_id") else "unknown"
    logger.error("Unhandled %s [request_id=%s]", type(exc).__name__, request_id)
    response = JSONResponse(
        status_code=500,
        content={"detail": "Internal server error.", "request_id": request_id},
    )
    apply_security_headers(response)
    return response


# ─── Endpoints ───────────────────────────────────────────────────────

@app.get("/health")
async def health():
    """Liveness probe — returns 200 with {"status": "ok"}."""
    return {"status": "ok"}


@app.post("/query", response_model=QueryResponse)
async def query_endpoint(request: QueryRequest, req: Request):
    """RAG query pipeline: authenticate → rate-limit → classify → retrieve → synthesize → respond."""
    # Authenticate
    auth_error = authenticate(req)
    if auth_error:
        raise HTTPException(status_code=401, detail=auth_error)

    # Rate limit
    client_ip = resolve_client_ip(req)
    if not check_rate_limit(client_ip):
        raise HTTPException(status_code=429, detail="Rate limit exceeded. Try again later.")

    with TimingContext() as total_timer:
        original_query = request.query

        # 1. Classify
        category = await classify_query_async(original_query)
        request_id = req.state.request_id if hasattr(req.state, "request_id") else "unknown"
        logger.info("Query: %s | Category: %s | request_id=%s", _safe_log_query(original_query), category, request_id)

        # 2. Rewrite if ambiguous
        final_query = original_query
        if category == "ambiguous":
            pass

        # 3. Retrieve
        with TimingContext() as retrieval_timer:
            search_results = perform_search(final_query, k=request.k)

        # 4. Synthesize
        with TimingContext() as synthesis_timer:
            synthesis_result = await synthesize_answer_async(final_query, search_results)

        # 5. Format Response
        citations = build_citations(search_results, synthesis_result, sanitize_output)

    # 6. Diagnostics (opt-in) — reuse the CitationAnalysis already
    #    computed during synthesis to avoid a redundant analyze_citations call.
    diagnostics = None
    if request.include_diagnostics:
        diagnostics = build_diagnostics(
            search_results, synthesis_result["answer"],
            retrieval_timer.ms, synthesis_timer.ms,
            synthesis_result["confidence"],
            citation_analysis=synthesis_result.get("_citation_analysis"),
        )

    return QueryResponse(
        query=sanitize_output(original_query),
        category=category,
        answer=sanitize_output(synthesis_result["answer"]),
        citations=citations,
        confidence=synthesis_result["confidence"],
        latency_ms=total_timer.ms,
        diagnostics=diagnostics,
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
