from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import Response
from src.api.schemas import QueryRequest, QueryResponse, Citation
from src.retrieval.search import perform_search
from src.llm.synthesize import synthesize_answer_async, classify_query_async
import logging
import os
import time

# Logs
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("rag_api")

app = FastAPI(
    title="RAG System with Citations",
    description="Production-ready RAG API with source attribution and confidence scoring",
    version="1.2.0",
    docs_url=None if os.getenv("DISABLE_DOCS") else "/docs",
    redoc_url=None if os.getenv("DISABLE_DOCS") else "/redoc",
)


class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        response: Response = await call_next(request)
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
        response.headers["Permissions-Policy"] = "camera=(), microphone=(), geolocation=()"
        response.headers["X-Permitted-Cross-Domain-Policies"] = "none"
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


@app.on_event("startup")
async def validate_env():
    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY environment variable is not set")


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.post("/query", response_model=QueryResponse)
async def query_endpoint(request: QueryRequest):
    start_total = time.perf_counter()
    original_query = request.query

    # 1. Classify (async - non-blocking)
    category = await classify_query_async(original_query)
    safe_query = original_query.replace("\n", " ").replace("\r", " ")[:200]
    logger.info("Query: %s | Category: %s", safe_query, category)

    # 2. Rewrite if ambiguous (placeholder for future enhancement)
    final_query = original_query
    if category == "ambiguous":
        # Could add query expansion or clarification here
        pass

    # 3. Retrieve (sync - FAISS is CPU-bound, fast enough)
    search_results = perform_search(final_query, k=request.k)

    # 4. Synthesize (async - non-blocking LLM call)
    synthesis_result = await synthesize_answer_async(final_query, search_results)

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

    return QueryResponse(
        query=original_query,
        category=category,
        answer=synthesis_result["answer"],
        citations=citations,
        confidence=synthesis_result["confidence"],
        latency_ms=round(latency, 2)
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
