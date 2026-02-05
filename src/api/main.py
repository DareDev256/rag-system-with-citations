from fastapi import FastAPI, HTTPException
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
    version="1.1.0"
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
    logger.info(f"Query: {original_query} | Category: {category}")

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
