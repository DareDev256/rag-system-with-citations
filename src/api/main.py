from fastapi import FastAPI, HTTPException
from src.api.schemas import QueryRequest, QueryResponse, Citation
from src.retrieval.search import perform_search
from src.llm.synthesize import synthesize_answer, classify_query
from src.utils.timing import measure_latency
import logging
import time

# Logs
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("rag_api")

app = FastAPI(title="RAG System with Citations")

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/query", response_model=QueryResponse)
@measure_latency
async def query_endpoint(request: QueryRequest):
    start_total = time.perf_counter()
    original_query = request.query
    
    # 1. Classify
    category = classify_query(original_query)
    logger.info(f"Query: {original_query} | Category: {category}")
    
    # 2. Rewrite if ambiguous (Mocked for now)
    final_query = original_query
    if category == "ambiguous":
        # Simple heuristic or LLM call to clarify could go here
        pass
    
    # 3. Retrieve
    # search results: [{doc_id, snippet, score, source}, ...]
    search_results = perform_search(final_query, k=3)
    
    # 4. Synthesize
    synthesis_result = synthesize_answer(final_query, search_results)
    
    # 5. Format Response
    citations = []
    for res in synthesis_result.get("citations_used", []):
        citations.append(Citation(
            doc_id=res["doc_id"],
            snippet=res["snippet"],
            score=res.get("score"),
            source=res.get("source")
        ))
    
    # Calculate Latency (injected by decorator, but we want total flow here if needed, 
    # though decorator handles the response object attribute injection).
    # Since we are returning a Pydantic model, we can just instantiate it.
    # The @measure_latency on this endpoint might need to wrap the return value.
    # Let's trust the decorator to update the dict response if we returned a dict, 
    # but here we return an object.
    # Let's compute it explicitly for the object.
    
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
