import os
import json
import time
from src.retrieval.search import perform_search
from src.llm.synthesize import synthesize_answer
from src.eval.metrics import calculate_citation_coverage
import pandas as pd

EVAL_DATA = [
    {"query": "What is RAG?", "expected_keywords": ["retrieval", "generation", "LLM"]},
    {"query": "Who walked on the moon?", "expected_keywords": ["Neil Armstrong", "Buzz Aldrin"]},
    {"query": "Tell me about Apollo 11.", "expected_keywords": ["1969", "Eagle"]}
]

def run_evaluation():
    print("Starting evaluation...")
    results = []
    
    for item in EVAL_DATA:
        query = item["query"]
        start_t = time.perf_counter()
        
        # 1. Pipeline
        search_res = perform_search(query, k=3)
        synth_res = synthesize_answer(query, search_res)
        
        latency = (time.perf_counter() - start_t) * 1000
        
        answer = synth_res["answer"]
        citations = synth_res.get("citations_used", [])
        
        # 2. Metrics
        coverage = calculate_citation_coverage(answer, citations)
        
        # Simple keyword recall
        hit = all(k.lower() in answer.lower() for k in item["expected_keywords"])
        
        results.append({
            "query": query,
            "latency_ms": latency,
            "citation_coverage": coverage,
            "keyword_match": hit,
            "answer_length": len(answer)
        })
        
    df = pd.DataFrame(results)
    os.makedirs("reports", exist_ok=True)
    df.to_csv("reports/eval_results.csv", index=False)
    
    print("\nEvaluation Results:")
    print(df.to_markdown())
    print(f"\nSaved to reports/eval_results.csv")

if __name__ == "__main__":
    run_evaluation()
