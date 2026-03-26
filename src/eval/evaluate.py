"""Offline evaluation pipeline for the RAG system.

Runs a fixed set of queries through the full pipeline (retrieve → synthesize),
measures latency, citation coverage, and keyword recall, then writes results
to ``reports/eval_results.csv`` for trend tracking.

Usage::

    python -m src.eval.evaluate
"""

import logging
import os
from src.retrieval.search import perform_search
from src.llm.synthesize import synthesize_answer
from src.eval.metrics import calculate_citation_coverage
from src.utils.timing import TimingContext
import pandas as pd

logger = logging.getLogger(__name__)

#: Built-in evaluation queries with expected keywords for recall checking.
EVAL_DATA = [
    {"query": "What is RAG?", "expected_keywords": ["retrieval", "generation", "LLM"]},
    {"query": "Who walked on the moon?", "expected_keywords": ["Neil Armstrong", "Buzz Aldrin"]},
    {"query": "Tell me about Apollo 11.", "expected_keywords": ["1969", "Eagle"]}
]


def run_evaluation():
    """Execute the evaluation pipeline and save results to CSV.

    For each query in :data:`EVAL_DATA`, runs the full RAG pipeline,
    computes citation coverage and keyword recall, then writes a
    ``reports/eval_results.csv`` with per-query metrics.
    """
    logger.info("Starting evaluation...")
    results = []
    
    for item in EVAL_DATA:
        query = item["query"]

        # 1. Pipeline
        with TimingContext() as pipeline_timer:
            search_res = perform_search(query, k=3)
            synth_res = synthesize_answer(query, search_res)

        answer = synth_res["answer"]
        citations = synth_res.get("citations_used", [])
        
        # 2. Metrics
        coverage = calculate_citation_coverage(answer, citations)
        
        # Simple keyword recall
        hit = all(k.lower() in answer.lower() for k in item["expected_keywords"])
        
        results.append({
            "query": query,
            "latency_ms": pipeline_timer.ms,
            "citation_coverage": coverage,
            "keyword_match": hit,
            "answer_length": len(answer)
        })
        
    df = pd.DataFrame(results)
    os.makedirs("reports", exist_ok=True)
    df.to_csv("reports/eval_results.csv", index=False)
    
    logger.info("Evaluation Results:\n%s", df.to_markdown())
    logger.info("Saved to reports/eval_results.csv")

if __name__ == "__main__":
    run_evaluation()
