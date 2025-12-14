from typing import List, Dict

def calculate_citation_coverage(answer: str, citations: List[Dict]) -> float:
    # A simple heuristic: check if citation IDs (e.g., [doc_1]) appear in the answer.
    # Returns % of provided citations that are actually used in the answer text.
    if not citations:
        return 0.0
    
    used_count = 0
    for cit in citations:
        doc_id = cit.get("doc_id")
        # Check for [doc_id] or just the id if prompt format varies
        if f"[{doc_id}]" in answer or doc_id in answer:
            used_count += 1
            
    return used_count / len(citations)

def estimate_hallucination_rate(answer: str, context_str: str) -> float:
    # True hallucination detection needs NLI or another LLM.
    # For this offline script, we'll placeholder it or use a simple overlap heuristic.
    # If the answer contains many named entities not in context, it might be hallucinated.
    # Low score = low hallucination (good).
    return 0.1 # Placeholder
