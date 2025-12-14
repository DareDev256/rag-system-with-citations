from typing import List, Dict

# Standard prompt for RAG
RAG_PROMPT_TEMPLATE = """You are a helpful assistant. Answer the user's question using ONLY the context provided below.
If the context does not contain the answer, say "I cannot answer this based on the provided documents."

Context:
{context_str}

User Question: {query}

Answer (include citations like [doc_id] where appropriate):"""

# Prompt for query classification
CLASSIFICATION_PROMPT_TEMPLATE = """Classify the following user query into one of three categories: "factual", "exploratory", or "ambiguous".
- factual: Specific questions looking for a precise fact (e.g., "What is the capital of France?").
- exploratory: Open-ended questions asking for explanations or summaries (e.g., "Tell me about RAG systems").
- ambiguous: Unclear or vague queries that might need clarification.

Query: {query}

Return only the category name in lowercase."""

def build_context_str(results: List[Dict]) -> str:
    context_parts = []
    for res in results:
        # Format: [doc_id] Content...
        context_parts.append(f"[{res['doc_id']}] {res['snippet']}")
    return "\n\n".join(context_parts)
