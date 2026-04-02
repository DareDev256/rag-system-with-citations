from typing import List, Dict

# Standard prompt for RAG
# NOTE: Use format_rag_prompt() / format_classification_prompt() instead of
# calling .format() directly — prevents KeyError when user query contains
# Python format placeholders like {context_str}.
_RAG_PROMPT_TEMPLATE = """You are a helpful assistant. Answer the user's question using ONLY the context provided below.
If the context does not contain the answer, say "I cannot answer this based on the provided documents."

Context:
{context_str}

User Question: {query}

Answer (include citations like [doc_id] where appropriate):"""

_CLASSIFICATION_PROMPT_TEMPLATE = """Classify the following user query into one of three categories: "factual", "exploratory", or "ambiguous".
- factual: Specific questions looking for a precise fact (e.g., "What is the capital of France?").
- exploratory: Open-ended questions asking for explanations or summaries (e.g., "Tell me about RAG systems").
- ambiguous: Unclear or vague queries that might need clarification.

Query: {query}

Return only the category name in lowercase."""


def format_rag_prompt(context_str: str, query: str) -> str:
    """Build RAG prompt safely — user query cannot trigger format expansion.

    Uses split-and-join to avoid double-replacement: if context_str contains
    the literal text '{query}', a naive chained .replace() would silently
    expand it into the user's actual query, corrupting the context.
    """
    parts = _RAG_PROMPT_TEMPLATE.split("{context_str}", 1)
    return parts[0] + context_str + parts[1].replace("{query}", query)


def format_classification_prompt(query: str) -> str:
    """Build classification prompt safely — user query cannot trigger format expansion."""
    return _CLASSIFICATION_PROMPT_TEMPLATE.replace("{query}", query)


def build_context_str(results: List[Dict]) -> str:
    context_parts = []
    for res in results:
        doc_id = res.get("doc_id")
        snippet = res.get("snippet")
        # Skip results with missing identity — they pollute the LLM context
        # with "[None] None" and can cause false citation matches downstream.
        # Use ``is None`` for doc_id (not falsy check) so valid values like
        # integer 0 are preserved — consistent with _sanitize_field and
        # calculate_citation_coverage.  Snippet uses ``not`` because empty
        # string snippets waste context with "[d1] " blank entries.
        if doc_id is None or not snippet:
            continue
        context_parts.append(f"[{doc_id}] {snippet}")
    return "\n\n".join(context_parts)
