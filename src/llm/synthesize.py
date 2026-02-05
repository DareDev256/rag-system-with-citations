import os
import re
import openai
from typing import List, Dict, Any, Set
from src.llm.prompt import RAG_PROMPT_TEMPLATE, CLASSIFICATION_PROMPT_TEMPLATE, build_context_str
from src.utils.timing import measure_latency

# Configuration from environment
DEFAULT_MODEL = "gpt-4o-mini"
CLASSIFICATION_MODEL = os.getenv("CLASSIFICATION_MODEL", "gpt-4o-mini")
SYNTHESIS_MODEL = os.getenv("SYNTHESIS_MODEL", DEFAULT_MODEL)

# Cached clients (sync and async)
_llm_client = None
_async_llm_client = None


def get_llm_client():
    global _llm_client
    if _llm_client is None:
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            print("Warning: OPENAI_API_KEY not found in env.")
        _llm_client = openai.OpenAI(api_key=api_key)
    return _llm_client


def get_async_llm_client():
    global _async_llm_client
    if _async_llm_client is None:
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            print("Warning: OPENAI_API_KEY not found in env.")
        _async_llm_client = openai.AsyncOpenAI(api_key=api_key)
    return _async_llm_client


def extract_cited_doc_ids(answer: str, available_ids: Set[str] = None) -> Set[str]:
    """Extract all [doc_id] citations from the answer text.

    If available_ids is provided, only returns IDs that match actual
    documents from search results, filtering out hallucinated citations.
    """
    pattern = r'\[([^\]]+)\]'
    matches = re.findall(pattern, answer)
    cited = set(matches)
    if available_ids is not None:
        cited = cited & available_ids
    return cited


def calculate_confidence(answer: str, search_results: List[Dict], cited_ids: Set[str]) -> float:
    """
    Calculate confidence score based on:
    - Citation coverage: % of retrieved docs that were cited
    - Grounding check: Whether the answer uses citations at all
    - Refusal detection: Lower confidence if LLM refused to answer
    """
    if not search_results:
        return 0.0

    # Check for refusal patterns
    refusal_patterns = [
        "cannot answer",
        "don't have enough",
        "not enough information",
        "no information",
        "unable to answer"
    ]
    answer_lower = answer.lower()
    if any(pattern in answer_lower for pattern in refusal_patterns):
        return 0.1  # Low but non-zero (the refusal itself is a valid response)

    # No citations used = low confidence (LLM may be hallucinating)
    if not cited_ids:
        return 0.3

    # Calculate citation coverage
    available_ids = {res["doc_id"] for res in search_results}
    valid_citations = cited_ids & available_ids

    if not valid_citations:
        return 0.3  # Citations don't match available docs

    # Base confidence from citation ratio
    citation_ratio = len(valid_citations) / len(search_results)

    # Scale: at least 1 citation = 0.6, all cited = 1.0
    confidence = 0.6 + (0.4 * citation_ratio)

    return round(confidence, 2)


@measure_latency
def classify_query(query: str) -> str:
    client = get_llm_client()
    try:
        response = client.chat.completions.create(
            model=CLASSIFICATION_MODEL,
            messages=[
                {"role": "system", "content": "You are a precise classifier."},
                {"role": "user", "content": CLASSIFICATION_PROMPT_TEMPLATE.format(query=query)}
            ],
            temperature=0,
            max_tokens=10
        )
        category = response.choices[0].message.content.strip().lower()
        if category not in ["factual", "exploratory", "ambiguous"]:
            return "exploratory"
        return category
    except Exception as e:
        print(f"Classification error: {e}")
        return "factual"


@measure_latency
def synthesize_answer(query: str, search_results: List[Dict]) -> Dict[str, Any]:
    client = get_llm_client()

    context_str = build_context_str(search_results)
    prompt = RAG_PROMPT_TEMPLATE.format(context_str=context_str, query=query)

    try:
        response = client.chat.completions.create(
            model=SYNTHESIS_MODEL,
            messages=[
                {"role": "system", "content": "You are a grounded QA assistant. Always cite your sources using [doc_id] format."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.0,
            max_tokens=500
        )
        answer = response.choices[0].message.content.strip()

        # Parse citations from the answer, validating against available doc IDs
        available_ids = {res["doc_id"] for res in search_results}
        cited_ids = extract_cited_doc_ids(answer, available_ids)

        # Filter to only actually cited sources
        citations_used = [
            res for res in search_results
            if res["doc_id"] in cited_ids
        ]

        # If no citations were parsed but we have results, include top result
        # (fallback for when LLM doesn't follow citation format)
        if not citations_used and search_results:
            citations_used = search_results[:1]

        # Calculate real confidence score
        confidence = calculate_confidence(answer, search_results, cited_ids)

        return {
            "answer": answer,
            "citations_used": citations_used,
            "confidence": confidence
        }
    except Exception as e:
        print(f"Synthesis error: {e}")
        return {
            "answer": "Error generating answer.",
            "citations_used": [],
            "confidence": 0.0
        }


# ============= ASYNC VERSIONS =============

async def classify_query_async(query: str) -> str:
    """Async version of classify_query for non-blocking API calls."""
    client = get_async_llm_client()
    try:
        response = await client.chat.completions.create(
            model=CLASSIFICATION_MODEL,
            messages=[
                {"role": "system", "content": "You are a precise classifier."},
                {"role": "user", "content": CLASSIFICATION_PROMPT_TEMPLATE.format(query=query)}
            ],
            temperature=0,
            max_tokens=10
        )
        category = response.choices[0].message.content.strip().lower()
        if category not in ["factual", "exploratory", "ambiguous"]:
            return "exploratory"
        return category
    except Exception as e:
        print(f"Classification error: {e}")
        return "factual"


async def synthesize_answer_async(query: str, search_results: List[Dict]) -> Dict[str, Any]:
    """Async version of synthesize_answer for non-blocking API calls."""
    client = get_async_llm_client()

    context_str = build_context_str(search_results)
    prompt = RAG_PROMPT_TEMPLATE.format(context_str=context_str, query=query)

    try:
        response = await client.chat.completions.create(
            model=SYNTHESIS_MODEL,
            messages=[
                {"role": "system", "content": "You are a grounded QA assistant. Always cite your sources using [doc_id] format."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.0,
            max_tokens=500
        )
        answer = response.choices[0].message.content.strip()

        # Parse citations from the answer, validating against available doc IDs
        available_ids = {res["doc_id"] for res in search_results}
        cited_ids = extract_cited_doc_ids(answer, available_ids)

        # Filter to only actually cited sources
        citations_used = [
            res for res in search_results
            if res["doc_id"] in cited_ids
        ]

        # Fallback if no citations parsed
        if not citations_used and search_results:
            citations_used = search_results[:1]

        # Calculate real confidence score
        confidence = calculate_confidence(answer, search_results, cited_ids)

        return {
            "answer": answer,
            "citations_used": citations_used,
            "confidence": confidence
        }
    except Exception as e:
        print(f"Synthesis error: {e}")
        return {
            "answer": "Error generating answer.",
            "citations_used": [],
            "confidence": 0.0
        }
