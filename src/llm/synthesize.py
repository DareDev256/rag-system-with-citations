import asyncio
import logging
import os
import re
import threading
import openai
from typing import List, Dict, Any, Set
from src.llm.prompt import format_rag_prompt, format_classification_prompt, build_context_str
from src.utils.env import safe_int_env
from src.utils.timing import measure_latency

logger = logging.getLogger("rag_api")

# Configuration from environment
DEFAULT_MODEL = "gpt-4o-mini"
CLASSIFICATION_MODEL = os.getenv("CLASSIFICATION_MODEL", "gpt-4o-mini")
SYNTHESIS_MODEL = os.getenv("SYNTHESIS_MODEL", DEFAULT_MODEL)

# Request timeout in seconds — prevents resource exhaustion from hung upstreams
_LLM_TIMEOUT = safe_int_env("LLM_TIMEOUT", 30, min_val=1)

# Cached clients (sync and async) — guarded by locks to prevent
# duplicate initialization under concurrent access (CWE-362).
_llm_client = None
_async_llm_client = None
_sync_client_lock = threading.Lock()
_async_client_lock = asyncio.Lock()


def _client_kwargs() -> dict:
    """Build shared kwargs for OpenAI client initialization.

    Supports OPENAI_BASE_URL for proxy compatibility — any OpenAI-compatible
    proxy (LiteLLM, OpenRouter, enterprise gateways) works by setting this
    env var alongside OPENAI_API_KEY.
    """
    kwargs = {}
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("Warning: OPENAI_API_KEY not found in env.")
    kwargs["api_key"] = api_key

    base_url = os.getenv("OPENAI_BASE_URL")
    if base_url:
        kwargs["base_url"] = base_url

    return kwargs


def get_llm_client():
    global _llm_client
    if _llm_client is not None:
        return _llm_client
    with _sync_client_lock:
        # Double-check after acquiring lock — another thread may have initialized
        if _llm_client is None:
            _llm_client = openai.OpenAI(**_client_kwargs(), timeout=_LLM_TIMEOUT)
    return _llm_client


async def get_async_llm_client():
    global _async_llm_client
    if _async_llm_client is not None:
        return _async_llm_client
    async with _async_client_lock:
        if _async_llm_client is None:
            _async_llm_client = openai.AsyncOpenAI(**_client_kwargs(), timeout=_LLM_TIMEOUT)
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


# ─── Shared post-processing helpers (sync/async DRY) ────────────────

_VALID_CATEGORIES = {"factual", "exploratory", "ambiguous"}

_CLASSIFICATION_MESSAGES = lambda query: [
    {"role": "system", "content": "You are a precise classifier."},
    {"role": "user", "content": format_classification_prompt(query)},
]

_SYNTHESIS_MESSAGES = lambda prompt: [
    {"role": "system", "content": "You are a grounded QA assistant. Always cite your sources using [doc_id] format."},
    {"role": "user", "content": prompt},
]


def _parse_classification(response) -> str:
    """Parse LLM classification response into a valid category."""
    category = response.choices[0].message.content.strip().lower()
    return category if category in _VALID_CATEGORIES else "exploratory"


def _parse_synthesis(response, search_results: List[Dict]) -> Dict[str, Any]:
    """Parse LLM synthesis response into answer dict with citations and confidence."""
    answer = response.choices[0].message.content.strip()

    available_ids = {res["doc_id"] for res in search_results}
    cited_ids = extract_cited_doc_ids(answer, available_ids)

    citations_used = [res for res in search_results if res["doc_id"] in cited_ids]

    # Fallback: include top result when LLM doesn't follow citation format
    if not citations_used and search_results:
        citations_used = search_results[:1]

    confidence = calculate_confidence(answer, search_results, cited_ids)

    return {
        "answer": answer,
        "citations_used": citations_used,
        "confidence": confidence,
    }


_SYNTHESIS_ERROR = {"answer": "Error generating answer.", "citations_used": [], "confidence": 0.0}


# ─── LLM call helpers (sync/async DRY) ───────────────────────────────

def _call_llm(client, model: str, messages: list, **kwargs):
    """Sync LLM call — single entry point for all blocking OpenAI requests."""
    return client.chat.completions.create(model=model, messages=messages, **kwargs)


async def _call_llm_async(client, model: str, messages: list, **kwargs):
    """Async LLM call — single entry point for all non-blocking OpenAI requests."""
    return await client.chat.completions.create(model=model, messages=messages, **kwargs)


# ─── Sync API ────────────────────────────────────────────────────────

@measure_latency
def classify_query(query: str) -> str:
    client = get_llm_client()
    try:
        response = _call_llm(client, CLASSIFICATION_MODEL, _CLASSIFICATION_MESSAGES(query), temperature=0, max_tokens=10)
        return _parse_classification(response)
    except Exception as e:
        logger.error("Classification failed: %s", type(e).__name__)
        return "exploratory"


@measure_latency
def synthesize_answer(query: str, search_results: List[Dict]) -> Dict[str, Any]:
    client = get_llm_client()
    prompt = format_rag_prompt(build_context_str(search_results), query)
    try:
        response = _call_llm(client, SYNTHESIS_MODEL, _SYNTHESIS_MESSAGES(prompt), temperature=0.0, max_tokens=500)
        return _parse_synthesis(response, search_results)
    except Exception as e:
        logger.error("Synthesis failed: %s", type(e).__name__)
        return dict(_SYNTHESIS_ERROR)


# ─── Async API ───────────────────────────────────────────────────────

async def classify_query_async(query: str) -> str:
    """Async version of classify_query for non-blocking API calls."""
    client = await get_async_llm_client()
    try:
        response = await _call_llm_async(client, CLASSIFICATION_MODEL, _CLASSIFICATION_MESSAGES(query), temperature=0, max_tokens=10)
        return _parse_classification(response)
    except Exception as e:
        logger.error("Async classification failed: %s", type(e).__name__)
        return "exploratory"


async def synthesize_answer_async(query: str, search_results: List[Dict]) -> Dict[str, Any]:
    """Async version of synthesize_answer for non-blocking API calls."""
    client = await get_async_llm_client()
    prompt = format_rag_prompt(build_context_str(search_results), query)
    try:
        response = await _call_llm_async(client, SYNTHESIS_MODEL, _SYNTHESIS_MESSAGES(prompt), temperature=0.0, max_tokens=500)
        return _parse_synthesis(response, search_results)
    except Exception as e:
        logger.error("Async synthesis failed: %s", type(e).__name__)
        return dict(_SYNTHESIS_ERROR)
