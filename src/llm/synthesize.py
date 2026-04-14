import logging
from typing import List, Dict, Any

logger = logging.getLogger("rag_api")

from src.llm.citations import (
    CITATION_PATTERN,
    CitationAnalysis,
    analyze_citations,
    calculate_confidence,
    extract_cited_doc_ids,
    get_available_doc_ids,
)
from src.llm.client import (
    CLASSIFICATION_MODEL,
    SYNTHESIS_MODEL,
    LLM_TIMEOUT as _LLM_TIMEOUT,
    get_llm_client,
    get_async_llm_client,
    call_llm as _call_llm,
    call_llm_async as _call_llm_async,
    _validate_base_url,
    _client_kwargs,
)
from src.llm.prompt import format_rag_prompt, format_classification_prompt, build_context_str
from src.utils.timing import measure_latency

# ─── Prompt Constants ──────────────────────────────────────────────────
# Surfaced as module-level constants so prompt engineering changes are
# visible in diffs and reviewable without reading through helper functions.

CLASSIFIER_SYSTEM_PROMPT = "You are a precise classifier."

SYNTHESIS_SYSTEM_PROMPT = (
    "You are a grounded QA assistant. "
    "Always cite your sources using [doc_id] format."
)


# ─── Shared post-processing helpers (sync/async DRY) ────────────────

_VALID_CATEGORIES = {"factual", "exploratory", "ambiguous"}

def _classification_messages(query: str) -> List[Dict[str, str]]:
    return [
        {"role": "system", "content": CLASSIFIER_SYSTEM_PROMPT},
        {"role": "user", "content": format_classification_prompt(query)},
    ]


def _build_synthesis_prompt(query: str, search_results: List[Dict]) -> str:
    """Build the full RAG prompt from query and search results.

    Centralizes the context-building → prompt-formatting pipeline that
    was previously duplicated in both sync and async entry points.
    """
    return format_rag_prompt(build_context_str(search_results), query)


def _synthesis_messages(prompt: str) -> List[Dict[str, str]]:
    return [
        {"role": "system", "content": SYNTHESIS_SYSTEM_PROMPT},
        {"role": "user", "content": prompt},
    ]


def _parse_classification(response: Any) -> str:
    """Parse LLM classification response into a valid category."""
    category = response.choices[0].message.content.strip().lower()
    return category if category in _VALID_CATEGORIES else "exploratory"


def _parse_synthesis(response: Any, search_results: List[Dict]) -> Dict[str, Any]:
    """Parse LLM synthesis response into answer dict with citations and confidence."""
    answer = response.choices[0].message.content.strip()

    analysis = analyze_citations(answer, search_results)

    citations_used = [
        res for res in search_results
        if res.get("doc_id") is not None and str(res["doc_id"]) in analysis.valid_cited_ids
    ]

    # Fallback: include first result with a valid doc_id when LLM
    # doesn't follow citation format.  Skips None doc_ids so the API
    # never returns a Citation with an empty-string identifier.
    if not citations_used:
        for res in search_results:
            if res.get("doc_id") is not None:
                citations_used = [res]
                break

    confidence = calculate_confidence(
        answer, search_results, analysis.valid_cited_ids,
        _available_ids=analysis.available_ids,
    )

    return {
        "answer": answer,
        "citations_used": citations_used,
        "confidence": confidence,
        "_citation_analysis": analysis,
    }


def _synthesis_error() -> Dict[str, Any]:
    """Return a fresh error-fallback dict on every call.

    Avoids sharing a mutable ``[]`` reference across all error responses —
    ``dict(constant)`` is a shallow copy, so the inner list would be the
    same object, creating a latent mutation bug if any downstream code
    appends to ``citations_used``.
    """
    return {"answer": "Error generating answer.", "citations_used": [], "confidence": 0.0}


def _safe_llm_call(model: str, messages: List[Dict[str, str]], parser, fallback, label: str, **kwargs: Any):
    """Sync LLM call with standardized error handling.

    Encapsulates the get-client → call → parse → catch pattern shared by
    all sync LLM entry points.  On failure, logs the exception type and
    returns *fallback* so the caller never sees a raw exception.
    """
    client = get_llm_client()
    try:
        response = _call_llm(client, model, messages, **kwargs)
        return parser(response)
    except Exception as e:
        logger.error("%s failed: %s", label, type(e).__name__)
        return fallback


async def _safe_llm_call_async(model: str, messages: List[Dict[str, str]], parser, fallback, label: str, **kwargs: Any):
    """Async LLM call with standardized error handling.

    Async mirror of :func:`_safe_llm_call` — identical contract, uses the
    async client and awaitable call path.
    """
    client = await get_async_llm_client()
    try:
        response = await _call_llm_async(client, model, messages, **kwargs)
        return parser(response)
    except Exception as e:
        logger.error("Async %s failed: %s", label, type(e).__name__)
        return fallback


# ─── Sync API ────────────────────────────────────────────────────────

@measure_latency
def classify_query(query: str) -> str:
    return _safe_llm_call(
        CLASSIFICATION_MODEL, _classification_messages(query),
        _parse_classification, "exploratory", "classification",
        temperature=0, max_tokens=10,
    )


@measure_latency
def synthesize_answer(query: str, search_results: List[Dict]) -> Dict[str, Any]:
    prompt = _build_synthesis_prompt(query, search_results)
    return _safe_llm_call(
        SYNTHESIS_MODEL, _synthesis_messages(prompt),
        lambda r: _parse_synthesis(r, search_results), _synthesis_error(),
        "synthesis", temperature=0.0, max_tokens=500,
    )


# ─── Async API ───────────────────────────────────────────────────────

async def classify_query_async(query: str) -> str:
    """Async version of classify_query for non-blocking API calls."""
    return await _safe_llm_call_async(
        CLASSIFICATION_MODEL, _classification_messages(query),
        _parse_classification, "exploratory", "classification",
        temperature=0, max_tokens=10,
    )


async def synthesize_answer_async(query: str, search_results: List[Dict]) -> Dict[str, Any]:
    """Async version of synthesize_answer for non-blocking API calls."""
    prompt = _build_synthesis_prompt(query, search_results)
    return await _safe_llm_call_async(
        SYNTHESIS_MODEL, _synthesis_messages(prompt),
        lambda r: _parse_synthesis(r, search_results), _synthesis_error(),
        "synthesis", temperature=0.0, max_tokens=500,
    )
