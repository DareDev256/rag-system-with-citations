import asyncio
import dataclasses
import logging
import os
import re
import threading
from urllib.parse import urlparse
import openai
from typing import List, Dict, Any, Optional, Set
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
_async_client_lock = None  # Lazy init — asyncio.Lock() requires a running event loop


# ─── SSRF Prevention (CWE-918) ──────────────────────────────────────
_ALLOWED_SCHEMES = {"https", "http"}
_BLOCKED_HOSTS = frozenset({
    "169.254.169.254",       # AWS/GCP metadata
    "metadata.google.internal",  # GCP metadata
    "100.100.100.200",       # Alibaba Cloud metadata
})


def _validate_base_url(url: str) -> str:
    """Validate OPENAI_BASE_URL to prevent SSRF attacks.

    Only allows http:// and https:// schemes; blocks known cloud metadata
    endpoints and rejects file://, ftp://, gopher://, etc.

    Raises:
        ValueError: If the URL uses a disallowed scheme or targets a blocked host.
    """
    parsed = urlparse(url)
    if parsed.scheme not in _ALLOWED_SCHEMES:
        raise ValueError(
            f"OPENAI_BASE_URL scheme must be http or https, got '{parsed.scheme}'"
        )
    hostname = parsed.hostname or ""
    if hostname in _BLOCKED_HOSTS:
        raise ValueError(
            f"OPENAI_BASE_URL targets a blocked metadata endpoint: {hostname}"
        )
    if not hostname:
        raise ValueError("OPENAI_BASE_URL has no hostname")
    return url


def _client_kwargs() -> dict:
    """Build shared kwargs for OpenAI client initialization.

    Supports OPENAI_BASE_URL for proxy compatibility — any OpenAI-compatible
    proxy (LiteLLM, OpenRouter, enterprise gateways) works by setting this
    env var alongside OPENAI_API_KEY. The URL is validated against an
    allowlist of schemes and a blocklist of cloud metadata endpoints to
    prevent SSRF (CWE-918).
    """
    kwargs = {}
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        logger.warning("OPENAI_API_KEY not found in env.")
    kwargs["api_key"] = api_key

    base_url = os.getenv("OPENAI_BASE_URL")
    if base_url:
        kwargs["base_url"] = _validate_base_url(base_url)

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
    global _async_llm_client, _async_client_lock
    if _async_llm_client is not None:
        return _async_llm_client
    # Lazy-init the lock inside the running event loop to avoid binding
    # to a stale or non-existent loop at module import time (Python 3.10+
    # deprecation → 3.12+ RuntimeError).
    if _async_client_lock is None:
        _async_client_lock = asyncio.Lock()
    async with _async_client_lock:
        if _async_llm_client is None:
            _async_llm_client = openai.AsyncOpenAI(**_client_kwargs(), timeout=_LLM_TIMEOUT)
    return _async_llm_client


CITATION_PATTERN = re.compile(r'\[([^\]]+)\]')

# ─── Prompt Constants ──────────────────────────────────────────────────
# Surfaced as module-level constants so prompt engineering changes are
# visible in diffs and reviewable without reading through helper functions.

CLASSIFIER_SYSTEM_PROMPT = "You are a precise classifier."

SYNTHESIS_SYSTEM_PROMPT = (
    "You are a grounded QA assistant. "
    "Always cite your sources using [doc_id] format."
)


def get_available_doc_ids(search_results: List[Dict]) -> Set[str]:
    """Extract the set of valid (non-None) doc_ids from search results.

    Centralizes the doc_id extraction logic used across the synthesis and
    diagnostics pipeline — confidence scoring, citation filtering, and
    hallucination detection all need this same set.  Filters out None/empty
    doc_ids so downstream comparisons don't false-match on broken metadata.
    """
    return {str(res["doc_id"]) for res in search_results if res.get("doc_id") is not None}


def extract_cited_doc_ids(answer: str, available_ids: Optional[Set[str]] = None) -> Set[str]:
    """Extract all [doc_id] citations from the answer text.

    If available_ids is provided, only returns IDs that match actual
    documents from search results, filtering out hallucinated citations.
    """
    matches = CITATION_PATTERN.findall(answer)
    cited = set(matches)
    if available_ids is not None:
        cited = cited & available_ids
    return cited


@dataclasses.dataclass(frozen=True, slots=True)
class CitationAnalysis:
    """Pre-computed citation metrics for an answer + search-results pair.

    Consolidates the citation set math that was previously duplicated across
    ``_parse_synthesis`` and ``build_diagnostics`` — available IDs, cited IDs,
    valid/hallucinated partitions, and coverage ratio are all computed once.
    """
    available_ids: frozenset
    all_cited_ids: frozenset
    valid_cited_ids: frozenset
    hallucinated_ids: frozenset
    coverage: float


def analyze_citations(answer: str, search_results: List[Dict]) -> CitationAnalysis:
    """Run full citation analysis in a single pass.

    Returns a frozen dataclass with every citation metric the pipeline needs,
    eliminating redundant ``get_available_doc_ids`` / ``extract_cited_doc_ids``
    calls across the synthesis and diagnostics code paths.
    """
    available = get_available_doc_ids(search_results)
    all_cited = extract_cited_doc_ids(answer)
    valid = all_cited & available
    hallucinated = all_cited - available
    coverage = round(len(valid) / len(available), 2) if available else 0.0
    return CitationAnalysis(
        available_ids=frozenset(available),
        all_cited_ids=frozenset(all_cited),
        valid_cited_ids=frozenset(valid),
        hallucinated_ids=frozenset(hallucinated),
        coverage=coverage,
    )


def calculate_confidence(
    answer: str,
    search_results: List[Dict],
    cited_ids: Set[str],
    *,
    _available_ids: Optional[Set[str]] = None,
) -> float:
    """
    Calculate confidence score based on:
    - Citation coverage: % of retrieved docs that were cited
    - Grounding check: Whether the answer uses citations at all
    - Refusal detection: Lower confidence if LLM refused to answer

    The optional ``_available_ids`` kwarg lets callers that have already
    computed the valid doc-id set (e.g. ``_parse_synthesis``) skip the
    redundant recomputation.
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

    # Reuse pre-computed set when available; otherwise compute on the fly
    available_ids = _available_ids if _available_ids is not None else get_available_doc_ids(search_results)
    valid_citations = cited_ids & available_ids

    if not valid_citations:
        return 0.3  # Citations don't match available docs

    # Explicit guard: all search results had None/empty doc_ids.
    # (valid_citations check above catches this transitively, but an
    # explicit guard prevents regressions if the logic is reordered.)
    if not available_ids:
        return 0.3

    # Base confidence from citation ratio — use available_ids (not search_results)
    # so broken metadata entries with None doc_ids don't deflate the score.
    citation_ratio = len(valid_citations) / len(available_ids)

    # Scale: at least 1 citation = 0.6, all cited = 1.0
    confidence = 0.6 + (0.4 * citation_ratio)

    return round(confidence, 2)


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

    # Fallback: include top result when LLM doesn't follow citation format
    if not citations_used and search_results:
        citations_used = search_results[:1]

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


_SYNTHESIS_ERROR = {"answer": "Error generating answer.", "citations_used": [], "confidence": 0.0}


# ─── LLM call helpers (sync/async DRY) ───────────────────────────────

def _call_llm(client: Any, model: str, messages: List[Dict[str, str]], **kwargs: Any) -> Any:
    """Sync LLM call — single entry point for all blocking OpenAI requests."""
    return client.chat.completions.create(model=model, messages=messages, **kwargs)


async def _call_llm_async(client: Any, model: str, messages: List[Dict[str, str]], **kwargs: Any) -> Any:
    """Async LLM call — single entry point for all non-blocking OpenAI requests."""
    return await client.chat.completions.create(model=model, messages=messages, **kwargs)


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
        lambda r: _parse_synthesis(r, search_results), dict(_SYNTHESIS_ERROR),
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
        lambda r: _parse_synthesis(r, search_results), dict(_SYNTHESIS_ERROR),
        "synthesis", temperature=0.0, max_tokens=500,
    )
