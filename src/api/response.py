"""Response builders for the /query endpoint.

Extracts citation assembly and diagnostics computation from the endpoint
handler so they're independently testable and the handler reads as a
clean pipeline orchestrator.
"""

from typing import Dict, List, Optional

from src.api.schemas import Citation, Diagnostics
from src.llm.synthesize import extract_cited_doc_ids, get_available_doc_ids


def _sanitize_field(text: str, sanitize_fn) -> str:
    """Apply output sanitizer to a single field, handling None gracefully."""
    return sanitize_fn(text) if text else ""


def build_citations(
    search_results: List[Dict],
    synthesis_result: Dict,
    sanitize_fn,
) -> List[Citation]:
    """Build sanitized Citation objects from synthesis results.

    Each corpus-sourced field (doc_id, snippet, source) passes through
    sanitize_fn to strip control characters before reaching the client.
    """
    return [
        Citation(
            doc_id=sanitize_fn(res["doc_id"]),
            snippet=sanitize_fn(res["snippet"]),
            score=res.get("score"),
            source=sanitize_fn(res["source"]) if res.get("source") else None,
        )
        for res in synthesis_result.get("citations_used", [])
    ]


def build_diagnostics(
    search_results: List[Dict],
    answer: str,
    retrieval_ms: float,
    synthesis_ms: float,
) -> Diagnostics:
    """Compute retrieval diagnostics: coverage, hallucinated citations, timing.

    Separated from the endpoint so diagnostics logic can be tested without
    standing up the full HTTP stack.
    """
    available_ids = get_available_doc_ids(search_results)
    all_cited = extract_cited_doc_ids(answer)
    hallucinated = sorted(all_cited - available_ids)
    valid_cited = all_cited & available_ids
    coverage = len(valid_cited) / len(search_results) if search_results else 0.0

    return Diagnostics(
        retrieval_ms=round(retrieval_ms, 2),
        synthesis_ms=round(synthesis_ms, 2),
        documents_searched=len(search_results),
        citation_coverage=round(coverage, 2),
        hallucinated_citations=hallucinated,
    )
