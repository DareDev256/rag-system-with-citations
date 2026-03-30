"""Response builders for the /query endpoint.

Extracts citation assembly and diagnostics computation from the endpoint
handler so they're independently testable and the handler reads as a
clean pipeline orchestrator.
"""

from typing import Dict, List, Optional

from src.api.schemas import Citation, Diagnostics
from src.eval.metrics import estimate_hallucination_rate, calculate_answer_quality
from src.llm.prompt import build_context_str
from src.llm.synthesize import analyze_citations


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
            doc_id=_sanitize_field(res.get("doc_id"), sanitize_fn),
            snippet=_sanitize_field(res.get("snippet"), sanitize_fn),
            score=res.get("score"),
            source=_sanitize_field(res.get("source"), sanitize_fn) or None,
        )
        for res in synthesis_result.get("citations_used", [])
    ]


def build_diagnostics(
    search_results: List[Dict],
    answer: str,
    retrieval_ms: float,
    synthesis_ms: float,
    confidence: float,
) -> Diagnostics:
    """Compute retrieval diagnostics: coverage, hallucination, quality, timing.

    Separated from the endpoint so diagnostics logic can be tested without
    standing up the full HTTP stack.
    """
    analysis = analyze_citations(answer, search_results)
    context_str = build_context_str(search_results)
    hallucination_rate = estimate_hallucination_rate(answer, context_str)
    quality = calculate_answer_quality(analysis.coverage, hallucination_rate, confidence)

    return Diagnostics(
        retrieval_ms=round(retrieval_ms, 2),
        synthesis_ms=round(synthesis_ms, 2),
        documents_searched=len(search_results),
        citation_coverage=analysis.coverage,
        hallucinated_citations=sorted(analysis.hallucinated_ids),
        hallucination_rate=hallucination_rate,
        answer_quality_score=quality,
    )
