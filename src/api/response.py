"""Response builders for the /query endpoint.

Extracts citation assembly and diagnostics computation from the endpoint
handler so they're independently testable and the handler reads as a
clean pipeline orchestrator.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Dict, List, Optional

from src.api.schemas import Citation, Diagnostics
from src.eval.metrics import estimate_hallucination_rate, calculate_answer_quality
from src.llm.prompt import build_context_str
from src.llm.synthesize import analyze_citations

if TYPE_CHECKING:
    from src.llm.synthesize import CitationAnalysis


def _sanitize_field(text, sanitize_fn) -> str:
    """Apply output sanitizer to a single field, handling None and non-str types.

    JSON metadata can contain non-string values (e.g. integer doc_ids).
    Coerces to str before sanitizing to prevent TypeError in re.sub().
    Uses explicit ``is None`` instead of falsy check so valid values
    like ``0`` are preserved rather than silently dropped.
    """
    if text is None:
        return ""
    return sanitize_fn(str(text))


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
    *,
    citation_analysis: Optional[CitationAnalysis] = None,
) -> Diagnostics:
    """Compute retrieval diagnostics: coverage, hallucination, quality, timing.

    When *citation_analysis* is supplied (e.g. from the synthesis result),
    skips the redundant ``analyze_citations`` call — same pure function,
    same inputs, no reason to pay twice.
    """
    analysis = citation_analysis or analyze_citations(answer, search_results)
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
