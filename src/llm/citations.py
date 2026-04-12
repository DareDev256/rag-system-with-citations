"""Citation analysis — pattern matching, set math, and confidence scoring.

Provides the citation extraction pipeline used across synthesis, diagnostics,
and evaluation: parse ``[doc_id]`` references from LLM output, partition them
into valid/hallucinated sets against retrieved documents, compute coverage
ratios and confidence scores.

Extracted from ``synthesize.py`` so citation logic is independently importable
without pulling in LLM client infrastructure.
"""

import dataclasses
import re
from typing import Dict, List, Optional, Set


CITATION_PATTERN = re.compile(r'\[([^\]]+)\]')


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
