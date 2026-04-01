"""Evaluation metrics for RAG answer quality.

Provides two complementary quality signals:

* **Citation coverage** — what fraction of provided citations actually
  appear in the answer text (measures citation discipline).
* **Hallucination rate** — what fraction of the answer's content words
  are *not* grounded in the retrieval context (measures factual drift).
"""

import re
from typing import List, Dict

# Words too common to signal hallucination
_STOP_WORDS = frozenset(
    "a an the is are was were be been being have has had do does did will "
    "would shall should may might can could of in to for on with at by from "
    "as into through during before after above below between out off over "
    "up down and but or nor not no so yet both either neither each every "
    "all any few more most other some such than too very it its this that "
    "these those i me my we our you your he him his she her they them their "
    "what which who whom whose when where why how if then else".split()
)


def calculate_citation_coverage(answer: str, citations: List[Dict]) -> float:
    """Return the fraction of *citations* whose ``doc_id`` appears in *answer*.

    Checks for both ``[doc_id]`` bracket notation and bare ``doc_id`` strings.
    Entries with ``None`` doc_id are silently skipped (don't inflate the
    denominator).  Returns 0.0 when *citations* is empty or all entries
    have ``None`` doc_ids.
    """
    if not citations:
        return 0.0

    used_count = 0
    valid_count = 0
    for cit in citations:
        doc_id = cit.get("doc_id")
        if doc_id is None:
            continue
        valid_count += 1
        # Check for [doc_id] or just the id if prompt format varies
        if f"[{doc_id}]" in answer or doc_id in answer:
            used_count += 1

    if valid_count == 0:
        return 0.0
    return used_count / valid_count


def _tokenize(text: str) -> set:
    """Extract content words (lowercase alphanumeric), excluding stop words.

    Module-level for reuse across metrics and independent testability.
    """
    return {w for w in re.findall(r"[a-z0-9]+", text.lower()) if w not in _STOP_WORDS}


def estimate_hallucination_rate(answer: str, context_str: str) -> float:
    """Estimate hallucination via word-overlap heuristic.

    Tokenises both strings, removes stop words, and returns the fraction
    of answer content words that do NOT appear in the context.
    Returns 0.0 (fully grounded) to 1.0 (fully hallucinated).
    """
    if not answer or not answer.strip():
        return 0.0

    answer_words = _tokenize(answer)
    if not answer_words:
        return 0.0

    context_words = _tokenize(context_str) if context_str else set()
    novel = answer_words - context_words
    return round(len(novel) / len(answer_words), 2)


def calculate_answer_quality(
    citation_coverage: float,
    hallucination_rate: float,
    confidence: float,
) -> float:
    """Compute a composite 0–1 answer quality score.

    Combines three independent signals into a single metric:
    - Citation coverage (40% weight) — are retrieved docs actually cited?
    - Inverse hallucination rate (35% weight) — is the answer grounded?
    - Confidence (25% weight) — does the citation ratio support the answer?

    Returns 0.0 for completely ungrounded answers, 1.0 for perfectly
    cited, fully grounded, high-confidence answers.
    """
    groundedness = max(0.0, 1.0 - hallucination_rate)
    raw = (0.40 * citation_coverage) + (0.35 * groundedness) + (0.25 * confidence)
    return round(min(1.0, max(0.0, raw)), 2)
