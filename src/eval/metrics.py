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
    # A simple heuristic: check if citation IDs (e.g., [doc_1]) appear in the answer.
    # Returns % of provided citations that are actually used in the answer text.
    if not citations:
        return 0.0

    used_count = 0
    for cit in citations:
        doc_id = cit.get("doc_id")
        if doc_id is None:
            continue
        # Check for [doc_id] or just the id if prompt format varies
        if f"[{doc_id}]" in answer or doc_id in answer:
            used_count += 1

    return used_count / len(citations)


def estimate_hallucination_rate(answer: str, context_str: str) -> float:
    """Estimate hallucination via word-overlap heuristic.

    Tokenises both strings, removes stop words, and returns the fraction
    of answer content words that do NOT appear in the context.
    Returns 0.0 (fully grounded) to 1.0 (fully hallucinated).
    """
    if not answer or not answer.strip():
        return 0.0

    def _tokenize(text: str) -> set:
        return {w for w in re.findall(r"[a-z0-9]+", text.lower()) if w not in _STOP_WORDS}

    answer_words = _tokenize(answer)
    if not answer_words:
        return 0.0

    context_words = _tokenize(context_str) if context_str else set()
    novel = answer_words - context_words
    return round(len(novel) / len(answer_words), 2)
