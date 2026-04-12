"""Tests for recently refactored code paths that lack direct coverage.

Targets:
- analyze_citations() — frozen CitationAnalysis dataclass (5fe4332)
- _build_synthesis_prompt() — extracted prompt builder (7b8971e)
- _parse_synthesis _citation_analysis threading (5fe4332)
- build_diagnostics citation_analysis kwarg optimization path
- _validate_dimension no-op when index is None (03abcae)
- Prompt constants stability (7b8971e)
"""

import pytest
from unittest.mock import MagicMock

from src.llm.citations import analyze_citations, CitationAnalysis
from src.llm.synthesize import (
    _build_synthesis_prompt,
    _parse_synthesis,
    CLASSIFIER_SYSTEM_PROMPT,
    SYNTHESIS_SYSTEM_PROMPT,
    _classification_messages,
    _synthesis_messages,
)
from src.api.response import build_diagnostics
from src.retrieval.vector_store import VectorStore


# ── helpers ──────────────────────────────────────────────────────

def _make_response(content: str):
    """Minimal OpenAI-shaped response object."""
    msg = MagicMock()
    msg.content = content
    choice = MagicMock()
    choice.message = msg
    resp = MagicMock()
    resp.choices = [choice]
    return resp


_TWO_DOCS = [
    {"doc_id": "a", "snippet": "Alpha content."},
    {"doc_id": "b", "snippet": "Beta content."},
]

_THREE_DOCS = [
    {"doc_id": "x", "snippet": "X text."},
    {"doc_id": "y", "snippet": "Y text."},
    {"doc_id": "z", "snippet": "Z text."},
]


# ── analyze_citations (direct unit tests) ────────────────────────

class TestAnalyzeCitations:
    """Direct tests for the CitationAnalysis factory — previously only
    tested indirectly through _parse_synthesis and build_diagnostics."""

    def test_all_cited(self):
        ca = analyze_citations("See [a] and [b].", _TWO_DOCS)
        assert ca.available_ids == frozenset({"a", "b"})
        assert ca.valid_cited_ids == frozenset({"a", "b"})
        assert ca.hallucinated_ids == frozenset()
        assert ca.coverage == 1.0

    def test_partial_citation(self):
        ca = analyze_citations("Only [x].", _THREE_DOCS)
        assert ca.valid_cited_ids == frozenset({"x"})
        assert ca.hallucinated_ids == frozenset()
        assert ca.coverage == pytest.approx(0.33, abs=0.01)

    def test_hallucinated_ids_partitioned(self):
        ca = analyze_citations("[a] and [phantom].", _TWO_DOCS)
        assert "phantom" in ca.hallucinated_ids
        assert "phantom" not in ca.valid_cited_ids
        assert "a" in ca.valid_cited_ids

    def test_no_citations_in_answer(self):
        ca = analyze_citations("Plain text answer.", _TWO_DOCS)
        assert ca.all_cited_ids == frozenset()
        assert ca.valid_cited_ids == frozenset()
        assert ca.coverage == 0.0

    def test_empty_search_results(self):
        ca = analyze_citations("[ghost].", [])
        assert ca.available_ids == frozenset()
        assert ca.hallucinated_ids == frozenset({"ghost"})
        assert ca.coverage == 0.0

    def test_none_doc_ids_excluded_from_available(self):
        """None doc_ids must not pollute available_ids — regression guard."""
        results = [{"doc_id": None, "snippet": "s"}, {"doc_id": "real", "snippet": "s"}]
        ca = analyze_citations("[real].", results)
        assert ca.available_ids == frozenset({"real"})
        assert ca.coverage == 1.0

    def test_integer_doc_ids_coerced_to_str(self):
        """JSON metadata often has integer doc_ids — must match string citations."""
        results = [{"doc_id": 1, "snippet": "s"}, {"doc_id": 2, "snippet": "s"}]
        ca = analyze_citations("[1] and [2].", results)
        assert ca.valid_cited_ids == frozenset({"1", "2"})
        assert ca.coverage == 1.0

    def test_frozen_dataclass_immutable(self):
        """CitationAnalysis must be frozen — mutation would corrupt shared state."""
        ca = analyze_citations("[a].", _TWO_DOCS)
        with pytest.raises(AttributeError):
            ca.coverage = 0.99

    def test_zero_doc_id_not_dropped(self):
        """Integer 0 is falsy but valid — must appear in available_ids."""
        results = [{"doc_id": 0, "snippet": "zero doc"}]
        ca = analyze_citations("[0].", results)
        assert "0" in ca.available_ids
        assert ca.coverage == 1.0


# ── _build_synthesis_prompt ──────────────────────────────────────

class TestBuildSynthesisPrompt:
    """Tests for the extracted prompt builder (7b8971e) — ensures context
    and query are correctly wired into the RAG prompt template."""

    def test_contains_query_and_context(self):
        prompt = _build_synthesis_prompt("What is FAISS?", _TWO_DOCS)
        assert "What is FAISS?" in prompt
        assert "Alpha content." in prompt
        assert "Beta content." in prompt

    def test_none_doc_ids_excluded_from_context(self):
        """build_context_str skips None doc_ids — prompt must not contain [None]."""
        results = [{"doc_id": None, "snippet": "ghost"}, {"doc_id": "ok", "snippet": "visible"}]
        prompt = _build_synthesis_prompt("q", results)
        assert "[None]" not in prompt
        assert "visible" in prompt

    def test_format_placeholder_in_query_not_expanded(self):
        """If the user query contains {context_str}, it must NOT cause
        double-replacement or KeyError — regression for format_rag_prompt."""
        prompt = _build_synthesis_prompt("{context_str}", _TWO_DOCS)
        assert "{context_str}" in prompt  # literal, not expanded

    def test_empty_results_produces_empty_context(self):
        prompt = _build_synthesis_prompt("q", [])
        assert "User Question: q" in prompt


# ── _parse_synthesis threads CitationAnalysis ────────────────────

class TestParseSynthesisCitationAnalysis:
    """Verify _parse_synthesis returns a _citation_analysis key (5fe4332)
    that build_diagnostics can reuse — eliminating redundant computation."""

    def test_citation_analysis_present_in_result(self):
        resp = _make_response("Answer from [a].")
        out = _parse_synthesis(resp, _TWO_DOCS)
        assert "_citation_analysis" in out
        assert isinstance(out["_citation_analysis"], CitationAnalysis)

    def test_citation_analysis_matches_answer(self):
        resp = _make_response("[x] and [z] say yes.")
        out = _parse_synthesis(resp, _THREE_DOCS)
        ca = out["_citation_analysis"]
        assert ca.valid_cited_ids == frozenset({"x", "z"})
        assert "y" not in ca.valid_cited_ids

    def test_citation_analysis_reusable_by_diagnostics(self):
        """The optimization path: passing pre-computed analysis to
        build_diagnostics should produce identical results."""
        answer = "Per [a] and [b], yes."
        resp = _make_response(answer)
        synthesis_out = _parse_synthesis(resp, _TWO_DOCS)
        ca = synthesis_out["_citation_analysis"]

        # With pre-computed analysis (optimization path)
        diag_fast = build_diagnostics(_TWO_DOCS, answer, 1.0, 2.0, 0.9, citation_analysis=ca)
        # Without (recomputation path)
        diag_slow = build_diagnostics(_TWO_DOCS, answer, 1.0, 2.0, 0.9)

        assert diag_fast.citation_coverage == diag_slow.citation_coverage
        assert diag_fast.hallucinated_citations == diag_slow.hallucinated_citations
        assert diag_fast.hallucination_rate == diag_slow.hallucination_rate


# ── Prompt constants stability ───────────────────────────────────

class TestPromptConstants:
    """Prompt constants (7b8971e) are module-level so they're diffable —
    verify they flow correctly into message builders."""

    def test_classifier_prompt_in_messages(self):
        msgs = _classification_messages("test query")
        assert msgs[0]["role"] == "system"
        assert msgs[0]["content"] == CLASSIFIER_SYSTEM_PROMPT

    def test_synthesis_prompt_in_messages(self):
        msgs = _synthesis_messages("user prompt here")
        assert msgs[0]["role"] == "system"
        assert msgs[0]["content"] == SYNTHESIS_SYSTEM_PROMPT
        assert msgs[1]["content"] == "user prompt here"

    def test_synthesis_prompt_mentions_citations(self):
        """The synthesis system prompt must instruct citation format."""
        assert "[doc_id]" in SYNTHESIS_SYSTEM_PROMPT


# ── _validate_dimension edge cases (03abcae) ─────────────────────

class TestValidateDimension:
    """Tests for the extracted dimension guard — ensures it's a no-op
    when no index exists and raises correctly on mismatch."""

    def test_no_index_passes_silently(self):
        """Before any index is created, validation must be a no-op."""
        store = VectorStore()
        assert store.index is None
        store._validate_dimension(128)  # should not raise

    def test_matching_dimension_passes(self):
        store = VectorStore()
        store.create_index(64)
        store._validate_dimension(64)  # should not raise

    def test_mismatched_dimension_raises(self):
        store = VectorStore()
        store.create_index(64)
        with pytest.raises(ValueError, match="dimension 128 does not match"):
            store._validate_dimension(128)
