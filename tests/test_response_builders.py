"""Unit tests for src/api/response.py — citation assembly and diagnostics computation.

These functions sit between LLM output and the API response, so bugs here
directly corrupt what the client sees.  Every test runs without HTTP, FAISS,
or OpenAI — pure functions with deterministic inputs.
"""

import pytest

from src.api.response import _sanitize_field, build_citations, build_diagnostics
from src.api.schemas import Citation, Diagnostics


# ── helpers ────────────────────────────────────────────────────────

def _noop_sanitize(text: str) -> str:
    """Identity sanitizer for tests that don't care about sanitization."""
    return text


def _strip_control(text: str) -> str:
    """Realistic sanitizer that strips control chars (mirrors main.py)."""
    return "".join(ch for ch in text if ch in "\n\r\t" or (32 <= ord(ch) < 127))


_SEARCH_RESULTS = [
    {"doc_id": "doc_1", "snippet": "First result.", "score": 0.95, "source": "a.txt"},
    {"doc_id": "doc_2", "snippet": "Second result.", "score": 0.80, "source": "b.txt"},
    {"doc_id": "doc_3", "snippet": "Third result.", "score": 0.60},
]


# ── _sanitize_field ───────────────────────────────────────────────

class TestSanitizeField:

    def test_applies_fn_to_text(self):
        assert _sanitize_field("hello", str.upper) == "HELLO"

    def test_none_returns_empty(self):
        assert _sanitize_field(None, str.upper) == ""

    def test_empty_string_returns_empty(self):
        """Empty string is falsy — should return empty, not crash."""
        assert _sanitize_field("", str.upper) == ""

    def test_strips_control_chars(self):
        assert _sanitize_field("ok\x00bad\x01", _strip_control) == "okbad"


# ── build_citations ───────────────────────────────────────────────

class TestBuildCitations:

    def test_happy_path_returns_citation_objects(self):
        synthesis = {"citations_used": _SEARCH_RESULTS[:2]}
        result = build_citations(_SEARCH_RESULTS, synthesis, _noop_sanitize)
        assert len(result) == 2
        assert all(isinstance(c, Citation) for c in result)
        assert result[0].doc_id == "doc_1"
        assert result[1].snippet == "Second result."

    def test_preserves_score(self):
        synthesis = {"citations_used": [_SEARCH_RESULTS[0]]}
        result = build_citations(_SEARCH_RESULTS, synthesis, _noop_sanitize)
        assert result[0].score == 0.95

    def test_source_none_stays_none(self):
        """Result without 'source' key should produce Citation.source = None."""
        synthesis = {"citations_used": [_SEARCH_RESULTS[2]]}  # no source key
        result = build_citations(_SEARCH_RESULTS, synthesis, _noop_sanitize)
        assert result[0].source is None

    def test_empty_citations_used(self):
        synthesis = {"citations_used": []}
        assert build_citations(_SEARCH_RESULTS, synthesis, _noop_sanitize) == []

    def test_missing_citations_used_key(self):
        """Synthesis dict without 'citations_used' should not crash."""
        synthesis = {"answer": "no citations here"}
        assert build_citations(_SEARCH_RESULTS, synthesis, _noop_sanitize) == []

    def test_sanitizer_applied_to_corpus_fields(self):
        """Control chars in doc_id, snippet, source get stripped."""
        dirty_result = {
            "doc_id": "doc\x00_1",
            "snippet": "clean\x01text",
            "source": "file\x02.txt",
            "score": 0.9,
        }
        synthesis = {"citations_used": [dirty_result]}
        result = build_citations([], synthesis, _strip_control)
        assert result[0].doc_id == "doc_1"
        assert result[0].snippet == "cleantext"
        assert result[0].source == "file.txt"

    def test_sanitizer_not_applied_to_score(self):
        """Score is numeric — should pass through untouched."""
        synthesis = {"citations_used": [_SEARCH_RESULTS[0]]}
        result = build_citations(_SEARCH_RESULTS, synthesis, _strip_control)
        assert result[0].score == 0.95


# ── build_diagnostics ─────────────────────────────────────────────

class TestBuildDiagnostics:

    def test_full_coverage(self):
        """Answer cites all retrieved docs → coverage = 1.0."""
        answer = "Based on [doc_1] and [doc_2] and [doc_3], yes."
        diag = build_diagnostics(_SEARCH_RESULTS, answer, 12.345, 45.678)
        assert isinstance(diag, Diagnostics)
        assert diag.citation_coverage == 1.0
        assert diag.hallucinated_citations == []
        assert diag.documents_searched == 3

    def test_partial_coverage(self):
        answer = "According to [doc_1], yes."
        diag = build_diagnostics(_SEARCH_RESULTS, answer, 10.0, 20.0)
        assert diag.citation_coverage == pytest.approx(1 / 3, abs=0.01)

    def test_no_citations_zero_coverage(self):
        answer = "The sky is blue."
        diag = build_diagnostics(_SEARCH_RESULTS, answer, 5.0, 10.0)
        assert diag.citation_coverage == 0.0

    def test_hallucinated_citation_detected(self):
        """Citation not in search results is flagged as hallucinated."""
        answer = "Per [doc_1] and [phantom_doc], definitely."
        diag = build_diagnostics(_SEARCH_RESULTS, answer, 1.0, 2.0)
        assert "phantom_doc" in diag.hallucinated_citations
        assert "doc_1" not in diag.hallucinated_citations

    def test_hallucinated_sorted_alphabetically(self):
        answer = "[zebra_doc] and [alpha_doc] say so."
        diag = build_diagnostics(_SEARCH_RESULTS, answer, 1.0, 1.0)
        assert diag.hallucinated_citations == ["alpha_doc", "zebra_doc"]

    def test_empty_search_results(self):
        """No docs retrieved → coverage 0, everything is hallucinated."""
        answer = "[doc_1] says yes."
        diag = build_diagnostics([], answer, 0.0, 0.0)
        assert diag.citation_coverage == 0.0
        assert diag.documents_searched == 0
        assert "doc_1" in diag.hallucinated_citations

    def test_none_doc_ids_excluded_from_coverage_denominator(self):
        """None doc_ids must not inflate the denominator — regression for
        the same bug fixed in calculate_confidence (v1.10.1) but missed
        in build_diagnostics.  With 2 valid + 1 None doc_id, citing 1
        valid doc should yield 0.5, not 0.33."""
        results_with_none = [
            {"doc_id": "doc_1", "snippet": "s1", "score": 0.9, "source": "a.txt"},
            {"doc_id": None, "snippet": "broken", "score": 0.5},
            {"doc_id": "doc_2", "snippet": "s2", "score": 0.7, "source": "b.txt"},
        ]
        answer = "Per [doc_1], the answer is yes."
        diag = build_diagnostics(results_with_none, answer, 1.0, 2.0)
        # 1 cited out of 2 valid = 0.5  (NOT 1/3 ≈ 0.33)
        assert diag.citation_coverage == 0.5
        assert diag.documents_searched == 3  # total results unchanged

    def test_all_none_doc_ids_zero_coverage(self):
        """When every result has None doc_id, coverage should be 0.0,
        not a ZeroDivisionError."""
        results_all_none = [
            {"doc_id": None, "snippet": "s1", "score": 0.9},
            {"doc_id": None, "snippet": "s2", "score": 0.5},
        ]
        answer = "Some answer [doc_1]."
        diag = build_diagnostics(results_all_none, answer, 1.0, 1.0)
        assert diag.citation_coverage == 0.0
        assert "doc_1" in diag.hallucinated_citations

    def test_timing_rounded(self):
        diag = build_diagnostics(_SEARCH_RESULTS, "hello", 1.23456, 7.89012)
        assert diag.retrieval_ms == 1.23
        assert diag.synthesis_ms == 7.89


# ── _parse_classification edge cases ─────────────────────────────

class TestParseClassification:
    """Tests _parse_classification with realistic mock LLM responses."""

    def _make_response(self, content: str):
        """Build a minimal OpenAI-shaped response object."""
        class Msg:
            pass
        class Choice:
            pass
        class Resp:
            pass
        msg = Msg()
        msg.content = content
        choice = Choice()
        choice.message = msg
        resp = Resp()
        resp.choices = [choice]
        return resp

    def test_valid_categories(self):
        from src.llm.synthesize import _parse_classification
        for cat in ("factual", "exploratory", "ambiguous"):
            assert _parse_classification(self._make_response(cat)) == cat

    def test_uppercase_normalized(self):
        from src.llm.synthesize import _parse_classification
        assert _parse_classification(self._make_response("FACTUAL")) == "factual"

    def test_whitespace_stripped(self):
        from src.llm.synthesize import _parse_classification
        assert _parse_classification(self._make_response("  factual\n")) == "factual"

    def test_unknown_category_falls_back(self):
        from src.llm.synthesize import _parse_classification
        assert _parse_classification(self._make_response("opinion")) == "exploratory"

    def test_empty_response_falls_back(self):
        from src.llm.synthesize import _parse_classification
        assert _parse_classification(self._make_response("")) == "exploratory"


# ── _parse_synthesis edge cases ───────────────────────────────────

class TestParseSynthesis:
    """Tests _parse_synthesis with shaped mock responses — no LLM calls."""

    def _make_response(self, content: str):
        class Msg:
            pass
        class Choice:
            pass
        class Resp:
            pass
        msg = Msg()
        msg.content = content
        choice = Choice()
        choice.message = msg
        resp = Resp()
        resp.choices = [choice]
        return resp

    def test_cites_matching_doc(self):
        from src.llm.synthesize import _parse_synthesis
        results = [{"doc_id": "d1", "snippet": "s1"}, {"doc_id": "d2", "snippet": "s2"}]
        resp = self._make_response("Answer from [d1].")
        out = _parse_synthesis(resp, results)
        assert out["answer"] == "Answer from [d1]."
        assert len(out["citations_used"]) == 1
        assert out["citations_used"][0]["doc_id"] == "d1"
        assert out["confidence"] > 0.5

    def test_no_citations_falls_back_to_top_result(self):
        from src.llm.synthesize import _parse_synthesis
        results = [{"doc_id": "d1", "snippet": "s1"}]
        resp = self._make_response("Just a plain answer.")
        out = _parse_synthesis(resp, results)
        assert len(out["citations_used"]) == 1  # fallback
        assert out["citations_used"][0]["doc_id"] == "d1"
        assert out["confidence"] == 0.3  # no citations → low

    def test_hallucinated_citation_filtered_out(self):
        from src.llm.synthesize import _parse_synthesis
        results = [{"doc_id": "real", "snippet": "s"}]
        resp = self._make_response("Per [fake] and [real], yes.")
        out = _parse_synthesis(resp, results)
        cited_ids = {c["doc_id"] for c in out["citations_used"]}
        assert "fake" not in cited_ids
        assert "real" in cited_ids

    def test_empty_search_results(self):
        from src.llm.synthesize import _parse_synthesis
        resp = self._make_response("I know nothing.")
        out = _parse_synthesis(resp, [])
        assert out["confidence"] == 0.0
        assert out["citations_used"] == []

    def test_none_doc_id_in_results_skipped(self):
        from src.llm.synthesize import _parse_synthesis
        results = [{"doc_id": None, "snippet": "s"}, {"doc_id": "d1", "snippet": "s"}]
        resp = self._make_response("Per [d1], yes.")
        out = _parse_synthesis(resp, results)
        cited_ids = {c.get("doc_id") for c in out["citations_used"]}
        assert None not in cited_ids
