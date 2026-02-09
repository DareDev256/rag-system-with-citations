"""Edge-case tests for RAG system core functions.

Covers: Unicode handling, missing keys, boundary values, exception propagation,
type coercion, and error paths not reached by existing test suites.
"""
import pytest
from pydantic import ValidationError

from src.llm.synthesize import extract_cited_doc_ids, calculate_confidence
from src.llm.prompt import build_context_str
from src.eval.metrics import calculate_citation_coverage, estimate_hallucination_rate
from src.utils.timing import measure_latency
from src.api.schemas import QueryRequest, Citation


# ── extract_cited_doc_ids: edge cases ────────────────────────────

class TestExtractCitedDocIdsEdgeCases:
    def test_unicode_doc_ids(self):
        assert extract_cited_doc_ids("Answer [日本語_001] about Japan.") == {"日本語_001"}

    def test_special_characters_in_brackets(self):
        result = extract_cited_doc_ids("See [doc-001] and [doc_001#section].")
        assert "doc-001" in result
        assert "doc_001#section" in result

    def test_available_ids_empty_set_filters_all(self):
        result = extract_cited_doc_ids("See [doc_001].", available_ids=set())
        assert result == set()

    def test_unclosed_bracket_greedy_match(self):
        """Regex is greedy: [x incomplete and [y] captures everything between first [ and ]."""
        result = extract_cited_doc_ids("See [doc_001 incomplete and [doc_002] valid.")
        assert "doc_001 incomplete and [doc_002" in result

    def test_brackets_with_spaces(self):
        assert extract_cited_doc_ids("See [doc 001] ref.") == {"doc 001"}

    def test_empty_string_input(self):
        assert extract_cited_doc_ids("") == set()

    def test_only_empty_brackets(self):
        assert extract_cited_doc_ids("[]") == set()

    def test_numeric_doc_ids(self):
        assert extract_cited_doc_ids("See [123] and [456].") == {"123", "456"}


# ── calculate_confidence: edge cases ─────────────────────────────

class TestCalculateConfidenceEdgeCases:
    def _r(self, *ids):
        return [{"doc_id": d} for d in ids]

    def test_refusal_case_insensitive(self):
        assert calculate_confidence("I CANNOT ANSWER that.", self._r("d1"), set()) == 0.1

    def test_partial_refusal_phrase_still_triggers(self):
        answer = "Unfortunately I cannot answer this question based on context."
        assert calculate_confidence(answer, self._r("d1"), set()) == 0.1

    def test_empty_string_answer_no_crash(self):
        assert calculate_confidence("", self._r("d1"), set()) == 0.3

    def test_single_result_single_citation_gives_max(self):
        assert calculate_confidence("[d1]", self._r("d1"), {"d1"}) == 1.0

    def test_mixed_valid_and_hallucinated_citations(self):
        conf = calculate_confidence("[d1] [d9]", self._r("d1", "d2"), {"d1", "d9"})
        assert conf == 0.8  # valid={d1}, ratio=1/2 → 0.6+0.2

    def test_empty_cited_ids_set(self):
        assert calculate_confidence("Answer.", self._r("d1"), set()) == 0.3


# ── build_context_str: edge cases ────────────────────────────────

class TestBuildContextStrEdgeCases:
    def test_missing_doc_id_key_raises_keyerror(self):
        with pytest.raises(KeyError):
            build_context_str([{"snippet": "text only"}])

    def test_missing_snippet_key_raises_keyerror(self):
        with pytest.raises(KeyError):
            build_context_str([{"doc_id": "doc_001"}])

    def test_unicode_in_snippets(self):
        assert build_context_str([{"doc_id": "d1", "snippet": "日本語テキスト"}]) == "[d1] 日本語テキスト"

    def test_none_values_become_string_none(self):
        assert build_context_str([{"doc_id": None, "snippet": None}]) == "[None] None"

    def test_empty_snippet(self):
        assert build_context_str([{"doc_id": "d1", "snippet": ""}]) == "[d1] "

    def test_snippet_with_newlines(self):
        assert "line1\nline2" in build_context_str([{"doc_id": "d1", "snippet": "line1\nline2"}])


# ── calculate_citation_coverage: edge cases ──────────────────────

class TestCitationCoverageEdgeCases:
    def test_missing_doc_id_key_skipped(self):
        """dict.get('doc_id') returns None → skipped, not matched as '[None]'."""
        assert calculate_citation_coverage("Has [None] in it.", [{"snippet": "x"}]) == 0.0

    def test_none_doc_id_skipped_gracefully(self):
        """Fixed: None doc_id is skipped instead of crashing."""
        result = calculate_citation_coverage("answer", [{"doc_id": None}])
        assert result == 0.0

    def test_none_doc_id_mixed_with_valid(self):
        """None doc_ids are skipped; valid ones still counted."""
        citations = [{"doc_id": None}, {"doc_id": "doc_001"}]
        result = calculate_citation_coverage("See [doc_001] here.", citations)
        assert result == 0.5

    def test_empty_doc_id_checks_for_empty_brackets(self):
        assert calculate_citation_coverage("Has [] here.", [{"doc_id": ""}]) == 1.0

    def test_unicode_doc_id_coverage(self):
        assert calculate_citation_coverage("See [日本語_001].", [{"doc_id": "日本語_001"}]) == 1.0

    def test_doc_id_without_brackets_still_matches(self):
        """Coverage checks both [doc_id] and bare doc_id."""
        assert calculate_citation_coverage("The doc_001 says so.", [{"doc_id": "doc_001"}]) == 1.0


# ── measure_latency: edge cases ──────────────────────────────────

class TestMeasureLatencyEdgeCases:
    def test_exception_propagates_through_decorator(self):
        @measure_latency
        def raises_error():
            raise ValueError("boom")
        with pytest.raises(ValueError, match="boom"):
            raises_error()

    def test_integer_result_not_modified(self):
        @measure_latency
        def returns_int():
            return 42
        assert returns_int() == 42

    def test_list_result_not_modified(self):
        @measure_latency
        def returns_list():
            return [1, 2, 3]
        assert returns_list() == [1, 2, 3]

    def test_empty_dict_gets_latency_injected(self):
        @measure_latency
        def returns_empty_dict():
            return {}
        assert "latency_ms" in returns_empty_dict()

    def test_bool_result_not_modified(self):
        @measure_latency
        def returns_bool():
            return True
        assert returns_bool() is True


# ── Schema validation: edge cases ────────────────────────────────

class TestSchemaEdgeCases:
    def test_query_request_int_query_coerced_or_rejected(self):
        try:
            req = QueryRequest(query=123)
            assert isinstance(req.query, str)
        except ValidationError:
            pass  # Strict mode rejects non-str

    def test_query_request_max_length_boundary(self):
        req = QueryRequest(query="x" * 1000)
        assert len(req.query) == 1000

    def test_citation_score_accepts_zero(self):
        assert Citation(doc_id="d", snippet="s", score=0.0).score == 0.0

    def test_citation_score_accepts_negative(self):
        """No validator prevents negative scores — documenting behavior."""
        assert Citation(doc_id="d", snippet="s", score=-1.0).score == -1.0


# ── estimate_hallucination_rate: tests ───────────────────────────

class TestEstimateHallucinationRate:
    def test_fully_grounded(self):
        """Answer words all appear in context → 0.0 hallucination."""
        assert estimate_hallucination_rate("Python is great", "Python is great") == 0.0

    def test_fully_hallucinated(self):
        """No answer words appear in context → 1.0 hallucination."""
        assert estimate_hallucination_rate("quantum entanglement", "Python Flask API") == 1.0

    def test_partial_overlap(self):
        """Some words grounded, some novel → between 0 and 1."""
        rate = estimate_hallucination_rate("Python quantum computing", "Python Flask API")
        assert 0.0 < rate < 1.0

    def test_empty_answer_returns_zero(self):
        assert estimate_hallucination_rate("", "some context") == 0.0

    def test_whitespace_only_answer_returns_zero(self):
        assert estimate_hallucination_rate("   ", "some context") == 0.0

    def test_empty_context_returns_one(self):
        """No context to ground against → all words are novel."""
        assert estimate_hallucination_rate("hello world", "") == 1.0

    def test_none_context_returns_one(self):
        """None context treated as empty → all words novel."""
        assert estimate_hallucination_rate("hello world", None) == 1.0

    def test_stop_words_ignored(self):
        """Stop words should not count toward overlap or novelty."""
        rate = estimate_hallucination_rate("the is a an", "some context")
        assert rate == 0.0  # Only stop words → nothing to measure

    def test_case_insensitive(self):
        assert estimate_hallucination_rate("PYTHON FLASK", "python flask api") == 0.0
