"""Unit tests for RAG system core logic.

Tests only pure functions — no FAISS, no API calls, no LLM calls.
"""
import pytest
from pydantic import ValidationError

from src.api.schemas import QueryRequest, QueryResponse, Citation
from src.llm.synthesize import extract_cited_doc_ids, calculate_confidence
from src.llm.prompt import build_context_str
from src.eval.metrics import calculate_citation_coverage, estimate_hallucination_rate


# ── build_context_str ────────────────────────────────────────────

class TestBuildContextStr:
    def test_formats_single_result(self):
        results = [{"doc_id": "doc_001", "snippet": "RAG is great."}]
        assert build_context_str(results) == "[doc_001] RAG is great."

    def test_formats_multiple_results(self):
        results = [
            {"doc_id": "doc_001", "snippet": "First chunk."},
            {"doc_id": "doc_002", "snippet": "Second chunk."},
        ]
        output = build_context_str(results)
        assert "[doc_001] First chunk." in output
        assert "[doc_002] Second chunk." in output
        assert "\n\n" in output

    def test_empty_results(self):
        assert build_context_str([]) == ""


# ── extract_cited_doc_ids ────────────────────────────────────────

class TestExtractCitedDocIds:
    def test_single_citation(self):
        assert extract_cited_doc_ids("Answer [doc_001].") == {"doc_001"}

    def test_multiple_citations(self):
        text = "See [doc_001] and [doc_002] for details."
        assert extract_cited_doc_ids(text) == {"doc_001", "doc_002"}

    def test_no_citations(self):
        assert extract_cited_doc_ids("No citations here.") == set()

    def test_duplicate_citations_deduplicated(self):
        text = "[doc_001] is cited twice [doc_001]."
        assert extract_cited_doc_ids(text) == {"doc_001"}

    def test_empty_brackets_ignored(self):
        # The regex r'\[([^\]]+)\]' requires 1+ chars inside brackets
        assert extract_cited_doc_ids("Empty [] brackets.") == set()

    def test_filters_to_available_ids(self):
        text = "See [doc_001] and [doc_999]."
        available = {"doc_001", "doc_002"}
        assert extract_cited_doc_ids(text, available) == {"doc_001"}

    def test_no_available_ids_returns_all(self):
        text = "See [doc_001] and [doc_999]."
        assert extract_cited_doc_ids(text) == {"doc_001", "doc_999"}

    def test_nested_brackets(self):
        # Regex should handle edge case of nested brackets
        text = "See [[doc_001]] for info."
        result = extract_cited_doc_ids(text)
        # Inner brackets: first match is [doc_001], second is the outer content
        assert "doc_001" in result or "[doc_001" in result


# ── calculate_confidence ─────────────────────────────────────────

class TestCalculateConfidence:
    def _make_results(self, *doc_ids):
        return [{"doc_id": did} for did in doc_ids]

    def test_no_results_returns_zero(self):
        assert calculate_confidence("any answer", [], set()) == 0.0

    def test_refusal_returns_low(self):
        results = self._make_results("doc_001")
        answer = "I cannot answer this based on the provided documents."
        assert calculate_confidence(answer, results, set()) == 0.1

    def test_refusal_not_enough_info(self):
        results = self._make_results("doc_001")
        answer = "I don't have enough information to answer."
        assert calculate_confidence(answer, results, set()) == 0.1

    def test_no_citations_returns_03(self):
        results = self._make_results("doc_001", "doc_002")
        assert calculate_confidence("Answer without refs.", results, set()) == 0.3

    def test_hallucinated_citations_ignored(self):
        results = self._make_results("doc_001")
        # cited_ids has doc_999 which doesn't exist in results
        cited = {"doc_999"}
        assert calculate_confidence("Answer [doc_999].", results, cited) == 0.3

    def test_partial_citation_coverage(self):
        results = self._make_results("doc_001", "doc_002")
        cited = {"doc_001"}
        confidence = calculate_confidence("Answer [doc_001].", results, cited)
        # 0.6 + 0.4 * (1/2) = 0.8
        assert confidence == 0.8

    def test_all_citations_returns_max(self):
        results = self._make_results("doc_001", "doc_002")
        cited = {"doc_001", "doc_002"}
        confidence = calculate_confidence("See [doc_001] [doc_002].", results, cited)
        # 0.6 + 0.4 * (2/2) = 1.0
        assert confidence == 1.0

    def test_confidence_is_rounded(self):
        results = self._make_results("doc_001", "doc_002", "doc_003")
        cited = {"doc_001"}
        confidence = calculate_confidence("Answer [doc_001].", results, cited)
        # 0.6 + 0.4 * (1/3) = 0.7333... → rounded to 0.73
        assert confidence == 0.73


# ── Citation schema ──────────────────────────────────────────────

class TestCitation:
    def test_full_citation(self):
        c = Citation(doc_id="doc_001", snippet="text", score=0.95, source="file.txt")
        assert c.doc_id == "doc_001"
        assert c.score == 0.95
        assert c.source == "file.txt"

    def test_minimal_citation(self):
        c = Citation(doc_id="doc_001", snippet="text")
        assert c.score is None
        assert c.source is None


# ── QueryRequest schema ─────────────────────────────────────────

class TestQueryRequest:
    def test_valid_request(self):
        req = QueryRequest(query="What is RAG?")
        assert req.query == "What is RAG?"
        assert req.k == 5  # default

    def test_custom_k(self):
        req = QueryRequest(query="test", k=10)
        assert req.k == 10

    def test_empty_query_rejected(self):
        with pytest.raises(ValidationError):
            QueryRequest(query="")

    def test_long_query_rejected(self):
        with pytest.raises(ValidationError):
            QueryRequest(query="x" * 1001)

    def test_k_below_min_rejected(self):
        with pytest.raises(ValidationError):
            QueryRequest(query="test", k=0)

    def test_k_above_max_rejected(self):
        with pytest.raises(ValidationError):
            QueryRequest(query="test", k=21)


# ── QueryResponse schema ────────────────────────────────────────

class TestQueryResponse:
    def test_valid_response(self):
        resp = QueryResponse(
            query="What is RAG?",
            category="factual",
            answer="RAG is ... [doc_001]",
            citations=[Citation(doc_id="doc_001", snippet="RAG combines...")],
            confidence=0.85,
            latency_ms=123.45,
        )
        assert resp.confidence == 0.85
        assert len(resp.citations) == 1


# ── citation_coverage (eval metrics) ────────────────────────────

class TestCitationCoverage:
    def test_all_cited(self):
        answer = "See [doc_001] and [doc_002] for details."
        citations = [
            {"doc_id": "doc_001"},
            {"doc_id": "doc_002"},
        ]
        assert calculate_citation_coverage(answer, citations) == 1.0

    def test_partial_cited(self):
        answer = "Only [doc_001] is mentioned."
        citations = [
            {"doc_id": "doc_001"},
            {"doc_id": "doc_002"},
        ]
        assert calculate_citation_coverage(answer, citations) == 0.5

    def test_none_cited(self):
        answer = "No references at all."
        citations = [{"doc_id": "doc_001"}]
        assert calculate_citation_coverage(answer, citations) == 0.0

    def test_empty_citations(self):
        assert calculate_citation_coverage("any answer", []) == 0.0


# ── hallucination_rate (eval metrics) ────────────────────────────

class TestHallucinationRate:
    def test_placeholder_returns_fixed_value(self):
        assert estimate_hallucination_rate("answer", "context") == 0.1
