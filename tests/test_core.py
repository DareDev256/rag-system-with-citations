"""Unit tests for RAG system core logic.

Tests only pure functions — no FAISS, no API calls, no LLM calls.
"""
import time

import pytest
from pydantic import ValidationError

from src.api.schemas import QueryRequest, QueryResponse, Citation
from src.llm.synthesize import extract_cited_doc_ids, calculate_confidence
from src.llm.prompt import build_context_str
from src.eval.metrics import calculate_citation_coverage, estimate_hallucination_rate
from src.data.ingest import load_documents
from src.utils.timing import measure_latency


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


# ── load_documents (data ingestion) ─────────────────────────────

class TestLoadDocuments:
    def test_nonexistent_directory_returns_empty(self, tmp_path):
        result = load_documents(str(tmp_path / "does_not_exist"))
        assert result == []

    def test_empty_directory_returns_empty(self, tmp_path):
        result = load_documents(str(tmp_path))
        assert result == []

    def test_single_file_single_paragraph(self, tmp_path):
        (tmp_path / "notes.txt").write_text("RAG combines retrieval with generation.")
        docs = load_documents(str(tmp_path))
        assert len(docs) == 1
        assert docs[0]["doc_id"] == "notes.txt_0"
        assert docs[0]["text"] == "RAG combines retrieval with generation."
        assert docs[0]["source"] == "notes.txt"

    def test_single_file_multiple_paragraphs(self, tmp_path):
        (tmp_path / "guide.txt").write_text("First paragraph.\n\nSecond paragraph.")
        docs = load_documents(str(tmp_path))
        assert len(docs) == 2
        texts = [d["text"] for d in docs]
        assert "First paragraph." in texts
        assert "Second paragraph." in texts

    def test_empty_paragraphs_skipped(self, tmp_path):
        # Double newlines with only whitespace between them
        (tmp_path / "sparse.txt").write_text("Content here.\n\n\n\n  \n\nMore content.")
        docs = load_documents(str(tmp_path))
        texts = [d["text"] for d in docs]
        assert len(texts) == 2
        assert "Content here." in texts
        assert "More content." in texts

    def test_multiple_files(self, tmp_path):
        (tmp_path / "a.txt").write_text("Alpha content.")
        (tmp_path / "b.txt").write_text("Beta content.")
        docs = load_documents(str(tmp_path))
        assert len(docs) == 2
        sources = {d["source"] for d in docs}
        assert sources == {"a.txt", "b.txt"}

    def test_non_txt_files_ignored(self, tmp_path):
        (tmp_path / "data.txt").write_text("Included.")
        (tmp_path / "data.csv").write_text("col1,col2\na,b")
        (tmp_path / "data.json").write_text('{"key": "value"}')
        docs = load_documents(str(tmp_path))
        assert len(docs) == 1
        assert docs[0]["source"] == "data.txt"

    def test_doc_id_format_includes_paragraph_index(self, tmp_path):
        (tmp_path / "paper.txt").write_text("Para zero.\n\nPara one.\n\nPara two.")
        docs = load_documents(str(tmp_path))
        doc_ids = [d["doc_id"] for d in docs]
        assert "paper.txt_0" in doc_ids
        assert "paper.txt_1" in doc_ids
        assert "paper.txt_2" in doc_ids

    def test_whitespace_stripped_from_paragraphs(self, tmp_path):
        (tmp_path / "messy.txt").write_text("  leading spaces  \n\n  trailing too  ")
        docs = load_documents(str(tmp_path))
        assert docs[0]["text"] == "leading spaces"
        assert docs[1]["text"] == "trailing too"


# ── measure_latency (timing decorator) ──────────────────────────

class TestMeasureLatency:
    def test_injects_latency_into_dict_result(self):
        @measure_latency
        def returns_dict():
            return {"answer": "hello"}

        result = returns_dict()
        assert "latency_ms" in result
        assert isinstance(result["latency_ms"], float)
        assert result["answer"] == "hello"

    def test_does_not_overwrite_existing_latency(self):
        @measure_latency
        def preset_latency():
            return {"answer": "hi", "latency_ms": 999.0}

        result = preset_latency()
        # The decorator checks 'latency_ms' not in result, so it skips
        assert result["latency_ms"] == 999.0

    def test_injects_latency_into_object_with_attribute(self):
        class Response:
            def __init__(self):
                self.latency_ms = -1.0  # sentinel value
                self.data = "test"

        @measure_latency
        def returns_obj():
            return Response()

        result = returns_obj()
        # Decorator should overwrite the sentinel with actual timing
        assert result.latency_ms >= 0.0
        assert result.data == "test"

    def test_handles_non_dict_non_object_result(self):
        @measure_latency
        def returns_string():
            return "plain string"

        # Should not raise — just returns the value unchanged
        result = returns_string()
        assert result == "plain string"

    def test_preserves_function_name(self):
        @measure_latency
        def my_function():
            """My docstring."""
            return {}

        assert my_function.__name__ == "my_function"
        assert my_function.__doc__ == "My docstring."

    def test_latency_reflects_actual_time(self):
        @measure_latency
        def slow_function():
            time.sleep(0.05)  # 50ms
            return {"data": True}

        result = slow_function()
        # Should be at least 40ms (allowing some scheduling variance)
        assert result["latency_ms"] >= 40.0

    def test_passes_args_and_kwargs(self):
        @measure_latency
        def add(a, b, extra=0):
            return {"sum": a + b + extra}

        result = add(2, 3, extra=10)
        assert result["sum"] == 15
        assert "latency_ms" in result

    def test_returns_none_without_error(self):
        @measure_latency
        def returns_none():
            return None

        result = returns_none()
        assert result is None
