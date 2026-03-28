"""Contract tests — behavioral promises each component makes to its callers.

Tests the *interfaces* between components, not internal implementation.
Catches integration bugs that unit tests miss: double-replacement attacks,
conditional kwargs, bounds checking, crash paths on malformed data.
"""

import os
import re
import pytest
from unittest.mock import patch, MagicMock

from src.llm.prompt import format_rag_prompt, format_classification_prompt, build_context_str
from src.llm.synthesize import (
    _parse_synthesis, _parse_classification, _client_kwargs,
    calculate_confidence, extract_cited_doc_ids, CITATION_PATTERN,
    _SYNTHESIS_ERROR, _classification_messages, _synthesis_messages,
)
from src.api.response import build_diagnostics, build_citations, _sanitize_field
from src.utils.env import safe_int_env
from src.utils.timing import measure_latency
from src.eval.metrics import estimate_hallucination_rate, calculate_citation_coverage


# ─── Prompt Safety Contract ─────────────────────────────────────────

class TestPromptSafetyContract:
    """format_rag_prompt must prevent double-replacement attacks."""

    def test_context_containing_query_placeholder_not_expanded(self):
        """If context contains literal '{query}', it must NOT be replaced with the user query."""
        context = "The template uses {query} as a placeholder."
        query = "INJECTED"
        result = format_rag_prompt(context, query)
        assert "{query}" in result  # literal preserved in context section
        assert result.count("INJECTED") == 1  # appears only in User Question slot

    def test_context_containing_context_str_placeholder_preserved(self):
        """If context contains literal '{context_str}', it must NOT trigger recursion."""
        context = "Docs mention {context_str} variable."
        query = "test"
        result = format_rag_prompt(context, query)
        assert "{context_str}" in result

    def test_classification_prompt_multiple_query_placeholders(self):
        """Query should replace the placeholder exactly once."""
        result = format_classification_prompt("my query")
        assert "my query" in result
        assert "{query}" not in result

    def test_empty_context_and_query(self):
        result = format_rag_prompt("", "")
        assert "Context:" in result
        assert "User Question:" in result


# ─── Client Factory Contract ────────────────────────────────────────

class TestClientKwargsContract:
    """_client_kwargs must include base_url only when OPENAI_BASE_URL is set."""

    @patch.dict(os.environ, {"OPENAI_API_KEY": "sk-test"}, clear=False)
    def test_base_url_absent_when_env_unset(self):
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("OPENAI_BASE_URL", None)
            kwargs = _client_kwargs()
            assert "base_url" not in kwargs
            assert kwargs["api_key"] == "sk-test"

    @patch.dict(os.environ, {"OPENAI_API_KEY": "sk-test", "OPENAI_BASE_URL": "http://proxy:8080"})
    def test_base_url_present_when_env_set(self):
        kwargs = _client_kwargs()
        assert kwargs["base_url"] == "http://proxy:8080"

    @patch.dict(os.environ, {}, clear=True)
    def test_missing_api_key_still_returns_dict(self):
        kwargs = _client_kwargs()
        assert kwargs["api_key"] is None


# ─── safe_int_env Contract ──────────────────────────────────────────

class TestSafeIntEnvContract:
    """safe_int_env must handle every malformed input gracefully."""

    @patch.dict(os.environ, {"TEST_VAL": "3.14"})
    def test_float_string_falls_back(self):
        assert safe_int_env("TEST_VAL", 10) == 10

    @patch.dict(os.environ, {"TEST_VAL": ""})
    def test_empty_string_falls_back(self):
        assert safe_int_env("TEST_VAL", 42) == 42

    @patch.dict(os.environ, {"TEST_VAL": "5"})
    def test_below_min_val_falls_back(self):
        assert safe_int_env("TEST_VAL", 30, min_val=10) == 30

    @patch.dict(os.environ, {"TEST_VAL": "10"})
    def test_at_min_val_boundary_accepted(self):
        assert safe_int_env("TEST_VAL", 99, min_val=10) == 10

    @patch.dict(os.environ, {"TEST_VAL": "-1"})
    def test_negative_accepted_when_no_min(self):
        assert safe_int_env("TEST_VAL", 0) == -1

    def test_missing_env_returns_default(self):
        assert safe_int_env("NONEXISTENT_KEY_XYZ", 77) == 77


# ─── Synthesis Pipeline Contract ────────────────────────────────────

class TestParseSynthesisContract:
    """_parse_synthesis must handle fallback and edge cases correctly."""

    def _make_response(self, content):
        resp = MagicMock()
        resp.choices[0].message.content = content
        return resp

    def test_fallback_to_top_result_when_no_citations(self):
        """When LLM answer has no brackets, include top search result as fallback."""
        results = [{"doc_id": "doc_1", "snippet": "info", "score": 0.9}]
        parsed = _parse_synthesis(self._make_response("Just a plain answer."), results)
        assert len(parsed["citations_used"]) == 1
        assert parsed["citations_used"][0]["doc_id"] == "doc_1"

    def test_hallucinated_citations_filtered_out(self):
        """Citations referencing non-existent docs must be excluded."""
        results = [{"doc_id": "real_doc", "snippet": "data"}]
        parsed = _parse_synthesis(
            self._make_response("See [real_doc] and [fake_doc]."), results
        )
        cited_ids = {r["doc_id"] for r in parsed["citations_used"]}
        assert "fake_doc" not in cited_ids
        assert "real_doc" in cited_ids

    def test_empty_search_results_returns_low_confidence(self):
        parsed = _parse_synthesis(self._make_response("Answer text."), [])
        assert parsed["confidence"] == 0.0

    def test_synthesis_error_sentinel_is_independent_copy(self):
        """dict(_SYNTHESIS_ERROR) must produce a new dict each call."""
        a = dict(_SYNTHESIS_ERROR)
        b = dict(_SYNTHESIS_ERROR)
        a["answer"] = "mutated"
        assert b["answer"] == "Error generating answer."


# ─── Classification Contract ────────────────────────────────────────

class TestParseClassificationContract:

    def _make_response(self, content):
        resp = MagicMock()
        resp.choices[0].message.content = content
        return resp

    def test_whitespace_padded_category_normalized(self):
        assert _parse_classification(self._make_response("  Factual  \n")) == "factual"

    def test_unknown_category_defaults_to_exploratory(self):
        assert _parse_classification(self._make_response("creative")) == "exploratory"

    def test_mixed_case_accepted(self):
        assert _parse_classification(self._make_response("AMBIGUOUS")) == "ambiguous"


# ─── Diagnostics Crash Path ─────────────────────────────────────────

class TestBuildDiagnosticsContract:
    """build_diagnostics must not crash on edge-case inputs."""

    def test_zero_search_results_no_division_error(self):
        diag = build_diagnostics([], "no docs", 10.0, 20.0, 0.0)
        assert diag.citation_coverage == 0.0
        assert diag.documents_searched == 0

    def test_hallucinated_citations_detected(self):
        results = [{"doc_id": "d1"}, {"doc_id": "d2"}]
        diag = build_diagnostics(results, "See [d1] and [phantom].", 5.0, 10.0, 0.7)
        assert "phantom" in diag.hallucinated_citations
        assert diag.citation_coverage == 0.5

    def test_timing_rounded_to_two_decimals(self):
        results = [{"doc_id": "x"}]
        diag = build_diagnostics(results, "[x]", 1.23456, 7.89012, 0.8)
        assert diag.retrieval_ms == 1.23
        assert diag.synthesis_ms == 7.89


# ─── measure_latency Contract ───────────────────────────────────────

class TestMeasureLatencyContract:
    """measure_latency must inject timing into dicts and pass through non-dicts."""

    def test_injects_into_dict_result(self):
        @measure_latency
        def returns_dict():
            return {"key": "value"}
        result = returns_dict()
        assert "latency_ms" in result
        assert isinstance(result["latency_ms"], float)

    def test_does_not_overwrite_existing_latency(self):
        @measure_latency
        def returns_dict_with_latency():
            return {"latency_ms": 999.0}
        result = returns_dict_with_latency()
        assert result["latency_ms"] == 999.0  # preserved, not overwritten

    def test_string_result_passed_through(self):
        @measure_latency
        def returns_string():
            return "hello"
        assert returns_string() == "hello"

    def test_none_result_passed_through(self):
        @measure_latency
        def returns_none():
            return None
        assert returns_none() is None


# ─── CITATION_PATTERN Contract ──────────────────────────────────────

class TestCitationPatternContract:
    """Compiled CITATION_PATTERN must match the documented extraction behavior."""

    def test_nested_brackets_captures_inner(self):
        # [[doc_1]] — outer brackets aren't a valid citation format
        matches = CITATION_PATTERN.findall("[[doc_1]]")
        assert "[doc_1" in matches or "doc_1" in matches

    def test_empty_brackets_no_match(self):
        assert CITATION_PATTERN.findall("[]") == []

    def test_multiple_citations_extracted(self):
        text = "[a] then [b] then [c]"
        assert set(CITATION_PATTERN.findall(text)) == {"a", "b", "c"}

    def test_bracket_in_prose_captured(self):
        """Non-doc-id content in brackets is still captured (caller must filter)."""
        matches = CITATION_PATTERN.findall("This is [not a doc id] really.")
        assert "not a doc id" in matches


# ─── Hallucination Estimator Contract ───────────────────────────────

class TestHallucinationContract:
    """estimate_hallucination_rate must handle adversarial inputs."""

    def test_answer_all_stop_words_returns_zero(self):
        """An answer of only stop words has no content words to hallucinate."""
        assert estimate_hallucination_rate("the a an is are", "unrelated context") == 0.0

    def test_fully_grounded_answer(self):
        context = "Python is a programming language"
        answer = "Python is a programming language"
        assert estimate_hallucination_rate(answer, context) == 0.0

    def test_fully_hallucinated_answer(self):
        rate = estimate_hallucination_rate("quantum entanglement theory", "")
        assert rate == 1.0

    def test_whitespace_only_answer_returns_zero(self):
        assert estimate_hallucination_rate("   \t\n  ", "context") == 0.0
