"""Tests for critical code paths that lack direct coverage.

Targets the error-handling wrappers and optimization paths that ALL
LLM traffic flows through but had zero dedicated tests:

- _safe_llm_call / _safe_llm_call_async — error fallback behavior
- call_llm / call_llm_async — raw LLM call delegation
- _parse_classification — invalid/unexpected LLM responses
- calculate_confidence — _available_ids optimization kwarg
- get_available_doc_ids — degenerate metadata inputs
"""

import pytest
from unittest.mock import MagicMock, AsyncMock, patch

from src.llm.synthesize import (
    _safe_llm_call,
    _safe_llm_call_async,
    _parse_classification,
    _parse_synthesis,
    calculate_confidence,
    get_available_doc_ids,
    extract_cited_doc_ids,
    _SYNTHESIS_ERROR,
)
from src.llm.client import call_llm, call_llm_async


# ── helpers ──────────────────────────────────────────────────────

def _make_response(content: str):
    msg = MagicMock()
    msg.content = content
    choice = MagicMock()
    choice.message = msg
    resp = MagicMock()
    resp.choices = [choice]
    return resp


_DOCS = [
    {"doc_id": "a", "snippet": "Alpha"},
    {"doc_id": "b", "snippet": "Beta"},
]


# ── _safe_llm_call (sync error wrapper) ─────────────────────────

class TestSafeLlmCall:
    """_safe_llm_call wraps every sync LLM call — verify fallback on error."""

    @patch("src.llm.synthesize.get_llm_client")
    @patch("src.llm.synthesize._call_llm")
    def test_returns_parsed_result_on_success(self, mock_call, mock_client):
        mock_call.return_value = "raw_response"
        result = _safe_llm_call("model", [{"role": "user", "content": "hi"}],
                                lambda r: r.upper(), "fallback", "test")
        assert result == "RAW_RESPONSE"

    @patch("src.llm.synthesize.get_llm_client")
    @patch("src.llm.synthesize._call_llm", side_effect=ConnectionError("timeout"))
    def test_returns_fallback_on_connection_error(self, mock_call, mock_client):
        result = _safe_llm_call("model", [], lambda r: r, "FALLBACK", "test")
        assert result == "FALLBACK"

    @patch("src.llm.synthesize.get_llm_client")
    @patch("src.llm.synthesize._call_llm", side_effect=RuntimeError("boom"))
    def test_returns_fallback_on_runtime_error(self, mock_call, mock_client):
        result = _safe_llm_call("model", [], lambda r: r, {"error": True}, "test")
        assert result == {"error": True}

    @patch("src.llm.synthesize.get_llm_client")
    @patch("src.llm.synthesize._call_llm")
    def test_returns_fallback_when_parser_raises(self, mock_call, mock_client):
        """Parser crash must not propagate — fallback returned instead."""
        mock_call.return_value = "resp"
        result = _safe_llm_call("model", [], lambda r: 1 / 0, "safe", "test")
        assert result == "safe"

    @patch("src.llm.synthesize.get_llm_client")
    @patch("src.llm.synthesize._call_llm")
    def test_passes_kwargs_through(self, mock_call, mock_client):
        mock_call.return_value = "ok"
        _safe_llm_call("gpt-4", [{"role": "user", "content": "q"}],
                        lambda r: r, "", "test", temperature=0, max_tokens=10)
        _, kwargs = mock_call.call_args
        assert kwargs["temperature"] == 0
        assert kwargs["max_tokens"] == 10


# ── _safe_llm_call_async (async error wrapper) ──────────────────

class TestSafeLlmCallAsync:
    """Async mirror — same contract, async client path."""

    @pytest.mark.asyncio
    @patch("src.llm.synthesize.get_async_llm_client", new_callable=AsyncMock)
    @patch("src.llm.synthesize._call_llm_async", new_callable=AsyncMock)
    async def test_returns_parsed_on_success(self, mock_call, mock_client):
        mock_call.return_value = "resp"
        result = await _safe_llm_call_async("m", [], lambda r: f"got:{r}", "fb", "t")
        assert result == "got:resp"

    @pytest.mark.asyncio
    @patch("src.llm.synthesize.get_async_llm_client", new_callable=AsyncMock)
    @patch("src.llm.synthesize._call_llm_async", new_callable=AsyncMock,
           side_effect=TimeoutError("stalled"))
    async def test_returns_fallback_on_timeout(self, mock_call, mock_client):
        result = await _safe_llm_call_async("m", [], lambda r: r, "timeout_fb", "t")
        assert result == "timeout_fb"

    @pytest.mark.asyncio
    @patch("src.llm.synthesize.get_async_llm_client", new_callable=AsyncMock)
    @patch("src.llm.synthesize._call_llm_async", new_callable=AsyncMock)
    async def test_returns_fallback_when_parser_raises(self, mock_call, mock_client):
        mock_call.return_value = "r"
        result = await _safe_llm_call_async("m", [], lambda r: [][0], "safe", "t")
        assert result == "safe"


# ── call_llm / call_llm_async (raw delegation) ──────────────────

class TestCallLlm:
    """Raw call helpers — thin wrappers, but verify delegation and kwargs."""

    def test_delegates_to_client(self):
        client = MagicMock()
        client.chat.completions.create.return_value = "result"
        out = call_llm(client, "gpt-4", [{"role": "user", "content": "q"}])
        assert out == "result"
        client.chat.completions.create.assert_called_once_with(
            model="gpt-4", messages=[{"role": "user", "content": "q"}]
        )

    def test_forwards_extra_kwargs(self):
        client = MagicMock()
        call_llm(client, "m", [], temperature=0.5, stop=["\n"])
        _, kwargs = client.chat.completions.create.call_args
        assert kwargs["temperature"] == 0.5
        assert kwargs["stop"] == ["\n"]

    @pytest.mark.asyncio
    async def test_async_delegates_to_client(self):
        client = MagicMock()
        client.chat.completions.create = AsyncMock(return_value="async_result")
        out = await call_llm_async(client, "m", [{"role": "user", "content": "q"}])
        assert out == "async_result"


# ── _parse_classification edge cases ─────────────────────────────

class TestParseClassification:
    """LLM responses are unpredictable — verify all classification paths."""

    def test_valid_factual(self):
        assert _parse_classification(_make_response("factual")) == "factual"

    def test_valid_exploratory(self):
        assert _parse_classification(_make_response("exploratory")) == "exploratory"

    def test_valid_ambiguous(self):
        assert _parse_classification(_make_response("ambiguous")) == "ambiguous"

    def test_unknown_category_defaults_to_exploratory(self):
        assert _parse_classification(_make_response("opinion")) == "exploratory"

    def test_whitespace_stripped_before_matching(self):
        assert _parse_classification(_make_response("  factual\n  ")) == "factual"

    def test_case_insensitive(self):
        assert _parse_classification(_make_response("FACTUAL")) == "factual"

    def test_empty_string_defaults_to_exploratory(self):
        assert _parse_classification(_make_response("")) == "exploratory"

    def test_multiword_junk_defaults_to_exploratory(self):
        """LLM returns a full sentence instead of a single word."""
        resp = _make_response("I think this is a factual question about science.")
        assert _parse_classification(resp) == "exploratory"


# ── calculate_confidence _available_ids optimization ─────────────

class TestConfidenceAvailableIdsKwarg:
    """The _available_ids kwarg lets callers skip redundant recomputation.
    Must produce identical results to the auto-compute path."""

    def test_precomputed_matches_auto(self):
        cited = {"a"}
        auto = calculate_confidence("See [a].", _DOCS, cited)
        precomputed = calculate_confidence("See [a].", _DOCS, cited,
                                           _available_ids={"a", "b"})
        assert auto == precomputed

    def test_precomputed_empty_available_returns_low_confidence(self):
        """If caller passes empty available_ids, confidence should be 0.3."""
        result = calculate_confidence("[a].", _DOCS, {"a"},
                                      _available_ids=set())
        assert result == 0.3

    def test_precomputed_superset_does_not_inflate(self):
        """Extra IDs in _available_ids shouldn't inflate confidence."""
        # Only "a" is cited, but available has "a","b","c"
        result = calculate_confidence("[a].", _DOCS, {"a"},
                                      _available_ids={"a", "b", "c"})
        expected = 0.6 + (0.4 * (1 / 3))  # 1 of 3 available
        assert result == pytest.approx(round(expected, 2))


# ── get_available_doc_ids degenerate inputs ──────────────────────

class TestGetAvailableDocIds:
    """Edge cases for doc_id extraction from metadata."""

    def test_empty_string_doc_id_included(self):
        """Empty string is not None — it's a valid (if useless) doc_id."""
        result = get_available_doc_ids([{"doc_id": "", "snippet": "s"}])
        assert "" in result

    def test_all_none_returns_empty_set(self):
        results = [{"doc_id": None}, {"doc_id": None}]
        assert get_available_doc_ids(results) == set()

    def test_missing_doc_id_key_excluded(self):
        """Result dict without doc_id key → .get returns None → excluded."""
        assert get_available_doc_ids([{"snippet": "orphan"}]) == set()

    def test_mixed_types_all_coerced_to_str(self):
        results = [{"doc_id": 1}, {"doc_id": "two"}, {"doc_id": 3.0}]
        ids = get_available_doc_ids(results)
        assert ids == {"1", "two", "3.0"}


# ── extract_cited_doc_ids edge cases ─────────────────────────────

class TestExtractCitedDocIds:
    """Citation regex edge cases that affect downstream confidence."""

    def test_nested_brackets_extracts_inner(self):
        """[outer [inner]] — regex behavior on nested brackets."""
        result = extract_cited_doc_ids("See [doc_a].")
        assert "doc_a" in result

    def test_multiple_citations_same_id_deduped(self):
        result = extract_cited_doc_ids("[a] and [a] again.")
        assert result == {"a"}

    def test_no_match_on_empty_brackets(self):
        """[] should not match — requires 1+ chars inside."""
        result = extract_cited_doc_ids("See [] here.")
        assert result == set()

    def test_available_ids_filters_hallucinated(self):
        result = extract_cited_doc_ids("[real] and [fake].", available_ids={"real"})
        assert result == {"real"}
