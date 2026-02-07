"""Tests for async LLM functions: classify_query_async and synthesize_answer_async.

These are the ACTUAL production code paths used by the FastAPI /query endpoint.
Uses AsyncMock to simulate the OpenAI AsyncOpenAI client without network calls.
"""
import pytest
from unittest.mock import patch, MagicMock, AsyncMock

from src.llm.synthesize import classify_query_async, synthesize_answer_async


def _mock_async_client(content=None, side_effect=None):
    """Build a mock async OpenAI client with a preset response or error."""
    client = MagicMock()
    if side_effect:
        client.chat.completions.create = AsyncMock(side_effect=side_effect)
    else:
        msg = MagicMock(content=content)
        choice = MagicMock(message=msg)
        resp = MagicMock(choices=[choice])
        client.chat.completions.create = AsyncMock(return_value=resp)
    return client


SEARCH_RESULTS = [
    {"doc_id": "doc_001", "snippet": "RAG combines retrieval and generation."},
    {"doc_id": "doc_002", "snippet": "FAISS enables fast similarity search."},
]

# ── classify_query_async ─────────────────────────────────────────


class TestClassifyQueryAsync:
    @pytest.mark.asyncio
    @patch("src.llm.synthesize.get_async_llm_client")
    async def test_returns_factual(self, mock_get):
        mock_get.return_value = _mock_async_client("factual")
        assert await classify_query_async("What is RAG?") == "factual"

    @pytest.mark.asyncio
    @patch("src.llm.synthesize.get_async_llm_client")
    async def test_returns_exploratory(self, mock_get):
        mock_get.return_value = _mock_async_client("exploratory")
        assert await classify_query_async("Tell me about AI") == "exploratory"

    @pytest.mark.asyncio
    @patch("src.llm.synthesize.get_async_llm_client")
    async def test_returns_ambiguous(self, mock_get):
        mock_get.return_value = _mock_async_client("ambiguous")
        assert await classify_query_async("stuff") == "ambiguous"

    @pytest.mark.asyncio
    @patch("src.llm.synthesize.get_async_llm_client")
    async def test_invalid_category_defaults_to_exploratory(self, mock_get):
        mock_get.return_value = _mock_async_client("nonsense_category")
        assert await classify_query_async("something") == "exploratory"

    @pytest.mark.asyncio
    @patch("src.llm.synthesize.get_async_llm_client")
    async def test_strips_whitespace(self, mock_get):
        mock_get.return_value = _mock_async_client("  factual  \n")
        assert await classify_query_async("What is X?") == "factual"

    @pytest.mark.asyncio
    @patch("src.llm.synthesize.get_async_llm_client")
    async def test_api_error_defaults_to_factual(self, mock_get):
        mock_get.return_value = _mock_async_client(side_effect=Exception("API down"))
        assert await classify_query_async("What is X?") == "factual"

    @pytest.mark.asyncio
    @patch("src.llm.synthesize.get_async_llm_client")
    async def test_calls_openai_with_correct_params(self, mock_get):
        client = _mock_async_client("factual")
        mock_get.return_value = client
        await classify_query_async("What is X?")
        call_kwargs = client.chat.completions.create.call_args[1]
        assert call_kwargs["temperature"] == 0
        assert call_kwargs["max_tokens"] == 10


# ── synthesize_answer_async ──────────────────────────────────────


class TestSynthesizeAnswerAsync:
    @pytest.mark.asyncio
    @patch("src.llm.synthesize.get_async_llm_client")
    async def test_returns_answer_with_citations(self, mock_get):
        mock_get.return_value = _mock_async_client(
            "RAG is great [doc_001] and fast [doc_002]."
        )
        result = await synthesize_answer_async("What is RAG?", SEARCH_RESULTS)
        assert "RAG is great" in result["answer"]
        cited_ids = {c["doc_id"] for c in result["citations_used"]}
        assert cited_ids == {"doc_001", "doc_002"}
        assert result["confidence"] == 1.0

    @pytest.mark.asyncio
    @patch("src.llm.synthesize.get_async_llm_client")
    async def test_partial_citations(self, mock_get):
        mock_get.return_value = _mock_async_client("Only this source [doc_001].")
        result = await synthesize_answer_async("What is RAG?", SEARCH_RESULTS)
        cited_ids = {c["doc_id"] for c in result["citations_used"]}
        assert cited_ids == {"doc_001"}
        assert result["confidence"] == 0.8  # 0.6 + 0.4 * (1/2)

    @pytest.mark.asyncio
    @patch("src.llm.synthesize.get_async_llm_client")
    async def test_no_citations_falls_back_to_top_result(self, mock_get):
        mock_get.return_value = _mock_async_client(
            "RAG is a technique for improving LLM answers."
        )
        result = await synthesize_answer_async("What is RAG?", SEARCH_RESULTS)
        assert len(result["citations_used"]) == 1
        assert result["citations_used"][0]["doc_id"] == "doc_001"
        assert result["confidence"] == 0.3

    @pytest.mark.asyncio
    @patch("src.llm.synthesize.get_async_llm_client")
    async def test_api_error_returns_error_dict(self, mock_get):
        mock_get.return_value = _mock_async_client(
            side_effect=Exception("API quota exceeded")
        )
        result = await synthesize_answer_async("What is RAG?", SEARCH_RESULTS)
        assert result["answer"] == "Error generating answer."
        assert result["citations_used"] == []
        assert result["confidence"] == 0.0

    @pytest.mark.asyncio
    @patch("src.llm.synthesize.get_async_llm_client")
    async def test_hallucinated_citation_filtered_out(self, mock_get):
        mock_get.return_value = _mock_async_client("Answer [doc_001] and [doc_999].")
        result = await synthesize_answer_async("question", SEARCH_RESULTS)
        cited_ids = {c["doc_id"] for c in result["citations_used"]}
        assert "doc_999" not in cited_ids
        assert "doc_001" in cited_ids

    @pytest.mark.asyncio
    @patch("src.llm.synthesize.get_async_llm_client")
    async def test_empty_search_results(self, mock_get):
        mock_get.return_value = _mock_async_client(
            "I cannot answer this based on the provided documents."
        )
        result = await synthesize_answer_async("question", [])
        assert result["confidence"] == 0.0

    @pytest.mark.asyncio
    @patch("src.llm.synthesize.get_async_llm_client")
    async def test_refusal_gets_low_confidence(self, mock_get):
        mock_get.return_value = _mock_async_client(
            "I cannot answer this question based on the provided context."
        )
        result = await synthesize_answer_async("obscure question", SEARCH_RESULTS)
        assert result["confidence"] == 0.1
