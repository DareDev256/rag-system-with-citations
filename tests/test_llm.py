"""Mock-based tests for LLM synthesize functions.

Tests classify_query, synthesize_answer, and async variants by mocking
the OpenAI client — no API key or network calls required.
"""
from unittest.mock import patch, MagicMock

from src.llm.synthesize import classify_query, synthesize_answer


def _mock_chat_response(content: str):
    """Build a fake OpenAI ChatCompletion response."""
    msg = MagicMock()
    msg.content = content
    choice = MagicMock()
    choice.message = msg
    resp = MagicMock()
    resp.choices = [choice]
    return resp


# ── classify_query (sync) ────────────────────────────────────────

class TestClassifyQuery:
    @patch("src.llm.synthesize.get_llm_client")
    def test_returns_factual(self, mock_get):
        client = MagicMock()
        client.chat.completions.create.return_value = _mock_chat_response("factual")
        mock_get.return_value = client

        # classify_query returns a string; measure_latency passes it through
        assert classify_query("What is RAG?") == "factual"

    @patch("src.llm.synthesize.get_llm_client")
    def test_returns_exploratory(self, mock_get):
        client = MagicMock()
        client.chat.completions.create.return_value = _mock_chat_response("exploratory")
        mock_get.return_value = client

        assert classify_query("Tell me about AI") == "exploratory"

    @patch("src.llm.synthesize.get_llm_client")
    def test_returns_ambiguous(self, mock_get):
        client = MagicMock()
        client.chat.completions.create.return_value = _mock_chat_response("ambiguous")
        mock_get.return_value = client

        assert classify_query("stuff") == "ambiguous"

    @patch("src.llm.synthesize.get_llm_client")
    def test_invalid_category_defaults_to_exploratory(self, mock_get):
        client = MagicMock()
        client.chat.completions.create.return_value = _mock_chat_response("nonsense_category")
        mock_get.return_value = client

        assert classify_query("something") == "exploratory"

    @patch("src.llm.synthesize.get_llm_client")
    def test_strips_whitespace_from_response(self, mock_get):
        client = MagicMock()
        client.chat.completions.create.return_value = _mock_chat_response("  factual  \n")
        mock_get.return_value = client

        assert classify_query("What is X?") == "factual"

    @patch("src.llm.synthesize.get_llm_client")
    def test_api_error_defaults_to_factual(self, mock_get):
        client = MagicMock()
        client.chat.completions.create.side_effect = Exception("API down")
        mock_get.return_value = client

        assert classify_query("What is X?") == "factual"

    @patch("src.llm.synthesize.get_llm_client")
    def test_calls_openai_with_correct_model(self, mock_get):
        client = MagicMock()
        client.chat.completions.create.return_value = _mock_chat_response("factual")
        mock_get.return_value = client

        classify_query("What is X?")
        call_kwargs = client.chat.completions.create.call_args[1]
        assert call_kwargs["temperature"] == 0
        assert call_kwargs["max_tokens"] == 10


# ── synthesize_answer (sync) ─────────────────────────────────────

class TestSynthesizeAnswer:
    SEARCH_RESULTS = [
        {"doc_id": "doc_001", "snippet": "RAG combines retrieval and generation."},
        {"doc_id": "doc_002", "snippet": "FAISS enables fast similarity search."},
    ]

    @patch("src.llm.synthesize.get_llm_client")
    def test_returns_answer_with_citations(self, mock_get):
        client = MagicMock()
        client.chat.completions.create.return_value = _mock_chat_response(
            "RAG is great [doc_001] and fast [doc_002]."
        )
        mock_get.return_value = client

        result = synthesize_answer("What is RAG?", self.SEARCH_RESULTS)
        assert "RAG is great" in result["answer"]
        cited_ids = {c["doc_id"] for c in result["citations_used"]}
        assert cited_ids == {"doc_001", "doc_002"}
        assert result["confidence"] == 1.0

    @patch("src.llm.synthesize.get_llm_client")
    def test_partial_citations(self, mock_get):
        client = MagicMock()
        client.chat.completions.create.return_value = _mock_chat_response(
            "Only this source [doc_001]."
        )
        mock_get.return_value = client

        result = synthesize_answer("What is RAG?", self.SEARCH_RESULTS)
        cited_ids = {c["doc_id"] for c in result["citations_used"]}
        assert cited_ids == {"doc_001"}
        assert result["confidence"] == 0.8  # 0.6 + 0.4 * (1/2)

    @patch("src.llm.synthesize.get_llm_client")
    def test_no_citations_falls_back_to_top_result(self, mock_get):
        client = MagicMock()
        client.chat.completions.create.return_value = _mock_chat_response(
            "RAG is a technique for improving LLM answers."
        )
        mock_get.return_value = client

        result = synthesize_answer("What is RAG?", self.SEARCH_RESULTS)
        # Fallback: includes top result even without citations
        assert len(result["citations_used"]) == 1
        assert result["citations_used"][0]["doc_id"] == "doc_001"
        assert result["confidence"] == 0.3

    @patch("src.llm.synthesize.get_llm_client")
    def test_api_error_returns_error_dict(self, mock_get):
        client = MagicMock()
        client.chat.completions.create.side_effect = Exception("API quota exceeded")
        mock_get.return_value = client

        result = synthesize_answer("What is RAG?", self.SEARCH_RESULTS)
        assert result["answer"] == "Error generating answer."
        assert result["citations_used"] == []
        assert result["confidence"] == 0.0

    @patch("src.llm.synthesize.get_llm_client")
    def test_hallucinated_citation_filtered_out(self, mock_get):
        client = MagicMock()
        client.chat.completions.create.return_value = _mock_chat_response(
            "Answer [doc_001] and [doc_999]."
        )
        mock_get.return_value = client

        result = synthesize_answer("question", self.SEARCH_RESULTS)
        cited_ids = {c["doc_id"] for c in result["citations_used"]}
        # doc_999 doesn't exist in search results — should be filtered
        assert "doc_999" not in cited_ids
        assert "doc_001" in cited_ids

    @patch("src.llm.synthesize.get_llm_client")
    def test_injects_latency_ms(self, mock_get):
        client = MagicMock()
        client.chat.completions.create.return_value = _mock_chat_response("ans [doc_001]")
        mock_get.return_value = client

        result = synthesize_answer("q", self.SEARCH_RESULTS)
        assert "latency_ms" in result
        assert isinstance(result["latency_ms"], float)

    @patch("src.llm.synthesize.get_llm_client")
    def test_empty_search_results(self, mock_get):
        client = MagicMock()
        client.chat.completions.create.return_value = _mock_chat_response(
            "I cannot answer this based on the provided documents."
        )
        mock_get.return_value = client

        result = synthesize_answer("question", [])
        assert result["confidence"] == 0.0  # no search results → 0.0
