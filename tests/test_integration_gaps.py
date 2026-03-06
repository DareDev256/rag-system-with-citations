"""Tests for previously-untested functions: validate_env, ingest, LLM client factories.

Mocks all external deps — no API keys, FAISS, or file I/O needed.
"""
from unittest.mock import patch, MagicMock, call

import pytest

# ---------------------------------------------------------------------------
# lifespan — FastAPI startup validation via lifespan context manager
# ---------------------------------------------------------------------------


class TestLifespan:
    @pytest.mark.asyncio
    async def test_raises_when_key_missing(self):
        from src.api.main import lifespan, app

        with patch("src.api.main.os.getenv", return_value=None):
            with pytest.raises(RuntimeError, match="OPENAI_API_KEY"):
                async with lifespan(app):
                    pass

    @pytest.mark.asyncio
    async def test_passes_when_key_set(self):
        from src.api.main import lifespan, app

        with patch("src.api.main.os.getenv", return_value="sk-test-123"):
            async with lifespan(app):
                pass  # should not raise

    @pytest.mark.asyncio
    async def test_raises_on_empty_string(self):
        from src.api.main import lifespan, app

        with patch("src.api.main.os.getenv", return_value=""):
            with pytest.raises(RuntimeError):
                async with lifespan(app):
                    pass


# ---------------------------------------------------------------------------
# ingest — data pipeline orchestrator
# ---------------------------------------------------------------------------


class TestIngest:
    @patch("src.data.ingest.VectorStore")
    @patch("src.data.ingest.get_embedder")
    @patch("src.data.ingest.load_documents")
    @patch("src.data.ingest.os.makedirs")
    def test_full_pipeline(self, mock_makedirs, mock_load, mock_embed, mock_vs_cls):
        from src.data.ingest import ingest

        mock_load.return_value = [
            {"doc_id": "f_0", "text": "hello world", "source": "f.txt"},
        ]
        mock_embedder = MagicMock()
        mock_embedder.encode.return_value = [[0.1, 0.2, 0.3]]
        mock_embed.return_value = mock_embedder

        mock_vs = MagicMock()
        mock_vs_cls.return_value = mock_vs

        ingest()

        mock_makedirs.assert_called_once()
        mock_embedder.encode.assert_called_once_with(["hello world"])
        mock_vs.create_index.assert_called_once_with(dimension=3)
        mock_vs.add_documents.assert_called_once()
        mock_vs.save_index.assert_called_once()

    @patch("src.data.ingest.load_documents", return_value=[])
    @patch("src.data.ingest.os.makedirs")
    def test_no_documents_exits_early(self, mock_makedirs, mock_load):
        from src.data.ingest import ingest

        ingest()  # should not raise
        mock_makedirs.assert_called_once_with("data_store", exist_ok=True)

    @patch("src.data.ingest.VectorStore")
    @patch("src.data.ingest.get_embedder")
    @patch("src.data.ingest.load_documents")
    @patch("src.data.ingest.os.makedirs")
    def test_creates_index_dir_atomically(self, mock_makedirs, mock_load, mock_embed, mock_vs_cls):
        from src.data.ingest import ingest

        mock_load.return_value = [{"doc_id": "a_0", "text": "content", "source": "a.txt"}]
        mock_embedder = MagicMock()
        mock_embedder.encode.return_value = [[0.5]]
        mock_embed.return_value = mock_embedder
        mock_vs_cls.return_value = MagicMock()

        ingest()
        mock_makedirs.assert_called_once_with("data_store", exist_ok=True)

    @patch("src.data.ingest.VectorStore")
    @patch("src.data.ingest.get_embedder")
    @patch("src.data.ingest.load_documents")
    @patch("src.data.ingest.os.makedirs")
    def test_multiple_documents(self, mock_makedirs, mock_load, mock_embed, mock_vs_cls):
        from src.data.ingest import ingest

        mock_load.return_value = [
            {"doc_id": "f_0", "text": "first", "source": "f.txt"},
            {"doc_id": "f_1", "text": "second", "source": "f.txt"},
        ]
        mock_embedder = MagicMock()
        mock_embedder.encode.return_value = [[0.1, 0.2], [0.3, 0.4]]
        mock_embed.return_value = mock_embedder
        mock_vs = MagicMock()
        mock_vs_cls.return_value = mock_vs

        ingest()

        mock_embedder.encode.assert_called_once_with(["first", "second"])
        mock_vs.add_documents.assert_called_once_with(
            [[0.1, 0.2], [0.3, 0.4]], mock_load.return_value
        )


# ---------------------------------------------------------------------------
# get_llm_client / get_async_llm_client — cached singleton factories
# ---------------------------------------------------------------------------


class TestGetLlmClient:
    def setup_method(self):
        """Reset cached clients before each test."""
        import src.llm.synthesize as mod
        mod._llm_client = None
        mod._async_llm_client = None

    @patch("src.llm.synthesize.openai.OpenAI")
    @patch("src.llm.synthesize.os.getenv", side_effect=lambda k, *a: {"OPENAI_API_KEY": "sk-test"}.get(k))
    def test_creates_client_with_key(self, mock_env, mock_openai):
        from src.llm.synthesize import get_llm_client

        client = get_llm_client()
        mock_openai.assert_called_once_with(api_key="sk-test", timeout=30)
        assert client is mock_openai.return_value

    @patch("src.llm.synthesize.openai.OpenAI")
    @patch("src.llm.synthesize.os.getenv", side_effect=lambda k, *a: {"OPENAI_API_KEY": "sk-test"}.get(k))
    def test_caches_client(self, mock_env, mock_openai):
        from src.llm.synthesize import get_llm_client

        c1 = get_llm_client()
        c2 = get_llm_client()
        assert c1 is c2
        mock_openai.assert_called_once()  # only created once

    @patch("src.llm.synthesize.openai.OpenAI")
    @patch("src.llm.synthesize.os.getenv", side_effect=lambda k, *a: {"OPENAI_API_KEY": "sk-test", "OPENAI_BASE_URL": "http://proxy:4000/v1"}.get(k))
    def test_creates_client_with_base_url(self, mock_env, mock_openai):
        from src.llm.synthesize import get_llm_client

        get_llm_client()
        mock_openai.assert_called_once_with(api_key="sk-test", base_url="http://proxy:4000/v1", timeout=30)

    @patch("src.llm.synthesize.openai.OpenAI")
    @patch("src.llm.synthesize.os.getenv", return_value=None)
    def test_warns_on_missing_key(self, mock_env, mock_openai, capsys):
        from src.llm.synthesize import get_llm_client

        get_llm_client()
        captured = capsys.readouterr()
        assert "OPENAI_API_KEY not found" in captured.out


class TestGetAsyncLlmClient:
    def setup_method(self):
        import src.llm.synthesize as mod
        mod._llm_client = None
        mod._async_llm_client = None

    @patch("src.llm.synthesize.openai.AsyncOpenAI")
    @patch("src.llm.synthesize.os.getenv", side_effect=lambda k, *a: {"OPENAI_API_KEY": "sk-async"}.get(k))
    def test_creates_async_client(self, mock_env, mock_async):
        from src.llm.synthesize import get_async_llm_client

        client = get_async_llm_client()
        mock_async.assert_called_once_with(api_key="sk-async", timeout=30)
        assert client is mock_async.return_value

    @patch("src.llm.synthesize.openai.AsyncOpenAI")
    @patch("src.llm.synthesize.os.getenv", side_effect=lambda k, *a: {"OPENAI_API_KEY": "sk-async"}.get(k))
    def test_caches_async_client(self, mock_env, mock_async):
        from src.llm.synthesize import get_async_llm_client

        c1 = get_async_llm_client()
        c2 = get_async_llm_client()
        assert c1 is c2
        mock_async.assert_called_once()

    @patch("src.llm.synthesize.openai.AsyncOpenAI")
    @patch("src.llm.synthesize.os.getenv", return_value="")
    def test_warns_on_empty_key(self, mock_env, mock_async, capsys):
        from src.llm.synthesize import get_async_llm_client

        get_async_llm_client()
        captured = capsys.readouterr()
        assert "OPENAI_API_KEY not found" in captured.out
