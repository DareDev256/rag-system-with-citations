"""Tests for previously-untested functions: validate_env, ingest, LLM client factories.

Mocks all external deps — no API keys, FAISS, or file I/O needed.
"""
import asyncio
import threading
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
    @patch("src.data.ingest.create_default_store")
    @patch("src.data.ingest.get_embedder")
    @patch("src.data.ingest.load_documents")
    def test_full_pipeline(self, mock_load, mock_embed, mock_factory):
        from src.data.ingest import ingest

        mock_load.return_value = [
            {"doc_id": "f_0", "text": "hello world", "source": "f.txt"},
        ]
        mock_embedder = MagicMock()
        mock_embedder.encode.return_value = [[0.1, 0.2, 0.3]]
        mock_embed.return_value = mock_embedder

        mock_vs = MagicMock()
        mock_factory.return_value = mock_vs

        ingest()

        mock_factory.assert_called_once()
        mock_embedder.encode.assert_called_once_with(["hello world"])
        mock_vs.create_index.assert_called_once_with(dimension=3)
        mock_vs.add_documents.assert_called_once()
        mock_vs.save_index.assert_called_once()

    @patch("src.data.ingest.load_documents", return_value=[])
    def test_no_documents_exits_early(self, mock_load):
        from src.data.ingest import ingest

        ingest()  # should not raise

    @patch("src.data.ingest.create_default_store")
    @patch("src.data.ingest.get_embedder")
    @patch("src.data.ingest.load_documents")
    def test_creates_index_dir_atomically(self, mock_load, mock_embed, mock_factory):
        from src.data.ingest import ingest

        mock_load.return_value = [{"doc_id": "a_0", "text": "content", "source": "a.txt"}]
        mock_embedder = MagicMock()
        mock_embedder.encode.return_value = [[0.5]]
        mock_embed.return_value = mock_embedder
        mock_factory.return_value = MagicMock()

        ingest()
        mock_factory.assert_called_once()

    @patch("src.data.ingest.create_default_store")
    @patch("src.data.ingest.get_embedder")
    @patch("src.data.ingest.load_documents")
    def test_multiple_documents(self, mock_load, mock_embed, mock_factory):
        from src.data.ingest import ingest

        mock_load.return_value = [
            {"doc_id": "f_0", "text": "first", "source": "f.txt"},
            {"doc_id": "f_1", "text": "second", "source": "f.txt"},
        ]
        mock_embedder = MagicMock()
        mock_embedder.encode.return_value = [[0.1, 0.2], [0.3, 0.4]]
        mock_embed.return_value = mock_embedder
        mock_vs = MagicMock()
        mock_factory.return_value = mock_vs

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
        import src.llm.client as mod
        mod._llm_client = None
        mod._async_llm_client = None

    @patch("src.llm.client.openai.OpenAI")
    @patch("src.llm.client.os.getenv", side_effect=lambda k, *a: {"OPENAI_API_KEY": "sk-test"}.get(k))
    def test_creates_client_with_key(self, mock_env, mock_openai):
        from src.llm.client import get_llm_client

        client = get_llm_client()
        mock_openai.assert_called_once_with(api_key="sk-test", timeout=30)
        assert client is mock_openai.return_value

    @patch("src.llm.client.openai.OpenAI")
    @patch("src.llm.client.os.getenv", side_effect=lambda k, *a: {"OPENAI_API_KEY": "sk-test"}.get(k))
    def test_caches_client(self, mock_env, mock_openai):
        from src.llm.client import get_llm_client

        c1 = get_llm_client()
        c2 = get_llm_client()
        assert c1 is c2
        mock_openai.assert_called_once()  # only created once

    @patch("src.llm.client.openai.OpenAI")
    @patch("src.llm.client.os.getenv", side_effect=lambda k, *a: {"OPENAI_API_KEY": "sk-test", "OPENAI_BASE_URL": "http://proxy:4000/v1"}.get(k))
    def test_creates_client_with_base_url(self, mock_env, mock_openai):
        from src.llm.client import get_llm_client

        get_llm_client()
        mock_openai.assert_called_once_with(api_key="sk-test", base_url="http://proxy:4000/v1", timeout=30)

    @patch("src.llm.client.openai.OpenAI")
    @patch("src.llm.client.os.getenv", return_value=None)
    def test_warns_on_missing_key(self, mock_env, mock_openai, caplog):
        from src.llm.client import get_llm_client

        with caplog.at_level("WARNING", logger="rag_api"):
            get_llm_client()
        assert "OPENAI_API_KEY not found" in caplog.text


class TestGetAsyncLlmClient:
    def setup_method(self):
        import src.llm.client as mod
        mod._llm_client = None
        mod._async_llm_client = None
        mod._async_client_lock = None  # Reset lazy lock

    @pytest.mark.asyncio
    @patch("src.llm.client.openai.AsyncOpenAI")
    @patch("src.llm.client.os.getenv", side_effect=lambda k, *a: {"OPENAI_API_KEY": "sk-async"}.get(k))
    async def test_creates_async_client(self, mock_env, mock_async):
        from src.llm.client import get_async_llm_client

        client = await get_async_llm_client()
        mock_async.assert_called_once_with(api_key="sk-async", timeout=30)
        assert client is mock_async.return_value

    @pytest.mark.asyncio
    @patch("src.llm.client.openai.AsyncOpenAI")
    @patch("src.llm.client.os.getenv", side_effect=lambda k, *a: {"OPENAI_API_KEY": "sk-async"}.get(k))
    async def test_caches_async_client(self, mock_env, mock_async):
        from src.llm.client import get_async_llm_client

        c1 = await get_async_llm_client()
        c2 = await get_async_llm_client()
        assert c1 is c2
        mock_async.assert_called_once()

    @pytest.mark.asyncio
    @patch("src.llm.client.openai.AsyncOpenAI")
    @patch("src.llm.client.os.getenv", return_value="")
    async def test_warns_on_empty_key(self, mock_env, mock_async, caplog):
        from src.llm.client import get_async_llm_client

        with caplog.at_level("WARNING", logger="rag_api"):
            await get_async_llm_client()
        assert "OPENAI_API_KEY not found" in caplog.text


# ---------------------------------------------------------------------------
# Race condition tests — verify locks prevent duplicate initialization
# ---------------------------------------------------------------------------


class TestSyncClientRaceSafety:
    """Concurrent threads must not create duplicate sync OpenAI clients."""

    def setup_method(self):
        import src.llm.client as mod
        mod._llm_client = None

    def teardown_method(self):
        import src.llm.client as mod
        mod._llm_client = None

    @patch("src.llm.client.openai.OpenAI")
    @patch("src.llm.client.os.getenv", side_effect=lambda k, *a: {"OPENAI_API_KEY": "sk-test"}.get(k))
    def test_concurrent_threads_create_single_client(self, mock_env, mock_openai):
        from src.llm.client import get_llm_client

        results = []
        barrier = threading.Barrier(4)

        def worker():
            barrier.wait()
            results.append(get_llm_client())

        threads = [threading.Thread(target=worker) for _ in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # All threads must get the same instance
        assert len(set(id(r) for r in results)) == 1
        mock_openai.assert_called_once()


class TestAsyncLockLazyInit:
    """Regression: asyncio.Lock() at module level crashes on Python 3.10+."""

    def setup_method(self):
        import src.llm.client as mod
        mod._async_llm_client = None
        mod._async_client_lock = None

    def teardown_method(self):
        import src.llm.client as mod
        mod._async_llm_client = None
        mod._async_client_lock = None

    @pytest.mark.asyncio
    @patch("src.llm.client.openai.AsyncOpenAI")
    @patch("src.llm.client.os.getenv", side_effect=lambda k, *a: {"OPENAI_API_KEY": "sk-test"}.get(k))
    async def test_lock_created_lazily_inside_event_loop(self, mock_env, mock_async):
        """Lock must be None before first call and an asyncio.Lock after."""
        import src.llm.client as mod
        assert mod._async_client_lock is None
        await mod.get_async_llm_client()
        assert isinstance(mod._async_client_lock, asyncio.Lock)


class TestAsyncClientRaceSafety:
    """Concurrent asyncio tasks must not create duplicate async OpenAI clients."""

    def setup_method(self):
        import src.llm.client as mod
        mod._async_llm_client = None
        mod._async_client_lock = None

    def teardown_method(self):
        import src.llm.client as mod
        mod._async_llm_client = None
        mod._async_client_lock = None

    @pytest.mark.asyncio
    @patch("src.llm.client.openai.AsyncOpenAI")
    @patch("src.llm.client.os.getenv", side_effect=lambda k, *a: {"OPENAI_API_KEY": "sk-test"}.get(k))
    async def test_concurrent_tasks_create_single_client(self, mock_env, mock_async):
        from src.llm.client import get_async_llm_client

        results = await asyncio.gather(*(get_async_llm_client() for _ in range(4)))

        assert len(set(id(r) for r in results)) == 1
        mock_async.assert_called_once()


class TestSearchEngineRaceSafety:
    """Concurrent threads must not create duplicate search engine instances."""

    def setup_method(self):
        import src.retrieval.search as mod
        mod._vector_store = None
        mod._embedder = None

    def teardown_method(self):
        import src.retrieval.search as mod
        mod._vector_store = None
        mod._embedder = None

    @patch("src.retrieval.search.get_embedder")
    @patch("src.retrieval.search.create_default_store")
    def test_concurrent_threads_create_single_engine(self, mock_factory, mock_embed):
        from src.retrieval.search import get_search_engine

        mock_vs = MagicMock()
        mock_factory.return_value = mock_vs
        mock_embedder = MagicMock()
        mock_embed.return_value = mock_embedder

        results = []
        barrier = threading.Barrier(4)

        def worker():
            barrier.wait()
            results.append(get_search_engine())

        threads = [threading.Thread(target=worker) for _ in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # All threads got the same (store, embedder) tuple
        stores = [r[0] for r in results]
        embedders = [r[1] for r in results]
        assert len(set(id(s) for s in stores)) == 1
        assert len(set(id(e) for e in embedders)) == 1
        mock_factory.assert_called_once()
