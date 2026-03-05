"""Hardening tests — security headers, prompt injection resistance, file I/O resilience.

These tests verify defensive behaviors that protect the system in production:
- SecurityHeadersMiddleware sets all required headers on every response
- Prompt templates resist format-string injection from user input
- Document ingestion gracefully skips bad files instead of aborting
- _client_kwargs builds correct OpenAI client config from env vars
"""
import os
import stat
from unittest.mock import patch, AsyncMock

import pytest
from fastapi.testclient import TestClient

from src.llm.prompt import format_rag_prompt, format_classification_prompt
from src.data.ingest import load_documents


# ── Security Headers Middleware ──────────────────────────────────


@pytest.fixture(autouse=True)
def _skip_env_validation():
    with patch("src.api.main.os.getenv", return_value="fake-key"):
        yield


@pytest.fixture()
def client():
    from src.api.main import app
    return TestClient(app, raise_server_exceptions=False)


class TestSecurityHeaders:
    """Verify every security header is present on responses."""

    def test_x_content_type_options(self, client):
        resp = client.get("/health")
        assert resp.headers["X-Content-Type-Options"] == "nosniff"

    def test_x_frame_options(self, client):
        resp = client.get("/health")
        assert resp.headers["X-Frame-Options"] == "DENY"

    def test_referrer_policy(self, client):
        resp = client.get("/health")
        assert resp.headers["Referrer-Policy"] == "strict-origin-when-cross-origin"

    def test_permissions_policy(self, client):
        resp = client.get("/health")
        assert "camera=()" in resp.headers["Permissions-Policy"]
        assert "microphone=()" in resp.headers["Permissions-Policy"]

    def test_csp_header(self, client):
        resp = client.get("/health")
        csp = resp.headers["Content-Security-Policy"]
        assert "default-src 'self'" in csp
        assert "frame-ancestors 'none'" in csp

    def test_hsts_header(self, client):
        resp = client.get("/health")
        hsts = resp.headers["Strict-Transport-Security"]
        assert "max-age=63072000" in hsts
        assert "includeSubDomains" in hsts
        assert "preload" in hsts

    def test_cross_domain_policies(self, client):
        resp = client.get("/health")
        assert resp.headers["X-Permitted-Cross-Domain-Policies"] == "none"

    def test_headers_present_on_error_responses(self, client):
        """Security headers must appear even on 422 validation errors."""
        resp = client.post("/query", json={})
        assert resp.status_code == 422
        assert resp.headers.get("X-Content-Type-Options") == "nosniff"
        assert "Content-Security-Policy" in resp.headers

    @patch("src.api.main.perform_search", return_value=[])
    @patch("src.api.main.synthesize_answer_async", new_callable=AsyncMock)
    @patch("src.api.main.classify_query_async", new_callable=AsyncMock)
    def test_headers_present_on_query_responses(self, mock_c, mock_s, mock_p, client):
        """Security headers must appear on successful /query responses too."""
        mock_c.return_value = "factual"
        mock_s.return_value = {"answer": "x", "citations_used": [], "confidence": 0.0}
        resp = client.post("/query", json={"query": "test"})
        assert resp.status_code == 200
        assert resp.headers.get("Strict-Transport-Security", "").startswith("max-age=")


# ── Prompt Injection Resistance ──────────────────────────────────


class TestPromptInjection:
    """format_rag_prompt/format_classification_prompt use .replace() instead of
    .format() specifically to prevent user queries containing {curly_braces}
    from causing KeyError or injecting template variables."""

    def test_rag_prompt_with_format_placeholders(self):
        """User query containing {context_str} must not expand the template variable."""
        result = format_rag_prompt("real context", "{context_str}")
        assert "{context_str}" in result  # literal, not expanded
        assert "real context" in result

    def test_rag_prompt_with_python_format_syntax(self):
        """Curly braces like {0}, {key} must pass through safely."""
        result = format_rag_prompt("ctx", "What is {0} and {key}?")
        assert "{0}" in result
        assert "{key}" in result

    def test_classification_prompt_with_braces(self):
        result = format_classification_prompt("{query}")
        # The literal {query} should appear in the output (user's input)
        # but the template's {query} placeholder should be replaced
        count = result.count("{query}")
        assert count == 1  # only the user's literal input, not the template var

    def test_rag_prompt_preserves_all_template_structure(self):
        """Verify the prompt still has the expected structure after replacement."""
        result = format_rag_prompt("my context", "my question")
        assert "Context:" in result
        assert "my context" in result
        assert "User Question: my question" in result
        assert "Answer (include citations" in result

    def test_rag_prompt_with_percent_format(self):
        """Percent-style format strings %(var)s should not cause issues."""
        result = format_rag_prompt("ctx", "%(query)s and %(context_str)s")
        assert "%(query)s" in result


# ── File I/O Error Resilience (load_documents) ──────────────────


class TestLoadDocumentsErrorPaths:
    """load_documents should skip bad files and continue ingesting the rest."""

    def test_encoding_error_skipped(self, tmp_path):
        """Binary file disguised as .txt should be skipped, not crash."""
        bad_file = tmp_path / "binary.txt"
        bad_file.write_bytes(b"\x80\x81\x82\x83" * 100)
        good_file = tmp_path / "good.txt"
        good_file.write_text("Valid content here.", encoding="utf-8")

        docs = load_documents(str(tmp_path))
        # Good file loaded, bad file skipped
        assert len(docs) == 1
        assert docs[0]["source"] == "good.txt"

    def test_permission_denied_skipped(self, tmp_path):
        """Unreadable file should be skipped gracefully."""
        blocked = tmp_path / "secret.txt"
        blocked.write_text("classified info")
        blocked.chmod(0o000)

        good = tmp_path / "public.txt"
        good.write_text("Open content.")

        try:
            docs = load_documents(str(tmp_path))
            sources = [d["source"] for d in docs]
            assert "public.txt" in sources
            assert "secret.txt" not in sources
        finally:
            blocked.chmod(stat.S_IRUSR | stat.S_IWUSR)  # restore for cleanup

    def test_empty_file_produces_no_docs(self, tmp_path):
        """A .txt file with only whitespace should produce no documents."""
        (tmp_path / "blank.txt").write_text("   \n\n   \n  ")
        docs = load_documents(str(tmp_path))
        assert docs == []

    def test_mixed_good_and_bad_files(self, tmp_path):
        """Multiple bad files don't prevent good files from loading."""
        (tmp_path / "a.txt").write_text("Alpha content.")
        (tmp_path / "bad1.txt").write_bytes(b"\xff\xfe" + b"\x00" * 50)
        (tmp_path / "b.txt").write_text("Beta content.")
        (tmp_path / "bad2.txt").write_bytes(bytes(range(128, 256)))

        docs = load_documents(str(tmp_path))
        sources = {d["source"] for d in docs}
        assert "a.txt" in sources
        assert "b.txt" in sources


# ── _client_kwargs Configuration ─────────────────────────────────


class TestClientKwargs:
    """Test the _client_kwargs helper that builds OpenAI client config."""

    def test_key_only(self):
        from src.llm.synthesize import _client_kwargs
        with patch("src.llm.synthesize.os.getenv", side_effect=lambda k, *a: {"OPENAI_API_KEY": "sk-test"}.get(k)):
            kwargs = _client_kwargs()
            assert kwargs == {"api_key": "sk-test"}

    def test_key_and_base_url(self):
        from src.llm.synthesize import _client_kwargs
        env = {"OPENAI_API_KEY": "sk-x", "OPENAI_BASE_URL": "http://proxy:4000/v1"}
        with patch("src.llm.synthesize.os.getenv", side_effect=lambda k, *a: env.get(k)):
            kwargs = _client_kwargs()
            assert kwargs["api_key"] == "sk-x"
            assert kwargs["base_url"] == "http://proxy:4000/v1"

    def test_missing_key_still_returns_none(self):
        from src.llm.synthesize import _client_kwargs
        with patch("src.llm.synthesize.os.getenv", return_value=None):
            kwargs = _client_kwargs()
            assert kwargs["api_key"] is None
            assert "base_url" not in kwargs

    def test_empty_base_url_not_included(self):
        from src.llm.synthesize import _client_kwargs
        env = {"OPENAI_API_KEY": "sk-y", "OPENAI_BASE_URL": ""}
        with patch("src.llm.synthesize.os.getenv", side_effect=lambda k, *a: env.get(k)):
            kwargs = _client_kwargs()
            # Empty string is falsy, should not be included
            assert "base_url" not in kwargs
