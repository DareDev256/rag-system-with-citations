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


_real_getenv = os.getenv


def _mock_getenv(key, *args, **kwargs):
    """Only fake OPENAI_API_KEY; let all other env vars pass through."""
    if key == "OPENAI_API_KEY":
        return "fake-key"
    return _real_getenv(key, *args, **kwargs)


@pytest.fixture(autouse=True)
def _skip_env_validation():
    with patch("src.api.main.os.getenv", side_effect=_mock_getenv):
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

    def test_csp_blocks_object_src(self, client):
        """object-src 'none' blocks Flash/Java plugin exploits."""
        csp = client.get("/health").headers["Content-Security-Policy"]
        assert "object-src 'none'" in csp

    def test_csp_restricts_base_uri(self, client):
        """base-uri 'self' prevents <base> tag injection attacks."""
        csp = client.get("/health").headers["Content-Security-Policy"]
        assert "base-uri 'self'" in csp

    def test_csp_restricts_form_action(self, client):
        """form-action 'self' blocks cross-origin form submissions."""
        csp = client.get("/health").headers["Content-Security-Policy"]
        assert "form-action 'self'" in csp

    def test_csp_upgrade_insecure(self, client):
        """upgrade-insecure-requests auto-upgrades HTTP to HTTPS."""
        csp = client.get("/health").headers["Content-Security-Policy"]
        assert "upgrade-insecure-requests" in csp

    @patch.dict(os.environ, {"CSP_POLICY": "default-src 'none'"})
    def test_csp_custom_override(self, client):
        """CSP_POLICY env var overrides the default policy."""
        csp = client.get("/health").headers["Content-Security-Policy"]
        assert csp == "default-src 'none'"

    def test_hsts_header(self, client):
        resp = client.get("/health")
        hsts = resp.headers["Strict-Transport-Security"]
        assert "max-age=63072000" in hsts
        assert "includeSubDomains" in hsts
        assert "preload" in hsts

    def test_cross_domain_policies(self, client):
        resp = client.get("/health")
        assert resp.headers["X-Permitted-Cross-Domain-Policies"] == "none"

    def test_dns_prefetch_control(self, client):
        """X-DNS-Prefetch-Control prevents DNS prefetching data leaks."""
        assert client.get("/health").headers["X-DNS-Prefetch-Control"] == "off"

    def test_cache_control_no_store(self, client):
        """API responses must not be cached (sensitive query data)."""
        assert client.get("/health").headers["Cache-Control"] == "no-store"

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

    def test_context_containing_query_placeholder_not_expanded(self):
        """If a document contains literal '{query}', it must NOT be replaced
        with the user's actual query — that would corrupt context data."""
        result = format_rag_prompt(
            "Tutorial: use {query} as a variable name",
            "how do I format strings?"
        )
        # The {query} inside the context must survive as-is
        assert "{query}" in result
        # The user's actual query appears once in the User Question slot
        assert "User Question: how do I format strings?" in result

    def test_mutual_cross_replacement_safety(self):
        """Neither context nor query should be able to inject into the other's slot."""
        result = format_rag_prompt("{query}", "{context_str}")
        # Context slot contains literal {query}
        assert "Context:\n{query}" in result
        # Query slot contains literal {context_str}
        assert "User Question: {context_str}" in result


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


# ── LLM Timeout Configuration ───────────────────────────────────


class TestLLMTimeout:
    """Verify OpenAI clients are created with timeout to prevent resource exhaustion."""

    def test_default_timeout_value(self):
        from src.llm.synthesize import _LLM_TIMEOUT
        assert _LLM_TIMEOUT == 30

    def test_client_created_with_timeout(self):
        """get_llm_client passes timeout kwarg to OpenAI constructor."""
        import src.llm.synthesize as synth
        synth._llm_client = None  # force re-creation
        with patch("src.llm.synthesize.openai.OpenAI") as mock_cls:
            mock_cls.return_value = "fake-client"
            with patch("src.llm.synthesize._client_kwargs", return_value={"api_key": "k"}):
                synth.get_llm_client()
            mock_cls.assert_called_once_with(api_key="k", timeout=30)
        synth._llm_client = None

    @pytest.mark.asyncio
    async def test_async_client_created_with_timeout(self):
        """get_async_llm_client passes timeout kwarg to AsyncOpenAI constructor."""
        import src.llm.synthesize as synth
        synth._async_llm_client = None
        with patch("src.llm.synthesize.openai.AsyncOpenAI") as mock_cls:
            mock_cls.return_value = "fake-async"
            with patch("src.llm.synthesize._client_kwargs", return_value={"api_key": "k"}):
                await synth.get_async_llm_client()
            mock_cls.assert_called_once_with(api_key="k", timeout=30)
        synth._async_llm_client = None


# ── Error Message Sanitization ───────────────────────────────────


class TestErrorSanitization:
    """Error handlers must not leak internal details (API keys, paths, etc)."""

    def test_classify_error_logs_type_not_message(self):
        """classify_query logs exception type name, not the full message."""
        import src.llm.synthesize as synth
        synth._llm_client = None
        mock_client = patch("src.llm.synthesize.get_llm_client").start()
        mock_client.return_value.chat.completions.create.side_effect = (
            RuntimeError("Connection to sk-secret-key-here failed")
        )
        with patch("src.llm.synthesize.logger") as mock_logger:
            result = synth.classify_query("test")
            assert result == "exploratory"
            log_msg = mock_logger.error.call_args[0][1]
            assert log_msg == "RuntimeError"
            assert "sk-secret" not in str(mock_logger.error.call_args)
        patch.stopall()

    def test_synthesize_error_returns_safe_message(self):
        """synthesize_answer returns generic error, not exception details."""
        import src.llm.synthesize as synth
        synth._llm_client = None
        mock_client = patch("src.llm.synthesize.get_llm_client").start()
        mock_client.return_value.chat.completions.create.side_effect = (
            ConnectionError("upstream at internal-proxy.corp:4000 refused")
        )
        result = synth.synthesize_answer("q", [{"doc_id": "d", "snippet": "s"}])
        assert result["answer"] == "Error generating answer."
        assert "internal-proxy" not in result["answer"]
        patch.stopall()


# ── Log Injection Control Chars ──────────────────────────────────


class TestLogInjectionControlChars:
    """Query logging must strip all control characters, not just \\n and \\r."""

    @patch("src.api.main.perform_search", return_value=[])
    @patch("src.api.main.synthesize_answer_async", new_callable=AsyncMock)
    @patch("src.api.main.classify_query_async", new_callable=AsyncMock)
    def test_null_bytes_stripped_from_log(self, mock_c, mock_s, mock_p, client):
        mock_c.return_value = "factual"
        mock_s.return_value = {"answer": "a", "citations_used": [], "confidence": 0.5}
        with patch("src.api.main.logger") as mock_log:
            client.post("/query", json={"query": "hello\x00world\x08test"})
            logged = mock_log.info.call_args[0][1]
            assert "\x00" not in logged
            assert "\x08" not in logged

    @patch("src.api.main.perform_search", return_value=[])
    @patch("src.api.main.synthesize_answer_async", new_callable=AsyncMock)
    @patch("src.api.main.classify_query_async", new_callable=AsyncMock)
    def test_tab_and_escape_stripped(self, mock_c, mock_s, mock_p, client):
        mock_c.return_value = "factual"
        mock_s.return_value = {"answer": "a", "citations_used": [], "confidence": 0.5}
        with patch("src.api.main.logger") as mock_log:
            client.post("/query", json={"query": "tab\there\x1bescape"})
            logged = mock_log.info.call_args[0][1]
            assert "\t" not in logged
            assert "\x1b" not in logged


# ── HSTS_MAX_AGE Validation ──────────────────────────────────────


class TestHSTSMaxAgeValidation:
    """_parse_hsts_max_age must handle invalid env var values gracefully."""

    def test_valid_value(self):
        from src.api.main import _parse_hsts_max_age
        with patch("src.api.main.os.getenv", return_value="3600"):
            assert _parse_hsts_max_age() == 3600

    def test_non_numeric_falls_back(self):
        from src.api.main import _parse_hsts_max_age
        with patch("src.api.main.os.getenv", return_value="not_a_number"):
            assert _parse_hsts_max_age() == 63072000

    def test_negative_falls_back(self):
        from src.api.main import _parse_hsts_max_age
        with patch("src.api.main.os.getenv", return_value="-1"):
            assert _parse_hsts_max_age() == 63072000


# ── Path Traversal Guard (load_documents) ────────────────────────


class TestPathTraversalGuard:
    """load_documents must reject symlinks that escape the corpus directory."""

    def test_symlink_outside_corpus_skipped(self, tmp_path):
        """A symlink pointing outside corpus_dir should be rejected."""
        outside = tmp_path / "outside"
        outside.mkdir()
        secret = outside / "leaked.txt"
        secret.write_text("secret data")

        corpus = tmp_path / "corpus"
        corpus.mkdir()
        (corpus / "legit.txt").write_text("Safe content.")
        link = corpus / "evil.txt"
        link.symlink_to(secret)

        docs = load_documents(str(corpus))
        sources = [d["source"] for d in docs]
        assert "legit.txt" in sources
        # Symlink target is outside corpus — should be skipped
        assert "leaked.txt" not in sources
        assert "evil.txt" not in sources


# ── Rate Limiting (CWE-770) ──────────────────────────────────────


class TestRateLimiting:
    """In-memory rate limiter protects /query from abuse and budget drain."""

    @patch("src.api.main.perform_search", return_value=[])
    @patch("src.api.main.synthesize_answer_async", new_callable=AsyncMock)
    @patch("src.api.main.classify_query_async", new_callable=AsyncMock)
    def test_requests_within_limit_succeed(self, mock_c, mock_s, mock_p, client):
        """Requests under the rate limit should succeed normally."""
        mock_c.return_value = "factual"
        mock_s.return_value = {"answer": "ok", "citations_used": [], "confidence": 0.5}
        resp = client.post("/query", json={"query": "test"})
        assert resp.status_code == 200

    @patch("src.api.main.perform_search", return_value=[])
    @patch("src.api.main.synthesize_answer_async", new_callable=AsyncMock)
    @patch("src.api.main.classify_query_async", new_callable=AsyncMock)
    def test_exceeding_rate_limit_returns_429(self, mock_c, mock_s, mock_p, client):
        """Exceeding RATE_LIMIT_RPM should return 429 Too Many Requests."""
        mock_c.return_value = "factual"
        mock_s.return_value = {"answer": "ok", "citations_used": [], "confidence": 0.5}
        from src.api.main import _rate_store, _RATE_LIMIT
        import time as t
        # Stuff the rate store to simulate exhausted quota
        _rate_store["testclient"] = [t.monotonic()] * (_RATE_LIMIT + 1)
        resp = client.post("/query", json={"query": "should fail"})
        assert resp.status_code == 429
        assert "Rate limit" in resp.json()["detail"]
        _rate_store.clear()  # cleanup

    def test_rate_limit_has_security_headers(self, client):
        """429 responses must still include security headers."""
        from src.api.main import _rate_store, _RATE_LIMIT
        import time as t
        _rate_store["testclient"] = [t.monotonic()] * (_RATE_LIMIT + 1)
        resp = client.post("/query", json={"query": "blocked"})
        assert resp.status_code == 429
        assert resp.headers.get("X-Content-Type-Options") == "nosniff"
        _rate_store.clear()


# ── Output Sanitization (LLM responses) ──────────────────────────


class TestOutputSanitization:
    """LLM output and reflected query must be sanitized before returning to client."""

    @patch("src.api.main.perform_search", return_value=[])
    @patch("src.api.main.synthesize_answer_async", new_callable=AsyncMock)
    @patch("src.api.main.classify_query_async", new_callable=AsyncMock)
    def test_control_chars_stripped_from_answer(self, mock_c, mock_s, mock_p, client):
        """Control chars in LLM output must not reach the client."""
        mock_c.return_value = "factual"
        mock_s.return_value = {
            "answer": "Hello\x00world\x08hidden\x1b[31mred",
            "citations_used": [],
            "confidence": 0.5,
        }
        resp = client.post("/query", json={"query": "test"})
        answer = resp.json()["answer"]
        assert "\x00" not in answer
        assert "\x08" not in answer
        assert "\x1b" not in answer
        assert "Helloworld" in answer

    @patch("src.api.main.perform_search", return_value=[])
    @patch("src.api.main.synthesize_answer_async", new_callable=AsyncMock)
    @patch("src.api.main.classify_query_async", new_callable=AsyncMock)
    def test_newlines_preserved_in_answer(self, mock_c, mock_s, mock_p, client):
        """Legitimate whitespace (newlines, tabs) in LLM output should survive."""
        mock_c.return_value = "factual"
        mock_s.return_value = {
            "answer": "Line 1\nLine 2\n\tIndented",
            "citations_used": [],
            "confidence": 0.5,
        }
        resp = client.post("/query", json={"query": "test"})
        answer = resp.json()["answer"]
        assert "\n" in answer
        assert "\t" in answer

    @patch("src.api.main.perform_search", return_value=[])
    @patch("src.api.main.synthesize_answer_async", new_callable=AsyncMock)
    @patch("src.api.main.classify_query_async", new_callable=AsyncMock)
    def test_reflected_query_sanitized(self, mock_c, mock_s, mock_p, client):
        """The echoed query in the response must be sanitized too."""
        mock_c.return_value = "factual"
        mock_s.return_value = {
            "answer": "ok",
            "citations_used": [],
            "confidence": 0.5,
        }
        resp = client.post("/query", json={"query": "test\x00inject\x1b"})
        echoed = resp.json()["query"]
        assert "\x00" not in echoed
        assert "\x1b" not in echoed

    def test_sanitize_output_unit(self):
        """Direct unit test for _sanitize_output."""
        from src.api.main import _sanitize_output
        assert _sanitize_output("clean") == "clean"
        assert _sanitize_output("a\x00b\x08c\x7fd") == "abcd"
        assert _sanitize_output("keep\nnewlines\tand\rtabs") == "keep\nnewlines\tand\rtabs"


# ── LLM_TIMEOUT Safe Parsing ─────────────────────────────────────


class TestLLMTimeoutParsing:
    """_parse_llm_timeout must handle invalid values like _parse_hsts_max_age."""

    def test_valid_value(self):
        from src.llm.synthesize import _parse_llm_timeout
        with patch("src.llm.synthesize.os.getenv", return_value="60"):
            assert _parse_llm_timeout() == 60

    def test_non_numeric_falls_back(self):
        from src.llm.synthesize import _parse_llm_timeout
        with patch("src.llm.synthesize.os.getenv", return_value="abc"):
            assert _parse_llm_timeout() == 30

    def test_zero_falls_back(self):
        from src.llm.synthesize import _parse_llm_timeout
        with patch("src.llm.synthesize.os.getenv", return_value="0"):
            assert _parse_llm_timeout() == 30

    def test_negative_falls_back(self):
        from src.llm.synthesize import _parse_llm_timeout
        with patch("src.llm.synthesize.os.getenv", return_value="-5"):
            assert _parse_llm_timeout() == 30


# ── Rate Limiter Memory Exhaustion (CWE-400) ─────────────────────


class TestRateLimiterMemory:
    """_rate_store must not grow unboundedly from IP rotation attacks."""

    def test_evict_stale_ips_removes_empty_entries(self):
        """IPs with no recent requests should be evicted."""
        from src.api.main import _evict_stale_ips, _rate_store
        _rate_store.clear()
        now = 1000.0
        _rate_store["stale-ip"] = [now - 120]  # expired (>60s ago)
        _rate_store["active-ip"] = [now - 10]   # still within window
        _rate_store["empty-ip"] = []             # empty list, should be evicted
        _evict_stale_ips(now)
        assert "stale-ip" not in _rate_store
        assert "empty-ip" not in _rate_store
        assert "active-ip" in _rate_store
        _rate_store.clear()

    def test_max_tracked_ips_triggers_eviction(self):
        """When IP count exceeds _MAX_TRACKED_IPS, stale entries are purged."""
        from src.api.main import _check_rate_limit, _rate_store, _MAX_TRACKED_IPS
        import time as t
        _rate_store.clear()
        now = t.monotonic()
        # Fill with stale IPs (timestamps expired)
        for i in range(_MAX_TRACKED_IPS + 1):
            _rate_store[f"ip-{i}"] = [now - 120]
        # Next check should trigger eviction
        _check_rate_limit("trigger-ip")
        assert len(_rate_store) < _MAX_TRACKED_IPS
        _rate_store.clear()


# ── Global Exception Handler (CWE-209) ───────────────────────────


class TestGlobalExceptionHandler:
    """Unhandled exceptions must not leak internal details to clients."""

    @patch("src.api.main.perform_search", side_effect=RuntimeError("secret internal path /opt/data"))
    @patch("src.api.main.classify_query_async", new_callable=AsyncMock, return_value="factual")
    def test_500_hides_internal_details(self, mock_c, mock_p, client):
        """Stack traces and internal paths must not appear in 500 responses."""
        resp = client.post("/query", json={"query": "crash me"})
        assert resp.status_code == 500
        body = resp.json()
        assert body["detail"] == "Internal server error."
        assert "/opt/data" not in str(body)
        assert "RuntimeError" not in str(body)

    @patch("src.api.main.perform_search", side_effect=ValueError("bad value"))
    @patch("src.api.main.classify_query_async", new_callable=AsyncMock, return_value="factual")
    def test_500_includes_request_id(self, mock_c, mock_p, client):
        """500 responses must include a request_id for incident correlation."""
        resp = client.post("/query", json={"query": "crash"})
        assert resp.status_code == 500
        assert "request_id" in resp.json()
        assert len(resp.json()["request_id"]) > 0

    @patch("src.api.main.perform_search", side_effect=Exception("boom"))
    @patch("src.api.main.classify_query_async", new_callable=AsyncMock, return_value="factual")
    def test_500_has_security_headers(self, mock_c, mock_p, client):
        """Security headers must still appear on 500 responses."""
        resp = client.post("/query", json={"query": "test"})
        assert resp.status_code == 500
        assert resp.headers.get("X-Content-Type-Options") == "nosniff"


# ── Request ID Middleware ─────────────────────────────────────────


class TestRequestID:
    """Every response must include X-Request-ID for security traceability."""

    def test_health_returns_request_id(self, client):
        """Even simple endpoints get a request ID."""
        resp = client.get("/health")
        assert "X-Request-ID" in resp.headers
        assert len(resp.headers["X-Request-ID"]) == 32  # uuid4 hex

    def test_client_provided_id_echoed(self, client):
        """Valid client-provided X-Request-ID should be preserved."""
        resp = client.get("/health", headers={"X-Request-ID": "client-trace-abc123"})
        assert resp.headers["X-Request-ID"] == "client-trace-abc123"

    def test_oversized_id_rejected(self, client):
        """Client IDs longer than 64 chars are replaced with a server-generated one."""
        long_id = "x" * 100
        resp = client.get("/health", headers={"X-Request-ID": long_id})
        assert resp.headers["X-Request-ID"] != long_id
        assert len(resp.headers["X-Request-ID"]) == 32

    def test_control_chars_in_id_rejected(self, client):
        """Client IDs with control characters are replaced to prevent log injection."""
        resp = client.get("/health", headers={"X-Request-ID": "bad\x00id\x1b"})
        assert "\x00" not in resp.headers["X-Request-ID"]
        assert len(resp.headers["X-Request-ID"]) == 32
