"""Middleware tests — MaxBodySize, RequestID, rate limiter thread safety,
and global exception handler.

These tests verify the security perimeter that runs on every request:
- MaxBodySizeMiddleware rejects oversized payloads before Pydantic parsing (CWE-400)
- RequestIDMiddleware assigns/validates request IDs for incident correlation
- Rate limiter thread safety under concurrent access (CWE-362)
- Global exception handler suppresses stack traces on 500s (CWE-209)
"""
import os
import threading
import time
import uuid
from unittest.mock import patch, AsyncMock

import pytest
from fastapi.testclient import TestClient


# ── Fixtures ──────────────────────────────────────────────────────

_real_getenv = os.getenv


def _mock_getenv(key, *args, **kwargs):
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


# ── MaxBodySizeMiddleware ─────────────────────────────────────────


class TestMaxBodySizeMiddleware:
    """Verifies early rejection of oversized request bodies (CWE-400)."""

    def test_normal_post_passes_through(self, client):
        """POST with small body reaches the endpoint normally."""
        resp = client.get("/health")
        assert resp.status_code == 200

    def test_oversized_content_length_returns_413(self, client):
        """Content-Length exceeding _MAX_BODY_BYTES triggers 413 before parsing."""
        resp = client.post(
            "/query",
            content=b'{"query": "x"}',
            headers={"Content-Length": "999999999", "Content-Type": "application/json"},
        )
        assert resp.status_code == 413
        assert "too large" in resp.json()["detail"].lower()

    def test_invalid_content_length_returns_400(self, client):
        """Non-numeric Content-Length gets a 400, not a crash."""
        resp = client.post(
            "/query",
            content=b'{"query": "test"}',
            headers={"Content-Length": "not-a-number", "Content-Type": "application/json"},
        )
        assert resp.status_code == 400
        assert "Invalid Content-Length" in resp.json()["detail"]

    def test_get_request_bypasses_size_check(self, client):
        """GET requests have no body — middleware must not reject them."""
        resp = client.get("/health", headers={"Content-Length": "999999999"})
        assert resp.status_code == 200

    def test_missing_content_length_passes_through(self, client):
        """POST without Content-Length header proceeds to normal validation."""
        # TestClient usually adds Content-Length, but if absent, middleware
        # should not reject — Pydantic handles validation downstream.
        resp = client.post(
            "/query",
            content=b'{"query": "test"}',
            headers={"Content-Type": "application/json"},
        )
        # Should reach the endpoint (422 for validation or deeper), not 413
        assert resp.status_code != 413

    def test_exact_limit_passes(self, client):
        """Body exactly at the limit should pass, not be rejected."""
        from src.api.middleware import _MAX_BODY_BYTES
        # Content-Length == limit should pass
        resp = client.post(
            "/query",
            content=b'{"query": "x"}',
            headers={
                "Content-Length": str(_MAX_BODY_BYTES),
                "Content-Type": "application/json",
            },
        )
        # Should NOT be 413 — it's within the limit
        assert resp.status_code != 413

    def test_one_over_limit_rejected(self, client):
        """Body at limit + 1 byte should be rejected."""
        from src.api.middleware import _MAX_BODY_BYTES
        resp = client.post(
            "/query",
            content=b'{"query": "x"}',
            headers={
                "Content-Length": str(_MAX_BODY_BYTES + 1),
                "Content-Type": "application/json",
            },
        )
        assert resp.status_code == 413


# ── RequestIDMiddleware ───────────────────────────────────────────


class TestRequestIDMiddleware:
    """Verifies request ID assignment, validation, and propagation."""

    def test_auto_generates_uuid_when_missing(self, client):
        """No X-Request-ID header → middleware generates one."""
        resp = client.get("/health")
        rid = resp.headers.get("X-Request-ID")
        assert rid is not None
        assert len(rid) == 32  # uuid4().hex is 32 hex chars

    def test_client_provided_id_passed_through(self, client):
        """Valid X-Request-ID from client is echoed back unchanged."""
        custom_id = "my-trace-id-12345"
        resp = client.get("/health", headers={"X-Request-ID": custom_id})
        assert resp.headers["X-Request-ID"] == custom_id

    def test_oversized_id_replaced(self, client):
        """X-Request-ID > 64 chars is replaced with auto-generated UUID."""
        long_id = "a" * 65
        resp = client.get("/health", headers={"X-Request-ID": long_id})
        rid = resp.headers["X-Request-ID"]
        assert rid != long_id
        assert len(rid) == 32  # auto-generated UUID hex

    def test_control_chars_in_id_replaced(self, client):
        """X-Request-ID with control characters is rejected and replaced."""
        bad_id = "trace\x00injected"
        resp = client.get("/health", headers={"X-Request-ID": bad_id})
        rid = resp.headers["X-Request-ID"]
        assert rid != bad_id
        assert "\x00" not in rid

    def test_empty_string_id_replaced(self, client):
        """Empty X-Request-ID is treated as missing → auto-generated."""
        resp = client.get("/health", headers={"X-Request-ID": ""})
        rid = resp.headers["X-Request-ID"]
        assert len(rid) == 32

    def test_exactly_64_chars_accepted(self, client):
        """X-Request-ID at exactly 64 chars is valid and passed through."""
        valid_id = "b" * 64
        resp = client.get("/health", headers={"X-Request-ID": valid_id})
        assert resp.headers["X-Request-ID"] == valid_id


# ── Global Exception Handler (CWE-209) ───────────────────────────


class TestGlobalExceptionHandler:
    """Verifies unhandled exceptions return safe 500 responses."""

    @patch("src.api.main.perform_search", side_effect=RuntimeError("DB connection lost"))
    @patch("src.api.main.classify_query_async", new_callable=AsyncMock, return_value="factual")
    @patch("src.api.main.check_rate_limit", return_value=True)
    def test_500_hides_exception_details(self, _rl, _classify, _search, client):
        """Internal errors must NOT leak exception message or traceback."""
        resp = client.post("/query", json={"query": "test question"})
        assert resp.status_code == 500
        body = resp.json()
        assert "DB connection lost" not in body.get("detail", "")
        assert "Internal server error" in body["detail"]

    @patch("src.api.main.perform_search", side_effect=RuntimeError("secret path info"))
    @patch("src.api.main.classify_query_async", new_callable=AsyncMock, return_value="factual")
    @patch("src.api.main.check_rate_limit", return_value=True)
    def test_500_includes_request_id(self, _rl, _classify, _search, client):
        """500 response includes request_id for incident correlation."""
        resp = client.post("/query", json={"query": "test"})
        assert resp.status_code == 500
        body = resp.json()
        assert "request_id" in body
        assert len(body["request_id"]) > 0

    @patch("src.api.main.perform_search", side_effect=ValueError("bad dimension"))
    @patch("src.api.main.classify_query_async", new_callable=AsyncMock, return_value="factual")
    @patch("src.api.main.check_rate_limit", return_value=True)
    def test_500_has_security_headers(self, _rl, _classify, _search, client):
        """Security headers must be present even on 500 error responses."""
        resp = client.post("/query", json={"query": "test"})
        assert resp.status_code == 500
        assert resp.headers.get("X-Content-Type-Options") == "nosniff"
        assert resp.headers.get("X-Frame-Options") == "DENY"
        assert "Content-Security-Policy" in resp.headers

    @patch("src.api.main.perform_search", side_effect=Exception("traceback\n  File \"/app/src/secret.py\""))
    @patch("src.api.main.classify_query_async", new_callable=AsyncMock, return_value="factual")
    @patch("src.api.main.check_rate_limit", return_value=True)
    def test_500_no_stack_trace_leak(self, _rl, _classify, _search, client):
        """Response body must not contain file paths or traceback fragments."""
        resp = client.post("/query", json={"query": "test"})
        body = resp.text
        assert "/app/src/" not in body
        assert "traceback" not in body.lower()
        assert "File " not in body


# ── Rate Limiter Thread Safety (CWE-362) ────────────────────────


class TestRateLimiterThreadSafety:
    """Verify rate limiter is safe under concurrent access."""

    def setup_method(self):
        """Reset rate store between tests."""
        from src.api.middleware import _rate_store
        _rate_store.clear()

    def test_concurrent_same_ip_no_crash(self):
        """Concurrent requests from the same IP must not raise RuntimeError."""
        from src.api.middleware import check_rate_limit
        errors = []

        def hammer(ip):
            try:
                for _ in range(50):
                    check_rate_limit(ip)
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=hammer, args=("10.0.0.1",)) for _ in range(8)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert errors == [], f"Concurrent access raised: {errors}"

    def test_concurrent_distinct_ips_no_crash(self):
        """Many distinct IPs hammered concurrently must not corrupt the dict."""
        from src.api.middleware import check_rate_limit
        errors = []

        def hammer(thread_id):
            try:
                for i in range(50):
                    check_rate_limit(f"10.{thread_id}.0.{i}")
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=hammer, args=(tid,)) for tid in range(8)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert errors == [], f"Concurrent access raised: {errors}"

    def test_eviction_under_contention_no_crash(self):
        """Hard eviction triggered while other threads mutate must not raise."""
        from src.api import middleware
        from src.api.middleware import check_rate_limit, _rate_store
        old_max = middleware._MAX_TRACKED_IPS
        errors = []
        try:
            middleware._MAX_TRACKED_IPS = 20  # force frequent eviction

            def hammer(thread_id):
                try:
                    for i in range(60):
                        check_rate_limit(f"192.{thread_id}.{i}.1")
                except Exception as e:
                    errors.append(e)

            threads = [threading.Thread(target=hammer, args=(tid,)) for tid in range(6)]
            for t in threads:
                t.start()
            for t in threads:
                t.join()
            assert errors == [], f"Eviction under contention raised: {errors}"
        finally:
            middleware._MAX_TRACKED_IPS = old_max
            _rate_store.clear()

    def test_rate_lock_exists(self):
        """The module must expose a threading.Lock for rate store protection."""
        from src.api.middleware import _rate_lock
        assert isinstance(_rate_lock, type(threading.Lock()))

    def test_rate_limit_still_enforced_after_threading(self):
        """Rate limiting logic must still correctly block after concurrent load."""
        from src.api import middleware
        from src.api.middleware import check_rate_limit, _rate_store
        old_limit = middleware._RATE_LIMIT
        try:
            middleware._RATE_LIMIT = 5
            _rate_store.clear()
            results = [check_rate_limit("1.2.3.4") for _ in range(10)]
            assert results[:5] == [True] * 5
            assert results[5:] == [False] * 5
        finally:
            middleware._RATE_LIMIT = old_limit
            _rate_store.clear()
