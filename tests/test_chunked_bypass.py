"""Tests for chunked transfer encoding body size enforcement.

Verifies that MaxBodySizeMiddleware correctly limits body size for
requests without Content-Length (chunked encoding), reading bytes
incrementally via ASGI receive wrapping instead of buffering the
full payload into memory before checking (CWE-400).
"""

import json

import pytest
from unittest.mock import AsyncMock

from src.api.middleware import MaxBodySizeMiddleware, _MAX_BODY_BYTES


# ── Helpers ──────────────────────────────────────────────────────


def _make_scope(method="POST", headers=None):
    """Build a minimal ASGI HTTP scope."""
    raw_headers = []
    for k, v in (headers or {}).items():
        raw_headers.append([k.encode(), v.encode()])
    return {"type": "http", "method": method, "headers": raw_headers, "path": "/query"}


def _chunked_receive(chunks):
    """Async receive that yields body chunks without Content-Length."""
    it = iter(chunks)

    async def receive():
        try:
            chunk = next(it)
            return {"type": "http.request", "body": chunk, "more_body": True}
        except StopIteration:
            return {"type": "http.request", "body": b"", "more_body": False}

    return receive


async def _passthrough_app(scope, receive, send):
    """Minimal ASGI app that reads the full body."""
    while True:
        msg = await receive()
        if not msg.get("more_body", False):
            break


async def _collect_send(responses):
    """Return a send callable that appends messages to the list."""

    async def send(msg):
        responses.append(msg)

    return send


# ── Tests ────────────────────────────────────────────────────────


class TestChunkedBodyEnforcement:
    """MaxBodySizeMiddleware with chunked (no Content-Length) requests."""

    @pytest.mark.asyncio
    async def test_small_chunked_body_passes(self):
        """Chunked request under the limit reaches the downstream app."""
        app_called = False

        async def app(scope, receive, send):
            nonlocal app_called
            await _passthrough_app(scope, receive, send)
            app_called = True

        mw = MaxBodySizeMiddleware(app)
        scope = _make_scope(headers={"content-type": "application/json"})
        await mw(scope, _chunked_receive([b'{"query": "hi"}']), AsyncMock())
        assert app_called

    @pytest.mark.asyncio
    async def test_oversized_chunked_body_returns_413(self):
        """Chunked request exceeding limit gets 413 without full buffering."""
        app_called = False

        async def app(scope, receive, send):
            nonlocal app_called
            await _passthrough_app(scope, receive, send)
            app_called = True

        mw = MaxBodySizeMiddleware(app)
        scope = _make_scope(headers={"content-type": "application/json"})
        chunk = b"x" * 1024
        num_chunks = (_MAX_BODY_BYTES // 1024) + 2
        receive = _chunked_receive([chunk] * num_chunks)

        responses = []
        await mw(scope, receive, await _collect_send(responses))
        assert not app_called
        assert responses[0]["status"] == 413
        body = json.loads(responses[1]["body"])
        assert "too large" in body["detail"].lower()

    @pytest.mark.asyncio
    async def test_exact_limit_chunked_passes(self):
        """Chunked body exactly at the byte limit should not be rejected."""
        app_called = False

        async def app(scope, receive, send):
            nonlocal app_called
            await _passthrough_app(scope, receive, send)
            app_called = True

        mw = MaxBodySizeMiddleware(app)
        scope = _make_scope()
        await mw(scope, _chunked_receive([b"x" * _MAX_BODY_BYTES]), AsyncMock())
        assert app_called

    @pytest.mark.asyncio
    async def test_one_byte_over_limit_rejected(self):
        """Chunked body at limit + 1 byte triggers 413."""
        app_called = False

        async def app(scope, receive, send):
            nonlocal app_called
            await _passthrough_app(scope, receive, send)
            app_called = True

        mw = MaxBodySizeMiddleware(app)
        scope = _make_scope()
        responses = []
        await mw(
            scope,
            _chunked_receive([b"x" * (_MAX_BODY_BYTES + 1)]),
            await _collect_send(responses),
        )
        assert not app_called
        assert responses[0]["status"] == 413

    @pytest.mark.asyncio
    async def test_get_bypasses_chunked_check(self):
        """GET requests skip body size enforcement entirely."""
        app_called = False

        async def app(scope, receive, send):
            nonlocal app_called
            app_called = True

        mw = MaxBodySizeMiddleware(app)
        scope = _make_scope(method="GET")
        await mw(scope, AsyncMock(), AsyncMock())
        assert app_called

    @pytest.mark.asyncio
    async def test_early_abort_limits_memory_usage(self):
        """Middleware aborts after exceeding limit — does NOT read all chunks."""
        chunks_read = 0

        async def counting_receive():
            nonlocal chunks_read
            chunks_read += 1
            if chunks_read <= 100:
                return {"type": "http.request", "body": b"x" * _MAX_BODY_BYTES, "more_body": True}
            return {"type": "http.request", "body": b"", "more_body": False}

        mw = MaxBodySizeMiddleware(_passthrough_app)
        scope = _make_scope()
        responses = []
        await mw(scope, counting_receive, await _collect_send(responses))

        # Must abort after ≤2 chunks, not read all 100
        assert chunks_read <= 2, f"Read {chunks_read} chunks — should have aborted early"
        assert responses[0]["status"] == 413

    @pytest.mark.asyncio
    async def test_multi_chunk_accumulation(self):
        """Multiple small chunks that together exceed the limit get rejected."""
        mw = MaxBodySizeMiddleware(_passthrough_app)
        scope = _make_scope()
        # Each chunk is half the limit; 3 chunks = 1.5x limit
        half = _MAX_BODY_BYTES // 2
        responses = []
        await mw(
            scope,
            _chunked_receive([b"x" * half, b"x" * half, b"x" * half]),
            await _collect_send(responses),
        )
        assert responses[0]["status"] == 413

    @pytest.mark.asyncio
    async def test_non_http_scope_passes_through(self):
        """WebSocket and other non-HTTP scopes bypass the middleware."""
        app_called = False

        async def app(scope, receive, send):
            nonlocal app_called
            app_called = True

        mw = MaxBodySizeMiddleware(app)
        await mw({"type": "websocket"}, AsyncMock(), AsyncMock())
        assert app_called
