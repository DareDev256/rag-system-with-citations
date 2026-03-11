"""Tests for trusted proxy IP resolution (CWE-348).

Validates that resolve_client_ip correctly extracts real client IPs from
X-Forwarded-For when behind reverse proxies, and falls back safely when not.
"""
from unittest.mock import MagicMock, patch

import pytest

from src.utils.ip import resolve_client_ip, _is_valid_ip


# ─── Helper ──────────────────────────────────────────────────────────
def _make_request(socket_ip="127.0.0.1", forwarded_for=None):
    """Create a mock FastAPI Request with optional X-Forwarded-For."""
    req = MagicMock()
    req.client.host = socket_ip
    headers = {}
    if forwarded_for is not None:
        headers["x-forwarded-for"] = forwarded_for
    req.headers = headers
    return req


# ─── IP Validation ───────────────────────────────────────────────────
class TestIsValidIp:
    def test_valid_ipv4(self):
        assert _is_valid_ip("192.168.1.1") is True

    def test_valid_ipv6(self):
        assert _is_valid_ip("::1") is True

    def test_invalid_string(self):
        assert _is_valid_ip("not-an-ip") is False

    def test_empty_string(self):
        assert _is_valid_ip("") is False

    def test_injection_attempt(self):
        assert _is_valid_ip("127.0.0.1; DROP TABLE") is False


# ─── Proxy Disabled (default) ───────────────────────────────────────
class TestProxyDisabled:
    """TRUSTED_PROXY_COUNT=0 — always use socket IP, ignore X-Forwarded-For."""

    @patch("src.utils.ip._TRUSTED_PROXY_COUNT", 0)
    def test_returns_socket_ip(self):
        req = _make_request(socket_ip="10.0.0.1")
        assert resolve_client_ip(req) == "10.0.0.1"

    @patch("src.utils.ip._TRUSTED_PROXY_COUNT", 0)
    def test_ignores_forwarded_header(self):
        req = _make_request(socket_ip="10.0.0.1", forwarded_for="1.2.3.4")
        assert resolve_client_ip(req) == "10.0.0.1"

    @patch("src.utils.ip._TRUSTED_PROXY_COUNT", 0)
    def test_no_client_returns_unknown(self):
        req = MagicMock()
        req.client = None
        req.headers = {}
        assert resolve_client_ip(req) == "unknown"


# ─── Single Proxy (e.g., nginx → app) ───────────────────────────────
class TestSingleProxy:
    """TRUSTED_PROXY_COUNT=1 — one reverse proxy between client and app."""

    @patch("src.utils.ip._TRUSTED_PROXY_COUNT", 1)
    def test_extracts_client_ip(self):
        # nginx appends client IP: X-Forwarded-For: real_client, nginx_ip
        req = _make_request(forwarded_for="203.0.113.50, 10.0.0.1")
        assert resolve_client_ip(req) == "203.0.113.50"

    @patch("src.utils.ip._TRUSTED_PROXY_COUNT", 1)
    def test_attacker_spoofed_prefix_ignored(self):
        # Attacker sends: X-Forwarded-For: fake_ip
        # Proxy appends: X-Forwarded-For: fake_ip, real_client, proxy
        req = _make_request(forwarded_for="6.6.6.6, 203.0.113.50, 10.0.0.1")
        assert resolve_client_ip(req) == "203.0.113.50"

    @patch("src.utils.ip._TRUSTED_PROXY_COUNT", 1)
    def test_missing_header_falls_back_to_socket(self):
        req = _make_request(socket_ip="10.0.0.1")
        assert resolve_client_ip(req) == "10.0.0.1"

    @patch("src.utils.ip._TRUSTED_PROXY_COUNT", 1)
    def test_too_few_entries_falls_back_to_socket(self):
        # Only 1 entry but need at least 2 (client + proxy)
        req = _make_request(socket_ip="10.0.0.1", forwarded_for="203.0.113.50")
        assert resolve_client_ip(req) == "10.0.0.1"

    @patch("src.utils.ip._TRUSTED_PROXY_COUNT", 1)
    def test_invalid_ip_in_header_falls_back(self):
        req = _make_request(socket_ip="10.0.0.1", forwarded_for="not-valid, 10.0.0.1")
        assert resolve_client_ip(req) == "10.0.0.1"


# ─── Double Proxy (e.g., CDN → LB → app) ────────────────────────────
class TestDoubleProxy:
    """TRUSTED_PROXY_COUNT=2 — CDN and load balancer in front of app."""

    @patch("src.utils.ip._TRUSTED_PROXY_COUNT", 2)
    def test_extracts_client_through_two_proxies(self):
        # X-Forwarded-For: real_client, cdn_ip, lb_ip
        req = _make_request(forwarded_for="203.0.113.50, 198.51.100.1, 10.0.0.1")
        assert resolve_client_ip(req) == "203.0.113.50"

    @patch("src.utils.ip._TRUSTED_PROXY_COUNT", 2)
    def test_spoofed_prefix_with_two_proxies(self):
        # Attacker spoofs: fake, then real path: real_client, cdn, lb
        req = _make_request(forwarded_for="6.6.6.6, 203.0.113.50, 198.51.100.1, 10.0.0.1")
        assert resolve_client_ip(req) == "203.0.113.50"

    @patch("src.utils.ip._TRUSTED_PROXY_COUNT", 2)
    def test_too_few_entries_for_double_proxy(self):
        # Only 2 entries but need at least 3
        req = _make_request(socket_ip="10.0.0.1", forwarded_for="203.0.113.50, 10.0.0.1")
        assert resolve_client_ip(req) == "10.0.0.1"


# ─── IPv6 Support ────────────────────────────────────────────────────
class TestIPv6:
    @patch("src.utils.ip._TRUSTED_PROXY_COUNT", 1)
    def test_ipv6_client(self):
        req = _make_request(forwarded_for="2001:db8::1, 10.0.0.1")
        assert resolve_client_ip(req) == "2001:db8::1"

    @patch("src.utils.ip._TRUSTED_PROXY_COUNT", 0)
    def test_ipv6_socket(self):
        req = _make_request(socket_ip="::1")
        assert resolve_client_ip(req) == "::1"
