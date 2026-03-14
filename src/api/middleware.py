"""Middleware stack for the RAG API.

Centralizes all request/response middleware — rate limiting, authentication,
security headers, body size enforcement, request ID tracking, and output
sanitization — so main.py stays focused on endpoint orchestration.
"""

import hmac
import logging
import os
import re
import time
import uuid
from collections import defaultdict

from fastapi import Request
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import Response

from src.utils.env import safe_int_env
from src.utils.ip import resolve_client_ip

logger = logging.getLogger("rag_api")


# ─── Output sanitization ─────────────────────────────────────────────
# LLM responses are untrusted — strip control chars before returning to client.
CONTROL_CHAR_RE = re.compile(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]')


def sanitize_output(text: str) -> str:
    """Strip C0 control chars from LLM output (preserves \\n, \\r, \\t)."""
    return CONTROL_CHAR_RE.sub('', text)


# ─── Request Body Size Limit (CWE-400) ───────────────────────────────
_MAX_BODY_BYTES = safe_int_env("MAX_BODY_BYTES", 65_536, min_val=1024)


class MaxBodySizeMiddleware(BaseHTTPMiddleware):
    """Reject requests with bodies exceeding _MAX_BODY_BYTES before parsing."""

    async def dispatch(self, request: Request, call_next):
        if request.method in ("POST", "PUT", "PATCH"):
            content_length = request.headers.get("content-length")
            if content_length:
                try:
                    if int(content_length) > _MAX_BODY_BYTES:
                        return JSONResponse(
                            status_code=413,
                            content={"detail": "Request body too large."},
                        )
                except ValueError:
                    return JSONResponse(
                        status_code=400,
                        content={"detail": "Invalid Content-Length header."},
                    )
        return await call_next(request)


# ─── Rate Limiter (in-memory, per-IP) ────────────────────────────────
_RATE_LIMIT = safe_int_env("RATE_LIMIT_RPM", 30, min_val=1)
_MAX_TRACKED_IPS = 10_000
_rate_store: dict = defaultdict(list)


def check_rate_limit(client_ip: str) -> bool:
    """Return True if request is allowed, False if rate-limited."""
    now = time.monotonic()
    window = now - 60
    _rate_store[client_ip] = [t for t in _rate_store[client_ip] if t > window]
    if len(_rate_store[client_ip]) >= _RATE_LIMIT:
        return False
    _rate_store[client_ip].append(now)
    if len(_rate_store) > _MAX_TRACKED_IPS:
        _evict_stale_ips(now)
    return True


def _evict_stale_ips(now: float) -> None:
    """Remove IPs with no recent requests from the rate store."""
    window = now - 60
    stale = [ip for ip, ts in _rate_store.items() if not ts or ts[-1] <= window]
    for ip in stale:
        del _rate_store[ip]


# ─── API Key Authentication (CWE-862) ────────────────────────────────
_API_KEYS: set = set()
_raw_keys = os.getenv("API_KEYS", "").strip()
if _raw_keys:
    _API_KEYS = {k.strip() for k in _raw_keys.split(",") if k.strip()}
    logger.info("API key auth enabled (%d key(s) loaded)", len(_API_KEYS))

_PUBLIC_PATHS = {"/health", "/docs", "/redoc", "/openapi.json"}


def authenticate(request: Request) -> str | None:
    """Return an error message if auth fails, None if OK."""
    if not _API_KEYS:
        return None
    if request.url.path in _PUBLIC_PATHS or request.method == "GET":
        return None
    auth_header = request.headers.get("authorization", "")
    if auth_header.startswith("Bearer "):
        token = auth_header[7:]
        if any(hmac.compare_digest(token, k) for k in _API_KEYS):
            return None
    api_key = request.headers.get("x-api-key", "")
    if api_key and any(hmac.compare_digest(api_key, k) for k in _API_KEYS):
        return None
    return "Invalid or missing API key."


# ─── Security Headers ────────────────────────────────────────────────
_DEFAULT_CSP = (
    "default-src 'self'; "
    "script-src 'self'; "
    "style-src 'self'; "
    "img-src 'self' data:; "
    "font-src 'self'; "
    "connect-src 'self'; "
    "object-src 'none'; "
    "base-uri 'self'; "
    "form-action 'self'; "
    "frame-ancestors 'none'; "
    "upgrade-insecure-requests"
)

_HSTS_MAX_AGE = safe_int_env("HSTS_MAX_AGE", 63072000, min_val=0)

_SECURITY_HEADERS = {
    "X-Content-Type-Options": "nosniff",
    "X-Frame-Options": "DENY",
    "X-DNS-Prefetch-Control": "off",
    "Referrer-Policy": "strict-origin-when-cross-origin",
    "Permissions-Policy": "camera=(), microphone=(), geolocation=()",
    "X-Permitted-Cross-Domain-Policies": "none",
    "Cache-Control": "no-store",
}


def apply_security_headers(response: Response) -> None:
    """Apply all security headers — shared by middleware and exception handler."""
    for header, value in _SECURITY_HEADERS.items():
        response.headers[header] = value
    response.headers["Content-Security-Policy"] = os.getenv("CSP_POLICY", _DEFAULT_CSP)
    response.headers["Strict-Transport-Security"] = (
        f"max-age={_HSTS_MAX_AGE}; includeSubDomains; preload"
    )


class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        response: Response = await call_next(request)
        apply_security_headers(response)
        return response


# ─── Request ID Middleware ───────────────────────────────────────────
class RequestIDMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        request_id = request.headers.get("X-Request-ID", "")
        if not request_id or len(request_id) > 64 or CONTROL_CHAR_RE.search(request_id):
            request_id = uuid.uuid4().hex
        request.state.request_id = request_id
        response = await call_next(request)
        response.headers["X-Request-ID"] = request_id
        return response
