"""Trusted proxy IP resolution (CWE-348).

Behind a reverse proxy (nginx, ALB, Cloudflare), req.client.host returns the
proxy IP — not the real client. This collapses all clients into one rate-limit
bucket, making rate limiting useless and enabling DoS amplification.

Fix: when TRUSTED_PROXY_COUNT is set, extract the real client IP from
X-Forwarded-For by counting back from the rightmost entry (which the last
trusted proxy appended). Counting from the RIGHT is critical — an attacker can
prepend arbitrary entries to X-Forwarded-For, but cannot forge the entries
added by your own infrastructure.

When TRUSTED_PROXY_COUNT=0 (default), the socket IP is used directly.
"""

import ipaddress
import logging
import os
import re

from fastapi import Request

from src.utils.env import safe_int_env

logger = logging.getLogger("rag_api")

_TRUSTED_PROXY_COUNT = safe_int_env("TRUSTED_PROXY_COUNT", 0, min_val=0)

# Validate extracted IPs — reject anything that isn't a real address
_IP_RE = re.compile(r'^[\d.]+$|^[0-9a-fA-F:]+$')


def _is_valid_ip(value: str) -> bool:
    """Check if a string is a valid IPv4 or IPv6 address."""
    try:
        ipaddress.ip_address(value)
        return True
    except ValueError:
        return False


def resolve_client_ip(request: Request) -> str:
    """Extract the real client IP, respecting trusted proxy configuration.

    With TRUSTED_PROXY_COUNT=0: returns socket IP (default, safe for direct clients).
    With TRUSTED_PROXY_COUNT=N: picks the Nth-from-right entry in X-Forwarded-For.

    Example: X-Forwarded-For: attacker_fake, real_client, proxy1
    With TRUSTED_PROXY_COUNT=1 (one proxy between us and client):
      → picks real_client (index -2, which is len-1-TRUSTED_PROXY_COUNT)

    Falls back to socket IP if the header is missing, malformed, or too short.
    """
    socket_ip = request.client.host if request.client else "unknown"

    if _TRUSTED_PROXY_COUNT == 0:
        return socket_ip

    forwarded = request.headers.get("x-forwarded-for", "")
    if not forwarded:
        return socket_ip

    # Split and strip — proxies append comma-separated IPs
    parts = [p.strip() for p in forwarded.split(",") if p.strip()]

    # We need at least TRUSTED_PROXY_COUNT + 1 entries:
    # [client, ..., proxy1, ..., proxyN]  where proxyN is the one that set the header
    if len(parts) < _TRUSTED_PROXY_COUNT + 1:
        logger.warning(
            "X-Forwarded-For has %d entries but TRUSTED_PROXY_COUNT=%d; using socket IP",
            len(parts), _TRUSTED_PROXY_COUNT,
        )
        return socket_ip

    # The client IP is at index -(TRUSTED_PROXY_COUNT + 1)
    candidate = parts[-(1 + _TRUSTED_PROXY_COUNT)]

    if not _is_valid_ip(candidate):
        logger.warning("Invalid IP '%s' in X-Forwarded-For; using socket IP", candidate[:64])
        return socket_ip

    return candidate
