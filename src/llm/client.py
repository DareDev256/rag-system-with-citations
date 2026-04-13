"""LLM client infrastructure — configuration, SSRF defense, and singletons.

Owns everything about *how* to connect to the LLM provider: environment-based
configuration, base URL validation (SSRF prevention), and thread-safe lazy
initialization of sync/async OpenAI clients.  The synthesis module imports
from here to get a ready-to-use client without caring about connection details.
"""

import asyncio
import ipaddress
import logging
import os
import threading
from typing import Any, Dict, List
from urllib.parse import urlparse

import openai

from src.utils.env import safe_int_env

logger = logging.getLogger("rag_api")

# ─── Model Configuration ────────────────────────────────────────────
DEFAULT_MODEL = "gpt-4o-mini"
CLASSIFICATION_MODEL = os.getenv("CLASSIFICATION_MODEL", "gpt-4o-mini")
SYNTHESIS_MODEL = os.getenv("SYNTHESIS_MODEL", DEFAULT_MODEL)

# Request timeout in seconds — prevents resource exhaustion from hung upstreams
LLM_TIMEOUT = safe_int_env("LLM_TIMEOUT", 30, min_val=1, max_val=300)

# ─── SSRF Prevention (CWE-918) ──────────────────────────────────────
_ALLOWED_SCHEMES = {"https", "http"}
_BLOCKED_HOSTS = frozenset({
    "169.254.169.254",           # AWS/GCP metadata
    "metadata.google.internal",  # GCP metadata
    "100.100.100.200",           # Alibaba Cloud metadata
})


_BLOCKED_HOSTNAMES = frozenset({"localhost", "localhost."})


def _is_private_ip(hostname: str) -> bool:
    """Check whether *hostname* is a private/loopback/link-local address.

    Catches SSRF attacks that bypass the hostname blocklist by using raw
    IPs (127.0.0.1, 10.x.x.x, 172.16.x.x, 192.168.x.x, ::1, fe80::, etc.),
    the special 0.0.0.0 address, or the ``localhost`` hostname.
    """
    if hostname.lower() in _BLOCKED_HOSTNAMES:
        return True
    try:
        addr = ipaddress.ip_address(hostname)
        return (
            addr.is_loopback        # 127.0.0.0/8, ::1
            or addr.is_private      # 10/8, 172.16/12, 192.168/16, fc00::/7
            or addr.is_link_local   # 169.254/16, fe80::/10
            or addr.is_reserved     # 0.0.0.0, 255.255.255.255, etc.
            or addr.is_multicast    # 224.0.0.0/4, ff00::/8
            or addr.is_unspecified  # 0.0.0.0, ::
        )
    except ValueError:
        return False  # Not an IP literal — hostname-based, checked against blocklist


def _validate_base_url(url: str) -> str:
    """Validate OPENAI_BASE_URL to prevent SSRF attacks.

    Defence layers:
    1. Scheme allowlist — only http:// and https://
    2. Hostname blocklist — known cloud metadata endpoints
    3. Private IP blocklist — loopback, RFC-1918, link-local, multicast

    Raises:
        ValueError: If the URL uses a disallowed scheme, targets a blocked
                    host, or resolves to a private IP range.
    """
    parsed = urlparse(url)
    if parsed.scheme not in _ALLOWED_SCHEMES:
        raise ValueError(
            f"OPENAI_BASE_URL scheme must be http or https, got '{parsed.scheme}'"
        )
    hostname = parsed.hostname or ""
    if not hostname:
        raise ValueError("OPENAI_BASE_URL has no hostname")
    if hostname in _BLOCKED_HOSTS:
        raise ValueError(
            f"OPENAI_BASE_URL targets a blocked metadata endpoint: {hostname}"
        )
    if _is_private_ip(hostname):
        raise ValueError(
            f"OPENAI_BASE_URL targets a private/loopback address: {hostname}"
        )
    return url


def _client_kwargs() -> dict:
    """Build shared kwargs for OpenAI client initialization.

    Supports OPENAI_BASE_URL for proxy compatibility — any OpenAI-compatible
    proxy (LiteLLM, OpenRouter, enterprise gateways) works by setting this
    env var alongside OPENAI_API_KEY. The URL is validated against an
    allowlist of schemes and a blocklist of cloud metadata endpoints to
    prevent SSRF (CWE-918).
    """
    kwargs = {}
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        logger.warning("OPENAI_API_KEY not found in env.")
    kwargs["api_key"] = api_key

    base_url = os.getenv("OPENAI_BASE_URL")
    if base_url:
        kwargs["base_url"] = _validate_base_url(base_url)

    return kwargs


# ─── Client Singletons ──────────────────────────────────────────────
# Guarded by locks to prevent duplicate initialization under concurrent
# access (CWE-362).
_llm_client = None
_async_llm_client = None
_sync_client_lock = threading.Lock()
_async_client_lock = None  # Lazy init — asyncio.Lock() requires a running event loop


def get_llm_client():
    global _llm_client
    if _llm_client is not None:
        return _llm_client
    with _sync_client_lock:
        # Double-check after acquiring lock — another thread may have initialized
        if _llm_client is None:
            _llm_client = openai.OpenAI(**_client_kwargs(), timeout=LLM_TIMEOUT)
    return _llm_client


async def get_async_llm_client():
    global _async_llm_client, _async_client_lock
    if _async_llm_client is not None:
        return _async_llm_client
    # Lazy-init the lock inside the running event loop to avoid binding
    # to a stale or non-existent loop at module import time (Python 3.10+
    # deprecation → 3.12+ RuntimeError).
    if _async_client_lock is None:
        _async_client_lock = asyncio.Lock()
    async with _async_client_lock:
        if _async_llm_client is None:
            _async_llm_client = openai.AsyncOpenAI(**_client_kwargs(), timeout=LLM_TIMEOUT)
    return _async_llm_client


# ─── Raw LLM Call Helpers ────────────────────────────────────────────

def call_llm(client: Any, model: str, messages: List[Dict[str, str]], **kwargs: Any) -> Any:
    """Sync LLM call — single entry point for all blocking OpenAI requests."""
    return client.chat.completions.create(model=model, messages=messages, **kwargs)


async def call_llm_async(client: Any, model: str, messages: List[Dict[str, str]], **kwargs: Any) -> Any:
    """Async LLM call — single entry point for all non-blocking OpenAI requests."""
    return await client.chat.completions.create(model=model, messages=messages, **kwargs)
