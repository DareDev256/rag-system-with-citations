"""LLM client infrastructure — configuration, SSRF defense, and singletons.

Owns everything about *how* to connect to the LLM provider: environment-based
configuration, base URL validation (SSRF prevention), and thread-safe lazy
initialization of sync/async OpenAI clients.  The synthesis module imports
from here to get a ready-to-use client without caring about connection details.
"""

import asyncio
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
LLM_TIMEOUT = safe_int_env("LLM_TIMEOUT", 30, min_val=1)

# ─── SSRF Prevention (CWE-918) ──────────────────────────────────────
_ALLOWED_SCHEMES = {"https", "http"}
_BLOCKED_HOSTS = frozenset({
    "169.254.169.254",           # AWS/GCP metadata
    "metadata.google.internal",  # GCP metadata
    "100.100.100.200",           # Alibaba Cloud metadata
})


def _validate_base_url(url: str) -> str:
    """Validate OPENAI_BASE_URL to prevent SSRF attacks.

    Only allows http:// and https:// schemes; blocks known cloud metadata
    endpoints and rejects file://, ftp://, gopher://, etc.

    Raises:
        ValueError: If the URL uses a disallowed scheme or targets a blocked host.
    """
    parsed = urlparse(url)
    if parsed.scheme not in _ALLOWED_SCHEMES:
        raise ValueError(
            f"OPENAI_BASE_URL scheme must be http or https, got '{parsed.scheme}'"
        )
    hostname = parsed.hostname or ""
    if hostname in _BLOCKED_HOSTS:
        raise ValueError(
            f"OPENAI_BASE_URL targets a blocked metadata endpoint: {hostname}"
        )
    if not hostname:
        raise ValueError("OPENAI_BASE_URL has no hostname")
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
