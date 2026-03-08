"""Safe environment variable parsing with defensive fallbacks."""

import logging
import os

logger = logging.getLogger("rag_api")


def safe_int_env(key: str, default: int, *, min_val: int = None) -> int:
    """Parse an integer env var with bounds checking and graceful fallback.

    Consolidates the defensive parsing pattern used across the codebase
    (LLM_TIMEOUT, HSTS_MAX_AGE, RATE_LIMIT_RPM) into a single function.
    Invalid, out-of-bounds, or missing values log a warning and return the default.
    """
    raw = os.getenv(key, str(default))
    try:
        val = int(raw)
        if min_val is not None and val < min_val:
            raise ValueError
        return val
    except (ValueError, TypeError):
        logger.warning("Invalid %s='%s', using default %d", key, raw, default)
        return default
