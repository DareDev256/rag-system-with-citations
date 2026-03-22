"""Latency measurement decorator for pipeline instrumentation.

Automatically injects ``latency_ms`` into the wrapped function's return
value (dict or object with a ``latency_ms`` attribute), giving every
pipeline stage free timing without manual ``perf_counter`` bookkeeping.
"""

import time
import functools
import logging

logger = logging.getLogger(__name__)


def measure_latency(func):
    """Decorator that times *func* and injects ``latency_ms`` into its result.

    If the result is a ``dict`` without a ``latency_ms`` key, the key is
    added.  If the result is an object with a ``latency_ms`` attribute, the
    attribute is set.  Other return types pass through unchanged.

    Timing uses ``time.perf_counter`` for sub-millisecond resolution.
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        end_time = time.perf_counter()
        latency_ms = (end_time - start_time) * 1000
        # Check if result is a dict (like our API response) and inject latency
        if isinstance(result, dict) and 'latency_ms' not in result:
             result['latency_ms'] = round(latency_ms, 2)
        elif hasattr(result, 'latency_ms'): # For objects
             result.latency_ms = round(latency_ms, 2)
        
        logger.debug("Function %s took %.2fms", func.__name__, latency_ms)
        return result
    return wrapper
