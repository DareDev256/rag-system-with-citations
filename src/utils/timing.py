"""Latency measurement utilities for pipeline instrumentation.

Provides two complementary timing tools:

* **measure_latency** — decorator that auto-injects ``latency_ms`` into a
  function's return value.
* **TimingContext** — context manager for inline timing blocks, replacing
  the repetitive ``start = perf_counter(); …; ms = (perf_counter() - start) * 1000``
  pattern used across the query endpoint and evaluation pipeline.
"""

import time
import functools
import logging

logger = logging.getLogger(__name__)


class TimingContext:
    """Context manager that captures elapsed wall-clock time in milliseconds.

    Usage::

        with TimingContext() as t:
            do_work()
        print(t.ms)  # elapsed milliseconds (float, rounded to 2dp)
    """

    __slots__ = ("ms", "_start")

    def __init__(self) -> None:
        self.ms: float = 0.0
        self._start: float = 0.0

    def __enter__(self) -> "TimingContext":
        self._start = time.perf_counter()
        return self

    def __exit__(self, *exc) -> bool:
        self.ms = round((time.perf_counter() - self._start) * 1000, 2)
        return False


def measure_latency(func):
    """Decorator that times *func* and injects ``latency_ms`` into its result.

    If the result is a ``dict`` without a ``latency_ms`` key, the key is
    added.  If the result is an object with a ``latency_ms`` attribute, the
    attribute is set.  Other return types pass through unchanged.

    Internally delegates to :class:`TimingContext` for consistent timing
    across the codebase (no manual ``perf_counter`` bookkeeping).
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        with TimingContext() as t:
            result = func(*args, **kwargs)
        if isinstance(result, dict) and 'latency_ms' not in result:
            result['latency_ms'] = t.ms
        elif hasattr(result, 'latency_ms'):
            result.latency_ms = t.ms
        logger.debug("Function %s took %.2fms", func.__name__, t.ms)
        return result
    return wrapper
