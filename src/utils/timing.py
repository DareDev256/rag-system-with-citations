import time
import functools
import logging

logger = logging.getLogger(__name__)

def measure_latency(func):
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
        
        logger.debug(f"Function {func.__name__} took {latency_ms:.2f}ms")
        return result
    return wrapper
