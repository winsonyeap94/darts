"""
General decorators.
"""

import time

from . import loguru_logger as _logger


# Timing decorator -- measures the time taken by a function
def log_time(func):
    def wrap(*args, **kwargs):
        start_time = time.time()
        results = func(*args, **kwargs)
        end_time = time.time()
        _logger.info(f"{func.__name__} took {(end_time - start_time):.2f} seconds")
        return results
    return wrap
