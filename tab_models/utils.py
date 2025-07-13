import time
import json
import functools


def log_timing():
    """Decorator to log the execution time of functions"""

    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.time()
            result = func(*args, **kwargs)
            end_time = time.time()
            print(f"{func.__name__} took {end_time - start_time:.4f} seconds")
            return result

        return wrapper

    return decorator


def write_json(data, filepath):
    """Write data to a JSON file"""
    with open(filepath, "w") as f:
        json.dump(data, f, indent=2)
