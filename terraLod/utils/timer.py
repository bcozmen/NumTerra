import time
from functools import wraps

class Timeit:
    def __init__(self):
        self.times = {}

    def __call__(self, func=None, *, name=None):

        # Called as @timeit(name="...")
        if func is None:
            def decorator(func):
                return self._wrap(func, name)
            return decorator

        # Called as @timeit
        return self._wrap(func, name)

    def _wrap(self, func, name=None):
        label = name or func.__name__

        @wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.perf_counter()

            result = func(*args, **kwargs)

            elapsed = time.perf_counter() - start_time

            if label not in self.times:
                self.times[label] = []

            self.times[label].append(elapsed)

            return result

        return wrapper

    def report(self):
        print("Timing Report:")
        for label, times in self.times.items():
            avg_time = sum(times) / len(times)
            print(f"  {label}: {avg_time:.4f}s (avg over {len(times)} runs)")

timeit = Timeit()