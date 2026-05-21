import time
from functools import wraps

class timeit:
    def __init__(self, label=None):
        self.label = label

    def __call__(self, func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.time()

            result = func(*args, **kwargs)

            end_time = time.time()

            name = self.label or func.__name__

            #print(f"{name} took {end_time - start_time:.2f} seconds")

            return result

        return wrapper