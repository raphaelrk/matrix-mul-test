# matrix-mul-test/py/index.js
# compare performance of matrix multiplication of various libraries

import numpy as np
import time

sz = 512
A = np.random.rand(sz, sz)
B = np.random.rand(sz, sz)

def run_test(func: callable, name: str):
    times = []
    for i in range(20):
        start = time.time()
        C = func()
        end = time.time()
        ms = (end - start) * 1000
        times.append(ms)

    str = "".join([
      f"{name}:".ljust(20),
      f"mean: {round(np.mean(times), 2)}ms".ljust(16),
      f"std: {round(np.std(times), 2)}ms".ljust(16),
      f"p0: {round(np.percentile(times, 0), 2)}ms".ljust(16),
      f"p5: {round(np.percentile(times, 5), 2)}ms".ljust(16),
      f"p25: {round(np.percentile(times, 25), 2)}ms".ljust(16),
      f"p50: {round(np.percentile(times, 50), 2)}ms".ljust(16),
      f"p75: {round(np.percentile(times, 75), 2)}ms".ljust(16),
      f"p95: {round(np.percentile(times, 95), 2)}ms".ljust(16),
      f"p100: {round(np.percentile(times, 100), 2)}ms".ljust(16),
    ])
    print(str)

run_test(lambda: np.matmul(A, B), "numpy.matmul")
