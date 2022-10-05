# matrix-mul-test/py/index.js
# compare performance of matrix multiplication of various libraries

print("Importing...")

from numba import jit
import numpy as np
import torch
import tensorflow as tf
import time

print("Imported.")

sz = 512
A = np.random.rand(sz, sz)
B = np.random.rand(sz, sz)
A_torch = torch.from_numpy(A)
B_torch = torch.from_numpy(B)
A_tf = tf.constant(A)
B_tf = tf.constant(B)

N = 100


def run_test(func: callable, name: str):
    times = []
    for i in range(N):
        start = time.perf_counter()
        C = func()
        end = time.perf_counter()
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


run_test(lambda: 1, "nil")
run_test(lambda: np.matmul(A, B), "numpy.matmul")


@jit(nopython=True)
def matmul_jit_three_loop(A: np.ndarray, B: np.ndarray):
    C = np.zeros((sz, sz))
    for i in range(sz):
        for j in range(sz):
            for k in range(sz):
                C[i, j] += A[i, k] * B[k, j]
    return C


@jit(nopython=True)
def matmul_jit_two_loop(A, B):
    C = np.zeros((sz, sz))
    for i in range(sz):
        for j in range(sz):
            C[i, j] = np.sum(A[i, :] * B[:, j])
    return C


@jit(nopython=True)
def matmul_jit_one_loop(A, B):
    C = np.zeros((sz, sz))
    for i in range(sz):
        C[i, :] = np.sum(A[i, :] * B, axis=1)
    return C


@jit(nopython=True)
def matmul_jit_no_loop(A, B):
    # return np.linalg.multi_dot([A, B])
    return A @ B


# compile first
matmul_jit_three_loop(A, B)
matmul_jit_two_loop(A, B)
matmul_jit_one_loop(A, B)
matmul_jit_no_loop(A, B)

# run numba tests
run_test(lambda: matmul_jit_three_loop(A, B), "numba three loop")
run_test(lambda: matmul_jit_two_loop(A, B), "numba two loop")
run_test(lambda: matmul_jit_one_loop(A, B), "numba one loop")
run_test(lambda: matmul_jit_no_loop(A, B), "numba no loop")

# run pytorch tests
run_test(lambda: torch.matmul(A_torch, B_torch), "pytorch.matmul")
run_test(lambda: A_torch @ B_torch, "pytorch @")

# run tensorflow tests
run_test(lambda: tf.matmul(A_tf, B_tf), "tf.matmul")
run_test(lambda: A_tf @ B_tf, "tf @")

# run tensorflow tests
run_test(lambda: tf.linalg.matmul(A_tf, B_tf), "tf.linalg.matmul")

# tf gives following warning:
# 2022-10-05 17:26:41.050547: I tensorflow/core/common_runtime/pluggable_device/pluggable_device_factory.cc:306] Could not identify NUMA node of platform GPU ID 0, defaulting to 0. Your kernel may not have been built with NUMA support.
# 2022-10-05 17:26:41.050691: I tensorflow/core/common_runtime/pluggable_device/pluggable_device_factory.cc:272] Created TensorFlow device (/job:localhost/replica:0/task:0/device:GPU:0 with 0 MB memory) -> physical PluggableDevice (device: 0, name: METAL, pci bus id: <undefined>)
# https://developer.apple.com/forums/thread/693292

