import timeit
import numpy as np
from numba import jit, cuda
import time
from numba.cuda.random import create_xoroshiro128p_states, xoroshiro128p_uniform_float32


# CPU版本，没有JIT加速
def calculate_pi_no_jit(n=1000):
    acc = 0
    for i in range(n):
        x = np.random.rand()
        y = np.random.rand()
        if x ** 2 + y ** 2 <= 1:
            acc += 1
    return 4 * acc / n


# CPU版本，使用JIT加速
@jit(nopython=True)
def calculate_pi_jit(n=1000):
    acc = 0
    for i in range(n):
        x = np.random.rand()
        y = np.random.rand()
        if x ** 2 + y ** 2 <= 1:
            acc += 1
    return 4 * acc / n


# GPU版本，使用CUDA随机数
@cuda.jit
def calculate_pi_gpu_kernel(rng_states, n, acc):
    i = cuda.grid(1)
    if i < n:
        x = xoroshiro128p_uniform_float32(rng_states, i)
        y = xoroshiro128p_uniform_float32(rng_states, i)
        if x ** 2 + y ** 2 <= 1:
            acc[i] = 1
        else:
            acc[i] = 0


def calculate_pi_gpu(n=1000):
    # 初始化随机数生成器状态
    rng_states = create_xoroshiro128p_states(n, seed=42)

    # 初始化结果数组
    acc = np.zeros(n, dtype=np.int32)
    d_acc = cuda.to_device(acc)

    # 定义线程块和网格大小
    threads_per_block = 256
    blocks_per_grid = (n + threads_per_block - 1) // threads_per_block

    # 执行 CUDA 内核函数
    calculate_pi_gpu_kernel[blocks_per_grid, threads_per_block](rng_states, n, d_acc)
    acc = d_acc.copy_to_host()

    return 4 * acc.sum() / n


# 对比各版本的执行时间
for n in [1000, 10000, 100000,100000000]:
    # 没有JIT加速的CPU版本
    start = time.time()
    pi_no_jit = calculate_pi_no_jit(n)
    end = time.time()
    print(f"Execution time for calculate_pi_no_jit with n={n}: {end - start:.5f} seconds, Result: {pi_no_jit}")

    # 使用JIT加速的CPU版本
    start = time.time()
    pi_jit = calculate_pi_jit(n)
    end = time.time()
    print(f"Execution time for calculate_pi_jit with n={n}: {end - start:.5f} seconds, Result: {pi_jit}")

    # GPU版本
    start = time.time()
    pi_gpu = calculate_pi_gpu(n)
    end = time.time()
    print(f"Execution time for calculate_pi_gpu with n={n}: {end - start:.5f} seconds, Result: {pi_gpu}")

    print()
