from numba import cuda
import numpy as np

# 定义 CUDA 核函数
@cuda.jit
def add_kernel(x, y, out):
    idx = cuda.grid(1)
    if idx < x.size:
        out[idx] = x[idx] + y[idx]

# 初始化数据
x = np.ones(10**6)
y = np.ones(10**6)
out = np.zeros(10**6)

# 将数据从 CPU 传到 GPU 并运行内核
d_x = cuda.to_device(x)
d_y = cuda.to_device(y)
d_out = cuda.to_device(out)

threads_per_block = 256
blocks_per_grid = (x.size + (threads_per_block - 1)) // threads_per_block
add_kernel[blocks_per_grid, threads_per_block](d_x, d_y, d_out)

# 将结果从 GPU 复制回 CPU 并验证
out = d_out.copy_to_host()
print("GPU addition successful:", np.all(out == x + y))
