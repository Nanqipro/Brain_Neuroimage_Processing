import torch
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import matplotlib
import os

# 测试CUDA环境
print("="*50)
print("CUDA环境检查")
print("="*50)
print("CUDA是否可用:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("CUDA版本:", torch.version.cuda)
    print("当前GPU:", torch.cuda.get_device_name(0))
    print("GPU数量:", torch.cuda.device_count())
    print("当前GPU索引:", torch.cuda.current_device())
    print("GPU计算能力:", torch.cuda.get_device_capability(0))
    
# 检查CUDA环境变量
print("\nCUDA环境变量:")
cuda_path = os.environ.get('CUDA_PATH')
if cuda_path:
    print(f"CUDA_PATH: {cuda_path}")
else:
    print("警告: CUDA_PATH 未设置")

# 测试NumPy
print("\n" + "="*50)
print("Python包版本")
print("="*50)
print("NumPy版本:", np.__version__)
print("Pandas版本:", pd.__version__)
print("Scikit-learn版本:", StandardScaler().__class__.__module__.split('.')[0])
print("Matplotlib版本:", matplotlib.__version__)
print("PyTorch版本:", torch.__version__)

# 测试PyTorch GPU功能
if torch.cuda.is_available():
    print("\n" + "="*50)
    print("PyTorch GPU测试")
    print("="*50)
    # 创建一个测试张量并移动到GPU
    x = torch.randn(3, 3)
    x = x.cuda()
    print("GPU张量测试成功！")
    print("张量设备位置:", x.device) 