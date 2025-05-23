# 神经图像处理状态分类器 - Python版本

这是从MATLAB代码转换而来的Python神经图像处理工具包，专门用于脑神经图像处理中的状态分类和相空间分析。

## 功能模块

### 1. 格式转换模块 (`format_convert.py`)
将数值数组转换为CSV格式字符串，便于数据导出和存储。

**主要功能：**
- 数值数组转逗号分隔字符串
- 支持numpy数组和Python列表输入
- 自动扁平化多维数组

### 2. 互信息计算模块 (`mutual.py`)
计算时间序列的时间延迟互信息，用于确定相空间重构的最佳时间延迟参数。

**主要功能：**
- 时间延迟互信息计算
- 最佳时间延迟自动估计
- 可视化互信息曲线
- 支持自定义分区数和最大延迟

### 3. 相空间重构模块 (`phasespace.py`)
将一维时间序列重构为高维相空间轨迹，揭示系统的动力学特性。

**主要功能：**
- 时间延迟嵌入相空间重构
- 2D/3D轨迹可视化
- 嵌入参数自动估计
- 洛伦兹吸引子测试数据生成

### 4. 数据裁剪模块 (`cellset2trim.py`)
统一细胞数组中相空间轨迹的长度，便于后续批处理和分析。

**主要功能：**
- 细胞数组批量裁剪
- 支持列表和字典两种数据格式
- 数据集统计信息分析
- 裁剪参数有效性验证

## 安装依赖

```bash
pip install -r requirements.txt
```

## 使用示例

### 基本使用
```python
import numpy as np
from src import format_convert, mutual, phasespace, cellset2trim

# 1. 格式转换示例
data = [1.23, 4.56, 7.89]
csv_string = format_convert(data)
print(f"CSV格式: {csv_string}")

# 2. 互信息计算示例
t = np.linspace(0, 10*np.pi, 1000)
signal = np.sin(t) + 0.1*np.random.randn(len(t))
mi_values = mutual(signal, partitions=16, tau=20, plot_result=True)
optimal_delay = find_optimal_delay(signal)

# 3. 相空间重构示例
Y = phasespace(signal, dim=3, tau=optimal_delay, plot_result=True)

# 4. 数据裁剪示例
dataset = [[np.random.randn(100, 3), None], 
           [np.random.randn(150, 3), np.random.randn(80, 3)]]
trimmed = cellset2trim(dataset, trim_len=75)
```

### 完整工作流程
```python
import numpy as np
from src import *

# 生成测试信号
np.random.seed(42)
t = np.linspace(0, 20*np.pi, 2000)
signal = np.sin(t) + 0.5*np.sin(3*t) + 0.1*np.random.randn(len(t))

# 步骤1: 估计最佳嵌入参数
print("估计最佳嵌入参数...")
optimal_dim, optimal_tau = estimate_embedding_params(signal)
print(f"最佳维度: {optimal_dim}, 最佳延迟: {optimal_tau}")

# 步骤2: 进行相空间重构  
print("进行相空间重构...")
Y = phasespace(signal, dim=optimal_dim, tau=optimal_tau, plot_result=True)

# 步骤3: 创建多轨迹数据集
print("创建数据集...")
dataset = []
for i in range(3):  # 3个细胞
    cell_data = []
    for j in range(2):  # 2个时间线
        if np.random.rand() > 0.2:  # 80%概率有数据
            trajectory_len = np.random.randint(100, 200)
            trajectory = phasespace(signal[:trajectory_len], dim=3, tau=optimal_tau)
            cell_data.append(trajectory)
        else:
            cell_data.append(None)
    dataset.append(cell_data)

# 步骤4: 统一轨迹长度
print("统一轨迹长度...")
stats = get_dataset_stats(dataset)
print(f"数据集统计: {stats}")

trim_length = int(stats['min_length'])
if validate_trim_length(dataset, trim_length):
    trimmed_dataset = cellset2trim(dataset, trim_length)
    print(f"裁剪完成，统一长度: {trim_length}")

# 步骤5: 导出数据
print("导出数据...")
for i, cell_data in enumerate(trimmed_dataset):
    for j, trajectory in enumerate(cell_data):
        if trajectory is not None:
            # 将轨迹转换为CSV格式
            csv_data = []
            for point in trajectory:
                csv_data.append(format_convert(point))
            print(f"细胞{i+1}_时间线{j+1}: {len(csv_data)}个数据点")
```

## 技术特点

### 1. 现代Python语法
- 使用类型注解提高代码可读性
- 遵循PEP 8编程规范
- 异常处理和参数验证

### 2. 高性能计算
- 基于NumPy进行矢量化计算
- 优化的内存管理
- 支持大规模数据处理

### 3. 可视化支持
- Matplotlib集成的图表绘制
- 2D/3D相空间轨迹可视化
- 互信息曲线分析图

### 4. 模块化设计
- 独立的功能模块
- 清晰的API接口
- 易于扩展和维护

## 算法原理

### 时间延迟互信息
使用香农熵和联合概率分布计算互信息：
```
MI(τ) = ∑∑ P(xi, xi+τ) log[P(xi, xi+τ) / (P(xi)P(xi+τ))]
```

### 相空间重构  
基于Takens嵌入定理进行时间延迟嵌入：
```
Y(t) = [x(t), x(t+τ), x(t+2τ), ..., x(t+(d-1)τ)]
```

### 数据裁剪
统一多变长轨迹到固定长度，保持数据一致性用于机器学习。

## 引用文献

1. H. Yang, "Multiscale Recurrence Quantification Analysis of Spatial Vectorcardiogram (VCG) Signals," IEEE Transactions on Biomedical Engineering, Vol. 58, No. 2, p339-347, 2011
2. Y. Chen and H. Yang, "Multiscale recurrence analysis of long-term nonlinear and nonstationary time series," Chaos, Solitons and Fractals, Vol. 45, No. 7, p978-987, 2012

## 许可证

本项目基于原MATLAB代码转换，保持相同的学术研究用途。使用时请引用相关文献。 