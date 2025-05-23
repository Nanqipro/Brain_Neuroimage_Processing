# SCN相空间处理Python版本

## 概述

这是将MATLAB脚本 `scn_phase_space_process.m` 转换为Python的等效实现。该脚本用于处理脑神经元钙成像数据，将时间序列转换为相空间流形，并构建图数据集。

## 主要功能

1. **数据加载**: 加载MAT格式的钙成像数据
2. **相空间重构**: 将一维时间序列转换为3D相空间轨迹
3. **互信息计算**: 自动确定最佳时间延迟参数
4. **图数据集生成**: 创建节点、边和图属性的CSV文件

## 环境要求

### Python版本
- Python 3.7 或更高版本

### 依赖包
```bash
pip install -r requirements_scn.txt
```

或者手动安装：
```bash
pip install numpy>=1.21.0 pandas>=1.3.0 scipy>=1.7.0 matplotlib>=3.4.0 tqdm>=4.60.0
```

## 使用方法

### 1. 准备数据
确保数据文件 `Dataset1_SCNProject.mat` 位于 `./SCNData/` 目录中。

### 2. 运行脚本
```bash
cd SCN-Research-Project-main/StateClassifier
python scn_phase_space_process.py
```

### 3. 输出文件
脚本会在 `./data/` 目录中生成以下文件：
- `nodes.csv`: 图节点数据，包含图ID、节点ID和特征
- `edges.csv`: 图边数据，包含图ID、源节点ID、目标节点ID和边特征
- `graphs.csv`: 图属性数据，包含图ID、图特征和标签

## 核心算法

### 1. 互信息计算 (`mutual_information`)
- 计算时间延迟互信息，用于确定相空间重构的最佳时间延迟
- 使用概率分布的离散化方法
- 返回互信息随延迟变化的数组

### 2. 相空间重构 (`phase_space_reconstruction`)
- 将一维时间序列嵌入到高维相空间
- 使用时间延迟嵌入方法
- 支持任意维度的相空间重构

### 3. 数据预处理
- Z-score标准化钙信号
- 统一相空间轨迹长度
- 格式化特征数据为CSV兼容格式

## 配置参数

在 `main()` 函数中可以调整以下参数：

```python
file_path = './SCNData/Dataset1_SCNProject.mat'  # 输入数据路径
frame_rate = 0.67  # 帧率 (Hz)
out_path = './data'  # 输出目录
xyz_len = 170  # 相空间轨迹统一长度
dim = 3  # 相空间嵌入维度
```

## 性能优化

- 使用进度条显示处理进度
- 支持大数据集的批量处理
- 内存优化的数据结构

## 与MATLAB版本的差异

1. **类型注解**: 添加了完整的Python类型注解
2. **错误处理**: 增强的异常处理机制
3. **进度显示**: 使用tqdm显示实时进度
4. **代码结构**: 模块化函数设计，便于测试和维护
5. **注释规范**: 符合Python docstring标准

## 故障排除

### 常见问题

1. **数据加载失败**
   - 检查文件路径是否正确
   - 确认MAT文件中包含变量 'F_set'

2. **内存不足**
   - 减少处理的细胞数量（修改循环中的范围）
   - 降低相空间轨迹长度 `xyz_len`

3. **相空间重构失败**
   - 检查时间序列长度是否足够
   - 调整时间延迟参数

### 调试模式
在代码中可以启用详细的错误信息：
```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

## 输出数据格式

### nodes.csv
```
graph_id,node_id,feat
1,1,"x1,y1,z1"
1,2,"x2,y2,z2"
...
```

### edges.csv
```
graph_id,src_id,dst_id,feat
1,1,2,1
1,2,3,1
...
```

### graphs.csv
```
graph_id,feat,label
1,"1,0,0,0,0,0",0
2,"1,0,0,0,0,0",0
...
```

## 许可和引用

如果您使用此代码，请引用相关的学术论文：

1. H. Yang, "Multiscale Recurrence Quantification Analysis of Spatial Vectorcardiogram (VCG) Signals," IEEE Transactions on Biomedical Engineering, Vol. 58, No. 2, p339-347, 2011
2. Y. Chen and H. Yang, "Multiscale recurrence analysis of long-term nonlinear and nonstationary time series," Chaos, Solitons and Fractals, Vol. 45, No. 7, p978-987, 2012 