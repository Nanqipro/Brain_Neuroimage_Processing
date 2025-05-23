# 脑神经网络状态分类器项目

基于图卷积神经网络(GCN)的脑神经元活动状态分类系统，专门用于处理和分析脑神经元钙成像数据。

## 快速开始 🚀

### 1. 安装依赖
```bash
pip install -r requirements.txt
```

### 2. 运行完整流程
```bash
# 一键运行数据处理和模型训练
python run.py --all

# 或者分步骤运行
python run.py --process    # 数据处理
python run.py --train      # 模型训练
```

### 3. 查看结果
训练完成后，查看生成的文件：
- `data/` - 图数据集文件
- `result/` - 训练好的模型
- `scn_classifier.log` - 详细日志

### 4. 其他有用命令
```bash
python run.py --check      # 检查运行环境
python run.py --config     # 查看配置参数
python run.py --test       # 测试核心模块
```

---

## 项目概述

该项目包含完整的数据处理管道：从原始钙离子信号到相空间重构，再到图神经网络分类。主要应用于超交叉核(SCN)神经元活动状态的自动识别和分类。

## 项目结构

```
StateClassifier/
├── src/                           # 核心数据处理模块
│   ├── __init__.py               # 包初始化文件
│   ├── format_convert.py         # 数据格式转换工具
│   ├── mutual.py                 # 时间延迟互信息计算
│   ├── phasespace.py            # 相空间重构算法
│   ├── cellset2trim.py          # 数据裁剪和标准化
│   ├── test_all_modules.py      # 核心模块测试脚本
│   └── README.md                # 核心模块详细说明
├── scn_phase_space_process.py   # 主数据处理脚本
├── model.py                     # GCN模型定义
├── utils.py                     # 数据加载和预处理工具
├── main.py                      # 模型训练和评估主程序
├── config.py                    # 项目配置管理
├── run.py                       # 便捷运行脚本
├── requirements.txt             # 项目依赖
└── README.md                    # 项目说明文档(本文件)
```

## 功能模块

### 1. 数据处理模块 (`src/`)

**核心算法模块**，从MATLAB代码转换而来：

- **`mutual.py`**: 时间延迟互信息计算，用于确定相空间重构的最佳时间延迟参数
- **`phasespace.py`**: 相空间重构算法，将一维时间序列转换为高维相空间轨迹
- **`cellset2trim.py`**: 数据裁剪工具，统一轨迹长度便于批处理
- **`format_convert.py`**: 数据格式转换工具，支持CSV导出

### 2. 主处理脚本 (`scn_phase_space_process.py`)

**端到端数据处理流程**：
- 加载原始钙离子信号数据(.mat格式)
- 执行Z-score标准化和相空间重构
- 生成图数据集(nodes.csv, edges.csv, graphs.csv)

### 3. 深度学习模块

- **`model.py`**: 多层GCN模型定义，包含全局池化和分类器
- **`utils.py`**: 数据加载器和数据集分割工具
- **`main.py`**: 模型训练、验证和测试主程序

### 4. 配置和运行

- **`config.py`**: 集中管理所有配置参数
- **`run.py`**: 提供便捷的命令行接口

## 安装依赖

```bash
pip install -r requirements.txt
```

### 主要依赖包：
- `numpy>=1.20.0` - 数值计算
- `pandas>=1.3.0` - 数据处理
- `scipy>=1.7.0` - 科学计算
- `matplotlib>=3.3.0` - 数据可视化
- `torch>=1.9.0` - 深度学习框架
- `torch-geometric>=2.0.0` - 图神经网络
- `tqdm>=4.60.0` - 进度条显示

## 使用方法

### 快速开始

1. **准备数据**：将SCN钙成像数据保存为`.mat`格式
2. **数据处理**：运行主处理脚本生成图数据集
3. **模型训练**：使用生成的数据集训练GCN分类器

```bash
# 1. 数据处理 - 生成图数据集
python run.py --process

# 2. 模型训练 - 训练GCN分类器
python run.py --train

# 或者一键运行完整流程
python run.py --all
```

### 详细工作流程

#### 步骤1: 数据预处理

```python
# 运行主处理脚本
python run.py --process
```

该脚本将：
- 加载原始钙信号数据
- 执行Z-score标准化
- 计算最佳时间延迟参数
- 进行3D相空间重构
- 生成标准化的图数据集

输出文件：
- `data/nodes.csv` - 节点特征数据
- `data/edges.csv` - 边连接信息
- `data/graphs.csv` - 图标签数据

#### 步骤2: 模型训练

```python
# 训练GCN分类器
python run.py --train
```

该程序将：
- 加载图数据集并按比例分割(训练60%/验证20%/测试20%)
- 训练多层GCN模型(160个epoch)
- 保存最佳性能模型
- 在测试集上评估最终性能

### 测试核心模块

```bash
# 测试数据处理模块
python run.py --test

# 显示可视化图表
cd src/
python test_all_modules.py --plot
```

## 算法原理

### 1. 时间延迟互信息
使用香农熵计算时间序列的自相关性：
```
MI(τ) = ∑∑ P(xi, xi+τ) log[P(xi, xi+τ) / (P(xi)P(xi+τ))]
```

### 2. 相空间重构
基于Takens嵌入定理：
```
Y(t) = [x(t), x(t+τ), x(t+2τ), ..., x(t+(d-1)τ)]
```

### 3. 图卷积网络
多层GCN架构：
- 3层图卷积层(GCNConv)
- 全局池化层(最大池化+平均池化)
- 全连接分类器

## 配置参数

所有配置参数都在`config.py`中集中管理：

### 数据处理参数
```python
frame_rate = 0.67        # 钙成像帧率(Hz)
xyz_len = 170           # 统一轨迹长度
embedding_dim = 3       # 相空间嵌入维度
```

### 模型训练参数
```python
num_epochs = 160        # 训练轮数
learning_rate = 0.001   # 初始学习率
batch_size = 1          # 批次大小
num_classes = 6         # 分类类别数
```

查看所有配置：
```bash
python run.py --config
```

## 数据集格式

### 输入数据
- **MAT文件**：包含细胞钙信号数据的MATLAB文件
- **数据结构**：`F_set[cell_num][timeline]` - 细胞×时间线的数据矩阵

### 输出数据
- **nodes.csv**：图节点特征数据
  - `graph_id`: 图ID
  - `node_id`: 节点ID  
  - `feat`: 3D坐标特征(x,y,z)

- **edges.csv**：图边连接信息
  - `graph_id`: 图ID
  - `src_id`: 源节点ID
  - `dst_id`: 目标节点ID
  - `feat`: 边权重

- **graphs.csv**：图标签数据
  - `graph_id`: 图ID
  - `feat`: 图级别特征
  - `label`: 分类标签

## 性能评估

模型评估指标：
- **Overall Accuracy**: 整体分类准确率
- **Balanced Accuracy**: 平衡准确率(处理类别不平衡)
- **Class-wise Accuracy**: 各类别分类准确率

## 扩展和定制

### 添加新的特征提取方法
在`src/`目录下添加新模块，遵循现有的API设计模式。

### 修改GCN架构
编辑`model.py`中的`MultiLayerGCN`类，调整：
- 卷积层数量和隐藏维度
- 池化策略
- 分类器结构

### 数据增强策略
在`utils.py`的`get_dataset`函数中修改数据增强参数。

### 修改配置参数
编辑`config.py`文件中的相应参数，或通过环境变量覆盖。

## 常见问题

### Q: 如何处理自己的数据？
A: 将数据保存为MAT格式，确保包含`F_set`字段，数据结构为`[cell_num, timeline]`的列表。

### Q: 如何调整模型性能？
A: 
1. 修改`config.py`中的训练参数
2. 调整GCN模型架构
3. 尝试不同的数据增强策略

### Q: 如何处理GPU内存不足？
A: 
1. 减小批次大小(`BATCH_SIZE`)
2. 减少模型隐藏层维度(`HIDDEN_DIM`)
3. 设置`DEVICE = "cpu"`使用CPU训练

### Q: 如何验证结果？
A: 
1. 运行核心模块测试：`python run.py --test`
2. 检查生成的CSV文件格式
3. 观察训练日志中的准确率变化

## 引用文献

如果使用本项目，请引用相关论文：

1. H. Yang, "Multiscale Recurrence Quantification Analysis of Spatial Vectorcardiogram (VCG) Signals," IEEE Transactions on Biomedical Engineering, Vol. 58, No. 2, p339-347, 2011

2. Y. Chen and H. Yang, "Multiscale recurrence analysis of long-term nonlinear and nonstationary time series," Chaos, Solitons and Fractals, Vol. 45, No. 7, p978-987, 2012

## 许可证

本项目基于学术研究目的开发，请在使用时引用相关文献。

## 联系方式

如有问题或建议，请联系：SCN研究小组

---

## 更新日志

- **v1.0.0** (2023): 初始版本，包含完整的MATLAB到Python转换
- **v1.1.0** (2023): 添加GCN分类器和完整训练流程
- **v1.2.0** (2023): 项目结构优化和文档完善
- **v1.3.0** (2023): 添加配置管理和便捷运行脚本 