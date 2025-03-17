# 神经网络LSTM与GNN分析平台

## 项目概述

本项目是一个神经元网络数据分析平台，结合LSTM（长短期记忆网络）和GNN（图神经网络）技术，专注于神经元活动序列分析、聚类和网络拓扑可视化。项目利用深度学习和图神经网络技术分析神经活动模式，旨在揭示神经网络功能连接的动态变化及其与行为的关联。

## 核心功能

- 基于LSTM的神经元活动时间序列分析
- 基于GNN/GCN的神经元网络拓扑结构分析
- 神经元功能连接的图模型构建与分析
- 基于GNN嵌入的拓扑结构可视化（新增！）
- K-means等聚类算法用于神经元功能分组
- 神经网络拓扑结构分析
- 交互式和静态网络可视化
- 神经元活动与行为之间的相关性分析
- 模型训练、测试和验证框架

## 项目结构

```
LSTM/
├── src/                    # 源代码目录
│   ├── lib/                # 核心库组件
│   │   ├── vis-9.1.2/      # 可视化库
│   │   ├── tom-select/     # 选择组件库
│   │   └── bindings/       # 语言绑定
│   ├── analysis_results.py # 结果分析脚本
│   ├── pos_topology_js.py  # 拓扑分析脚本
│   ├── analysis_config.py  # 分析配置
│   ├── kmeans_lstm_analysis.py # LSTM聚类分析
│   ├── neuron_gnn.py       # 神经元GNN分析模块
│   ├── gnn_visualization.py # GNN可视化模块
│   ├── gnn_topology.py     # GNN拓扑结构分析与可视化
│   ├── visualization.py    # 可视化工具
│   ├── analysis_utils.py   # 分析工具函数
│   ├── enhanced_analysis.py # 增强分析方法
│   └── test_env.py         # 环境测试脚本
├── datasets/               # 数据集目录
├── models/                 # 训练模型存储
│   ├── neuron_lstm_model_Day3.pth # 第3天神经元模型
│   ├── neuron_lstm_model_Day6.pth # 第6天神经元模型
│   └── neuron_lstm_model_Day9.pth # 第9天神经元模型
├── results/                # 分析结果输出目录
│   └── gnn_results/        # GNN分析结果目录
│       ├── gcn_topology.png             # GCN拓扑静态可视化
│       ├── gcn_interactive_topology.html # GCN拓扑交互式可视化
│       ├── gat_topology.png             # GAT拓扑静态可视化
│       ├── gat_interactive_topology.html # GAT拓扑交互式可视化
│       └── gnn_analysis_results.json    # GNN分析结果JSON
├── README_interactive_network.md # 交互式网络使用指南
└── requirements.txt        # 项目依赖
```

## 安装要求

运行此项目需要以下依赖项（推荐Python 3.8+）：

```
# 核心依赖
torch>=2.1.0
torchvision>=0.16.0
torchaudio>=2.1.0
numpy>=1.21.0
pandas>=1.3.0
scikit-learn>=1.0.0
matplotlib>=3.4.0
seaborn>=0.11.0
networkx>=2.6.0

# GNN相关依赖
torch-geometric>=2.0.0
torch-scatter>=2.0.9
torch-sparse>=0.6.13
```

安装依赖：

```bash
pip install -r requirements.txt
```

## 使用指南

### 环境测试

首先，验证环境配置是否正确：

```bash
python src/test_env.py
```

### 数据分析工作流程

1. **配置分析参数**
   
   编辑`src/analysis_config.py`文件设置分析参数，包括GNN相关参数

2. **运行LSTM分析**

   ```bash
   python src/kmeans_lstm_analysis.py
   ```

3. **结果分析和可视化**

   ```bash
   python src/analysis_results.py
   ```
   
   此步骤现在也会执行GNN分析和基于GNN的拓扑可视化

4. **可视化网络拓扑**

   ```bash
   python src/pos_topology_js.py
   ```

### 交互式网络可视化

详细说明请参阅`README_interactive_network.md`

## 主要模块说明

### LSTM模型

项目使用长短期记忆网络（LSTM）编码和分析神经元活动时间序列。模型结构遵循项目的标准架构，包括：

- 多层LSTM编码器
- 梯度裁剪
- 学习率调度器
- 标准化输入处理

### GNN/GCN模型

项目使用图神经网络分析神经元功能连接拓扑结构：

- **GCN（图卷积网络）**：用于神经元行为预测，捕捉神经元活动与行为之间的关系
- **GAT（图注意力网络）**：用于识别神经元功能模块，自动发现具有相似功能的神经元群组
- **时间GNN**：结合GNN和LSTM分析神经元活动随时间变化的特征

### 基于GNN的拓扑结构分析（新增！）

项目新增了基于GNN学习的神经元拓扑结构分析功能：

- 使用GNN学习的嵌入表示重新构建神经元连接结构
- 基于功能相似性而非仅仅相关性的连接创建
- 揭示传统相关性分析难以发现的神经元功能模块
- 提供静态拓扑图与交互式可视化
- 从GCN（行为视角）和GAT（模块视角）两个不同角度进行拓扑构建

### 聚类分析

使用K-means和其他聚类算法分析神经元功能组织：

- 自动参数选择
- 聚类稳定性分析
- 跟踪聚类随时间的演变

### 拓扑分析

分析神经元之间的功能连接：

- 基于相关性的连接强度计算
- 社区检测算法
- 中心性和路径长度分析
- GNN增强的模块识别

### 可视化

提供各种可视化工具：

- 静态网络图
- 交互式网络可视化
- 活动模式热图
- 聚类结果可视化
- GNN嵌入可视化
- GAT注意力权重可视化
- 时间序列GNN结果可视化
- 基于GNN的拓扑结构可视化（新增！）

## 结果解释

分析结果保存在`results`目录中，包括：

- 训练和验证指标
- 聚类结果
- 网络分析结果
- GNN分析结果（保存在`results/gnn_results`目录）
- 基于GNN的拓扑结构图（保存在`results/gnn_results`目录）
- 可视化图表

### GNN分析结果

GNN分析生成以下结果：

- 神经元行为预测模型（GCN）
- 神经元功能模块识别（GAT）
- 基于GNN重新构建的拓扑结构（新增！）
- 时间序列GNN分析
- 节点嵌入可视化
- 交互式GNN网络可视化

## 常见问题解决

常见问题及解决方案：

1. **CUDA错误**：确保正确设置GPU环境变量并安装兼容的CUDA版本
2. **内存错误**：减小批量大小或使用数据采样
3. **可视化问题**：确保安装所有相关的可视化库
4. **PyTorch Geometric安装问题**：根据您的CUDA版本安装正确版本的torch-scatter和torch-sparse

## 联系方式与贡献

欢迎提出问题和改进建议！

## 引用

如果您在研究中使用此平台，请引用：

```
@software{neural_network_lstm_gnn_platform,
  author = {Your Research Team},
  title = {Neural Network LSTM and GNN Analysis Platform},
  year = {2023},
  url = {https://github.com/yourusername/neural-network-lstm-gnn}
}
```

## 许可证

此项目根据MIT许可证授权 - 详情见LICENSE文件。

