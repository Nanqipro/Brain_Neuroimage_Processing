# 脑神经影像处理平台 | Brain Neuroimage Processing Platform

<div align="center">
    <img src="docs/images/banner.png" alt="项目标识" width="600" style="margin-bottom: 20px"/>
</div>

[English](#project-overview) | [中文](#项目概述)

## 项目概述

本项目提供了一套全面的工具集，用于分析小鼠脑神经元的钙成像数据。该平台由四个集成模块组成，支持从预处理到高级机器学习和拓扑分析的端到端神经数据分析。

### 项目背景与意义

神经钙成像技术是研究神经活动的重要手段，它能够在细胞分辨率水平上捕捉神经元活动。然而，从原始成像数据中提取有意义的信息需要复杂的计算方法。本项目旨在提供一个集成的分析平台，使神经科学研究人员能够：

- 高效处理大规模钙成像数据
- 发现神经元群体中的功能连接模式
- 关联神经活动与行为事件
- 利用先进的机器学习技术挖掘时间序列中的隐藏模式

## 主要功能模块

### 1. 预分析模块 (Pre_analysis)

预处理和探索性数据分析工具包，为高级分析准备神经成像数据：

- **数据整合与清洗**：合并和标准化来自多个成像会话的数据
- **探索性数据分析 (EDA)**：神经活动模式的初步统计分析和可视化
- **特征提取**：识别钙波信号中的关键特征
- **平滑算法**：降低噪声同时保留重要信号特征
- **周期性分析**：检测和量化神经活动中的节律模式
- **相关性分析**：研究神经信号与行为标记之间的关系

### 2. 聚类分析模块 (Cluster_analysis)

实现各种聚类算法来识别神经元的功能组：

- **多种聚类算法**：K-means、DBSCAN、GMM、层次聚类、谱聚类
- **降维技术**：PCA、t-SNE、UMAP用于可视化高维神经数据
- **距离度量**：欧氏距离、曼哈顿距离、EMD、Hausdorff距离等多种相似性度量
- **指标提取**：评估聚类质量的定量指标
- **活跃神经元可视化**：显示神经元活动模式的动态条形图

### 3. 拓扑分析模块 (Topology_analysis)

分析神经元网络的拓扑结构：

- **拓扑结构生成**：基于神经元活动状态构建时间序列拓扑结构
- **拓扑矩阵转换**：将拓扑结构转换为标准化矩阵格式
- **多算法聚类分析**：识别拓扑矩阵中的模式
- **时空模式分析**：结合时间和空间信息进行全面分析
- **交互式可视化**：神经网络拓扑的2D/3D可视化

### 4. LSTM分析模块 (LSTM)

利用深度学习分析神经活动中的时间模式：

- **基于LSTM的时间序列分析**：编码和预测神经元活动序列
- **时间模式聚类**：按时间激活特征对神经元分组
- **神经网络拓扑分析**：基于LSTM嵌入检查功能连接
- **交互式网络可视化**：神经网络及其变化的动态可视化
- **与行为的相关性**：将神经时间模式与行为事件关联起来

## 目录结构

```
Brain_Neuroimage_Processing/
├── Pre_analysis/            # 预处理和初步分析
│   ├── src/                 # 预处理源代码
│   │   ├── EDA/             # 探索性数据分析工具
│   │   ├── DataIntegration/ # 数据整合脚本
│   │   ├── Feature/         # 特征提取工具
│   │   ├── smooth/          # 信号平滑算法
│   │   ├── Periodic/        # 周期性分析
│   │   ├── oneNeuronal/     # 单神经元分析
│   │   ├── heatmap/         # 热图可视化
│   │   └── Comparative/     # 比较分析
│   ├── datasets/            # 原始和处理后的数据集
│   └── graph/               # 生成的可视化
│
├── Cluster_analysis/        # 聚类工具和算法
│   ├── src/                 # 聚类算法实现
│   │   ├── k-means-*.py     # 各种K-means实现
│   │   ├── DBSCAN.py        # DBSCAN聚类
│   │   ├── GMM.py           # 高斯混合模型
│   │   ├── Hierarchical.py  # 层次聚类
│   │   ├── Spectral.py      # 谱聚类
│   │   ├── *_analysis.py    # 降维工具
│   │   └── Active_bar_chart.py # 活动可视化
│   └── datasets/            # 聚类输入数据
│
├── Topology_analysis/       # 网络拓扑分析
│   ├── src/                 # 拓扑分析代码
│   │   ├── TopologyToMatrix*.py # 拓扑矩阵生成
│   │   ├── Cluster_topology*.py # 拓扑聚类
│   │   ├── Pos_topology.py      # 空间拓扑分析
│   │   ├── Time_topology.py     # 时间拓扑分析
│   │   └── Dynamic_Sorting.py   # 动态结构分析
│   ├── datasets/            # 拓扑数据集
│   ├── result/              # 分析结果
│   ├── graph/               # 拓扑可视化
│   └── requirements.txt     # 拓扑模块依赖
│
└── LSTM/                    # 基于LSTM的时间分析
    ├── src/                 # LSTM分析源代码
    │   ├── lib/             # 支持库
    │   ├── kmeans_lstm_analysis.py # 结合聚类的LSTM
    │   ├── analysis_results.py     # 结果处理
    │   ├── pos_topology_js.py      # 网络可视化
    │   └── visualization.py        # 可视化工具
    ├── datasets/            # LSTM输入数据
    ├── models/              # 训练好的LSTM模型
    ├── results/             # LSTM分析结果
    ├── README_interactive_network.md # 交互式网络指南
    └── requirements.txt     # LSTM模块依赖
```

## 安装要求

每个模块都有自己的依赖项，在各自的`requirements.txt`文件中指定。整个平台的核心依赖项包括：

```
numpy>=1.20.0
pandas>=1.3.0
scikit-learn>=1.0.0
matplotlib>=3.4.0
seaborn>=0.11.0
scipy>=1.7.0
torch>=2.1.0 (LSTM模块需要)
networkx>=2.6.0 (拓扑模块需要)
plotly>=5.3.0
```

### 环境设置

```bash
# 克隆仓库
git clone https://github.com/yourusername/Brain_Neuroimage_Processing.git
cd Brain_Neuroimage_Processing

# 创建虚拟环境（推荐）
python -m venv venv
source venv/bin/activate  # Windows系统上: venv\Scripts\activate

# 安装核心依赖
pip install -r requirements.txt

# 安装特定模块的依赖（按需）
pip install -r LSTM/requirements.txt
pip install -r Topology_analysis/requirements.txt
```

### 快速入门

对于快速测试平台功能，可以使用示例数据集：

```bash
# 运行示例分析管道
python scripts/run_example.py

# 查看结果
python scripts/view_results.py
```

## 使用工作流

### 1. 数据预处理 (Pre_analysis)

```bash
# 运行探索性数据分析
python Pre_analysis/src/EDA/init_show.py

# 执行相关性分析
python Pre_analysis/src/EDA/Correlation_Analysis.py

# 提取特征
python Pre_analysis/src/Feature/extract_features.py
```

### 2. 聚类分析

```bash
# 运行K-means聚类
python Cluster_analysis/src/k-means-ed.py

# 运行UMAP降维
python Cluster_analysis/src/umap_analysis.py

# 可视化聚类结果
python Cluster_analysis/src/visualize_clusters.py
```

### 3. 拓扑分析

```bash
# 生成拓扑矩阵
python Topology_analysis/src/TopologyToMatrix.py

# 对拓扑矩阵进行聚类
python Topology_analysis/src/Cluster_topology.py

# 可视化空间拓扑
python Topology_analysis/src/Pos_topology.py
```

### 4. LSTM分析

```bash
# 运行结合K-means的LSTM分析
python LSTM/src/kmeans_lstm_analysis.py

# 分析和可视化结果
python LSTM/src/analysis_results.py

# 生成交互式网络可视化
python LSTM/src/pos_topology_js.py
```

## 分析结果

每个模块在其各自的输出目录中保存其结果：

- **Pre_analysis/graph**: 初始数据探索和特征提取结果
- **Cluster_analysis/results**: 聚类结果和降维可视化
- **Topology_analysis/result**: 拓扑矩阵和聚类日志
- **Topology_analysis/graph**: 网络可视化和动画
- **LSTM/results**: LSTM模型性能指标和预测可视化
- **LSTM/models**: 训练好的神经网络模型

### 可视化示例

<div align="center">
    <img src="docs/images/cluster_example.png" alt="聚类分析示例" width="400" style="margin-right: 20px"/>
    <img src="docs/images/topology_example.png" alt="拓扑分析示例" width="400"/>
    <p><em>左图：神经元聚类分析结果 | 右图：神经网络拓扑结构可视化</em></p>
</div>

## 常见问题解答

### 1. 如何处理自定义数据格式？

在`Pre_analysis/src/DataIntegration/`目录中，我们提供了多种数据适配器。对于特定的数据格式，可以创建新的适配器类，继承自基本的`DataAdapter`类。

### 2. 系统资源要求是什么？

- **最低配置**：8GB RAM，双核处理器，用于基本的预处理和聚类分析
- **推荐配置**：16GB+ RAM，多核处理器，CUDA兼容GPU（用于LSTM模块）

### 3. 如何优化大规模数据集的处理？

对于大型数据集，建议：
- 使用数据分块处理功能（见`Pre_analysis/src/DataIntegration/chunked_processor.py`）
- 启用内存映射选项（对大多数模块可用，通过添加`--memory-mapped`参数）
- 对LSTM分析使用GPU加速（确保已正确安装CUDA和PyTorch的GPU版本）

## 未来计划

- **深度学习模块扩展**：增加Transformer和GNN模型支持
- **实时分析功能**：开发在线分析流程，支持实时神经成像数据
- **跨物种数据集支持**：扩展框架以支持更多物种（如斑马鱼、果蝇）的神经成像数据
- **行为数据集成**：增强神经活动与行为数据的关联分析工具
- **Web界面**：开发基于浏览器的用户界面，简化分析流程

## 贡献指南

欢迎对任何模块做出贡献。请按照以下步骤进行：

1. Fork本仓库
2. 创建特性分支（`git checkout -b feature/YourFeature`）
3. 提交您的更改（`git commit -am '添加某功能'`）
4. 推送到分支（`git push origin feature/YourFeature`）
5. 创建新的Pull Request

### 编码规范

- 遵循PEP 8 Python代码风格指南
- 为所有函数和类添加中文和英文注释
- 使用类型提示增强代码可读性
- 确保所有新功能都有相应的单元测试

## 致谢

本项目受益于多个开源项目和研究团队的贡献：

- [neuMap](https://github.com/example/neuMap) - 神经元映射库
- [CalciumWave](https://github.com/example/calciumwave) - 钙成像分析工具包
- [TDA-Toolkit](https://github.com/example/tda-toolkit) - 拓扑数据分析工具包

特别感谢以下实验室和机构提供数据集和专业指导：
- 中国科学院神经科学研究所
- 复旦大学脑科学研究院
- 北京大学IDG/McGovern脑科学研究所

## 许可证

本项目采用MIT许可证 - 详情请参见LICENSE文件。

## 联系方式

- **作者**：赵劲
- **邮箱**：ZhaoJ@example.com
- **项目主页**：[https://github.com/yourusername/Brain_Neuroimage_Processing](https://github.com/yourusername/Brain_Neuroimage_Processing)
- **文档**：[https://brain-neuroimage-processing.readthedocs.io](https://brain-neuroimage-processing.readthedocs.io)

对于特定模块的问题，请参考每个模块目录中的README文件。

---

## Project Overview

This repository contains a comprehensive set of tools for analyzing neural calcium imaging data from mouse brain neurons. The platform consists of four integrated modules that facilitate end-to-end neural data analysis, from preprocessing to advanced machine learning and topological analysis.

## Key Components

### 1. Pre-analysis Module

A preprocessing and exploratory data analysis toolkit that prepares neural imaging data for advanced analysis:

- **Data Integration & Cleaning**: Combines and standardizes data from multiple imaging sessions
- **Exploratory Data Analysis (EDA)**: Initial statistical analysis and visualization of neural activity patterns
- **Feature Extraction**: Identifies key characteristics in calcium wave signals
- **Smoothing Algorithms**: Reduces noise while preserving important signal features
- **Periodicity Analysis**: Detects and quantifies rhythmic patterns in neural activity
- **Correlation Analysis**: Examines relationships between neural signals and behavioral markers

### 2. Cluster Analysis Module

Implements various clustering algorithms to identify functional groups of neurons:

- **Multiple Clustering Algorithms**: K-means, DBSCAN, GMM, Hierarchical, Spectral
- **Dimensionality Reduction**: PCA, t-SNE, UMAP for visualizing high-dimensional neural data
- **Distance Metrics**: Euclidean, Manhattan, EMD, Hausdorff for different similarity measures
- **Indicator Extraction**: Quantitative metrics for evaluating cluster quality
- **Active Neuron Visualization**: Dynamic bar charts showing neuronal activity patterns

### 3. Topology Analysis Module

Analyzes the topological structure of neuronal networks:

- **Topology Structure Generation**: Builds time-series topological structures based on neuronal activity states
- **Topology Matrix Conversion**: Converts topological structures into standardized matrix formats
- **Multi-algorithm Clustering Analysis**: Identifies patterns in topology matrices
- **Spatiotemporal Pattern Analysis**: Combines time and space information for comprehensive analysis
- **Interactive Visualization**: 2D/3D visualization of neural network topology

### 4. LSTM Analysis Module

Leverages deep learning to analyze temporal patterns in neural activity:

- **LSTM-based Time Series Analysis**: Encodes and predicts neuronal activity sequences
- **Clustering of Temporal Patterns**: Groups neurons by their temporal activation profiles
- **Neural Network Topology Analysis**: Examines functional connectivity based on LSTM embeddings
- **Interactive Network Visualization**: Dynamic visualization of neural networks and their changes
- **Correlation with Behavior**: Associates neural temporal patterns with behavioral events

## Directory Structure

```
Brain_Neuroimage_Processing/
├── Pre_analysis/            # Preprocessing and initial analysis
│   ├── src/                 # Source code for preprocessing
│   │   ├── EDA/             # Exploratory data analysis tools
│   │   ├── DataIntegration/ # Data integration scripts
│   │   ├── Feature/         # Feature extraction tools
│   │   ├── smooth/          # Signal smoothing algorithms
│   │   ├── Periodic/        # Periodicity analysis
│   │   ├── oneNeuronal/     # Single neuron analysis
│   │   ├── heatmap/         # Heatmap visualization
│   │   └── Comparative/     # Comparative analysis
│   ├── datasets/            # Raw and processed datasets
│   └── graph/               # Generated visualizations
│
├── Cluster_analysis/        # Clustering tools and algorithms
│   ├── src/                 # Clustering algorithms implementation
│   │   ├── k-means-*.py     # Various K-means implementations
│   │   ├── DBSCAN.py        # DBSCAN clustering
│   │   ├── GMM.py           # Gaussian Mixture Models
│   │   ├── Hierarchical.py  # Hierarchical clustering
│   │   ├── Spectral.py      # Spectral clustering
│   │   ├── *_analysis.py    # Dimensionality reduction tools
│   │   └── Active_bar_chart.py # Activity visualization
│   └── datasets/            # Input data for clustering
│
├── Topology_analysis/       # Network topology analysis
│   ├── src/                 # Topology analysis code
│   │   ├── TopologyToMatrix*.py # Topology matrix generation
│   │   ├── Cluster_topology*.py # Topology clustering
│   │   ├── Pos_topology.py      # Spatial topology analysis
│   │   ├── Time_topology.py     # Temporal topology analysis
│   │   └── Dynamic_Sorting.py   # Dynamic structure analysis
│   ├── datasets/            # Topology datasets
│   ├── result/              # Analysis results
│   ├── graph/               # Topology visualizations
│   └── requirements.txt     # Topology module dependencies
│
└── LSTM/                    # LSTM-based temporal analysis
    ├── src/                 # LSTM analysis source code
    │   ├── lib/             # Support libraries
    │   ├── kmeans_lstm_analysis.py # LSTM with clustering
    │   ├── analysis_results.py     # Results processing
    │   ├── pos_topology_js.py      # Network visualization
    │   └── visualization.py        # Visualization tools
    ├── datasets/            # LSTM input data
    ├── models/              # Trained LSTM models
    ├── results/             # LSTM analysis results
    ├── README_interactive_network.md # Interactive network guide
    └── requirements.txt     # LSTM module dependencies
```

## Installation Requirements

Each module has its own dependencies specified in respective `requirements.txt` files. The core dependencies for the entire platform include:

```
numpy>=1.20.0
pandas>=1.3.0
scikit-learn>=1.0.0
matplotlib>=3.4.0
seaborn>=0.11.0
scipy>=1.7.0
torch>=2.1.0 (for LSTM module)
networkx>=2.6.0 (for Topology module)
plotly>=5.3.0
```

To set up the environment:

```bash
# Clone the repository
git clone https://github.com/yourusername/Brain_Neuroimage_Processing.git
cd Brain_Neuroimage_Processing

# Create a virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install core dependencies
pip install -r requirements.txt

# Install module-specific dependencies (as needed)
pip install -r LSTM/requirements.txt
pip install -r Topology_analysis/requirements.txt
```

## Usage Workflow

### 1. Data Preprocessing (Pre_analysis)

```bash
# Run exploratory data analysis
python Pre_analysis/src/EDA/init_show.py

# Perform correlation analysis
python Pre_analysis/src/EDA/Correlation_Analysis.py
```

### 2. Cluster Analysis

```bash
# Run K-means clustering
python Cluster_analysis/src/k-means-ed.py

# Run UMAP dimensionality reduction
python Cluster_analysis/src/umap_analysis.py
```

### 3. Topology Analysis

```bash
# Generate topology matrices
python Topology_analysis/src/TopologyToMatrix.py

# Perform clustering on topology matrices
python Topology_analysis/src/Cluster_topology.py

# Visualize spatial topology
python Topology_analysis/src/Pos_topology.py
```

### 4. LSTM Analysis

```bash
# Run LSTM analysis with K-means
python LSTM/src/kmeans_lstm_analysis.py

# Analyze and visualize results
python LSTM/src/analysis_results.py

# Generate interactive network visualization
python LSTM/src/pos_topology_js.py
```

## Analysis Results

Each module saves its results in its respective output directory:

- **Pre_analysis**: Initial data exploration and feature extraction results
- **Cluster_analysis**: Clustering results and dimensionality reduction visualizations
- **Topology_analysis/result**: Topology matrices and clustering logs
- **Topology_analysis/graph**: Network visualizations and animations
- **LSTM/results**: LSTM model performance metrics and prediction visualizations
- **LSTM/models**: Trained neural network models

## Contributing

Contributions to any module are welcome. Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/YourFeature`)
3. Commit your changes (`git commit -am 'Add some feature'`)
4. Push to the branch (`git push origin feature/YourFeature`)
5. Create a new Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Contact

- **Author**: ZhaoJin
- **Email**: ZhaoJ@example.com

For module-specific questions, please refer to the README files in each module directory.
