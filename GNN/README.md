# 神经元拓扑结构分析

该项目使用图神经网络（GNN）对小鼠脑神经的钙离子浓度波动数据进行分析，构建神经元之间的拓扑结构。

## 项目简介

该项目基于GNN模型（GCN或GAT）对神经元钙离子浓度波动数据进行建模，通过学习神经元之间的关系，构建拓扑结构图，从而揭示神经元之间的功能联系。

主要功能：
- 加载和预处理神经元钙离子浓度波动数据
- 基于数据相关性构建初始图结构
- 使用GNN模型（GCN或GAT）学习神经元节点表示
- 基于GNN模型学习的节点表示构建优化的拓扑结构
- 静态和交互式可视化拓扑结构
- 分析社区结构和节点中心性

## 环境配置

### 必要的依赖项

```bash
pip install -r requirements.txt
```

### 主要依赖

- torch
- torch-geometric
- networkx
- matplotlib
- numpy
- pandas
- scikit-learn
- plotly

## 项目结构

```
GNN/
├── datasets/             # 数据集
│   └── Day6_with_behavior_labels_filled.xlsx  # 神经元数据
├── models/               # 保存的模型
├── results/              # 分析结果
├── src/                  # 源代码
│   ├── config/           # 配置
│   │   └── config.py     # 配置类
│   ├── core/             # 核心功能
│   │   ├── gcn_model.py  # GCN模型
│   │   ├── gat_model.py  # GAT模型
│   │   └── train.py      # 模型训练
│   └── utils/            # 工具函数
│       ├── data_processor.py  # 数据处理
│       └── visualize.py       # 可视化
├── main.py               # 主程序
└── requirements.txt      # 依赖项
```

## 使用方法

### 基本使用

```bash
python main.py
```

这将使用默认参数运行分析，包括使用GCN模型，不训练新模型（除非找不到已有模型），不执行可视化和社区分析。

### 训练新模型

```bash
python main.py --train
```

### 可视化拓扑结构

```bash
python main.py --visualize
```

### 分析社区结构

```bash
python main.py --analyze
```

### 使用GAT模型

```bash
python main.py --model_type gat
```

### 使用自定义参数

```bash
python main.py --hidden_channels 128 --out_channels 64 --num_layers 3 --dropout 0.3 --lr 0.0005 --epochs 200 --correlation_threshold 0.6
```

## 分析结果

执行分析后，结果将保存在`results/`目录下：

1. **原始网络结构**
   - `original_graph.png`: 基于相关性构建的初始图结构

2. **训练历史**
   - `training_history_gcn.png`/`training_history_gat.png`: 训练历史曲线

3. **嵌入可视化**
   - `embeddings_gcn.png`/`embeddings_gat.png`: 神经元嵌入可视化

4. **拓扑结构**
   - `topology_gcn.json`/`topology_gat.json`: 拓扑结构数据
   - `topology_static_gcn.png`/`topology_static_gat.png`: 静态拓扑图
   - `topology_interactive_gcn.html`/`topology_interactive_gat.html`: 交互式拓扑图

5. **社区分析**
   - `community_analysis_gcn.json`/`community_analysis_gat.json`: 社区分析结果 