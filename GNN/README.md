# 神经元网络图神经网络分析模块

基于BrainGB框架的神经元网络图神经网络分析模块，用于优化神经元之间的连接分析。

## 功能特点

- **多种节点特征表示**：支持连接配置文件(Connection Profile)、特征向量(Eigen)、度配置文件(Degree Profile)和独热编码(Identity)等多种节点特征提取方法
- **灵活的消息传递机制**：实现了四种消息传递类型，包括边权重(Edge Weighted)、箱拼接(Bin Concat)、节点-边拼接(Node-Edge Concat)和节点拼接(Node Concat)
- **注意力增强**：添加了基于图注意力网络(GAT)的注意力机制，增强了模型捕捉关键连接的能力
- **动态图分析**：支持基于滑动窗口的神经元动态网络分析，揭示神经元连接随时间的变化
- **GNN-LSTM混合模型**：将图神经网络与LSTM结合，同时捕捉神经元的拓扑结构和时序模式
- **社区结构分析**：基于图算法检测神经元的功能模块和社区结构
- **节点嵌入可视化**：通过降维技术(t-SNE/PCA)可视化神经元节点嵌入

## 系统架构

整个系统分为以下几个核心组件：

1. **节点特征提取器**：从神经元相关矩阵中提取节点特征
2. **消息传递层**：实现不同类型的图神经网络消息传递机制
3. **BrainGNN**：基于BrainGB架构的图神经网络模型
4. **NeuronGNNLSTM**：将GNN和LSTM结合的混合模型
5. **GNNLSTMIntegrator**：用于集成LSTM和GNN分析的接口类

## 使用方法

### 基本用法

```python
from gnn_lstm_integration import GNNLSTMIntegrator
from analysis_config import AnalysisConfig

# 创建配置
config = AnalysisConfig()

# 创建集成器
integrator = GNNLSTMIntegrator(config)

# 加载和预处理数据
X_scaled, y, behavior_labels = processor.preprocess_data()

# 运行完整分析
results = integrator.run_full_analysis(X_scaled, y)
```

### 高级用法

您可以单独使用各个组件进行更精细的分析：

```python
# 创建神经元功能连接图
G, correlation_matrix, available_neurons = integrator.create_neuron_graph(X_scaled)

# 分析社区结构
communities, metrics = integrator.analyze_community_structure(G, correlation_matrix, available_neurons)

# 创建动态图数据
correlation_matrices, time_indices = integrator.create_windowed_correlation_matrices(X_scaled)
graph_data_list = integrator.prepare_graph_data(correlation_matrices)

# 训练独立的GNN模型
gnn_model, node_embeddings = integrator.train_gnn_model(graph_data_list, labels, feature_dim)

# 可视化节点嵌入
integrator.visualize_embeddings(node_embeddings, labels)
```

## 参数配置

在`analysis_config.py`中可以自定义以下GNN相关参数：

```python
# GNN模型参数
self.gnn_feature_type = 'connection_profile'  # 节点特征类型
self.gnn_message_type = 'node_edge_concat'    # 消息传递类型
self.gnn_hidden_dim = 64                      # GNN隐藏层维度
self.gnn_num_layers = 2                       # GNN层数
self.gnn_pooling = 'mean'                     # 池化方法
self.gnn_dropout = 0.2                        # Dropout比率

# GNN-LSTM混合模型参数
self.window_size = 20                         # 滑动窗口大小
self.step_size = 10                           # 滑动步长
self.correlation_threshold = 0.3              # 相关性阈值
```

## 安装依赖

安装所需依赖：

```bash
pip install -r requirements.txt
```

对于PyG(PyTorch Geometric)，建议参考[官方安装指南](https://pytorch-geometric.readthedocs.io/en/latest/notes/installation.html)，确保安装与您的CUDA版本匹配的版本。

## 参考文献

1. Zhang, Y., Huang, C., et al. (2023). BrainGB: A Benchmark for Brain Network Analysis with Graph Neural Networks. *IEEE Transactions on Medical Imaging*.
2. Li, X., Zhou, Y., et al. (2021). BrainGNN: Interpretable Brain Graph Neural Network for fMRI Analysis. *Medical Image Analysis*.
3. Ktena, S. I., Parisot, S., et al. (2018). Metric learning with spectral graph convolutions on brain connectivity networks. *NeuroImage*. 