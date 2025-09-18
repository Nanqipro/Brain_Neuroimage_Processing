# SOTA 图神经网络模型说明

## 新增的SOTA模型

在原有的4个模型（GCN, GAT, GraphSAGE, Hybrid）基础上，新增了4个SOTA模型：

### 1. GIN (Graph Isomorphism Network)
- **论文**: How Powerful are Graph Neural Networks? (ICLR 2019)
- **特点**: 
  - 理论上最强大的GNN架构之一
  - 使用MLP作为聚合函数，能够区分不同的图结构
  - 采用多层表示拼接，保留不同尺度的信息
- **适用场景**: 需要强大表达能力的图分类任务

### 2. GraphTransformer
- **特点**:
  - 使用注意力机制的图神经网络
  - 能够捕获全局信息和长程依赖
  - 使用多头注意力机制
  - 结合三种池化方式（mean, max, add）
- **适用场景**: 大规模图、需要全局信息的任务

### 3. ChebNet (Chebyshev Network)
- **论文**: Convolutional Neural Networks on Graphs with Fast Localized Spectral Filtering (NeurIPS 2016)
- **特点**:
  - 使用切比雪夫多项式近似谱卷积
  - 计算效率高，避免特征分解
  - K阶多项式控制感受野大小
- **适用场景**: 需要高效计算的大规模图

### 4. EnsembleGNN
- **特点**:
  - 集成多种GNN架构（GIN + Transformer + ChebNet）
  - 融合不同架构的优势
  - 自适应学习不同卷积的贡献
- **适用场景**: 追求最高性能，计算资源充足的场景

## 使用方法

### 1. 运行单个SOTA模型
```bash
# 运行GIN模型
python run.py --model gin --runs 50 --window_size 20

# 运行GraphTransformer
python run.py --model transformer --runs 50 --window_size 50

# 运行ChebNet
python run.py --model chebnet --runs 50 --window_size 100

# 运行集成模型
python run.py --model ensemble --runs 50
```

### 2. 批量实验（默认运行4个主要模型）
```bash
# 运行GCN, GAT, GraphSAGE, GIN
python batch_experiments.py

# 运行所有8个模型
python batch_experiments.py --models gcn gat sage hybrid gin transformer chebnet ensemble

# 快速测试
python batch_experiments.py --quick_test --models gin transformer
```

## 性能考虑

1. **计算复杂度**:
   - GCN, GAT, GraphSAGE: 标准复杂度
   - GIN: 稍高（因为MLP）
   - GraphTransformer: 较高（注意力机制）
   - ChebNet: 中等（取决于K值）
   - EnsembleGNN: 最高（集成多个模型）

2. **显存占用**:
   - 对于大图，建议减小batch_size
   - GraphTransformer和EnsembleGNN占用显存较多
   - 如遇到OOM，可以调整hidden_dim参数

## 实验建议

1. **基准测试**: 先运行GCN作为基准
2. **逐步提升**: GCN → GAT/GraphSAGE → GIN → GraphTransformer
3. **集成模型**: 最后尝试EnsembleGNN，追求最高性能
4. **参数调优**: 
   - hidden_dim: 32, 64, 128
   - dropout: 0.2, 0.3, 0.5
   - 对于GraphTransformer: heads可以尝试2, 4, 8
   - 对于ChebNet: K可以尝试2, 3, 5

## 预期结果

根据文献和经验：
- **准确率排序（通常）**: EnsembleGNN > GIN ≈ GraphTransformer > GAT > GraphSAGE ≈ GCN
- **速度排序**: GCN > GraphSAGE > GAT ≈ ChebNet > GIN > GraphTransformer > EnsembleGNN
- **稳定性**: GIN和ChebNet通常较稳定

注意：实际结果会因数据集特性而异。
