# GCN标准数据集下载工具

一个用于下载常用GCN（图卷积网络）标准数据集的Python脚本。

## 支持的数据集

| 数据集 | 节点数 | 边数 | 特征维度 | 类别数 | 任务类型 |
|--------|--------|------|----------|--------|----------|
| **Cora** | 2,708 | 5,429 | 1,433 | 7 | 节点分类（论文引用网络） |
| **CiteSeer** | 3,327 | 4,732 | 3,703 | 6 | 节点分类（论文引用网络） |
| **Pubmed** | 19,717 | 44,338 | 500 | 3 | 节点分类（论文引用网络） |
| **Reddit** | 232,965 | 11,606,919 | 602 | 41 | 大规模节点分类（社交网络） |
| **PPI** | 24图 | - | 50 | 121 | 多标签节点分类（蛋白质网络） |

## 安装依赖

### 方法1: 使用PyTorch Geometric（推荐）

```bash
pip install torch torch-geometric
```

### 方法2: 使用DGL

```bash
pip install torch dgl
```

### 方法3: 使用ModelScope

```bash
pip install modelscope
```

## 使用方法

### 基本用法

```bash
# 下载Cora数据集（最常用）
python download_gcn_datasets.py --datasets cora --method pyg

# 下载多个数据集
python download_gcn_datasets.py --datasets cora citeseer pubmed --method pyg

# 下载所有数据集
python download_gcn_datasets.py --datasets all --method pyg
```

### 参数说明

- `--datasets`: 要下载的数据集名称
  - 可选值: `cora`, `citeseer`, `pubmed`, `reddit`, `ppi`, `all`
  - 可以指定多个数据集，用空格分隔
  
- `--method`: 下载方法
  - `pyg`: 使用PyTorch Geometric（推荐）
  - `dgl`: 使用Deep Graph Library
  - `modelscope`: 使用ModelScope（主要用于中文环境）
  
- `--data_root`: 数据集保存路径（可选）
  - 默认值: `./data/gcn_datasets`

### 使用示例

```bash
# 示例1: 快速开始 - 下载Cora数据集
python download_gcn_datasets.py --datasets cora --method pyg

# 示例2: 下载论文常用的三个基准数据集
python download_gcn_datasets.py --datasets cora citeseer pubmed --method pyg

# 示例3: 下载所有数据集
python download_gcn_datasets.py --datasets all --method pyg

# 示例4: 指定自定义保存路径
python download_gcn_datasets.py --datasets cora --method pyg --data_root /path/to/your/data

# 示例5: 使用DGL下载
python download_gcn_datasets.py --datasets cora --method dgl

# 示例6: 查看帮助信息
python download_gcn_datasets.py --help
```

## 在代码中使用下载的数据集

### PyTorch Geometric

```python
from torch_geometric.datasets import Planetoid

# 加载数据集
dataset = Planetoid(root='./data/gcn_datasets', name='Cora')
data = dataset[0]

# 查看数据集信息
print(f'节点数: {data.num_nodes}')
print(f'边数: {data.num_edges}')
print(f'特征维度: {dataset.num_features}')
print(f'类别数: {dataset.num_classes}')
```

### DGL

```python
from dgl.data import CoraGraphDataset

# 加载数据集
dataset = CoraGraphDataset(raw_dir='./data/gcn_datasets')
graph = dataset[0]

# 查看数据集信息
print(f'节点数: {graph.num_nodes()}')
print(f'边数: {graph.num_edges()}')
print(f'特征维度: {graph.ndata["feat"].shape[1]}')
print(f'类别数: {dataset.num_classes}')
```

## 数据集选择建议

- **快速原型开发**: 使用 **Cora**（规模最小，速度最快）
- **标准论文基准**: 使用 **Cora + CiteSeer + Pubmed**（最常用组合）
- **大规模图测试**: 使用 **Reddit**（需要GPU）
- **归纳学习测试**: 使用 **PPI**（多图、多标签）

## 常见问题

### Q1: 下载失败或速度很慢？
```bash
# 尝试使用DGL替代PyG
python download_gcn_datasets.py --datasets cora --method dgl

# 或设置代理
export http_proxy=http://your-proxy:port
export https_proxy=http://your-proxy:port
```

### Q2: 显示"未安装 torch_geometric"错误？
```bash
pip install torch torch-geometric
```

### Q3: Reddit数据集太大？
Reddit数据集约7GB，如果存储空间不足，建议只下载Cora/CiteSeer/Pubmed。

### Q4: 如何查看已下载的数据集？
```bash
ls -lh ./data/gcn_datasets/
```

## 参考文献

- **GCN原论文**: [Semi-Supervised Classification with Graph Convolutional Networks (ICLR 2017)](https://arxiv.org/abs/1609.02907)
- **Cora/CiteSeer/Pubmed**: [Revisiting Semi-Supervised Learning with Graph Embeddings](https://arxiv.org/abs/1603.08861)
- **Reddit**: [Inductive Representation Learning on Large Graphs (NeurIPS 2017)](https://arxiv.org/abs/1706.02216)
- **PPI**: [Predicting Multicellular Function through Multi-layer Tissue Networks](https://arxiv.org/abs/1707.04638)

## 相关资源

- [PyTorch Geometric 文档](https://pytorch-geometric.readthedocs.io/)
- [DGL 文档](https://www.dgl.ai/)
- [Open Graph Benchmark](https://ogb.stanford.edu/)

## 许可证

本脚本遵循MIT许可证。数据集版权归原作者所有。

