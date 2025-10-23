# GCN数据集下载工具

一个统一的数据集下载脚本，支持**图分类**和**节点分类**两种类型的GCN数据集。

## ⚠️ 重要提示

**您的代码是图分类任务**（使用 `DataLoader` + `global_pooling`），请下载**图分类数据集**！

## 快速开始

```bash
# 1. 查看所有可用数据集
python download_gcn_datasets.py --list

# 2. 下载推荐的图分类数据集（适合您的代码）
python download_gcn_datasets.py --dataset MUTAG --task graph

# 3. 下载中等规模数据集
python download_gcn_datasets.py --dataset PROTEINS --task graph

# 4. 下载大规模OGB数据集
python download_gcn_datasets.py --dataset ogbg-molhiv --task graph --source ogb
```

## 图分类数据集（推荐！）

这些数据集**适合您的代码**，可以直接使用 `DataLoader` + `global_pooling`。

### TUDataset 系列

| 数据集 | 图数量 | 类别数 | 平均节点数 | 平均边数 | 领域 | 描述 |
|--------|--------|--------|-----------|---------|------|------|
| **MUTAG** | 188 | 2 | ~18 | ~20 | 生物化学 | 分子图分类 - 判断化合物致癌性 |
| **PROTEINS** | 1,113 | 2 | ~39 | ~73 | 生物 | 蛋白质图分类 - 判断蛋白质是否为酶 |
| **PTC_MR** | 344 | 2 | ~14 | ~15 | 生物化学 | 化合物致癌性预测 |
| **ENZYMES** | 600 | 6 | ~33 | ~62 | 生物 | 蛋白质图分类 - 酶类型分类 |
| **NCI1** | 4,110 | 2 | ~30 | ~32 | 生物化学 | 化合物抗癌活性预测（标准基准） |
| **NCI109** | 4,127 | 2 | ~30 | ~32 | 生物化学 | 化合物抗癌活性预测 |
| **IMDB-BINARY** | 1,000 | 2 | ~20 | ~96 | 社交网络 | 电影协作网络 - 判断电影类型 |
| **IMDB-MULTI** | 1,500 | 3 | ~13 | ~66 | 社交网络 | 电影协作网络 - 判断电影类型（3类） |
| **REDDIT-BINARY** | 2,000 | 2 | ~430 | ~498 | 社交网络 | Reddit社区图分类 |
| **DD** | 1,178 | 2 | ~284 | ~716 | 生物 | 蛋白质图分类 - 酶/非酶 |

### OGB (Open Graph Benchmark)

| 数据集 | 图数量 | 任务类型 | 平均节点数 | 平均边数 | 领域 | 描述 |
|--------|--------|----------|-----------|---------|------|------|
| **ogbg-molhiv** | 41,127 | 二分类 | ~26 | ~27 | 药物发现 | 分子图 - HIV抑制剂预测 |
| **ogbg-molpcba** | 437,929 | 多标签分类 | ~26 | ~28 | 药物发现 | 分子图 - 生物活性预测（大规模） |
| **ogbg-ppa** | 158,100 | 多分类 | ~243 | ~2,266 | 生物 | 蛋白质-蛋白质相互作用图 |

## 使用方法

### 查看所有数据集

```bash
python download_gcn_datasets.py --list
```

### 下载图分类数据集

```bash
# TUDataset系列（推荐从小数据集开始）
python download_gcn_datasets.py --dataset MUTAG --task graph
python download_gcn_datasets.py --dataset PROTEINS --task graph
python download_gcn_datasets.py --dataset NCI1 --task graph

# OGB系列（需要安装ogb: pip install ogb）
python download_gcn_datasets.py --dataset ogbg-molhiv --task graph --source ogb

# 指定保存路径
python download_gcn_datasets.py --dataset MUTAG --task graph --data_root /path/to/data
```

## 在代码中使用数据集

### TUDataset 示例

```python
from torch_geometric.datasets import TUDataset
from torch_geometric.loader import DataLoader
from model import ImprovedGCN  # 您的模型

# 1. 加载数据集
dataset = TUDataset(root='./data/TUDataset', name='MUTAG')

print(f"数据集信息:")
print(f"  - 图数量: {len(dataset)}")
print(f"  - 特征维度: {dataset.num_features}")
print(f"  - 类别数: {dataset.num_classes}")

# 2. 划分训练/验证/测试集
train_size = int(len(dataset) * 0.8)
val_size = int(len(dataset) * 0.1)

train_dataset = dataset[:train_size]
val_dataset = dataset[train_size:train_size+val_size]
test_dataset = dataset[train_size+val_size:]

# 3. 创建DataLoader（与您的代码兼容）
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32)
test_loader = DataLoader(test_dataset, batch_size=32)

# 4. 初始化模型
model = ImprovedGCN(
    num_features=dataset.num_features,
    hidden_dim=64,
    num_classes=dataset.num_classes,
    dropout=0.3
)

# 5. 训练模型（使用您现有的train.py）
# optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
# for epoch in range(200):
#     train_metrics = train_model(model, train_loader, optimizer, device)
#     val_metrics = evaluate_model(model, val_loader, device)
#     ...
```

### OGB 示例

```python
from ogb.graphproppred import PygGraphPropPredDataset
from torch_geometric.loader import DataLoader
from model import ImprovedGCN

# 1. 加载数据集（OGB自带数据划分）
dataset = PygGraphPropPredDataset(name='ogbg-molhiv', root='./data/OGB')
split_idx = dataset.get_idx_split()

print(f"数据集信息:")
print(f"  - 训练集: {len(split_idx['train'])}")
print(f"  - 验证集: {len(split_idx['valid'])}")
print(f"  - 测试集: {len(split_idx['test'])}")

# 2. 创建DataLoader
train_loader = DataLoader(dataset[split_idx['train']], batch_size=32, shuffle=True)
val_loader = DataLoader(dataset[split_idx['valid']], batch_size=32)
test_loader = DataLoader(dataset[split_idx['test']], batch_size=32)

# 3. 初始化模型（注意：OGB使用num_tasks而不是num_classes）
model = ImprovedGCN(
    num_features=dataset.num_features,
    hidden_dim=64,
    num_classes=dataset.num_tasks,  # OGB使用num_tasks
    dropout=0.3
)
```

## 推荐的学习路径

| 阶段 | 数据集 | 图规模 | 原因 |
|------|--------|--------|------|
| 1️⃣ 快速测试 | MUTAG (188图) | 小图 (~18节点) | 最小数据集，快速验证代码 |
| 2️⃣ 初步实验 | PROTEINS (1,113图) | 小图 (~39节点) | 中小规模，适合调参 |
| 3️⃣ 标准基准 | NCI1 (4,110图) | 小图 (~30节点) | 论文常用基准，结果可对比 |
| 4️⃣ 多分类测试 | ENZYMES (600图, 6类) | 小图 (~33节点) | 测试多分类性能 |
| 5️⃣ 大规模测试 | ogbg-molhiv (41,127图) | 小图 (~26节点) | 大规模标准基准 |

## 节点分类数据集（仅供参考）

⚠️ **这些数据集与您当前的图分类代码不兼容！** 需要不同的代码结构。

- **Cora**: 论文引用网络 (2,708节点, 7类)
- **CiteSeer**: 论文引用网络 (3,327节点, 6类)
- **Pubmed**: 论文引用网络 (19,717节点, 3类)

如需下载节点分类数据集：
```bash
python download_gcn_datasets.py --dataset cora --task node --source pyg
```

## 安装依赖

### 基础依赖（TUDataset）

```bash
pip install torch torch-geometric
```

### OGB数据集

```bash
pip install ogb
```

### 完整安装

```bash
# PyTorch Geometric完整安装
pip install torch-scatter torch-sparse -f https://data.pyg.org/whl/torch-2.0.0+cu118.html
pip install torch-geometric

# OGB
pip install ogb
```

## 常见问题

### Q1: 如何选择合适的数据集？

- **快速验证代码**: MUTAG (188图，每图~18节点，几秒钟)
- **调参实验**: PROTEINS (1,113图，每图~39节点，几分钟)
- **论文对比**: NCI1 (4,110图，每图~30节点，标准基准)
- **大规模实验**: ogbg-molhiv (41,127图，每图~26节点，需要GPU)

💡 **图大小说明**：
- 小分子数据集（MUTAG、NCI1、ogbg-molhiv等）：每张图通常10-30个节点，适合快速训练
- 蛋白质数据集（PROTEINS、DD等）：每张图30-300个节点，计算量中等
- 社交网络数据集（REDDIT-BINARY）：每张图可达数百节点，需要更多计算资源

### Q2: 数据集下载很慢？

- 首次下载会从网络获取，请耐心等待
- 数据会缓存在本地，下次使用会很快
- 可以尝试设置代理加速下载

### Q3: 如何查看数据集详细信息？

```python
from torch_geometric.datasets import TUDataset

dataset = TUDataset(root='./data/TUDataset', name='MUTAG')
print(f"图数量: {len(dataset)}")
print(f"特征维度: {dataset.num_features}")
print(f"类别数: {dataset.num_classes}")

# 查看第一个图
data = dataset[0]
print(f"节点数: {data.num_nodes}")
print(f"边数: {data.num_edges}")
print(f"节点特征: {data.x.shape}")
print(f"标签: {data.y}")
```

### Q4: Cora等数据集能用吗？

不能！Cora/CiteSeer/Pubmed是**节点分类数据集**（单个大图），您的代码是**图分类任务**（多个小图）。请使用MUTAG、PROTEINS、NCI1等图分类数据集。

### Q5: 如何知道我的代码是图分类还是节点分类？

看您的模型代码：
- ✅ **图分类**: 使用 `global_mean_pool` 或 `global_add_pool` → 用MUTAG等
- ❌ **节点分类**: 没有global pooling，直接输出节点特征 → 用Cora等

您的代码是图分类！

### Q6: 表格中的图大小（节点数和边数）是什么意思？

- **平均节点数**：数据集中每张图的平均节点（顶点）数量
- **平均边数**：数据集中每张图的平均边（连接）数量
- **示例**：MUTAG数据集有188张图，每张图平均有~18个节点和~20条边

这些数据可以帮助您：
1. **估算计算量**：节点/边越多，训练越慢，显存占用越大
2. **选择batch_size**：大图需要小的batch_size，小图可以用大的batch_size
3. **预估训练时间**：MUTAG（小图）几秒钟，REDDIT-BINARY（大图）可能需要几分钟

## 参考文献

### TUDataset
- [TUDataset: A collection of benchmark datasets for graph classification](https://chrsmrrs.github.io/datasets/)

### OGB
- **ogbg-molhiv**: [Open Graph Benchmark: Datasets for Machine Learning on Graphs](https://ogb.stanford.edu/)
- 论文: Hu et al. "Open Graph Benchmark: Datasets for Machine Learning on Graphs" (NeurIPS 2020)

### 相关资源
- [PyTorch Geometric 文档](https://pytorch-geometric.readthedocs.io/)
- [OGB 官方网站](https://ogb.stanford.edu/)
- [TUDataset 数据集列表](https://chrsmrrs.github.io/datasets/docs/datasets/)

## 技术支持

遇到问题？
1. 检查是否安装了 `torch-geometric`
2. 查看 `python download_gcn_datasets.py --list` 确认数据集名称
3. 确保网络连接正常
4. OGB数据集需要额外安装: `pip install ogb`

---

**祝您实验顺利！** 🚀
