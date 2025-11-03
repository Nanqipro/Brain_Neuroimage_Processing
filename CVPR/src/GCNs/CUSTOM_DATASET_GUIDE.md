# 自定义图数据集使用指南

## 📊 数据集概述

本指南介绍如何使用 `custom` 数据源加载和训练自定义图结构数据集，特别是 **random 数据集**。

### Random 数据集结构

数据集位置：`/app/ZJ/gitlocal/Brain_Neuroimage_Processing/CVPR/data/random/`

包含4个不同规模的数据集：

| 数据集 | 图数量 | 平均节点数 | 平均边数 | 文件大小 | 文件扩展名 |
|--------|--------|-----------|---------|----------|-----------|
| **graphs_100** | 1,000 | ~100 | ~200 | 1.4K | 无 |
| **graphs_1000** | 1,000 | ~151 | ~20,000 | 148K | .txt |
| **graphs_10000** | 1,000 | ~151 | ~20,000 | 148K | 无 |
| **graphs_100000** | 1,000 | ~457 | ~200,000 | 1.7M | 无 |

### 文件格式

- **格式**: 边列表（Edge List）
- **内容**: 每行两个整数，表示一条边的两个端点
- **节点编号**: 从 0 开始
- **示例**:
  ```
  0 35
  0 92
  1 87
  3 4
  3 77
  ```

## 🚀 快速开始

### 1. 基本使用（使用随机特征）

```bash
cd /home/ZJ/.cursor/worktrees/Brain_Neuroimage_Processing__SSH__ZJ_/UNWum/CVPR/src/GCNs

# 训练 graphs_100 数据集（最小规模，快速测试）
python run.py \
    --data_source custom \
    --dataset /app/ZJ/gitlocal/Brain_Neuroimage_Processing/CVPR/data/random/graphs_100 \
    --model gcn \
    --epochs 100 \
    --feature_dim 16 \
    --num_classes 2
```

### 2. 使用节点度数作为特征

```bash
python run.py \
    --data_source custom \
    --dataset /app/ZJ/gitlocal/Brain_Neuroimage_Processing/CVPR/data/random/graphs_100 \
    --model gcn \
    --epochs 100 \
    --use_degree_feature
```

### 3. 使用更强大的模型（GIN）

```bash
python run.py \
    --data_source custom \
    --dataset /app/ZJ/gitlocal/Brain_Neuroimage_Processing/CVPR/data/random/graphs_100 \
    --model gin \
    --hidden_dim 64 \
    --dropout 0.5 \
    --epochs 200 \
    --save_results
```

## 🔬 完整实验示例

### 测试所有规模的数据集

```bash
#!/bin/bash
# 测试4个不同规模的random数据集

DATA_ROOT="/app/ZJ/gitlocal/Brain_Neuroimage_Processing/CVPR/data/random"
MODEL="gin"

# 1. graphs_100 (最小，快速测试)
python run.py --data_source custom \
    --dataset ${DATA_ROOT}/graphs_100 \
    --model ${MODEL} \
    --epochs 100 \
    --save_results \
    --gpu_id 3

# 2. graphs_1000 (中等规模)
python run.py --data_source custom \
    --dataset ${DATA_ROOT}/graphs_1000 \
    --model ${MODEL} \
    --epochs 100 \
    --save_results \
    --gpu_id 3

# 3. graphs_10000 (大规模)
python run.py --data_source custom \
    --dataset ${DATA_ROOT}/graphs_10000 \
    --model ${MODEL} \
    --epochs 50 \
    --batch_size 16 \
    --save_results \
    --gpu_id 3

# 4. graphs_100000 (超大规模)
python run.py --data_source custom \
    --dataset ${DATA_ROOT}/graphs_100000 \
    --model ${MODEL} \
    --epochs 20 \
    --batch_size 8 \
    --save_results \
    --gpu_id 3
```

### 对比不同模型性能

```bash
#!/bin/bash
# 在graphs_100上对比不同模型

DATASET="/app/ZJ/gitlocal/Brain_Neuroimage_Processing/CVPR/data/random/graphs_100"

for model in gcn gat sage gin pna gatv2; do
    echo "Testing model: $model"
    python run.py \
        --data_source custom \
        --dataset ${DATASET} \
        --model ${model} \
        --epochs 100 \
        --save_results \
        --gpu_id 3
done
```

## ⚙️ 参数说明

### 必需参数

- `--data_source custom`: 指定使用自定义数据源
- `--dataset PATH`: 数据集目录路径
- `--model MODEL`: 模型类型（gcn, gat, gin, 等）

### 自定义数据集专用参数

- `--feature_dim N`: 随机节点特征维度（默认16）
  - 较大的值可能提高表达能力，但会增加计算量
  - 推荐范围：8-64
  
- `--num_classes N`: 类别数（默认2）
  - 用于生成随机标签
  - 当前实现：标签 = 图索引 % num_classes
  
- `--use_degree_feature`: 使用节点度数作为特征
  - 如果设置，将忽略 `--feature_dim`
  - 特征维度自动变为1（度数）

### 训练参数

- `--train_ratio`: 训练集比例（默认0.6）
- `--val_ratio`: 验证集比例（默认0.2）
- `--batch_size`: 批大小（默认32）
  - 大图推荐减小：graphs_100000 建议 4-8
  - 小图可增大：graphs_100 可以 64-128
- `--epochs`: 训练轮数
- `--lr`: 学习率（默认0.001）
- `--hidden_dim`: 隐藏层维度（默认64）
- `--dropout`: Dropout比例（默认0.5）

### 其他参数

- `--gpu_id N`: GPU编号（默认3）
- `--save_results`: 保存详细结果和图表
- `--save_model`: 保存最佳模型
- `--num_runs N`: 重复训练次数
- `--seed N`: 随机种子

## 📈 预期结果

### 训练输出示例

```
加载自定义图数据集: /app/ZJ/gitlocal/Brain_Neuroimage_Processing/CVPR/data/random/graphs_100
  - 找到 1000 个边列表文件
  - 使用16维随机特征
  - 生成 2 个类别的随机标签

数据集信息:
  - 图数量: 1000
  - 节点特征维度: 16
  - 类别数: 2
  - 标签分布: Counter({0: 500, 1: 500})
  - 平均节点数: 100.0
  - 平均边数: 200.0

数据集划分:
  - 训练集: 600 图
  - 验证集: 200 图
  - 测试集: 200 图

模型信息:
  - 模型类型: gin
  - 参数量: 30,000
  - 隐藏层维度: 64
  - Dropout: 0.5

开始训练 (最多 100 轮)...
Epoch 010: Loss=0.6891, Train F1=0.5234, Val F1=0.5167, Val Acc=0.5200, Time=0.123s
Epoch 020: Loss=0.6523, Train F1=0.6012, Val F1=0.5834, Val Acc=0.5850, Time=0.121s
...
```

### 性能基准（graphs_100）

基于GIN模型的预期性能：

| 指标 | 预期范围 |
|------|---------|
| 训练时间/epoch | 0.1-0.3秒 |
| 验证准确率 | 50-70% |
| 测试F1分数 | 50-70% |

*注意：由于使用随机生成的特征和标签，实际性能可能接近随机猜测（50%）*

## 🔍 数据集详细信息

### 支持的文件格式

`load_custom_graph_dataset()` 函数支持多种格式：

1. **PyTorch格式** (`.pt`, `.pth`)
   - 单文件包含所有图：`graphs_list.pt`
   - 多文件：`graph_0.pt`, `graph_1.pt`, ...

2. **Pickle格式** (`.pkl`)
   - 单文件：`graphs_list.pkl`

3. **边列表格式** (无扩展名或`.txt`)
   - `graph_0001_seed42`
   - `graph_0001_seed42.txt`
   - 每行格式：`源节点ID 目标节点ID`

4. **GraphML格式** (`.graphml`)
   - 标准NetworkX格式

### 自动特征生成

由于random数据集只包含图结构（边），系统会自动生成节点特征：

**方式1：随机特征**（默认）
```python
x = torch.randn(num_nodes, feature_dim)  # 正态分布随机特征
```

**方式2：度数特征**（`--use_degree_feature`）
```python
degree = 每个节点的连接边数
x = degree.unsqueeze(1)  # (num_nodes, 1)
```

### 自动标签生成

```python
label = graph_index % num_classes
```

例如，如果 `num_classes=2`：
- graph_0 → label 0
- graph_1 → label 1
- graph_2 → label 0
- graph_3 → label 1
- ...

## 💡 最佳实践

### 1. 选择合适的特征表示

**使用随机特征**（推荐用于快速测试）：
```bash
--feature_dim 16  # 轻量级
--feature_dim 32  # 平衡
--feature_dim 64  # 高表达力
```

**使用度数特征**（更有意义的结构特征）：
```bash
--use_degree_feature  # 单维度，计算快
```

### 2. 根据图规模调整批大小

```bash
# graphs_100 (小图)
--batch_size 64

# graphs_1000 (中图)
--batch_size 32

# graphs_10000 (大图)
--batch_size 16

# graphs_100000 (超大图)
--batch_size 4
```

### 3. 选择合适的模型

**快速测试**：
```bash
--model gcn  # 最快
```

**最佳性能**：
```bash
--model gin   # 理论最强
--model pna   # 多聚合器
```

**处理大图**：
```bash
--model chebnet    # 高效
--model deepergcn  # 深层网络
```

### 4. 训练策略

**快速验证**（graphs_100）：
```bash
--epochs 50
--lr 0.01
--patience 10
```

**完整训练**：
```bash
--epochs 200
--lr 0.001
--patience 20
--weight_decay 5e-4
```

**大规模数据集**（graphs_100000）：
```bash
--epochs 20
--lr 0.0001
--batch_size 4
--patience 5
```

## 🐛 常见问题

### Q1: 数据加载很慢？

**答**: 大规模数据集（如graphs_100000）加载时间较长，可以：
1. 使用更少的图（通过修改目录）
2. 预先转换为 `.pt` 格式
3. 增加内存

### Q2: 训练准确率很低（~50%）？

**答**: 这是正常的！因为：
1. 使用随机生成的节点特征
2. 使用简单的标签生成规则
3. 图结构可能与标签无关

解决方法：
- 使用真实的节点特征和标签
- 或者将此作为基准测试（测试模型容量）

### Q3: Out of Memory 错误？

**答**: 减小批大小：
```bash
--batch_size 8  # 或更小
```

或减小隐藏层维度：
```bash
--hidden_dim 32  # 默认64
```

### Q4: 如何使用真实的特征和标签？

**答**: 有两种方式：

**方式1**: 修改边列表文件，添加节点特征和图标签
- 参考 GraphML 格式

**方式2**: 预先转换为 PyG 格式并保存：
```python
from process import save_custom_graph_dataset

# 构建 data_list（包含真实特征和标签）
data_list = [...]  # 你的PyG Data对象列表

# 保存
save_custom_graph_dataset(
    data_list, 
    'path/to/save/graphs_list.pt', 
    format='pt'
)
```

## 📚 代码示例

### 加载数据集（Python代码）

```python
from process import load_custom_graph_dataset

# 加载数据集
data_list, num_features, num_classes = load_custom_graph_dataset(
    data_dir='/app/ZJ/gitlocal/Brain_Neuroimage_Processing/CVPR/data/random/graphs_100',
    feature_dim=16,
    num_classes=2,
    use_degree_feature=False
)

print(f"加载了 {len(data_list)} 个图")
print(f"节点特征维度: {num_features}")
print(f"类别数: {num_classes}")

# 查看第一个图
data = data_list[0]
print(f"节点数: {data.num_nodes}")
print(f"边数: {data.num_edges}")
print(f"特征形状: {data.x.shape}")
print(f"标签: {data.y.item()}")
```

### 保存预处理数据

```python
from process import load_custom_graph_dataset, save_custom_graph_dataset

# 加载原始数据
data_list, _, _ = load_custom_graph_dataset(
    '/app/ZJ/gitlocal/Brain_Neuroimage_Processing/CVPR/data/random/graphs_100',
    feature_dim=32,
    use_degree_feature=False
)

# 保存为PyTorch格式（下次加载更快）
save_custom_graph_dataset(
    data_list,
    '/path/to/save/graphs_100_preprocessed.pt',
    format='pt'
)
```

## 🎯 总结

### 功能清单

✅ 支持边列表格式的图数据集加载  
✅ 自动节点特征生成（随机或度数）  
✅ 自动标签生成  
✅ 支持多种文件格式（.pt, .pkl, .txt, graphml）  
✅ 完整的训练流程  
✅ 结果可视化和保存  
✅ 支持多次重复实验  
✅ GPU加速支持  

### 下一步

1. **测试基本功能**：
   ```bash
   python run.py --data_source custom --dataset /app/ZJ/gitlocal/Brain_Neuroimage_Processing/CVPR/data/random/graphs_100 --model gcn --epochs 10
   ```

2. **完整实验**：
   ```bash
   python run.py --data_source custom --dataset /app/ZJ/gitlocal/Brain_Neuroimage_Processing/CVPR/data/random/graphs_100 --model gin --epochs 200 --save_results --gpu_id 3
   ```

3. **大规模测试**：
   依次测试 graphs_100 → graphs_1000 → graphs_10000 → graphs_100000

---

**作者**: AI Assistant  
**更新时间**: 2025-11-03  
**版本**: 1.0

