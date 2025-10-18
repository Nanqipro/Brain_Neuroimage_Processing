# 多数据集训练指南

## 概述

`run.py` 脚本现在支持三种运行模式：

1. **单数据集模式** - 在单个数据集上训练
2. **多数据集批量模式** - 在多个数据集上依次训练并生成汇总报告
3. **配置文件模式** - 使用JSON配置文件运行多个实验

---

## 1. 单数据集模式

在单个数据集上训练模型（原有功能）。

### TUDataset示例
```bash
python run.py --data_source tudataset --dataset MUTAG --model gcn --save_results
```

### CSV数据示例
```bash
python run.py --data_source csv --dataset dataset/processed3.csv --model gcn --save_results
```

---

## 2. 多数据集批量模式

在多个数据集上运行相同的模型配置，自动生成汇总报告。

### 基本用法

使用逗号分隔多个数据集名称：

```bash
python run.py --data_source tudataset \
    --dataset MUTAG,PROTEINS,DD,ENZYMES \
    --model gcn \
    --save_results
```

### 完整示例

```bash
# 在5个数据集上测试GIN模型
python run.py --data_source tudataset \
    --dataset MUTAG,PROTEINS,DD,ENZYMES,COLLAB \
    --model gin \
    --hidden_dim 128 \
    --dropout 0.5 \
    --epochs 200 \
    --batch_size 64 \
    --save_results
```

### 输出结果

运行后会显示：

1. **实时进度** - 每个数据集的训练进度
2. **汇总表格** - 所有数据集的性能对比
3. **平均性能** - 平均准确率、F1等指标（含标准差）
4. **保存文件**：
   - `result/summary/{model}/summary_{timestamp}.json` - JSON格式汇总
   - `result/summary/{model}/summary_{timestamp}.csv` - CSV格式汇总

### 示例输出

```
================================================================================
实验汇总报告
================================================================================

模型: gcn

  dataset  accuracy  precision    recall        f1
    MUTAG    0.8889     0.8750    0.9333    0.9032
 PROTEINS    0.7250     0.7100    0.7400    0.7247
       DD    0.7800     0.7650    0.7950    0.7797

平均性能:
  - 准确率: 0.7980 ± 0.0702
  - 精确率: 0.7833 ± 0.0703
  - 召回率: 0.8228 ± 0.0816
  - F1分数: 0.8025 ± 0.0754

================================================================================
```

---

## 3. 配置文件模式

通过JSON配置文件批量运行多个不同配置的实验。

### 创建配置文件

创建 `experiments_config.json`：

```json
{
    "description": "对比不同模型在多个数据集上的性能",
    "experiments": [
        {
            "data_source": "tudataset",
            "dataset": "MUTAG",
            "model": "gcn",
            "hidden_dim": 64,
            "dropout": 0.5,
            "batch_size": 32,
            "lr": 0.001,
            "epochs": 200,
            "seed": 42
        },
        {
            "data_source": "tudataset",
            "dataset": "MUTAG",
            "model": "gin",
            "hidden_dim": 64,
            "dropout": 0.5,
            "batch_size": 32,
            "lr": 0.001,
            "epochs": 200,
            "seed": 42
        },
        {
            "data_source": "tudataset",
            "dataset": "PROTEINS",
            "model": "gcn",
            "hidden_dim": 64,
            "dropout": 0.5,
            "batch_size": 32,
            "lr": 0.001,
            "epochs": 200,
            "seed": 42
        }
    ]
}
```

### 运行配置文件

```bash
python run.py --config experiments_config.json
```

### 支持的配置选项

在配置文件中，也可以使用多数据集语法：

```json
{
    "experiments": [
        {
            "data_source": "tudataset",
            "dataset": "MUTAG,PROTEINS,DD",
            "model": "gcn",
            "hidden_dim": 64,
            "dropout": 0.5,
            "batch_size": 32,
            "lr": 0.001,
            "epochs": 200,
            "seed": 42
        }
    ]
}
```

### 输出结果

- 每个实验的单独结果保存在各自的目录
- 总体汇总保存在 `result/batch_summary/batch_results_{timestamp}.json`

---

## 4. 常用数据集

### TUDataset常用数据集

| 数据集 | 类型 | 图数量 | 平均节点数 | 类别数 |
|--------|------|--------|-----------|--------|
| MUTAG | 小分子 | 188 | 17.9 | 2 |
| PROTEINS | 蛋白质 | 1113 | 39.1 | 2 |
| DD | 蛋白质 | 1178 | 284.3 | 2 |
| ENZYMES | 蛋白质 | 600 | 32.6 | 6 |
| COLLAB | 社交网络 | 5000 | 74.5 | 3 |
| IMDB-BINARY | 社交网络 | 1000 | 19.8 | 2 |
| REDDIT-BINARY | 社交网络 | 2000 | 429.6 | 2 |

### 推荐的基准测试组合

**小型测试**（快速验证）：
```bash
python run.py --data_source tudataset \
    --dataset MUTAG,ENZYMES \
    --model gcn \
    --save_results
```

**中型测试**（标准基准）：
```bash
python run.py --data_source tudataset \
    --dataset MUTAG,PROTEINS,DD,ENZYMES \
    --model gin \
    --save_results
```

**完整测试**（全面评估）：
```bash
python run.py --data_source tudataset \
    --dataset MUTAG,PROTEINS,DD,ENZYMES,COLLAB,IMDB-BINARY \
    --model gcn \
    --save_results
```

---

## 5. 高级使用

### 对比不同模型

创建配置文件 `model_comparison.json`：

```json
{
    "description": "对比所有模型在MUTAG数据集上的性能",
    "experiments": [
        {"data_source": "tudataset", "dataset": "MUTAG", "model": "gcn", "epochs": 200, "seed": 42},
        {"data_source": "tudataset", "dataset": "MUTAG", "model": "gat", "epochs": 200, "seed": 42},
        {"data_source": "tudataset", "dataset": "MUTAG", "model": "gin", "epochs": 200, "seed": 42},
        {"data_source": "tudataset", "dataset": "MUTAG", "model": "sage", "epochs": 200, "seed": 42},
        {"data_source": "tudataset", "dataset": "MUTAG", "model": "chebnet", "epochs": 200, "seed": 42},
        {"data_source": "tudataset", "dataset": "MUTAG", "model": "gatv2", "epochs": 200, "seed": 42}
    ]
}
```

运行：
```bash
python run.py --config model_comparison.json
```

### 不同随机种子实验

```json
{
    "description": "多次运行以评估稳定性",
    "experiments": [
        {"data_source": "tudataset", "dataset": "MUTAG", "model": "gcn", "seed": 42},
        {"data_source": "tudataset", "dataset": "MUTAG", "model": "gcn", "seed": 123},
        {"data_source": "tudataset", "dataset": "MUTAG", "model": "gcn", "seed": 456},
        {"data_source": "tudataset", "dataset": "MUTAG", "model": "gcn", "seed": 789},
        {"data_source": "tudataset", "dataset": "MUTAG", "model": "gcn", "seed": 1024}
    ]
}
```

### 超参数网格搜索

```json
{
    "description": "隐藏层维度和dropout的网格搜索",
    "experiments": [
        {"data_source": "tudataset", "dataset": "MUTAG", "model": "gcn", "hidden_dim": 32, "dropout": 0.3},
        {"data_source": "tudataset", "dataset": "MUTAG", "model": "gcn", "hidden_dim": 32, "dropout": 0.5},
        {"data_source": "tudataset", "dataset": "MUTAG", "model": "gcn", "hidden_dim": 64, "dropout": 0.3},
        {"data_source": "tudataset", "dataset": "MUTAG", "model": "gcn", "hidden_dim": 64, "dropout": 0.5},
        {"data_source": "tudataset", "dataset": "MUTAG", "model": "gcn", "hidden_dim": 128, "dropout": 0.3},
        {"data_source": "tudataset", "dataset": "MUTAG", "model": "gcn", "hidden_dim": 128, "dropout": 0.5}
    ]
}
```

---

## 6. 结果分析

### 查看汇总结果

汇总结果以JSON和CSV两种格式保存：

**JSON格式** (`result/summary/{model}/summary_{timestamp}.json`)：
```json
{
    "model": "gcn",
    "datasets": ["MUTAG", "PROTEINS", "DD"],
    "results": [
        {
            "dataset": "MUTAG",
            "model": "gcn",
            "accuracy": 0.8889,
            "precision": 0.8750,
            "recall": 0.9333,
            "f1": 0.9032,
            "status": "success"
        },
        ...
    ],
    "timestamp": "20241018_123456"
}
```

**CSV格式** (`result/summary/{model}/summary_{timestamp}.csv`)：

可直接用Excel或Pandas分析：

```python
import pandas as pd

# 读取CSV结果
df = pd.read_csv('result/summary/gcn/summary_20241018_123456.csv')

# 查看统计信息
print(df.describe())

# 绘制性能对比图
import matplotlib.pyplot as plt
df.plot(x='dataset', y=['accuracy', 'precision', 'recall', 'f1'], kind='bar')
plt.show()
```

---

## 7. 错误处理

脚本会自动处理单个数据集的失败：

- 失败的数据集会被标记但不会中断整体流程
- 汇总报告会列出所有失败的实验
- 只有成功的实验会被纳入统计

示例输出：
```
失败的实验 (1):
  - REDDIT-BINARY: CUDA out of memory
```

---

## 8. 性能优化建议

### 对于大型数据集

```bash
# 使用更大的批大小
python run.py --data_source tudataset \
    --dataset DD,REDDIT-BINARY \
    --model gcn \
    --batch_size 128 \
    --epochs 100
```

### 快速原型测试

```bash
# 减少训练轮数和耐心值
python run.py --data_source tudataset \
    --dataset MUTAG,PROTEINS \
    --model gcn \
    --epochs 50 \
    --patience 10
```

---

## 9. 完整参数列表

### 命令行参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--config` | str | None | JSON配置文件路径 |
| `--data_source` | str | - | csv 或 tudataset |
| `--dataset` | str | - | 数据集名称（支持逗号分隔） |
| `--model` | str | gcn | 模型类型 |
| `--hidden_dim` | int | 64 | 隐藏层维度 |
| `--dropout` | float | 0.5 | Dropout比例 |
| `--batch_size` | int | 32 | 批大小 |
| `--lr` | float | 0.001 | 学习率 |
| `--weight_decay` | float | 5e-4 | 权重衰减 |
| `--epochs` | int | 200 | 最大训练轮数 |
| `--patience` | int | 20 | Early stopping耐心值 |
| `--seed` | int | 42 | 随机种子 |
| `--save_results` | flag | False | 保存详细结果 |
| `--save_model` | flag | False | 保存模型权重 |

### 支持的模型

- `gcn` - Graph Convolutional Network
- `gat` - Graph Attention Network
- `sage` - GraphSAGE
- `gin` - Graph Isomorphism Network
- `chebnet` - Chebyshev GCN
- `edgeconv` - Edge Convolution Network
- `gunet` - Graph U-Net
- `pna` - Principal Neighbourhood Aggregation
- `gatv2` - Graph Attention Network v2
- `deepergcn` - DeeperGCN

---

## 10. 常见问题

**Q: 如何只在部分数据集上运行？**

A: 使用逗号分隔你需要的数据集：
```bash
python run.py --data_source tudataset --dataset MUTAG,PROTEINS --model gcn
```

**Q: 可以混合CSV和TUDataset吗？**

A: 不可以在一次运行中混合，但可以在配置文件中分别配置：
```json
{
    "experiments": [
        {"data_source": "csv", "dataset": "dataset/data1.csv", "model": "gcn"},
        {"data_source": "tudataset", "dataset": "MUTAG", "model": "gcn"}
    ]
}
```

**Q: 如何跳过已完成的实验？**

A: 当前版本会重新运行所有实验。你可以修改配置文件只包含未完成的实验。

**Q: 内存不足怎么办？**

A: 尝试减小batch_size或使用更小的数据集：
```bash
python run.py --dataset MUTAG,PROTEINS --batch_size 16
```

---

## 快速开始示例

```bash
# 1. 快速测试单个数据集
python run.py --data_source tudataset --dataset MUTAG --model gcn

# 2. 在3个数据集上测试GCN
python run.py --data_source tudataset --dataset MUTAG,PROTEINS,ENZYMES --model gcn --save_results

# 3. 使用配置文件批量运行
python run.py --config experiments_config.json

# 4. 完整的基准测试
python run.py --data_source tudataset \
    --dataset MUTAG,PROTEINS,DD,ENZYMES \
    --model gin \
    --hidden_dim 128 \
    --epochs 200 \
    --save_results \
    --save_model
```

祝实验顺利！🚀

