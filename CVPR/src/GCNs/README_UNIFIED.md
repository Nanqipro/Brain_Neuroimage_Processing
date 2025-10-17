# 通用图分类训练脚本使用指南

## 🎯 统一脚本的优势

**`run_unified.py`** - 一个脚本支持所有数据源：
- ✅ TUDataset标准数据集（MUTAG, PROTEINS, NCI1等）
- ✅ 原始CSV数据（神经元钙离子数据）
- ✅ 统一的API和参数
- ✅ 代码更简洁，易于维护

## 🚀 快速开始

### TUDataset（标准基准测试）

```bash
# 在MUTAG上训练GCN
python run_unified.py --data_source tudataset --dataset MUTAG --model gcn

# 在PROTEINS上训练GAT
python run_unified.py --data_source tudataset --dataset PROTEINS --model gat

# 保存完整结果
python run_unified.py --data_source tudataset --dataset MUTAG --model gcn --save_results
```

### CSV数据（神经元数据）

```bash
# 使用原始CSV数据训练
python run_unified.py --data_source csv --dataset ../../dataset/processed3.csv --model gcn

# 自定义参数
python run_unified.py \
    --data_source csv \
    --dataset ../../dataset/processed3.csv \
    --model gcn \
    --hidden_dim 64 \
    --dropout 0.3 \
    --epochs 200 \
    --save_results
```

## 📋 完整参数说明

### 必需参数

- `--data_source`: 数据源类型
  - `tudataset`: TUDataset标准数据集
  - `csv`: CSV文件（神经元数据）

- `--dataset`: 数据集名称或路径
  - TUDataset: `MUTAG`, `PROTEINS`, `NCI1` 等
  - CSV: 文件路径，如 `../../dataset/processed3.csv`

### 数据参数

- `--data_root`: TUDataset根目录（默认：`../../data/TUDataset`）
- `--train_ratio`: 训练集比例（默认：0.8）
- `--val_ratio`: 验证集比例（默认：0.1）

### 模型参数

- `--model`: 模型类型（默认：`gcn`）
  - `gcn`: 纯GCN
  - `gat`: 纯GAT
  - `sage`: 纯GraphSAGE
  - `hybrid`: 混合模型

- `--hidden_dim`: 隐藏层维度（默认：64）
- `--dropout`: Dropout比例（默认：0.5）

### 训练参数

- `--batch_size`: 批大小（默认：32）
- `--lr`: 学习率（默认：0.001）
- `--weight_decay`: 权重衰减（默认：5e-4）
- `--epochs`: 最大训练轮数（默认：200）
- `--patience`: Early stopping耐心值（默认：20）
- `--seed`: 随机种子（默认：42）
- `--print_every`: 打印间隔（默认：10）

### 保存参数

- `--save_results`: 保存详细结果和图表
- `--save_model`: 保存最佳模型

## 💡 使用场景对比

### 场景1: 标准基准测试（论文）

**使用TUDataset**

```bash
# 在标准数据集上对比不同模型
for model in gcn gat sage hybrid; do
    python run_unified.py \
        --data_source tudataset \
        --dataset MUTAG \
        --model $model \
        --save_results
done

# 在多个数据集上测试
for dataset in MUTAG PROTEINS NCI1; do
    python run_unified.py \
        --data_source tudataset \
        --dataset $dataset \
        --model gcn \
        --save_results
done
```

### 场景2: 神经元数据分析

**使用CSV数据**

```bash
# 训练神经元数据模型
python run_unified.py \
    --data_source csv \
    --dataset ../../dataset/processed3.csv \
    --model hybrid \
    --hidden_dim 64 \
    --dropout 0.3 \
    --epochs 200 \
    --save_results
```

## 📊 结果保存结构

### TUDataset结果

```
result/tudataset/{数据集}/{模型}/{时间戳}/
├── best_model.pth
├── experiment_results.json
├── training_metrics.png
└── confusion_matrix.png
```

### CSV数据结果

```
result/csv/{模型}/{时间戳}/
├── best_model.pth
├── experiment_results.json
├── training_metrics.png
└── confusion_matrix.png
```

## 🔄 与原有脚本的对比

| 功能 | `run.py` (原) | `run_tudataset.py` | `run_unified.py` ⭐ |
|------|--------------|-------------------|-------------------|
| CSV数据 | ✅ | ❌ | ✅ |
| TUDataset | ❌ | ✅ | ✅ |
| 统一API | ❌ | ❌ | ✅ |
| 代码维护 | 需维护两份 | 需维护两份 | 只维护一份 ✅ |

## 🎨 使用示例

### 示例1: 快速验证（TUDataset）

```bash
python run_unified.py \
    --data_source tudataset \
    --dataset MUTAG \
    --model gcn \
    --epochs 50
```

### 示例2: 完整实验（TUDataset）

```bash
python run_unified.py \
    --data_source tudataset \
    --dataset PROTEINS \
    --model gcn \
    --hidden_dim 128 \
    --dropout 0.5 \
    --batch_size 64 \
    --epochs 200 \
    --save_results \
    --save_model
```

### 示例3: CSV数据训练

```bash
python run_unified.py \
    --data_source csv \
    --dataset ../../dataset/processed3.csv \
    --model gcn \
    --epochs 200 \
    --save_results
```

### 示例4: 超参数搜索

```bash
# 测试不同隐藏层维度
for hidden_dim in 32 64 128 256; do
    python run_unified.py \
        --data_source tudataset \
        --dataset MUTAG \
        --model gcn \
        --hidden_dim $hidden_dim \
        --save_results
done

# 测试不同dropout
for dropout in 0.3 0.5 0.7; do
    python run_unified.py \
        --data_source tudataset \
        --dataset MUTAG \
        --model gcn \
        --dropout $dropout \
        --save_results
done
```

## ❓ 常见问题

### Q1: 我应该使用哪个脚本？

**推荐使用 `run_unified.py`**，它可以替代：
- `run.py`（原始CSV数据脚本）
- `run_tudataset.py`（TUDataset专用脚本）

### Q2: 原来的脚本还能用吗？

可以！但建议统一使用 `run_unified.py`：
- 功能完全相同
- API更统一
- 更易维护

### Q3: 如何迁移到统一脚本？

```bash
# 原来的命令（run.py）
python run.py --model gcn --dataset dataset/processed3.csv

# 改为（run_unified.py）
python run_unified.py --data_source csv --dataset ../../dataset/processed3.csv --model gcn

# 原来的命令（run_tudataset.py）
python run_tudataset.py --dataset MUTAG --model gcn

# 改为（run_unified.py）
python run_unified.py --data_source tudataset --dataset MUTAG --model gcn
```

### Q4: 查看帮助信息

```bash
python run_unified.py --help
```

## 🎯 推荐工作流

1. **快速测试** (2分钟)
   ```bash
   python run_unified.py --data_source tudataset --dataset MUTAG --model gcn --epochs 10
   ```

2. **模型对比** (15分钟)
   ```bash
   for model in gcn gat sage; do
       python run_unified.py --data_source tudataset --dataset MUTAG --model $model --save_results
   done
   ```

3. **多数据集评估** (30分钟)
   ```bash
   for dataset in MUTAG PROTEINS NCI1; do
       python run_unified.py --data_source tudataset --dataset $dataset --model gcn --save_results
   done
   ```

4. **最终论文实验** (1-2小时)
   ```bash
   python run_unified.py \
       --data_source tudataset \
       --dataset NCI1 \
       --model gcn \
       --hidden_dim 128 \
       --epochs 300 \
       --save_results \
       --save_model
   ```

## 📖 相关文档

- 数据集下载: `../../README_GCN_DATASETS.md`
- 模型定义: `model.py`
- 训练函数: `train.py`
- 数据处理: `process.py`

---

**统一脚本，简化工作流！** 🚀

