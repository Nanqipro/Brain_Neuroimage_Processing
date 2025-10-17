# 使用TUDataset训练模型指南

## 快速开始

### 1. 确保已下载数据集

```bash
cd /app/ZJ/gitlocal/Brain_Neuroimage_Processing/CVPR

# 查看可用数据集
python download_gcn_datasets.py --list

# 下载MUTAG数据集（如果还没下载）
python download_gcn_datasets.py --dataset MUTAG --task graph
```

### 2. 训练模型

```bash
cd src/GCNs

# 使用默认参数在MUTAG上训练GCN
python run_tudataset.py --dataset MUTAG --model gcn

# 使用自定义参数
python run_tudataset.py --dataset MUTAG --model gcn --hidden_dim 64 --dropout 0.5 --epochs 200

# 保存详细结果
python run_tudataset.py --dataset MUTAG --model gcn --save_results --save_model

# 尝试其他模型
python run_tudataset.py --dataset MUTAG --model gat      # GAT模型
python run_tudataset.py --dataset MUTAG --model sage     # GraphSAGE模型
python run_tudataset.py --dataset MUTAG --model hybrid   # 混合模型
```

### 3. 在其他数据集上训练

```bash
# PROTEINS数据集
python run_tudataset.py --dataset PROTEINS --model gcn --batch_size 64

# NCI1数据集
python run_tudataset.py --dataset NCI1 --model gcn --batch_size 128

# ENZYMES数据集（多分类）
python run_tudataset.py --dataset ENZYMES --model gcn
```

## 完整参数说明

### 数据集参数
- `--dataset`: 数据集名称（MUTAG, PROTEINS, NCI1, ENZYMES等）
- `--data_root`: 数据集根目录（默认：`../../data/TUDataset`）
- `--train_ratio`: 训练集比例（默认：0.8）
- `--val_ratio`: 验证集比例（默认：0.1）

### 模型参数
- `--model`: 模型类型
  - `gcn`: 纯GCN（3层）
  - `gat`: 纯GAT（3层，带注意力机制）
  - `sage`: 纯GraphSAGE（3层）
  - `hybrid`: 混合模型（GCN+SAGE+GAT）
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

## 使用示例

### 示例1: 快速测试（MUTAG，最小数据集）

```bash
python run_tudataset.py \
    --dataset MUTAG \
    --model gcn \
    --epochs 100 \
    --print_every 10
```

### 示例2: 完整实验（保存所有结果）

```bash
python run_tudataset.py \
    --dataset PROTEINS \
    --model gcn \
    --hidden_dim 128 \
    --dropout 0.5 \
    --batch_size 64 \
    --lr 0.001 \
    --epochs 200 \
    --patience 30 \
    --save_results \
    --save_model
```

### 示例3: 对比不同模型

```bash
# GCN
python run_tudataset.py --dataset MUTAG --model gcn --save_results

# GAT  
python run_tudataset.py --dataset MUTAG --model gat --save_results

# GraphSAGE
python run_tudataset.py --dataset MUTAG --model sage --save_results

# 混合模型
python run_tudataset.py --dataset MUTAG --model hybrid --save_results
```

### 示例4: 多数据集实验

```bash
# 在不同规模数据集上测试
for dataset in MUTAG PROTEINS NCI1; do
    python run_tudataset.py --dataset $dataset --model gcn --save_results
done
```

## 结果保存位置

训练结果会保存在：
```
result/tudataset/{数据集名称}/{模型名称}/{时间戳}/
├── best_model.pth                  # 最佳模型（如果使用--save_model）
├── experiment_results.json         # 详细结果JSON
├── training_metrics.png            # 训练指标图
├── learning_curve.png              # 学习曲线
└── confusion_matrix.png            # 混淆矩阵
```

## 代码结构说明

### 新增文件

- **`run_tudataset.py`**: 专门用于TUDataset的训练脚本
  - 自动加载TUDataset数据
  - 自动划分训练/验证/测试集
  - 支持多种模型和参数配置

### 修改文件

- **`process.py`**: 添加了注释说明
  - 标注了哪些函数是用于原始神经元数据的
  - 标注了哪些函数是通用的

### 保持不变

- **`model.py`**: 模型定义（4种模型）
- **`train.py`**: 训练和评估函数
- **`run.py`**: 用于原始神经元数据的训练脚本

## 与原始代码的区别

| 功能 | 原始代码 (run.py) | TUDataset代码 (run_tudataset.py) |
|------|-------------------|-----------------------------------|
| 数据来源 | CSV文件（神经元数据） | TUDataset（标准图数据集） |
| 数据加载 | `load_data()` | `TUDataset()` 直接加载 |
| 图构建 | 需要计算相关性矩阵 | 已有图结构 |
| 数据预处理 | SMOTE过采样 | 随机/分层划分 |
| 使用场景 | 神经元钙离子数据 | 标准图分类基准 |

## 常见问题

### Q1: 如何查看数据集信息？

运行训练脚本时会自动打印数据集信息：
```bash
python run_tudataset.py --dataset MUTAG --model gcn
```

或者在Python中：
```python
from torch_geometric.datasets import TUDataset

dataset = TUDataset(root='../../data/TUDataset', name='MUTAG')
print(f"图数量: {len(dataset)}")
print(f"特征维度: {dataset.num_features}")
print(f"类别数: {dataset.num_classes}")
```

### Q2: 数据集保存在哪里？

默认保存在 `/app/ZJ/gitlocal/Brain_Neuroimage_Processing/CVPR/data/TUDataset/`

可以通过 `--data_root` 参数修改。

### Q3: 如何调整超参数？

从小到大调整：
1. **hidden_dim**: 32 → 64 → 128 → 256
2. **dropout**: 0.3 → 0.5 → 0.7
3. **lr**: 0.01 → 0.001 → 0.0001
4. **batch_size**: 根据数据集大小（MUTAG用32，NCI1用128）

### Q4: 训练很慢怎么办？

- 减小 `batch_size`
- 减少 `epochs`
- 使用GPU（自动检测）
- 选择更小的数据集（MUTAG < PROTEINS < NCI1）

### Q5: 原始的神经元数据还能用吗？

可以！使用原来的 `run.py`：
```bash
python run.py --model gcn --dataset dataset/processed3.csv
```

## 推荐实验流程

1. **第一步**: 在MUTAG上快速验证
   ```bash
   python run_tudataset.py --dataset MUTAG --model gcn --epochs 100
   ```

2. **第二步**: 对比不同模型
   ```bash
   for model in gcn gat sage hybrid; do
       python run_tudataset.py --dataset MUTAG --model $model --save_results
   done
   ```

3. **第三步**: 在更大数据集上测试
   ```bash
   python run_tudataset.py --dataset PROTEINS --model gcn --save_results
   python run_tudataset.py --dataset NCI1 --model gcn --save_results
   ```

4. **第四步**: 调优最佳模型
   ```bash
   python run_tudataset.py --dataset NCI1 --model gcn \
       --hidden_dim 128 --dropout 0.5 --batch_size 128 \
       --lr 0.001 --epochs 300 --save_results --save_model
   ```

## 预期结果

### MUTAG (188图, 2类)
- GCN: ~80-85% 准确率
- GAT: ~80-85% 准确率
- 训练时间: < 1分钟 (CPU)

### PROTEINS (1,113图, 2类)
- GCN: ~70-75% 准确率
- GAT: ~72-76% 准确率
- 训练时间: 2-5分钟 (CPU)

### NCI1 (4,110图, 2类)
- GCN: ~75-80% 准确率
- GAT: ~76-81% 准确率
- 训练时间: 5-15分钟 (CPU)

---

**祝实验顺利！** 🚀

如有问题，请查看代码注释或联系开发者。

