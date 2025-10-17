# GCN图分类模型库

## 🚀 快速开始

```bash
# 在MUTAG数据集上训练GIN模型（推荐）
python run_unified.py --data_source tudataset --dataset MUTAG --model gin

# 查看帮助
python run_unified.py --help
```

## 📊 可用模型（8个）

### 基础模型
1. **GCN** (`--model gcn`) - 经典图卷积网络
2. **GAT** (`--model gat`) - 图注意力网络
3. **GraphSAGE** (`--model sage`) - 图采样聚合
4. **Hybrid** (`--model hybrid`) - GCN+SAGE+GAT混合

### 先进模型（2016-2019）
5. **GIN** (`--model gin`) - 🏆 理论最强，ICLR 2019
6. **ChebNet** (`--model chebnet`) - Chebyshev卷积，NIPS 2016
7. **EdgeConv** (`--model edgeconv`) - 动态图CNN，TOG 2019
8. **GraphUNet** (`--model gunet`) - U-Net架构，ICML 2019

## 💻 使用方法

### TUDataset标准数据集
```bash
# 基本训练
python run_unified.py --data_source tudataset --dataset MUTAG --model gin

# 完整参数
python run_unified.py \
    --data_source tudataset \
    --dataset MUTAG \
    --model gin \
    --hidden_dim 64 \
    --dropout 0.5 \
    --epochs 200 \
    --save_results
```

### CSV数据（神经元数据）
```bash
python run_unified.py \
    --data_source csv \
    --dataset ../../dataset/processed3.csv \
    --model gcn \
    --save_results
```

## 🎯 选择模型

- **最佳性能**: `gin` (理论最强，准确率最高)
- **快速原型**: `gcn` (简单快速)
- **大规模图**: `chebnet` (计算高效)
- **可解释性**: `gat` (注意力权重)

## 📖 详细文档

查看 `MODEL_GUIDE.md` 了解每个模型的详细信息、论文引用和使用建议。

## 🔬 实验示例

```bash
# 对比所有模型
for model in gcn gat sage hybrid gin chebnet edgeconv gunet; do
    python run_unified.py --data_source tudataset --dataset MUTAG --model $model --save_results
done

# 多数据集实验
for dataset in MUTAG PROTEINS NCI1; do
    python run_unified.py --data_source tudataset --dataset $dataset --model gin --save_results
done
```

## 📈 预期性能（MUTAG数据集）

| 模型 | 准确率 | 参数量 | 速度 |
|------|--------|--------|------|
| **GIN** | **~85-90%** | 30K | 中 |
| GCN | ~80-85% | 18K | 快 |
| GAT | ~80-85% | 18K | 中 |
| ChebNet | ~80-85% | 35K | 快 |

## 🛠️ 文件说明

- `model.py` - 8个模型实现
- `run_unified.py` - 统一训练脚本（支持CSV和TUDataset）
- `train.py` - 训练和评估函数
- `process.py` - 数据处理函数
- `MODEL_GUIDE.md` - 详细模型指南

## 📚 论文引用

详见 `MODEL_GUIDE.md`
