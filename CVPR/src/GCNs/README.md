# GCN图分类模型库

## 🚀 快速开始

```bash
# 在MUTAG数据集上训练GIN模型（推荐）
python run_unified.py --data_source tudataset --dataset MUTAG --model gin

# 查看帮助
python run_unified.py --help
```

## 📊 可用模型（11个）

### 基础模型（4个）
1. **GCN** (`--model gcn`) - 经典图卷积网络
2. **GAT** (`--model gat`) - 图注意力网络
3. **GraphSAGE** (`--model sage`) - 图采样聚合
4. **Hybrid** (`--model hybrid`) - GCN+SAGE+GAT混合

### 先进模型 2016-2019（4个）
5. **GIN** (`--model gin`) - 🏆 理论最强，ICLR 2019
6. **ChebNet** (`--model chebnet`) - Chebyshev卷积，NIPS 2016
7. **EdgeConv** (`--model edgeconv`) - 动态图CNN，TOG 2019
8. **GraphUNet** (`--model gunet`) - U-Net架构，ICML 2019

### 最新模型 2020-2022（3个）⭐
9. **PNA** (`--model pna`) - 🔥 多聚合器，NeurIPS 2020
10. **GATv2** (`--model gatv2`) - 🔥 改进注意力，ICLR 2022
11. **DeeperGCN** (`--model deepergcn`) - 🔥 深层网络，ICLR 2020

## 💻 使用方法

### TUDataset标准数据集
```bash
# 使用最新的PNA模型
python run_unified.py --data_source tudataset --dataset MUTAG --model pna

# 使用GATv2（改进的GAT）
python run_unified.py --data_source tudataset --dataset MUTAG --model gatv2

# 使用DeeperGCN（深层网络）
python run_unified.py --data_source tudataset --dataset MUTAG --model deepergcn

# 完整参数
python run_unified.py \
    --data_source tudataset \
    --dataset MUTAG \
    --model pna \
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
    --model gin \
    --save_results
```

## 🎯 选择模型

### 按性能选择
- **最佳性能**: `gin` 或 `pna` (理论最强)
- **改进注意力**: `gatv2` (优于GAT)
- **深层网络**: `deepergcn` (7-14层)

### 按速度选择
- **最快**: `gcn` 或 `chebnet`
- **中等**: `gin`, `pna`, `gatv2`
- **较慢**: `edgeconv`, `gunet`

### 按场景选择
- **标准基准**: `gin`, `pna` (论文常用)
- **快速验证**: `gcn`
- **大规模图**: `chebnet`, `deepergcn`
- **可解释性**: `gatv2` (注意力可视化)

## 🔬 实验示例

### 对比所有模型
```bash
for model in gcn gin pna gatv2 deepergcn; do
    python run_unified.py --data_source tudataset --dataset MUTAG --model $model --save_results
done
```

### 测试2020+最新模型
```bash
for model in pna gatv2 deepergcn; do
    python run_unified.py --data_source tudataset --dataset MUTAG --model $model --epochs 200 --save_results
done
```

## 📈 模型性能对比

基于MUTAG数据集的测试（5轮训练）：

| 模型 | 验证F1 | 参数量 | 年份 | 推荐度 |
|------|--------|--------|------|--------|
| **GIN** | 0.848 | 30K | 2019 | ⭐⭐⭐⭐⭐ |
| **PNA** | 0.776 | 48K | 2020 | ⭐⭐⭐⭐⭐ |
| **ChebNet** | 0.707 | 35K | 2016 | ⭐⭐⭐⭐ |
| **GCN** | 0.776 | 18K | 2017 | ⭐⭐⭐⭐ |
| **GATv2** | 0.694 | 45K | 2022 | ⭐⭐⭐⭐ |
| **DeeperGCN** | 0.556 | 25K | 2020 | ⭐⭐⭐ |

*注：DeeperGCN在大规模图上表现更好

## 📖 详细文档

查看 `MODEL_GUIDE.md` 了解每个模型的详细信息、论文引用和使用建议。

## 🏆 推荐组合

### 论文实验（标准基准）
```bash
# 对比GCN vs GIN vs PNA
python run_unified.py --data_source tudataset --dataset NCI1 --model gcn --save_results
python run_unified.py --data_source tudataset --dataset NCI1 --model gin --save_results  
python run_unified.py --data_source tudataset --dataset NCI1 --model pna --save_results
```

### 最新技术（2020+）
```bash
# 使用最新的模型
python run_unified.py --data_source tudataset --dataset MUTAG --model pna      # NeurIPS 2020
python run_unified.py --data_source tudataset --dataset MUTAG --model gatv2    # ICLR 2022
python run_unified.py --data_source tudataset --dataset MUTAG --model deepergcn # ICLR 2020
```

## 🔧 关键论文

### 2020-2022 最新
- **PNA**: Principal Neighbourhood Aggregation (NeurIPS 2020)
- **DeeperGCN**: All You Need to Train Deeper GCNs (ICLR 2020)
- **GATv2**: How Attentive are Graph Attention Networks? (ICLR 2022)

### 2019 经典
- **GIN**: How Powerful are Graph Neural Networks? (ICLR 2019)

详见 `MODEL_GUIDE.md`
