# 图神经网络模型指南

## 📚 可用模型总览

当前共有 **8个模型** 可用于实验：4个基础模型 + 4个先进模型

### 基础模型（原有）

| 模型 | 参数 | 特点 | 适用场景 |
|------|------|------|----------|
| **GCN** | `--model gcn` | 经典图卷积网络 | 基准测试、快速验证 |
| **GAT** | `--model gat` | 图注意力网络 | 需要注意力机制的任务 |
| **GraphSAGE** | `--model sage` | 图采样聚合 | 大规模图、归纳学习 |
| **Hybrid** | `--model hybrid` | GCN+SAGE+GAT混合 | 综合性能 |

### 先进模型（2016-2019）

| 模型 | 参数 | 论文年份 | 特点 | 优势 |
|------|------|---------|------|------|
| **GIN** | `--model gin` | ICLR 2019 | 理论最强的图区分能力 | WL测试等价，精确区分图结构 |
| **ChebNet** | `--model chebnet` | NIPS 2016 | Chebyshev多项式近似 | 计算高效，适合大规模图 |
| **EdgeConv** | `--model edgeconv` | TOG 2019 | 动态图卷积+边特征 | 捕捉边信息，多尺度融合 |
| **GraphUNet** | `--model gunet` | ICML 2019 | U-Net架构+池化 | 多尺度特征，分层表示 |

## 🎯 模型详细介绍

### 1. GIN (Graph Isomorphism Network)

**论文**: How Powerful are Graph Neural Networks? (ICLR 2019)  
**作者**: Xu et al.  
**链接**: https://arxiv.org/abs/1810.00826

**核心思想**:
- 理论证明与WL（Weisfeiler-Lehman）图同构测试等价
- 使用MLP作为聚合函数，而不是简单的求和
- 具有最强的图结构区分能力

**技术特点**:
```python
# GIN的聚合方式
h_v = MLP((1 + ε) · h_v + Σ h_u)  # ε是可学习参数
```

**适用场景**:
- ✅ 需要精确区分复杂图结构
- ✅ 图分类任务
- ✅ 理论性能要求高的场景

**使用示例**:
```bash
# 基本使用
python run_unified.py --data_source tudataset --dataset MUTAG --model gin

# 自定义层数
python run_unified.py --data_source tudataset --dataset MUTAG --model gin --hidden_dim 64
```

**预期性能**:
- MUTAG: ~85-90%
- PROTEINS: ~75-80%
- NCI1: ~80-85%

---

### 2. ChebNet (Chebyshev GCN)

**论文**: Convolutional Neural Networks on Graphs with Fast Localized Spectral Filtering (NIPS 2016)  
**作者**: Defferrard et al.  
**链接**: https://arxiv.org/abs/1606.09375

**核心思想**:
- 使用Chebyshev多项式近似图卷积核
- 基于谱图理论
- K阶多项式可以捕获K-hop邻居信息

**技术特点**:
```python
# Chebyshev卷积
h = Σ θ_k T_k(L̃) x  # T_k是k阶Chebyshev多项式
```

**适用场景**:
- ✅ 大规模图（计算效率高）
- ✅ 需要控制感受野大小
- ✅ 谱域方法

**使用示例**:
```bash
# K=3 (默认，捕获3-hop邻居)
python run_unified.py --data_source tudataset --dataset PROTEINS --model chebnet

# 调整多项式阶数需要修改model.py中的K参数
```

**优势**:
- 计算复杂度: O(K|E|)，线性于边数
- 参数效率高
- 理论基础坚实

---

### 3. EdgeConv (Dynamic Graph CNN)

**论文**: Dynamic Graph CNN for Learning on Point Clouds (TOG 2019)  
**作者**: Wang et al.  
**链接**: https://arxiv.org/abs/1801.07829

**核心思想**:
- 边卷积：同时考虑节点和其邻居的差异
- 动态更新图结构
- 多尺度特征融合

**技术特点**:
```python
# 边卷积
h_i = max_j MLP(h_i || h_j - h_i)  # ||表示拼接
```

**适用场景**:
- ✅ 需要捕捉边特征
- ✅ 点云数据
- ✅ 动态图结构

**使用示例**:
```bash
python run_unified.py --data_source tudataset --dataset NCI1 --model edgeconv
```

**特色**:
- 3层EdgeConv，每层提取不同尺度特征
- 最后融合所有尺度的特征
- 深层MLP分类器

---

### 4. GraphUNet

**论文**: Graph U-Nets (ICML 2019)  
**作者**: Gao et al.  
**链接**: https://arxiv.org/abs/1905.05178

**核心思想**:
- 将U-Net架构应用于图
- 使用TopK池化进行下采样
- 编码器-解码器结构

**技术特点**:
```python
# TopK池化
score = σ(Wx)  # 计算节点重要性
idx = top_k(score)  # 选择最重要的k个节点
```

**适用场景**:
- ✅ 需要多尺度特征
- ✅ 分层图表示
- ✅ 节点/图分类

**使用示例**:
```bash
python run_unified.py --data_source tudataset --dataset ENZYMES --model gunet
```

**特色**:
- 2层池化，逐步压缩图
- 保留重要节点
- 结合mean和max池化

---

## 🔬 模型对比实验

### 实验1: 在MUTAG上对比所有模型

```bash
# 对比所有模型
for model in gcn gat sage hybrid gin chebnet edgeconv gunet; do
    python run_unified.py \
        --data_source tudataset \
        --dataset MUTAG \
        --model $model \
        --epochs 200 \
        --save_results
done
```

### 实验2: 测试GIN在不同数据集上的性能

```bash
# GIN跨数据集实验
for dataset in MUTAG PROTEINS NCI1 ENZYMES; do
    python run_unified.py \
        --data_source tudataset \
        --dataset $dataset \
        --model gin \
        --epochs 200 \
        --save_results
done
```

### 实验3: 超参数对比

```bash
# 测试不同隐藏层维度
for hidden_dim in 32 64 128; do
    python run_unified.py \
        --data_source tudataset \
        --dataset MUTAG \
        --model gin \
        --hidden_dim $hidden_dim \
        --save_results
done
```

## 📊 性能参考

基于文献和经验的预期性能（准确率%）：

| 数据集 | GCN | GAT | SAGE | GIN | ChebNet | EdgeConv | GraphUNet |
|--------|-----|-----|------|-----|---------|----------|-----------|
| MUTAG | 80-85 | 80-85 | 75-80 | **85-90** | 80-85 | 82-87 | 80-85 |
| PROTEINS | 70-75 | 72-76 | 70-74 | **75-80** | 70-75 | 73-77 | 71-76 |
| NCI1 | 75-80 | 76-81 | 74-79 | **80-85** | 75-80 | 77-82 | 76-81 |
| ENZYMES | 55-60 | 57-62 | 55-60 | **60-65** | 56-61 | 58-63 | 57-62 |

**注意**: 实际性能会受到随机种子、超参数等影响。

## 🎯 选择模型的建议

### 按任务类型选择

**1. 标准基准测试（论文对比）**
- 首选: **GIN** - 理论最强，性能最好
- 备选: GCN, GAT (经典基准)

**2. 快速原型开发**
- 首选: **GCN** - 简单快速
- 备选: ChebNet (如果数据量大)

**3. 需要解释性**
- 首选: **GAT** - 注意力权重可视化
- 备选: EdgeConv (边特征重要性)

**4. 大规模图**
- 首选: **ChebNet** - 计算效率高
- 备选: GraphSAGE

**5. 多尺度特征**
- 首选: **GraphUNet** - U-Net架构
- 备选: EdgeConv (多尺度融合)

### 按数据集大小选择

**小数据集 (< 1000图)**
- GIN, GAT, EdgeConv - 表达能力强

**中等数据集 (1000-10000图)**
- GCN, GIN, ChebNet - 平衡性能和效率

**大数据集 (> 10000图)**
- ChebNet, GraphSAGE - 计算效率优先

## 💡 使用技巧

### 1. 超参数调优顺序

```bash
# 步骤1: 选择最佳模型（固定其他参数）
for model in gin edgeconv gunet; do
    python run_unified.py --data_source tudataset --dataset MUTAG --model $model
done

# 步骤2: 调整隐藏层维度
for hid in 32 64 128; do
    python run_unified.py --data_source tudataset --dataset MUTAG --model gin --hidden_dim $hid
done

# 步骤3: 调整dropout
for drop in 0.3 0.5 0.7; do
    python run_unified.py --data_source tudataset --dataset MUTAG --model gin --dropout $drop
done
```

### 2. 避免过拟合

小数据集上（如MUTAG）:
- ✅ 增大dropout (0.5-0.7)
- ✅ 减少层数
- ✅ 使用Early stopping
- ✅ 数据增强

### 3. 提升性能

- ✅ 尝试集成学习（多模型投票）
- ✅ 调整池化策略
- ✅ 增加数据预处理
- ✅ 使用更复杂的分类器

## 📖 相关论文列表

### 基础GNN论文
1. **GCN**: Semi-Supervised Classification with Graph Convolutional Networks (ICLR 2017)
2. **GraphSAGE**: Inductive Representation Learning on Large Graphs (NeurIPS 2017)
3. **GAT**: Graph Attention Networks (ICLR 2018)

### 新增模型论文
4. **ChebNet**: Convolutional Neural Networks on Graphs with Fast Localized Spectral Filtering (NIPS 2016)
5. **GIN**: How Powerful are Graph Neural Networks? (ICLR 2019)
6. **EdgeConv**: Dynamic Graph CNN for Learning on Point Clouds (TOG 2019)
7. **GraphUNet**: Graph U-Nets (ICML 2019)

### 综述论文
- A Comprehensive Survey on Graph Neural Networks (IEEE TNNLS 2020)
- Graph Neural Networks: A Review of Methods and Applications (AI Open 2020)

## 🔧 故障排除

### 问题1: 模型训练不收敛

**可能原因**:
- 学习率过大
- 模型过于复杂

**解决方案**:
```bash
# 降低学习率
python run_unified.py --data_source tudataset --dataset MUTAG --model gin --lr 0.0001

# 减小模型规模
python run_unified.py --data_source tudataset --dataset MUTAG --model gin --hidden_dim 32
```

### 问题2: 内存不足

**解决方案**:
```bash
# 减小batch size
python run_unified.py --data_source tudataset --dataset MUTAG --model gin --batch_size 16

# 使用更简单的模型
python run_unified.py --data_source tudataset --dataset MUTAG --model gcn
```

### 问题3: 性能不如预期

**检查清单**:
- [ ] 数据预处理正确吗？
- [ ] 超参数合适吗？
- [ ] 运行足够多的epochs吗？
- [ ] 尝试了多个随机种子吗？

---

**祝实验顺利！** 🚀

如有问题，请查看模型代码注释或相关论文。

