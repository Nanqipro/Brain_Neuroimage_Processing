# 实验运行快速指南

## 环境准备

1. 确保已安装必要的包：
```bash
pip install torch torch-geometric pandas numpy scikit-learn matplotlib seaborn
```

2. 确保在 GCNs 目录下运行命令：
```bash
cd /app/ZJ/gitlocal/Brain_Neuroimage_Processing/ICLR/src/GCNs
```

## 实验步骤（按照README要求）

### 步骤1: 验证数据适配 ✅
代码已适配 `../../data/graphs/` 下的图文件格式：
- 支持 window_20, window_50, window_100
- 自动从文件名提取标签（Sleep/Wake）
- 转换为PyTorch Geometric格式

### 步骤2: 快速测试（推荐先执行）
```bash
# 测试单个模型，5次实验
python run.py --model gcn --runs 5 --window_size 20

# 快速批量测试所有模型
python batch_experiments.py --quick_test --window_sizes 20
```

### 步骤3: 运行50-100组完整实验

#### 选项A: 使用批量脚本（推荐）
```bash
# 运行默认4个模型，每个50组实验
python batch_experiments.py --runs 50

# 运行所有8个模型，每个50组实验
python batch_experiments.py --runs 50 --models gcn gat sage hybrid gin transformer chebnet ensemble

# 只测试特定窗口大小
python batch_experiments.py --runs 100 --window_sizes 50
```

#### 选项B: 分别运行每个模型
```bash
# 基础模型
python run.py --model gcn --runs 100
python run.py --model gat --runs 100
python run.py --model sage --runs 100

# SOTA模型
python run.py --model gin --runs 100
python run.py --model transformer --runs 100
python run.py --model chebnet --runs 100

# 集成模型（可能较慢）
python run.py --model ensemble --runs 50
```

### 步骤4: 查看结果

实验结果保存在以下位置：
- 单个模型结果: `result/{model_name}/summary_*.json`
- 批量实验结果: `batch_results/`
  - `batch_results.json`: 详细结果
  - `batch_results.csv`: 表格格式
  - `experiment_report.json`: 汇总报告

### 步骤5: 收集的指标 ✅
每个实验自动记录：
- **accuracy**: 准确率
- **recall**: 召回率
- **f1-score**: F1分数
- **precision**: 精确率
- **training_time**: 训练时间
- **predicting_time**: 预测时间
- **run_time**: 总运行时间

## 常见问题

### 1. 显存不足（OOM）
```bash
# 减小隐藏层维度
python run.py --model gat --hidden_dim 32

# 或修改 run.py 中的 batch_size (第129行)
```

### 2. 运行时间过长
```bash
# 只测试一个窗口大小
python batch_experiments.py --window_sizes 20

# 减少实验次数
python batch_experiments.py --runs 30
```

### 3. 查看实时进度
批量实验会实时显示：
- 当前进度: X/Y
- 每个模型的平均指标
- 预计剩余时间

## 数据分析建议

完成实验后，可以：
1. 使用 `batch_results.csv` 进行数据分析
2. 比较不同模型在不同窗口大小下的表现
3. 分析训练时间与性能的权衡
4. 检查模型稳定性（通过std）

## 实验计划示例

完整实验计划（约需8-12小时）：
```bash
# 1. 快速验证（10分钟）
python batch_experiments.py --quick_test

# 2. 主要模型50组实验（3-4小时）
python batch_experiments.py --runs 50 --models gcn gat sage gin

# 3. 高级模型测试（4-6小时）
python batch_experiments.py --runs 30 --models transformer chebnet ensemble

# 4. 特定配置深入测试（2-3小时）
python run.py --model gin --runs 100 --window_size 50 --hidden_dim 128
```

祝实验顺利！如有问题，请查看错误日志或调整参数。
