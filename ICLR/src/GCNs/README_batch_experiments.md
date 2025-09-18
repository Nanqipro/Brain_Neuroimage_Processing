# 批量实验脚本使用说明

## 概述

本批量实验脚本用于自动化进行多组GCN实验，符合README中的要求：
- 进行50-100组实验
- 收集各项指标：accuracy, recall, f1-score, training time, predicting time, run time
- 支持多种模型和窗口大小的组合测试

## 文件说明

- `batch_experiments.py` - 主要的批量实验脚本
- `run_batch.sh` - Linux/Mac运行脚本
- `run_batch.bat` - Windows运行脚本
- `experiment_config.json` - 实验配置文件
- `README_batch_experiments.md` - 本说明文档

## 快速开始

### Windows用户
```bash
# 在GCNs目录下运行
run_batch.bat
```

### Linux/Mac用户
```bash
# 在GCNs目录下运行
chmod +x run_batch.sh
./run_batch.sh
```

### 直接使用Python
```bash
# 标准实验（50次运行）
python batch_experiments.py

# 快速测试（5次运行）
python batch_experiments.py --quick_test

# 自定义参数
python batch_experiments.py --runs 100 --models gcn gat --window_sizes 20 50
```

## 命令行参数

- `--runs N` - 每个实验的运行次数（默认：50）
- `--models MODEL1 MODEL2` - 要测试的模型（可选：gcn, gat, hybrid）
- `--window_sizes SIZE1 SIZE2` - 要测试的窗口大小（可选：20, 50, 100）
- `--output_dir DIR` - 结果输出目录（默认：batch_results）
- `--quick_test` - 快速测试模式（每个实验只运行5次）

## 实验配置

### 默认实验设置
- **模型**: GCN, GAT, Hybrid
- **窗口大小**: 20, 50, 100
- **运行次数**: 50次/实验
- **总实验数**: 3模型 × 3窗口 = 9个实验组合

### 预设实验场景
1. **快速测试** (`--quick_test`)
   - 1个模型 × 1个窗口 × 5次运行
   - 用于验证脚本功能

2. **标准实验** (默认)
   - 3个模型 × 3个窗口 × 50次运行
   - 符合README要求

3. **全面实验**
   - 3个模型 × 3个窗口 × 100次运行
   - 更高精度的结果

## 输出结果

### 文件结构
```
batch_results/
├── batch_results.json          # 详细实验结果（JSON格式）
├── batch_results.csv           # 实验结果表格（CSV格式）
└── experiment_report.json      # 实验报告摘要
```

### 收集的指标
- **accuracy_mean/std** - 准确率均值和标准差
- **precision_mean/std** - 精确率均值和标准差
- **recall_mean/std** - 召回率均值和标准差
- **f1_mean/std** - F1分数均值和标准差
- **training_time_mean** - 平均训练时间
- **total_run_time** - 总运行时间

## 使用示例

### 1. 标准实验
```bash
# 运行所有模型和窗口大小的组合，每个50次
python batch_experiments.py
```

### 2. 只测试特定模型
```bash
# 只测试GCN和GAT模型
python batch_experiments.py --models gcn gat
```

### 3. 只测试特定窗口大小
```bash
# 只测试窗口大小20和50
python batch_experiments.py --window_sizes 20 50
```

### 4. 高精度实验
```bash
# 每个实验运行100次
python batch_experiments.py --runs 100
```

### 5. 自定义输出目录
```bash
# 结果保存到custom_results目录
python batch_experiments.py --output_dir custom_results
```

## 实验时间估算

基于单次实验的平均时间：
- **快速测试**: ~5分钟
- **标准实验**: ~4-6小时
- **全面实验**: ~8-12小时

实际时间取决于：
- 硬件配置（CPU/GPU）
- 数据集大小
- 模型复杂度

## 注意事项

1. **磁盘空间**: 确保有足够的磁盘空间存储结果
2. **内存使用**: 大窗口大小可能需要更多内存
3. **中断恢复**: 脚本支持Ctrl+C中断，会保存当前结果
4. **并行运行**: 目前不支持并行，建议单独运行
5. **环境要求**: 需要安装所有依赖包（torch, torch-geometric等）

## 结果分析

实验完成后，可以使用以下方式分析结果：

### 1. 查看CSV结果
```python
import pandas as pd
df = pd.read_csv('batch_results/batch_results.csv')
print(df.groupby('model')[['accuracy_mean', 'f1_mean']].mean())
```

### 2. 查看JSON报告
```python
import json
with open('batch_results/experiment_report.json', 'r') as f:
    report = json.load(f)
print(json.dumps(report['performance_summary'], indent=2))
```

## 故障排除

### 常见问题
1. **ImportError**: 检查是否安装了所有依赖包
2. **FileNotFoundError**: 确保在GCNs目录下运行脚本
3. **CUDA错误**: 检查GPU驱动和CUDA版本
4. **内存不足**: 减少batch_size或使用更小的窗口大小

### 调试模式
```bash
# 使用快速测试模式进行调试
python batch_experiments.py --quick_test
```

## 联系支持

如有问题，请检查：
1. 依赖包是否正确安装
2. 数据路径是否正确
3. 运行目录是否正确
4. 系统资源是否充足