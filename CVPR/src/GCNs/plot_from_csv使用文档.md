# plot_from_csv.py 使用文档

## 📋 功能概述

`plot_from_csv.py` 是一个用于训练结果可视化的Python脚本，支持两种主要功能：

1. **训练历史可视化**：从CSV文件读取训练历史数据并生成多种可视化图表
2. **数据集规模扩展性分析**：分析不同数据集规模和模型的性能表现

---

## 📦 依赖要求

### 必需的Python包

```bash
pip install pandas matplotlib numpy
```

或者使用requirements.txt：

```bash
pip install -r requirements.txt
```

### Python版本
- Python 3.6+

---

## 🚀 使用方法

### 方法一：训练历史可视化（CSV模式）

从单个或多个CSV文件生成训练指标的可视化图表。

#### 基本语法

```bash
python plot_from_csv.py --csv <CSV文件路径> [选项]
```

#### 参数说明

| 参数 | 类型 | 必需 | 说明 |
|------|------|------|------|
| `--csv` | str+ | 是* | CSV文件路径，可指定多个文件进行对比 |
| `--labels` | str+ | 否 | 图例标签，与CSV文件一一对应 |
| `--output_dir` | str | 否 | 输出目录（默认：`plots`） |
| `--no_stats` | flag | 否 | 不打印统计信息 |

*注：在CSV模式下必需，在scaling模式下不需要

#### 使用示例

**示例1：单个CSV文件可视化**
```bash
python plot_from_csv.py --csv result/tudataset/MUTAG/gcn/20251103_190727/training_history.csv
```

**示例2：对比多个模型**
```bash
python plot_from_csv.py \
  --csv gcn_training.csv gat_training.csv gin_training.csv \
  --labels GCN GAT GIN
```

**示例3：指定输出目录**
```bash
python plot_from_csv.py \
  --csv training_history.csv \
  --output_dir my_custom_plots
```

**示例4：不显示统计信息**
```bash
python plot_from_csv.py \
  --csv training_history.csv \
  --no_stats
```

#### 输出内容（CSV模式）

在指定的输出目录下会生成以下图表：

1. **cumulative_time.png** - 累积训练时间曲线
2. **epoch_time.png** - 每个epoch的执行时间
3. **training_metrics.png** - 训练和验证指标（2x2子图）
   - Loss & F1 Score
   - F1 Score (训练vs验证)
   - Accuracy (训练vs验证)
   - Precision (训练vs验证)

#### CSV文件格式要求

CSV文件应包含以下列：

```
epoch,train_loss,train_f1,train_accuracy,train_precision,val_f1,val_accuracy,val_precision,epoch_time,cumulative_time
```

示例：
```csv
epoch,train_loss,train_f1,train_accuracy,train_precision,val_f1,val_accuracy,val_precision,epoch_time,cumulative_time
1,0.6931,0.5234,0.7123,0.6234,0.6123,0.6834,0.6543,2.345,2.345
2,0.5234,0.6345,0.7834,0.7123,0.6834,0.7234,0.7012,2.123,4.468
...
```

---

### 方法二：数据集规模扩展性分析（Scaling模式）

分析不同数据集规模下各个模型的训练时间和性能表现。

#### 基本语法

```bash
python plot_from_csv.py --scaling [选项]
```

#### 参数说明

| 参数 | 类型 | 必需 | 说明 |
|------|------|------|------|
| `--scaling` | flag | 是 | 启用扩展性分析模式 |
| `--result_dir` | str | 否 | 结果目录（默认：`result`） |
| `--output_dir` | str | 否 | 输出目录（默认：`scaling_plots`） |
| `--models` | str+ | 否 | 指定要分析的模型列表，不指定则分析所有模型 |

#### 使用示例

**示例1：分析所有模型**
```bash
python plot_from_csv.py \
  --scaling \
  --result_dir result \
  --output_dir scaling_plots
```

**示例2：只分析指定模型**
```bash
python plot_from_csv.py \
  --scaling \
  --models gcn gat gin \
  --output_dir gcn_gat_gin_analysis
```

**示例3：只对比GCN和GIN**
```bash
python plot_from_csv.py \
  --scaling \
  --models gcn gin \
  --output_dir gcn_vs_gin
```

#### 输出内容（Scaling模式）

在指定的输出目录下会生成以下图表：

1. **total_training_time_vs_scale.png** - 数据集规模 vs 总训练时间
2. **avg_training_time_vs_scale.png** - 数据集规模 vs 平均每epoch训练时间
3. **time_per_graph_vs_scale.png** - 数据集规模 vs 单个图处理时间（如果有数据）
4. **time_per_graph_per_epoch_vs_scale.png** - 数据集规模 vs 每个epoch中单个图处理时间（如果有数据）
5. **peak_cpu_memory_vs_scale.png** - 数据集规模 vs 峰值CPU内存占用（MB）⭐ 新增
6. **peak_gpu_memory_vs_scale.png** - 数据集规模 vs 峰值GPU内存占用（MB）⭐ 新增

#### 目录结构要求

扩展性分析需要以下目录结构：

```
result/
├── custom/
│   ├── graphs_100/
│   │   ├── gcn/
│   │   │   └── experiment_results.json
│   │   ├── gat/
│   │   │   └── experiment_results.json
│   │   └── gin/
│   │       └── experiment_results.json
│   ├── graphs_1000/
│   │   ├── gcn/
│   │   │   └── experiment_results.json
│   │   └── ...
│   └── graphs_10000/
│       └── ...
```

#### experiment_results.json 格式要求

```json
{
  "time_statistics": {
    "total_training_time": 123.45,
    "avg_training_time_per_epoch": 6.17,
    "time_per_graph": 0.01234,
    "time_per_graph_ms": 12.34,
    "time_per_graph_per_epoch": 0.00617,
    "time_per_graph_per_epoch_ms": 6.17
  },
  "resource_usage": {
    "cpu_memory": {
      "average_mb": 1267.91,
      "peak_mb": 1267.91,
      "min_mb": 1267.91
    },
    "gpu_memory": {
      "average_allocated_mb": 16.70,
      "peak_allocated_mb": 140.21,
      "average_reserved_mb": 164.0,
      "peak_reserved_mb": 164.0
    }
  }
}
```

**注意**：`resource_usage` 部分为新增字段，用于记录内存占用情况。如果JSON文件中没有这些字段，脚本会自动跳过内存图表的绘制。

---

## 🆕 v2.1 更新：优化可视化与分析功能

### 主要改进

从 v2.1 开始，`plot_from_csv.py` 进行了重大优化：

1. ✅ **横坐标改为边数量**：更准确地反映图数据的复杂度
2. ✅ **曲线拟合**：使用幂律拟合生成平滑曲线，弱化折线影响
3. ✅ **OOM标注**：自动检测并标注内存溢出(Out-of-Memory)点
4. ✅ **图表优化**：缩小图宽度至10x6，提升可读性
5. ✅ **CPU/GPU内存分析**：绘制不同数据集规模下的峰值内存占用曲线
6. ✅ **统一超参数**：run_all_experiments_multi.sh确保所有实验使用batch_size=32

### 数据来源

内存数据从 `run.py` 训练脚本自动记录，保存在 `experiment_results.json` 文件的 `resource_usage` 字段中：
- **CPU内存**：使用 `psutil` 库记录进程内存占用
- **GPU内存**：使用 `torch.cuda.memory_allocated()` 和 `torch.cuda.max_memory_allocated()` 记录

### 使用方式

使用 scaling 模式时自动生成，无需额外参数：

```bash
python plot_from_csv.py --scaling --result_dir result --output_dir scaling_plots
```

### 详细功能说明

#### 1. 横坐标改为边数量

**原因**：边数量比图数量更能反映数据复杂度和计算负载

**实现**：
- 从 `experiment_results.json` 的 `dataset_info.num_edges` 读取边数量
- 如果数据不可用，估算为图数量×10

**示例**：
```
横坐标：10^3 ← 10^4 ← 10^5 ← 10^6 (边数量)
而不是：100 ← 1000 ← 10000 ← 100000 (图数量)
```

#### 2. 曲线拟合

**方法**：幂律拟合（Power Law: y = ax^b）
- 在对数空间进行线性拟合
- 生成100个点的平滑曲线
- 原始数据点以半透明误差条显示

**优点**：
- 清晰展示增长趋势
- 弱化数据点之间的折线
- 更适合表示非线性增长

#### 3. OOM标注

**检测方法**：
- GPU内存超过30GB视为可能OOM
- 从日志中检测OOM错误信息

**标注样式**：
- 红色X标记（200点大小）
- 深红色边框（2像素）
- 图例中标注"OOM"

**示例**：
```python
if peak_gpu_mem > 30000:  # MB
    # 标注为OOM点
```

#### 4. 图表优化

**尺寸变化**：
- 原尺寸：12×7英寸
- 新尺寸：10×6英寸

**其他优化**：
- 网格透明度：0.3 → 0.25
- 误差条透明度：0.8 → 0.6
- 曲线透明度：0.8（保持）
- 标记大小：10 → 8
- 字体大小：14/16 → 13/15

#### 5. 输出图表特性

所有6个图表均采用统一样式：
- **横坐标**：边数量（对数刻度）
- **纵坐标**：对应指标（对数刻度）
- **误差条**：半透明显示标准差
- **平滑曲线**：幂律拟合曲线
- **OOM标注**：红色X标记
- **分辨率**：300 DPI

### 统计信息

在终端统计输出中，会显示每个数据集规模下的内存占用信息：

```
【GCN】
Dataset Size    Total Time (s)            Avg Time (s/epoch)        Time/Graph (ms)      Peak CPU (MB)        Peak GPU (MB)       
--------------------------------------------------------------------------------------------------------------------------------------------
100                73.05 ± 0.56           0.2435 ± 0.0019        121.755 ± 0.941        1275.7 ± 41.1           28.7 ± 0.0    
1000               73.74 ± 0.69           0.2458 ± 0.0023        122.899 ± 1.146        1254.7 ± 14.8           43.0 ± 0.0    
10000             101.83 ± 6.80           0.3394 ± 0.0227        169.724 ± 11.335       2057.2 ± 188.8         761.3 ± 0.1    
```

### 兼容性

- 如果 `experiment_results.json` 中不包含 `resource_usage` 字段，脚本会自动跳过内存图表的绘制
- 如果没有边数量信息，会估算为图数量×10
- 适用于所有支持的模型和数据集规模

---

## 🖥️ 硬件信息记录（run_all_experiments_multi.sh）

### 自动记录硬件配置

运行实验脚本时，会自动记录完整的硬件信息到 `multi_run_logs_YYYYMMDD_HHMMSS/hardware_info.log`

### 记录内容

#### CPU信息
- 型号名称（Model name）
- CPU数量和核心数
- 线程数
- 主频（MHz）

#### 内存信息
- 总内存大小
- 可用内存
- 使用率

#### GPU信息
- GPU索引和型号
- 显存容量
- 驱动版本
- 计算能力（Compute Capability）
- 完整的nvidia-smi输出

### 使用示例

查看硬件配置：
```bash
cat multi_run_logs_20251105_HHMMSS/hardware_info.log
```

示例输出：
```
=====================================================================
🖥️  硬件配置信息
=====================================================================
记录时间: 2025-11-05 15:59:29

【CPU信息】
Model name:            Intel(R) Xeon(R) CPU E5-2698 v4 @ 2.20GHz
CPU(s):                80
Thread(s) per core:    2
Core(s) per socket:    20
Socket(s):             2
CPU MHz:               2200.000

【内存信息】
              total        used        free      shared  buff/cache   available
Mem:          251Gi        45Gi       180Gi       1.2Gi        25Gi       203Gi

【GPU信息】
index, name, memory.total [MiB], driver_version, compute_cap
0, NVIDIA A100-SXM4-80GB, 81920 MiB, 535.104.05, 8.0
1, NVIDIA A100-SXM4-80GB, 81920 MiB, 535.104.05, 8.0
2, NVIDIA A100-SXM4-80GB, 81920 MiB, 535.104.05, 8.0
3, NVIDIA A100-SXM4-80GB, 81920 MiB, 535.104.05, 8.0
```

### 超参数统一性保证

`run_all_experiments_multi.sh` v2.1 确保：
- ✅ 所有数据集使用 **batch_size=32**
- ✅ 所有实验使用 **epochs=300**
- ✅ 所有实验使用 **lr=0.001**
- ✅ 训练/验证/测试比例统一：**60%/20%/20%**

**修改记录**：
- graphs_10000: batch_size 64 → 32
- graphs_100000: batch_size 128 → 32
- graphs_1000000: batch_size 128 → 32

---

## 📊 统计信息输出

默认情况下，脚本会在终端输出统计信息，包括：

- 总Epoch数
- 累积总训练时间
- 平均Epoch时间（均值 ± 标准差）
- 最快/最慢Epoch时间
- 最终训练F1分数
- 最佳验证F1分数及对应的Epoch
- 最终验证F1分数
- **峰值CPU内存占用**（新增）
- **峰值GPU内存占用**（新增）

示例输出：
```
================================================================================
📊 训练历史统计
================================================================================

【GCN】
  文件: gcn_training.csv
  总Epoch数: 100
  累积总时间: 234.567秒
  平均Epoch时间: 2.346 ± 0.123秒
  最快Epoch: 2.123秒
  最慢Epoch: 2.678秒
  最终训练F1: 0.8934
  最佳验证F1: 0.8567 (Epoch 87)
  最终验证F1: 0.8512
================================================================================
```

---

## 🎨 图表特性

### 通用特性
- **高分辨率输出**：所有图表以300 DPI保存
- **误差条**：显示标准差（多次运行的情况下）
- **对数坐标**：扩展性分析图使用对数坐标以更好地展示不同规模的数据
- **网格线**：所有图表包含网格线以便读数
- **图例**：清晰的图例标注不同模型或运行

### 颜色编码
- 使用matplotlib的tab10调色板
- 支持最多10个不同模型的对比

---

## ⚠️ 常见问题

### Q1: 提示"❌ 错误: 文件不存在"
**解决方法**：检查CSV文件路径是否正确，使用绝对路径或从正确的工作目录运行脚本。

### Q2: 图表中某些指标缺失
**解决方法**：检查CSV文件是否包含所有必需的列。

### Q3: Scaling模式未找到数据
**解决方法**：
- 确认`result_dir`路径正确
- 确认目录结构符合要求
- 确认存在`experiment_results.json`文件

### Q4: 标签与文件数量不匹配
**解决方法**：确保`--labels`参数的数量与`--csv`参数的文件数量一致。

### Q5: 内存不足
**解决方法**：如果处理大量数据，可以分批处理或增加系统内存。

---

## 💡 高级技巧

### 技巧1：批量处理多个实验

```bash
# 创建一个shell脚本
#!/bin/bash

models=("gcn" "gat" "gin" "graphsage")
for model in "${models[@]}"; do
  python plot_from_csv.py \
    --csv result/custom/graphs_1000/${model}/*/training_history.csv \
    --output_dir plots/${model}
done
```

### 技巧2：自动查找所有CSV文件

```bash
# 使用find命令查找所有training_history.csv
python plot_from_csv.py \
  --csv $(find result/ -name "training_history.csv") \
  --output_dir all_results
```

### 技巧3：只生成特定类型的分析

修改main()函数中的绘图调用，注释掉不需要的图表函数。

---

## 📝 完整示例工作流

### 场景：对比3个模型在不同数据集规模下的表现

```bash
# 步骤1：对比单一规模下的训练过程
python plot_from_csv.py \
  --csv result/custom/graphs_1000/gcn/run1/training_history.csv \
        result/custom/graphs_1000/gat/run1/training_history.csv \
        result/custom/graphs_1000/gin/run1/training_history.csv \
  --labels GCN GAT GIN \
  --output_dir plots/graphs_1000_comparison

# 步骤2：扩展性分析
python plot_from_csv.py \
  --scaling \
  --result_dir result \
  --models gcn gat gin \
  --output_dir plots/scaling_analysis

# 步骤3：只看GCN的不同规模
python plot_from_csv.py \
  --scaling \
  --result_dir result \
  --models gcn \
  --output_dir plots/gcn_scaling
```

---

## 📞 技术支持

如果遇到问题，请检查：
1. Python和依赖包版本
2. CSV文件格式
3. 目录结构
4. 文件权限

---

## 📄 版本信息

- **脚本版本**：基于当前代码
- **兼容性**：Python 3.6+
- **最后更新**：2025-11-05

---

## 🔗 相关文件

- `plot_from_csv.py` - 主脚本
- `training_history.csv` - 训练历史CSV文件
- `experiment_results.json` - 实验结果JSON文件

---

**祝使用愉快！** 🎉

python plot_from_csv.py --scaling --result_dir result --output_dir scaling_plots --exclude_models hybrid gunet