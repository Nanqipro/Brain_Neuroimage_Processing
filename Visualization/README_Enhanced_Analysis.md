# 增强版神经元放电状态分析系统

## 概述

本系统是对原有神经元状态分析的增强版本，将放电状态从6种调整为4种，并大幅扩展了可视化功能。系统基于多维度特征提取和机器学习技术，能够自动识别神经元的4种典型放电状态。

## 4种神经元放电状态

1. **State I: 高频连续振荡状态**
   - 特征：持续的高频振荡活动
   - 颜色标识：深红色 (#8B0000)

2. **State II: 规律性脉冲放电状态**
   - 特征：周期性的脉冲放电模式
   - 颜色标识：金黄色 (#FFD700)

3. **State III: 间歇性突发状态**
   - 特征：间歇性的突发活动
   - 颜色标识：森林绿 (#228B22)

4. **State IV: 不规律波动状态**
   - 特征：随机的不规律波动
   - 颜色标识：皇家蓝 (#4169E1)

## 主要功能特性

### 1. 多维度特征提取
- **时域特征**：统计量、峰值特征、活动性指标
- **频域特征**：功率谱密度、主导频率、频带功率
- **非线性特征**：样本熵、Hurst指数、分形维数、Lyapunov指数
- **形态学特征**：峰值宽度、突出度、上升/下降时间
- **相空间特征**：基于Takens嵌入定理的相空间重构特征
- **图特征**：时间序列转换为图结构的特征

### 2. 智能状态分类
- **集成聚类**：结合多种聚类算法的最优结果
- **质量评估**：轮廓系数和Calinski-Harabasz指数评估
- **参数自动优化**：如时间延迟τ的自动确定

### 3. 增强版可视化功能

#### 3.1 传统可视化
- **状态波形图**：每种状态的典型神经元波形
- **状态分布图**：饼图和柱状图显示状态分布
- **相空间重构**：3D相空间轨迹可视化
- **特征分析图**：PCA/UMAP降维可视化和特征重要性分析

#### 3.2 新增可视化功能
- **状态热图**：神经元状态分配的热图表示
- **时间动态分析**：
  - 各状态平均活动强度随时间变化
  - 活动强度分布箱线图
  - 峰值频率分析
  - 变异系数分析
- **频率域分析**：
  - 功率谱密度对比
  - 主导频率分布
  - 频带功率分析
  - 频谱重心分析
- **相关性分析**：各状态内神经元间的相关性矩阵
- **状态转换分析**：基于滑动窗口的时间活动模式分析

#### 3.3 交互式可视化（需要Plotly）
- **3D PCA散点图**：交互式3D特征空间可视化
- **时间序列图**：交互式多状态时间序列对比
- **特征重要性图**：交互式特征重要性分析

## 技术架构

### 1. 核心类结构
```
PhaseSpaceAnalyzer
├── 相空间重构
├── 互信息计算
├── 最优时间延迟寻找
└── 动力学特征提取

TemporalGraphBuilder
├── 时序图构建
├── 节点特征提取
└── 边连接构建

EnhancedStateAnalyzer
├── 特征提取管理
├── 状态分类
├── 可视化生成
└── 结果保存
```

### 2. 依赖库
- **核心依赖**：numpy, pandas, matplotlib, seaborn, scipy, scikit-learn
- **可选依赖**：
  - UMAP：用于降维可视化
  - PyTorch Geometric：用于图神经网络
  - Plotly：用于交互式可视化

## 使用方法

### 1. 命令行使用
```bash
python State_analysis.py --input data.xlsx --output-dir results/ --method ensemble --n-states 4
```

### 2. 参数说明
- `--input`：输入数据文件路径（支持Excel和CSV）
- `--output-dir`：输出目录
- `--method`：聚类方法（kmeans/dbscan/ensemble/gcn）
- `--n-states`：状态数量（默认为4）
- `--sampling-rate`：采样频率（默认4.8Hz）

### 3. 编程接口
```python
from State_analysis import EnhancedStateAnalyzer

# 初始化分析器
analyzer = EnhancedStateAnalyzer(sampling_rate=4.8)

# 加载数据
data = analyzer.load_data('neuron_data.xlsx')

# 提取特征
features, feature_names, neuron_names = analyzer.extract_comprehensive_features(data)

# 识别状态
labels = analyzer.identify_states_enhanced(features, method='ensemble', n_states=4)

# 生成可视化
analyzer.visualize_enhanced_states(data, labels, neuron_names, 'output_dir/')

# 保存结果
analyzer.save_enhanced_results(data, labels, neuron_names, features, 
                              feature_names, 'results.xlsx')
```

## 输出文件说明

### 1. 可视化文件
- `enhanced_state_waveforms.png`：状态典型波形图
- `enhanced_state_distribution.png`：状态分布统计图
- `phase_space_reconstruction.png`：相空间重构可视化
- `enhanced_feature_analysis.png`：特征降维和重要性分析
- `state_heatmap.png`：状态分配热图
- `temporal_dynamics.png`：时间动态特性分析
- `frequency_analysis.png`：频率域分析
- `correlation_matrices.png`：相关性矩阵
- `state_transitions.png`：状态转换分析

### 2. 交互式文件（如果Plotly可用）
- `interactive_3d_pca.html`：交互式3D PCA可视化
- `interactive_timeseries.html`：交互式时间序列图
- `interactive_feature_importance.html`：交互式特征重要性图

### 3. 数据文件
- `enhanced_neuron_states_analysis.xlsx`：完整分析结果
  - Sheet 1: 神经元状态分类结果
  - Sheet 2: 状态统计信息
  - Sheet 3: 特征数据
  - Sheet 4: 特征重要性排序

## 测试功能

运行测试脚本验证系统功能：
```bash
python test_enhanced_analysis.py
```

测试脚本会：
1. 生成4种不同特征的模拟神经元数据
2. 执行完整的分析流程
3. 生成所有可视化文件
4. 验证系统的正确性

## 更新日志

### v2.0 主要更新
1. **状态数量调整**：从6种状态调整为4种更具代表性的状态
2. **新增可视化**：增加5类新的可视化分析功能
3. **交互式支持**：支持Plotly交互式可视化
4. **特征增强**：新增相空间特征和图特征
5. **代码优化**：改进了代码结构和错误处理

## 注意事项

1. **内存使用**：大数据集可能需要较多内存，建议分批处理
2. **计算时间**：相空间重构和图特征提取可能较耗时
3. **参数调优**：不同数据集可能需要调整聚类参数
4. **可视化质量**：建议使用高DPI设置获得更好的图像质量

## 引用

如果使用本系统进行研究，请引用相关的理论基础：
- Takens嵌入定理用于相空间重构
- 随机森林用于特征重要性分析
- 集成聚类用于状态分类 