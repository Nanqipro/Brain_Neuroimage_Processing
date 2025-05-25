# 钙爆发聚类分析可视化改进

## 概述

基于提供的学术论文图片，我们对 `cluster-integrate.py` 代码进行了重要改进，使其生成的可视化图表更符合学术论文标准。

## 主要改进内容

### 1. Gap Statistic方法
- **新增功能**: 实现了Gap Statistic算法来确定最佳聚类数
- **科学价值**: Gap Statistic比轮廓系数更加准确和科学
- **函数**: `calculate_gap_statistic()`
- **输出**: 在最佳聚类数确定图中新增Gap Statistic子图

### 2. 钙爆发间隔时间分析
- **新增功能**: 计算连续钙爆发之间的时间间隔
- **函数**: `calculate_burst_intervals()`
- **可视化**: 生成类似参考图片中子图d的间隔时间直方图
- **过滤**: 自动过滤异常值（>3000秒）

### 3. 钙爆发频率分析
- **新增功能**: 计算每个神经元的钙爆发频率（Hz）
- **函数**: `calculate_burst_frequency()`
- **可视化**: 生成类似参考图片中子图e的频率直方图
- **单位**: 频率以Hz为单位显示

### 4. 学术风格多面板可视化
- **新增功能**: `visualize_burst_attributes_academic()`
- **布局**: 2×3子图布局，完全模仿参考图片风格
- **子图内容**:
  - a: Gap Statistic确定最佳聚类数
  - b: 钙爆发持续时间分布
  - c: 钙爆发振幅分布
  - d: 钙爆发间隔时间分布
  - e: 钙爆发频率分布
  - f: 聚类分布统计

### 5. 学术风格设计元素
- **颜色方案**: 使用学术红色(#D62728)和蓝色(#1F77B4)
- **图表样式**: 
  - 移除顶部和右侧边框
  - 取消网格线
  - 使用专业字体和尺寸
  - 红色填充的直方图
- **标注**: 添加异常值过滤信息
- **标题**: 使用a-f子图标记方式

### 6. 增强的最佳聚类数确定
- **改进功能**: `determine_optimal_k()`现在包含三种方法：
  - 肘部法则
  - 轮廓系数
  - Gap Statistic（新增）
- **决策逻辑**: 优先采用Gap Statistic建议的K值
- **可视化**: 三种方法并排比较显示

### 7. 改进的特征分布可视化
- **学术风格**: 更新`visualize_feature_distribution()`
- **设计**: 更专业的箱线图样式
- **配色**: 一致的学术配色方案

## 使用方法

### 基本使用
```bash
python cluster-integrate.py --input your_data.xlsx
```

### 启用学术风格可视化
```bash
python cluster-integrate.py --input your_data.xlsx --academic_style
```

### 完整参数示例
```bash
python cluster-integrate.py \
    --input ../results/all_datasets_transients/all_datasets_transients.xlsx \
    --output ../results/academic_clustering \
    --academic_style \
    --k 5 \
    --raw_data_dir ../datasets \
    --weights "amplitude:2,duration:2,rise_time:1.5"
```

## 输出文件

新的可视化功能会生成以下文件：

1. `burst_attributes_academic_style.png` - 学术风格的6面板特征分析图
2. `optimal_k_determination_enhanced.png` - 增强的最佳K值确定图（包含Gap Statistic）
3. `cluster_feature_distribution_academic.png` - 学术风格的特征分布箱线图

## 技术特点

### Gap Statistic算法
- 使用均匀分布参考数据集
- 计算聚类内平方和的对数期望值
- 自动选择Gap值最大的K作为最佳聚类数

### 异常值处理
- 持续时间：过滤>300秒的事件
- 振幅：过滤>3 ΔF/F的事件
- 间隔时间：过滤>3000秒的间隔

### 颜色一致性
- 所有可视化使用统一的学术配色方案
- 聚类颜色在不同图表中保持一致
- 使用专业的科学出版物常用色彩

## 科学价值

这些改进使代码生成的图表：
1. 符合Nature、Science等顶级期刊的图表标准
2. 提供更全面的钙爆发特征分析
3. 使用更科学的聚类数确定方法
4. 便于直接用于学术论文发表

## 依赖要求

确保安装以下Python包：
```bash
pip install numpy pandas matplotlib seaborn scikit-learn scipy
```

## 注意事项

1. Gap Statistic计算可能需要较长时间，特别是对于大数据集
2. 学术风格可视化需要合适的数据字段（amplitude, duration等）
3. 建议在使用前确保数据质量和完整性 