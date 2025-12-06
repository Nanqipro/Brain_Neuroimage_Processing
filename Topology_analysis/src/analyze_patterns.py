"""
图模式(Pattern)与行为(Behavior)关系分析脚本

分析方案：
1. 描述性统计：计算每个behavior下各pattern的均值、标准差
2. 显著性检验：使用Kruskal-Wallis H检验（非参数）判断各behavior组间是否存在显著差异
3. 效应量计算：计算Cohen's d或Eta-squared评估差异大小
4. 特征重要性：使用随机森林计算各pattern对behavior分类的重要性
5. 可视化：箱线图、热力图展示pattern在不同behavior下的分布
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings('ignore')

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans', 'Arial']
plt.rcParams['axes.unicode_minus'] = False

# 读取数据
data_file = 'counts/results_2980.csv'
df = pd.read_csv(data_file)

# 提取数据集名称并创建输出目录
import os
dataset_name = os.path.basename(data_file).replace('results_', '').replace('.csv', '')
output_dir = f'analysis/{dataset_name}'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# 过滤掉behavior为NaN的数据
original_count = len(df)
df = df.dropna(subset=['behavior'])
if len(df) < original_count:
    print(f"\n警告: 移除了 {original_count - len(df)} 条behavior为NaN的数据")
    print(f"剩余有效数据: {len(df)} 条\n")

# 获取所有pattern列（排除graph_id和behavior）
pattern_cols = [col for col in df.columns if col not in ['graph_id', 'behavior']]
behaviors = df['behavior'].unique()

print("="*80)
print("图模式(Pattern)与行为(Behavior)关系分析")
print("="*80)
print(f"\n数据概览：")
print(f"  总样本数: {len(df)}")
print(f"  Pattern数量: {len(pattern_cols)}")
print(f"  Behavior类别: {behaviors}")
print(f"  各Behavior样本数: \n{df['behavior'].value_counts()}")

# ============================================================================
# 1. 描述性统计分析
# ============================================================================
print("\n" + "="*80)
print("1. 描述性统计分析")
print("="*80)

descriptive_stats = []
for behavior in behaviors:
    behavior_data = df[df['behavior'] == behavior][pattern_cols]
    stats_dict = {
        'behavior': behavior,
        'n_samples': len(behavior_data)
    }
    for col in pattern_cols:
        stats_dict[f'{col}_mean'] = behavior_data[col].mean()
        stats_dict[f'{col}_std'] = behavior_data[col].std()
    descriptive_stats.append(stats_dict)

desc_df = pd.DataFrame(descriptive_stats)
desc_df.to_csv(f'{output_dir}/descriptive_stats.csv', index=False)
print(f"已保存: {output_dir}/descriptive_stats.csv")

# ============================================================================
# 2. 统计显著性检验（Kruskal-Wallis H检验）
# ============================================================================
print("\n" + "="*80)
print("2. Kruskal-Wallis H检验（检验不同behavior在各pattern上是否有显著差异）")
print("="*80)

kruskal_results = []
skipped_patterns = []
for col in pattern_cols:
    # 准备各behavior组的数据
    groups = [df[df['behavior'] == b][col].values for b in behaviors]
    
    # 检查是否所有值都相同（方差为0）
    if df[col].std() == 0:
        skipped_patterns.append(col)
        continue
    
    try:
        # 进行Kruskal-Wallis检验
        h_stat, p_value = stats.kruskal(*groups)
        
        kruskal_results.append({
            'pattern': col,
            'H_statistic': h_stat,
            'p_value': p_value,
            'significant': 'Yes' if p_value < 0.05 else 'No'
        })
    except ValueError as e:
        # 如果仍然出错，跳过该pattern
        skipped_patterns.append(col)
        continue

if skipped_patterns:
    print(f"\n跳过的patterns（值全部相同或无差异）: {len(skipped_patterns)}")
    print(f"  {', '.join(skipped_patterns)}")

kruskal_df = pd.DataFrame(kruskal_results).sort_values('p_value')
kruskal_df.to_csv(f'{output_dir}/kruskal_wallis_test.csv', index=False)
print(f"\n显著性pattern数量（p < 0.05）: {(kruskal_df['p_value'] < 0.05).sum()}/{len(pattern_cols)}")
print("\nTop 10 最显著的patterns:")
print(kruskal_df.head(10)[['pattern', 'p_value', 'significant']])
print(f"\n已保存: {output_dir}/kruskal_wallis_test.csv")

# ============================================================================
# 3. 效应量分析（Eta-squared）
# ============================================================================
print("\n" + "="*80)
print("3. 效应量分析（Eta-squared）")
print("="*80)

effect_sizes = []
for col in pattern_cols:
    # 计算组间平方和（SSB）和总平方和（SST）
    grand_mean = df[col].mean()
    ssb = sum([len(df[df['behavior'] == b]) * (df[df['behavior'] == b][col].mean() - grand_mean)**2 
               for b in behaviors])
    sst = sum((df[col] - grand_mean)**2)
    eta_squared = ssb / sst if sst > 0 else 0
    
    effect_sizes.append({
        'pattern': col,
        'eta_squared': eta_squared,
        'effect_size': 'Large' if eta_squared > 0.14 else ('Medium' if eta_squared > 0.06 else 'Small')
    })

effect_df = pd.DataFrame(effect_sizes).sort_values('eta_squared', ascending=False)
effect_df.to_csv(f'{output_dir}/effect_sizes.csv', index=False)
print("\nTop 10 效应量最大的patterns:")
print(effect_df.head(10))
print(f"\n已保存: {output_dir}/effect_sizes.csv")

# ============================================================================
# 4. 特征重要性分析（随机森林）
# 注意：这里使用随机森林的目的是评估各pattern对区分behavior的相对重要性，
# 而不是建立预测模型。准确率高可能因为：
# 1. 样本分布不均衡（Close-arm占78%）
# 2. 使用全部数据训练和测试（过拟合）
# 3. 某些patterns与特定behavior高度相关
# 特征重要性分数反映了各pattern在decision tree分裂时的贡献度
# ============================================================================
print("\n" + "="*80)
print("4. 随机森林特征重要性分析")
print("="*80)

X = df[pattern_cols].values
y = LabelEncoder().fit_transform(df['behavior'].values)

rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
rf.fit(X, y)

feature_importance = pd.DataFrame({
    'pattern': pattern_cols,
    'importance': rf.feature_importances_
}).sort_values('importance', ascending=False)

feature_importance.to_csv(f'{output_dir}/feature_importance.csv', index=False)
print("\nTop 10 最重要的patterns:")
print(feature_importance.head(10))
print(f"\n随机森林分类准确率: {rf.score(X, y):.4f}")
print(f"\n已保存: {output_dir}/feature_importance.csv")

# ============================================================================
# 5. 可视化
# ============================================================================
print("\n" + "="*80)
print("5. 生成可视化图表")
print("="*80)

# 5.1 只展示有变化的patterns的箱线图（过滤掉全为0的）
valid_patterns = [p for p in kruskal_df['pattern'].values if df[p].std() > 0]
top_patterns = valid_patterns[:min(5, len(valid_patterns))]  # 最多展示5个，除非有效数据不足5个

if len(top_patterns) > 0:
    n_patterns = len(top_patterns)
    fig, axes = plt.subplots(1, n_patterns, figsize=(5*n_patterns, 5))
    if n_patterns == 1:
        axes = [axes]
    
    for idx, pattern in enumerate(top_patterns):
        ax = axes[idx]
        df.boxplot(column=pattern, by='behavior', ax=ax)
        p_val = kruskal_df[kruskal_df["pattern"]==pattern]["p_value"].values[0]
        ax.set_title(f'{pattern}\n(p={p_val:.2e})', fontsize=12)
        ax.set_xlabel('Behavior', fontsize=10)
        ax.set_ylabel('Count', fontsize=10)
        # 旋转x轴标签避免重叠
        ax.tick_params(axis='x', rotation=45, labelsize=8)
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    plt.suptitle(f'Top {n_patterns} 显著性Patterns在不同Behavior下的分布', fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(f'{output_dir}/top_patterns_boxplot.png', dpi=300, bbox_inches='tight')
    print(f"已保存: {output_dir}/top_patterns_boxplot.png")
    plt.close()
else:
    print("警告: 没有有效的pattern可用于箱线图")

# 5.2 各behavior下pattern均值热力图
heatmap_data = []
for behavior in behaviors:
    means = df[df['behavior'] == behavior][pattern_cols].mean().values
    heatmap_data.append(means)

heatmap_df = pd.DataFrame(heatmap_data, index=behaviors, columns=pattern_cols)

plt.figure(figsize=(16, 6))
sns.heatmap(heatmap_df.T, cmap='YlOrRd', annot=False, cbar_kws={'label': 'Mean Count'})
plt.title('各Behavior下Pattern平均计数热力图', fontsize=14, pad=20)
plt.xlabel('Behavior', fontsize=12)
plt.ylabel('Pattern', fontsize=12)
plt.tight_layout()
plt.savefig(f'{output_dir}/pattern_behavior_heatmap.png', dpi=300, bbox_inches='tight')
print(f"已保存: {output_dir}/pattern_behavior_heatmap.png")
plt.close()

# 5.3 特征重要性条形图
top_features = feature_importance.head(15)
plt.figure(figsize=(10, 8))
plt.barh(range(len(top_features)), top_features['importance'].values)
plt.yticks(range(len(top_features)), top_features['pattern'].values)
plt.xlabel('Feature Importance', fontsize=12)
plt.ylabel('Pattern', fontsize=12)
plt.title('Top 15 Pattern特征重要性（随机森林）', fontsize=14)
plt.gca().invert_yaxis()
plt.tight_layout()
plt.savefig(f'{output_dir}/feature_importance_barplot.png', dpi=300, bbox_inches='tight')
print(f"已保存: {output_dir}/feature_importance_barplot.png")
plt.close()

# 5.4 综合对比图：显著性 vs 效应量 vs 特征重要性
merged = kruskal_df.merge(effect_df, on='pattern').merge(feature_importance, on='pattern')
merged['-log10(p)'] = -np.log10(merged['p_value'] + 1e-300)  # 避免log(0)

top_merged = merged.nlargest(15, 'importance')

fig, ax = plt.subplots(figsize=(12, 8))
scatter = ax.scatter(top_merged['-log10(p)'], top_merged['importance'], 
                    s=top_merged['eta_squared']*3000, 
                    alpha=0.6, c=range(len(top_merged)), cmap='viridis')

for idx, row in top_merged.iterrows():
    ax.annotate(row['pattern'], (row['-log10(p)'], row['importance']), 
               fontsize=8, alpha=0.7)

ax.set_xlabel('-log10(p-value) [显著性]', fontsize=12)
ax.set_ylabel('Random Forest Importance [特征重要性]', fontsize=12)
ax.set_title('Pattern综合评估：显著性 × 特征重要性 × 效应量（气泡大小）', fontsize=14)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(f'{output_dir}/comprehensive_comparison.png', dpi=300, bbox_inches='tight')
print(f"已保存: {output_dir}/comprehensive_comparison.png")
plt.close()

# ============================================================================
# 6. 生成综合报告
# ============================================================================
print("\n" + "="*80)
print("6. 生成综合分析报告")
print("="*80)

report = f"""
图模式(Pattern)与行为(Behavior)关系分析报告
{'='*80}

一、数据概况
- 总样本数: {len(df)}
- Pattern类型数: {len(pattern_cols)}
- Behavior类别数: {len(behaviors)}
- Behavior分布: {dict(df['behavior'].value_counts())}

二、主要发现

1. 显著性检验结果（Kruskal-Wallis H检验）
   - 显著性patterns（p < 0.05）: {(kruskal_df['p_value'] < 0.05).sum()}/{len(pattern_cols)}
   - Top 5 最显著patterns:
{kruskal_df.head(5)[['pattern', 'p_value']].to_string(index=False)}

2. 效应量分析（Eta-squared）
   - Large效应（η² > 0.14）: {(effect_df['eta_squared'] > 0.14).sum()}个
   - Medium效应（0.06 < η² ≤ 0.14）: {((effect_df['eta_squared'] > 0.06) & (effect_df['eta_squared'] <= 0.14)).sum()}个
   - Top 5 效应量最大patterns:
{effect_df.head(5)[['pattern', 'eta_squared', 'effect_size']].to_string(index=False)}

3. 特征重要性（随机森林）
   注意：准确率 {rf.score(X, y):.4f} 基于训练集评估（存在过拟合），仅用于参考
   实际意义：反映各pattern对区分behavior的相对重要性
   - Top 5 最重要patterns:
{feature_importance.head(5).to_string(index=False)}

三、结论与建议
1. 显著性高的patterns可用于区分不同behavior
2. 效应量大的patterns在behavior间差异明显
3. 特征重要性高的patterns对分类贡献最大
4. 综合三个指标选择关键patterns进行深入分析

所有结果文件已保存至 {output_dir}/ 目录
"""

with open(f'{output_dir}/analysis_report.txt', 'w', encoding='utf-8') as f:
    f.write(report)

print(report)
print(f"\n已保存: {output_dir}/analysis_report.txt")
print("\n" + "="*80)
print(f"分析完成！所有结果已保存至 {output_dir}/ 目录")
print("="*80)
