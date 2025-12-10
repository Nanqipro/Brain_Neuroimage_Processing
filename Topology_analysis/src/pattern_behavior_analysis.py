"""
================================================================================
图模式(Pattern)与行为(Behavior)关系的统计分析框架
================================================================================

本脚本提供完整的统计分析流程，包括：
1. 描述性统计
2. 非参数显著性检验（Kruskal-Wallis H检验）
3. 效应量分析（Eta-squared）
4. 特征重要性评估（随机森林）
5. 多维度可视化

所有分析都基于严格的数学公式和统计理论。
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import warnings
import os
import glob
from datetime import datetime

warnings.filterwarnings('ignore')

# 设置绘图参数
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans', 'Arial']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.dpi'] = 100


class PatternBehaviorAnalyzer:
    """
    图模式与行为关系分析器
    
    核心数学框架：
    1. Kruskal-Wallis H统计量：H = (12/(N(N+1))) * Σ(R_i²/n_i) - 3(N+1)
    2. Eta-squared效应量：η² = SS_between / SS_total
    3. 随机森林Gini重要性：I(f) = Σ p(t) * ΔGini(t)
    """
    
    def __init__(self, data_file):
        """
        初始化分析器
        
        参数:
            data_file: CSV文件路径，包含graph_id, patterns, behavior列
        """
        self.data_file = data_file
        self.dataset_name = os.path.basename(data_file).replace('results_', '').replace('.csv', '')
        self.output_dir = f'final_analysis/{self.dataset_name}'
        
        # 创建输出目录
        os.makedirs(self.output_dir, exist_ok=True)
        
        # 加载数据
        print("="*80)
        print(f"分析数据集: {self.dataset_name}")
        print("="*80)
        
        self.df = pd.read_csv(data_file)
        
        # 数据清洗：移除NaN
        original_count = len(self.df)
        self.df = self.df.dropna(subset=['behavior'])
        if len(self.df) < original_count:
            print(f"⚠️ 移除了 {original_count - len(self.df)} 条behavior为NaN的数据")
        
        # 识别pattern列
        self.pattern_cols = [col for col in self.df.columns if col not in ['graph_id', 'behavior']]
        self.behaviors = sorted(self.df['behavior'].unique())  # 排序以保证一致性
        
        print(f"✓ 有效样本数: {len(self.df)}")
        print(f"✓ Pattern数量: {len(self.pattern_cols)}")
        print(f"✓ Behavior类别: {len(self.behaviors)}")
        print()
        
    def compute_descriptive_statistics(self):
        """
        步骤1: 描述性统计
        
        数学公式:
            均值: μ = (1/n) * Σx_i
            标准差: σ = sqrt((1/n) * Σ(x_i - μ)²)
            
        目的: 
            了解各behavior组在每个pattern上的中心趋势和离散程度
        """
        print("\n" + "="*80)
        print("步骤1: 描述性统计分析")
        print("="*80)
        print("数学公式: 均值 μ = (1/n)Σx_i, 标准差 σ = sqrt((1/n)Σ(x_i-μ)²)")
        print()
        
        descriptive_stats = []
        
        # 按样本数从大到小排序behaviors
        behavior_counts = self.df['behavior'].value_counts()
        sorted_behaviors = behavior_counts.index.tolist()
        
        for behavior in sorted_behaviors:
            behavior_data = self.df[self.df['behavior'] == behavior][self.pattern_cols]
            stats_dict = {
                'behavior': behavior,
                'n_samples': len(behavior_data)
            }
            for col in self.pattern_cols:
                stats_dict[f'{col}_mean'] = behavior_data[col].mean()
                stats_dict[f'{col}_std'] = behavior_data[col].std()
                stats_dict[f'{col}_median'] = behavior_data[col].median()
            descriptive_stats.append(stats_dict)
        
        self.desc_df = pd.DataFrame(descriptive_stats)
        self.desc_df.to_csv(f'{self.output_dir}/01_descriptive_statistics.csv', index=False)
        
        print(f"各Behavior样本分布:")
        for behavior in sorted_behaviors:
            count = behavior_counts[behavior]
            pct = count / len(self.df) * 100
            print(f"  {behavior}: {count} ({pct:.1f}%)")
        
        print(f"\n✓ 已保存: {self.output_dir}/01_descriptive_statistics.csv")
        
    def kruskal_wallis_test(self):
        """
        步骤2: Kruskal-Wallis H检验（非参数显著性检验）
        
        数学公式:
            H = [12/(N(N+1))] * Σ(R_i²/n_i) - 3(N+1)
            
            其中:
            - N: 总样本数
            - k: 组数（behavior数量）
            - n_i: 第i组样本数
            - R_i: 第i组的秩和
            
        原假设H0: 所有组的分布相同
        备择假设H1: 至少有一对组的分布不同
        
        p值含义:
            在H0为真的条件下，观察到当前H统计量或更极端值的概率
            
        判断标准:
            - p < 0.001: ***极显著
            - p < 0.01:  **高度显著  
            - p < 0.05:  *显著
            - p ≥ 0.05:  不显著(ns)
            
        多重检验校正:
            Bonferroni校正: α_adjusted = α / m
            其中m为检验次数（pattern数量）
        """
        print("\n" + "="*80)
        print("步骤2: Kruskal-Wallis H检验（非参数显著性检验）")
        print("="*80)
        print("数学公式: H = [12/(N(N+1))] * Σ(R_i²/n_i) - 3(N+1)")
        print("原假设H0: 所有behavior组的pattern分布相同")
        print("p值: 在H0为真时观察到当前差异的概率")
        print()
        
        kruskal_results = []
        skipped_patterns = []
        
        for col in self.pattern_cols:
            # 检查方差
            if self.df[col].std() == 0:
                skipped_patterns.append(col)
                continue
            
            # 准备各组数据
            groups = [self.df[self.df['behavior'] == b][col].values for b in self.behaviors]
            
            try:
                # 执行Kruskal-Wallis检验
                h_stat, p_value = stats.kruskal(*groups)
                
                kruskal_results.append({
                    'pattern': col,
                    'H_statistic': h_stat,
                    'p_value': p_value,
                    'df': len(self.behaviors) - 1  # 自由度
                })
            except ValueError:
                skipped_patterns.append(col)
                continue
        
        # 创建结果DataFrame并按p值排序
        self.kruskal_df = pd.DataFrame(kruskal_results).sort_values('p_value', ascending=True)
        
        # 多重检验校正
        n_tests = len(self.kruskal_df)
        bonferroni_alpha = 0.05 / n_tests if n_tests > 0 else 0.05
        
        # 添加显著性标记
        def get_sig_level(p):
            if p < 0.001: return '***'
            elif p < 0.01: return '**'
            elif p < 0.05: return '*'
            else: return 'ns'
        
        self.kruskal_df['sig_level'] = self.kruskal_df['p_value'].apply(get_sig_level)
        self.kruskal_df['significant_0.05'] = self.kruskal_df['p_value'] < 0.05
        self.kruskal_df['significant_bonferroni'] = self.kruskal_df['p_value'] < bonferroni_alpha
        
        # 保存结果
        self.kruskal_df.to_csv(f'{self.output_dir}/02_kruskal_wallis_test.csv', index=False)
        
        # 统计报告
        print(f"跳过的patterns（无变异）: {len(skipped_patterns)}")
        print(f"\n显著性统计:")
        print(f"  极显著 (p < 0.001): {(self.kruskal_df['p_value'] < 0.001).sum()}/{n_tests}")
        print(f"  高度显著 (p < 0.01): {(self.kruskal_df['p_value'] < 0.01).sum()}/{n_tests}")
        print(f"  显著 (p < 0.05): {(self.kruskal_df['p_value'] < 0.05).sum()}/{n_tests}")
        print(f"  Bonferroni校正 (α={bonferroni_alpha:.4f}): {self.kruskal_df['significant_bonferroni'].sum()}/{n_tests}")
        
        print(f"\nTop 5 最显著patterns:")
        print(self.kruskal_df.head(5)[['pattern', 'p_value', 'sig_level']])
        
        print(f"\n✓ 已保存: {self.output_dir}/02_kruskal_wallis_test.csv")
        
    def compute_effect_sizes(self):
        """
        步骤3: 效应量分析（Eta-squared）
        
        数学公式:
            η² = SS_between / SS_total
            
            其中:
            SS_between = Σ n_i * (μ_i - μ_grand)²  (组间平方和)
            SS_total = Σ(x_ij - μ_grand)²          (总平方和)
            
            - n_i: 第i组样本数
            - μ_i: 第i组均值
            - μ_grand: 总体均值
            - x_ij: 第i组第j个观测值
            
        解释:
            η²表示behavior分组能解释pattern变异的比例
            
        判断标准（Cohen, 1988）:
            - η² ≥ 0.14: Large (大效应)
            - 0.06 ≤ η² < 0.14: Medium (中效应)
            - η² < 0.06: Small (小效应)
            
        注意:
            η²与p值独立，p值显著不代表效应量大
        """
        print("\n" + "="*80)
        print("步骤3: 效应量分析（Eta-squared）")
        print("="*80)
        print("数学公式: η² = SS_between / SS_total")
        print("含义: behavior分组能解释pattern变异的比例")
        print("判断: Large(≥0.14), Medium(0.06-0.14), Small(<0.06)")
        print()
        
        effect_sizes = []
        
        for col in self.pattern_cols:
            # 计算总体均值
            grand_mean = self.df[col].mean()
            
            # 计算组间平方和 SS_between
            ss_between = sum([
                len(self.df[self.df['behavior'] == b]) * 
                (self.df[self.df['behavior'] == b][col].mean() - grand_mean)**2
                for b in self.behaviors
            ])
            
            # 计算总平方和 SS_total
            ss_total = sum((self.df[col] - grand_mean)**2)
            
            # 计算eta-squared
            eta_squared = ss_between / ss_total if ss_total > 0 else 0
            
            # 判断效应大小
            if eta_squared >= 0.14:
                effect_size = 'Large'
            elif eta_squared >= 0.06:
                effect_size = 'Medium'
            else:
                effect_size = 'Small'
            
            effect_sizes.append({
                'pattern': col,
                'eta_squared': eta_squared,
                'effect_size': effect_size,
                'SS_between': ss_between,
                'SS_total': ss_total
            })
        
        # 按eta_squared从大到小排序
        self.effect_df = pd.DataFrame(effect_sizes).sort_values('eta_squared', ascending=False)
        self.effect_df.to_csv(f'{self.output_dir}/03_effect_sizes.csv', index=False)
        
        # 统计报告
        print(f"效应量分级:")
        print(f"  Large (η² ≥ 0.14): {(self.effect_df['eta_squared'] >= 0.14).sum()}/{len(self.pattern_cols)}")
        print(f"  Medium (0.06 ≤ η² < 0.14): {((self.effect_df['eta_squared'] >= 0.06) & (self.effect_df['eta_squared'] < 0.14)).sum()}/{len(self.pattern_cols)}")
        print(f"  Small (η² < 0.06): {(self.effect_df['eta_squared'] < 0.06).sum()}/{len(self.pattern_cols)}")
        
        print(f"\nTop 5 最大效应量patterns:")
        print(self.effect_df.head(5)[['pattern', 'eta_squared', 'effect_size']])
        
        print(f"\n✓ 已保存: {self.output_dir}/03_effect_sizes.csv")
        
    def feature_importance_analysis(self):
        """
        步骤4: 特征重要性分析（随机森林）
        
        数学公式:
            Gini不纯度: G(t) = 1 - Σ p_k²
            
            特征重要性: I(f) = Σ [p(t) * ΔG(t,f)]
            
            其中:
            - p_k: 节点t中类别k的比例
            - p(t): 到达节点t的样本比例
            - ΔG(t,f): 使用特征f分裂节点t带来的Gini减少量
            
        目的:
            评估各pattern对behavior分类的相对贡献度
            
        注意:
            1. 准确率基于训练集，存在过拟合，仅供参考
            2. 重点关注特征重要性排序，而非绝对值
            3. 重要性之和 = 1.0
        """
        print("\n" + "="*80)
        print("步骤4: 特征重要性分析（随机森林）")
        print("="*80)
        print("数学公式: Gini不纯度 G(t) = 1 - Σp_k², 重要性 I(f) = Σ[p(t)*ΔG(t,f)]")
        print("目的: 评估各pattern对behavior分类的相对贡献")
        print()
        
        # 准备数据
        X = self.df[self.pattern_cols].values
        y = LabelEncoder().fit_transform(self.df['behavior'].values)
        
        # 训练随机森林
        rf = RandomForestClassifier(
            n_estimators=100, 
            random_state=42, 
            max_depth=10,  # 限制深度减少过拟合
            min_samples_split=20,
            n_jobs=-1
        )
        rf.fit(X, y)
        
        # 提取特征重要性
        importance_df = pd.DataFrame({
            'pattern': self.pattern_cols,
            'importance': rf.feature_importances_
        }).sort_values('importance', ascending=False)
        
        # 计算累积重要性
        importance_df['cumulative_importance'] = importance_df['importance'].cumsum()
        
        # 保存
        importance_df.to_csv(f'{self.output_dir}/04_feature_importance.csv', index=False)
        
        # 统计报告
        train_acc = rf.score(X, y)
        top3_cumsum = importance_df.head(3)['importance'].sum()
        
        print(f"随机森林参数: n_estimators=100, max_depth=10")
        print(f"训练集准确率: {train_acc:.4f} (注意: 存在过拟合)")
        print(f"\nTop 3 patterns累积贡献: {top3_cumsum:.1%}")
        print(f"\nTop 5 最重要patterns:")
        print(importance_df.head(5)[['pattern', 'importance', 'cumulative_importance']])
        
        print(f"\n⚠️ 注意: 重点关注重要性排序，不要过度解读准确率！")
        print(f"\n✓ 已保存: {self.output_dir}/04_feature_importance.csv")
        
        self.importance_df = importance_df
        self.rf_model = rf
        
    def generate_summary_table(self):
        """
        步骤5: 生成综合汇总表
        
        整合三个维度的结果：
        1. 显著性（p值）
        2. 效应量（η²）
        3. 特征重要性（Gini importance）
        """
        print("\n" + "="*80)
        print("步骤5: 生成综合汇总表")
        print("="*80)
        
        # 合并三个分析结果
        summary = self.kruskal_df[['pattern', 'p_value', 'sig_level', 'significant_0.05']].copy()
        summary = summary.merge(
            self.effect_df[['pattern', 'eta_squared', 'effect_size']], 
            on='pattern', 
            how='outer'
        )
        summary = summary.merge(
            self.importance_df[['pattern', 'importance']], 
            on='pattern', 
            how='outer'
        )
        
        # 填充缺失值（对于跳过的patterns）
        summary = summary.fillna({
            'p_value': 1.0,
            'sig_level': 'ns',
            'significant_0.05': False,
            'eta_squared': 0.0,
            'effect_size': 'Small',
            'importance': 0.0
        })
        
        # 按综合重要性排序（优先p值，其次η²，最后importance）
        summary['rank_score'] = (
            -np.log10(summary['p_value'] + 1e-300) * 0.4 +  # 显著性权重40%
            summary['eta_squared'] * 100 * 0.3 +              # 效应量权重30%
            summary['importance'] * 0.3                       # RF重要性权重30%
        )
        summary = summary.sort_values('rank_score', ascending=False)
        
        # 保存
        summary.to_csv(f'{self.output_dir}/05_comprehensive_summary.csv', index=False)
        
        print(f"综合排序方法: rank = -log10(p)*0.4 + η²*100*0.3 + importance*0.3")
        print(f"\nTop 5 综合最重要patterns:")
        print(summary.head(5)[['pattern', 'p_value', 'eta_squared', 'importance']])
        
        print(f"\n✓ 已保存: {self.output_dir}/05_comprehensive_summary.csv")
        
        self.summary_df = summary
        
    def visualize_results(self):
        """
        步骤6: 多维度可视化
        
        生成图表:
        1. 箱线图: Top 5显著patterns（子图内按behavior均值排序）
        2. 热力图: Pattern-Behavior矩阵（双向排序）
        3. 条形图: 特征重要性Top 15
        4. 散点图: 三维综合评估（p值 × 重要性 × 效应量）
        """
        print("\n" + "="*80)
        print("步骤6: 生成可视化图表")
        print("="*80)
        
        # 6.1 箱线图（只展示p<0.05的patterns）
        sig_patterns = self.summary_df[self.summary_df['significant_0.05']]['pattern'].head(5).tolist()
        
        if len(sig_patterns) > 0:
            fig, axes = plt.subplots(1, len(sig_patterns), figsize=(5*len(sig_patterns), 5))
            if len(sig_patterns) == 1:
                axes = [axes]
            
            for idx, pattern in enumerate(sig_patterns):
                ax = axes[idx]
                
                # 计算每个behavior的中位数用于排序
                medians = self.df.groupby('behavior')[pattern].median().sort_values(ascending=False)
                sorted_behaviors = medians.index.tolist()
                
                # 准备数据
                data_to_plot = [self.df[self.df['behavior'] == b][pattern].values 
                               for b in sorted_behaviors]
                
                # 绘制箱线图
                bp = ax.boxplot(data_to_plot, labels=sorted_behaviors, patch_artist=True)
                
                # 美化
                for patch in bp['boxes']:
                    patch.set_facecolor('lightblue')
                
                p_val = self.summary_df[self.summary_df['pattern'] == pattern]['p_value'].values[0]
                eta = self.summary_df[self.summary_df['pattern'] == pattern]['eta_squared'].values[0]
                
                ax.set_title(f'{pattern}\np={p_val:.2e}, η²={eta:.3f}', fontsize=11, fontweight='bold')
                ax.set_xlabel('Behavior (按中位数排序)', fontsize=9)
                ax.set_ylabel('Count', fontsize=9)
                ax.tick_params(axis='x', rotation=45, labelsize=8)
                ax.grid(True, alpha=0.3)
                plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
            
            plt.suptitle(f'Top {len(sig_patterns)} 显著Patterns在不同Behavior下的分布', 
                        fontsize=14, fontweight='bold')
            plt.tight_layout()
            plt.savefig(f'{self.output_dir}/06_boxplots_top_patterns.png', dpi=300, bbox_inches='tight')
            plt.close()
            print(f"✓ 箱线图已保存")
        else:
            print("⚠️ 无显著patterns，跳过箱线图")
        
        # 6.2 热力图（双向排序）
        heatmap_data = []
        for behavior in self.behaviors:
            means = self.df[self.df['behavior'] == behavior][self.pattern_cols].mean()
            heatmap_data.append(means)
        
        heatmap_df = pd.DataFrame(heatmap_data, index=self.behaviors, columns=self.pattern_cols)
        
        # 按patterns的总体均值排序
        pattern_order = heatmap_df.mean(axis=0).sort_values(ascending=False).index
        heatmap_df = heatmap_df[pattern_order]
        
        # 按behaviors的样本数排序
        behavior_order = self.df['behavior'].value_counts().index
        heatmap_df = heatmap_df.loc[behavior_order]
        
        plt.figure(figsize=(16, 8))
        sns.heatmap(heatmap_df.T, cmap='YlOrRd', annot=False, 
                   cbar_kws={'label': 'Mean Count'}, linewidths=0.5)
        plt.title('Pattern-Behavior均值热力图（双向排序）', fontsize=14, fontweight='bold', pad=20)
        plt.xlabel('Behavior (按样本数排序)', fontsize=12)
        plt.ylabel('Pattern (按总体均值排序)', fontsize=12)
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/07_heatmap_pattern_behavior.png', dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✓ 热力图已保存")
        
        # 6.3 特征重要性条形图
        top15 = self.importance_df.head(15)
        
        plt.figure(figsize=(10, 8))
        colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(top15)))
        plt.barh(range(len(top15)), top15['importance'].values, color=colors)
        plt.yticks(range(len(top15)), top15['pattern'].values)
        plt.xlabel('Feature Importance (Gini)', fontsize=12, fontweight='bold')
        plt.ylabel('Pattern', fontsize=12, fontweight='bold')
        plt.title('Top 15 Pattern特征重要性（随机森林）', fontsize=14, fontweight='bold')
        plt.gca().invert_yaxis()
        plt.grid(axis='x', alpha=0.3)
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/08_barplot_feature_importance.png', dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✓ 条形图已保存")
        
        # 6.4 综合散点图
        merged = self.summary_df.head(15).copy()
        merged['-log10(p)'] = -np.log10(merged['p_value'] + 1e-300)
        
        plt.figure(figsize=(12, 8))
        scatter = plt.scatter(
            merged['-log10(p)'], 
            merged['importance'],
            s=merged['eta_squared'] * 5000 + 50,  # 气泡大小
            c=merged['rank_score'],
            cmap='coolwarm',
            alpha=0.7,
            edgecolors='black',
            linewidths=1
        )
        
        for _, row in merged.iterrows():
            plt.annotate(
                row['pattern'], 
                (row['-log10(p)'], row['importance']),
                fontsize=9,
                alpha=0.8,
                xytext=(5, 5),
                textcoords='offset points'
            )
        
        plt.xlabel('-log10(p-value) [显著性]', fontsize=12, fontweight='bold')
        plt.ylabel('Random Forest Importance [特征重要性]', fontsize=12, fontweight='bold')
        plt.title('Pattern综合评估：显著性 × 特征重要性 × 效应量（气泡大小）', 
                 fontsize=14, fontweight='bold')
        plt.colorbar(scatter, label='综合排序分数')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/09_scatter_comprehensive.png', dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✓ 散点图已保存")
        
    def generate_report(self):
        """
        步骤7: 生成完整分析报告
        """
        print("\n" + "="*80)
        print("步骤7: 生成完整分析报告")
        print("="*80)
        
        n_sig_005 = (self.kruskal_df['p_value'] < 0.05).sum()
        n_large_effect = (self.effect_df['eta_squared'] >= 0.14).sum()
        top3_importance = self.importance_df.head(3)['importance'].sum()
        
        report = f"""
{'='*80}
图模式(Pattern)与行为(Behavior)关系统计分析报告
{'='*80}

数据集: {self.dataset_name}
生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

{'='*80}
一、数据概况
{'='*80}

总样本数: {len(self.df)}
Pattern类型数: {len(self.pattern_cols)}
Behavior类别数: {len(self.behaviors)}

Behavior分布:
{self.df['behavior'].value_counts().to_string()}

{'='*80}
二、分析方法论
{'='*80}

本分析基于以下严格的数学框架：

1. Kruskal-Wallis H检验（非参数显著性检验）
   
   公式: H = [12/(N(N+1))] * Σ(R_i²/n_i) - 3(N+1)
   
   - 目的: 检验k个独立组在某变量上是否存在显著差异
   - 原假设H0: 所有组的分布相同
   - p值: 在H0为真时观察到当前H值或更极端值的概率
   - 判断: p < 0.05拒绝H0，认为组间存在显著差异
   - 自由度: df = k - 1 = {len(self.behaviors) - 1}
   
2. Eta-squared效应量（η²）
   
   公式: η² = SS_between / SS_total
   
   - SS_between = Σ n_i(μ_i - μ_grand)² (组间平方和)
   - SS_total = Σ(x - μ_grand)² (总平方和)
   - 含义: behavior分组能解释的pattern变异比例
   - 判断: Large(≥0.14), Medium(0.06-0.14), Small(<0.06)
   
3. 随机森林特征重要性
   
   公式: I(f) = Σ [p(t) * ΔGini(t,f)]
   
   - Gini不纯度: G(t) = 1 - Σ p_k²
   - 含义: 特征f在所有决策树节点分裂时的平均Gini减少量
   - 归一化: Σ I(f) = 1.0
   
4. 多重检验校正
   
   Bonferroni校正: α_adjusted = α / m
   
   - α = 0.05 (显著性水平)
   - m = {len(self.pattern_cols)} (检验次数)
   - α_adjusted = {0.05/len(self.pattern_cols):.4f}

{'='*80}
三、主要发现
{'='*80}

3.1 显著性检验结果

检验统计量分布:
- 极显著 (p < 0.001, ***): {(self.kruskal_df['p_value'] < 0.001).sum()} patterns
- 高度显著 (p < 0.01, **): {(self.kruskal_df['p_value'] < 0.01).sum()} patterns
- 显著 (p < 0.05, *): {n_sig_005} patterns
- Bonferroni校正后显著: {self.kruskal_df['significant_bonferroni'].sum()} patterns

Top 5 最显著patterns:
{self.kruskal_df.head(5)[['pattern', 'p_value', 'sig_level']].to_string(index=False)}

解释: 这些patterns在不同behavior间的分布存在统计学显著差异

3.2 效应量分析

效应量分布:
- Large效应 (η² ≥ 0.14): {n_large_effect} patterns
- Medium效应 (0.06 ≤ η² < 0.14): {((self.effect_df['eta_squared'] >= 0.06) & (self.effect_df['eta_squared'] < 0.14)).sum()} patterns
- Small效应 (η² < 0.06): {(self.effect_df['eta_squared'] < 0.06).sum()} patterns

Top 5 最大效应量patterns:
{self.effect_df.head(5)[['pattern', 'eta_squared', 'effect_size']].to_string(index=False)}

解释: η²越大，behavior分组对该pattern变异的解释力越强

3.3 特征重要性分析

随机森林性能:
- 训练集准确率: {self.rf_model.score(self.df[self.pattern_cols].values, LabelEncoder().fit_transform(self.df['behavior'].values)):.4f}
- Top 3 patterns累积贡献: {top3_importance:.1%}

Top 5 最重要patterns:
{self.importance_df.head(5)[['pattern', 'importance']].to_string(index=False)}

解释: 重要性高的patterns对behavior分类贡献大

3.4 综合评估

综合Top 5 patterns（考虑显著性+效应量+重要性）:
{self.summary_df.head(5)[['pattern', 'p_value', 'eta_squared', 'importance', 'rank_score']].to_string(index=False)}

{'='*80}
四、结论与建议
{'='*80}

4.1 核心发现

1. 统计显著性: {n_sig_005}/{len(self.pattern_cols)} patterns具有统计学显著性
2. 实际效应: {n_large_effect} patterns具有大效应量
3. 分类贡献: Top 3 patterns贡献{top3_importance:.1%}的分类信息

4.2 方法学意义

- p值回答"有无差异": 显著性检验判断差异是否真实存在
- η²回答"差异多大": 效应量评估差异的实际大小
- 重要性回答"如何使用": 特征重要性指导实际应用

三者互补，缺一不可！

4.3 研究建议

1. 重点关注同时满足以下条件的patterns:
   - p < 0.05 (有统计学显著性)
   - η² > 0.06 (有中等以上效应)
   - importance > 平均值 (对分类有贡献)

2. 谨慎解读随机森林准确率，避免过度拟合

3. 对于显著但效应小的patterns，需结合领域知识判断其实际意义

{'='*80}
五、输出文件清单
{'='*80}

01_descriptive_statistics.csv     - 描述性统计
02_kruskal_wallis_test.csv       - 显著性检验结果
03_effect_sizes.csv              - 效应量分析
04_feature_importance.csv        - 特征重要性
05_comprehensive_summary.csv     - 综合汇总表
06_boxplots_top_patterns.png     - 箱线图
07_heatmap_pattern_behavior.png  - 热力图
08_barplot_feature_importance.png- 重要性条形图
09_scatter_comprehensive.png     - 综合散点图
10_analysis_report.txt           - 本报告

{'='*80}
报告结束
{'='*80}
"""
        
        # 保存报告
        with open(f'{self.output_dir}/10_analysis_report.txt', 'w', encoding='utf-8') as f:
            f.write(report)
        
        print(f"✓ 完整报告已保存")
        print(f"\n所有分析结果已保存至: {self.output_dir}/")
        
    def run_full_analysis(self):
        """执行完整分析流程"""
        self.compute_descriptive_statistics()
        self.kruskal_wallis_test()
        self.compute_effect_sizes()
        self.feature_importance_analysis()
        self.generate_summary_table()
        self.visualize_results()
        self.generate_report()
        
        print("\n" + "="*80)
        print("✅ 完整分析流程执行完毕！")
        print("="*80)


def main():
    """
    主函数：批量处理counts目录下的所有数据集
    """
    print("\n" + "="*80)
    print("图模式-行为关系统计分析系统")
    print("="*80)
    print()
    
    # 查找所有results_*.csv文件
    data_files = glob.glob('counts/results_*.csv')
    
    if not data_files:
        print("❌ 错误: 在counts/目录下未找到results_*.csv文件")
        return
    
    print(f"找到 {len(data_files)} 个数据集:")
    for f in data_files:
        print(f"  - {os.path.basename(f)}")
    print()
    
    # 逐个分析
    for data_file in sorted(data_files):
        try:
            analyzer = PatternBehaviorAnalyzer(data_file)
            analyzer.run_full_analysis()
            print()
        except Exception as e:
            print(f"❌ 分析 {os.path.basename(data_file)} 时出错: {str(e)}")
            import traceback
            traceback.print_exc()
            print()
    
    print("\n" + "="*80)
    print("🎉 所有数据集分析完成！")
    print("="*80)
    print(f"\n结果保存在: final_analysis/ 目录")
    print()


if __name__ == "__main__":
    main()
