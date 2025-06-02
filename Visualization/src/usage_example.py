#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
时间窗口神经元状态分析 - 增强版使用示例

该示例展示如何使用完善后的State_analysis.py进行：
1. GCN功能验证
2. 传统时间窗口分析
3. GCN时间窗口分析
4. 先进GCN分析（包含注意力机制和时序特征）
5. 结果对比和分析
"""

import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 添加路径以导入State_analysis模块
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from State_analysis import EnhancedStateAnalyzer, verify_gcn_functionality

def main():
    """主函数：演示完整的GCN增强分析流程"""
    
    print("🧠 神经元状态分析 - GCN增强版演示")
    print("=" * 60)
    
    # 1. 初始化分析器
    analyzer = EnhancedStateAnalyzer(
        sampling_rate=4.8,
        window_duration=30.0,
        overlap_ratio=0.5
    )
    
    # 2. 首先验证GCN功能
    print("\n📋 步骤1: 验证GCN功能")
    verify_gcn_functionality(analyzer)
    
    # 3. 加载示例数据
    print("\n📋 步骤2: 加载数据")
    data_path = '../datasets/processed_EMtrace01.xlsx'
    
    if not os.path.exists(data_path):
        print(f"❌ 数据文件不存在: {data_path}")
        print("💡 请确保数据文件存在，或修改data_path变量")
        return
    
    try:
        data = analyzer.load_data(data_path)
        print(f"✅ 数据加载成功: {data.shape}")
        print(f"📊 神经元数量: {len([col for col in data.columns if col.startswith('n')])}")
        print(f"⏱️  时间点数: {len(data)}")
    except Exception as e:
        print(f"❌ 数据加载失败: {e}")
        return
    
    # 4. 创建输出目录
    output_base = '../results/gcn_enhanced_analysis'
    os.makedirs(output_base, exist_ok=True)
    
    # 5. 运行不同的分析方法并比较结果
    methods_to_test = [
        ('ensemble', '传统集成方法'),
        ('gcn_temporal', 'GCN时间窗口分析'),
        ('advanced_gcn', '先进GCN分析（基础）'),
        ('advanced_gcn_enhanced', '先进GCN分析（完整功能）')
    ]
    
    analysis_results = {}
    
    for method_key, method_name in methods_to_test:
        print(f"\n📋 步骤3.{len(analysis_results)+1}: {method_name}")
        print("-" * 40)
        
        method_output_dir = os.path.join(output_base, method_key)
        os.makedirs(method_output_dir, exist_ok=True)
        
        try:
            if method_key == 'ensemble':
                # 传统集成方法
                labels, results_df = analyzer.analyze_temporal_states(
                    data, method='ensemble', n_states=4
                )
                
            elif method_key == 'gcn_temporal':
                # GCN时间窗口分析
                labels, results_df = analyzer.gcn_temporal_analysis(
                    data, method='gcn', n_states=4
                )
                
            elif method_key == 'advanced_gcn':
                # 先进GCN分析（基础配置）
                labels, results_df = analyzer.advanced_gcn_analysis(
                    data, 
                    method='advanced_gcn',
                    n_states=4,
                    use_attention=False,
                    use_temporal_features=False
                )
                
            elif method_key == 'advanced_gcn_enhanced':
                # 先进GCN分析（完整功能）
                labels, results_df = analyzer.advanced_gcn_analysis(
                    data,
                    method='advanced_gcn', 
                    n_states=4,
                    use_attention=True,
                    use_temporal_features=True
                )
            
            # 保存结果
            analysis_results[method_key] = {
                'labels': labels,
                'results_df': results_df,
                'method_name': method_name
            }
            
            # 生成可视化
            analyzer.visualize_temporal_states(data, results_df, method_output_dir)
            
            # 输出关键统计信息
            print(f"   ✅ 分析完成")
            print(f"   📊 时间窗口数: {len(results_df)}")
            print(f"   🧠 神经元数: {results_df['neuron_id'].nunique()}")
            print(f"   🎯 识别状态数: {results_df['state_label'].nunique()}")
            
            # 如果是GCN方法，显示额外信息
            if 'graph_nodes' in results_df.columns:
                print(f"   🔗 平均图节点数: {results_df['graph_nodes'].mean():.1f}")
            if 'graph_edges' in results_df.columns:
                print(f"   🌐 平均图边数: {results_df['graph_edges'].mean():.1f}")
            if 'feature_dim' in results_df.columns:
                print(f"   📏 特征维度: {results_df['feature_dim'].iloc[0] if len(results_df) > 0 else 'N/A'}")
            
        except Exception as e:
            print(f"   ❌ {method_name} 分析失败: {e}")
            analyzer.logger.error(f"{method_name} 分析失败: {e}")
    
    # 6. 结果对比分析
    if len(analysis_results) > 1:
        print(f"\n📋 步骤4: 结果对比分析")
        print("-" * 40)
        
        comparison_results = compare_analysis_results(analysis_results)
        plot_comparison_results(comparison_results, output_base)
        
        # 保存对比结果
        comparison_df = pd.DataFrame(comparison_results).T
        comparison_file = os.path.join(output_base, 'method_comparison.xlsx')
        comparison_df.to_excel(comparison_file)
        print(f"📊 对比结果保存至: {comparison_file}")
    
    # 7. 生成使用报告
    print(f"\n📋 步骤5: 生成使用报告")
    generate_usage_report(analysis_results, output_base)
    
    print(f"\n🎉 所有分析完成！")
    print(f"📁 结果保存在: {output_base}")
    print(f"📚 查看各方法的详细结果和可视化")


def compare_analysis_results(analysis_results):
    """
    对比不同分析方法的结果
    
    Parameters
    ----------
    analysis_results : dict
        各方法的分析结果
        
    Returns
    -------
    dict
        对比结果字典
    """
    comparison = {}
    
    for method_key, result in analysis_results.items():
        results_df = result['results_df']
        method_name = result['method_name']
        
        # 基本统计
        comparison[method_key] = {
            'method_name': method_name,
            'total_windows': len(results_df),
            'num_neurons': results_df['neuron_id'].nunique(),
            'num_states': results_df['state_label'].nunique(),
            'avg_window_duration': results_df['duration'].mean(),
        }
        
        # 状态分布均匀性 (熵)
        state_counts = results_df['state_label'].value_counts()
        state_probs = state_counts / state_counts.sum()
        entropy = -np.sum(state_probs * np.log2(state_probs + 1e-10))
        comparison[method_key]['state_entropy'] = entropy
        
        # 神经元状态多样性
        neuron_diversity = results_df.groupby('neuron_id')['state_label'].nunique()
        comparison[method_key]['avg_neuron_diversity'] = neuron_diversity.mean()
        comparison[method_key]['max_neuron_diversity'] = neuron_diversity.max()
        
        # GCN特定指标
        if 'graph_nodes' in results_df.columns:
            comparison[method_key]['avg_graph_nodes'] = results_df['graph_nodes'].mean()
            comparison[method_key]['avg_graph_edges'] = results_df['graph_edges'].mean()
            comparison[method_key]['graph_density'] = (results_df['graph_edges'].mean() / 
                                                     (results_df['graph_nodes'].mean() * 
                                                      (results_df['graph_nodes'].mean() - 1)))
        else:
            comparison[method_key]['avg_graph_nodes'] = 'N/A'
            comparison[method_key]['avg_graph_edges'] = 'N/A'
            comparison[method_key]['graph_density'] = 'N/A'
    
    return comparison


def plot_comparison_results(comparison_results, output_dir):
    """
    绘制方法对比结果图表
    
    Parameters
    ----------
    comparison_results : dict
        对比结果
    output_dir : str
        输出目录
    """
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    methods = list(comparison_results.keys())
    method_names = [comparison_results[m]['method_name'] for m in methods]
    
    # 1. 识别状态数对比
    state_nums = [comparison_results[m]['num_states'] for m in methods]
    axes[0, 0].bar(method_names, state_nums)
    axes[0, 0].set_title('Identified States by Method')
    axes[0, 0].set_ylabel('Number of States')
    axes[0, 0].tick_params(axis='x', rotation=45)
    
    # 2. 状态分布熵对比
    entropies = [comparison_results[m]['state_entropy'] for m in methods]
    axes[0, 1].bar(method_names, entropies)
    axes[0, 1].set_title('State Distribution Entropy')
    axes[0, 1].set_ylabel('Entropy (bits)')
    axes[0, 1].tick_params(axis='x', rotation=45)
    
    # 3. 神经元状态多样性对比
    diversities = [comparison_results[m]['avg_neuron_diversity'] for m in methods]
    axes[0, 2].bar(method_names, diversities)
    axes[0, 2].set_title('Average Neuron State Diversity')
    axes[0, 2].set_ylabel('States per Neuron')
    axes[0, 2].tick_params(axis='x', rotation=45)
    
    # 4. 图节点数对比（仅GCN方法）
    gcn_methods = [m for m in methods if comparison_results[m]['avg_graph_nodes'] != 'N/A']
    if gcn_methods:
        gcn_names = [comparison_results[m]['method_name'] for m in gcn_methods]
        node_nums = [comparison_results[m]['avg_graph_nodes'] for m in gcn_methods]
        axes[1, 0].bar(gcn_names, node_nums)
        axes[1, 0].set_title('Average Graph Nodes (GCN Methods)')
        axes[1, 0].set_ylabel('Number of Nodes')
        axes[1, 0].tick_params(axis='x', rotation=45)
    else:
        axes[1, 0].text(0.5, 0.5, 'No GCN Methods', ha='center', va='center', transform=axes[1, 0].transAxes)
        axes[1, 0].set_title('Average Graph Nodes (GCN Methods)')
    
    # 5. 图边数对比（仅GCN方法）
    if gcn_methods:
        edge_nums = [comparison_results[m]['avg_graph_edges'] for m in gcn_methods]
        axes[1, 1].bar(gcn_names, edge_nums)
        axes[1, 1].set_title('Average Graph Edges (GCN Methods)')
        axes[1, 1].set_ylabel('Number of Edges')
        axes[1, 1].tick_params(axis='x', rotation=45)
    else:
        axes[1, 1].text(0.5, 0.5, 'No GCN Methods', ha='center', va='center', transform=axes[1, 1].transAxes)
        axes[1, 1].set_title('Average Graph Edges (GCN Methods)')
    
    # 6. 计算效率对比（模拟）
    # 在实际应用中，这里应该记录实际的计算时间
    efficiency_scores = []
    for method in methods:
        if 'advanced_gcn' in method:
            score = 0.6  # GCN方法计算较慢但结果可能更好
        elif 'gcn' in method:
            score = 0.7
        else:
            score = 0.9  # 传统方法较快
        efficiency_scores.append(score)
    
    axes[1, 2].bar(method_names, efficiency_scores)
    axes[1, 2].set_title('Computational Efficiency (Simulated)')
    axes[1, 2].set_ylabel('Efficiency Score')
    axes[1, 2].tick_params(axis='x', rotation=45)
    axes[1, 2].set_ylim(0, 1)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'method_comparison.png'), dpi=300, bbox_inches='tight')
    plt.close()


def generate_usage_report(analysis_results, output_dir):
    """
    生成使用报告
    
    Parameters
    ----------
    analysis_results : dict
        分析结果
    output_dir : str
        输出目录
    """
    report_file = os.path.join(output_dir, 'usage_report.md')
    
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write("# 神经元状态分析 - GCN增强版使用报告\n\n")
        f.write("## 分析概述\n\n")
        f.write("本次分析使用了多种方法来识别神经元的放电状态，包括传统机器学习方法和最新的图神经网络方法。\n\n")
        
        f.write("## 方法对比\n\n")
        f.write("| 方法 | 时间窗口数 | 神经元数 | 识别状态数 | 平均神经元多样性 |\n")
        f.write("|------|------------|----------|------------|------------------|\n")
        
        for method_key, result in analysis_results.items():
            method_name = result['method_name']
            results_df = result['results_df']
            
            neuron_diversity = results_df.groupby('neuron_id')['state_label'].nunique().mean()
            
            f.write(f"| {method_name} | {len(results_df)} | {results_df['neuron_id'].nunique()} | "
                   f"{results_df['state_label'].nunique()} | {neuron_diversity:.2f} |\n")
        
        f.write("\n## GCN方法特点\n\n")
        f.write("### 优势\n")
        f.write("- 🧠 **图结构建模**: 通过相空间重构将时间序列转换为图结构，能够捕获复杂的非线性动态\n")
        f.write("- 🔗 **关系学习**: GCN能够学习节点之间的复杂关系和模式\n")
        f.write("- ⚙️ **注意力机制**: 先进GCN包含注意力机制，能够突出重要特征\n")
        f.write("- 📈 **时序特征**: 可以集成速度、加速度等时序特征，提供更丰富的信息\n\n")
        
        f.write("### 使用建议\n")
        f.write("- 📊 对于复杂的非线性神经元动态，推荐使用 `advanced_gcn` 方法\n")
        f.write("- ⏱️ 对于快速分析，可以使用传统的 `ensemble` 方法\n")
        f.write("- 🔄 对于中等复杂度的分析，`gcn_temporal` 提供了平衡的选择\n\n")
        
        f.write("## 命令行使用示例\n\n")
        f.write("```bash\n")
        f.write("# 验证GCN功能\n")
        f.write("python State_analysis.py --verify-gcn\n\n")
        f.write("# 使用传统方法\n")
        f.write("python State_analysis.py --method ensemble --window-duration 30\n\n")
        f.write("# 使用GCN时间窗口分析\n")
        f.write("python State_analysis.py --method gcn_temporal --window-duration 60\n\n")
        f.write("# 使用完整功能的先进GCN\n")
        f.write("python State_analysis.py --method advanced_gcn --use-attention --use-temporal-features\n")
        f.write("```\n\n")
        
        f.write("## 结果解释\n\n")
        f.write("- **State Label**: 识别的状态编号（0, 1, 2, ...）\n")
        f.write("- **Graph Nodes**: 相空间重构后的图节点数量\n")
        f.write("- **Graph Edges**: 图中的边数量，反映了轨迹的连接复杂度\n")
        f.write("- **Feature Dim**: 增强特征的维度（包含位置、速度、加速度等）\n\n")
        
        f.write("## 注意事项\n\n")
        f.write("- 🔧 确保安装了PyTorch和PyTorch Geometric: `pip install torch torch-geometric`\n")
        f.write("- 💾 GCN方法需要更多内存和计算资源\n")
        f.write("- ⏳ 训练时间会比传统方法长，但结果通常更准确\n")
        f.write("- 📏 建议根据数据复杂度调整窗口大小和重叠比例\n")
    
    print(f"📄 使用报告已生成: {report_file}")


if __name__ == "__main__":
    main() 