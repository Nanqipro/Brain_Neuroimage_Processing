import numpy as np
import pandas as pd
import torch
from analysis_config import AnalysisConfig
from analysis_utils import DataProcessor, StatisticalAnalyzer, ResultSaver
from visualization import VisualizationManager
from kmeans_lstm_analysis import NeuronDataProcessor
import warnings
warnings.filterwarnings('ignore')

class EnhancedAnalyzer:
    """
    增强型分析器类
    用于整合数据处理、统计分析、可视化等多个功能模块，
    实现对神经元活动数据的综合分析
    """
    def __init__(self):
        # 初始化各个功能模块
        self.config = AnalysisConfig()
        self.data_processor = DataProcessor()
        self.statistical_analyzer = StatisticalAnalyzer(self.config)
        self.result_saver = ResultSaver(self.config)
        self.visualizer = VisualizationManager(self.config)
        self.neuron_processor = NeuronDataProcessor(self.config)
    
    def prepare_data(self):
        """
        数据准备和预处理
        包括数据加载、标准化、平衡处理和稀有行为合并
        """
        print("Loading and preprocessing data...")
        X_scaled, y = self.neuron_processor.preprocess_data()
        behavior_labels = self.neuron_processor.label_encoder.classes_
        
        # 如果需要，对数据进行平衡处理
        if self.config.analysis_params['min_samples_per_behavior'] > 0:
            X_scaled, y = self.data_processor.balance_data(
                X_scaled, y, 
                self.config.analysis_params['min_samples_per_behavior']
            )
        
        # 合并样本数过少的稀有行为
        X_scaled, y, behavior_labels = self.data_processor.merge_rare_behaviors(
            X_scaled, y, behavior_labels,
            self.config.analysis_params['min_samples_per_behavior']
        )
        
        return X_scaled, y, behavior_labels
    
    def perform_statistical_analysis(self, X_scaled, y, behavior_labels):
        """
        执行统计分析
        包括方差分析、效应量计算和时间相关性分析
        
        参数:
        X_scaled: 标准化后的神经元活动数据
        y: 行为标签
        behavior_labels: 行为类别名称
        """
        print("\nPerforming statistical analysis...")
        
        # 执行单因素方差分析(ANOVA)
        f_values, p_values = self.statistical_analyzer.perform_anova(X_scaled, y)
        
        # 计算效应量（Cohen's d）
        effect_sizes = self.statistical_analyzer.calculate_effect_sizes(
            X_scaled, y, behavior_labels
        )
        
        # 分析时间相关性
        temporal_correlations = self.statistical_analyzer.analyze_temporal_correlations(
            X_scaled, y
        )
        
        return f_values, p_values, effect_sizes, temporal_correlations
    
    def analyze_behavior_patterns(self, X_scaled, y, behavior_labels):
        """
        分析行为模式
        研究行为转换概率和神经元活动与行为的关系
        
        参数:
        X_scaled: 标准化后的神经元活动数据
        y: 行为标签
        behavior_labels: 行为类别名称
        """
        print("\nAnalyzing behavior patterns...")
        
        # 计算行为转换矩阵
        transitions = np.zeros((len(behavior_labels), len(behavior_labels)))
        for i in range(len(y)-1):
            transitions[y[i], y[i+1]] += 1
        
        # 归一化转换概率
        row_sums = transitions.sum(axis=1)
        transitions_norm = transitions / row_sums[:, np.newaxis]
        
        # 计算每种行为下的平均神经元活动
        behavior_means = {}
        for behavior_idx, behavior in enumerate(behavior_labels):
            behavior_mask = (y == behavior_idx)
            behavior_means[behavior] = np.mean(X_scaled[behavior_mask], axis=0)
        
        # 创建行为-神经元活动关系数据框
        behavior_activity_df = pd.DataFrame(behavior_means).T
        behavior_activity_df.columns = [f'Neuron {i+1}' for i in range(behavior_activity_df.shape[1])]
        
        return transitions_norm, behavior_activity_df
    
    def save_and_visualize_results(self, results):
        """
        保存和可视化分析结果
        生成各类图表并输出关键发现
        
        参数:
        results: 包含所有分析结果的元组
        """
        print("\nSaving and visualizing results...")
        
        # 解包结果数据
        (f_values, p_values, effect_sizes, temporal_correlations,
         transitions_norm, behavior_activity_df, behavior_labels) = results
        
        # 保存统计分析结果
        self.result_saver.save_statistical_results(f_values, p_values, effect_sizes)
        self.result_saver.save_temporal_correlations(temporal_correlations)
        
        # 创建可视化图表
        self.visualizer.set_plot_style()
        self.visualizer.plot_behavior_neuron_correlation(behavior_activity_df)  # 行为-神经元相关性热图
        self.visualizer.plot_behavior_transitions(transitions_norm, behavior_labels)  # 行为转换概率图
        self.visualizer.plot_neuron_network(effect_sizes)  # 神经元网络图
        self.visualizer.plot_temporal_correlations(temporal_correlations)  # 时间相关性图
        self.visualizer.plot_statistical_summary(f_values, p_values, effect_sizes)  # 统计分析总结图
        
        # 输出关键发现
        print("\nKey Findings:")
        # 1. 显示统计显著的神经元（按效应量排序）
        print("\n1. Significant Neurons (p < 0.05, sorted by effect size):")
        significant_neurons = np.where(p_values < self.config.analysis_params['p_value_threshold'])[0]
        
        # 计算显著性神经元的平均效应量
        mean_effect_sizes = np.zeros(len(significant_neurons))
        for i, neuron in enumerate(significant_neurons):
            neuron_effects = [data['effect_sizes'][neuron] for data in effect_sizes.values() 
                            if neuron in data['significant_neurons']]
            mean_effect_sizes[i] = np.mean(neuron_effects) if neuron_effects else 0
            
        # 按效应量排序
        sorted_indices = np.argsort(mean_effect_sizes)[::-1]  # 降序排列
        sorted_neurons = significant_neurons[sorted_indices]
        sorted_effects = mean_effect_sizes[sorted_indices]
        
        print(f"Found {len(sorted_neurons)} significant neurons:")
        for neuron, effect in zip(sorted_neurons + 1, sorted_effects):
            print(f"Neuron {neuron}: mean effect size = {effect:.3f}")
            
        # 绘制显著性神经元的效应量分布图
        self.visualizer.plot_significant_neurons_effect_sizes(sorted_neurons + 1, sorted_effects)
        
        # 2. 显示每种行为特异性神经元
        print("\n2. Behavior-Specific Neurons:")
        for behavior, data in effect_sizes.items():
            significant = data['significant_neurons']
            if len(significant) > 0:
                print(f"\n{behavior}:")
                print(f"Significant neurons: {significant + 1}")
                print(f"Top effect sizes: {data['effect_sizes'][significant][:5]}")
        
        # 3. 显示时间相关性分析结果
        print("\n3. Temporal Correlation Summary:")
        for window, corr_values in temporal_correlations.items():
            mean_corr = np.mean(corr_values)
            print(f"Window size {window}: Mean correlation = {mean_corr:.3f}")
    
    def run_analysis(self):
        """
        运行完整的分析流程
        包括数据准备、统计分析、模式分析和结果可视化
        """
        try:
            # 初始化设置
            self.config.setup_directories()
            self.config.validate_paths()
            
            # 数据准备
            X_scaled, y, behavior_labels = self.prepare_data()
            
            # 统计分析
            f_values, p_values, effect_sizes, temporal_correlations = (
                self.perform_statistical_analysis(X_scaled, y, behavior_labels)
            )
            
            # 行为模式分析
            transitions_norm, behavior_activity_df = (
                self.analyze_behavior_patterns(X_scaled, y, behavior_labels)
            )
            
            # 保存和可视化结果
            results = (f_values, p_values, effect_sizes, temporal_correlations,
                      transitions_norm, behavior_activity_df, behavior_labels)
            self.save_and_visualize_results(results)
            
            print(f"\nAnalysis completed! Results saved to: {self.config.analysis_dir}")
            
        except Exception as e:
            print(f"Error during analysis: {str(e)}")
            raise

def main():
    """主函数：创建分析器实例并运行分析"""
    analyzer = EnhancedAnalyzer()
    analyzer.run_analysis()

if __name__ == "__main__":
    main() 