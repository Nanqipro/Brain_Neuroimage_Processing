import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler, LabelEncoder
import os

from kmeans_lstm_analysis import NeuronLSTM, NeuronDataProcessor
from analysis_config import AnalysisConfig

import torch.nn.functional as F

class ResultAnalyzer:
    """
    结果分析器类：用于分析神经元活动数据和行为之间的关系
    主要功能包括：
    1. 加载训练好的模型和数据
    2. 分析行为和神经元活动的相关性
    3. 分析神经元活动的时间模式
    4. 分析行为转换模式
    5. 识别关键神经元
    6. 分析时间相关性
    """
    def __init__(self, config):
        """
        初始化结果分析器
        参数：
            config: 配置对象，包含所有必要的参数和路径
        """
        self.config = config
        self.processor = NeuronDataProcessor(config)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
    def balance_data(self, X, y, min_samples):
        """
        对数据进行平衡处理
        参数：
            X: 输入数据
            y: 标签数据
            min_samples: 每个类别的最小样本数
        返回：
            balanced_X: 平衡后的输入数据
            balanced_y: 平衡后的标签数据
        """
        unique_labels, counts = np.unique(y, return_counts=True)
        if np.all(counts >= min_samples):
            return X, y

        balanced_X = []
        balanced_y = []

        for label in unique_labels:
            mask = (y == label)
            label_X = X[mask]
            label_y = y[mask]

            if len(label_X) < min_samples:
                # 如果样本数不足，通过随机重复采样来增加样本
                indices = np.random.choice(len(label_X), min_samples, replace=True)
                label_X = label_X[indices]
                label_y = label_y[indices]
            else:
                # 如果样本数过多，随机选择指定数量的样本
                indices = np.random.choice(len(label_X), min_samples, replace=False)
                label_X = label_X[indices]
                label_y = label_y[indices]

            balanced_X.append(label_X)
            balanced_y.append(label_y)

        return np.vstack(balanced_X), np.concatenate(balanced_y)

    def merge_rare_behaviors(self, X, y, behavior_labels, min_samples):
        """
        合并样本数过少的稀有行为
        参数：
            X: 输入数据
            y: 标签数据
            behavior_labels: 行为标签名称
            min_samples: 最小样本数阈值
        返回：
            X: 处理后的输入数据
            y: 处理后的标签数据
            new_behavior_labels: 更新后的行为标签名称
        """
        unique_labels, counts = np.unique(y, return_counts=True)
        rare_labels = unique_labels[counts < min_samples]

        if len(rare_labels) == 0:
            return X, y, behavior_labels

        # 将稀有行为合并为"其他"类别
        new_y = y.copy()
        other_label = len(unique_labels)
        for label in rare_labels:
            new_y[y == label] = other_label

        # 更新行为标签列表
        new_behavior_labels = [label for i, label in enumerate(behavior_labels) 
                             if i not in rare_labels]
        new_behavior_labels.append("其他")

        # 重新编码标签使其连续
        label_map = {old: new for new, old in enumerate(np.unique(new_y))}
        final_y = np.array([label_map[label] for label in new_y])

        return X, final_y, new_behavior_labels

    def load_model_and_data(self):
        """
        加载预训练模型和预处理数据
        返回：
            model: 训练好的LSTM模型
            X_scaled: 标准化后的神经元数据
            y: 行为标签
        """
        # Load and preprocess data
        X_scaled, y = self.processor.preprocess_data()
        self.behavior_labels = self.processor.label_encoder.classes_
        
        # 数据平衡处理
        if hasattr(self.config, 'analysis_params') and 'min_samples_per_behavior' in self.config.analysis_params:
            min_samples = self.config.analysis_params['min_samples_per_behavior']
            X_scaled, y = self.balance_data(X_scaled, y, min_samples)
            # 合并稀有行为
            X_scaled, y, self.behavior_labels = self.merge_rare_behaviors(
                X_scaled, y, self.behavior_labels, min_samples
            )

        # Load trained model
        input_size = X_scaled.shape[1] + 1  # +1 for cluster label
        num_classes = len(np.unique(y))
        
        model = NeuronLSTM(
            input_size=input_size,
            hidden_size=self.config.hidden_size,
            num_layers=self.config.num_layers,
            num_classes=num_classes
        ).to(self.device)
        
        model.load_state_dict(torch.load(self.config.model_path))
        model.eval()
        
        return model, X_scaled, y
    
    def analyze_behavior_neuron_correlation(self, X_scaled, y):
        """
        分析行为和神经元活动之间的相关性
        参数：
            X_scaled: 标准化后的神经元数据
            y: 行为标签
        返回：
            behavior_activity_df: 行为-神经元活动相关性数据框
        """
        # Calculate mean activity for each behavior
        behavior_means = {}
        for behavior_idx in range(len(self.behavior_labels)):
            behavior_mask = (y == behavior_idx)
            behavior_means[self.behavior_labels[behavior_idx]] = np.mean(X_scaled[behavior_mask], axis=0)
        
        # Create correlation heatmap
        behavior_activity_df = pd.DataFrame(behavior_means).T
        behavior_activity_df.columns = [f'Neuron {i+1}' for i in range(behavior_activity_df.shape[1])]
        
        plt.figure(figsize=self.config.visualization_params['figure_sizes']['correlation'])
        
        # 设置字体样式
        plt.rcParams['font.family'] = 'Arial'
        
        sns.heatmap(behavior_activity_df, 
                   cmap=self.config.visualization_params['colormaps']['correlation'],
                   center=0,
                   annot=False,  # 移除数字标注
                   cbar_kws={'label': 'Correlation Coefficient'})
        
        plt.title('Average Correlation between Behaviors and Neuron Activities', 
                 fontsize=14, 
                 fontfamily='Arial')
        plt.xlabel('Neurons', fontsize=12, fontfamily='Arial')
        plt.ylabel('Behaviors', fontsize=12, fontfamily='Arial')
        plt.xticks(fontsize=10, fontfamily='Arial')
        plt.yticks(fontsize=10, fontfamily='Arial')
        
        # 调整布局以确保所有标签都可见
        plt.tight_layout()
        
        plt.savefig(self.config.correlation_plot, 
                   dpi=self.config.visualization_params['dpi'],
                   bbox_inches='tight')
        plt.close()
        
        return behavior_activity_df
    
    def calculate_temporal_correlations(self, X_scaled, window_size):
        """
        计算给定时间窗口大小下的神经元活动相关性
        
        参数：
            X_scaled: 标准化后的神经元数据
            window_size: 时间窗口大小
            
        返回：
            correlations: 神经元活动的相关性矩阵
        """
        # 检查数据是否足够计算相关性
        if len(X_scaled) <= window_size:
            print(f"警告: 数据长度({len(X_scaled)})小于等于窗口大小({window_size})，无法计算相关性")
            return None
            
        n_neurons = X_scaled.shape[1]
        n_windows = len(X_scaled) - window_size + 1
        
        try:
            # 计算每个窗口内的平均活动
            window_activities = np.array([
                np.mean(X_scaled[i:i+window_size], axis=0)
                for i in range(n_windows)
            ])
            
            # 检查计算结果
            if window_activities.size == 0:
                print(f"警告: 窗口活动计算结果为空")
                return None
                
            # 计算相关性矩阵
            correlations = np.corrcoef(window_activities.T)
            
            # 确保结果是2D数组
            if correlations.ndim != 2:
                print(f"警告: 相关性矩阵维度不正确 (shape={correlations.shape})")
                return None
                
            return correlations
            
        except Exception as e:
            print(f"计算时间相关性时出错: {str(e)}")
            return None

    def analyze_temporal_correlations(self, X_scaled, y):
        """
        分析不同时间窗口下的神经元活动相关性
        
        参数：
            X_scaled: 标准化后的神经元数据
            y: 行为标签
        """
        print("\n分析时间相关性...")
        
        # 确保输出目录存在
        os.makedirs(self.config.temporal_correlation_dir, exist_ok=True)
        
        for window_size in self.config.analysis_params['correlation_windows']:
            print(f"处理时间窗口大小: {window_size}")
            
            # 计算时间相关性
            correlations = self.calculate_temporal_correlations(X_scaled, window_size)
            
            # 检查计算结果是否有效
            if correlations is None:
                print(f"跳过窗口大小 {window_size} 的可视化")
                continue
                
            # 绘制和保存相关性图
            plt.figure(figsize=self.config.visualization_params['figure_sizes']['temporal'])
            try:
                sns.heatmap(
                    correlations, 
                    cmap=self.config.visualization_params['colormaps']['correlation'],
                    center=0,
                    vmin=-1,
                    vmax=1,
                    annot=False,
                    cbar_kws={'label': 'Correlation Coefficient'}
                )
                plt.title(f'Neuron Temporal Correlation (Window Size: {window_size})', fontsize=14)
                plt.xlabel('Neuron ID', fontsize=12)
                plt.ylabel('Neuron ID', fontsize=12)
                # 保存图表
                output_path = self.config.get_temporal_correlation_path(window_size)
                plt.savefig(
                    output_path,
                    dpi=self.config.visualization_params['dpi'],
                    bbox_inches='tight'
                )
                plt.close()
                print(f"已保存时间相关性图表: {output_path}")
                
            except Exception as e:
                print(f"绘制时间相关性图表时出错 (窗口大小 {window_size}): {str(e)}")
                plt.close()  # 确保关闭图表
                continue

    def analyze_temporal_patterns(self, X_scaled, y):
        """
        分析每种行为下神经元活动的时间模式
        
        参数：
            X_scaled: 标准化后的神经元数据
            y: 行为标签
        """
        print("\n分析时间模式...")
        
        # 确保输出目录存在
        os.makedirs(self.config.temporal_pattern_dir, exist_ok=True)
        
        # 获取配置的窗口大小
        window_size = self.config.analysis_params['temporal_window_size']
        min_samples_required = 3  # 最少需要3个时间点才能看出趋势
        
        for behavior_idx, behavior in enumerate(self.behavior_labels):
            print(f"处理行为: {behavior}")
            behavior_mask = (y == behavior_idx)
            behavior_data = X_scaled[behavior_mask]
            
            # 根据数据长度动态调整窗口大小
            actual_window_size = min(window_size, len(behavior_data) // 3)
            
            if len(behavior_data) >= min_samples_required:
                try:
                    # 计算移动平均
                    rolling_mean = np.array([
                        np.mean(behavior_data[i:i+actual_window_size], axis=0) 
                        for i in range(0, len(behavior_data)-actual_window_size+1, max(1, actual_window_size//2))
                    ])
                    
                    if len(rolling_mean) > 0:
                        plt.figure(figsize=(15, 8))  # 增加图表大小以适应更多神经元
                        
                        # 显示前10个神经元的活动
                        colors = plt.cm.tab10(np.linspace(0, 1, 10))  # 使用10种不同的颜色
                        for neuron in range(min(10, rolling_mean.shape[1])):
                            plt.plot(
                                rolling_mean[:, neuron],
                                label=f'Neuron {neuron+1}',
                                linewidth=self.config.visualization_params['line_width'],
                                color=colors[neuron]
                            )
                        
                        plt.title(f'Neuron Activity Temporal Pattern During {behavior}\n(Window Size: {actual_window_size})', fontsize=14)
                        plt.xlabel('Time Window', fontsize=12)
                        plt.ylabel('Normalized Activity', fontsize=12)
                        plt.xticks(fontsize=11)
                        plt.yticks(fontsize=11)
                        plt.legend(fontsize=11, bbox_to_anchor=(1.05, 1), loc='upper left')
                        plt.tight_layout()
                        
                        # 保存图表
                        output_path = self.config.get_temporal_pattern_path(behavior)
                        plt.savefig(
                            output_path,
                            dpi=self.config.visualization_params['dpi'],
                            bbox_inches='tight'
                        )
                        plt.close()
                        print(f"已保存时间模式图表: {output_path}")
                    else:
                        print(f"警告: 行为 {behavior} 的时间序列计算结果为空")
                        
                except Exception as e:
                    print(f"警告: 处理行为 {behavior} 时出错: {str(e)}")
                    plt.close()  # 确保关闭图表
                    
            else:
                print(f"警告: 行为 {behavior} 的样本数量({len(behavior_data)})不足以进行时间模式分析，需要至少{min_samples_required}个样本")
    
    def analyze_behavior_transitions(self, y):
        """
        分析行为之间的转换模式
        参数：
            y: 行为标签
        返回：
            transitions_norm: 归一化后的行为转换概率矩阵
        """
        transitions = np.zeros((len(self.behavior_labels), len(self.behavior_labels)))
        
        for i in range(len(y)-1):
            transitions[y[i], y[i+1]] += 1
        
        # Normalize transitions
        row_sums = transitions.sum(axis=1)
        transitions_norm = transitions / row_sums[:, np.newaxis]
        
        plt.figure(figsize=self.config.visualization_params['figure_sizes']['transitions'])
        sns.heatmap(transitions_norm, 
                   xticklabels=self.behavior_labels,
                   yticklabels=self.behavior_labels,
                   cmap=self.config.visualization_params['colormaps']['transitions'],
                   annot=True,
                   fmt='.2f',
                   annot_kws={'size': 10},
                   cbar_kws={'label': 'Transition Probability'})
        plt.title('Behavior Transition Probability Matrix', fontsize=14)
        plt.xlabel('Target Behavior', fontsize=12)
        plt.ylabel('Initial Behavior', fontsize=12)
        plt.xticks(fontsize=11, rotation=45)
        plt.yticks(fontsize=11, rotation=0)
        plt.savefig(self.config.transition_plot)
        plt.close()
        
        return transitions_norm
    
    def identify_key_neurons(self, X_scaled, y):
        """
        识别对每种行为最具判别性的关键神经元
        使用Cohen's d效应量来衡量神经元的重要性
        
        参数：
            X_scaled: 标准化后的神经元数据
            y: 行为标签
        返回：
            behavior_importance: 每种行为的关键神经元及其重要性
        """
        behavior_importance = {}
        
        for behavior_idx, behavior in enumerate(self.behavior_labels):
            behavior_mask = (y == behavior_idx)
            behavior_data = X_scaled[behavior_mask]
            other_data = X_scaled[~behavior_mask]
            
            # Calculate effect size (Cohen's d) for each neuron
            behavior_mean = np.mean(behavior_data, axis=0)
            other_mean = np.mean(other_data, axis=0)
            behavior_std = np.std(behavior_data, axis=0)
            other_std = np.std(other_data, axis=0)
            
            pooled_std = np.sqrt((behavior_std**2 + other_std**2) / 2)
            effect_size = np.abs(behavior_mean - other_mean) / pooled_std
            
            # Get top neurons
            top_neurons = np.argsort(effect_size)[-self.config.analysis_params['top_neurons_count']:][::-1]
            behavior_importance[behavior] = {
                'neurons': top_neurons + 1,  # +1 for 1-based indexing
                'effect_sizes': effect_size[top_neurons]
            }
        
        # Plot results
        plt.figure(figsize=self.config.visualization_params['figure_sizes']['key_neurons'])
        x_pos = np.arange(len(behavior_importance))
        width = 0.15
        
        # 创建颜色映射
        colors = plt.cm.viridis(np.linspace(0, 1, self.config.analysis_params['top_neurons_count']))
        
        # 绘制柱状图并添加神经元编号标注
        for i in range(self.config.analysis_params['top_neurons_count']):
            effect_sizes = [behavior_importance[b]['effect_sizes'][i] for b in self.behavior_labels]
            neuron_numbers = [behavior_importance[b]['neurons'][i] for b in self.behavior_labels]
            bars = plt.bar(x_pos + i*width, effect_sizes, width, 
                         color=colors[i], label=f'Top {i+1}')
            
            # 在每个柱子上方添加神经元编号
            for idx, (rect, neuron_num) in enumerate(zip(bars, neuron_numbers)):
                height = rect.get_height()
                plt.text(rect.get_x() + rect.get_width()/2., height,
                        f'N{neuron_num}',
                        ha='center', va='bottom', rotation=45,
                        fontsize=10)
        
        plt.xlabel('Behavior', fontsize=12)
        plt.ylabel('Effect Size', fontsize=12)
        plt.title('Key Neurons for Each Behavior', fontsize=14)
        plt.xticks(x_pos + width*2, self.behavior_labels, rotation=45, fontsize=11)
        plt.legend(fontsize=11)
        plt.tight_layout()
        plt.savefig(self.config.key_neurons_plot, dpi=300, bbox_inches='tight')
        plt.close()
        return behavior_importance

def main():
    """
    主函数：执行完整的分析流程
    1. 加载配置
    2. 初始化分析器
    3. 加载模型和数据
    4. 执行各种分析
    5. 保存分析结果
    """
    # Initialize configuration
    config = AnalysisConfig()
    
    try:
        # Setup and validate directories
        config.setup_directories()
        config.validate_paths()
        
        # Initialize analyzer
        analyzer = ResultAnalyzer(config)
        
        print("加载模型和数据...")
        model, X_scaled, y = analyzer.load_model_and_data()
        
        print("\n分析行为-神经元相关性...")
        behavior_activity_df = analyzer.analyze_behavior_neuron_correlation(X_scaled, y)
        print(f"相关性分析完成。结果保存在: {config.correlation_plot}")
        
        print("\n分析时间模式...")
        analyzer.analyze_temporal_patterns(X_scaled, y)
        print(f"时间模式分析完成。结果保存在: {config.temporal_pattern_dir}")
        
        print("\n分析时间相关性...")
        analyzer.analyze_temporal_correlations(X_scaled, y)
        print(f"时间相关性分析完成。结果保存在: {config.temporal_correlation_dir}")
        
        print("\n分析行为转换...")
        transitions = analyzer.analyze_behavior_transitions(y)
        print(f"转换分析完成。结果保存在: {config.transition_plot}")
        
        print("\n识别关键神经元...")
        behavior_importance = analyzer.identify_key_neurons(X_scaled, y)
        print("\n每种行为的关键神经元:")
        for behavior, data in behavior_importance.items():
            print(f"\n{behavior}:")
            for i, (neuron, effect) in enumerate(zip(data['neurons'], data['effect_sizes'])):
                print(f"  神经元 {neuron}: 效应量 = {effect:.3f}")
        
        print(f"\n分析完成！所有结果已保存到: {config.analysis_dir}")
        
    except Exception as e:
        print(f"分析过程中出现错误: {str(e)}")
        raise

if __name__ == "__main__":
    main() 