import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib as mpl

# 字体配置：解决'Arial'字体警告问题
mpl.rcParams['font.family'] = 'sans-serif'
mpl.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Liberation Sans', 'FreeSans', 'Helvetica', 'sans-serif']
# 避免使用Arial字体
mpl.rcParams['mathtext.fontset'] = 'cm'
# 设置中文显示支持（如有需要）
mpl.rcParams['axes.unicode_minus'] = False

from sklearn.metrics import confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler, LabelEncoder
import os
import io
import sys
import contextlib
from datetime import datetime
from torch.serialization import add_safe_globals
import networkx as nx
from scipy.stats import pearsonr
from scipy.cluster import hierarchy
from community import community_louvain
from sklearn.cluster import SpectralClustering
import json
from hmmlearn import hmm
from scipy.stats import zscore
from sklearn.preprocessing import MinMaxScaler
from scipy.signal import find_peaks
import warnings
warnings.filterwarnings('ignore')
from sklearn.decomposition import PCA
import random

from kmeans_lstm_analysis import EnhancedNeuronLSTM, NeuronDataProcessor
from analysis_config import AnalysisConfig

import torch.nn.functional as F


# 添加安全的全局变量
add_safe_globals(['_reconstruct'])

# 添加到导入部分后
import logging
logging.getLogger('matplotlib.font_manager').setLevel(logging.ERROR)

# 导入GNN相关模块
try:
    import torch_geometric
    from neuron_gnn import GNNAnalyzer, NeuronGCN, NeuronGAT, TemporalGNN, ModuleGNN
    from neuron_gnn import train_gnn_model, plot_gnn_results, visualize_node_embeddings
    from gnn_topology import (create_gnn_based_topology, visualize_gnn_topology,
                             create_interactive_gnn_topology, save_gnn_topology_data, 
                             analyze_gnn_topology, visualize_gcn_topology, visualize_gat_topology)
    from gnn_visualization import GNNVisualizer
    # 设置GNN支持标志为True，表示成功导入了所有GNN相关模块
    # 此标志将在后续代码中用于条件性地启用GNN分析功能
    HAS_GNN_SUPPORT = True
    print("成功加载GNN支持模块，GNN分析功能已启用")
except ImportError as e:
    print(f"警告: 未找到GNN支持模块 ({str(e)})，将禁用GNN分析功能")
    print("请确保已安装PyTorch Geometric及相关依赖项")
    print("安装命令: pip install torch-geometric torch-scatter torch-sparse")
    HAS_GNN_SUPPORT = False

# 导入GNN拓扑可视化模块
try:
    from gnn_topology import (
        create_gnn_based_topology, 
        visualize_gnn_topology, 
        create_interactive_gnn_topology,
        save_gnn_topology_data,
        analyze_gnn_topology
    )
    # 设置GNN拓扑支持标志为True，表示成功导入了所有GNN拓扑相关模块
    # 此标志将在后续代码中用于条件性地启用GNN拓扑分析功能
    HAS_GNN_TOPOLOGY = True
except ImportError:
    print("警告: 未找到GNN拓扑可视化模块，将禁用GNN拓扑功能")
    HAS_GNN_TOPOLOGY = False

def check_gnn_dependencies():
    """
    检查GNN相关依赖项是否正确安装
    
    返回:
        status: 依赖项检查状态
        missing: 缺失的依赖项列表
    """
    dependencies = {
        'torch': False,
        'torch_geometric': False,
        'torch_scatter': False,
        'torch_sparse': False,
    }
    
    missing = []
    
    # 检查PyTorch
    try:
        import torch
        dependencies['torch'] = True
        print(f"PyTorch版本: {torch.__version__}")
    except ImportError:
        missing.append('torch')
    
    # 检查PyTorch Geometric
    try:
        import torch_geometric
        dependencies['torch_geometric'] = True
        print(f"PyTorch Geometric版本: {torch_geometric.__version__}")
    except ImportError:
        missing.append('torch_geometric')
    
    # 检查torch_scatter
    try:
        import torch_scatter
        dependencies['torch_scatter'] = True
        print(f"Torch Scatter版本: {torch_scatter.__version__}")
    except ImportError:
        missing.append('torch_scatter')
    
    # 检查torch_sparse
    try:
        import torch_sparse
        dependencies['torch_sparse'] = True
        print(f"Torch Sparse版本: {torch_sparse.__version__}")
    except ImportError:
        missing.append('torch_sparse')
    
    # 检查是否能导入我们的GNN模块
    try:
        import neuron_gnn
        print("成功导入neuron_gnn模块")
    except ImportError as e:
        print(f"无法导入neuron_gnn模块: {str(e)}")
        print(f"当前Python路径: {sys.path}")
        missing.append('neuron_gnn')
    
    status = all(dependencies.values())
    
    return status, missing

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
        返回:
            X_scaled: 标准化后的神经元数据
            y: 行为标签
            behavior_labels: 行为标签名称列表
        """
        # 加载和预处理数据
        X_scaled, y = self.processor.preprocess_data()
        self.behavior_labels = self.processor.label_encoder.classes_
        
        # 打印数据集信息
        n_neurons = X_scaled.shape[1]
        print(f"\n数据集信息:")
        print(f"神经元数量: {n_neurons}")
        print(f"样本数量: {len(X_scaled)}")
        print(f"行为类别数: {len(self.behavior_labels)}")
        
        # 打印行为标签映射和样本数量
        print("\n行为标签统计:")
        unique_labels, counts = np.unique(y, return_counts=True)
        for label, count in zip(self.behavior_labels, counts):
            print(f"{label}: {count} 个样本")
        
        # 数据平衡处理
        if hasattr(self.config, 'analysis_params') and 'min_samples_per_behavior' in self.config.analysis_params:
            min_samples = self.config.analysis_params['min_samples_per_behavior']
            X_scaled, y = self.balance_data(X_scaled, y, min_samples)
            # 合并稀有行为
            X_scaled, y, self.behavior_labels = self.merge_rare_behaviors(
                X_scaled, y, self.behavior_labels, min_samples
            )
        
        # 为了兼容旧版代码，仍然加载模型，但不返回它
        try:
            # 初始化模型（但我们不需要它进行分析）
            input_size = X_scaled.shape[1] + 1  # +1 for cluster label
            num_classes = len(np.unique(y))
            
            model = EnhancedNeuronLSTM(
                input_size=input_size,
                hidden_size=self.config.hidden_size,
                num_layers=self.config.num_layers,
                num_classes=num_classes,
                latent_dim=self.config.analysis_params.get('latent_dim', 32),
                num_heads=self.config.analysis_params.get('num_heads', 4),
                dropout=self.config.analysis_params.get('dropout', 0.2)
            ).to(self.device)
            
            # 尝试加载模型权重（仅用于保持完整性）
            try:
                checkpoint = torch.load(self.config.model_path, weights_only=True)
                model.load_state_dict(checkpoint['model_state_dict'])
                print("成功加载模型权重")
            except Exception as e1:
                try:
                    print("尝试使用 weights_only=False 加载模型...")
                    checkpoint = torch.load(self.config.model_path, weights_only=False)
                    model.load_state_dict(checkpoint['model_state_dict'])
                    print("成功加载模型权重")
                except Exception as e2:
                    print(f"注意: 无法加载模型，但这不影响分析过程: {str(e1)}")
        except Exception as e:
            print(f"注意: 模型加载过程出错，但这不影响分析过程: {str(e)}")
        
        # 只返回数据和标签，不返回模型
        return X_scaled, y, self.behavior_labels
    
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
        
        # 使用temporal_window_sizes参数，如果不存在则使用默认值
        window_sizes = self.config.analysis_params.get('temporal_window_sizes', [10, 20, 50, 100])
        print("\n分析时间相关性...")
        
        for window_size in window_sizes:
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
        window_size = self.config.analysis_params.get('temporal_window_size', 10)
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
        effect_size_data = []  # 用于存储所有效应量数据
        
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
            
            # 获取所有神经元的效应量，并按效应量大小排序
            sorted_indices = np.argsort(effect_size)[::-1]  # 从大到小排序
            sorted_effect_sizes = effect_size[sorted_indices]
            neuron_numbers = sorted_indices + 1  # +1 for 1-based indexing
            
            # 将该行为的所有神经元效应量数据添加到列表中
            effect_size_data.append({
                'behavior': behavior,
                'neuron_numbers': neuron_numbers,
                'effect_sizes': sorted_effect_sizes
            })
            
            # Get top neurons for visualization
            top_k = self.config.analysis_params.get('key_neurons_per_behavior', 5)  # 使用key_neurons_per_behavior代替top_neurons_count
            top_neurons = sorted_indices[:top_k]
            behavior_importance[behavior] = {
                'neurons': top_neurons + 1,  # +1 for 1-based indexing
                'effect_sizes': effect_size[top_neurons]
            }
        
        # 保存效应量数据到CSV文件
        # 创建一个包含所有神经元的DataFrame
        all_neurons_df = pd.DataFrame()
        
        for data in effect_size_data:
            behavior = data['behavior']
            # 创建该行为的效应量数据字典
            behavior_dict = {
                f'Neuron_{num}': effect_size 
                for num, effect_size in zip(data['neuron_numbers'], data['effect_sizes'])
            }
            # 将行为名称添加到字典中
            behavior_dict['Behavior'] = behavior
            # 将该行为的数据添加到DataFrame中
            all_neurons_df = pd.concat([all_neurons_df, pd.DataFrame([behavior_dict])], ignore_index=True)
        
        # 设置'Behavior'列为索引
        all_neurons_df.set_index('Behavior', inplace=True)
        
        # 保存到CSV文件
        csv_path = os.path.join(self.config.analysis_dir, 'neuron_effect_sizes.csv')
        all_neurons_df.to_csv(csv_path)
        print(f"\n所有神经元的效应量数据已保存到: {csv_path}")
        
        # Plot results
        plt.figure(figsize=self.config.visualization_params['figure_sizes']['key_neurons'])
        x_pos = np.arange(len(behavior_importance))
        width = 0.15
        
        # 使用key_neurons_per_behavior代替top_neurons_count
        top_k = self.config.analysis_params.get('key_neurons_per_behavior', 5)
        
        # 创建颜色映射
        colors = plt.cm.viridis(np.linspace(0, 1, top_k))
        
        # 绘制柱状图并添加神经元编号标注
        for i in range(top_k):
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

    def build_neuron_network(self, X_scaled, threshold=0.35):
        """
        构建神经元功能连接网络
        参数:
            X_scaled: 标准化后的神经元活动数据
            threshold: 相关性阈值
        返回:
            G: NetworkX图对象
            correlation_matrix: 相关性矩阵
            available_neurons: 可用神经元列表
            threshold: 相关性阈值  此处设置为0.35
        """
        print("\n构建神经元功能连接网络...")
        
        # 获取实际的神经元数量
        n_neurons = X_scaled.shape[1]
        
        # 获取可用的神经元编号列表（排除缺失的神经元）
        neuron_cols = [f'n{i}' for i in range(1, 63)]  # 假设总共有62个可能的神经元
        available_neurons = self.processor.available_neuron_cols
        
        if not hasattr(self.processor, 'available_neuron_cols'):
            # 如果处理器没有保存可用神经元列表，使用默认方法
            print("警告: 找不到可用神经元列表，使用连续编号")
            available_neurons = [f'n{i+1}' for i in range(n_neurons)]
        
        # 创建相关性矩阵
        correlation_matrix = np.zeros((n_neurons, n_neurons))
        
        # 计算相关性矩阵
        for i in range(n_neurons):
            for j in range(i+1, n_neurons):
                corr, _ = pearsonr(X_scaled[:, i], X_scaled[:, j])
                correlation_matrix[i, j] = corr
                correlation_matrix[j, i] = corr
        
        # 创建图对象
        G = nx.Graph()
        
        # 添加节点(使用实际的神经元编号)
        for neuron in available_neurons:
            G.add_node(neuron)
        
        # 添加边(基于相关性阈值)
        for i in range(n_neurons):
            for j in range(i+1, n_neurons):
                if abs(correlation_matrix[i, j]) >= threshold:
                    G.add_edge(available_neurons[i], 
                              available_neurons[j], 
                              weight=abs(correlation_matrix[i, j]))
        
        print(f"网络构建完成: {len(G.nodes())} 个节点, {len(G.edges())} 条边")
        return G, correlation_matrix, available_neurons

    def extract_main_connections(self, G, correlation_matrix, available_neurons, method='threshold', **kwargs):
        """
        提取神经元网络中的主要连接线路
        
        参数:
            G: 原始神经元网络图对象
            correlation_matrix: 相关性矩阵
            available_neurons: 可用神经元列表
            method: 提取方法，可选值包括:
                   'threshold' - 使用相关性阈值
                   'top_edges' - 保留每个节点的top-k连接
                   'mst' - 最小生成树
            **kwargs: 其他参数，如:
                      threshold - threshold方法的阈值
                      k - top_edges方法要保留的每个节点的边数
                      
        返回:
            main_G: 主要连接线路的NetworkX图对象
            info: 关于提取过程的附加信息字典
        """
        print(f"\n正在使用'{method}'方法提取主要连接线路...")
        
        if method == 'threshold':
            # 使用更高阈值找出最强的相关性连接
            threshold = kwargs.get('threshold', 0.35)
            print(f"使用更高的相关性阈值: {threshold}")
            
            # 创建新图
            main_G = nx.Graph()
            main_G.add_nodes_from(G.nodes())
            
            # 添加超过阈值的边
            for u, v, data in G.edges(data=True):
                if abs(data['weight']) >= threshold:
                    main_G.add_edge(u, v, **data)
            
            info = {
                'method': 'threshold', 
                'threshold': threshold,
                'original_edges': G.number_of_edges(),
                'filtered_edges': main_G.number_of_edges(),
                'retention_rate': main_G.number_of_edges() / G.number_of_edges() if G.number_of_edges() > 0 else 0
            }
            
            print(f"主要连接线路提取完成: {len(main_G.nodes())} 个节点, {len(main_G.edges())} 条边")
            
        elif method == 'top_edges':
            # 为每个神经元保留k个最强连接
            k = kwargs.get('k', 5)
            print(f"为每个节点保留{k}个最强连接")
            
            # 创建新图
            main_G = nx.Graph()
            main_G.add_nodes_from(G.nodes())
            
            # 为每个节点找出top-k边
            for node in G.nodes():
                edges = list(G.edges(node, data=True))
                if not edges:
                    continue
                    
                # 按权重绝对值排序
                edges.sort(key=lambda x: abs(x[2]['weight']), reverse=True)
                
                # 保留top-k边
                for i, (u, v, data) in enumerate(edges):
                    if i < k:
                        if not main_G.has_edge(u, v):  # 避免重复添加
                            main_G.add_edge(u, v, **data)
            
            info = {
                'method': 'top_edges', 
                'k': k,
                'original_edges': G.number_of_edges(),
                'filtered_edges': main_G.number_of_edges(),
                'retention_rate': main_G.number_of_edges() / G.number_of_edges() if G.number_of_edges() > 0 else 0
            }
            
            print(f"主要连接线路提取完成: {len(main_G.nodes())} 个节点, {len(main_G.edges())} 条边")
            
        elif method == 'mst':
            # 使用最小生成树算法提取网络骨架
            print("使用最小生成树算法提取网络骨架")
            
            # 创建正权图（因为MST算法是找最小权重）
            positive_G = nx.Graph()
            positive_G.add_nodes_from(G.nodes())
            
            for u, v, data in G.edges(data=True):
                # 将权重转换为正数，且越强的相关性对应越小的权重
                # 1 - abs(corr) 将强相关变为小权重
                positive_G.add_edge(u, v, weight=1.0 - abs(data['weight']))
            
            # 使用Kruskal算法计算MST
            mst_edges = list(nx.minimum_spanning_edges(positive_G, algorithm='kruskal', data=True))
            
            # 创建MST图
            main_G = nx.Graph()
            main_G.add_nodes_from(G.nodes())
            
            for u, v, data in mst_edges:
                # 恢复原始权重
                orig_weight = G[u][v]['weight']
                main_G.add_edge(u, v, weight=orig_weight)
            
            info = {
                'method': 'mst',
                'original_edges': G.number_of_edges(),
                'filtered_edges': main_G.number_of_edges(),
                'retention_rate': main_G.number_of_edges() / G.number_of_edges() if G.number_of_edges() > 0 else 0
            }
            
            print(f"主要连接线路提取完成: {len(main_G.nodes())} 个节点, {len(main_G.edges())} 条边")
            
        else:
            raise ValueError(f"未知的提取方法: {method}")
        
        # 导出为JSON文件
        output_file = os.path.join(self.config.analysis_dir, f'neuron_network_main_{method}.json')
        self.export_network_to_json(main_G, output_file)
        print(f"{method}方法的主要连接线路已导出到: {output_file}")
        
        return main_G, info

    def export_network_to_json(self, G, output_path, include_attributes=True):
        """
        将网络导出为JSON格式文件
        
        参数:
            G: NetworkX图对象
            output_path: 输出JSON文件路径
            include_attributes: 是否包含节点和边的属性
        """
        print(f"\n正在将网络导出为JSON格式...")
        
        # 准备节点数据
        nodes_data = []
        for node in G.nodes():
            node_data = {
                'id': str(node),
                'label': str(node)
            }
            
            # 添加节点属性
            if include_attributes:
                for attr, value in G.nodes[node].items():
                    node_data[attr] = convert_to_serializable(value)
                    
            nodes_data.append(node_data)
            
        # 准备边数据
        edges_data = []
        for u, v, data in G.edges(data=True):
            edge_data = {
                'source': str(u),
                'target': str(v)
            }
            
            # 添加边权重和其他属性
            if 'weight' in data:
                edge_data['weight'] = float(data['weight'])
                
            if include_attributes:
                for attr, value in data.items():
                    if attr != 'weight':  # 已经添加过权重
                        edge_data[attr] = convert_to_serializable(value)
                        
            edges_data.append(edge_data)
            
        # 创建网络数据结构
        network_data = {
            'nodes': nodes_data,
            'links': edges_data
        }
        
        # 导出为JSON文件
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(network_data, f, indent=2, ensure_ascii=False)
            
        print(f"网络已成功导出到: {output_path}")
        
        return output_path

    def analyze_network_topology(self, G):
        """
        分析神经元网络的拓扑特征
        参数:
            G: NetworkX图对象
        返回:
            metrics: 包含各种拓扑指标的字典
        """
        print("\n分析网络拓扑特征...")
        
        metrics = {}
        
        try:
            # 基本网络指标
            metrics['node_count'] = G.number_of_nodes()
            metrics['edge_count'] = G.number_of_edges()
            metrics['average_degree'] = float(2 * G.number_of_edges()) / G.number_of_nodes()
            
            # 中心性指标
            print("计算中心性指标...")
            metrics['degree_centrality'] = nx.degree_centrality(G)
            metrics['betweenness_centrality'] = nx.betweenness_centrality(G)
            metrics['clustering_coefficient'] = nx.clustering(G)
            metrics['average_clustering'] = nx.average_clustering(G)
            
            # 连通性分析
            print("分析网络连通性...")
            if nx.is_connected(G):
                metrics['is_connected'] = True
                metrics['average_shortest_path_length'] = nx.average_shortest_path_length(G)
                metrics['diameter'] = nx.diameter(G)
            else:
                metrics['is_connected'] = False
                components = list(nx.connected_components(G))
                metrics['number_of_components'] = len(components)
                metrics['largest_component_size'] = len(max(components, key=len))
            
            # 社区检测
            print("执行社区检测...")
            try:
                communities_generator = nx.community.girvan_newman(G)
                communities = tuple(sorted(c) for c in next(communities_generator))
                
                # 将社区信息转换为字典格式
                community_dict = {}
                for i, community in enumerate(communities):
                    for node in community:
                        community_dict[node] = i
                        
                metrics['communities'] = community_dict
                metrics['number_of_communities'] = len(communities)
                
                # 计算模块度
                metrics['modularity'] = nx.community.modularity(G, communities)
                
            except Exception as e:
                print(f"社区检测警告: {str(e)}")
                # 使用连通分量作为备选方案
                connected_components = list(nx.connected_components(G))
                community_dict = {}
                for i, component in enumerate(connected_components):
                    for node in component:
                        community_dict[node] = i
                metrics['communities'] = community_dict
                metrics['number_of_communities'] = len(connected_components)
                metrics['modularity'] = 0.0
            
            # 网络密度和其他统计指标
            metrics['density'] = nx.density(G)
            metrics['average_neighbor_degree'] = nx.average_neighbor_degree(G)
            
            print("网络分析完成")
            
        except Exception as e:
            print(f"网络分析过程中出现错误: {str(e)}")
            # 返回基本指标
            metrics['node_count'] = G.number_of_nodes()
            metrics['edge_count'] = G.number_of_edges()
            metrics['error'] = str(e)
        
        return metrics

    def identify_functional_modules(self, G, correlation_matrix, available_neurons):
        """
        识别神经元功能模块
        
        参数：
            G: networkx图对象
            correlation_matrix: 相关性矩阵
            available_neurons: 可用神经元列表
            
        返回：
            modules: 识别出的功能模块列表
        """
        print("\n识别功能模块...")
        
        # 使用谱聚类识别功能模块
        n_clusters = min(int(np.sqrt(len(G.nodes))), 10)  # 动态确定模块数量
        clustering = SpectralClustering(n_clusters=n_clusters, 
                                      affinity='precomputed',
                                      random_state=42)
        
        # 将相关性矩阵转换为相似度矩阵
        similarity_matrix = np.abs(correlation_matrix)
        np.fill_diagonal(similarity_matrix, 1)
        
        # 执行聚类
        labels = clustering.fit_predict(similarity_matrix)
        
        # 创建神经元索引到名称的映射
        if len(available_neurons) != len(labels):
            print(f"警告: 神经元列表长度({len(available_neurons)})与标签长度({len(labels)})不匹配")
            
        # 确保有足够的神经元标签
        neurons_by_index = {i: neuron for i, neuron in enumerate(available_neurons)}
        
        # 整理模块信息
        modules = {}
        for i in range(n_clusters):
            # 使用映射获取该模块的神经元
            module_neurons = [neurons_by_index[j] for j in range(len(labels)) if labels[j] == i]
            modules[f'Module_{i+1}'] = {
                'neurons': module_neurons,
                'size': len(module_neurons),
                'internal_density': self._calculate_module_density(G, module_neurons)
            }
        
        return modules

    def _calculate_module_density(self, G, module_neurons):
        """
        计算模块内部的连接密度
        """
        subgraph = G.subgraph(module_neurons)
        n = len(module_neurons)
        if n <= 1:
            return 0
        max_edges = (n * (n - 1)) / 2
        return len(subgraph.edges()) / max_edges if max_edges > 0 else 0

    def visualize_network_topology(self, G, metrics, modules):
        """
        可视化网络拓扑分析结果
        
        参数：
            G: networkx图对象
            metrics: 拓扑分析指标
            modules: 功能模块信息
        """
        print("\n可视化网络拓扑分析结果...")
        
        # 1. 网络结构可视化
        plt.figure(figsize=(15, 15))
        pos = nx.spring_layout(G, k=1/np.sqrt(len(G.nodes())), iterations=50)
        
        # 根据模块给节点着色
        colors = plt.cm.rainbow(np.linspace(0, 1, len(modules)))
        node_colors = []
        for node in G.nodes():
            for i, (_, module) in enumerate(modules.items()):
                if node in module['neurons']:
                    node_colors.append(colors[i])
                    break
        
        # 创建节点ID标签 - 为每个节点分配1,2,3...的编号
        node_id_labels = {node: str(i+1) for i, node in enumerate(G.nodes())}
        
        # 绘制网络
        nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=100)
        nx.draw_networkx_edges(G, pos, alpha=0.2)
        
        # 添加节点ID标签
        nx.draw_networkx_labels(G, pos, labels=node_id_labels, font_size=8, font_color='black', font_weight='bold')
        
        plt.title('Neuron Functional Network', fontsize=16)
        plt.savefig(os.path.join(self.config.analysis_dir, 'neuron_network.png'), 
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        # 2. 中心性指标可视化
        plt.figure(figsize=(12, 6))
        degree_centrality = list(metrics['degree_centrality'].values())
        betweenness_centrality = list(metrics['betweenness_centrality'].values())
        
        node_indices = list(range(len(G.nodes())))
        
        plt.bar(node_indices, degree_centrality, alpha=0.6, label='Degree Centrality')
        plt.bar(node_indices, betweenness_centrality, alpha=0.6, label='Betweenness Centrality')
        
        plt.title('Node Centrality Measures', fontsize=14)
        plt.xlabel('Node ID', fontsize=12)
        plt.ylabel('Centrality Value', fontsize=12)
        plt.legend()
        plt.savefig(os.path.join(self.config.analysis_dir, 'centrality_measures.png'),
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        # 3. 社区结构可视化
        plt.figure(figsize=(10, 8))
        if 'communities' in metrics:
            community_sizes = {}
            for node, community_id in metrics['communities'].items():
                if community_id not in community_sizes:
                    community_sizes[community_id] = 0
                community_sizes[community_id] += 1
            
            communities = sorted(community_sizes.items(), key=lambda x: x[1], reverse=True)
            sizes = [size for _, size in communities]
            labels = [f"Community {comm_id+1}" for comm_id, _ in communities]
            
            plt.bar(labels, sizes)
            plt.title('Community Size Distribution', fontsize=14)
            plt.xlabel('Community', fontsize=12)
            plt.ylabel('Number of Neurons', fontsize=12)
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.savefig(os.path.join(self.config.analysis_dir, 'community_distribution.png'),
                       dpi=300, bbox_inches='tight')
            plt.close()

    def create_interactive_visualization(self, G, correlation_matrix, available_neurons):
        """
        创建交互式神经元网络可视化
        
        参数:
            G: NetworkX图对象
            correlation_matrix: 相关性矩阵
            available_neurons: 可用神经元列表
        """
        try:
            # 准备可视化数据
            nodes_data = []
            for i, node in enumerate(G.nodes()):
                # 确保node是字符串类型
                node_str = str(node)
                nodes_data.append({
                    'id': i,
                    'label': f'N{node_str}',
                    'value': G.degree(node),
                    'title': f'Neuron {node_str}<br>Degree: {G.degree(node)}'
                })
            
            edges_data = []
            for u, v, data in G.edges(data=True):
                u_idx = list(G.nodes()).index(u)
                v_idx = list(G.nodes()).index(v)
                edges_data.append({
                    'from': u_idx,
                    'to': v_idx,
                    'value': abs(data.get('weight', 0.5)),
                    'title': f'Weight: {data.get("weight", 0.5):.3f}'
                })
            
            # 创建HTML内容
            html_content = f"""
            <!DOCTYPE html>
            <html>
            <head>
                <title>Interactive Neuron Network</title>
                <script type="text/javascript" src="https://unpkg.com/vis-network/standalone/umd/vis-network.min.js"></script>
                <style type="text/css">
                    #mynetwork {{
                        width: 100%;
                        height: 800px;
                        border: 1px solid lightgray;
                    }}
                    body {{
                        font-family: sans-serif;
                        margin: 20px;
                    }}
                    .info {{
                        margin-bottom: 20px;
                    }}
                </style>
            </head>
            <body>
                <h1>Interactive Neuron Network Visualization</h1>
                <div class="info">
                    <p>This visualization shows the functional connections between neurons.</p>
                    <p>Number of neurons: {len(G.nodes())}</p>
                    <p>Number of connections: {len(G.edges())}</p>
                </div>
                <div id="mynetwork"></div>
                <script type="text/javascript">
                    // 创建节点和边的数据结构
                    var nodes = new vis.DataSet({nodes_data});
                    var edges = new vis.DataSet({edges_data});
                    
                    // 创建网络数据
                    var data = {{
                        nodes: nodes,
                        edges: edges
                    }};
                    
                    // 网络配置
                    var options = {{
                        nodes: {{
                            shape: 'dot',
                            scaling: {{
                                min: 10,
                                max: 30,
                                label: {{
                                    enabled: true,
                                    min: 14,
                                    max: 22,
                                }}
                            }},
                            font: {{
                                size: 14,
                                face: 'Tahoma'
                            }}
                        }},
                        edges: {{
                            width: 2,
                            scaling: {{
                                min: 0.1,
                                max: 5,
                                label: {{
                                    enabled: true
                                }}
                            }},
                            smooth: {{
                                type: 'continuous'
                            }}
                        }},
                        physics: {{
                            stabilization: false,
                            barnesHut: {{
                                gravitationalConstant: -80000,
                                springConstant: 0.01,
                                springLength: 150
                            }}
                        }},
                        interaction: {{
                            navigationButtons: true,
                            keyboard: true
                        }}
                    }};
                    
                    // 创建网络
                    var container = document.getElementById('mynetwork');
                    var network = new vis.Network(container, data, options);
                </script>
            </body>
            </html>
            """
            
            # 确保输出目录存在
            os.makedirs(self.config.interactive_dir, exist_ok=True)
            
            # 保存HTML文件
            output_path = os.path.join(self.config.interactive_dir, 'interactive_neuron_network.html')
            with open(output_path, 'w') as f:
                f.write(html_content)
            
            print(f"交互式神经元网络已保存到: {output_path}")
            return output_path
        except Exception as e:
            print(f"创建交互式可视化时出错: {str(e)}")
            return None

    def create_method_visualization(self, G, correlation_matrix, available_neurons, method):
        """
        为指定方法创建交互式可视化
        
        参数:
            G: NetworkX图对象
            correlation_matrix: 相关性矩阵
            available_neurons: 可用神经元列表
            method: 方法名称 (threshold, top_edges, mst)
        """
        try:
            # 根据方法提取网络
            if method == 'threshold':
                threshold = self.config.analysis_params.get('correlation_threshold', 0.35)
                if threshold < 0.6:  # 使用更高阈值以减少复杂性
                    threshold = 0.6
                print(f"\n正在使用'{method}'方法提取主要连接线路...")
                print(f"使用更高的相关性阈值: {threshold}")
                main_G, _ = self.extract_main_connections(G, correlation_matrix, available_neurons, 
                                                       method=method, threshold=threshold)
            elif method == 'top_edges':
                k = 3  # 减少每个节点的边数以降低复杂性
                print(f"\n正在使用'{method}'方法提取主要连接线路...")
                print(f"为每个节点保留{k}个最强连接")
                main_G, _ = self.extract_main_connections(G, correlation_matrix, available_neurons, 
                                                       method=method, k=k)
            elif method == 'mst':
                print(f"\n正在使用'{method}'方法提取主要连接线路...")
                print("使用最小生成树算法提取网络骨架")
                main_G, _ = self.extract_main_connections(G, correlation_matrix, available_neurons, 
                                                       method=method)
            else:
                print(f"未知方法: {method}")
                return None
            
            print(f"主要连接线路提取完成: {len(main_G.nodes())} 个节点, {len(main_G.edges())} 条边")
            
            # 准备可视化数据
            nodes_data = []
            for i, node in enumerate(main_G.nodes()):
                # 确保node是字符串类型
                node_str = str(node)
                nodes_data.append({
                    'id': i,
                    'label': f'N{node_str}',
                    'value': main_G.degree(node),
                    'title': f'Neuron {node_str}<br>Degree: {main_G.degree(node)}'
                })
            
            edges_data = []
            for u, v, data in main_G.edges(data=True):
                u_idx = list(main_G.nodes()).index(u)
                v_idx = list(main_G.nodes()).index(v)
                edges_data.append({
                    'from': u_idx,
                    'to': v_idx,
                    'value': abs(data.get('weight', 0.5)),
                    'title': f'Weight: {data.get("weight", 0.5):.3f}'
                })
            
            # 创建HTML内容
            html_content = f"""
            <!DOCTYPE html>
            <html>
            <head>
                <title>Interactive Neuron Network ({method})</title>
                <script type="text/javascript" src="https://unpkg.com/vis-network/standalone/umd/vis-network.min.js"></script>
                <style type="text/css">
                    #mynetwork {{
                        width: 100%;
                        height: 800px;
                        border: 1px solid lightgray;
                    }}
                    body {{
                        font-family: sans-serif;
                        margin: 20px;
                    }}
                    .info {{
                        margin-bottom: 20px;
                    }}
                </style>
            </head>
            <body>
                <h1>Interactive Neuron Network Visualization ({method})</h1>
                <div class="info">
                    <p>This visualization shows the functional connections between neurons using the {method} method.</p>
                    <p>Number of neurons: {len(main_G.nodes())}</p>
                    <p>Number of connections: {len(main_G.edges())}</p>
                </div>
                <div id="mynetwork"></div>
                <script type="text/javascript">
                    // 创建节点和边的数据结构
                    var nodes = new vis.DataSet({nodes_data});
                    var edges = new vis.DataSet({edges_data});
                    
                    // 创建网络数据
                    var data = {{
                        nodes: nodes,
                        edges: edges
                    }};
                    
                    // 网络配置
                    var options = {{
                        nodes: {{
                            shape: 'dot',
                            scaling: {{
                                min: 10,
                                max: 30,
                                label: {{
                                    enabled: true,
                                    min: 14,
                                    max: 22,
                                }}
                            }},
                            font: {{
                                size: 14,
                                face: 'Tahoma'
                            }}
                        }},
                        edges: {{
                            width: 2,
                            scaling: {{
                                min: 0.1,
                                max: 5,
                                label: {{
                                    enabled: true
                                }}
                            }},
                            smooth: {{
                                type: 'continuous'
                            }}
                        }},
                        physics: {{
                            stabilization: false,
                            barnesHut: {{
                                gravitationalConstant: -80000,
                                springConstant: 0.01,
                                springLength: 150
                            }}
                        }},
                        interaction: {{
                            navigationButtons: true,
                            keyboard: true
                        }}
                    }};
                    
                    // 创建网络
                    var container = document.getElementById('mynetwork');
                    var network = new vis.Network(container, data, options);
                </script>
            </body>
            </html>
            """
            
            # 确保输出目录存在
            os.makedirs(self.config.gnn_results_dir, exist_ok=True)
            
            # 保存HTML文件
            output_path = os.path.join(self.config.gnn_results_dir, f'interactive_network_main_{method}.html')
            with open(output_path, 'w') as f:
                f.write(html_content)
            
            print(f"生成主要连接({method})交互式可视化完成。结果保存在: {output_path}")
            return output_path
        except Exception as e:
            print(f"创建{method}方法的交互式可视化时出错: {str(e)}")
            return None

    def analyze_behavior_state_transitions(self, X_scaled, y):
        """
        使用HMM分析行为状态转换
        
        参数：
            X_scaled: 标准化后的神经元活动数据
            y: 行为标签
            
        返回：
            hmm_results: HMM分析结果字典
        """
        print("\n分析行为状态转换...")
        
        # 行为样本平衡处理
        X_balanced, y_balanced = self.balance_behaviors_for_hmm(X_scaled, y)
        
        # 进一步降维处理，减少特征数量
        n_components_pca = min(X_balanced.shape[1] // 5, 10)  # 更激进的降维
        print(f"对输入数据进行PCA降维，从 {X_balanced.shape[1]} 减少到 {n_components_pca} 个特征")
        pca = PCA(n_components=n_components_pca)
        X_reduced = pca.fit_transform(X_balanced)
        
        # 使用模型选择找到最优HMM参数
        best_params = self.select_optimal_hmm_params(X_reduced, y_balanced, max_states=4)
        
        # 如果没有找到有效参数，则尝试更小的降维和状态数
        if not best_params:
            print("尝试更简单的模型配置...")
            n_components_pca = min(X_balanced.shape[1] // 10, 5)  # 更极端的降维
            print(f"重新进行PCA降维，从 {X_balanced.shape[1]} 减少到 {n_components_pca} 个特征")
            pca = PCA(n_components=n_components_pca)
            X_reduced = pca.fit_transform(X_balanced)
            
            n_states = 3  # 固定使用3个状态
            print(f"使用最小状态数: {n_states}")
            
            model = hmm.GaussianHMM(
                n_components=n_states, 
                covariance_type="tied",  # 使用共享协方差矩阵
                n_iter=50,
                random_state=42
            )
        else:
            # 使用找到的最优参数
            n_states = best_params['n_states']
            model = best_params['model']
        
        # 计算并显示模型参数数量
        n_features = X_reduced.shape[1]
        if hasattr(model, 'covariance_type') and model.covariance_type == 'diag':
            n_params = n_states * n_states + n_states * n_features + n_states * n_features
        elif hasattr(model, 'covariance_type') and model.covariance_type == 'tied':
            n_params = n_states * n_states + n_states * n_features + n_features*(n_features+1)//2
        else:
            n_params = n_states * n_states + n_states * n_features + n_states * n_features * (n_features+1) // 2
            
        n_samples = X_reduced.shape[0]
        print(f"HMM模型参数数量: {n_params}, 数据样本数: {n_samples}")
        
        # 训练HMM模型
        try:
            if not best_params:
                model.fit(X_reduced)
            
            # 预测隐藏状态序列
            hidden_states = model.predict(X_reduced)
            
            # 计算转移概率矩阵
            transition_matrix = model.transmat_
            
            # 分析状态持续时间
            state_durations = []
            current_state = hidden_states[0]
            current_duration = 1
            
            for state in hidden_states[1:]:
                if state == current_state:
                    current_duration += 1
                else:
                    state_durations.append((current_state, current_duration))
                    current_state = state
                    current_duration = 1
            state_durations.append((current_state, current_duration))
            
            # 计算每个状态的平均持续时间
            avg_durations = {}
            for state in range(n_states):
                durations = [d for s, d in state_durations if s == state]
                avg_durations[state] = np.mean(durations) if durations else 0
            
            # 分析状态-行为映射关系
            mapping_probs = self.analyze_state_behavior_mapping(hidden_states, y_balanced)
            
            # 整理分析结果
            model_score = model.score(X_reduced)
            hmm_results = {
                'transition_matrix': transition_matrix,
                'hidden_states': hidden_states,
                'avg_durations': avg_durations,
                'model_score': model_score,
                'pca': pca,
                'X_reduced': X_reduced,
                'mapping_probs': mapping_probs,
                'n_states': n_states,
                'covariance_type': model.covariance_type if hasattr(model, 'covariance_type') else 'unknown',
                'pca_components': n_components_pca,
                'pca_explained_variance': pca.explained_variance_ratio_.tolist()
            }
            
            # 可视化HMM结果
            self._visualize_hmm_results(hmm_results, self.behavior_labels)
            
            return hmm_results
            
        except Exception as e:
            print(f"HMM分析错误: {str(e)}")
            # 保存更详细的错误信息
            import traceback
            print(f"详细错误信息: {traceback.format_exc()}")
            return None
    
    def analyze_neuron_state_relationships(self, X_scaled, hidden_states):
        """
        分析神经元活动模式与状态转换之间的关系
        
        参数：
            X_scaled: 标准化后的神经元活动数据
            hidden_states: HMM预测的隐藏状态序列
            
        返回：
            relationships: 神经元-状态关系分析结果
        """
        print("\n分析神经元活动与状态转换关系...")
        
        n_neurons = X_scaled.shape[1]
        n_states = len(np.unique(hidden_states))
        
        # 计算每个状态下的神经元活动特征
        state_features = {}
        for state in range(n_states):
            state_mask = (hidden_states == state)
            state_data = X_scaled[state_mask]
            
            state_features[state] = {
                'mean_activity': np.mean(state_data, axis=0),
                'std_activity': np.std(state_data, axis=0),
                'peak_frequency': np.sum(state_data > np.mean(state_data), axis=0) / len(state_data)
            }
        
        # Identify characteristic neurons for each state
        characteristic_neurons = {}
        for state in range(n_states):
            # Calculate neuron significance scores for the state
            significance_scores = (
                state_features[state]['mean_activity'] * 
                state_features[state]['peak_frequency'] / 
                (state_features[state]['std_activity'] + 1e-6)
            )
            
            # Select neurons with highest significance scores
            top_neurons = np.argsort(significance_scores)[-5:]  # Top 5 neurons for each state
            characteristic_neurons[state] = {
                'neuron_indices': top_neurons,
                'significance_scores': significance_scores[top_neurons]
            }
        
        # Analyze neuron activity patterns during state transitions
        transition_patterns = self._analyze_transition_patterns(X_scaled, hidden_states)
        
        relationships = {
            'state_features': state_features,
            'characteristic_neurons': characteristic_neurons,
            'transition_patterns': transition_patterns
        }
        
        # Visualize results
        self._visualize_neuron_state_relationships(relationships, n_states)
        
        return relationships
    
    def predict_transition_points(self, X_scaled, hidden_states):
        """
        预测状态转换的关键时间点
        
        参数：
            X_scaled: 标准化后的神经元活动数据
            hidden_states: HMM预测的隐藏状态序列
            
        返回：
            transition_points: 预测的转换点信息
        """
        print("\n预测状态转换点...")
        
        # 计算神经元活动的整体变化率
        activity_derivative = np.diff(X_scaled, axis=0)
        activity_change = np.sum(np.abs(activity_derivative), axis=1)
        
        # 标准化变化率
        activity_change = zscore(activity_change)
        
        # 检测峰值作为潜在的转换点
        peaks, properties = find_peaks(activity_change, 
                                     height=1.5,  # 仅考虑显著的峰值
                                     distance=10)  # 峰值之间的最小距离
        
        # 获取实际的状态转换点
        true_transitions = np.where(np.diff(hidden_states) != 0)[0]
        
        # 分析预测的转换点
        transition_points = {
            'predicted': peaks,
            'actual': true_transitions,
            'activity_change': activity_change,
            'prediction_scores': properties['peak_heights']
        }
        
        # 评估预测准确性
        prediction_accuracy = self._evaluate_transition_predictions(
            peaks, true_transitions, tolerance=5
        )
        transition_points['accuracy'] = prediction_accuracy
        
        # 可视化转换点预测结果
        self._visualize_transition_points(transition_points, X_scaled)
        
        return transition_points
    
    def _analyze_transition_patterns(self, X_scaled, hidden_states):
        """
        分析状态转换期间的神经元活动模式
        
        参数:
            X_scaled: 标准化后的神经元活动数据
            hidden_states: 隐藏状态序列
            
        返回:
            patterns: 转换模式信息列表
        """
        transitions = np.where(np.diff(hidden_states) != 0)[0]
        window_size = 5  # 转换前后的时间窗口大小
        
        patterns = []
        for t in transitions:
            if t >= window_size and t < len(X_scaled) - window_size:
                before_transition = X_scaled[t-window_size:t]
                after_transition = X_scaled[t+1:t+window_size+1]
                
                patterns.append({
                    'time_point': t,
                    'from_state': hidden_states[t],
                    'to_state': hidden_states[t+1],
                    'before_pattern': np.mean(before_transition, axis=0),
                    'after_pattern': np.mean(after_transition, axis=0),
                    'change_magnitude': np.mean(np.abs(after_transition - before_transition))
                })
        
        return patterns
    
    def _evaluate_transition_predictions(self, predicted, actual, tolerance):
        """
        评估转换点预测的准确性
        
        参数:
            predicted: 预测的转换点
            actual: 实际的转换点
            tolerance: 容忍误差范围
            
        返回:
            评估指标字典，包含精确率、召回率和F1分数
        """
        correct_predictions = 0
        
        for pred in predicted:
            # 检查预测点是否在实际转换点的容忍范围内
            if np.any(np.abs(actual - pred) <= tolerance):
                correct_predictions += 1
        
        precision = correct_predictions / len(predicted) if len(predicted) > 0 else 0
        recall = correct_predictions / len(actual) if len(actual) > 0 else 0
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        return {
            'precision': precision,
            'recall': recall,
            'f1_score': f1_score,
            'tolerance': tolerance
        }
    
    def _visualize_hmm_results(self, hmm_results, behavior_labels):
        """可视化HMM分析结果"""
        # 1. 状态转换概率矩阵热图
        plt.figure(figsize=(12, 5))
        
        # 获取实际的状态数量
        n_states = len(hmm_results['transition_matrix'])
        
        # 生成适当的状态标签
        state_labels = [f'State {i+1}' for i in range(n_states)]
        
        plt.subplot(1, 2, 1)
        sns.heatmap(hmm_results['transition_matrix'], 
                   annot=True, 
                   fmt='.2f',
                   xticklabels=state_labels,
                   yticklabels=state_labels)
        plt.title('State Transition Probability Matrix', fontsize=14)
        plt.xlabel('Target State', fontsize=12)
        plt.ylabel('Initial State', fontsize=12)
        
        # 2. 状态持续时间分布
        plt.subplot(1, 2, 2)
        durations = list(hmm_results['avg_durations'].values())
        plt.bar(range(len(durations)), durations)
        plt.xticks(range(len(durations)), state_labels)
        plt.title('Average State Duration', fontsize=14)
        plt.ylabel('Time Steps', fontsize=12)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.config.analysis_dir, 'hmm_analysis.png'),
                   dpi=300, bbox_inches='tight')
        plt.close()
    
    def _visualize_neuron_state_relationships(self, relationships, n_states):
        """可视化神经元与状态关系分析结果"""
        plt.figure(figsize=(15, 10))
        
        # 绘制各状态的特征神经元活动模式
        for state in range(n_states):
            plt.subplot(n_states, 1, state+1)
            neurons = relationships['characteristic_neurons'][state]['neuron_indices']
            scores = relationships['characteristic_neurons'][state]['significance_scores']
            
            plt.bar(neurons, scores)
            plt.title(f'Characteristic Neurons for State {state+1}', fontsize=14)
            plt.xlabel('Neuron ID', fontsize=12)
            plt.ylabel('Significance Score', fontsize=12)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.config.analysis_dir, 'neuron_state_relationships.png'),
                   dpi=300, bbox_inches='tight')
        plt.close()
    
    def _visualize_transition_points(self, transition_points, X_scaled):
        """可视化状态转换点预测结果"""
        plt.figure(figsize=(15, 8))
        
        # 绘制神经元活动变化率和预测的转换点
        plt.plot(transition_points['activity_change'], 'b-', label='Neural Activity Change Rate')
        plt.plot(transition_points['predicted'], 
                transition_points['activity_change'][transition_points['predicted']], 
                'ro', label='Predicted Transition Points')
        plt.plot(transition_points['actual'],
                transition_points['activity_change'][transition_points['actual']],
                'go', label='Actual Transition Points')
        
        plt.title('State Transition Point Prediction', fontsize=14)
        plt.xlabel('Time Steps', fontsize=12)
        plt.ylabel('Neural Activity Change Rate', fontsize=12)
        plt.legend(fontsize=11)
        
        # 添加预测准确性信息
        accuracy = transition_points['accuracy']
        plt.text(0.02, 0.98, 
                f'Precision: {accuracy["precision"]:.2f}\nRecall: {accuracy["recall"]:.2f}\nF1 Score: {accuracy["f1_score"]:.2f}',
                transform=plt.gca().transAxes,
                verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.config.analysis_dir, 'transition_points.png'),
                   dpi=300, bbox_inches='tight')
        plt.close()

    def balance_behaviors_for_hmm(self, X_scaled, y):
        """
        平衡行为样本用于HMM分析
        
        参数：
            X_scaled: 标准化后的神经元活动数据
            y: 行为标签
            
        返回：
            X_filtered: 处理后的神经元活动数据
            y_filtered: 处理后的行为标签
        """
        unique_labels, counts = np.unique(y, return_counts=True)
        min_samples = 5  # 每个类别至少需要5个样本
        
        # 找出样本数不足的行为
        rare_behaviors = [label for label, count in zip(unique_labels, counts) if count < min_samples]
        
        if not rare_behaviors:
            return X_scaled, y
        
        print(f"对样本量少于{min_samples}的行为进行处理: {[self.behavior_labels[b] for b in rare_behaviors]}")
        
        # 处理方法1: 移除极少样本的行为
        mask = np.array([label not in rare_behaviors for label in y])
        X_filtered = X_scaled[mask]
        y_filtered = y[mask]
        
        # 更新样本统计信息
        remaining_labels, remaining_counts = np.unique(y_filtered, return_counts=True)
        print("处理后的行为样本统计:")
        for label, count in zip(remaining_labels, remaining_counts):
            print(f"  {self.behavior_labels[label]}: {count} 个样本")
        
        return X_filtered, y_filtered

    def select_optimal_hmm_params(self, X_reduced, y, max_states=5):
        """
        选择最优HMM参数
        
        参数:
            X_reduced: 降维后的数据
            y: 行为标签
            max_states: 最大状态数
            
        返回:
            best_params: 包含最优参数的字典
        """
        print("选择最优HMM参数...")
        best_score = -np.inf
        best_params = {}
        
        # 测试不同的状态数量
        for n_states in range(2, min(max_states, len(np.unique(y)))+1):
            # 测试不同的协方差类型
            for cov_type in ['diag', 'tied']:
                try:
                    model = hmm.GaussianHMM(
                        n_components=n_states,
                        covariance_type=cov_type,
                        n_iter=50,
                        random_state=42
                    )
                    
                    # 计算参数数量
                    n_features = X_reduced.shape[1]
                    if cov_type == 'diag':
                        n_params = n_states * n_states + n_states * n_features + n_states * n_features
                    elif cov_type == 'tied':
                        n_params = n_states * n_states + n_states * n_features + n_features*(n_features+1)//2
                    else:
                        n_params = n_states * n_states + n_states * n_features + n_states * n_features * (n_features+1) // 2
                    
                    n_samples = X_reduced.shape[0]
                    
                    # 如果参数过多，跳过此配置
                    if n_params >= n_samples:
                        print(f"  跳过: 状态数={n_states}, 协方差类型={cov_type}, 参数数量({n_params})>=样本数({n_samples})")
                        continue
                        
                    print(f"  尝试: 状态数={n_states}, 协方差类型={cov_type}, 参数数量={n_params}")
                    model.fit(X_reduced)
                    score = model.score(X_reduced)
                    
                    print(f"  结果: 状态数={n_states}, 协方差类型={cov_type}, 分数={score:.2f}")
                    
                    if score > best_score:
                        best_score = score
                        best_params = {
                            'n_states': n_states,
                            'cov_type': cov_type,
                            'score': score,
                            'model': model
                        }
                except Exception as e:
                    print(f"  尝试参数(状态数={n_states}, 协方差类型={cov_type})失败: {str(e)}")
                    continue
        
        if best_params:
            print(f"最优HMM参数: 状态数={best_params['n_states']}, 协方差类型={best_params['cov_type']}, 分数={best_params['score']:.2f}")
        else:
            print("未找到有效的HMM参数配置")
            
        return best_params

    def analyze_state_behavior_mapping(self, hidden_states, y):
        """
        分析HMM状态与行为标签的对应关系
        
        参数:
            hidden_states: HMM预测的隐藏状态序列
            y: 行为标签
            
        返回:
            mapping_probs: 状态-行为映射概率矩阵
        """
        print("\n分析状态-行为映射关系...")
        n_states = len(np.unique(hidden_states))
        n_behaviors = len(np.unique(y))
        
        # 创建映射矩阵（状态 x 行为）
        mapping_matrix = np.zeros((n_states, n_behaviors))
        
        # 统计每个状态对应不同行为的频次
        for i in range(len(y)):
            state = hidden_states[i]
            behavior = y[i]
            mapping_matrix[state, behavior] += 1
        
        # 按行归一化，得到每个状态对应各行为的概率
        row_sums = mapping_matrix.sum(axis=1).reshape(-1, 1)
        mapping_probs = np.zeros_like(mapping_matrix)
        non_zero_rows = (row_sums > 0).flatten()
        
        if np.any(non_zero_rows):
            mapping_probs[non_zero_rows] = mapping_matrix[non_zero_rows] / row_sums[non_zero_rows]
        
        # 生成行为标签
        behavior_labels = [self.behavior_labels[i] for i in range(n_behaviors) if i in np.unique(y)]
        
        # 可视化映射关系
        plt.figure(figsize=(10, 8))
        sns.heatmap(mapping_probs, 
                   cmap='viridis',
                   annot=True,
                   fmt='.2f',
                   xticklabels=behavior_labels,
                   yticklabels=[f'State {i+1}' for i in range(n_states)])
        plt.title('HMM State-Behavior Mapping Relationship', fontsize=14)
        plt.xlabel('Behavior Labels', fontsize=12)
        plt.ylabel('HMM States', fontsize=12)
        plt.tight_layout()
        plt.savefig(os.path.join(self.config.analysis_dir, 'state_behavior_mapping.png'),
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        # 输出映射结果
        print("状态-行为映射概率:")
        for state in range(n_states):
            state_behaviors = []
            for behavior_idx in range(n_behaviors):
                if behavior_idx in np.unique(y) and mapping_probs[state, behavior_idx] > 0.1:
                    state_behaviors.append(f"{self.behavior_labels[behavior_idx]}({mapping_probs[state, behavior_idx]:.2f})")
            
            print(f"  状态{state+1} 主要对应行为: {', '.join(state_behaviors)}")
        
        return mapping_probs

    def analyze_network_with_gnn(self, G, correlation_matrix, X_scaled, y, available_neurons):
        """
        使用图神经网络分析神经元网络
        
        参数:
            G: NetworkX图对象
            correlation_matrix: 神经元相关性矩阵
            X_scaled: 标准化后的神经元活动数据
            y: 行为标签
            available_neurons: 可用神经元列表
            
        返回:
            gnn_results: GNN分析结果字典
        """
        print("\n使用图神经网络分析神经元网络...")
        
        # 导入GNN相关模块
        try:
            from gnn_visualization import GNNVisualizer
        except ImportError:
            print("警告: 无法导入GNNVisualizer，将使用简化版可视化")
            # 创建简化版GNNVisualizer作为后备方案
            class SimpleGNNVisualizer:
                def __init__(self, config):
                    self.config = config
                
                def plot_training_metrics(self, epochs, train_metrics, val_metrics, 
                                     metric_name='Accuracy', title=None, filename=None):
                    """简化版训练指标绘图函数"""
                    plt.figure(figsize=(10, 6))
                    plt.plot(epochs, train_metrics, label='Train')
                    plt.plot(epochs, val_metrics, label='Validation')
                    plt.xlabel('Epochs')
                    plt.ylabel(metric_name)
                    if title:
                        plt.title(title)
                    plt.legend()
                    
                    # 保存图像
                    if filename:
                        output_path = os.path.join(self.config.gnn_dir, filename)
                    else:
                        output_path = os.path.join(self.config.gnn_dir, f"{metric_name.lower()}_curve.png")
                    plt.savefig(output_path, dpi=300, bbox_inches='tight')
                    plt.close()
                    return output_path
            
            GNNVisualizer = SimpleGNNVisualizer
        
        # 数据兼容性检查和修复
        issues, fixed_data = check_data_compatibility(
            X_scaled=X_scaled,
            y=y,
            correlation_matrix=correlation_matrix,
            available_neurons=available_neurons
        )
        
        # 使用修复后的数据
        if issues:
            print("使用修复后的数据进行GNN分析")
            X_scaled = fixed_data['X_scaled']
            y = fixed_data['y']
            correlation_matrix = fixed_data['correlation_matrix']
            available_neurons = fixed_data['available_neurons']
        
        # 初始化GNN分析器
        from neuron_gnn import GNNAnalyzer
        gnn_analyzer = GNNAnalyzer(self.config)
        
        # 初始化可视化器
        visualizer = GNNVisualizer(self.config)
        
        # 结果字典
        gnn_results = {}
        
        # 标准化特征
        X_normalized = X_scaled.copy()
        
        # 将NetworkX图转换为PyTorch Geometric数据格式
        data = gnn_analyzer.convert_network_to_gnn_format(G, X_normalized, y)
        
        # 1. 行为预测GCN模型
        print("\n训练行为预测GCN模型...")
        try:
            # 获取类别数
            num_classes = len(np.unique(y))
            
            # 创建GCN模型
            gcn_model = NeuronGCN(
                in_channels=X_normalized.shape[1],
                hidden_channels=self.config.analysis_params.get('gcn_hidden_channels', 128),  # 增加隐藏层维度
                out_channels=num_classes,
                dropout=self.config.analysis_params.get('gnn_dropout', 0.4),  # 稍微增加dropout
                num_layers=self.config.analysis_params.get('gcn_num_layers', 4),  # 增加层数
                heads=self.config.analysis_params.get('gcn_heads', 4),  # 增加注意力头数
                use_batch_norm=self.config.analysis_params.get('gcn_use_batch_norm', True),
                activation=self.config.analysis_params.get('gcn_activation', 'leaky_relu'),
                alpha=self.config.analysis_params.get('gcn_alpha', 0.2),
                residual=self.config.analysis_params.get('gcn_residual', True)
            )
            
            # 设置训练参数
            epochs = self.config.analysis_params.get('gnn_epochs', 200)
            lr = self.config.analysis_params.get('gnn_learning_rate', 0.008)
            weight_decay = self.config.analysis_params.get('gnn_weight_decay', 1e-3)
            patience = self.config.analysis_params.get('gnn_early_stop_patience', 20)
            early_stopping_enabled = self.config.analysis_params.get('early_stopping_enabled', False)
            
            # 训练模型   
            trained_model, losses, accuracies = train_gnn_model(
                model=gcn_model,
                data=data,
                epochs=epochs,
                lr=lr,
                weight_decay=weight_decay,
                patience=patience,
                device=gnn_analyzer.device,
                early_stopping_enabled=early_stopping_enabled
            )
            
            # 绘制训练曲线
            loss_plot_path = self.config.gcn_training_plot
            plot_gnn_results(losses, loss_plot_path)
            
            # 绘制准确率曲线
            # 创建GNN可视化器用于绘制准确率曲线
            from gnn_visualization import GNNVisualizer
            visualizer = GNNVisualizer(self.config)
            acc_epochs = list(range(1, len(accuracies['train'])+1))
            accuracy_plot_path = visualizer.plot_training_metrics(
                acc_epochs, 
                accuracies['train'], 
                accuracies['val'], 
                metric_name='Accuracy',
                title='GCN Model Training Accuracy Change',
                filename='gcn_accuracy_curve.png'
            )
            
            # 可视化节点嵌入
            gnn_topo_path = self.config.gcn_topology_png
            visualize_node_embeddings(
                trained_model, 
                data, 
                save_path=gnn_topo_path,
                title='GCN Node Embeddings'
            )
            
            # 评估模型
            trained_model.eval()
            with torch.no_grad():
                out = trained_model(data)
                _, predicted = torch.max(out, 1)
                correct = (predicted == data.y).sum().item()
                accuracy = correct / len(data.y)
            
            print(f"GCN行为预测模型完成，准确率: {accuracy:.4f}")
            
            # 基于GCN模型创建拓扑结构
            print("\n基于GCN模型创建神经元拓扑结构...")
            G_gnn = create_gnn_based_topology(
                trained_model, 
                data, 
                G.copy(), 
                node_names=[f"N{i+1}" for i in range(len(available_neurons))]
            )
            
            print(f"GCN拓扑结构创建完成: {G_gnn.number_of_nodes()} 个节点, {G_gnn.number_of_edges()} 条边")
            
            # 保存GCN拓扑数据
            save_gnn_topology_data(
                G_gnn,
                trained_model.get_embeddings(data).detach().cpu().numpy(),
                nx.adjacency_matrix(G_gnn).todense(),
                [f"N{i+1}" for i in range(len(available_neurons))],
                self.config.gcn_topology_data
            )
            
            # 生成静态拓扑结构图
            visualize_gcn_topology(self.config.gcn_topology_data)
            
            # 创建交互式可视化
            create_interactive_gnn_topology(
                G=G_gnn,
                embeddings=trained_model.get_embeddings(data).detach().cpu().numpy(),
                similarities=nx.adjacency_matrix(G_gnn).todense().astype(float),
                node_names=[f"N{i+1}" for i in range(len(available_neurons))],
                output_path=self.config.gcn_interactive_topology
            )
            
            # 记录GCN分析结果
            gnn_results = {
                'gcn_behavior_prediction': {
                    'accuracy': accuracy,
                    'loss_plot_path': loss_plot_path,
                    'topology_plot_path': gnn_topo_path,
                    'accuracy_plot_path': accuracy_plot_path,
                    'train_acc_history': accuracies['train'],
                    'val_acc_history': accuracies['val'],
                    'epochs': epochs
                }
            }
        except Exception as e:
            import traceback
            error_trace = traceback.format_exc()
            print(f"GCN行为预测模型训练出错: {str(e)}")
            print(f"错误详情:\n{error_trace}")
        
        # 2. 神经元模块识别GAT模型
        print("\n训练神经元模块识别GAT模型...")
        try:
            # 创建GAT模型
            # 先使用社区检测算法获取社区作为标签
            try:
                # 添加数据验证和兼容性检查
                print(f"GNN数据转换完成: {data}")
                print(f"X_normalized形状: {X_normalized.shape}")
                print(f"可用神经元数量: {len(available_neurons)}")
                
                # 检查数据维度是否匹配
                if data.x.shape[0] != len(G.nodes()):
                    print(f"警告: 数据维度不匹配 - 数据点数量({data.x.shape[0]})与图节点数量({len(G.nodes())})不一致")
                    print("正在调整数据以适应社区检测...")
                    
                    # 方法1: 如果数据点多于节点，取平均值
                    if data.x.shape[0] > len(G.nodes()):
                        print("数据点多于节点，计算每个神经元的平均活动")
                        # 重新组织数据，按神经元聚合
                        X_reshaped = X_normalized.reshape(-1, len(available_neurons))
                        X_avg = np.mean(X_reshaped, axis=0)
                        # 更新相关性矩阵
                        correlation_matrix = np.corrcoef(X_avg)
                    
                    # 方法2: 如果节点多于数据点，使用可用的数据点
                    else:
                        print("节点多于数据点，使用可用数据点进行社区检测")
                        # 保持原有相关性矩阵
                
                # 确保相关性矩阵是方阵且维度与节点数量匹配
                if correlation_matrix.shape[0] != len(G.nodes()) or correlation_matrix.shape[1] != len(G.nodes()):
                    print(f"警告: 相关性矩阵维度({correlation_matrix.shape})与图节点数量({len(G.nodes())})不匹配")
                    print("重新计算相关性矩阵...")
                    
                    # 重新计算相关性矩阵
                    if len(available_neurons) == len(G.nodes()):
                        # 提取神经元活动数据
                        neuron_activities = []
                        for i, neuron in enumerate(available_neurons):
                            if i < X_normalized.shape[1]:
                                neuron_activities.append(X_normalized[:, i])
                            else:
                                # 如果神经元索引超出范围，使用零向量
                                neuron_activities.append(np.zeros(X_normalized.shape[0]))
                        
                        # 计算新的相关性矩阵
                        correlation_matrix = np.corrcoef(np.array(neuron_activities))
                
                # 确保相关性矩阵是有效的
                if np.isnan(correlation_matrix).any():
                    print("警告: 相关性矩阵包含NaN值，将其替换为0")
                    correlation_matrix = np.nan_to_num(correlation_matrix)
                
                # 使用更健壮的社区检测方法
                try:
                    import community as community_louvain
                    communities = community_louvain.best_partition(G)
                except Exception as louvain_error:
                    print(f"Louvain社区检测失败: {str(louvain_error)}，尝试使用备选方法...")
                    
                    # 备选方法1: 使用连通分量作为社区
                    connected_components = list(nx.connected_components(G))
                    communities = {}
                    for i, component in enumerate(connected_components):
                        for node in component:
                            communities[node] = i
                    
                    # 如果连通分量太少，使用度中心性聚类
                    if len(connected_components) <= 1:
                        print("图是完全连通的，使用度中心性进行社区划分...")
                        degree_dict = dict(G.degree())
                        sorted_nodes = sorted(degree_dict.items(), key=lambda x: x[1], reverse=True)
                        
                        # 将节点分为高度、中度和低度三组
                        n_nodes = len(sorted_nodes)
                        communities = {}
                        for i, (node, _) in enumerate(sorted_nodes):
                            if i < n_nodes // 3:
                                communities[node] = 0  # 高度中心性
                            elif i < 2 * (n_nodes // 3):
                                communities[node] = 1  # 中度中心性
                            else:
                                communities[node] = 2  # 低度中心性
                
                community_labels = np.array([communities[node] for node in G.nodes()])
                num_communities = len(set(communities.values()))
                
                print(f"社区检测完成，识别出{num_communities}个社区")
                
                # 更新数据标签
                data.y = torch.tensor(community_labels, dtype=torch.long)
                
                # 创建GAT模型
                gat_model = NeuronGAT(
                    in_channels=data.x.shape[1],  # 使用data.x.shape[1]而不是X_normalized.shape[1]
                    hidden_channels=self.config.analysis_params.get('gat_hidden_channels', 128),
                    out_channels=num_communities,
                    heads=self.config.analysis_params.get('gat_heads', 4),
                    dropout=self.config.analysis_params.get('gat_dropout', 0.3),
                    residual=self.config.analysis_params.get('gat_residual', True),
                    num_layers=self.config.analysis_params.get('gat_num_layers', 3),
                    alpha=self.config.analysis_params.get('gat_alpha', 0.2)
                )
                
                # 设置训练参数
                epochs = self.config.analysis_params.get('gnn_epochs', 200)
                lr = self.config.analysis_params.get('gnn_learning_rate', 0.008)
                weight_decay = self.config.analysis_params.get('gnn_weight_decay', 1e-3)
                patience = self.config.analysis_params.get('gnn_early_stop_patience', 20)
                early_stopping_enabled = self.config.analysis_params.get('early_stopping_enabled', False)
                
                # 训练模型
                trained_gat, gat_losses, gat_accuracies = train_gnn_model(
                    model=gat_model,
                    data=data,
                    epochs=epochs,
                    lr=lr,
                    weight_decay=weight_decay,
                    patience=patience,
                    device=gnn_analyzer.device,
                    early_stopping_enabled=early_stopping_enabled
                )
                
                # 绘制GAT训练曲线
                gat_loss_plot_path = self.config.gat_training_plot
                plot_gnn_results(gat_losses, gat_loss_plot_path)
                
                # 绘制GAT准确率曲线
                gat_acc_epochs = list(range(1, len(gat_accuracies['train'])+1))
                gat_accuracy_plot_path = visualizer.plot_training_metrics(
                    gat_acc_epochs, 
                    gat_accuracies['train'], 
                    gat_accuracies['val'], 
                    metric_name='Accuracy',
                    title='GAT Model Training Accuracy Change',
                    filename='gat_accuracy_curve.png'
                )
                
                # 评估模型
                trained_gat.eval()
                with torch.no_grad():
                    out = trained_gat(data)
                    _, predicted = torch.max(out, 1)
                    gat_accuracy = (predicted == data.y).sum().item() / len(data.y)
                
                print(f"GAT模块识别模型完成，准确率: {gat_accuracy:.4f}")
                
                # 保存结果
                gnn_results['gat_module_detection'] = {
                    'accuracy': gat_accuracy,
                    'loss_plot_path': gat_loss_plot_path,
                    'accuracy_plot_path': gat_accuracy_plot_path,
                    'num_communities': num_communities,
                    'train_acc_history': gat_accuracies['train'],
                    'val_acc_history': gat_accuracies['val']
                }
                
                # 基于GAT创建新的拓扑结构
                try:
                    print("\n基于GAT模型创建神经元拓扑结构...")
                    
                    # 使用GNN模型创建拓扑结构
                    gat_G = create_gnn_based_topology(
                        model=trained_gat,
                        data=data,
                        G=G.copy(),
                        node_names=[f"N{i+1}" for i in range(len(available_neurons))],
                        threshold=0.6 # 使用更高阈值突出模块结构
                    )
                    
                    # 获取节点嵌入和相似度矩阵
                    with torch.no_grad():
                        gat_embeddings = trained_gat.get_embeddings(data).detach().cpu().numpy()
                    gat_similarities = nx.adjacency_matrix(gat_G).todense()
                    
                    print(f"GAT拓扑结构创建完成: {gat_G.number_of_nodes()} 个节点, {gat_G.number_of_edges()} 条边")
                    
                    # # 可视化GAT拓扑结构
                    # gat_topo_path = self.config.gat_topology_png
                    # visualize_gnn_topology(
                    #     G=gat_G,
                    #     embeddings=gat_embeddings,
                    #     similarities=gat_similarities,
                    #     node_names=[f"N{i+1}" for i in range(len(available_neurons))],
                    #     output_path=gat_topo_path,
                    #     title="GAT-Based Module Detection Topology"
                    # )
                    
                    # 创建交互式可视化
                    create_interactive_gnn_topology(
                        G=gat_G,
                        embeddings=gat_embeddings,
                        similarities=gat_similarities.astype(float) if hasattr(gat_similarities, 'astype') else gat_similarities,
                        node_names=[f"N{i+1}" for i in range(len(available_neurons))],
                        output_path=self.config.gat_interactive_topology
                    )
                    
                    # 保存拓扑数据
                    save_gnn_topology_data(
                        G=gat_G,
                        embeddings=gat_embeddings,
                        similarities=gat_similarities,
                        node_names=[f"N{i+1}" for i in range(len(available_neurons))],
                        output_path=self.config.gat_topology_data
                    )
                    
                    # 使用visualize_gat_topology生成GAT静态拓扑图
                    gat_static_topo_path = visualize_gat_topology(self.config.gat_topology_data)
                    
                    # 分析GNN拓扑结构
                    topo_metrics = analyze_gnn_topology(gat_G, gat_similarities)
                    
                    # 保存结果
                    gnn_results['gat_topology'] = {
                        'visualization_path': gat_static_topo_path,  # 使用静态拓扑图路径替代原来的可视化路径
                        'interactive_path': self.config.gat_interactive_topology,
                        'data_path': self.config.gat_topology_data,
                        'metrics': topo_metrics,
                        'node_count': gat_G.number_of_nodes(),
                        'edge_count': gat_G.number_of_edges()
                    }
                except Exception as e:
                    import traceback
                    error_trace = traceback.format_exc()
                    print(f"无法进行社区检测，跳过GAT模型: {str(e)}")
                    print(f"详细错误信息:\n{error_trace}")
                    gnn_results['gat_topology'] = {'error': str(e)}
            
            except Exception as e:
                print(f"无法进行社区检测，跳过GAT模型: {str(e)}")
                gnn_results['gat_module_detection'] = {'error': str(e)}
            
        except Exception as e:
            print(f"GAT模块识别模型训练出错: {str(e)}")
            gnn_results['gat_module_detection'] = {'error': str(e)}
        
        # 3. 时间序列GNN分析
        print("\n准备时间序列GNN数据...")
        try:
            # 设置时间窗口参数
            window_size = self.config.analysis_params.get('temporal_window_size', 10)
            stride = self.config.analysis_params.get('temporal_stride', 5)
            
            # 准备时间序列数据
            temporal_data = gnn_analyzer.prepare_temporal_gnn_data(
                G, X_scaled, window_size=window_size, stride=stride
            )
            
            print(f"时间序列GNN数据准备完成，共 {len(temporal_data)} 个窗口")
            
            # 保存结果
            gnn_results['temporal_gnn'] = {
                'windows_count': len(temporal_data),
                'window_size': window_size,
                'stride': stride
            }
            
        except Exception as e:
            print(f"时间序列GNN数据准备出错: {str(e)}")
            gnn_results['temporal_gnn'] = {'error': str(e)}
        
        # 保存GNN分析结果到JSON文件
        gnn_results_file = self.config.gnn_analysis_results
        try:
            # 将不可序列化的对象转换为字符串
            serializable_results = convert_to_serializable(gnn_results)
            with open(gnn_results_file, 'w') as f:
                json.dump(serializable_results, f, indent=4)
            print(f"GNN分析结果已保存到 {gnn_results_file}")
            
            # 更新网络分析结果
            if hasattr(self, 'network_analysis_results') and self.network_analysis_results is not None:
                self.network_analysis_results['gnn_analysis'] = serializable_results
                
                # 保存更新后的网络分析结果
                network_results_path = os.path.join(self.config.analysis_dir, 'network_analysis_results.json')
                with open(network_results_path, 'w') as f:
                    json.dump(convert_to_serializable(self.network_analysis_results), f, indent=4)
                print(f"更新的网络分析结果（包含GNN分析）已保存到 {network_results_path}")
        except Exception as e:
            import traceback
            error_trace = traceback.format_exc()
            print(f"保存GNN分析结果时出错: {str(e)}")
            print(f"错误详情:\n{error_trace}")
        
        return gnn_results

def convert_to_serializable(obj):
    """
    将对象转换为可JSON序列化的格式
    
    参数:
        obj: 需要转换的对象
        
    返回:
        转换后的对象
    """
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (np.bool_, bool)):
        return bool(obj)
    elif isinstance(obj, (str, int, float, type(None))):
        return obj
    elif isinstance(obj, dict):
        return {convert_to_serializable(k): convert_to_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [convert_to_serializable(i) for i in obj]
    else:
        try:
            # 尝试转换为可序列化对象
            return str(obj)
        except:
            return f"不可序列化对象: {type(obj)}"

def set_all_random_seeds(seed=42):
    """
    设置所有相关库的随机种子，确保结果可重现
    """
    import random
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)
    
def check_data_compatibility(X_scaled, y=None, correlation_matrix=None, available_neurons=None):
    """
    检查数据兼容性，确保各种数据维度匹配
    
    参数:
        X_scaled: 标准化后的神经元活动数据，形状为(时间点数, 神经元数)
        y: 可选的行为标签
        correlation_matrix: 可选的相关性矩阵
        available_neurons: 可选的可用神经元列表
        
    返回:
        issues: 发现的问题列表
        fixed_data: 包含修复后数据的字典
    """
    issues = []
    fixed_data = {
        'X_scaled': X_scaled,
        'y': y,
        'correlation_matrix': correlation_matrix,
        'available_neurons': available_neurons
    }
    
    print("\n检查数据兼容性...")
    
    # 基本数据形状检查
    print(f"X_scaled形状: {X_scaled.shape}")
    if y is not None:
        print(f"标签数量: {len(y)}")
    if correlation_matrix is not None:
        print(f"相关性矩阵形状: {correlation_matrix.shape}")
    if available_neurons is not None:
        print(f"可用神经元数量: {len(available_neurons)}")
    
    # 检查1: X_scaled维度与神经元数量匹配
    if available_neurons is not None and X_scaled.shape[1] != len(available_neurons):
        issues.append(f"特征维度({X_scaled.shape[1]})与神经元数量({len(available_neurons)})不匹配")
        
        # 修复: 调整X_scaled或available_neurons
        if X_scaled.shape[1] > len(available_neurons):
            print(f"特征维度大于神经元数量，截断特征")
            fixed_data['X_scaled'] = X_scaled[:, :len(available_neurons)]
        else:
            print(f"特征维度小于神经元数量，截断神经元列表")
            fixed_data['available_neurons'] = available_neurons[:X_scaled.shape[1]]
    
    # 检查2: 相关性矩阵维度匹配
    if correlation_matrix is not None:
        if correlation_matrix.shape[0] != correlation_matrix.shape[1]:
            issues.append(f"相关性矩阵不是方阵: {correlation_matrix.shape}")
            
            # 修复: 创建方阵
            n = min(correlation_matrix.shape)
            fixed_data['correlation_matrix'] = correlation_matrix[:n, :n]
            print(f"截断相关性矩阵为{n}x{n}方阵")
        
        if available_neurons is not None and correlation_matrix.shape[0] != len(available_neurons):
            issues.append(f"相关性矩阵维度({correlation_matrix.shape[0]})与神经元数量({len(available_neurons)})不匹配")
            
            # 修复: 重新计算相关性矩阵
            print("重新计算相关性矩阵...")
            n_neurons = min(X_scaled.shape[1], len(available_neurons))
            neuron_activities = X_scaled[:, :n_neurons].T  # 转置为(神经元数, 时间点数)
            fixed_data['correlation_matrix'] = np.corrcoef(neuron_activities)
            fixed_data['available_neurons'] = available_neurons[:n_neurons]
            fixed_data['X_scaled'] = X_scaled[:, :n_neurons]
    
    # 检查3: 标签长度与时间点数匹配
    if y is not None and len(y) != X_scaled.shape[0]:
        issues.append(f"标签长度({len(y)})与时间点数({X_scaled.shape[0]})不匹配")
        
        # 修复: 调整标签长度
        if len(y) > X_scaled.shape[0]:
            print(f"标签太多，截断至{X_scaled.shape[0]}个")
            fixed_data['y'] = y[:X_scaled.shape[0]]
        else:
            print(f"标签太少，填充至{X_scaled.shape[0]}个")
            # 使用最后一个标签填充
            last_label = y[-1]
            padding = np.array([last_label] * (X_scaled.shape[0] - len(y)))
            fixed_data['y'] = np.concatenate([y, padding])
    
    # 检查4: 数据中是否有NaN或无穷值
    if np.isnan(X_scaled).any() or np.isinf(X_scaled).any():
        issues.append("X_scaled包含NaN或无穷值")
        
        # 修复: 替换为0
        print("替换X_scaled中的NaN和无穷值为0")
        fixed_data['X_scaled'] = np.nan_to_num(X_scaled)
    
    if correlation_matrix is not None and (np.isnan(correlation_matrix).any() or np.isinf(correlation_matrix).any()):
        issues.append("相关性矩阵包含NaN或无穷值")
        
        # 修复: 替换为0
        print("替换相关性矩阵中的NaN和无穷值为0")
        fixed_data['correlation_matrix'] = np.nan_to_num(correlation_matrix)
    
    # 报告结果
    if issues:
        print(f"发现{len(issues)}个数据兼容性问题:")
        for i, issue in enumerate(issues):
            print(f"  {i+1}. {issue}")
        print("已自动修复这些问题")
    else:
        print("数据兼容性检查通过，未发现问题")
    
    return issues, fixed_data

def main():
    """
    主函数：执行神经元网络行为分析流程
    """
    import argparse
    import sys
    import os
    
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='神经元网络行为分析工具')
    parser.add_argument('--data', type=str, help='数据文件路径')
    parser.add_argument('--no-auto-adapt', action='store_true', help='禁用自动参数调整')
    parser.add_argument('--correlation-threshold', type=float, help='神经元网络相关性阈值')
    parser.add_argument('--skip-gnn', action='store_true', help='跳过GNN分析')
    args = parser.parse_args()
    
    # 创建配置对象
    config = AnalysisConfig()
    
    # 如果指定了数据文件，更新配置
    if args.data:
        if not os.path.exists(args.data):
            print(f"错误: 指定的数据文件不存在: {args.data}")
            sys.exit(1)
        config.update_for_data_file(args.data)
    
    # 禁用自动适应功能
    if args.no_auto_adapt:
        config.auto_adapt_data = False
        print("已禁用自动参数调整功能")
    
    # 如果指定了相关性阈值，更新配置
    if args.correlation_threshold is not None:
        if not hasattr(config, 'analysis_params'):
            config.analysis_params = {}
        config.analysis_params['correlation_threshold'] = args.correlation_threshold
        print(f"已设置相关性阈值为: {args.correlation_threshold}")
    
    # 创建分析器对象
    analyzer = ResultAnalyzer(config)
    
    # 设置日志文件
    log_file = open(config.log_file, 'w')
    original_stdout = sys.stdout
    
    try:
        # 重定向标准输出到日志文件
        sys.stdout = StdoutTee(sys.stdout, log_file)
        
        # 检查GNN依赖项
        status, missing = check_gnn_dependencies()
        print(f"\n检查GNN依赖项...")
        if status:
            print("GNN依赖项检查通过，将启用GNN分析功能")
            USE_GNN = not args.skip_gnn
        else:
            print(f"GNN依赖项检查失败，将禁用GNN分析功能")
            print(f"缺失的依赖项: {', '.join(missing)}")
            USE_GNN = False
        
        if args.skip_gnn:
            print("用户指定跳过GNN分析")
        
        print(f"使用GNN: {'启用' if USE_GNN else '禁用'}")
        
        # 记录分析开始时间
        start_time = datetime.now()
        print(f"分析开始时间: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"数据标识符: {config.data_identifier}")
        print("="*50)
        
        # 加载模型和数据
        X, y, behavior_labels = analyzer.load_model_and_data()
        
        # 自动调整参数（如果启用）
        if config.auto_adapt_data:
            num_neurons = X.shape[1]
            num_samples = X.shape[0]
            config.auto_tune_parameters(num_neurons, num_samples)
        
        # 行为神经元相关性分析
        analyzer.analyze_behavior_neuron_correlation(X, y)
        
        # 时间模式分析
        analyzer.analyze_temporal_patterns(X, y)
        
        # 时间相关性分析
        analyzer.analyze_temporal_correlations(X, y)
        
        # 行为转换分析
        analyzer.analyze_behavior_transitions(y)
        
        # 识别关键神经元
        analyzer.identify_key_neurons(X, y)
        
        print("\n开始神经元网络拓扑分析...")
        
        # 构建神经元网络
        print("\n构建神经元功能连接网络...")
        correlation_threshold = config.analysis_params.get('correlation_threshold', 0.35)
        G, correlation_matrix, available_neurons = analyzer.build_neuron_network(X, threshold=correlation_threshold)
        print(f"网络构建完成: {len(G.nodes())} 个节点, {len(G.edges())} 条边")
        
        # 提取主要连接
        main_connections = {}
        
        # 使用threshold方法
        print("\n使用threshold方法提取主要连接线路...")
        G_threshold, threshold_output = analyzer.extract_main_connections(G, correlation_matrix, available_neurons, method='threshold', threshold=correlation_threshold)
        main_connections['threshold'] = threshold_output
        
        # 使用top_edges方法
        print("\n使用top_edges方法提取主要连接线路...")
        G_top_edges, top_edges_output = analyzer.extract_main_connections(G, correlation_matrix, available_neurons, method='top_edges', k=8)
        main_connections['top_edges'] = top_edges_output
        
        # 使用mst方法
        print("\n使用mst方法提取主要连接线路...")
        G_mst, mst_output = analyzer.extract_main_connections(G, correlation_matrix, available_neurons, method='mst')
        main_connections['mst'] = mst_output
        
        # 网络拓扑分析
        network_metrics = analyzer.analyze_network_topology(G)
        
        # 功能模块识别
        print("\n识别功能模块...")
        modules = analyzer.identify_functional_modules(G, correlation_matrix, available_neurons)
        
        # 可视化网络拓扑
        print("\n可视化网络拓扑分析结果...")
        analyzer.visualize_network_topology(G, network_metrics, modules)
        
        # 创建交互式可视化
        print("\n创建交互式神经元网络可视化...")
        analyzer.create_interactive_visualization(G, correlation_matrix, available_neurons)
        
        # 为主要连接创建交互式可视化
        for method in ['threshold', 'top_edges', 'mst']:
            analyzer.create_method_visualization(G, correlation_matrix, available_neurons, method)
        
        # 开始行为状态转换分析
        print("\n开始行为状态转换分析...")
        
        # 行为状态转换分析
        print("\n分析行为状态转换...")
        hmm_results = analyzer.analyze_behavior_state_transitions(X, y)
        
        # 整合结果
        network_analysis_results = {
            'network_metrics': network_metrics,
            'modules': modules,
            'main_connections': main_connections,
            'hmm_results': hmm_results
        }
        
        # GNN分析（如果启用）
        if USE_GNN:
            print("\n使用GNN进行神经元网络分析...")
            gnn_results = analyzer.analyze_network_with_gnn(G, correlation_matrix, X, y, available_neurons)
            network_analysis_results['gnn_analysis'] = gnn_results
        
        # 保存分析结果
        results_file = os.path.join(config.analysis_dir, 'network_analysis_results.json')
        with open(results_file, 'w') as f:
            json.dump(convert_to_serializable(network_analysis_results), f, indent=2)
        print(f"保存分析结果到: {results_file}")
        
        # 记录分析结束时间
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        print("="*50)
        print(f"分析结束时间: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"总耗时: {duration:.2f} 秒")
        print(f"分析完成！所有结果已保存。")
        
    except Exception as e:
        import traceback
        print(f"分析过程中出现错误: {str(e)}")
        print(traceback.format_exc())
    finally:
        # 恢复标准输出
        sys.stdout = original_stdout
        log_file.close()
        print(f"分析日志已保存到: {config.log_file}")

# 输出重定向类
class StdoutTee:
    def __init__(self, stdout, file):
        self.stdout = stdout
        self.file = file
        
    def write(self, data):
        self.stdout.write(data)
        self.file.write(data)
        
    def flush(self):
        self.stdout.flush()
        self.file.flush()

if __name__ == "__main__":
    main() 

