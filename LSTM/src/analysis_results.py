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
                             analyze_gnn_topology)
    from gnn_visualization import GNNVisualizer
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
            model: 训练好的LSTM模型
            X_scaled: 标准化后的神经元数据
            y: 行为标签
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
        
        # 加载训练好的模型
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
        
        try:
            # 尝试使用 weights_only=True 加载模型
            checkpoint = torch.load(self.config.model_path, weights_only=True)
            model.load_state_dict(checkpoint['model_state_dict'])
        except Exception as e1:
            try:
                print("尝试使用 weights_only=False 加载模型...")
                checkpoint = torch.load(self.config.model_path, weights_only=False)
                model.load_state_dict(checkpoint['model_state_dict'])
            except Exception as e2:
                raise RuntimeError(f"加载模型失败。错误1: {str(e1)}\n错误2: {str(e2)}")
        
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
            top_neurons = sorted_indices[:self.config.analysis_params['top_neurons_count']]
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

    def build_neuron_network(self, X_scaled, threshold=0.3):
        """
        构建神经元功能连接网络
        参数:
            X_scaled: 标准化后的神经元活动数据
            threshold: 相关性阈值
        返回:
            G: NetworkX图对象
            correlation_matrix: 相关性矩阵
            available_neurons: 可用神经元列表
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

    def extract_main_connections(self, G, correlation_matrix, available_neurons, method='top_edges', **kwargs):
        """
        从完整网络中提取主要连接线路
        
        参数:
            G: 原始网络图
            correlation_matrix: 相关性矩阵
            available_neurons: 可用神经元列表
            method: 提取方法，可选 'threshold', 'top_edges', 'mst', 'backbone'
            **kwargs: 其他参数
                - threshold: 更高的相关性阈值（用于'threshold'方法）
                - top_k: 每个节点保留的最强连接数（用于'top_edges'方法）
                - alpha: 显著性水平（用于'backbone'方法）
                
        返回:
            main_G: 提取出的主要连接网络
        """
        print(f"\n正在使用'{method}'方法提取主要连接线路...")
        
        # 复制原图以避免修改原始数据
        main_G = nx.Graph()
        
        # 添加所有节点
        for node in G.nodes():
            main_G.add_node(node)
        
        if method == 'threshold':
            # 使用更高的阈值过滤边
            higher_threshold = kwargs.get('threshold', 0.6)  # 默认提高到0.6
            print(f"使用更高的相关性阈值: {higher_threshold}")
            
            for u, v, data in G.edges(data=True):
                if data['weight'] >= higher_threshold:
                    main_G.add_edge(u, v, weight=data['weight'])
                    
        elif method == 'top_edges':
            # 为每个节点只保留top_k个最强连接
            top_k = kwargs.get('top_k', 5)  # 默认每个节点保留3个最强连接
            print(f"为每个节点保留{top_k}个最强连接")
            
            for node in G.nodes():
                edges = [(node, neighbor, G[node][neighbor]['weight']) 
                        for neighbor in G.neighbors(node)]
                
                if edges:
                    # 按权重降序排序
                    edges.sort(key=lambda x: x[2], reverse=True)
                    # 只保留前top_k个
                    for u, v, weight in edges[:top_k]:
                        if not main_G.has_edge(u, v):  # 避免重复添加
                            main_G.add_edge(u, v, weight=weight)
                            
        elif method == 'mst':
            # 使用最小生成树提取骨架网络
            print("使用最小生成树算法提取网络骨架")
            
            # 创建边权重为距离（1-相关性）的图用于最小生成树
            mst_graph = nx.Graph()
            for node in G.nodes():
                mst_graph.add_node(node)
                
            for u, v, data in G.edges(data=True):
                # 将权重转换为距离（权重越大距离越小）
                distance = 1.0 - data['weight']
                mst_graph.add_edge(u, v, weight=distance)
            
            # 计算最小生成树
            mst_edges = nx.minimum_spanning_edges(mst_graph, data=True)
            
            # 将原始权重添加回主图
            for u, v, mst_data in mst_edges:
                orig_weight = G[u][v]['weight'] if G.has_edge(u, v) else 0
                main_G.add_edge(u, v, weight=orig_weight)
                
        elif method == 'backbone':
            # 使用显著性过滤提取骨架网络
            alpha = kwargs.get('alpha', 0.05)  # 默认显著性水平0.05
            print(f"使用显著性骨架网络提取算法 (alpha={alpha})")
            
            try:
                import networkx.algorithms.community as nxcom
                
                # 计算节点的度
                degrees = dict(G.degree())
                
                # 对每条边进行显著性测试
                for u, v, data in G.edges(data=True):
                    weight = data['weight']
                    k_u = degrees[u]
                    k_v = degrees[v]
                    
                    # 计算边权重的显著性
                    if k_u > 1 and k_v > 1:  # 避免度为1的节点导致的除零错误
                        p_ij = weight / (k_u * k_v)
                        if p_ij < alpha:  # 如果显著，则保留该边
                            main_G.add_edge(u, v, weight=weight)
            except ImportError:
                print("警告: 无法导入社区检测算法，使用度中心性过滤作为替代")
                # 使用度中心性作为替代
                centrality = nx.degree_centrality(G)
                centrality_threshold = np.percentile(list(centrality.values()), 70)  # 保留前30%的中心节点
                
                for u, v, data in G.edges(data=True):
                    if centrality[u] >= centrality_threshold and centrality[v] >= centrality_threshold:
                        main_G.add_edge(u, v, weight=data['weight'])
        else:
            raise ValueError(f"未知的提取方法: {method}")
            
        print(f"主要连接线路提取完成: {len(main_G.nodes())} 个节点, {len(main_G.edges())} 条边")
        return main_G
        
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
        degree_values = list(metrics['degree_centrality'].values())
        betweenness_values = list(metrics['betweenness_centrality'].values())
        
        plt.subplot(1, 2, 1)
        plt.hist(degree_values, bins=20)
        plt.title('Degree Centrality Distribution')
        plt.xlabel('Degree Centrality')
        plt.ylabel('Count')
        
        plt.subplot(1, 2, 2)
        plt.hist(betweenness_values, bins=20)
        plt.title('Betweenness Centrality Distribution')
        plt.xlabel('Betweenness Centrality')
        plt.ylabel('Count')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.config.analysis_dir, 'centrality_metrics.png'),
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        # 3. 模块分析结果可视化
        plt.figure(figsize=(10, 6))
        module_sizes = [m['size'] for m in modules.values()]
        module_densities = [m['internal_density'] for m in modules.values()]
        
        plt.bar(range(len(modules)), module_densities)
        plt.title('Module Internal Density')
        plt.xlabel('Module ID')
        plt.ylabel('Density')
        plt.xticks(range(len(modules)), [f'M{i+1}' for i in range(len(modules))])
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.config.analysis_dir, 'module_analysis.png'),
                   dpi=300, bbox_inches='tight')
        plt.close()

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
            
            n_states = 2  # 固定使用2个状态
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
        使用GNN分析神经元网络
        
        参数:
            G: NetworkX图对象
            correlation_matrix: 相关性矩阵
            X_scaled: 标准化的神经元活动数据
            y: 行为标签
            available_neurons: 可用神经元列表
            
        返回:
            gnn_results: GNN分析结果字典
        """
        print("\n使用GNN进行神经元网络分析...")
        
        # 创建GNN分析器
        gnn_analyzer = GNNAnalyzer(self.config)
        
        # 特征归一化 - 确保特征在合适的范围内
        from sklearn.preprocessing import StandardScaler
        X_normalized = StandardScaler().fit_transform(X_scaled)
        
        # 转换为GNN格式
        print("\n开始使用GNN进行神经元网络分析...")
        print(f"GNN分析器将使用设备: {gnn_analyzer.device}")
        print("将神经元网络转换为GNN数据格式...")
        data = gnn_analyzer.convert_network_to_gnn_format(G, X_normalized, y)
        print(f"GNN数据转换完成: {data}")
        
        # GNN分析结果字典
        gnn_results = {}
        
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
            epochs = self.config.analysis_params.get('gnn_epochs', 100)
            lr = self.config.analysis_params.get('gnn_learning_rate', 0.008)
            weight_decay = self.config.analysis_params.get('gnn_weight_decay', 1e-3)
            patience = self.config.analysis_params.get('gnn_early_stop_patience', 20)
            early_stopping_enabled = self.config.analysis_params.get('early_stopping_enabled', False)
            
            # 训练模型
            trained_model, losses = train_gnn_model(
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
            
            # 创建交互式可视化
            create_interactive_gnn_topology(
                G=G_gnn,
                embeddings=trained_model.get_embeddings(data).detach().cpu().numpy(),
                similarities=nx.adjacency_matrix(G_gnn).todense().astype(float),
                node_names=[f"N{i+1}" for i in range(len(available_neurons))],
                output_path=self.config.gcn_interactive_topology
            )
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
                communities = community_louvain.best_partition(G)
                community_labels = np.array([communities[node] for node in G.nodes()])
                num_communities = len(set(communities.values()))
                
                # 更新数据标签
                data.y = torch.tensor(community_labels, dtype=torch.long)
                
                # 创建GAT模型
                gat_model = NeuronGAT(
                    in_channels=X_normalized.shape[1],
                    hidden_channels=self.config.analysis_params.get('gat_hidden_channels', 56),
                    out_channels=num_communities,
                    heads=self.config.analysis_params.get('gat_heads', 4),
                    dropout=self.config.analysis_params.get('gat_dropout', 0.3)
                )
                
                # 设置训练参数
                epochs = self.config.analysis_params.get('gnn_epochs', 100)
                lr = self.config.analysis_params.get('gnn_learning_rate', 0.008)
                weight_decay = self.config.analysis_params.get('gnn_weight_decay', 1e-3)
                patience = self.config.analysis_params.get('gnn_early_stop_patience', 20)
                early_stopping_enabled = self.config.analysis_params.get('early_stopping_enabled', False)
                
                # 训练模型
                trained_gat, gat_losses = train_gnn_model(
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
                    'num_communities': num_communities
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
                        threshold=0.7  # 使用更高阈值突出模块结构
                    )
                    
                    # 获取节点嵌入和相似度矩阵
                    with torch.no_grad():
                        gat_embeddings = trained_gat.get_embeddings(data).detach().cpu().numpy()
                    gat_similarities = nx.adjacency_matrix(gat_G).todense()
                    
                    print(f"GAT拓扑结构创建完成: {gat_G.number_of_nodes()} 个节点, {gat_G.number_of_edges()} 条边")
                    
                    # 可视化GAT拓扑结构
                    gat_topo_path = self.config.gat_topology_png
                    visualize_gnn_topology(
                        G=gat_G,
                        embeddings=gat_embeddings,
                        similarities=gat_similarities,
                        node_names=[f"N{i+1}" for i in range(len(available_neurons))],
                        output_path=gat_topo_path,
                        title="GAT-Based Module Detection Topology"
                    )
                    
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
                    
                    # 分析GNN拓扑结构
                    topo_metrics = analyze_gnn_topology(gat_G, gat_similarities)
                    
                    # 保存结果
                    gnn_results['gat_topology'] = {
                        'visualization_path': gat_topo_path,
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
        
        # 创建StringIO对象以捕获所有输出
        output_buffer = io.StringIO()
        
        # 使用redirect_stdout重定向标准输出
        with contextlib.redirect_stdout(output_buffer):
            # 检查GNN依赖项
            print("\n检查GNN依赖项...")
            gnn_ok, missing_deps = check_gnn_dependencies()
            if gnn_ok:
                print("GNN依赖项检查通过，将启用GNN分析功能")
                if hasattr(config, 'use_gnn'):
                    print(f"使用GNN: {'启用' if config.use_gnn else '禁用'}")
                else:
                    print("配置中未找到use_gnn属性，将默认启用GNN")
                    config.use_gnn = True
            else:
                print(f"GNN依赖项检查失败，以下组件缺失: {', '.join(missing_deps)}")
                print("将禁用GNN分析功能")
                config.use_gnn = False
                print("""
如需启用GNN功能，请安装以下依赖项:
pip install torch-geometric torch-scatter torch-sparse

注意: torch-scatter和torch-sparse可能需要根据您的CUDA版本安装特定版本
详情请参考: https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html
""")
                
            # 记录分析开始时间
            start_time = datetime.now()
            print(f"分析开始时间: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
            print(f"数据标识符: {config.data_identifier}")
            print("=" * 50)
            
            # Initialize analyzer
            analyzer = ResultAnalyzer(config)
            
            # 加载模型和数据
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
            
            print("\n开始神经元网络拓扑分析...")
            # 构建神经元功能连接网络
            G, correlation_matrix, available_neurons = analyzer.build_neuron_network(
                X_scaled, 
                threshold=config.analysis_params['correlation_threshold']
            )
            
            # 提取主要连接线路
            main_methods = ['threshold', 'top_edges', 'mst']
            for method in main_methods:
                print(f"\n使用{method}方法提取主要连接线路...")
                if method == 'threshold':
                    main_G = analyzer.extract_main_connections(
                        G, correlation_matrix, available_neurons,
                        method=method, threshold=0.6
                    )
                elif method == 'top_edges':
                    main_G = analyzer.extract_main_connections(
                        G, correlation_matrix, available_neurons,
                        method=method, top_k=3
                    )
                elif method == 'mst':
                    main_G = analyzer.extract_main_connections(
                        G, correlation_matrix, available_neurons,
                        method=method
                    )
                
                # 导出为JSON文件
                output_file = os.path.join(config.analysis_dir, f'neuron_network_main_{method}.json')
                analyzer.export_network_to_json(main_G, output_file)
                print(f"{method}方法的主要连接线路已导出到: {output_file}")
                
                # 可视化主要连接线路
                plt.figure(figsize=(12, 12))
                pos = nx.spring_layout(main_G, k=1/np.sqrt(len(main_G.nodes())), iterations=50)
                node_sizes = [300 for _ in main_G.nodes()]
                edge_weights = [main_G[u][v]['weight'] * 5 for u, v in main_G.edges()]
                
                nx.draw_networkx(
                    main_G, pos,
                    with_labels=True,
                    node_size=node_sizes,
                    node_color='skyblue',
                    font_size=10,
                    width=edge_weights,
                    edge_color='gray',
                    alpha=0.8
                )
                
                plt.title(f'Neuron Connection Network (Method: {method})', fontsize=16)
                plt.axis('off')
                plt.tight_layout()
                plt.savefig(os.path.join(config.analysis_dir, f'neuron_network_main_{method}.png'), 
                           dpi=300, bbox_inches='tight')
                plt.close()
            
            # 分析网络拓扑特征
            topology_metrics = analyzer.analyze_network_topology(G)
            
            # 识别功能模块
            functional_modules = analyzer.identify_functional_modules(G, correlation_matrix, available_neurons)
            
            # 可视化分析结果
            analyzer.visualize_network_topology(G, topology_metrics, functional_modules)
            
            # 生成交互式神经元网络可视化
            try:
                from visualization import VisualizationManager
                visualizer = VisualizationManager(config)
                interactive_path = visualizer.plot_interactive_neuron_network(G, topology_metrics, functional_modules)
                if interactive_path:
                    print(f"生成交互式神经元网络可视化完成。结果保存在: {interactive_path}")
            except Exception as e:
                print(f"生成交互式神经元网络可视化时出错: {str(e)}")
            
            # 尝试为主要连接也生成交互式可视化
            try:
                from visualization import VisualizationManager
                visualizer = VisualizationManager(config)
                for method in main_methods:
                    main_G = analyzer.extract_main_connections(
                        G, correlation_matrix, available_neurons,
                        method=method, threshold=0.6 if method == 'threshold' else None,
                        top_k=3 if method == 'top_edges' else None
                    )
                    interactive_path = visualizer.plot_interactive_neuron_network(
                        main_G, 
                        topology_metrics, 
                        functional_modules, 
                        output_path=config.gnn_interactive_template.format(method)
                    )
                    if interactive_path:
                        print(f"生成主要连接({method})交互式可视化完成。结果保存在: {interactive_path}")
            except Exception as e:
                print(f"生成主要连接交互式可视化时出错: {str(e)}")
            
            # 执行行为状态转换分析
            print("\n开始行为状态转换分析...")
            
            # 1. HMM分析
            hmm_results = analyzer.analyze_behavior_state_transitions(X_scaled, y)
            if hmm_results is not None:
                # 2. 分析神经元与状态转换的关系
                relationships = analyzer.analyze_neuron_state_relationships(
                    X_scaled, 
                    hmm_results['hidden_states']
                )
                
                # 3. 预测状态转换点
                transition_points = analyzer.predict_transition_points(
                    X_scaled,
                    hmm_results['hidden_states']
                )
                
                # 将状态转换分析结果添加到总结果中
                results = {
                    'topology_metrics': topology_metrics,
                    'functional_modules': functional_modules,
                    'state_transitions': {
                        'hmm_results': {
                            'transition_matrix': hmm_results['transition_matrix'],
                            'avg_durations': hmm_results['avg_durations'],
                            'model_score': float(hmm_results['model_score']),
                            'n_states': hmm_results['n_states'],
                            'covariance_type': hmm_results['covariance_type'],
                            'pca_components': hmm_results['pca_components'],
                            'pca_explained_variance': hmm_results['pca_explained_variance'],
                            'mapping_probs': hmm_results['mapping_probs'].tolist() if 'mapping_probs' in hmm_results else None
                        },
                        'neuron_state_relationships': relationships,
                        'transition_points': transition_points
                    }
                }
            else:
                results = {
                    'topology_metrics': topology_metrics,
                    'functional_modules': functional_modules
                }
            
            # 将结果保存为JSON文件
            results_path = os.path.join(config.analysis_dir, 'network_analysis_results.json')
            print(f"\n保存分析结果到: {results_path}")
            
            # 添加网络图对象到结果中
            if 'topology_metrics' in results:
                results['topology_metrics']['graph'] = G
                
            with open(results_path, 'w', encoding='utf-8') as f:
                serializable_results = convert_to_serializable(results)
                json.dump(serializable_results, f, indent=4, ensure_ascii=False)
            
            # 记录分析结束时间
            end_time = datetime.now()
            duration = end_time - start_time
            print(f"=" * 50)
            print(f"分析结束时间: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
            print(f"总耗时: {duration.total_seconds():.2f} 秒")
            print("分析完成！所有结果已保存。")
        
        # 将捕获的输出保存到日志文件
        log_content = output_buffer.getvalue()
        
        # 同时打印到控制台
        print(log_content)
        
        # 保存日志到文件
        with open(config.log_file, 'w', encoding='utf-8') as log_file:
            log_file.write(log_content)
            
        print(f"分析日志已保存到: {config.log_file}")
            
        # 在现有网络分析后添加GNN分析
        if hasattr(config, 'use_gnn') and config.use_gnn:
            print("\n使用GNN进行神经元网络分析...")
            gnn_results = analyzer.analyze_network_with_gnn(
                G, correlation_matrix, X_scaled, y, available_neurons
            )
            
            # 将GNN结果添加到分析结果中
            if gnn_results:
                # 读取已保存的分析结果
                try:
                    with open(config.network_analysis_file, 'r') as f:
                        results = json.load(f)
                    
                    # 添加GNN分析结果
                    results['gnn_analysis'] = gnn_results
                    
                    # 更新网络分析结果文件
                    with open(config.network_analysis_file, 'w') as f:
                        # 使用convert_to_serializable函数确保所有数据都可JSON序列化
                        serializable_results = convert_to_serializable(results)
                        json.dump(serializable_results, f, indent=4)
                    print(f"更新的网络分析结果（包含GNN分析）已保存到 {config.network_analysis_file}")
                except Exception as e:
                    print(f"更新网络分析结果文件时出错: {str(e)}")
        
    except Exception as e:
        error_message = f"分析过程中出现错误: {str(e)}"
        print(error_message)
        
        # 尝试将错误也写入日志文件
        try:
            with open(config.log_file, 'w', encoding='utf-8') as log_file:
                log_file.write(f"分析开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                log_file.write(f"错误信息: {error_message}\n")
                import traceback
                log_file.write(f"详细错误信息:\n{traceback.format_exc()}")
        except:
            pass
            
        raise

if __name__ == "__main__":
    main() 

