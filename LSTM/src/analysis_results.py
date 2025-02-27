import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler, LabelEncoder
import os
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

from kmeans_lstm_analysis import EnhancedNeuronLSTM, NeuronDataProcessor
from analysis_config import AnalysisConfig

import torch.nn.functional as F

# 添加安全的全局变量
add_safe_globals(['_reconstruct'])

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
        available_neurons = [f'n{i+1}' for i in range(n_neurons)]
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
        
        # 整理模块信息
        modules = {}
        for i in range(n_clusters):
            module_neurons = [available_neurons[j] for j in range(len(labels)) if labels[j] == i]
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
        
        # 绘制网络
        nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=100)
        nx.draw_networkx_edges(G, pos, alpha=0.2)
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
        Analyze behavior state transitions using HMM
        Args:
            X_scaled: Standardized neural activity data
            y: Behavior labels
        Returns:
            hmm_results: HMM analysis results dictionary
        """
        print("\nAnalyzing behavior state transitions...")
        
        # Initialize HMM model
        n_states = len(np.unique(y))
        model = hmm.GaussianHMM(
            n_components=n_states, 
            covariance_type="full",
            n_iter=100,
            random_state=42
        )
        
        # Train HMM model
        try:
            model.fit(X_scaled)
            
            # Predict hidden state sequence
            hidden_states = model.predict(X_scaled)
            
            # Calculate transition probability matrix
            transition_matrix = model.transmat_
            
            # Analyze state durations
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
            
            # Calculate average duration for each state
            avg_durations = {}
            for state in range(n_states):
                durations = [d for s, d in state_durations if s == state]
                avg_durations[state] = np.mean(durations) if durations else 0
            
            hmm_results = {
                'transition_matrix': transition_matrix,
                'hidden_states': hidden_states,
                'avg_durations': avg_durations,
                'model_score': model.score(X_scaled)
            }
            
            # Visualize HMM results
            self._visualize_hmm_results(hmm_results, self.behavior_labels)
            
            return hmm_results
            
        except Exception as e:
            print(f"HMM analysis error: {str(e)}")
            return None
    
    def analyze_neuron_state_relationships(self, X_scaled, hidden_states):
        """
        Analyze relationships between neuron activity patterns and state transitions
        Args:
            X_scaled: Standardized neural activity data
            hidden_states: Hidden state sequence predicted by HMM
        Returns:
            relationships: Neuron-state relationship analysis results
        """
        print("\nAnalyzing neuron activity and state transition relationships...")
        
        n_neurons = X_scaled.shape[1]
        n_states = len(np.unique(hidden_states))
        
        # Calculate neuron activity features for each state
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
        Predict key time points of state transitions
        Args:
            X_scaled: Standardized neural activity data
            hidden_states: Hidden state sequence predicted by HMM
        Returns:
            transition_points: Predicted transition point information
        """
        print("\nPredicting state transition points...")
        
        # Calculate overall change rate of neural activity
        activity_derivative = np.diff(X_scaled, axis=0)
        activity_change = np.sum(np.abs(activity_derivative), axis=1)
        
        # Standardize change rate
        activity_change = zscore(activity_change)
        
        # Detect peaks as potential transition points
        peaks, properties = find_peaks(activity_change, 
                                     height=1.5,  # Only consider significant peaks
                                     distance=10)  # Minimum distance between peaks
        
        # Get actual state transition points
        true_transitions = np.where(np.diff(hidden_states) != 0)[0]
        
        # Analyze predicted transition points
        transition_points = {
            'predicted': peaks,
            'actual': true_transitions,
            'activity_change': activity_change,
            'prediction_scores': properties['peak_heights']
        }
        
        # Evaluate prediction accuracy
        prediction_accuracy = self._evaluate_transition_predictions(
            peaks, true_transitions, tolerance=5
        )
        transition_points['accuracy'] = prediction_accuracy
        
        # Visualize transition point prediction results
        self._visualize_transition_points(transition_points, X_scaled)
        
        return transition_points
    
    def _analyze_transition_patterns(self, X_scaled, hidden_states):
        """
        Analyze neuron activity patterns during state transitions
        Args:
            X_scaled: Standardized neural activity data
            hidden_states: Hidden state sequence
        Returns:
            patterns: List of transition pattern information
        """
        transitions = np.where(np.diff(hidden_states) != 0)[0]
        window_size = 5  # Time window before and after transition
        
        patterns = []
        for t in transitions:
            if t >= window_size and t < len(X_scaled) - window_size:
                before_transition = X_scaled[t-window_size:t]
                after_transition = X_scaled[t+1:t+window_size+1]
                
                patterns.append({
                    'time_point': t,
                    'from_state': hidden_states[t],
                    'tostate': hidden_states[t+1],
                    'before_pattern': np.mean(before_transition, axis=0),
                    'after_pattern': np.mean(after_transition, axis=0),
                    'change_magnitude': np.mean(np.abs(after_transition - before_transition))
                })
        
        return patterns
    
    def _evaluate_transition_predictions(self, predicted, actual, tolerance):
        """Evaluate prediction accuracy"""
        correct_predictions = 0
        
        for pred in predicted:
            # Check if predicted point is within tolerance range of actual transition points
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
        """Visualize HMM analysis results"""
        # 1. State transition probability matrix heatmap
        plt.figure(figsize=(12, 5))
        
        plt.subplot(1, 2, 1)
        sns.heatmap(hmm_results['transition_matrix'], 
                   annot=True, 
                   fmt='.2f',
                   xticklabels=behavior_labels,
                   yticklabels=behavior_labels)
        plt.title('State Transition Probability Matrix')
        
        # 2. State duration distribution
        plt.subplot(1, 2, 2)
        durations = list(hmm_results['avg_durations'].values())
        plt.bar(range(len(durations)), durations)
        plt.xticks(range(len(durations)), behavior_labels)
        plt.title('Average State Duration')
        plt.ylabel('Time Steps')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.config.analysis_dir, 'hmm_analysis.png'),
                   dpi=300, bbox_inches='tight')
        plt.close()
    
    def _visualize_neuron_state_relationships(self, relationships, n_states):
        """Visualize neuron-state relationship analysis results"""
        plt.figure(figsize=(15, 10))
        
        # Plot characteristic neuron activity patterns
        for state in range(n_states):
            plt.subplot(n_states, 1, state+1)
            neurons = relationships['characteristic_neurons'][state]['neuron_indices']
            scores = relationships['characteristic_neurons'][state]['significance_scores']
            
            plt.bar(neurons, scores)
            plt.title(f'Characteristic Neurons for State {state}')
            plt.xlabel('Neuron ID')
            plt.ylabel('Significance Score')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.config.analysis_dir, 'neuron_state_relationships.png'),
                   dpi=300, bbox_inches='tight')
        plt.close()
    
    def _visualize_transition_points(self, transition_points, X_scaled):
        """Visualize state transition point prediction results"""
        plt.figure(figsize=(15, 8))
        
        # Plot neural activity change rate and predicted transition points
        plt.plot(transition_points['activity_change'], 'b-', label='Activity Change Rate')
        plt.plot(transition_points['predicted'], 
                transition_points['activity_change'][transition_points['predicted']], 
                'ro', label='Predicted Transitions')
        plt.plot(transition_points['actual'],
                transition_points['activity_change'][transition_points['actual']],
                'go', label='Actual Transitions')
        
        plt.title('State Transition Point Prediction')
        plt.xlabel('Time Steps')
        plt.ylabel('Neural Activity Change Rate')
        plt.legend()
        
        # Add prediction accuracy information
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

def convert_to_serializable(obj):
    """
    将对象转换为可JSON序列化的格式
    
    参数：
        obj: 需要转换的对象
    返回：
        转换后的可序列化对象
    """
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {k: convert_to_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [convert_to_serializable(item) for item in obj]
    elif isinstance(obj, (np.int64, np.int32, np.int16, np.int8)):
        return int(obj)
    elif isinstance(obj, (np.float64, np.float32, np.float16)):
        return float(obj)
    elif isinstance(obj, (set, frozenset)):
        return list(obj)
    return obj

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
        
        print("\n开始神经元网络拓扑分析...")
        # 构建神经元功能连接网络
        G, correlation_matrix, available_neurons = analyzer.build_neuron_network(
            X_scaled, 
            threshold=config.analysis_params['correlation_threshold']
        )
        
        # 分析网络拓扑特征
        topology_metrics = analyzer.analyze_network_topology(G)
        
        # 识别功能模块
        functional_modules = analyzer.identify_functional_modules(G, correlation_matrix, available_neurons)
        
        # 可视化分析结果
        analyzer.visualize_network_topology(G, topology_metrics, functional_modules)
        
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
                    'hmm_results': hmm_results,
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
        with open(results_path, 'w', encoding='utf-8') as f:
            serializable_results = convert_to_serializable(results)
            json.dump(serializable_results, f, indent=4, ensure_ascii=False)
            
        print("分析完成！所有结果已保存。")
        
    except Exception as e:
        print(f"分析过程中出现错误: {str(e)}")
        raise

if __name__ == "__main__":
    main() 