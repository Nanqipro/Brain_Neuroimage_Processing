import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import networkx as nx
from matplotlib.colors import LinearSegmentedColormap
import os

class VisualizationManager:
    """可视化管理器类,用于生成各种数据分析的可视化图表"""
    def __init__(self, config):
        self.config = config
        # 设置 seaborn 样式
        sns.set_style("whitegrid")
        # 设置 matplotlib 的基本样式
        plt.style.use('default')
        
    def set_plot_style(self):
        """设置全局绘图样式"""
        plt.rcParams['font.size'] = self.config.visualization_params['font_size']
        plt.rcParams['lines.linewidth'] = self.config.visualization_params['line_width']
        
    def plot_behavior_neuron_correlation(self, behavior_activity_df):
        """绘制行为和神经元之间的相关性热图"""
        plt.figure(figsize=self.config.visualization_params['figure_sizes']['correlation'])
        sns.heatmap(behavior_activity_df, 
                   cmap=self.config.visualization_params['colormaps']['correlation'],
                   center=0,
                   annot=True,
                   fmt='.2f',
                   cbar_kws={'label': 'Correlation Coefficient'})
        plt.title('Mean Neuron Activity by Behavior')
        plt.xlabel('Neurons')
        plt.ylabel('Behaviors')
        plt.tight_layout()
        plt.savefig(self.config.correlation_plot, 
                   dpi=self.config.visualization_params['dpi'])
        plt.close()
    
    def plot_temporal_patterns(self, X_scaled, y, behavior_labels):
        """绘制每种行为的时间模式图
        
        参数:
            X_scaled: 标准化后的神经元数据
            y: 行为标签
            behavior_labels: 行为标签列表
        """
        window_size = self.config.analysis_params['temporal_window_size']
        
        for behavior_idx, behavior in enumerate(behavior_labels):
            behavior_mask = (y == behavior_idx)
            behavior_data = X_scaled[behavior_mask]
            
            if len(behavior_data) > window_size:
                # 计算移动平均
                rolling_mean = np.array([
                    np.mean(behavior_data[i:i+window_size], axis=0)
                    for i in range(0, len(behavior_data)-window_size, window_size)
                ])
                
                plt.figure(figsize=self.config.visualization_params['figure_sizes']['temporal'])
                for neuron in range(min(5, rolling_mean.shape[1])):
                    plt.plot(rolling_mean[:, neuron], 
                            label=f'Neuron {neuron+1}',
                            linewidth=self.config.visualization_params['line_width'])
                
                plt.title(f'Temporal Pattern of Neuron Activity During {behavior}')
                plt.xlabel('Time Windows')
                plt.ylabel('Standardized Activity')
                plt.legend()
                plt.grid(True, alpha=0.3)
                plt.tight_layout()
                plt.savefig(self.config.get_temporal_pattern_path(behavior),
                          dpi=self.config.visualization_params['dpi'])
                plt.close()
    
    def plot_behavior_transitions(self, transitions, behavior_labels):
        """绘制行为转换矩阵热图
        
        参数:
            transitions: 行为转换概率矩阵
            behavior_labels: 行为标签列表
        """
        plt.figure(figsize=self.config.visualization_params['figure_sizes']['transitions'])
        sns.heatmap(transitions,
                   xticklabels=behavior_labels,
                   yticklabels=behavior_labels,
                   cmap=self.config.visualization_params['colormaps']['transitions'],
                   annot=True,
                   fmt='.2f',
                   cbar_kws={'label': 'Transition Probability'})
        plt.title('Behavior Transition Probabilities')
        plt.xlabel('To Behavior')
        plt.ylabel('From Behavior')
        plt.tight_layout()
        plt.savefig(self.config.transition_plot,
                   dpi=self.config.visualization_params['dpi'])
        plt.close()
    
    def plot_neuron_network(self, behavior_importance):
        """绘制行为-神经元网络关系图
        
        参数:
            behavior_importance: 包含行为重要性数据的字典
        """
        G = nx.Graph()
        
        # 添加节点和边
        for behavior, data in behavior_importance.items():
            G.add_node(behavior, node_type='behavior')
            for neuron, effect in zip(data['significant_neurons'], data['effect_sizes'][data['significant_neurons']]):
                neuron_name = f'N{neuron+1}'
                G.add_node(neuron_name, node_type='neuron')
                if effect > self.config.analysis_params['neuron_significance_threshold']:
                    G.add_edge(behavior, neuron_name, weight=effect)
        
        # 创建布局
        pos = nx.spring_layout(G, k=1, iterations=50)
        
        # 绘制图形
        plt.figure(figsize=self.config.visualization_params['figure_sizes']['network'])
        
        # 绘制节点
        behavior_nodes = [node for node in G.nodes() if G.nodes[node]['node_type'] == 'behavior']
        neuron_nodes = [node for node in G.nodes() if G.nodes[node]['node_type'] == 'neuron']
        
        nx.draw_networkx_nodes(G, pos, nodelist=behavior_nodes, 
                             node_color='lightblue', node_size=2000, alpha=0.7)
        nx.draw_networkx_nodes(G, pos, nodelist=neuron_nodes,
                             node_color='lightgreen', node_size=1500, alpha=0.7)
        
        # 根据权重绘制边
        edges = G.edges(data=True)
        weights = [d['weight'] for (u, v, d) in edges]
        nx.draw_networkx_edges(G, pos, width=weights, alpha=0.5)
        
        # 添加标签
        nx.draw_networkx_labels(G, pos, font_size=8)
        
        plt.title('Behavior-Neuron Interaction Network')
        plt.axis('off')
        plt.tight_layout()
        plt.savefig(self.config.network_plot,
                   dpi=self.config.visualization_params['dpi'])
        plt.close()
    
    def plot_temporal_correlations(self, correlations):
        """绘制不同时间窗口大小的时间相关性图
        
        参数:
            correlations: 包含不同窗口大小相关性值的字典
        """
        for window, corr_values in correlations.items():
            plt.figure(figsize=self.config.visualization_params['figure_sizes']['temporal'])
            plt.plot(corr_values, 
                    linewidth=self.config.visualization_params['line_width'])
            plt.title(f'Temporal Correlations (Window Size: {window})')
            plt.xlabel('Time')
            plt.ylabel('Mean Correlation')
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(self.config.get_temporal_correlation_path(window),
                       dpi=self.config.visualization_params['dpi'])
            plt.close()
    
    def plot_statistical_summary(self, f_values, p_values, effect_sizes):
        """绘制统计分析总结图
        
        参数:
            f_values: F检验值列表
            p_values: P值列表
            effect_sizes: 效应量字典
        """
        # 创建总结图
        plt.figure(figsize=(15, 10))
        
        # 图1: F值分布
        plt.subplot(2, 2, 1)
        plt.hist(f_values, bins=30)
        plt.title('Distribution of F-values')
        plt.xlabel('F-value')
        plt.ylabel('Count')
        
        # 图2: P值分布
        plt.subplot(2, 2, 2)
        plt.hist(p_values, bins=30)
        plt.title('Distribution of P-values')
        plt.xlabel('P-value')
        plt.ylabel('Count')
        
        # 图3: 各行为的效应量
        plt.subplot(2, 2, 3)
        behaviors = list(effect_sizes.keys())
        mean_effects = [np.mean(effect_sizes[b]['effect_sizes']) for b in behaviors]
        plt.bar(behaviors, mean_effects)
        plt.xticks(rotation=45)
        plt.title('Mean Effect Size by Behavior')
        plt.xlabel('Behavior')
        plt.ylabel('Mean Effect Size')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.config.analysis_dir, 'statistical_summary.png'),
                   dpi=self.config.visualization_params['dpi'])
        plt.close() 