"""
GNN神经元网络可视化模块

该模块提供用于可视化GNN分析结果的函数和类
"""

import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from torch_geometric.utils import to_networkx
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import json
import seaborn as sns
from sklearn.manifold import TSNE
from typing import Dict, List, Any, Optional, Tuple, Union

class GNNVisualizer:
    """
    GNN神经元网络可视化器
    """
    def __init__(self, config):
        """
        初始化GNN可视化器
        
        参数:
            config: 分析配置对象
        """
        self.config = config
        # 设置可视化输出目录
        self.viz_dir = config.gnn_results_dir
        os.makedirs(self.viz_dir, exist_ok=True)
    
    def visualize_gnn_embeddings(self, embeddings, labels, neuron_names=None, title='神经元GNN嵌入可视化'):
        """
        可视化GNN嵌入
        
        参数:
            embeddings: 节点嵌入向量
            labels: 节点标签
            neuron_names: 神经元名称列表
            title: 图表标题
        """
        # 使用t-SNE降维
        tsne = TSNE(n_components=2, random_state=42)
        embeddings_2d = tsne.fit_transform(embeddings)
        
        # 绘制散点图
        plt.figure(figsize=(12, 10))
        scatter = plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], 
                            c=labels, cmap='viridis', s=100, alpha=0.8)
        
        # 添加标签
        if neuron_names is not None:
            for i, name in enumerate(neuron_names):
                plt.annotate(name, (embeddings_2d[i, 0], embeddings_2d[i, 1]), 
                           fontsize=8, alpha=0.7)
        
        plt.colorbar(scatter, label='类别')
        plt.title(title)
        plt.xlabel('t-SNE维度1')
        plt.ylabel('t-SNE维度2')
        plt.tight_layout()
        
        # 保存图像
        viz_path = os.path.join(self.viz_dir, 'gnn_embeddings.png')
        plt.savefig(viz_path, dpi=300)
        plt.close()
        return viz_path
    
    def visualize_attention_weights(self, attention_weights, neuron_names, title='神经元GAT注意力权重'):
        """
        可视化GAT注意力权重
        
        参数:
            attention_weights: 注意力权重矩阵
            neuron_names: 神经元名称列表
            title: 图表标题
        """
        plt.figure(figsize=(14, 12))
        sns.heatmap(attention_weights, annot=False, cmap='viridis', 
                   xticklabels=neuron_names, yticklabels=neuron_names)
        plt.title(title)
        plt.tight_layout()
        
        # 保存图像
        viz_path = os.path.join(self.viz_dir, 'gat_attention_weights.png')
        plt.savefig(viz_path, dpi=300)
        plt.close()
        return viz_path
    
    def visualize_gnn_communities(self, G, communities, pos=None, title='GNN社区检测结果'):
        """
        可视化GNN检测的社区
        
        参数:
            G: NetworkX图对象
            communities: 社区分配字典，键为节点，值为社区ID
            pos: 节点位置字典
            title: 图表标题
        """
        plt.figure(figsize=(14, 12))
        
        # 如果没有提供位置，使用spring布局
        if pos is None:
            pos = nx.spring_layout(G)
        
        # 为每个社区分配颜色
        unique_communities = set(communities.values())
        colors = plt.cm.rainbow(np.linspace(0, 1, len(unique_communities)))
        color_map = {comm: colors[i] for i, comm in enumerate(unique_communities)}
        
        # 为每个节点分配颜色
        node_colors = [color_map[communities[node]] for node in G.nodes()]
        
        # 绘制图形
        nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=100)
        nx.draw_networkx_edges(G, pos, alpha=0.3)
        nx.draw_networkx_labels(G, pos, font_size=8)
        
        plt.title(title)
        plt.axis('off')
        
        # 保存图像
        viz_path = os.path.join(self.viz_dir, 'gnn_communities.png')
        plt.savefig(viz_path, dpi=300)
        plt.close()
        return viz_path
    
    def visualize_gnn_behavior_prediction(self, embeddings, true_labels, predicted_labels, 
                                        neuron_names=None, behavior_names=None,
                                        title='行为预测GNN结果'):
        """
        可视化GNN行为预测结果
        
        参数:
            embeddings: 节点嵌入向量
            true_labels: 真实标签
            predicted_labels: 预测标签
            neuron_names: 神经元名称列表
            behavior_names: 行为名称列表
            title: 图表标题
        """
        # 使用t-SNE降维
        tsne = TSNE(n_components=2, random_state=42)
        embeddings_2d = tsne.fit_transform(embeddings)
        
        # 创建子图，一个显示真实标签，一个显示预测标签
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
        
        # 绘制真实标签
        scatter1 = ax1.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], 
                             c=true_labels, cmap='tab10', s=100, alpha=0.8)
        ax1.set_title('真实行为标签')
        
        # 绘制预测标签
        scatter2 = ax2.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], 
                             c=predicted_labels, cmap='tab10', s=100, alpha=0.8)
        ax2.set_title('GNN预测行为标签')
        
        # 添加神经元标签（如果提供）
        if neuron_names is not None:
            for i, name in enumerate(neuron_names):
                ax1.annotate(name, (embeddings_2d[i, 0], embeddings_2d[i, 1]), 
                           fontsize=8, alpha=0.7)
                ax2.annotate(name, (embeddings_2d[i, 0], embeddings_2d[i, 1]), 
                           fontsize=8, alpha=0.7)
        
        # 添加颜色条图例
        if behavior_names is not None:
            # 创建自定义图例
            from matplotlib.lines import Line2D
            legend_elements = [Line2D([0], [0], marker='o', color='w', 
                                    markerfacecolor=plt.cm.tab10(i), markersize=10, 
                                    label=behavior_names[i])
                             for i in range(len(behavior_names))]
            ax1.legend(handles=legend_elements, loc='upper right')
            ax2.legend(handles=legend_elements, loc='upper right')
        else:
            # 使用普通颜色条
            plt.colorbar(scatter1, ax=ax1, label='行为类别')
            plt.colorbar(scatter2, ax=ax2, label='行为类别')
        
        fig.suptitle(title, fontsize=16)
        plt.tight_layout()
        
        # 保存图像
        viz_path = os.path.join(self.viz_dir, 'gnn_behavior_prediction.png')
        plt.savefig(viz_path, dpi=300)
        plt.close()
        return viz_path
    
    def create_interactive_gnn_visualization(self, G, embeddings, labels, neuron_names=None,
                                          title='交互式神经元GNN可视化'):
        """
        创建交互式GNN可视化
        
        参数:
            G: NetworkX图对象
            embeddings: 节点嵌入向量
            labels: 节点标签
            neuron_names: 神经元名称列表
            title: 图表标题
            
        返回:
            html_path: 保存的HTML文件路径
        """
        # 获取节点位置（使用t-SNE嵌入或spring布局）
        if embeddings is not None and embeddings.shape[1] >= 2:
            # 直接使用前两个维度作为坐标
            pos = {node: (embeddings[i, 0], embeddings[i, 1]) 
                 for i, node in enumerate(G.nodes())}
        else:
            # 使用spring布局
            pos = nx.spring_layout(G)
        
        # 创建边迹
        edge_trace = go.Scatter(
            x=[],
            y=[],
            line=dict(width=0.5, color='#888'),
            hoverinfo='none',
            mode='lines')
        
        # 添加边坐标
        for edge in G.edges():
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]
            edge_trace.x += (x0, x1, None)
            edge_trace.y += (y0, y1, None)
        
        # 创建节点迹
        node_trace = go.Scatter(
            x=[],
            y=[],
            text=[],
            mode='markers',
            hoverinfo='text',
            marker=dict(
                showscale=True,
                colorscale='Viridis',
                size=10,
                colorbar=dict(
                    thickness=15,
                    title='节点类别',
                    xanchor='left',
                    titleside='right'
                ),
                line=dict(width=2)))
        
        # 添加节点坐标
        for node in G.nodes():
            x, y = pos[node]
            node_trace.x += (x,)
            node_trace.y += (y,)
        
        # 添加节点文本和颜色
        node_info = []
        node_colors = []
        for i, node in enumerate(G.nodes()):
            if neuron_names is not None:
                node_label = neuron_names[i]
            else:
                node_label = str(node)
            
            node_info.append(f'神经元: {node_label}<br>类别: {labels[i]}')
            node_colors.append(labels[i])
        
        node_trace.marker.color = node_colors
        node_trace.text = node_info
        
        # 创建图形布局
        layout = go.Layout(
            title=title,
            titlefont=dict(size=16),
            showlegend=False,
            hovermode='closest',
            margin=dict(b=20, l=5, r=5, t=40),
            annotations=[dict(text="", showarrow=False, xref="paper", yref="paper")],
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)
        )
        
        # 创建图形
        fig = go.Figure(data=[edge_trace, node_trace], layout=layout)
        
        # 保存为HTML文件
        html_path = os.path.join(self.viz_dir, 'interactive_gnn_visualization.html')
        fig.write_html(html_path)
        
        return html_path
    
    def visualize_time_series_gnn(self, temporal_data, window_size, title='时间序列GNN结果'):
        """
        可视化时间序列GNN结果
        
        参数:
            temporal_data: 时间窗口数据列表
            window_size: 窗口大小
            title: 图表标题
        """
        # 分析窗口之间的变化
        if isinstance(temporal_data, list) and len(temporal_data) > 1:
            # 计算每个窗口的汇总统计数据
            window_stats = []
            for window_idx, window_data in enumerate(temporal_data):
                # 假设我们有一些测量窗口特征的方法
                # 这里使用简单的汇总统计数据
                if hasattr(window_data, 'x'):
                    mean_activity = torch.mean(window_data.x).item()
                    max_activity = torch.max(window_data.x).item()
                    window_stats.append({
                        'window_idx': window_idx,
                        'mean_activity': mean_activity,
                        'max_activity': max_activity
                    })
            
            # 绘制统计数据
            if window_stats:
                # 创建DataFrame
                import pandas as pd
                stats_df = pd.DataFrame(window_stats)
                
                # 绘制统计数据
                plt.figure(figsize=(12, 6))
                plt.plot(stats_df['window_idx'], stats_df['mean_activity'], 
                       label='平均活动', marker='o')
                plt.plot(stats_df['window_idx'], stats_df['max_activity'], 
                       label='最大活动', marker='x')
                plt.xlabel('时间窗口索引')
                plt.ylabel('活动值')
                plt.title(f'时间窗口（大小={window_size}）的神经元活动变化')
                plt.legend()
                plt.grid(True, alpha=0.3)
                
                # 保存图像
                viz_path = os.path.join(self.viz_dir, 'temporal_gnn_activity.png')
                plt.savefig(viz_path, dpi=300)
                plt.close()
                return viz_path
        
        return None 