"""
可视化模块

该模块提供神经元拓扑结构的可视化功能
"""

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import json
from pathlib import Path
import plotly.graph_objects as go
from typing import Dict, List, Tuple, Optional, Union, Any
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import plotly.io as pio

class TopologyVisualizer:
    """
    拓扑结构可视化器
    
    用于可视化神经元拓扑结构
    """
    
    def __init__(self, config):
        """
        初始化可视化器
        
        参数:
            config: 配置对象，包含可视化参数
        """
        self.config = config
        
    def load_topology(self, topology_path: Optional[Path] = None):
        """
        加载拓扑结构
        
        参数:
            topology_path: 拓扑结构文件路径，如果为None则使用配置中的默认路径
            
        返回:
            拓扑结构数据和构建的NetworkX图
        """
        if topology_path is None:
            model_type = self.config.model_type
            topology_path = self.config.get_results_path(f"topology_{model_type}.json")
            
        print(f"加载拓扑结构: {topology_path}")
        
        with open(topology_path, 'r') as f:
            topology_data = json.load(f)
            
        # 构建NetworkX图
        G = nx.Graph()
        
        # 添加节点
        for node in topology_data['nodes']:
            G.add_node(node['id'], embedding=node['embedding'])
            
        # 添加边
        for edge in topology_data['edges']:
            G.add_edge(edge['source'], edge['target'], weight=edge['weight'])
            
        print(f"拓扑结构加载完成：{len(G.nodes())} 个节点，{len(G.edges())} 条边")
        
        return topology_data, G
    
    def visualize_topology_static(self, G, output_path: Optional[Path] = None):
        """
        静态可视化拓扑结构
        
        参数:
            G: NetworkX图对象
            output_path: 输出图像路径，如果为None则使用配置中的默认路径
        """
        if output_path is None:
            model_type = self.config.model_type
            output_path = self.config.get_results_path(f"topology_static_{model_type}.png")
            
        # 创建图形
        plt.figure(figsize=(16, 12))
        
        # 设置布局
        pos = nx.spring_layout(G, seed=42, k=0.5)
        
        # 获取节点度数，用于设置节点大小
        node_degrees = dict(G.degree())
        node_sizes = [self.config.node_size * (0.5 + 0.1 * node_degrees[n]) for n in G.nodes()]
        
        # 获取边权重，用于设置边宽度
        edge_weights = [G[u][v]['weight'] * self.config.edge_width for u, v in G.edges()]
        
        # 获取社区结构
        communities = list(nx.algorithms.community.greedy_modularity_communities(G))
        
        # 为每个社区分配颜色
        cmap = plt.get_cmap(self.config.color_scheme)
        node_colors = []
        node_community = {}
        
        for i, community in enumerate(communities):
            color_idx = i % self.config.max_groups
            for node in community:
                node_community[node] = i
                node_colors.append(cmap(color_idx))
        
        # 绘制节点
        nx.draw_networkx_nodes(
            G, 
            pos, 
            node_size=node_sizes, 
            node_color=node_colors, 
            alpha=0.8
        )
        
        # 绘制边
        nx.draw_networkx_edges(
            G, 
            pos, 
            width=edge_weights, 
            alpha=0.5, 
            edge_color='gray'
        )
        
        # 绘制标签
        nx.draw_networkx_labels(
            G, 
            pos, 
            font_size=10, 
            font_family='sans-serif'
        )
        
        # 添加图例
        legend_elements = []
        for i, community in enumerate(communities):
            if i < self.config.max_groups:
                color_idx = i % self.config.max_groups
                legend_elements.append(
                    plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=cmap(color_idx),
                              markersize=10, label=f'社区 {i+1} ({len(community)}个节点)')
                )
                
        plt.legend(handles=legend_elements, loc='upper right', fontsize=12)
        
        plt.title(f"神经元拓扑结构 ({self.config.model_type.upper()})\n{len(G.nodes())} 个节点，{len(G.edges())} 条边，{len(communities)} 个社区", fontsize=16)
        plt.axis('off')
        plt.tight_layout()
        
        # 保存图像
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"静态拓扑图已保存至 {output_path}")
        
    def visualize_topology_interactive(self, G, output_path: Optional[Path] = None):
        """
        交互式可视化拓扑结构
        
        参数:
            G: NetworkX图对象
            output_path: 输出HTML文件路径，如果为None则使用配置中的默认路径
        """
        if output_path is None:
            model_type = self.config.model_type
            output_path = self.config.get_results_path(f"topology_interactive_{model_type}.html")
            
        # 获取节点嵌入
        embeddings = np.array([data['embedding'] for _, data in G.nodes(data=True)])
        
        # 使用t-SNE降维
        tsne = TSNE(n_components=2, random_state=42)
        embeddings_2d = tsne.fit_transform(embeddings)
        
        # 基于嵌入进行聚类
        kmeans = KMeans(n_clusters=min(8, len(G.nodes())), random_state=42)
        clusters = kmeans.fit_predict(embeddings)
        
        # 设置节点位置
        pos = {node: (float(embeddings_2d[i, 0]), float(embeddings_2d[i, 1])) for i, node in enumerate(G.nodes())}
        
        # 获取节点度数
        node_degrees = dict(G.degree())
        
        # 创建边迹线
        edge_x = []
        edge_y = []
        edge_weights = []
        
        for u, v, data in G.edges(data=True):
            x0, y0 = pos[u]
            x1, y1 = pos[v]
            edge_x.extend([x0, x1, None])
            edge_y.extend([y0, y1, None])
            edge_weights.append(float(data['weight']))
            
        # 创建节点迹线
        node_x = []
        node_y = []
        node_labels = []
        node_sizes = []
        node_clusters = []
        
        for node in G.nodes():
            x, y = pos[node]
            node_x.append(float(x))
            node_y.append(float(y))
            node_labels.append(str(node))
            node_sizes.append(float(10 + 5 * node_degrees[node]))
            node_clusters.append(int(clusters[list(G.nodes()).index(node)]))
            
        # 创建边迹线对象
        edge_trace = go.Scatter(
            x=edge_x, y=edge_y,
            line=dict(width=0.7, color='#888'),
            hoverinfo='none',
            mode='lines'
        )
        
        # 创建节点迹线对象
        node_trace = go.Scatter(
            x=node_x, y=node_y,
            mode='markers',
            hoverinfo='text',
            marker=dict(
                showscale=True,
                colorscale='Viridis',
                size=node_sizes,
                color=node_clusters,
                colorbar=dict(
                    thickness=15,
                    title='节点社区',
                    xanchor='left',
                    titleside='right'
                ),
                line=dict(width=2, color='DarkSlateGrey')
            )
        )
        
        # 设置悬停文本
        hover_texts = []
        for i, node in enumerate(G.nodes()):
            hover_texts.append(f"节点: {node}<br>度: {node_degrees[node]}<br>社区: {node_clusters[i]}")
        node_trace.text = hover_texts
        
        # 创建图形
        fig = go.Figure(
            data=[edge_trace, node_trace],
            layout=go.Layout(
                title=f"神经元拓扑结构 ({self.config.model_type.upper()})<br>{len(G.nodes())} 个节点，{len(G.edges())} 条边",
                titlefont=dict(size=16),
                showlegend=False,
                hovermode='closest',
                margin=dict(b=20, l=5, r=5, t=40),
                xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                height=800,
                annotations=[
                    dict(
                        text="基于GNN学习的神经元表示构建的拓扑结构",
                        showarrow=False,
                        xref="paper", yref="paper",
                        x=0.005, y=-0.002
                    )
                ]
            )
        )
        
        try:
            # 保存HTML文件
            pio.write_html(fig, file=str(output_path), auto_open=False)
            print(f"交互式拓扑图已保存至 {output_path}")
        except Exception as e:
            print(f"保存交互式拓扑图时出错: {str(e)}")
            print("尝试使用备选方法保存...")
            try:
                # 备选保存方法
                html_content = pio.to_html(fig, include_plotlyjs='cdn', include_mathjax='cdn', full_html=True)
                with open(output_path, 'w', encoding='utf-8') as f:
                    f.write(html_content)
                print(f"使用备选方法成功保存交互式拓扑图至 {output_path}")
            except Exception as e2:
                print(f"备选保存方法也失败: {str(e2)}")
                print("跳过交互式拓扑图保存")
        
    def analyze_communities(self, G, output_path: Optional[Path] = None):
        """
        分析社区结构
        
        参数:
            G: NetworkX图对象
            output_path: 输出JSON文件路径，如果为None则使用配置中的默认路径
            
        返回:
            社区分析结果
        """
        if output_path is None:
            model_type = self.config.model_type
            output_path = self.config.get_results_path(f"community_analysis_{model_type}.json")
            
        # 获取社区结构
        communities = list(nx.algorithms.community.greedy_modularity_communities(G))
        
        # 分析每个社区
        community_analysis = []
        
        for i, community in enumerate(communities):
            community_nodes = list(community)
            
            # 计算社区内部连接密度
            subgraph = G.subgraph(community_nodes)
            internal_edges = subgraph.number_of_edges()
            max_possible_edges = len(community_nodes) * (len(community_nodes) - 1) / 2
            internal_density = 0 if max_possible_edges == 0 else internal_edges / max_possible_edges
            
            # 计算社区直径（最长最短路径）
            try:
                diameter = nx.diameter(subgraph)
            except nx.NetworkXError:
                diameter = 0  # 如果社区不连通
                
            # 计算社区中心性
            if len(community_nodes) > 1:
                centrality = nx.degree_centrality(subgraph)
                avg_centrality = sum(centrality.values()) / len(centrality)
                max_centrality_node = max(centrality.items(), key=lambda x: x[1])[0]
            else:
                avg_centrality = 0
                max_centrality_node = community_nodes[0] if community_nodes else None
                
            # 社区分析结果，确保所有值都是Python原生类型
            community_info = {
                'id': i + 1,
                'size': len(community_nodes),
                'nodes': list(community_nodes),
                'internal_density': float(internal_density),
                'diameter': int(diameter),
                'avg_centrality': float(avg_centrality),
                'central_node': max_centrality_node
            }
            
            community_analysis.append(community_info)
            
        # 全局图分析
        global_analysis = {
            'num_nodes': int(G.number_of_nodes()),
            'num_edges': int(G.number_of_edges()),
            'num_communities': len(communities),
            'density': float(nx.density(G)),
            'avg_clustering': float(nx.average_clustering(G)),
            'modularity': float(self._calculate_modularity(G, communities))
        }
        
        # 保存分析结果
        analysis_result = {
            'global': global_analysis,
            'communities': community_analysis
        }
        
        # 确保所有数值都是Python原生类型
        analysis_result = self._convert_to_native_types(analysis_result)
        
        try:
            with open(output_path, 'w') as f:
                json.dump(analysis_result, f, indent=2)
                
            print(f"社区分析结果已保存至 {output_path}")
        except Exception as e:
            print(f"保存社区分析结果时出错: {str(e)}")
            print("尝试使用备选方法保存...")
            try:
                # 转换为字符串后尝试重新解析，确保所有类型兼容
                analysis_result_str = json.dumps(analysis_result, ensure_ascii=False)
                analysis_result_clean = json.loads(analysis_result_str)
                
                with open(output_path, 'w', encoding='utf-8') as f:
                    json.dump(analysis_result_clean, f, indent=2, ensure_ascii=False)
                    
                print(f"使用备选方法成功保存社区分析结果至 {output_path}")
            except Exception as e2:
                print(f"备选保存方法也失败: {str(e2)}")
                print("跳过社区分析结果保存")
        
        return analysis_result
    
    def _calculate_modularity(self, G, communities):
        """
        计算社区模块度
        
        参数:
            G: NetworkX图对象
            communities: 社区列表
            
        返回:
            模块度值
        """
        # 准备社区成员映射
        community_dict = {}
        for i, community in enumerate(communities):
            for node in community:
                community_dict[node] = i
                
        # 总边数
        m = G.number_of_edges()
        if m == 0:
            return 0
            
        # 计算模块度
        modularity = 0
        for u, v in G.edges():
            if community_dict[u] == community_dict[v]:
                modularity += 1/m - G.degree(u) * G.degree(v) / (2 * m * m)
                
        return float(modularity)  # 确保返回Python原生float类型
    
    def _convert_to_native_types(self, obj):
        """
        递归地将NumPy类型转换为Python原生类型
        
        参数:
            obj: 任意对象，可能包含NumPy类型
            
        返回:
            转换后的对象，所有NumPy类型替换为Python原生类型
        """
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return self._convert_to_native_types(obj.tolist())
        elif isinstance(obj, dict):
            return {self._convert_to_native_types(key): self._convert_to_native_types(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [self._convert_to_native_types(item) for item in obj]
        elif isinstance(obj, tuple):
            return tuple(self._convert_to_native_types(item) for item in obj)
        else:
            return obj 