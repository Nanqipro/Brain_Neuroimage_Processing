"""
GNN拓扑可视化模块

该模块提供基于GNN训练结果的神经元网络拓扑结构可视化工具
"""

import numpy as np
import torch
import matplotlib.pyplot as plt
import networkx as nx
import os
from sklearn.manifold import TSNE
from sklearn.metrics.pairwise import cosine_similarity
import json
from typing import Dict, List, Tuple, Optional, Any, Union

def create_gnn_based_topology(model, data, G, node_names, threshold=0.5):
    """
    基于GNN模型学习到的节点表示创建新的拓扑结构图
    
    参数:
        model: 训练好的GNN模型
        data: PyTorch Geometric数据对象
        G: 原始NetworkX图对象
        node_names: 节点名称列表
        threshold: 相似度阈值，用于确定是否添加边
        
    返回:
        new_G: 基于GNN嵌入的新NetworkX图对象
    """
    # 1. 提取模型的节点嵌入
    model.eval()
    with torch.no_grad():
        # 获取GNN的节点嵌入
        embeddings = model.get_embeddings(data).cpu().numpy()
    
    # 2. 基于嵌入计算新的相似度
    similarities = cosine_similarity(embeddings)
    
    # 3. 创建新的图结构
    new_G = nx.Graph()
    
    # 添加节点
    for i, node in enumerate(G.nodes()):
        new_G.add_node(node, embedding=embeddings[i].tolist())
    
    # 添加边（基于相似度阈值）
    for i, node_i in enumerate(G.nodes()):
        for j, node_j in enumerate(G.nodes()):
            if i < j and similarities[i, j] > threshold:
                new_G.add_edge(node_i, node_j, weight=float(similarities[i, j]))
    
    return new_G, embeddings, similarities

def visualize_gnn_topology(G, embeddings, similarities, node_names, output_path, title="GNN-Based Neuron Topology"):
    """
    可视化基于GNN嵌入的神经元拓扑结构
    
    参数:
        G: NetworkX图对象
        embeddings: 节点嵌入矩阵
        similarities: 节点间相似度矩阵
        node_names: 节点名称列表
        output_path: 输出图像路径
        title: 图表标题
    """
    plt.figure(figsize=(15, 15))
    
    # 使用t-SNE降维确定节点位置
    tsne = TSNE(n_components=2, random_state=42)
    tsne_pos = tsne.fit_transform(embeddings)
    
    pos = {node: (tsne_pos[i, 0], tsne_pos[i, 1]) for i, node in enumerate(G.nodes())}
    
    # 获取边权重，用于确定边的宽度
    edge_weights = [G[u][v]['weight'] * 2 for u, v in G.edges()]
    
    # 获取节点度，用于确定节点大小
    node_degrees = dict(G.degree())
    node_sizes = [50 + 10 * node_degrees[node] for node in G.nodes()]
    
    # 绘制节点
    nx.draw_networkx_nodes(G, pos, node_size=node_sizes, alpha=0.8, node_color='lightblue')
    
    # 绘制边
    nx.draw_networkx_edges(G, pos, width=edge_weights, alpha=0.5, edge_color='gray')
    
    # 绘制节点标签
    nx.draw_networkx_labels(G, pos, font_size=8, font_color='black')
    
    plt.title(title, fontsize=16)
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
def create_interactive_gnn_topology(G, embeddings, similarities, node_names, output_path):
    """
    创建基于GNN嵌入的交互式神经元拓扑可视化
    
    参数:
        G: NetworkX图对象
        embeddings: 节点嵌入矩阵
        similarities: 节点间相似度矩阵
        node_names: 节点名称列表
        output_path: 输出HTML文件路径
    """
    try:
        from pyvis.network import Network
    except ImportError:
        print("警告: 未找到pyvis库，请安装: pip install pyvis")
        return None
    
    # 创建交互式网络
    net = Network(height="800px", width="100%", notebook=False)
    
    # 使用t-SNE降维确定节点位置
    tsne = TSNE(n_components=2, random_state=42)
    tsne_pos = tsne.fit_transform(embeddings)
    
    # 添加节点
    for i, node in enumerate(G.nodes()):
        x, y = tsne_pos[i, 0] * 10, tsne_pos[i, 1] * 10  # 放大坐标以便更好地显示
        net.add_node(
            node, 
            label=node, 
            title=f"神经元: {node}\n度: {G.degree(node)}", 
            size=10 + 5 * G.degree(node),
            x=float(x), 
            y=float(y)
        )
    
    # 添加边
    for u, v, data in G.edges(data=True):
        weight = data['weight']
        width = weight * 5  # 根据相似度调整边宽度
        net.add_edge(u, v, value=weight, width=width, title=f"相似度: {weight:.3f}", color={'opacity': 0.7})
    
    # 设置物理布局参数
    net.set_options("""
    {
      "physics": {
        "enabled": true,
        "forceAtlas2Based": {
          "gravitationalConstant": -100,
          "centralGravity": 0.01,
          "springLength": 200,
          "springConstant": 0.08
        },
        "solver": "forceAtlas2Based",
        "stabilization": {
          "iterations": 150
        }
      },
      "interaction": {
        "navigationButtons": true,
        "keyboard": true
      }
    }
    """)
    
    # 保存为HTML文件
    net.save_graph(output_path)
    
    return output_path

def save_gnn_topology_data(G, embeddings, similarities, node_names, output_path):
    """
    将GNN拓扑分析数据保存为JSON文件
    
    参数:
        G: NetworkX图对象
        embeddings: 节点嵌入矩阵
        similarities: 节点间相似度矩阵
        node_names: 节点名称列表
        output_path: 输出JSON文件路径
    """
    # 准备数据
    data = {
        "nodes": [],
        "edges": [],
        "embeddings": {},
        "similarity_matrix": {}
    }
    
    # 添加节点数据
    for i, node in enumerate(G.nodes()):
        node_data = {
            "id": node,
            "degree": G.degree(node),
            "embedding_dim": embeddings.shape[1]
        }
        data["nodes"].append(node_data)
        data["embeddings"][node] = embeddings[i].tolist()
    
    # 添加边数据
    for u, v, attr in G.edges(data=True):
        edge_data = {
            "source": u,
            "target": v,
            "similarity": attr["weight"]
        }
        data["edges"].append(edge_data)
    
    # 添加相似度矩阵
    for i, node_i in enumerate(G.nodes()):
        data["similarity_matrix"][node_i] = {}
        for j, node_j in enumerate(G.nodes()):
            data["similarity_matrix"][node_i][node_j] = float(similarities[i, j])
    
    # 保存为JSON文件
    with open(output_path, 'w') as f:
        json.dump(data, f, indent=2)
    
    return output_path

def analyze_gnn_topology(G, similarities):
    """
    分析基于GNN嵌入的拓扑结构
    
    参数:
        G: NetworkX图对象
        similarities: 相似度矩阵
        
    返回:
        metrics: 拓扑分析指标
    """
    metrics = {}
    
    # 基本网络指标
    metrics['node_count'] = G.number_of_nodes()
    metrics['edge_count'] = G.number_of_edges()
    metrics['average_degree'] = float(2 * G.number_of_edges()) / G.number_of_nodes()
    metrics['density'] = nx.density(G)
    
    # 中心性指标
    metrics['degree_centrality'] = nx.degree_centrality(G)
    metrics['closeness_centrality'] = nx.closeness_centrality(G)
    metrics['betweenness_centrality'] = nx.betweenness_centrality(G)
    metrics['clustering_coefficient'] = nx.clustering(G)
    metrics['average_clustering'] = nx.average_clustering(G)
    
    # 连通性分析
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
    try:
        import community as community_louvain
        communities = community_louvain.best_partition(G)
        metrics['communities'] = communities
        metrics['modularity'] = community_louvain.modularity(communities, G)
        metrics['number_of_communities'] = len(set(communities.values()))
    except:
        # 如果社区检测失败，则使用备选方法
        metrics['communities'] = {}
        metrics['modularity'] = 0.0
        metrics['number_of_communities'] = 0
    
    # 高相似度节点对分析
    high_similarity_pairs = []
    node_list = list(G.nodes())
    for i in range(len(node_list)):
        for j in range(i+1, len(node_list)):
            if similarities[i, j] > 0.8:  # 高相似度阈值
                high_similarity_pairs.append((node_list[i], node_list[j], float(similarities[i, j])))
    
    # 排序并获取前10个高相似度对
    high_similarity_pairs.sort(key=lambda x: x[2], reverse=True)
    metrics['high_similarity_pairs'] = high_similarity_pairs[:10]
    
    return metrics 