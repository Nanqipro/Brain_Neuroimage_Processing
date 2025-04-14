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
import math

def create_gnn_based_topology(model, data, G, node_names, threshold=0.6):
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
        try:
            # 获取GNN的节点嵌入
            embeddings = model.get_embeddings(data)
            # 如果是tensor，转为numpy
            if torch.is_tensor(embeddings):
                embeddings = embeddings.cpu().detach().numpy()
        except Exception as e:
            print(f"获取节点嵌入时出错: {str(e)}")
            print("尝试使用前向传播的输出...")
            # 作为后备方案，使用前向传播的中间层
            output = model(data)
            if hasattr(model, 'last_hidden'):
                embeddings = model.last_hidden.cpu().detach().numpy()
            else:
                # 如果无法获取嵌入，使用随机嵌入
                print("无法获取嵌入，使用随机嵌入...")
                embeddings = np.random.randn(len(G.nodes()), 64)
    
    # 2. 基于嵌入计算新的相似度
    similarities = cosine_similarity(embeddings)
    
    # 3. 创建新的图结构
    new_G = nx.Graph()
    
    # 添加节点
    for i, node in enumerate(G.nodes()):
        node_name = node_names[i] if i < len(node_names) else f"Node_{i}"
        new_G.add_node(node_name, embedding=embeddings[i].tolist())
    
    # 添加边（基于相似度阈值）
    for i in range(len(G.nodes())):
        node_i = node_names[i] if i < len(node_names) else f"Node_{i}"
        for j in range(i+1, len(G.nodes())):
            node_j = node_names[j] if j < len(node_names) else f"Node_{j}"
            if similarities[i, j] > threshold:
                new_G.add_edge(node_i, node_j, weight=similarities[i, j])
    
    return new_G

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
    
    try:
        print(f"开始创建交互式GNN拓扑可视化: {output_path}")
        print(f"输入数据类型: embeddings={type(embeddings)}, similarities={type(similarities)}")
        
        # 确保目录存在
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # 创建交互式网络
        net = Network(height="800px", width="100%", notebook=False)
        
        # 确保embeddings是numpy数组，以便t-SNE可以处理
        if isinstance(embeddings, list):
            print("embeddings是列表类型，转换为numpy数组")
            embeddings_array = np.array(embeddings)
        else:
            embeddings_array = embeddings
            
        print(f"t-SNE输入数据类型: {type(embeddings_array)}, 形状: {embeddings_array.shape if hasattr(embeddings_array, 'shape') else '未知'}")
        
        # 使用t-SNE降维确定节点位置
        try:
            tsne = TSNE(n_components=2, random_state=42)
            tsne_pos = tsne.fit_transform(embeddings_array)
            print(f"t-SNE降维成功，结果形状: {tsne_pos.shape}")
        except Exception as tsne_err:
            print(f"t-SNE降维失败: {str(tsne_err)}")
            # 如果t-SNE失败，尝试使用随机位置
            num_nodes = len(G.nodes())
            print(f"生成随机位置作为备选，节点数: {num_nodes}")
            tsne_pos = np.random.rand(num_nodes, 2)
        
        # 现在将结果转换为Python原生类型用于JSON序列化
        tsne_pos = numpy_to_python_type(tsne_pos)
        embeddings_list = numpy_to_python_type(embeddings)
        
        # 检测社区结构，为不同社区分配不同颜色
        try:
            import community as community_louvain
            partition = community_louvain.best_partition(G)
            # 将社区ID转换为Python原生类型
            partition = {str(k): int(v) for k, v in partition.items()}
            
            # 为每个社区分配一个颜色
            community_colors = [
                "#FF6347", "#4682B4", "#32CD32", "#FFD700", "#9370DB", 
                "#20B2AA", "#FF69B4", "#8A2BE2", "#00CED1", "#FF8C00",
                "#1E90FF", "#FF1493", "#00FA9A", "#DC143C", "#BA55D3"
            ]
            
            # 如果社区数量超过颜色列表长度，则循环使用
            num_communities = max(partition.values()) + 1
            if num_communities > len(community_colors):
                community_colors = community_colors * (num_communities // len(community_colors) + 1)
                
            print(f"检测到{num_communities}个社区")
        except ImportError:
            print("警告: 未找到python-louvain库，无法检测社区结构")
            partition = {node: 0 for node in G.nodes()}
            community_colors = ["#97C2FC"]  # 默认颜色
        except Exception as comm_err:
            print(f"社区检测失败: {str(comm_err)}")
            partition = {node: 0 for node in G.nodes()}
            community_colors = ["#97C2FC"]  # 默认颜色
        
        # 添加节点
        for i, node in enumerate(G.nodes()):
            x, y = tsne_pos[i][0] * 10, tsne_pos[i][1] * 10  # 放大坐标以便更好地显示
            # 节点编号为N+索引，从1开始
            node_number = i + 1
            
            # 获取节点度 - 并转换为Python类型
            node_degree = numpy_to_python_type(G.degree(node))
            
            # 节点名称
            node_label = node_names[i] if i < len(node_names) else f"N{node_number}"
            
            # 获取节点所属社区并分配颜色
            community_id = partition.get(str(node), 0)
            node_color = community_colors[community_id % len(community_colors)]
            
            net.add_node(
                i,  # 使用整数ID避免序列化问题
                label=f"N{node_number}", 
                title=f"Neuron: {node_label}\nID: N{node_number}\nDegree: {node_degree}\n社区: {community_id+1}",
                size=15,  # 固定节点大小
                font={"size": 12, "face": "Arial", "color": "black"},
                x=x, 
                y=y,
                color=node_color,  # 根据社区分配颜色
            )
        
        # 添加边 - 确保所有数值都是Python内置类型，并设置为非常细
        for u, v, attr in G.edges(data=True):
            # 获取节点索引
            u_idx = list(G.nodes()).index(u)
            v_idx = list(G.nodes()).index(v)
            
            # 获取权重 - 转换为Python float
            if 'weight' in attr:
                weight = numpy_to_python_type(attr['weight'])
            else:
                weight = 1.0
            
            # 边宽度设置为非常细 (0.5-1.5范围内)
            width = 0.5 + min(1.0, weight)
            
            # 所有数值都转换为Python内置类型
            edge_data = {
                "value": width,
                "title": f"Similarity: {weight:.4f}",
                "width": width,  # 确保边非常细
                "color": {"opacity": 1}  # 不透明
            }
            
            net.add_edge(u_idx, v_idx, **edge_data)
        
        # 设置网络选项 - 确保使用Python内置类型
        physics_options = {
            "enabled": True,
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
        }
        
        # 保存网络数据到临时文件，进行修改后再加载
        # 这是为了避免pyvis内部序列化问题
        tmp_nodes_path = os.path.join(os.path.dirname(output_path), "temp_nodes.json")
        tmp_edges_path = os.path.join(os.path.dirname(output_path), "temp_edges.json")
        
        # 确保节点和边数据是Python内置类型
        nodes_data = numpy_to_python_type(net.nodes)
        edges_data = numpy_to_python_type(net.edges)
        
        # 保存为临时文件
        with open(tmp_nodes_path, 'w') as f:
            json.dump(nodes_data, f)
        with open(tmp_edges_path, 'w') as f:
            json.dump(edges_data, f)
            
        # 重新加载数据（现在是Python内置类型）
        with open(tmp_nodes_path, 'r') as f:
            net.nodes = json.load(f)
        with open(tmp_edges_path, 'r') as f:
            net.edges = json.load(f)
            
        # 删除临时文件
        try:
            os.remove(tmp_nodes_path)
            os.remove(tmp_edges_path)
        except:
            pass
        
        # 设置交互选项
        net.set_options("""
        var options = {
          "nodes": {
            "borderWidth": 2,
            "borderWidthSelected": 4,
            "shape": "dot",
            "scaling": {
              "min": 20,
              "max": 50
            }
          },
          "edges": {
            "color": {
              "inherit": true
            },
            "smooth": {
              "enabled": false,
              "type": "continuous"
            }
          },
          "interaction": {
            "hover": true,
            "navigationButtons": true,
            "keyboard": true
          },
          "physics": %s
        }
        """ % json.dumps(physics_options))
        
        # 保存为HTML文件
        try:
            print(f"尝试保存交互式网络图到: {output_path}")
            net.save_graph(output_path)
            print(f"交互式神经元网络已成功保存到: {output_path}")
            return output_path
        except Exception as save_err:
            print(f"保存交互式网络时出错: {str(save_err)}")
            import traceback
            print(f"保存错误详情:\n{traceback.format_exc()}")
            
            # 尝试使用更基本的设置重新保存
            try:
                print("尝试使用简化设置重新保存...")
                simple_net = Network(height="600px", width="100%", notebook=False)
                for i, node in enumerate(G.nodes()):
                    simple_net.add_node(i, label=f"N{i+1}")
                
                for u, v in G.edges():
                    u_idx = list(G.nodes()).index(u)
                    v_idx = list(G.nodes()).index(v)
                    simple_net.add_edge(u_idx, v_idx)
                
                backup_path = output_path.replace('.html', '_backup.html')
                simple_net.save_graph(backup_path)
                print(f"简化版交互式网络已保存到: {backup_path}")
                return backup_path
            except Exception as simple_save_err:
                print(f"简化版保存也失败了: {str(simple_save_err)}")
                return None
        
    except Exception as e:
        import traceback
        error_trace = traceback.format_exc()
        print(f"创建交互式可视化时出错: {str(e)}")
        print(f"错误详情:\n{error_trace}")
        return None

def save_gnn_topology_data(G, embeddings, similarities, node_names, output_path):
    """
    将GNN拓扑分析数据保存为JSON文件
    
    参数:
        G: NetworkX图对象
        embeddings: 节点嵌入矩阵
        similarities: 节点相似度矩阵
        node_names: 节点名称列表
        output_path: 输出文件路径
    """
    try:
        # 确保目录存在
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # 准备节点数据
        nodes_data = []
        for i, node in enumerate(G.nodes()):
            node_data = {
                "id": i,
                "name": node_names[i] if i < len(node_names) else f"Node_{i}",
                "degree": G.degree(node),
                "embedding_dim": embeddings.shape[1] if hasattr(embeddings, 'shape') else len(embeddings[i])
            }
            nodes_data.append(node_data)
        
        # 准备边数据
        edges_data = []
        for u, v, data in G.edges(data=True):
            # 获取节点索引
            u_idx = list(G.nodes()).index(u) if isinstance(u, str) else u
            v_idx = list(G.nodes()).index(v) if isinstance(v, str) else v
            
            edge_data = {
                "source": u_idx,
                "target": v_idx,
                "similarity": float(data.get("weight", 0)) if "weight" in data else 0
            }
            edges_data.append(edge_data)
        
        # 使用numpy_to_python_type处理全部数据
        # 构建完整数据结构
        data = {
            "nodes": nodes_data,
            "edges": edges_data,
            "embeddings": embeddings,
            "similarity_matrix": similarities
        }
        
        # 转换所有数据为Python内置类型
        serializable_data = numpy_to_python_type(data)
        
        # 保存为JSON
        with open(output_path, 'w') as f:
            json.dump(serializable_data, f, indent=2)
        
        print(f"GNN拓扑数据已保存到: {output_path}")
    except Exception as e:
        import traceback
        error_trace = traceback.format_exc()
        print(f"保存GNN拓扑数据时出错: {str(e)}")
        print(f"错误详情:\n{error_trace}")

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

def numpy_to_python_type(obj):
    """
    将NumPy类型转换为Python内置类型，用于JSON序列化
    
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
    elif isinstance(obj, dict):
        return {str(k): numpy_to_python_type(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [numpy_to_python_type(i) for i in obj]
    else:
        # 其他类型直接返回
        return obj

def visualize_gcn_topology(topology_data_path):
    """
    根据GCN拓扑数据文件生成静态拓扑结构图
    
    参数:
        topology_data_path: GCN拓扑数据JSON文件路径
    """
    print("\n开始生成GCN静态拓扑结构图...")
    
    try:
        # 读取拓扑数据
        with open(topology_data_path, 'r') as f:
            data = json.load(f)
        
        # 创建NetworkX图对象
        G = nx.Graph()
        
        # 添加节点
        for node in data['nodes']:
            G.add_node(node['id'])
        
        # 添加边
        for edge in data['edges']:
            G.add_edge(edge['source'], edge['target'], weight=edge['similarity'])
        
        # 检测社区结构
        import community as community_louvain
        communities = community_louvain.best_partition(G)
        
        # 为不同社区分配不同颜色
        community_colors = [
            "#FF6347", "#4682B4", "#32CD32", "#FFD700", "#9370DB", 
            "#20B2AA", "#FF69B4", "#8A2BE2", "#00CED1", "#FF8C00",
            "#1E90FF", "#FF1493", "#00FA9A", "#DC143C", "#BA55D3"
        ]
        
        # 获取社区数量
        num_communities = len(set(communities.values()))
        if num_communities > len(community_colors):
            community_colors = community_colors * (num_communities // len(community_colors) + 1)
        
        # 创建节点颜色映射
        node_colors = [community_colors[communities[node]] for node in G.nodes()]
        
        # 设置绘图参数
        plt.figure(figsize=(15, 15))
        
        # 使用spring布局，增加k值和迭代次数使节点更分散
        pos = nx.spring_layout(G, 
                             k=2.0,           # 增加节点间理想距离
                             iterations=100,   # 增加迭代次数
                             seed=42)         # 设置随机种子保证结果可重现
        
        # 绘制边（使用权重确定边的宽度，降低alpha值使边更透明）
        edge_weights = [G[u][v]['weight'] * 1.5 for u, v in G.edges()]  # 稍微减小边的宽度系数
        nx.draw_networkx_edges(G, pos, width=edge_weights, alpha=0.2, edge_color='gray')  # 降低边的不透明度
        
        # 绘制节点（稍微增大节点大小）
        node_sizes = [400 for _ in G.nodes()]  # 增加节点大小
        nx.draw_networkx_nodes(G, pos, node_size=node_sizes, node_color=node_colors, alpha=0.9)  # 增加节点不透明度
        
        # 添加节点标签（调整字体大小）
        labels = {node: f'N{node+1}' for node in G.nodes()}
        nx.draw_networkx_labels(G, pos, labels, font_size=9, font_weight='bold')  # 稍微增加字体大小
        
        plt.title('GCN-Based Neuron Network Topology', fontsize=16)
        plt.axis('off')
        
        # 添加社区图例
        legend_elements = [plt.Line2D([0], [0], marker='o', color='w', 
                                    markerfacecolor=community_colors[i],
                                    markersize=10, label=f'Community {i+1}')
                         for i in range(num_communities)]
        plt.legend(handles=legend_elements, loc='center left', bbox_to_anchor=(1, 0.5))
        
        # 保存图像
        output_path = os.path.join(os.path.dirname(topology_data_path), 'gcn_topology_static.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"GCN静态拓扑结构图已保存至: {output_path}")
        return output_path
        
    except Exception as e:
        print(f"生成GCN静态拓扑结构图时出错: {str(e)}")
        import traceback
        print(f"错误详情:\n{traceback.format_exc()}")
        return None 
    
def visualize_gat_topology(topology_data_path):
    """
    根据GAT拓扑数据文件生成静态拓扑结构图
    
    参数:
        topology_data_path: GAT拓扑数据JSON文件路径
    """
    print("\n开始生成GAT静态拓扑结构图...")
    
    try:
        # 读取拓扑数据
        with open(topology_data_path, 'r') as f:
            data = json.load(f)
        
        # 创建NetworkX图对象
        G = nx.Graph()
        
        # 添加节点
        for node in data['nodes']:
            G.add_node(node['id'])
        
        # 添加边
        for edge in data['edges']:
            G.add_edge(edge['source'], edge['target'], weight=edge['similarity'])
        
        # 检测社区结构
        import community as community_louvain
        communities = community_louvain.best_partition(G)
        
        # 为不同社区分配不同颜色
        community_colors = [
            "#FF6347", "#4682B4", "#32CD32", "#FFD700", "#9370DB", 
            "#20B2AA", "#FF69B4", "#8A2BE2", "#00CED1", "#FF8C00",
            "#1E90FF", "#FF1493", "#00FA9A", "#DC143C", "#BA55D3"
        ]
        
        # 获取社区数量
        num_communities = len(set(communities.values()))
        if num_communities > len(community_colors):
            community_colors = community_colors * (num_communities // len(community_colors) + 1)
        
        # 创建节点颜色映射
        node_colors = [community_colors[communities[node]] for node in G.nodes()]
        
        # 设置绘图参数
        plt.figure(figsize=(15, 15))
        
        # 使用spring布局，增加k值和迭代次数使节点更分散
        pos = nx.spring_layout(G, 
                             k=2.0,           # 增加节点间理想距离
                             iterations=100,   # 增加迭代次数
                             seed=42)         # 设置随机种子保证结果可重现
        
        # 绘制边（使用权重确定边的宽度，降低alpha值使边更透明）
        edge_weights = [G[u][v]['weight'] * 1.5 for u, v in G.edges()]  # 稍微减小边的宽度系数
        nx.draw_networkx_edges(G, pos, width=edge_weights, alpha=0.2, edge_color='gray')  # 降低边的不透明度
        
        # 绘制节点（稍微增大节点大小）
        node_sizes = [400 for _ in G.nodes()]  # 增加节点大小
        nx.draw_networkx_nodes(G, pos, node_size=node_sizes, node_color=node_colors, alpha=0.9)  # 增加节点不透明度
        
        # 添加节点标签（调整字体大小）
        labels = {node: f'N{node+1}' for node in G.nodes()}
        nx.draw_networkx_labels(G, pos, labels, font_size=9, font_weight='bold')  # 稍微增加字体大小
        
        plt.title('GAT-Based Neuron Network Topology', fontsize=16)
        plt.axis('off')
        
        # 添加社区图例
        legend_elements = [plt.Line2D([0], [0], marker='o', color='w', 
                                    markerfacecolor=community_colors[i],
                                    markersize=10, label=f'Community {i+1}')
                         for i in range(num_communities)]
        plt.legend(handles=legend_elements, loc='center left', bbox_to_anchor=(1, 0.5))
        
        # 保存图像
        output_path = os.path.join(os.path.dirname(topology_data_path), 'gat_topology_static.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"GAT静态拓扑结构图已保存至: {output_path}")
        return output_path
        
    except Exception as e:
        print(f"生成GAT静态拓扑结构图时出错: {str(e)}")
        import traceback
        print(f"错误详情:\n{traceback.format_exc()}")
        return None 

def visualize_gcn_topology_with_real_positions(topology_data_path, position_data_path):
    """
    根据GCN拓扑数据文件和真实神经元位置坐标生成静态拓扑结构图
    
    参数:
        topology_data_path: GCN拓扑数据JSON文件路径
        position_data_path: 神经元位置坐标CSV文件路径
    
    返回:
        output_path: 生成的图像文件路径
    """
    print("\n开始基于真实位置生成GCN静态拓扑结构图...")
    
    try:
        # 读取拓扑数据
        with open(topology_data_path, 'r') as f:
            topo_data = json.load(f)
        
        # 读取位置数据
        import pandas as pd
        position_df = pd.read_csv(position_data_path)
        
        # 检查CSV列名
        columns = position_df.columns.tolist()
        print(f"位置数据CSV列名: {columns}")
        
        # 自动检测坐标列名
        x_col = None
        y_col = None
        id_col = None
        
        # 尝试查找常见的坐标列名
        for col in columns:
            col_lower = col.lower()
            if 'x' in col_lower or 'lon' in col_lower:
                x_col = col
            elif 'y' in col_lower or 'lat' in col_lower:
                y_col = col
            elif 'num' in col_lower or 'id' in col_lower or 'index' in col_lower:
                id_col = col
        
        # 如果没有找到合适的列名，使用默认值
        if x_col is None and 'relative_x' in columns:
            x_col = 'relative_x'
        if y_col is None and 'relative_y' in columns:
            y_col = 'relative_y'
        if id_col is None and 'number' in columns:
            id_col = 'number'
            
        # 如果仍然没有找到，使用第一列作为ID，第二列作为x，第三列作为y
        if x_col is None and len(columns) >= 2:
            x_col = columns[1]
        if y_col is None and len(columns) >= 3:
            y_col = columns[2]
        if id_col is None and len(columns) >= 1:
            id_col = columns[0]
            
        print(f"使用的列名: ID={id_col}, X={x_col}, Y={y_col}")
        
        # 创建节点编号到CSV行索引的映射，确保正确匹配位置数据
        node_to_csv_row = {}
        neuron_ids = position_df[id_col].values
        
        # 将CSV文件中的neuron ID映射到行索引
        for idx, neuron_id in enumerate(neuron_ids):
            node_to_csv_row[int(neuron_id)] = idx
        
        print(f"从CSV文件中加载了{len(node_to_csv_row)}个神经元位置数据")
        
        # 创建NetworkX图对象
        G = nx.Graph()
        
        # 添加节点
        for node in topo_data['nodes']:
            G.add_node(node['id'])
        
        # 添加边
        for edge in topo_data['edges']:
            G.add_edge(edge['source'], edge['target'], weight=edge['similarity'])
        
        # 检测社区结构
        import community as community_louvain
        communities = community_louvain.best_partition(G)
        
        # 为不同社区分配不同颜色
        community_colors = [
            "#FF6347", "#4682B4", "#32CD32", "#FFD700", "#9370DB", 
            "#20B2AA", "#FF69B4", "#8A2BE2", "#00CED1", "#FF8C00",
            "#1E90FF", "#FF1493", "#00FA9A", "#DC143C", "#BA55D3"
        ]
        
        # 获取社区数量
        num_communities = len(set(communities.values()))
        if num_communities > len(community_colors):
            community_colors = community_colors * (num_communities // len(community_colors) + 1)
        
        # 创建节点颜色映射
        node_colors = [community_colors[communities[node]] for node in G.nodes()]
        
        # 创建位置字典
        pos = {}
        nodes_with_positions = 0
        nodes_with_random_positions = 0
        
        # 找出所有节点的最大和最小坐标，用于归一化
        all_x = []
        all_y = []
        for i in range(len(position_df)):
            all_x.append(position_df.iloc[i][x_col])
            all_y.append(position_df.iloc[i][y_col])
        
        min_x, max_x = min(all_x), max(all_x)
        min_y, max_y = min(all_y), max(all_y)
        
        for i, node in enumerate(G.nodes()):
            # 寻找与该节点对应的神经元编号
            node_num = node + 1  # 节点ID+1，因为position_df中的编号从1开始
            
            try:
                # 首先尝试直接查找节点编号
                if node_num in node_to_csv_row:
                    row_idx = node_to_csv_row[node_num]
                    
                    # 获取坐标
                    x_pos = position_df.iloc[row_idx][x_col]
                    y_pos = position_df.iloc[row_idx][y_col]
                    
                    # 反转Y坐标以修复上下颠倒问题
                    # 使用1减去归一化后的y值
                    y_pos = 1.0 - ((y_pos - min_y) / (max_y - min_y)) if max_y > min_y else y_pos
                    # 归一化x坐标
                    x_pos = (x_pos - min_x) / (max_x - min_x) if max_x > min_x else x_pos
                    
                    pos[node] = (x_pos, y_pos)
                    nodes_with_positions += 1
                    
                    # 打印前几个节点的真实位置日志
                    if i < 10:
                        print(f"节点{node}(N{node_num})位置: 原始({position_df.iloc[row_idx][x_col]:.3f}, {position_df.iloc[row_idx][y_col]:.3f}) => 转换后({x_pos:.3f}, {y_pos:.3f})")
                
                # 回退: 尝试使用node作为索引
                elif node < len(position_df):
                    # 使用相对坐标作为节点位置
                    x_pos = position_df.iloc[node][x_col]
                    y_pos = position_df.iloc[node][y_col]
                    
                    # 反转Y坐标以修复上下颠倒问题
                    y_pos = 1.0 - ((y_pos - min_y) / (max_y - min_y)) if max_y > min_y else y_pos
                    x_pos = (x_pos - min_x) / (max_x - min_x) if max_x > min_x else x_pos
                    
                    pos[node] = (x_pos, y_pos)
                    nodes_with_positions += 1
                    
                    if i < 10:
                        print(f"节点{node}(回退)位置: 原始({position_df.iloc[node][x_col]:.3f}, {position_df.iloc[node][y_col]:.3f}) => 转换后({x_pos:.3f}, {y_pos:.3f})")
                else:
                    # 如果找不到对应位置，使用随机位置
                    pos[node] = (np.random.random(), np.random.random())
                    nodes_with_random_positions += 1
                    if i < 10:
                        print(f"警告: 无法找到节点{node}的位置数据，使用随机位置")
            except Exception as e:
                print(f"处理节点{node}位置时出错: {str(e)}")
                pos[node] = (np.random.random(), np.random.random())
                nodes_with_random_positions += 1
        
        print(f"节点总数: {len(G.nodes())}, 使用真实位置节点数: {nodes_with_positions}, 使用随机位置节点数: {nodes_with_random_positions}")
        
        # 设置绘图参数
        plt.figure(figsize=(15, 15))
        
        # 绘制边（使用权重确定边的宽度，降低alpha值使边更透明）
        edge_weights = [G[u][v]['weight'] * 1.5 for u, v in G.edges()]
        nx.draw_networkx_edges(G, pos, width=edge_weights, alpha=0.2, edge_color='gray')
        
        # 绘制节点（节点大小调整）
        node_sizes = [400 for _ in G.nodes()]
        nx.draw_networkx_nodes(G, pos, node_size=node_sizes, node_color=node_colors, alpha=0.9)
        
        # 添加节点标签
        labels = {}
        for node in G.nodes():
            node_num = node + 1  # 根据需要调整
            if node_num in node_to_csv_row:
                # 使用CSV中的实际编号
                actual_num = int(position_df.iloc[node_to_csv_row[node_num]][id_col])
                labels[node] = f'N{actual_num}'
            else:
                # 回退到默认编号
                labels[node] = f'N{node_num}'
        
        nx.draw_networkx_labels(G, pos, labels, font_size=9, font_weight='bold')
        
        plt.title('GCN-Based Neuron Network Topology (Real Positions)', fontsize=16)
        plt.axis('off')
        
        # 添加社区图例
        legend_elements = [plt.Line2D([0], [0], marker='o', color='w', 
                                    markerfacecolor=community_colors[i],
                                    markersize=10, label=f'Community {i+1}')
                         for i in range(num_communities)]
        plt.legend(handles=legend_elements, loc='center left', bbox_to_anchor=(1, 0.5))
        
        # 保存图像
        output_path = os.path.join(os.path.dirname(topology_data_path), 'gcn_topology_real_positions.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"基于真实位置的GCN拓扑结构图已保存至: {output_path}")
        return output_path
        
    except Exception as e:
        print(f"生成基于真实位置的GCN拓扑结构图时出错: {str(e)}")
        import traceback
        print(f"错误详情:\n{traceback.format_exc()}")
        return None


def analyze_community_behaviors(G, communities, X_scaled, y, behavior_labels):
    """
    分析GNN拓扑中的社区与行为标签的关联
    
    参数:
        G: 基于GNN分析生成的NetworkX图对象
        communities: 社区字典，键为节点，值为社区编号
        X_scaled: 标准化后的神经元活动数据
        y: 行为标签数据
        behavior_labels: 行为标签名称列表
        
    返回:
        community_behavior_mapping: 社区-行为映射字典
    """
    print("\n分析GNN拓扑社区与行为的关联...")
    
    # 初始化结果字典
    community_behavior_mapping = {}
    
    # 确保社区字典有效
    if not communities or len(communities) == 0:
        print("警告: 没有有效的社区信息")
        return {}
    
    # 确保有行为标签
    if behavior_labels is None or len(behavior_labels) == 0:
        print("警告: 没有有效的行为标签信息")
        return {}
    
    # 创建行为标签索引映射，出现改变时确保matching
    behavior_indices = {}
    for i, behavior in enumerate(behavior_labels):
        behavior_indices[behavior] = i
    
    # 将社区分组
    comm_groups = {}
    for node, comm_id in communities.items():
        if comm_id not in comm_groups:
            comm_groups[comm_id] = []
        comm_groups[comm_id].append(node)
    
    # 准备节点名称到神经元数据索引的映射
    node_to_index = {}
    for i, node in enumerate(G.nodes()):
        node_to_index[node] = i if i < X_scaled.shape[1] else None
    
    # 分析每个社区与行为的关联
    for comm_id, nodes in comm_groups.items():
        # 获取该社区节点对应的神经元索引
        neuron_indices = []
        for node in nodes:
            if node in node_to_index and node_to_index[node] is not None:
                neuron_indices.append(node_to_index[node])
        
        if not neuron_indices:  # 如果没有有效的索引，跳过此社区
            continue
        
        # 计算每种行为与该社区神经元的关联强度
        behavior_associations = {}
        
        # 对每个行为标签计算
        for behavior_idx, behavior in enumerate(behavior_labels):
            # 获取当前行为的索引
            curr_behavior_idx = behavior_indices.get(behavior, behavior_idx)
            
            # 该行为的样本掩码
            behavior_mask = (y == curr_behavior_idx)
            
            if np.sum(behavior_mask) == 0:  # 如果没有该行为的样本，跳过
                continue
                
            # 计算该行为下社区神经元的平均活动
            try:
                community_activity_in_behavior = np.mean(X_scaled[behavior_mask][:, neuron_indices], axis=0)
                
                # 计算其他行为下社区神经元的平均活动
                other_activity = np.mean(X_scaled[~behavior_mask][:, neuron_indices], axis=0)
                
                # 计算效应量（Cohen's d）
                behavior_std = np.std(X_scaled[behavior_mask][:, neuron_indices], axis=0)
                other_std = np.std(X_scaled[~behavior_mask][:, neuron_indices], axis=0)
                pooled_std = np.sqrt((behavior_std**2 + other_std**2) / 2)
                effect_size = np.mean(np.abs(community_activity_in_behavior - other_activity) / (pooled_std + 1e-10))
                
                # 保存该行为的关联强度
                behavior_associations[behavior] = {
                    'effect_size': float(effect_size),  # 确保是原生类型
                    'mean_activity': float(np.mean(community_activity_in_behavior)),
                    'mean_activity_diff': float(np.mean(community_activity_in_behavior - other_activity))
                }
            except Exception as e:
                print(f"计算社区{comm_id}与行为{behavior}的关联时出错: {str(e)}")
                continue
        
        # 找出与该社区关联最强的行为
        if behavior_associations:
            strongest_behavior = max(behavior_associations.items(), key=lambda x: x[1]['effect_size'])
            
            # 保存社区-行为映射
            community_behavior_mapping[f'Community_{comm_id}'] = {
                'behavior': strongest_behavior[0],
                'effect_size': strongest_behavior[1]['effect_size'],
                'mean_activity': strongest_behavior[1]['mean_activity'],
                'mean_activity_diff': strongest_behavior[1]['mean_activity_diff'],
                'neurons': [str(node) for node in nodes],  # 确保是字符串类型
                'size': len(nodes),
                'behavior_associations': behavior_associations
            }
            
            print(f"社区 {comm_id} ({len(nodes)} 个神经元) 与行为 '{strongest_behavior[0]}' 最相关 (效应量: {strongest_behavior[1]['effect_size']:.3f})")
    
    print(f"分析完成，共发现{len(community_behavior_mapping)}个社区与行为的关联")
    return community_behavior_mapping


def visualize_community_behavior_mapping(G, community_behavior_mapping, output_path):
    """
    可视化社区与行为之间的关联
    
    参数:
        G: NetworkX图对象
        community_behavior_mapping: 社区-行为映射字典
        output_path: 输出文件路径
    """
    if not community_behavior_mapping:
        print("警告: 没有社区-行为映射数据可视化")
        return None
    
    # 确保输出目录存在
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
    # 准备数据
    communities = []
    behaviors = []
    effect_sizes = []
    community_sizes = []
    
    for comm_id, data in community_behavior_mapping.items():
        communities.append(comm_id)
        behaviors.append(data['behavior'])
        effect_sizes.append(data['effect_size'])
        community_sizes.append(data['size'])
    
    # 创建图表
    plt.figure(figsize=(14, 8))
    
    # 创建颜色映射
    unique_behaviors = list(set(behaviors))
    behavior_colors = plt.cm.tab10(np.linspace(0, 1, len(unique_behaviors)))
    behavior_color_map = {b: behavior_colors[i] for i, b in enumerate(unique_behaviors)}
    
    # 将社区按大小排序
    sorted_indices = np.argsort(community_sizes)[::-1]  # 从大到小排序
    
    # 创建柱状图
    bar_positions = np.arange(len(communities))
    bars = plt.bar(bar_positions, 
                   [effect_sizes[i] for i in sorted_indices], 
                   width=0.8,
                   color=[behavior_color_map[behaviors[i]] for i in sorted_indices],
                   alpha=0.8)
    
    # 添加社区大小标注
    for i, bar in enumerate(bars):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.05,
                f'{community_sizes[sorted_indices[i]]} neurons',
                ha='center', va='bottom', rotation=0,
                fontsize=9)
    # 添加标签和标题
    plt.xlabel('Neural communities', fontsize=12)
    plt.ylabel('Behavioral association intensity (effect size)', fontsize=12)
    plt.title('Neural community-behavior association analysis', fontsize=14)
    
    # 设置x轴刻度
    plt.xticks(bar_positions, [communities[i] for i in sorted_indices], rotation=45)
    
    # 添加行为图例
    legend_elements = [plt.Rectangle((0,0), 1, 1, color=behavior_color_map[b], alpha=0.8, label=b) 
                      for b in unique_behaviors]
    plt.legend(handles=legend_elements, title='Behavior type', loc='upper right')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"社区-行为关联可视化已保存到: {output_path}")
    
    # 创建网络可视化
    network_output_path = os.path.join(os.path.dirname(output_path), 'community_behavior_network.png')
    visualize_community_behavior_network(G, community_behavior_mapping, network_output_path)
    
    return output_path


def visualize_community_behavior_network(G, community_behavior_mapping, output_path):
    """
    可视化基于社区划分的神经元网络，不同社区使用不同颜色，并根据行为关联标注
    
    参数:
        G: NetworkX图对象
        community_behavior_mapping: 社区-行为映射字典
        output_path: 输出文件路径
    """
    plt.figure(figsize=(16, 16))
    
    # 创建颜色映射
    unique_behaviors = list(set([data['behavior'] for data in community_behavior_mapping.values()]))
    behavior_colors = plt.cm.tab10(np.linspace(0, 1, len(unique_behaviors)))
    behavior_color_map = {b: behavior_colors[i] for i, b in enumerate(unique_behaviors)}
    
    # 创建节点颜色映射
    node_colors = {}
    node_communities = {}
    
    # 解析社区ID
    for comm_id, data in community_behavior_mapping.items():
        # 从'Community_X'格式中提取数字
        comm_num = int(comm_id.split('_')[1]) if '_' in comm_id else int(comm_id)
        behavior = data['behavior']
        color = behavior_color_map[behavior]
        
        for node_str in data['neurons']:
            # 将节点字符串转换为合适的节点ID
            try:
                if node_str.isdigit():
                    node = int(node_str)
                else:
                    node = node_str
                    
                node_colors[node] = color
                node_communities[node] = comm_id
            except Exception as e:
                print(f"处理节点{node_str}时出错: {str(e)}")
                continue
    
    # 使用spring布局 - 控制随机种子以确保每次可视化结果一致
    pos = nx.spring_layout(G, seed=42, k=0.3)
    
    # 绘制节点
    for node in G.nodes():
        if node in node_colors:
            nx.draw_networkx_nodes(G, pos, nodelist=[node], 
                                 node_color=[node_colors[node]], 
                                 node_size=100, alpha=0.8)
        else:
            # 对于没有社区的节点，使用灰色
            nx.draw_networkx_nodes(G, pos, nodelist=[node], 
                                 node_color=['lightgray'], 
                                 node_size=50, alpha=0.5)
    
    # 绘制边 - 根据是否连接同一社区的节点使用不同透明度
    same_community_edges = []
    diff_community_edges = []
    
    for u, v in G.edges():
        if u in node_communities and v in node_communities:
            if node_communities[u] == node_communities[v]:
                same_community_edges.append((u, v))
            else:
                diff_community_edges.append((u, v))
        else:
            diff_community_edges.append((u, v))
    
    # 同一社区内的边使用高透明度
    nx.draw_networkx_edges(G, pos, edgelist=same_community_edges, 
                         width=1.5, alpha=0.7, edge_color='gray')
    # 不同社区之间的边使用低透明度
    nx.draw_networkx_edges(G, pos, edgelist=diff_community_edges, 
                         width=0.5, alpha=0.2, edge_color='lightgray')
    
    # 添加社区标签
    # 计算每个社区的中心位置
    community_centers = {}
    for comm_id, data in community_behavior_mapping.items():
        nodes = []
        for node_str in data['neurons']:
            try:
                if node_str.isdigit():
                    node = int(node_str)
                else:
                    node = node_str
                nodes.append(node)
            except:
                continue
                
        if not nodes:
            continue
            
        # 计算社区节点的平均位置
        centers_x = []
        centers_y = []
        
        for node in nodes:
            if node in pos:
                centers_x.append(pos[node][0])
                centers_y.append(pos[node][1])
        
        if centers_x and centers_y:
            center_x = sum(centers_x) / len(centers_x)
            center_y = sum(centers_y) / len(centers_y)
            community_centers[comm_id] = (center_x, center_y)
    
    # 绘制社区标签
    for comm_id, center in community_centers.items():
        behavior = community_behavior_mapping[comm_id]['behavior']
        plt.text(center[0], center[1], 
                f"{comm_id}\n({behavior})", 
                fontsize=12, fontweight='bold', 
                ha='center', va='center',
                bbox=dict(facecolor='white', alpha=0.7, edgecolor='none'))
    
    # 添加图例
    legend_elements = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=color, 
                                markersize=10, label=behavior) 
                      for behavior, color in behavior_color_map.items()]
    plt.legend(handles=legend_elements, loc='upper right', title='Behavior type')
    
    plt.title('Neural community-behavior association network', fontsize=16)
    plt.axis('off')
    plt.tight_layout()
    
    # 确保输出目录存在
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"社区-行为网络可视化已保存到: {output_path}")
    return output_path