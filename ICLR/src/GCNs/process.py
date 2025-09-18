import pandas as pd
import numpy as np
import torch
import torch.nn.functional as F
import networkx as nx
import matplotlib.pyplot as plt
from torch_geometric.data import Data, DataLoader
from imblearn.over_sampling import SMOTE
from collections import Counter
from scipy.stats import pearsonr
from sklearn.utils.class_weight import compute_class_weight
from sklearn.preprocessing import StandardScaler, LabelEncoder
import os
import glob
import re

def load_data(data_path, window_sizes=None, label_mapping=None):
    """
    加载数据，支持CSV文件和图文件目录两种格式
    
    Args:
        data_path: 数据路径，可以是CSV文件路径或图数据根目录
        
    Returns:
        如果是CSV: features_scaled, labels_encoded, class_weights, class_names
        如果是图目录: data_list, class_weights, class_names
    """
    if data_path.endswith('.csv'):
        # 原有的CSV加载逻辑
        data = pd.read_csv(data_path)
        features = data.loc[:, 'n1':'n43'].values
        labels = data['behavior'].values

        class_counts = Counter(labels)
        print(f"Class distribution: {class_counts}")

        scaler = StandardScaler()
        features_scaled = scaler.fit_transform(features)
        encoder = LabelEncoder()
        labels_encoded = encoder.fit_transform(labels)

        class_weights = compute_class_weight('balanced', classes=np.unique(labels_encoded), y=labels_encoded)
        class_weights = torch.FloatTensor(class_weights)
        print(f"Class weights: {class_weights}")

        return features_scaled, labels_encoded, class_weights, encoder.classes_
    else:
        # 新的图文件加载逻辑，支持按窗口过滤
        return load_graph_data_from_directory(data_path, window_sizes, label_mapping)

def load_graph_data_from_directory(data_root_path, window_sizes=None, label_mapping=None):
    """
    从图文件目录直接加载数据
    
    Args:
        data_root_path: 数据根目录路径
        window_sizes: 要加载的窗口大小列表，默认加载所有
        
    Returns:
        data_list: PyG Data对象列表
        class_weights: 类别权重
        class_names: 类别名称
    """
    
    graphs_path = os.path.join(data_root_path, 'graphs')
    
    # 获取所有窗口目录
    if window_sizes is None:
        window_dirs = glob.glob(os.path.join(graphs_path, 'window_*'))
    else:
        window_dirs = [os.path.join(graphs_path, f'window_{size}') for size in window_sizes]
        window_dirs = [d for d in window_dirs if os.path.exists(d)]
    
    print(f"Found {len(window_dirs)} window directories")
    
    # 收集所有文件和标签
    all_files = []
    all_labels = []
    
    for window_dir in window_dirs:
        if not os.path.exists(window_dir):
            continue
            
        graph_files = [f for f in os.listdir(window_dir) if f.startswith('graph_')]
        print(f"Processing {len(graph_files)} files in {os.path.basename(window_dir)}")
        
        for filename in graph_files:
            # 从文件名提取标签
            match = re.search(r'graph_\d+_\d+_(\w+)', filename)
            if match:
                label = match.group(1)
                # 应用标签映射（如果提供）
                if label_mapping and label in label_mapping:
                    label = label_mapping[label]
                all_labels.append(label)
                all_files.append(os.path.join(window_dir, filename))
    
    # 编码标签
    encoder = LabelEncoder()
    unique_labels = list(set(all_labels))
    encoder.fit(unique_labels)
    labels_encoded = encoder.transform(all_labels)
    
    print(f"Found labels: {unique_labels}")
    print(f"Label distribution: {Counter(all_labels)}")
    
    # 加载图数据并创建PyG对象
    data_list = []
    valid_labels = []
    
    for i, (file_path, label) in enumerate(zip(all_files, all_labels)):
        if i % 1000 == 0:
            print(f"Processing file {i+1}/{len(all_files)}")
            
        # 加载图数据
        edges, nodes = load_graph_from_txt_file(file_path)
        
        if len(nodes) > 0:  # 只处理非空图
            try:
                # 创建PyG数据对象
                data = create_pyg_data_from_graph(edges, nodes, label, encoder)
                data_list.append(data)
                valid_labels.append(encoder.transform([label])[0])
            except Exception as e:
                print(f"Error processing {file_path}: {e}")
                continue
    
    # 计算类别权重
    if valid_labels:
        # 确保所有类别都有对应的权重
        unique_valid_labels = np.unique(valid_labels)
        class_weights = compute_class_weight('balanced', classes=unique_valid_labels, y=valid_labels)
        
        # 创建完整的权重张量，为所有可能的类别分配权重
        full_class_weights = torch.ones(len(encoder.classes_))
        for i, class_idx in enumerate(unique_valid_labels):
            full_class_weights[class_idx] = class_weights[i]
        
        class_weights = full_class_weights
    else:
        class_weights = torch.ones(len(encoder.classes_))
    
    print(f"\nDataset statistics:")
    print(f"Total valid graphs: {len(data_list)}")
    print(f"Class weights: {class_weights}")
    
    return data_list, class_weights, encoder.classes_

def load_graph_from_txt_file(file_path):
    """
    从txt文件加载图数据
    
    Args:
        file_path: 图文件路径
        
    Returns:
        edges: 边列表 [(src, dst), ...]
        nodes: 节点集合
    """
    edges = []
    nodes = set()
    
    try:
        with open(file_path, 'r') as f:
            for line in f:
                line = line.strip()
                if line:
                    parts = line.split()
                    if len(parts) >= 2:
                        src, dst = int(parts[0]), int(parts[1])
                        edges.append((src, dst))
                        nodes.add(src)
                        nodes.add(dst)
    except Exception as e:
        print(f"Error loading graph from {file_path}: {e}")
        return [], set()
        
    return edges, nodes

def create_pyg_data_from_graph(edges, nodes, label, label_encoder):
    """
    从图数据创建PyG Data对象
    
    Args:
        edges: 边列表
        nodes: 节点集合
        label: 标签字符串
        label_encoder: 标签编码器
        
    Returns:
        data: PyG Data对象
    """
    if len(nodes) == 0 or len(edges) == 0:
        # 处理空图的情况
        nodes = {0}
        edges = [(0, 0)]  # 添加自环
    
    # 创建节点映射（确保从0开始连续）
    node_mapping = {}
    sorted_nodes = sorted(nodes)
    for i, node in enumerate(sorted_nodes):
        node_mapping[node] = i
    
    # 重新映射边
    edge_src = [node_mapping[src] for src, dst in edges]
    edge_dst = [node_mapping[dst] for src, dst in edges]
    
    edge_index = torch.tensor([edge_src, edge_dst], dtype=torch.long)
    
    # 创建节点特征（随机特征模拟钙离子浓度）
    num_nodes = len(sorted_nodes)
    x = torch.randn(num_nodes, 1)  # 每个节点1维特征
    
    # 编码标签
    y = torch.tensor(label_encoder.transform([label])[0], dtype=torch.long)
    
    # 创建PyG Data对象
    data = Data(
        x=x,
        edge_index=edge_index,
        y=y,
        num_nodes=num_nodes
    )
    
    return data

def load_graph_data(data_root_path, window_sizes=None, node_feature_method='random'):
    """
    使用内置函数加载图数据（替代原来的GraphDataAdapter）
    
    Args:
        data_root_path: 数据根目录路径
        window_sizes: 要加载的窗口大小列表
        node_feature_method: 节点特征生成方法
        
    Returns:
        data_list: PyG Data对象列表
        class_weights: 类别权重
        class_names: 类别名称
    """
    return load_graph_data_from_directory(data_root_path, window_sizes)

def oversample_data(features, labels, ramdom_state):
    smote = SMOTE(random_state=ramdom_state)
    features_resampled, labels_resampled = smote.fit_resample(features, labels)
    print("SMOTE 后样本分布:", Counter(labels_resampled))
    return features_resampled, labels_resampled

# 计算特征之间的相关性矩阵, Pearson 相关系数
def compute_correlation_matrix(features):
    features_T = features.T
    num_neurons = features_T.shape[0]
    correlation_matrix = np.zeros((num_neurons, num_neurons))

    for i in range(num_neurons):
        for j in range(i, num_neurons):
            corr, _ = pearsonr(features_T[i], features_T[j])
            correlation_matrix[i, j] = corr
            correlation_matrix[j, i] = corr

    return correlation_matrix

def generate_graph(sample_features, correlation_matrix, threshold=0.4):
    num_neurons = len(sample_features)
    edges_src = [] # 源节点
    edges_dst = [] # 目标节点
    edges_weights = []

    for i in range(num_neurons):
        for j in range(i+1, num_neurons):
            corr = correlation_matrix[i, j]
            if abs(corr) > threshold:
                edges_src.append(i)
                edges_dst.append(j)
                edges_weights.append(abs(corr))
                # 无向图，所以需要添加反向边
                edges_src.append(j)
                edges_dst.append(i)
                edges_weights.append(abs(corr))

    # 如果没有边，则添加一些简单的边
    if len(edges_src) == 0:
        print("No edges found in the graph here!")
        for i in range(num_neurons):
            j = (i + 1) % num_neurons
            edges_src.extend([i, j])
            edges_dst.extend([j, i])
            edges_weights.extend([0.1, 0.1])

    # 将边的列表转换为 PyTorch 张量，(2, num_edges)
    edge_index = torch.tensor([edges_src, edges_dst], dtype=torch.long)
    # 将边的权值转换为 PyTorch 张量，(num_edges,)
    edge_attr = torch.tensor(edges_weights, dtype=torch.float)

    return edge_index, edge_attr

# 生成 PyG 数据对象
def create_pyg_dataset(features, labels, correlation_matrix, threshold=0.4):
    data_list = []
    for i in range(len(features)):
        sample = features[i]
        x = torch.tensor(sample.reshape(-1, 1), dtype=torch.float)  # 钙离子浓度 (num_neurons, 1)
        edge_index, edge_attr = generate_graph(sample, correlation_matrix, threshold)
        data = Data(
            x=x, # 钙离子浓度, (num_neurons, 1)
            edge_index=edge_index, # 边索引, (2, num_edges)
            edge_attr=edge_attr, # 边权值, (num_edges,)
            y=torch.tensor(labels[i], dtype=torch.long), # 标签, (1,)
        )
        
        data_list.append(data)
    return data_list
    
# 将生成的拓扑图可视化
def visualize_graph(data, sample_index=0, title="神经元连接图", result_dir='result'):
    plt.figure(figsize=(10, 10))
    graph_data = data[sample_index]
    
    G = nx.Graph()
    for i in range(graph_data.x.shape[0]):
        node_value = float(graph_data.x[i][0])
        G.add_node(i, value=node_value)
    for i in range(graph_data.edge_index.shape[1]):
        src = int(graph_data.edge_index[0, i])
        dst = int(graph_data.edge_index[1, i])
        weight = float(graph_data.edge_attr[i]) if graph_data.edge_attr is not None else 1.0
        G.add_edge(src, dst, weight=weight)

    try:
        pos = nx.kamada_kawai_layout(G)
    except:
        pos = nx.spring_layout(G, seed=42)

    # 获取节点值以用于颜色映射
    node_values = [G.nodes[i]['value'] for i in range(len(G.nodes))]
    vmin = min(node_values)
    vmax = max(node_values)
    
    # 根据边权重确定边的宽度
    edge_weights = [G.edges[edge]['weight'] * 3 for edge in G.edges]
    
    # 创建颜色映射
    cmap = plt.cm.coolwarm

    # 绘制节点
    nodes = nx.draw_networkx_nodes(
        G, pos, 
        node_color=node_values,
        cmap=cmap,
        node_size=350,
        alpha=0.9,
        vmin=vmin,
        vmax=vmax,
        edgecolors='black',
        linewidths=0.5
    )
    
    # 绘制边
    edges = nx.draw_networkx_edges(
        G, pos,
        width=edge_weights,
        edge_color='gray',
        alpha=0.6,
        connectionstyle='arc3,rad=0.1'  # 使边弯曲，避免重叠
    )
    
    # 绘制节点标签
    nx.draw_networkx_labels(
        G, pos,
        font_size=9,
        font_family='sans-serif',
        font_weight='bold'
    )
    
    # 添加颜色条
    cbar = plt.colorbar(nodes, label='钙离子浓度', shrink=0.8)
    cbar.ax.tick_params(labelsize=9)
    
    # 添加标题和信息
    behavior_label = graph_data.y.item()
    plt.title(f"{title}\n样本标签: {behavior_label}", fontsize=14, fontweight='bold')
    plt.text(0.02, 0.02, f"节点数量: {G.number_of_nodes()}, 边数量: {G.number_of_edges()}",
             transform=plt.gca().transAxes, fontsize=10)
    
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(f'{result_dir}/graph_visualization.png', dpi=300, bbox_inches='tight')
    plt.close()