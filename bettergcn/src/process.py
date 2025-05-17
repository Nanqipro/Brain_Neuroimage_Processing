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

def load_data(data_path, min_samples=50):
    # 根据文件扩展名决定使用哪种方法读取数据
    if data_path.endswith('.csv'):
        data = pd.read_csv(data_path)
    elif data_path.endswith('.xlsx') or data_path.endswith('.xls'):
        data = pd.read_excel(data_path)
    else:
        raise ValueError(f"不支持的文件格式: {data_path}，请使用.csv或.xlsx/.xls格式")
    
    # 自动检测神经元特征列（假设所有神经元列都是以'n'开头后跟数字的格式）
    neuron_cols = [col for col in data.columns if col.startswith('n') and col[1:].isdigit()]
    
    if not neuron_cols:
        raise ValueError("未找到神经元特征列，请确保神经元列名以'n'开头后跟数字")
    
    # 按照编号顺序排序神经元列
    neuron_cols.sort(key=lambda x: int(x[1:]))
    print(f"自动检测到{len(neuron_cols)}个神经元特征列: {neuron_cols[:5]}...等")
    
    # 提取特征和标签
    features = data.loc[:, neuron_cols].values
    
    if 'behavior' not in data.columns:
        raise ValueError("未找到'behavior'标签列，请确保数据包含此列")
    
    labels = data['behavior'].values

    # 统计每个标签的样本数量
    class_counts = Counter(labels)
    print(f"原始类别分布: {class_counts}")
    
    # 过滤掉样本数量少于min_samples的标签
    valid_classes = [cls for cls, count in class_counts.items() if count >= min_samples]
    if len(valid_classes) < len(class_counts):
        removed_classes = [cls for cls, count in class_counts.items() if count < min_samples]
        print(f"已移除样本数少于{min_samples}的标签: {removed_classes}")
        
        # 创建过滤后的数据掩码
        valid_mask = np.isin(labels, valid_classes)
        features = features[valid_mask]
        labels = labels[valid_mask]
        
        # 更新类别统计
        class_counts = Counter(labels)
        print(f"过滤后类别分布: {class_counts}")
    
    if len(class_counts) < 2:
        raise ValueError(f"过滤后的数据集中只有{len(class_counts)}个类别，至少需要2个类别进行分类")

    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)
    encoder = LabelEncoder()
    labels_encoded = encoder.fit_transform(labels)

    class_weights = compute_class_weight('balanced', classes=np.unique(labels_encoded), y=labels_encoded)
    class_weights = torch.FloatTensor(class_weights)
    print(f"类别权重: {class_weights}")

    return features_scaled, labels_encoded, class_weights, encoder.classes_

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
def visualize_graph(data, sample_index=0, title="Neuron Connection Graph", result_dir='result'):
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
    cbar = plt.colorbar(nodes, label='Calcium Concentration', shrink=0.8)
    cbar.ax.tick_params(labelsize=9)
    
    # 添加标题和信息
    behavior_label = graph_data.y.item()
    plt.title(f"{title}\nSample Label: {behavior_label}", fontsize=14, fontweight='bold')
    plt.text(0.02, 0.02, f"Number of Nodes: {G.number_of_nodes()}, Number of Edges: {G.number_of_edges()}",
             transform=plt.gca().transAxes, fontsize=10)
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(f'{result_dir}/graph_visualization.png', dpi=300, bbox_inches='tight')
    plt.close()