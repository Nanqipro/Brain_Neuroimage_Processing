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

# ============================================================================
# 以下函数用于原始的神经元数据（从CSV构建图）
# 如果使用TUDataset等标准数据集，这些函数不需要使用
# ============================================================================

def load_data(data_path):
    """
    【仅用于原始神经元CSV数据】
    从CSV文件加载神经元数据并进行预处理
    
    注意: 如果使用TUDataset等标准数据集，不需要调用此函数
    """
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

def oversample_data(features, labels, ramdom_state):
    """
    【仅用于原始神经元CSV数据】
    使用SMOTE进行数据过采样
    
    注意: 如果使用TUDataset等标准数据集，不需要调用此函数
    """
    smote = SMOTE(random_state=ramdom_state)
    features_resampled, labels_resampled = smote.fit_resample(features, labels)
    print("SMOTE 后样本分布:", Counter(labels_resampled))
    return features_resampled, labels_resampled

# 计算特征之间的相关性矩阵, Pearson 相关系数
def compute_correlation_matrix(features):
    """
    【仅用于原始神经元CSV数据】
    计算神经元之间的Pearson相关系数矩阵
    
    注意: 如果使用TUDataset等标准数据集，不需要调用此函数
    """
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
    """
    【仅用于原始神经元CSV数据】
    根据相关性矩阵生成图结构
    
    注意: 如果使用TUDataset等标准数据集，不需要调用此函数
    """
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
    """
    【仅用于原始神经元CSV数据】
    从神经元特征创建PyG数据对象列表
    
    注意: 如果使用TUDataset等标准数据集，不需要调用此函数
          标准数据集已经是PyG格式，可以直接使用
    """
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


# ============================================================================
# 以下函数用于加载自定义图结构数据集（如random数据集）
# ============================================================================

def load_custom_graph_dataset(data_dir, feature_dim=16, num_classes=2, use_degree_feature=False):
    """
    【用于自定义图结构数据集】
    从目录加载图数据集，支持多种格式
    
    支持的文件格式：
    1. PyTorch格式 (.pt 或 .pth): 包含PyG Data对象列表
    2. Pickle格式 (.pkl): 包含PyG Data对象列表
    3. NetworkX格式 (.graphml): 每个文件一个图
    4. 边列表格式 (无扩展名或.txt): 每行两个整数表示一条边
    
    目录结构示例：
        data_dir/
            graphs_list.pt  # PyTorch格式：包含所有图的列表
        或
        data_dir/
            graph_0001_seed42  # 边列表格式（无扩展名）
            graph_0002_seed43
            ...
        或
        data_dir/
            graph_0001_seed42.txt  # 边列表格式（.txt扩展名）
            graph_0002_seed43.txt
            ...
    
    Args:
        data_dir: 数据集目录路径
        feature_dim: 随机节点特征维度（仅用于边列表格式）
        num_classes: 类别数（仅用于边列表格式）
        use_degree_feature: 是否使用节点度数作为特征（仅用于边列表格式）
    
    Returns:
        data_list: PyG Data对象列表
        num_features: 节点特征维度
        num_classes: 类别数
    """
    import os
    import pickle
    
    data_dir = os.path.abspath(data_dir)
    print(f"\n加载自定义图数据集: {data_dir}")
    
    # 方法1: 尝试加载单个PyTorch文件（包含所有图）
    pt_files = ['graphs_list.pt', 'graphs.pt', 'dataset.pt', 'data.pt']
    for pt_file in pt_files:
        pt_path = os.path.join(data_dir, pt_file)
        if os.path.exists(pt_path):
            print(f"  - 从文件加载: {pt_file}")
            data_list = torch.load(pt_path)
            if isinstance(data_list, list) and len(data_list) > 0:
                num_features = data_list[0].num_node_features
                labels = [data.y.item() for data in data_list]
                num_classes = len(set(labels))
                
                print(f"\n数据集信息:")
                print(f"  - 图数量: {len(data_list)}")
                print(f"  - 特征维度: {num_features}")
                print(f"  - 类别数: {num_classes}")
                print(f"  - 标签分布: {Counter(labels)}")
                
                return data_list, num_features, num_classes
    
    # 方法2: 尝试加载Pickle文件
    pkl_files = ['graphs_list.pkl', 'graphs.pkl', 'dataset.pkl', 'data.pkl']
    for pkl_file in pkl_files:
        pkl_path = os.path.join(data_dir, pkl_file)
        if os.path.exists(pkl_path):
            print(f"  - 从文件加载: {pkl_file}")
            with open(pkl_path, 'rb') as f:
                data_list = pickle.load(f)
            if isinstance(data_list, list) and len(data_list) > 0:
                num_features = data_list[0].num_node_features
                labels = [data.y.item() for data in data_list]
                num_classes = len(set(labels))
                
                print(f"\n数据集信息:")
                print(f"  - 图数量: {len(data_list)}")
                print(f"  - 特征维度: {num_features}")
                print(f"  - 类别数: {num_classes}")
                print(f"  - 标签分布: {Counter(labels)}")
                
                return data_list, num_features, num_classes
    
    # 方法3: 尝试加载多个单独的.pt文件
    graph_files = sorted([f for f in os.listdir(data_dir) if f.endswith('.pt') and f.startswith('graph_')])
    if len(graph_files) > 0:
        print(f"  - 找到 {len(graph_files)} 个单独的图文件")
        data_list = []
        for graph_file in graph_files:
            graph_path = os.path.join(data_dir, graph_file)
            graph_data = torch.load(graph_path)
            data_list.append(graph_data)
        
        num_features = data_list[0].num_node_features
        labels = [data.y.item() for data in data_list]
        num_classes = len(set(labels))
        
        print(f"\n数据集信息:")
        print(f"  - 图数量: {len(data_list)}")
        print(f"  - 特征维度: {num_features}")
        print(f"  - 类别数: {num_classes}")
        print(f"  - 标签分布: {Counter(labels)}")
        
        return data_list, num_features, num_classes
    
    # 方法4: 尝试加载GraphML文件
    graphml_files = sorted([f for f in os.listdir(data_dir) if f.endswith('.graphml')])
    if len(graphml_files) > 0:
        print(f"  - 找到 {len(graphml_files)} 个GraphML文件")
        data_list = []
        for graphml_file in graphml_files:
            graphml_path = os.path.join(data_dir, graphml_file)
            G = nx.read_graphml(graphml_path)
            
            # 转换为PyG格式
            # 假设节点特征存储在节点属性'features'中，标签存储在图属性'label'中
            x = torch.tensor([[float(G.nodes[n].get('feature', 0))] for n in G.nodes()], dtype=torch.float)
            edge_index = torch.tensor([[int(e[0]), int(e[1])] for e in G.edges()], dtype=torch.long).t()
            y = torch.tensor([int(G.graph.get('label', 0))], dtype=torch.long)
            
            data = Data(x=x, edge_index=edge_index, y=y)
            data_list.append(data)
        
        num_features = data_list[0].num_node_features
        labels = [data.y.item() for data in data_list]
        num_classes = len(set(labels))
        
        print(f"\n数据集信息:")
        print(f"  - 图数量: {len(data_list)}")
        print(f"  - 特征维度: {num_features}")
        print(f"  - 类别数: {num_classes}")
        print(f"  - 标签分布: {Counter(labels)}")
        
        return data_list, num_features, num_classes
    
    # 方法5: 尝试加载边列表格式文件（如random数据集）
    # 查找所有符合graph_*格式的文件（有或无.txt扩展名）
    all_files = os.listdir(data_dir)
    edge_list_files = sorted([f for f in all_files if f.startswith('graph_') and (not f.endswith('.pt') and not f.endswith('.pkl'))])
    
    if len(edge_list_files) > 0:
        print(f"  - 找到 {len(edge_list_files)} 个边列表文件")
        print(f"  - 使用{'节点度数' if use_degree_feature else f'{feature_dim}维随机'}特征")
        print(f"  - 生成 {num_classes} 个类别的随机标签")
        
        data_list = []
        np.random.seed(42)  # 固定随机种子以保证可重复性
        
        for idx, edge_file in enumerate(edge_list_files):
            edge_path = os.path.join(data_dir, edge_file)
            
            try:
                # 读取边列表
                edges = []
                with open(edge_path, 'r') as f:
                    for line in f:
                        line = line.strip()
                        if line:
                            parts = line.split()
                            if len(parts) >= 2:
                                src, dst = int(parts[0]), int(parts[1])
                                edges.append([src, dst])
                
                if len(edges) == 0:
                    continue
                
                # 转换为PyG格式
                edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
                
                # 获取节点数量
                num_nodes = int(edge_index.max().item()) + 1
                
                # 生成节点特征
                if use_degree_feature:
                    # 使用节点度数作为特征
                    degrees = torch.zeros(num_nodes, dtype=torch.float)
                    for i in range(edge_index.shape[1]):
                        degrees[edge_index[0, i]] += 1
                        degrees[edge_index[1, i]] += 1
                    x = degrees.unsqueeze(1)  # (num_nodes, 1)
                    actual_feature_dim = 1
                else:
                    # 使用随机特征
                    x = torch.randn(num_nodes, feature_dim, dtype=torch.float)
                    actual_feature_dim = feature_dim
                
                # 生成随机标签（基于图索引的哈希，确保一致性）
                y = torch.tensor([idx % num_classes], dtype=torch.long)
                
                data = Data(x=x, edge_index=edge_index, y=y)
                data_list.append(data)
                
            except Exception as e:
                print(f"  警告: 跳过文件 {edge_file}: {e}")
                continue
        
        if len(data_list) == 0:
            raise ValueError(f"未能从 {data_dir} 中加载任何有效的图数据")
        
        num_features = data_list[0].num_node_features
        labels = [data.y.item() for data in data_list]
        actual_num_classes = len(set(labels))
        
        print(f"\n数据集信息:")
        print(f"  - 图数量: {len(data_list)}")
        print(f"  - 节点特征维度: {num_features}")
        print(f"  - 类别数: {actual_num_classes}")
        print(f"  - 标签分布: {Counter(labels)}")
        print(f"  - 平均节点数: {np.mean([data.num_nodes for data in data_list]):.1f}")
        print(f"  - 平均边数: {np.mean([data.num_edges for data in data_list]):.1f}")
        
        return data_list, num_features, actual_num_classes
    
    raise FileNotFoundError(
        f"在目录 {data_dir} 中未找到有效的图数据集文件。\n"
        f"支持的格式：\n"
        f"  1. PyTorch格式: graphs_list.pt, graphs.pt, dataset.pt, data.pt\n"
        f"  2. Pickle格式: graphs_list.pkl, graphs.pkl, dataset.pkl, data.pkl\n"
        f"  3. 多文件格式: graph_0.pt, graph_1.pt, ...\n"
        f"  4. GraphML格式: *.graphml\n"
        f"  5. 边列表格式: graph_* (每行两个整数表示一条边)"
    )


def save_custom_graph_dataset(data_list, save_path, format='pt'):
    """
    【用于自定义图结构数据集】
    保存PyG格式图数据集到文件
    
    Args:
        data_list: PyG Data对象列表
        save_path: 保存路径（文件或目录）
        format: 保存格式 ('pt', 'pkl', 'separate')
            - 'pt': 保存为单个PyTorch文件
            - 'pkl': 保存为单个Pickle文件
            - 'separate': 保存为多个单独的.pt文件
    """
    import os
    import pickle
    
    os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)
    
    if format == 'pt':
        torch.save(data_list, save_path)
        print(f"数据集已保存到: {save_path} (PyTorch格式)")
    elif format == 'pkl':
        with open(save_path, 'wb') as f:
            pickle.dump(data_list, f)
        print(f"数据集已保存到: {save_path} (Pickle格式)")
    elif format == 'separate':
        save_dir = save_path
        os.makedirs(save_dir, exist_ok=True)
        for i, data in enumerate(data_list):
            graph_path = os.path.join(save_dir, f'graph_{i}.pt')
            torch.save(data, graph_path)
        print(f"数据集已保存到: {save_dir} ({len(data_list)}个单独文件)")
    else:
        raise ValueError(f"不支持的格式: {format}，请使用 'pt', 'pkl', 或 'separate'")


# ============================================================================
# 以下函数是通用的可视化工具，可以用于任何PyG图数据
# ============================================================================

# 将生成的拓扑图可视化
def visualize_graph(data, sample_index=0, title="神经元连接图", result_dir='result'):
    """
    【通用函数】
    可视化PyG图数据
    
    适用于任何PyG格式的图数据，包括TUDataset
    """
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