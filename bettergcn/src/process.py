# import pandas as pd
# import numpy as np
# from sklearn.preprocessing import StandardScaler, LabelEncoder
# from sklearn.model_selection import train_test_split
# import torch
# import torch_geometric.utils as utils
# from scipy.sparse import coo_matrix
# from torch_geometric.data import Data, DataLoader
# from imblearn.over_sampling import SMOTE
# from collections import Counter
# from sklearn.utils.class_weight import compute_class_weight
# from sklearn.feature_selection import SelectKBest, f_classif
# from sklearn.neighbors import kneighbors_graph

# def load_data(data_path):
#     # 根据文件扩展名决定使用哪种方法读取数据
#     if data_path.endswith('.csv'):
#         data = pd.read_csv(data_path)
#     elif data_path.endswith('.xlsx') or data_path.endswith('.xls'):
#         data = pd.read_excel(data_path)
#     else:
#         raise ValueError(f"不支持的文件格式: {data_path}，请使用.csv或.xlsx/.xls格式")
    
#     # 自动检测神经元特征列（假设所有神经元列都是以'n'开头后跟数字的格式）
#     neuron_cols = [col for col in data.columns if col.startswith('n') and col[1:].isdigit()]
    
#     if not neuron_cols:
#         raise ValueError("未找到神经元特征列，请确保神经元列名以'n'开头后跟数字")
    
#     print(f"自动检测到{len(neuron_cols)}个神经元特征列: {neuron_cols[:5]}...等")
    
#     # 按照编号顺序排序神经元列
#     neuron_cols.sort(key=lambda x: int(x[1:]))
    
#     # 提取特征和标签
#     features = data.loc[:, neuron_cols].values
#     labels = data['behavior'].values

#     class_counts = Counter(labels)
#     print("Class counts before SMOTE:", class_counts)
    
#     scaler = StandardScaler()
#     features_scaled = scaler.fit_transform(features) 
    
#     ### 选择最重要的35个特征 ### -- need further tuning 目前来看 35 效果最好
#     selector = SelectKBest(f_classif, k=35)  
#     features_selected = selector.fit_transform(features_scaled, labels)
    
#     # feature_indices = selector.get_support(indices=True)
#     # print(f"选择了{len(feature_indices)}个特征: {feature_indices}")
    
#     encoder = LabelEncoder()
#     labels_encoded = encoder.fit_transform(labels)
    
#     class_weights = compute_class_weight('balanced', classes=np.unique(labels_encoded), y=labels_encoded)
#     class_weights = torch.FloatTensor(class_weights)
#     # print(f"类别权重: {class_weights}")
    
#     return features_selected, labels_encoded, encoder, class_weights

# def apply_smote(features, labels, random_state=42):
#     print("applying smote")
    
#     class_counts = Counter(labels)
#     min_samples = min(class_counts.values())
    
#     # 根据最小样本数调整k_neighbors
#     k_neighbors = min(5, min_samples - 1)
    
#     try:
#         # SMOTE
#         smote = SMOTE(random_state=random_state, k_neighbors=k_neighbors)
#         features_resampled, labels_resampled = smote.fit_resample(features, labels)
        
#         print(f"after SMOTE: {Counter(labels_resampled)}")
#         return features_resampled, labels_resampled
#     except Exception as e:
#         print(f"SMOTE: {e}")
#         print("return original data")
#         return features, labels

# def generate_graph(features, k=10, threshold=None):

#     # 使用KNN构建邻接矩阵
#     A = kneighbors_graph(features, k, mode='distance', include_self=False)
    
#     # 将距离转换为相似度
#     A.data = np.exp(-A.data)
    
#     if threshold is not None:
#         A.data[A.data < threshold] = 0
#         A.eliminate_zeros()
    
#     # 获取边的起点和终点
#     rows, cols = A.nonzero()
#     edge_index = torch.tensor(np.array([rows, cols]), dtype=torch.long)
    
#     # print(f"图结构: {features.shape[0]}个节点, {edge_index.shape[1]}条边")
#     # print(f"平均每个节点的边数: {edge_index.shape[1]/features.shape[0]:.2f}")
    
#     return edge_index

# def create_dataset(features, labels, edge_index):
#     dataset = []
#     feature_dim = features.shape[1]
    
#     for i in range(len(features)):
#         x = torch.tensor(features[i], dtype=torch.float).reshape(1, feature_dim)
#         y = torch.tensor(labels[i], dtype=torch.long)
        
#         # 为每个样本创建一个自环，保证索引在有效范围内
#         sample_edge_index = torch.tensor([[0], [0]], dtype=torch.long)
        
#         # 使用单节点的自环代替全局edge_index
#         data_obj = Data(x=x, edge_index=sample_edge_index, y=y)
#         dataset.append(data_obj)
    
#     return dataset
    
# def split_data(features, labels, test_size=0.2, random_state=42):
#     return train_test_split(features, labels, test_size=test_size, random_state=random_state)

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import torch
import torch_geometric.utils as utils
from scipy.sparse import coo_matrix
from torch_geometric.data import Data, DataLoader
from imblearn.over_sampling import SMOTE
from collections import Counter
from sklearn.utils.class_weight import compute_class_weight
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.neighbors import kneighbors_graph

def generate_graph(node_features_for_one_graph, k=10, threshold=None):
    """
    Generates an edge_index for a single graph based on its node features.
    node_features_for_one_graph: 2D numpy array (num_nodes, num_node_features)
    k: Desired number of neighbors for KNN.
    threshold: Similarity threshold to prune edges.
    """
    num_nodes = node_features_for_one_graph.shape[0]

    if num_nodes == 0:
        return torch.empty((2, 0), dtype=torch.long)
    if num_nodes == 1: # Single node graph, can have a self-loop or no edges.
        return torch.tensor([[0], [0]], dtype=torch.long) # Example: self-loop

    # Adjust k for kneighbors_graph: k must be < num_nodes.
    actual_k = min(k, num_nodes - 1)
    
    if actual_k <= 0: # Cannot perform KNN if k is not positive or no other nodes exist
        return torch.empty((2, 0), dtype=torch.long) # Return graph with no edges

    try:
        A = kneighbors_graph(node_features_for_one_graph, 
                             n_neighbors=actual_k, 
                             mode='distance', 
                             include_self=False)
        A.data = np.exp(-A.data) # Convert distance to similarity
        
        if threshold is not None:
            A.data[A.data < threshold] = 0
            A.eliminate_zeros()
        
        rows, cols = A.nonzero()
        edge_index = torch.tensor(np.array([rows, cols]), dtype=torch.long)
    except Exception as e:
        # print(f"Warning: Graph generation failed for a sample: {e}. Returning empty edge_index.")
        edge_index = torch.empty((2, 0), dtype=torch.long)
        
    return edge_index


def load_data(data_path, 
              num_nodes_per_graph=35, # Assumption: 35 selected features form 35 nodes
              node_feature_dim=1,     # Assumption: Each of these nodes has 1 feature
              k_for_graph_gen=5,      # k for KNN when generating each graph's internal edges
              threshold_for_graph_gen=0.1): # Threshold for edge generation
    data = pd.read_csv(data_path)
    features_raw = data.loc[:, 'n1':'n43'].values # Renamed to avoid confusion
    labels_raw = data['behavior'].values # Renamed

    class_counts_initial = Counter(labels_raw)
    print("Initial class counts:", class_counts_initial)
    
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features_raw) 
    
    # Assuming k_selected_features is the first dimension of your node features for each graph
    # or will be reshaped into num_nodes_per_graph * node_feature_dim
    k_selected_features = num_nodes_per_graph * node_feature_dim 
    if k_selected_features > features_scaled.shape[1]:
        print(f"Warning: Requested {k_selected_features} features for graph nodes, but only {features_scaled.shape[1]} available after scaling. Using all available.")
        k_selected_features = features_scaled.shape[1]
        # You might need to adjust num_nodes_per_graph or node_feature_dim if this happens
        if node_feature_dim > 0 :
             num_nodes_per_graph = k_selected_features // node_feature_dim # Re-calculate
        else: # Should not happen if node_feature_dim is validated
             num_nodes_per_graph = k_selected_features


    # Select KBest features that will form the (flat) feature vector for each graph
    # If k_selected_features is, for example, 35, this selects 35 features.
    selector = SelectKBest(f_classif, k=k_selected_features)  
    # `labels_raw` should be used here for fitting selector, not encoded ones yet.
    flat_features_per_graph = selector.fit_transform(features_scaled, labels_raw)
    
    encoder = LabelEncoder()
    labels_encoded = encoder.fit_transform(labels_raw)
    
    class_weights = compute_class_weight('balanced', classes=np.unique(labels_encoded), y=labels_encoded)
    class_weights_tensor = torch.FloatTensor(class_weights)
    all_individual_edge_indices = []
    for i in range(flat_features_per_graph.shape[0]):
        # Reshape the flat features of the current graph into its node feature matrix
        # For example, if flat_features_per_graph[i] is (35,)
        # and num_nodes_per_graph=35, node_feature_dim=1, this becomes (35, 1)
        if node_feature_dim == 0: # Avoid division by zero if misconfigured
            print("Warning: node_feature_dim is 0. Cannot reshape node features.")
            node_features_current_graph_np = np.empty((0,0))
        elif flat_features_per_graph[i].size == 0: # No features for this graph
             node_features_current_graph_np = np.empty((0,0))
        else:
            try:
                node_features_current_graph_np = flat_features_per_graph[i].reshape(num_nodes_per_graph, node_feature_dim)
            except ValueError as e:
                raise ValueError(f"Cannot reshape flat features of size {flat_features_per_graph[i].size} "
                                 f"into ({num_nodes_per_graph}, {node_feature_dim}). Error: {e}. "
                                 "Ensure num_nodes_per_graph * node_feature_dim == k_selected_features.")

        current_graph_edge_index = generate_graph(
            node_features_current_graph_np, 
            k=k_for_graph_gen, 
            threshold=threshold_for_graph_gen
        )
        all_individual_edge_indices.append(current_graph_edge_index)
    
    return flat_features_per_graph, labels_encoded, encoder, class_weights_tensor, all_individual_edge_indices

def apply_smote(features, labels, random_state=42):
    print("applying smote")
    
    class_counts = Counter(labels)
    min_samples = min(class_counts.values())
    
    # 根据最小样本数调整k_neighbors
    k_neighbors = min(5, min_samples - 1)
    
    try:
        # SMOTE
        smote = SMOTE(random_state=random_state, k_neighbors=k_neighbors)
        features_resampled, labels_resampled = smote.fit_resample(features, labels)
        
        print(f"after SMOTE: {Counter(labels_resampled)}")
        return features_resampled, labels_resampled
    except Exception as e:
        print(f"SMOTE: {e}")
        print("return original data")
        return features, labels


def create_dataset(all_graphs_flat_features, 
                   all_graphs_labels, 
                   all_individual_edge_indices, # New parameter
                   num_nodes_per_graph=35,      # Must match assumption in load_data
                   node_feature_dim=1):         # Must match assumption in load_data
    dataset = []
    if not (len(all_graphs_flat_features) == len(all_graphs_labels) == len(all_individual_edge_indices)):
        raise ValueError("Mismatch in lengths of features, labels, and edge_indices lists.")

    for i in range(len(all_graphs_flat_features)):
        current_flat_features = all_graphs_flat_features[i]
        
        # Reshape flat features to node feature matrix for the current graph
        if node_feature_dim == 0:
            node_feature_matrix_np = np.empty((0,0)) # Or handle as error
        elif current_flat_features.size == 0:
            node_feature_matrix_np = np.empty((0,0))
        else:
            try:
                node_feature_matrix_np = current_flat_features.reshape(num_nodes_per_graph, node_feature_dim)
            except ValueError as e:
                 raise ValueError(f"Cannot reshape flat features of size {current_flat_features.size} for graph {i} "
                                 f"into ({num_nodes_per_graph}, {node_feature_dim}). Error: {e}")

        x_tensor = torch.tensor(node_feature_matrix_np, dtype=torch.float)
        y_tensor = torch.tensor(all_graphs_labels[i], dtype=torch.long)
        current_edge_index = all_individual_edge_indices[i] # Use the pre-generated edge_index
        
        data_obj = Data(x=x_tensor, edge_index=current_edge_index, y=y_tensor)
        dataset.append(data_obj)
    
    return dataset
    
def split_data(features, labels, edge_indices_list, test_size=0.2, random_state=42):
    # Ensure labels are available for stratification if there's more than one class
    stratify_on = None
    if labels is not None and len(np.unique(labels)) > 1:
        stratify_on = labels

    # Split features and labels
    features_train, features_test, labels_train, labels_test = train_test_split(
        features, labels, test_size=test_size, random_state=random_state, stratify=stratify_on
    )
    # Split edge_indices_list accordingly
    # This requires knowing the indices from the train_test_split, or splitting indices first
    # A simpler way if not needing perfect stratification for edge_indices:
    if stratify_on is not None: # If stratified, indices are shuffled, so we need to split indices too
        indices = np.arange(len(features))
        train_idx, test_idx = train_test_split(indices, test_size=test_size, random_state=random_state, stratify=labels)
        
        edge_indices_train = [edge_indices_list[i] for i in train_idx]
        edge_indices_test = [edge_indices_list[i] for i in test_idx]
        
        # features_train, labels_train are already derived from these indices by sklearn
        # but to be sure they align, re-index features and labels based on train_idx, test_idx
        features_train = features[train_idx]
        labels_train = labels[train_idx]
        features_test = features[test_idx]
        labels_test = labels[test_idx]

    else:
        indices = np.arange(len(features))
        train_idx, test_idx = train_test_split(indices, test_size=test_size, random_state=random_state, shuffle=True) # shuffle=True is default

        edge_indices_train = [edge_indices_list[i] for i in train_idx]
        edge_indices_test = [edge_indices_list[i] for i in test_idx]
        
        features_train = features[train_idx]
        labels_train = labels[train_idx]
        features_test = features[test_idx]
        labels_test = labels[test_idx]


    return features_train, features_test, labels_train, labels_test, edge_indices_train, edge_indices_test