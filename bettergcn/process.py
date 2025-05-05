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

def load_data(data_path=None, features=None, labels=None, feature_select_k=35):
    """
    加载和预处理数据：可以从文件加载或直接传入特征和标签
    
    参数:
        data_path: 数据文件路径，如为None则使用传入的features和labels
        features: 特征矩阵，如果data_path为None则使用此参数
        labels: 标签数组，如果data_path为None则使用此参数
        feature_select_k: 特征选择数量，默认35
        
    返回:
        features_selected: 标准化并选择后的特征
        labels_encoded: 编码后的标签
        encoder: 标签编码器
        class_weights: 类别权重
    """
    # 从文件或内存加载数据
    if data_path is not None:
        print(f"从{data_path}加载数据...")
        try:
            # 尝试使用pandas加载
            data = pd.read_csv(data_path)
            # 确定特征列，默认假设除了'behavior'外的所有列都是特征
            feature_cols = [col for col in data.columns if col != 'behavior']
            features = data[feature_cols].values
            labels = data['behavior'].values
        except Exception as e:
            print(f"从文件加载数据出错: {e}")
            print("将尝试加载numpy数组...")
            # 尝试加载numpy文件
            try:
                data = np.load(data_path)
                features = data['features'] if isinstance(data, dict) else data
                labels = data['labels'] if isinstance(data, dict) and 'labels' in data else None
            except Exception as e2:
                print(f"加载numpy数据出错: {e2}")
                raise ValueError(f"无法加载数据文件: {data_path}")
    elif features is None or labels is None:
        raise ValueError("必须提供data_path或同时提供features和labels")
    
    # 打印类别统计
    class_counts = Counter(labels)
    print("类别统计:", class_counts)
    
    # 标准化特征
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)
    
    # 特征选择 - 如果特征太多，可以选择最重要的k个
    if feature_select_k is not None and feature_select_k < features.shape[1]:
        print(f"选择最重要的{feature_select_k}个特征...")
        try:
            selector = SelectKBest(f_classif, k=feature_select_k)
            features_selected = selector.fit_transform(features_scaled, labels)
            
            # 输出选择的特征索引（用于调试）
            feature_indices = selector.get_support(indices=True)
            print(f"选择了{len(feature_indices)}个特征: {feature_indices}")
        except Exception as e:
            print(f"特征选择出错: {e}")
            print("将使用所有特征")
            features_selected = features_scaled
    else:
        features_selected = features_scaled
    
    # 标签编码
    encoder = LabelEncoder()
    labels_encoded = encoder.fit_transform(labels)
    
    # 计算类别权重
    class_weights = compute_class_weight('balanced', classes=np.unique(labels_encoded), y=labels_encoded)
    class_weights = torch.FloatTensor(class_weights)
    print(f"类别权重: {class_weights}")
    
    return features_selected, labels_encoded, encoder, class_weights

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

def generate_graph(features, k=10, threshold=None):

    # 使用KNN构建邻接矩阵
    A = kneighbors_graph(features, k, mode='distance', include_self=False)
    
    # 将距离转换为相似度
    A.data = np.exp(-A.data)
    
    if threshold is not None:
        A.data[A.data < threshold] = 0
        A.eliminate_zeros()
    
    # 获取边的起点和终点
    rows, cols = A.nonzero()
    edge_index = torch.tensor(np.array([rows, cols]), dtype=torch.long)
    
    # print(f"图结构: {features.shape[0]}个节点, {edge_index.shape[1]}条边")
    # print(f"平均每个节点的边数: {edge_index.shape[1]/features.shape[0]:.2f}")
    
    return edge_index

def create_dataset(features, labels, edge_index):
    dataset = []
    feature_dim = features.shape[1]
    
    for i in range(len(features)):
        x = torch.tensor(features[i], dtype=torch.float).reshape(1, feature_dim)
        y = torch.tensor(labels[i], dtype=torch.long)
        
        sample_edge_index = torch.tensor([[0], [0]], dtype=torch.long)
        
        data_obj = Data(x=x, edge_index=sample_edge_index, y=y)
        dataset.append(data_obj)
    
    return dataset
    
def split_data(features, labels, test_size=0.2, random_state=42):
    return train_test_split(features, labels, test_size=test_size, random_state=random_state)

def knn_graph(features, k=10, threshold=None):
    """
    使用KNN算法构建图结构 (generate_graph的别名)
    
    参数:
        features: 特征矩阵
        k: KNN算法的k值
        threshold: 相似度阈值，低于此值的边将被移除
        
    返回:
        edge_index: 边索引张量
    """
    return generate_graph(features, k, threshold)
