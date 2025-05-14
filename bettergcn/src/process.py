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

def load_data(data_path):
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
    
    print(f"自动检测到{len(neuron_cols)}个神经元特征列: {neuron_cols[:5]}...等")
    
    # 按照编号顺序排序神经元列
    neuron_cols.sort(key=lambda x: int(x[1:]))
    
    # 提取特征和标签
    features = data.loc[:, neuron_cols].values
    labels = data['behavior'].values

    class_counts = Counter(labels)
    print("Class counts before SMOTE:", class_counts)
    
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features) 
    
    ### 选择最重要的35个特征 ### -- need further tuning 目前来看 35 效果最好
    selector = SelectKBest(f_classif, k=35)  
    features_selected = selector.fit_transform(features_scaled, labels)
    
    # feature_indices = selector.get_support(indices=True)
    # print(f"选择了{len(feature_indices)}个特征: {feature_indices}")
    
    encoder = LabelEncoder()
    labels_encoded = encoder.fit_transform(labels)
    
    class_weights = compute_class_weight('balanced', classes=np.unique(labels_encoded), y=labels_encoded)
    class_weights = torch.FloatTensor(class_weights)
    # print(f"类别权重: {class_weights}")
    
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
        
        # 为每个样本创建一个自环，保证索引在有效范围内
        sample_edge_index = torch.tensor([[0], [0]], dtype=torch.long)
        
        # 使用单节点的自环代替全局edge_index
        data_obj = Data(x=x, edge_index=sample_edge_index, y=y)
        dataset.append(data_obj)
    
    return dataset
    
def split_data(features, labels, test_size=0.2, random_state=42):
    return train_test_split(features, labels, test_size=test_size, random_state=random_state)
