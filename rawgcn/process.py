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

def load_data(data_path):
    # Check file extension and use appropriate pandas function
    if data_path.endswith('.xlsx') or data_path.endswith('.xls'):
        data = pd.read_excel(data_path)
    else:
        data = pd.read_csv(data_path)
    features = data.loc[:, 'n1':'n43'].values
    labels = data['behavior'].values

    class_counts = Counter(labels)
    print("Class counts before SMOTE:", class_counts)
    
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)
    
    encoder = LabelEncoder()
    labels_encoded = encoder.fit_transform(labels)
    
    return features_scaled, labels_encoded, encoder

# def apply_smote(features, labels, ramdom_state=42):
#     print("Applying SMOTE...")
#     smote = SMOTE(random_state=ramdom_state)
#     features_resampled, labels_resampled = smote.fit_resample(features, labels)
#     return features_resampled, labels_resampled

def generate_graph(features, threshold=0.5):
    corr_matrix = pd.DataFrame(features).corr().values
    adj_matrix = (np.abs(corr_matrix) >= threshold).astype(int)
    np.fill_diagonal(adj_matrix, 0)
    edge_index, _ = utils.from_scipy_sparse_matrix(coo_matrix(adj_matrix))
    return edge_index
    
def split_data(features, labels, test_size=0.2, random_state=42):
    return train_test_split(features, labels, test_size=test_size, random_state=random_state)

def create_dataset(features, labels, edge_index):
    dataset = []
    for i in range(len(features)):
        x = torch.tensor(features[i], dtype=torch.float).view(43, 1)
        y = torch.tensor(labels[i], dtype=torch.long)
        data_obj = Data(x=x, edge_index=edge_index, y=y)
        dataset.append(data_obj)
    return dataset
