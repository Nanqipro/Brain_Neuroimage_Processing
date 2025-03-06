import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch_geometric.nn import GCNConv, GATv2Conv, SAGEConv, TransformerConv
from torch_geometric.nn import global_mean_pool, global_max_pool, global_add_pool
from torch_geometric.data import Data
import networkx as nx

class BrainNodeFeatureExtractor:
    """
    实现BrainGB中的节点特征提取方法
    """
    def __init__(self, feature_type='connection_profile'):
        """
        初始化节点特征提取器
        
        参数:
            feature_type: 特征类型，可选 'connection_profile', 'eigen', 'degree', 'identity'
        """
        self.feature_type = feature_type
        
    def extract_features(self, adj_matrix):
        """
        根据指定类型提取节点特征
        
        参数:
            adj_matrix: 邻接矩阵
            
        返回:
            node_features: 节点特征矩阵
        """
        n_nodes = adj_matrix.shape[0]
        
        if self.feature_type == 'connection_profile':
            # 使用连接配置文件作为特征（在ABCD数据集上AUC达91.33%）
            return adj_matrix
            
        elif self.feature_type == 'eigen':
            # 使用特征向量作为特征（适合小样本数据）
            k = min(10, n_nodes - 1)  # 取前k个特征向量
            try:
                eigenvalues, eigenvectors = np.linalg.eigh(adj_matrix)
                # 取最大的k个特征值对应的特征向量
                indices = np.argsort(eigenvalues)[-k:]
                return eigenvectors[:, indices]
            except np.linalg.LinAlgError:
                print("特征分解失败，返回连接配置文件作为备选")
                return adj_matrix
                
        elif self.feature_type == 'degree':
            # 使用度配置文件作为特征
            degree = adj_matrix.sum(axis=1, keepdims=True)
            # 计算邻居度的统计特征
            neighbor_degrees = []
            for i in range(n_nodes):
                neighbors = np.where(adj_matrix[i] > 0)[0]
                if len(neighbors) > 0:
                    neighbor_deg = degree[neighbors].flatten()
                    stats = [np.min(neighbor_deg), np.max(neighbor_deg), 
                             np.mean(neighbor_deg), np.std(neighbor_deg)]
                else:
                    stats = [0, 0, 0, 0]
                neighbor_degrees.append(stats)
                
            neighbor_degrees = np.array(neighbor_degrees)
            # 返回[度, 邻居度min, 邻居度max, 邻居度mean, 邻居度std]
            return np.hstack([degree, neighbor_degrees])
            
        elif self.feature_type == 'identity':
            # 使用独热编码作为特征（最基础的方法）
            return np.eye(n_nodes)
            
        else:
            raise ValueError(f"不支持的特征类型: {self.feature_type}")


class MessagePassingLayer(nn.Module):
    """
    实现BrainGB中的不同消息传递机制
    """
    def __init__(self, in_channels, out_channels, message_type='node_edge_concat'):
        """
        初始化消息传递层
        
        参数:
            in_channels: 输入特征维度
            out_channels: 输出特征维度
            message_type: 消息传递类型，可选 'edge_weighted', 'bin_concat', 'node_edge_concat', 'node_concat'
        """
        super(MessagePassingLayer, self).__init__()
        self.message_type = message_type
        
        if message_type == 'edge_weighted':
            self.conv = GCNConv(in_channels, out_channels)
            
        elif message_type == 'bin_concat':
            self.conv = TransformerConv(in_channels, out_channels)
            
        elif message_type == 'node_edge_concat':
            self.conv = GATv2Conv(in_channels, out_channels, edge_dim=1)
            
        elif message_type == 'node_concat':
            self.conv = SAGEConv(in_channels, out_channels)
            
        else:
            raise ValueError(f"不支持的消息传递类型: {message_type}")
    
    def forward(self, x, edge_index, edge_attr=None):
        """
        前向传播
        
        参数:
            x: 节点特征
            edge_index: 边索引
            edge_attr: 边属性
            
        返回:
            x: 更新后的节点特征
        """
        if self.message_type == 'edge_weighted':
            return self.conv(x, edge_index, edge_attr)
        
        elif self.message_type == 'bin_concat':
            return self.conv(x, edge_index, edge_attr)
        
        elif self.message_type == 'node_edge_concat':
            return self.conv(x, edge_index, edge_attr)
        
        elif self.message_type == 'node_concat':
            return self.conv(x, edge_index)


class BrainGNN(nn.Module):
    """
    基于BrainGB论文的脑图神经网络模型
    """
    def __init__(self, in_channels, hidden_channels, out_channels, message_type='node_edge_concat', 
                 num_layers=2, dropout=0.2, pooling='concat'):
        """
        初始化BrainGNN模型
        
        参数:
            in_channels: 输入特征维度
            hidden_channels: 隐藏层维度
            out_channels: 输出类别数
            message_type: 消息传递类型
            num_layers: GNN层数
            dropout: Dropout比率
            pooling: 池化类型，可选 'concat', 'mean', 'max', 'add'
        """
        super(BrainGNN, self).__init__()
        
        self.num_layers = num_layers
        self.pooling = pooling
        self.dropout = dropout
        
        # GNN层
        self.convs = nn.ModuleList()
        self.batch_norms = nn.ModuleList()
        
        self.convs.append(MessagePassingLayer(in_channels, hidden_channels, message_type))
        self.batch_norms.append(nn.BatchNorm1d(hidden_channels))
        
        for _ in range(num_layers - 1):
            self.convs.append(MessagePassingLayer(hidden_channels, hidden_channels, message_type))
            self.batch_norms.append(nn.BatchNorm1d(hidden_channels))
            
        # 处理池化结果的分类层
        if pooling == 'concat':
            # 假设最大有100个节点
            max_nodes = 100
            self.classifier = nn.Sequential(
                nn.Linear(hidden_channels * max_nodes, hidden_channels * 2),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.BatchNorm1d(hidden_channels * 2),
                nn.Linear(hidden_channels * 2, hidden_channels),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.BatchNorm1d(hidden_channels),
                nn.Linear(hidden_channels, out_channels)
            )
        else:
            self.classifier = nn.Sequential(
                nn.Linear(hidden_channels, hidden_channels),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.BatchNorm1d(hidden_channels),
                nn.Linear(hidden_channels, out_channels)
            )
    
    def forward(self, data):
        """
        前向传播
        
        参数:
            data: PyG数据对象，包含x, edge_index, edge_attr等属性
            
        返回:
            out: 输出预测
        """
        x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch
        
        # 消息传递
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index, edge_attr)
            x = self.batch_norms[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
            
        # 池化
        if self.pooling == 'concat':
            # 获取每个图的节点数
            num_nodes = torch.bincount(batch)
            max_nodes = num_nodes.max().item()
            
            # 初始化结果张量
            batch_size = batch.max().item() + 1
            result = torch.zeros(batch_size, max_nodes * x.size(1), device=x.device)
            
            # 为每个图填充节点特征
            node_idx = 0
            for i in range(batch_size):
                n_nodes = num_nodes[i].item()
                # 取当前图的节点特征
                features = x[node_idx:node_idx+n_nodes]
                # 如果节点数小于最大值，填充零
                if n_nodes < max_nodes:
                    padding = torch.zeros(max_nodes - n_nodes, x.size(1), device=x.device)
                    features = torch.cat([features, padding], dim=0)
                # 展平并存储
                result[i] = features.reshape(1, -1)
                node_idx += n_nodes
            
            # 分类
            out = self.classifier(result)
            
        elif self.pooling == 'mean':
            pooled = global_mean_pool(x, batch)
            out = self.classifier(pooled)
            
        elif self.pooling == 'max':
            pooled = global_max_pool(x, batch)
            out = self.classifier(pooled)
            
        elif self.pooling == 'add':
            pooled = global_add_pool(x, batch)
            out = self.classifier(pooled)
            
        else:
            raise ValueError(f"不支持的池化类型: {self.pooling}")
            
        return out
        
        
class NeuronGNNLSTM(nn.Module):
    """
    将GNN和LSTM结合的混合模型，用于神经元网络时序分析
    """
    def __init__(self, input_size, gnn_hidden_size, lstm_hidden_size, num_layers, num_classes,
                 message_type='node_edge_concat', gnn_layers=2, latent_dim=32, 
                 num_heads=4, dropout=0.2, pooling='mean'):
        """
        初始化混合模型
        
        参数:
            input_size: 输入特征维度
            gnn_hidden_size: GNN隐藏层维度
            lstm_hidden_size: LSTM隐藏层维度
            num_layers: LSTM层数
            num_classes: 输出类别数
            message_type: GNN消息传递类型
            gnn_layers: GNN层数
            latent_dim: 潜在特征维度
            num_heads: 注意力头数
            dropout: Dropout比率
            pooling: GNN池化类型
        """
        super(NeuronGNNLSTM, self).__init__()
        
        self.input_size = input_size
        self.gnn_hidden_size = gnn_hidden_size
        self.lstm_hidden_size = lstm_hidden_size
        self.num_layers = num_layers
        self.num_classes = num_classes
        
        # 确保隐藏层大小是多头注意力头数的倍数
        if lstm_hidden_size % num_heads != 0:
            adjusted_hidden_size = (lstm_hidden_size // num_heads) * num_heads
            print(f"警告: 隐藏层大小({lstm_hidden_size})不是头数({num_heads})的倍数，调整为{adjusted_hidden_size}")
            lstm_hidden_size = adjusted_hidden_size
            self.lstm_hidden_size = lstm_hidden_size
        
        # GNN部分
        self.gnn = BrainGNN(
            in_channels=input_size,
            hidden_channels=gnn_hidden_size,
            out_channels=latent_dim,
            message_type=message_type,
            num_layers=gnn_layers,
            dropout=dropout,
            pooling=pooling
        )
        
        # LSTM部分
        self.lstm = nn.LSTM(
            input_size=latent_dim,
            hidden_size=lstm_hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        # 多头注意力机制
        self.multihead_attention = nn.MultiheadAttention(
            embed_dim=lstm_hidden_size * 2,
            num_heads=num_heads,
            dropout=dropout
        )
        
        # 分类器
        self.classifier = nn.Sequential(
            nn.Linear(lstm_hidden_size * 2, lstm_hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.BatchNorm1d(lstm_hidden_size),
            nn.Linear(lstm_hidden_size, lstm_hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.BatchNorm1d(lstm_hidden_size // 2),
            nn.Linear(lstm_hidden_size // 2, num_classes)
        )
        
    def forward(self, time_series_data, graph_data_list):
        """
        前向传播
        
        参数:
            time_series_data: 时间序列数据 [batch_size, seq_len, num_neurons]
            graph_data_list: 每个时间点的图数据列表 [seq_len个PyG数据对象]
            
        返回:
            output: 分类结果
            attention_weights: 注意力权重
        """
        batch_size, seq_len, _ = time_series_data.shape
        
        # 使用GNN处理每个时间点的图
        gnn_outputs = []
        for t in range(seq_len):
            # 处理当前时间点的图
            data_t = graph_data_list[t]
            gnn_out = self.gnn(data_t)
            gnn_outputs.append(gnn_out)
            
        # 将GNN输出堆叠成序列
        gnn_sequence = torch.stack(gnn_outputs, dim=1)  # [batch_size, seq_len, latent_dim]
        
        # 通过LSTM处理序列
        lstm_out, _ = self.lstm(gnn_sequence)
        
        # 应用多头注意力 - 转换维度以适应nn.MultiheadAttention
        lstm_out_t = lstm_out.transpose(0, 1)  # [seq_len, batch_size, lstm_hidden_size*2]
        attended, attention_weights = self.multihead_attention(
            lstm_out_t, lstm_out_t, lstm_out_t
        )
        
        # 取最后一个时间步的输出
        last_hidden = attended[-1]  # [batch_size, lstm_hidden_size*2]
        
        # 分类
        output = self.classifier(last_hidden)
        
        return output, attention_weights


def convert_networkx_to_pyg(G, node_features=None):
    """
    将NetworkX图转换为PyG数据对象
    
    参数:
        G: NetworkX图
        node_features: 节点特征字典 {node_id: feature_vector}
        
    返回:
        data: PyG数据对象
    """
    # 提取边索引和边属性
    edge_index = []
    edge_attr = []
    
    for u, v, data in G.edges(data=True):
        # 获取节点索引
        u_idx = list(G.nodes()).index(u)
        v_idx = list(G.nodes()).index(v)
        
        # 添加边
        edge_index.append([u_idx, v_idx])
        edge_index.append([v_idx, u_idx])  # 添加反向边
        
        # 添加边属性
        weight = data.get('weight', 1.0)
        edge_attr.append([weight])
        edge_attr.append([weight])  # 反向边的权重
    
    # 转换为张量
    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
    edge_attr = torch.tensor(edge_attr, dtype=torch.float)
    
    # 处理节点特征
    if node_features is None:
        # 使用度作为默认特征
        x = torch.tensor([[G.degree(node)] for node in G.nodes()], dtype=torch.float)
    else:
        # 使用提供的特征
        x = torch.tensor([node_features[node] for node in G.nodes()], dtype=torch.float)
    
    # 创建数据对象
    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
    
    return data


def create_dynamic_graph_data(correlation_matrices, time_steps, threshold=0.3, feature_type='connection_profile'):
    """
    创建时间序列动态图数据
    
    参数:
        correlation_matrices: 时间序列相关矩阵列表
        time_steps: 时间步数
        threshold: 相关性阈值
        feature_type: 节点特征类型
        
    返回:
        graph_data_list: PyG数据对象列表
    """
    graph_data_list = []
    feature_extractor = BrainNodeFeatureExtractor(feature_type)
    
    for t in range(time_steps):
        corr_matrix = correlation_matrices[t]
        
        # 提取节点特征
        node_features = feature_extractor.extract_features(corr_matrix)
        
        # 创建NetworkX图
        G = nx.Graph()
        
        # 添加节点
        n_nodes = corr_matrix.shape[0]
        for i in range(n_nodes):
            G.add_node(i, features=node_features[i])
        
        # 添加边
        for i in range(n_nodes):
            for j in range(i+1, n_nodes):
                if abs(corr_matrix[i, j]) >= threshold:
                    G.add_edge(i, j, weight=abs(corr_matrix[i, j]))
        
        # 转换为PyG数据
        data = convert_networkx_to_pyg(G, {i: node_features[i] for i in range(n_nodes)})
        graph_data_list.append(data)
    
    return graph_data_list 