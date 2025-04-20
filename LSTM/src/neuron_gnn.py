import torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import copy
from sklearn.model_selection import train_test_split
from torch_geometric.nn import GCNConv, SAGEConv, GATConv, global_mean_pool, global_add_pool, global_max_pool
from imblearn.over_sampling import SMOTE
from collections import Counter
from sklearn.utils.class_weight import compute_class_weight
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.neighbors import kneighbors_graph
from torch_geometric.nn import JumpingKnowledge, GlobalAttention
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import from_networkx, to_networkx, add_self_loops, degree
import os
import networkx as nx
from torch.nn import Sequential, Linear, BatchNorm1d, Dropout, ReLU, LeakyReLU, ELU, GELU
from sklearn.manifold import TSNE
from torch_geometric.data import Data
from typing import List, Dict, Tuple, Optional, Union, Any

class GNNAnalyzer:
    """
    使用图神经网络分析神经元网络拓扑结构
    """
    def __init__(self, config):
        """
        初始化GNN分析器
        
        参数:
            config: 配置对象，包含必要的参数设置
        """
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"GNN分析器将使用设备: {self.device}")
        
        # 确保保存GNN结果的目录存在
        self.gnn_output_dir = config.gnn_results_dir
        os.makedirs(self.gnn_output_dir, exist_ok=True)
    
    def convert_network_to_gnn_format(self, G, X_scaled, y=None, create_individual_graphs=True):
        """
        将NetworkX神经元网络转换为PyTorch Geometric格式
        
        参数:
            G: NetworkX图对象
            X_scaled: 标准化后的神经元活动数据
            y: 可选的行为标签
            create_individual_graphs: 是否为每个样本创建单独的图（解决BatchNorm问题）
        
        返回:
            data: PyTorch Geometric Data对象或Data对象列表
        """
        print("将神经元网络转换为GNN数据格式...")
        
        # 特征选择（从bettergcn/process.py集成）
        if X_scaled.shape[1] > 35:  # 只有当特征数量较多时才应用特征选择
            try:
                print(f"应用特征选择，从{X_scaled.shape[1]}个特征中选择前35个最重要的特征")
                selector = SelectKBest(f_classif, k=35)
                if y is not None:
                    X_scaled = selector.fit_transform(X_scaled, y)
                else:
                    print("警告：无标签数据，跳过特征选择")
            except Exception as e:
                print(f"特征选择出错：{e}，使用原始特征")
        
        # 处理类别不平衡（从bettergcn/process.py集成）
        if y is not None and len(np.unique(y)) > 1:
            try:
                class_counts = Counter(y)
                print(f"类别分布: {class_counts}")
                
                min_samples = min(class_counts.values())
                # 根据最小样本数调整k_neighbors
                k_neighbors = min(5, min_samples - 1)
                
                if min(class_counts.values()) < 100:  # 对小类别应用SMOTE
                    print("应用SMOTE处理类别不平衡...")
                    smote = SMOTE(random_state=42, k_neighbors=k_neighbors)
                    X_scaled, y = smote.fit_resample(X_scaled, y)
                    print(f"SMOTE后的类别分布: {Counter(y)}")
            except Exception as e:
                print(f"SMOTE处理出错：{e}，使用原始数据")
        
        # 创建节点ID映射（神经元名称到索引）
        node_map = {node: i for i, node in enumerate(G.nodes())}
        
        # 为每个样本创建单独的图（解决BatchNorm问题）
        if create_individual_graphs and X_scaled.shape[0] > 1:
            print(f"为每个样本创建单独的图结构，样本数量: {X_scaled.shape[0]}")
            dataset = []
            feature_dim = X_scaled.shape[1]
            
            # 使用KNN构建图结构（从bettergcn/process.py集成）
            print("使用KNN构建图结构...")
            try:
                # 为所有样本构建一个共享的KNN图结构
                A = kneighbors_graph(X_scaled, 10, mode='distance', include_self=False)
                A.data = np.exp(-A.data)  # 将距离转换为相似度
                rows, cols = A.nonzero()
                shared_edge_index = torch.tensor(np.array([rows % feature_dim, cols % feature_dim]), dtype=torch.long)
            except Exception as e:
                print(f"KNN图构建出错：{e}，使用简单的全连接图")
                # 使用简单的全连接图作为后备方案
                nodes = range(feature_dim)
                edges = [(i, j) for i in nodes for j in nodes if i != j]
                shared_edge_index = torch.tensor(list(zip(*edges)), dtype=torch.long)
            
            for i in range(X_scaled.shape[0]):
                x = torch.tensor(X_scaled[i], dtype=torch.float).reshape(feature_dim, 1)
                
                # 每个样本使用相同的边结构
                data_obj = Data(x=x, edge_index=shared_edge_index)
                
                # 如果提供了标签，添加到数据对象
                if y is not None:
                    data_obj.y = torch.tensor(y[i], dtype=torch.long)
                    
                dataset.append(data_obj)
                
            print(f"创建了{len(dataset)}个样本图，每个图有{feature_dim}个节点")
            return dataset
        
        # 传统方法：创建单个大图
        else:
            # 准备边索引和权重
            edge_index = []
            edge_weight = []
            
            # 提取边的信息
            for u, v, attr in G.edges(data=True):
                edge_index.append([node_map[u], node_map[v]])
                edge_index.append([node_map[v], node_map[u]])  # 添加双向边
                # 使用相关性作为边权重
                edge_weight.append(attr['weight'])
                edge_weight.append(attr['weight'])
            
            # 转换为PyTorch张量
            edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
            edge_weight = torch.tensor(edge_weight, dtype=torch.float)
            
            # 节点特征矩阵 - 使用神经元活动序列
            # 确保神经元顺序与图中节点顺序一致
            n_neurons = len(G.nodes())
            if X_scaled.shape[1] == n_neurons:
                # 如果特征维度与节点数匹配，假设它们按顺序对应
                x = torch.tensor(X_scaled, dtype=torch.float)
            else:
                # 否则使用节点度作为特征（作为备选方案）
                print(f"警告：特征维度 ({X_scaled.shape[1]}) 与节点数 ({n_neurons}) 不匹配，使用节点度作为特征")
                node_degrees = np.array([d for _, d in G.degree()])
                x = torch.tensor(node_degrees.reshape(-1, 1), dtype=torch.float)
            
            # 创建PyTorch Geometric数据对象
            data = Data(x=x, edge_index=edge_index, edge_attr=edge_weight)
            
            # 如果提供了标签，添加到数据对象
            if y is not None:
                data.y = torch.tensor(y, dtype=torch.long)
            
            print(f"GNN数据转换完成: {data}")
            return data
    
    def prepare_temporal_gnn_data(self, G, X_scaled, window_size=10, stride=5):
        """
        准备时间序列GNN数据
        
        参数:
            G: NetworkX图对象
            X_scaled: 标准化后的神经元活动数据
            window_size: 时间窗口大小
            stride: 窗口移动步长
        
        返回:
            temporal_data: 时间窗口GNN数据列表
        """
        print(f"准备时间序列GNN数据 (窗口大小: {window_size}, 步长: {stride})...")
        
        n_samples = X_scaled.shape[0]
        temporal_data = []
        
        # 滑动窗口生成时间序列数据
        for i in range(0, n_samples - window_size + 1, stride):
            window_data = X_scaled[i:i+window_size]
            
            # 为每个时间窗口创建一个单独的图
            window_G = G.copy()
            
            # 使用当前窗口的数据计算节点特征
            node_features = []
            for node_idx, node in enumerate(G.nodes()):
                # 提取该神经元在窗口期间的活动
                neuron_activity = window_data[:, node_idx]
                # 可以添加更多特征计算，例如活动统计量
                node_features.append(neuron_activity)
            
            # 更新图中的节点特征
            for i, node in enumerate(G.nodes()):
                window_G.nodes[node]['features'] = node_features[i]
            
            # 转换为PyTorch Geometric格式
            data = from_networkx(window_G)
            temporal_data.append(data)
        
        print(f"时间序列GNN数据准备完成，生成了 {len(temporal_data)} 个时间窗口")
        return temporal_data

class NeuronGCN(torch.nn.Module):
    """
    增强型图卷积网络 (GCN) 用于神经元行为预测
    
    特点:
    1. 多层GCN结构
    2. GraphSAGE和GAT层级联
    3. 批归一化
    4. 全局池化
    5. 多头注意力
    6. 改进的前向传播
    """
    def __init__(self, in_channels, hidden_channels, out_channels, dropout=0.3, 
                 num_layers=4, heads=2, use_batch_norm=True, activation='leaky_relu', 
                 alpha=0.2, residual=True):
        """
        初始化增强型GCN模型
        
        参数:
            in_channels: 输入特征维度
            hidden_channels: 隐藏层维度
            out_channels: 输出类别数
            dropout: dropout率
            num_layers: GCN层数 (此参数在新实现中不直接使用)
            heads: 注意力头数
            use_batch_norm: 是否使用批归一化
            activation: 激活函数类型 ('relu', 'leaky_relu', 'elu', 'gelu')
            alpha: LeakyReLU的alpha参数
            residual: 是否使用残差连接
        """
        super(NeuronGCN, self).__init__()
        
        # 存储参数，保持与原接口兼容
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels
        self.dropout = dropout
        self.num_layers = num_layers
        self.use_batch_norm = use_batch_norm
        self.activation = activation
        self.alpha = alpha
        self.residual = residual
        
        # 采用ImprovedGCN的架构，集成GCN、GraphSAGE和GAT
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = SAGEConv(hidden_channels, hidden_channels)
        self.conv3 = GATConv(hidden_channels, hidden_channels, heads=heads, concat=False)
        
        # BatchNorm层
        self.bn1 = BatchNorm1d(hidden_channels)
        self.bn2 = BatchNorm1d(hidden_channels)
        self.bn3 = BatchNorm1d(hidden_channels)
        
        # 多层感知机(MLP)用于分类
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(hidden_channels*2, hidden_channels),
            torch.nn.BatchNorm1d(hidden_channels),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(hidden_channels, out_channels)
        )
        
    
    def forward(self, x, edge_index, edge_weight=None, batch=None):
        """
        前向传播函数
        
        参数:
            x: 节点特征矩阵
            edge_index: 边索引
            edge_weight: 边权重（可选）
            batch: 批处理索引，指示每个节点属于哪个图（用于多图训练）
            
        返回:
            out: 分类结果
        """
        # GCN
        x1 = self.conv1(x, edge_index)
        x1 = self.bn1(x1)
        x1 = F.relu(x1)
        x1 = self.dropout(x1)
        
        # GraphSAGE
        x2 = self.conv2(x1, edge_index)
        x2 = self.bn2(x2)
        x2 = F.relu(x2)
        x2 = self.dropout(x2)
        
        # GAT
        x3 = self.conv3(x2, edge_index)
        x3 = self.bn3(x3)
        x3 = F.relu(x3)
        
        # 全局池化
        # 注意：如果是单个图，batch为None
        if batch is None:
            # 如果没有批处理信息，直接对整个特征矩阵进行处理
            # 在这种情况下，我们可以认为所有节点属于同一个图
            x_mean = torch.mean(x3, dim=0, keepdim=True)  # 全局均值池化
            x_sum = torch.sum(x3, dim=0, keepdim=True)    # 全局求和池化
        else:
            # 有批处理信息，使用PyG提供的全局池化函数
            x_mean = global_mean_pool(x3, batch)
            x_sum = global_add_pool(x3, batch)
        
        # 连接并分类
        x_combined = torch.cat([x_mean, x_sum], dim=1)
        out = self.mlp(x_combined)
        
        return out

class LabelSmoothingCrossEntropy(nn.Module):
    """
    标签平滑交叉熵损失函数
    
    参数:
        smoothing: 平滑系数，通常在0.1到0.2之间
    """
    def __init__(self, smoothing=0.1):
        super(LabelSmoothingCrossEntropy, self).__init__()
        self.smoothing = smoothing
        self.confidence = 1.0 - smoothing
        
    def forward(self, x, target):
        logprobs = F.log_softmax(x, dim=-1)
        nll_loss = -logprobs.gather(dim=-1, index=target.unsqueeze(1))
        nll_loss = nll_loss.squeeze(1)
        smooth_loss = -logprobs.mean(dim=-1)
        loss = self.confidence * nll_loss + self.smoothing * smooth_loss
        return loss.mean()

def custom_gnn_loss(outputs, data, lambda_reg=0.001):
    """
    自定义GNN损失函数，结合了多种损失成分
    
    参数:
        outputs: 模型输出
        data: PyTorch Geometric数据对象
        lambda_reg: 正则化系数
    
    返回:
        loss: 计算得到的损失值
    """
    # 如果存在标签，使用交叉熵
    if hasattr(data, 'y') and data.y is not None:
        ce_loss = F.cross_entropy(outputs, data.y)
        loss = ce_loss
    else:
        # 如果没有标签，可以使用自监督损失如重构损失
        loss = F.mse_loss(outputs, data.x)
    
    # L2正则化已经通过weight_decay实现，这里不再重复添加
    
    # 添加拓扑一致性损失，相连节点应该有相似的嵌入
    if hasattr(data, 'edge_index'):
        src, dst = data.edge_index
        src_emb = outputs[src]
        dst_emb = outputs[dst]
        topo_loss = F.mse_loss(src_emb, dst_emb)
        
        # 加权组合各种损失
        loss = loss + 0.1 * topo_loss
    
    return loss

def train_gnn_model(model, data, epochs, lr=0.01, weight_decay=1e-3, device='cpu', patience=15, early_stopping_enabled=False, batch_size=32):
    """
    训练GNN模型
    
    参数:
        model: GNN模型
        data: PyTorch Geometric数据对象
        epochs: 训练轮数
        lr: 学习率
        weight_decay: 权重衰减
        device: 设备(cuda/cpu)
        patience: 早停耐心值
        early_stopping_enabled: 是否启用早停机制，默认为False
    
    返回:
        model: 训练好的模型
        losses: 损失历史
        metrics: 其他指标历史（如准确率）
    """
    # 检查数据类型
    if isinstance(data, list):
        # 数据是列表，不要尝试直接访问.x属性
        print(f"列表数据接收成功，列表长度: {len(data)}")
        if len(data) > 0:
            sample = data[0]
            print(f"第一个元素类型: {type(sample).__name__}")
            if hasattr(sample, 'x') and hasattr(sample, 'edge_index'):
                print(f"样本数据 - x形状: {sample.x.size()}, 边形状: {sample.edge_index.size()}")
                if hasattr(sample, 'y'):
                    print(f"标签类型: {type(sample.y).__name__}")
    else:
        # 单个数据对象
        print(f"数据检查 - x形状: {data.x.size()}, y形状: {data.y.size() if hasattr(data, 'y') and data.y is not None else 'None'}")
        print(f"边索引形状: {data.edge_index.size()}, 边属性形状: {data.edge_attr.size() if hasattr(data, 'edge_attr') else 'None'}")
        
        # 将单个数据对象移至设备
        data = data.to(device)
        
    # 将模型移至设备
    model = model.to(device)
    
    # 准备优化器 - 使用带动量的SGD或AdamW
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay, betas=(0.9, 0.999))
    
    # 准备交叉熵损失函数
    criterion = torch.nn.CrossEntropyLoss()
    
    # 检查数据类型并决定是否使用DataLoader
    if isinstance(data, list):
        # 数据是一个列表，可能是Data对象的列表
        try:
            # 检查数据列表是否非空
            if len(data) == 0:
                raise ValueError("数据列表为空")
                
            # 检查首个元素是否为Data对象
            sample = data[0]
            if not hasattr(sample, 'x') or not hasattr(sample, 'edge_index'):
                raise ValueError(f"数据列表中的元素不是有效的Data对象: {type(sample)}")
                
            # 输出样本信息
            print(f"检测到数据列表，样本数量: {len(data)}")
            print(f"样本类型: {type(sample).__name__}, 特征形状: {sample.x.size()}, 边形状: {sample.edge_index.size()}")
            if hasattr(sample, 'y'):
                all_labels = []
                for d in data:
                    if hasattr(d, 'y'):
                        if isinstance(d.y, torch.Tensor):
                            all_labels.append(d.y.item())
                        else:
                            all_labels.append(d.y)
                
                if all_labels:
                    label_counts = torch.bincount(torch.tensor(all_labels))
                    print(f"标签分布: {label_counts}")
            
            # 划分训练集和验证集
            n_samples = len(data)
            indices = list(range(n_samples))
            
            # 如果有标签，使用分层采样
            if hasattr(sample, 'y'):
                all_labels = []
                for d in data:
                    if hasattr(d, 'y'):
                        if isinstance(d.y, torch.Tensor):
                            all_labels.append(d.y.item())
                        else:
                            all_labels.append(d.y)
                
                if len(all_labels) == n_samples:  # 确保标签数量与样本数量相等
                    from sklearn.model_selection import train_test_split
                    train_indices, val_indices = train_test_split(
                        indices, test_size=0.2, random_state=42, stratify=all_labels
                    )
                else:
                    # 如果标签数量不匹配，使用随机划分
                    from sklearn.model_selection import train_test_split
                    train_indices, val_indices = train_test_split(
                        indices, test_size=0.2, random_state=42
                    )
            else:
                # 没有标签，使用随机划分
                from sklearn.model_selection import train_test_split
                train_indices, val_indices = train_test_split(
                    indices, test_size=0.2, random_state=42
                )
            
            # 创建DataLoader
            from torch_geometric.loader import DataLoader
            
            # 分割数据集
            train_dataset = [data[i] for i in train_indices]
            val_dataset = [data[i] for i in val_indices]
            
            # 为训练和验证创建DataLoader
            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
            val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
            
            print(f"创建了DataLoader，批大小: {batch_size}")
            print(f"数据集划分: 训练集 {len(train_dataset)} 样本, 验证集 {len(val_dataset)} 样本")
            
            using_dataloader = True
            
        except Exception as e:
            print(f"创建DataLoader时发生错误: {e}")
            import traceback
            traceback.print_exc()
            raise
    else:
        # 单个数据对象，移至设备
        # 数据已在前面被移至设备
        using_dataloader = False
        print(f"使用单个数据对象, x形状: {data.x.size()}, 边形状: {data.edge_index.size()}")
        
        # 在非DataLoader模式下创建训练/验证数据划分
        indices = list(range(data.x.size(0)))
        
        # 如果有标签则使用分层采样
        if hasattr(data, 'y') and data.y is not None:
            from sklearn.model_selection import train_test_split
            stratify = data.y.cpu().numpy()
            train_indices, val_indices = train_test_split(
                indices, test_size=0.2, random_state=42, stratify=stratify
            )
        else:
            from sklearn.model_selection import train_test_split
            train_indices, val_indices = train_test_split(
                indices, test_size=0.2, random_state=42
            )
        
        # 创建训练和验证掌码
        train_mask = torch.zeros(data.x.size(0), dtype=torch.bool, device=device)
        train_mask[train_indices] = True
        
        val_mask = torch.zeros(data.x.size(0), dtype=torch.bool, device=device)
        val_mask[val_indices] = True
        
        print(f"划分了 {train_mask.sum().item()} 个训练样本和 {val_mask.sum().item()} 个验证样本")
            
    # 记录损失和准确率
    losses = {"train": [], "val": []}
    metrics = {"train_acc": [], "val_acc": []}
    best_val_loss = float('inf')
    no_improve_count = 0
    best_model_state = None
    
    # 训练循环
    for epoch in range(epochs):
        model.train()
        
        if using_dataloader:
            # 使用DataLoader的批处理方式
            train_loss = 0.0
            train_correct = 0
            train_total = 0
            batch_count = 0
            
            for batch in train_loader:
                try:
                    batch_count += 1
                    batch = batch.to(device)
                    optimizer.zero_grad()
                    
                    # 打印第一个批次的详细信息（仅在第一轮第一个批次）
                    if epoch == 0 and batch_count == 1:
                        print(f"批次数据 - 批次大小: {batch.num_graphs}, x形状: {batch.x.size()}")
                        print(f"边索引形状: {batch.edge_index.size()}, 批次索引: {batch.batch.size()}")
                        if hasattr(batch, 'y'):
                            print(f"该批次标签: {batch.y[:min(10, len(batch.y))]} (显示前10个)")
                    
                    # 运行前向传播 - 处理批处理数据
                    out = model(batch.x, batch.edge_index, batch.batch)
                    
                    # 计算损失
                    if hasattr(batch, 'y') and batch.y is not None:
                        loss = criterion(out, batch.y)
                        
                        # 计算准确率
                        if batch.y.dim() == 1:  # 分类任务
                            pred = out.argmax(dim=1)
                            train_correct += pred.eq(batch.y).sum().item()
                            train_total += batch.y.size(0)
                    else:
                        # 如果没有标签，使用重构损失
                        loss = F.mse_loss(out, batch.x)
                    
                    # 反向传播与参数更新
                    loss.backward()
                    optimizer.step()
                    
                    train_loss += loss.item() * batch.num_graphs
                    
                except Exception as e:
                    print(f"处理批次 {batch_count} 时出错: {e}")
                    # 继续处理下一个批次
                    continue
            
            # 计算平均损失和准确率
            if batch_count > 0:  # 确保至少处理了一个批次
                train_loss = train_loss / (batch_count * batch_size) if batch_count * batch_size > 0 else 0
                train_acc = train_correct / train_total if train_total > 0 else 0
            else:
                train_loss = 0.0
                train_acc = 0.0
                print("警告: 没有成功处理任何批次")
            
            # 验证模式
            model.eval()
            val_loss = 0.0
            val_correct = 0
            val_total = 0
            
            # 创建验证集（使用相同的数据列表但采用不同的划分）
            if epoch == 0:  # 只在第一轮训练创建验证集
                # 划分验证集，使用 20% 的数据
                n_samples = len(data)
                indices = list(range(n_samples))
                
                # 如果有标签，使用分层采样
                if hasattr(data[0], 'y'):
                    # 收集所有标签
                    labels = [d.y.item() if isinstance(d.y, torch.Tensor) else d.y for d in data]
                    train_indices, val_indices = train_test_split(
                        indices, test_size=0.2, random_state=42, stratify=labels
                    )
                else:
                    # 如果没有标签，使用标准划分
                    train_indices, val_indices = train_test_split(indices, test_size=0.2, random_state=42)
                
                # 创建验证集DataLoader
                val_dataset = [data[i] for i in val_indices]
                val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
                print(f"验证集创建完成: {len(val_dataset)}个样本")
            
            # 验证过程
            with torch.no_grad():
                for batch in val_loader:
                    batch = batch.to(device)
                    
                    # 前向传播
                    out = model(batch.x, batch.edge_index, batch.batch if hasattr(batch, 'batch') else None)
                    
                    # 计算损失和分类结果
                    if hasattr(batch, 'y') and batch.y is not None:
                        loss = criterion(out, batch.y)
                        pred = out.argmax(dim=1)
                        val_correct += pred.eq(batch.y).sum().item()
                        val_total += batch.y.size(0)
                    else:
                        loss = F.mse_loss(out, batch.x)
                    
                    val_loss += loss.item() * batch.num_graphs
                
                # 计算平均验证集损失和准确率
                if val_total > 0:  # 防止零除错误
                    val_loss = val_loss / val_total
                    val_acc = val_correct / val_total
                else:
                    val_loss = 0
                    val_acc = 0
        else:
            # 传统方式：使用单个大图的训练方法
            model.train()
            
            # 第一轮时创建训练/验证划分
            if epoch == 0:
                # 创建训练和验证索引
                indices = list(range(data.x.size(0)))
                
                # 如果有标签则使用分层采样
                if hasattr(data, 'y') and data.y is not None:
                    stratify = data.y.cpu().numpy()
                    train_indices, val_indices = train_test_split(
                        indices, test_size=0.2, random_state=42, stratify=stratify
                    )
                else:
                    train_indices, val_indices = train_test_split(
                        indices, test_size=0.2, random_state=42
                    )
                
                # 创建训练和验证掩码
                train_mask = torch.zeros(data.x.size(0), dtype=torch.bool, device=device)
                train_mask[train_indices] = True
                
                val_mask = torch.zeros(data.x.size(0), dtype=torch.bool, device=device)
                val_mask[val_indices] = True
                
                print(f"划分了 {train_mask.sum().item()} 个训练样本和 {val_mask.sum().item()} 个验证样本")
            
            # 前向传播和优化
            optimizer.zero_grad()
            out = model(data.x, data.edge_index, data.edge_attr if hasattr(data, 'edge_attr') else None)
            
            if hasattr(data, 'y') and data.y is not None:
                # 分类任务
                loss = criterion(out[train_mask], data.y[train_mask])
                loss.backward()
                optimizer.step()
                
                # 计算训练准确率
                preds = out[train_mask].argmax(dim=1)
                train_correct = preds.eq(data.y[train_mask]).sum().item()
                train_total = train_mask.sum().item()
                train_acc = train_correct / train_total if train_total > 0 else 0
                train_loss = loss.item()
                
                # 验证
                model.eval()
                with torch.no_grad():
                    val_preds = out[val_mask].argmax(dim=1)
                    val_correct = val_preds.eq(data.y[val_mask]).sum().item()
                    val_total = val_mask.sum().item()
                    val_acc = val_correct / val_total if val_total > 0 else 0
                    val_loss = criterion(out[val_mask], data.y[val_mask]).item()
            else:
                # 如果没有标签，使用重构损失
                loss = F.mse_loss(out[train_mask], data.x[train_mask])
                loss.backward()
                optimizer.step()
                
                train_loss = loss.item()
                train_acc = 0  # 无类别标签，没有准确率
                
                # 验证
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                
                optimizer.step()
                
                train_loss += loss.item()
                
                # 计算平均损失
                train_loss /= len(train_indices)
                train_acc = 0.0  # 非分类任务无准确率
            
            # 验证模式
            model.eval()
            val_loss = 0.0
            val_correct = 0
            val_total = 0
            
            # 创建验证掩码
            val_mask = torch.zeros(data.x.size(0), dtype=torch.bool, device=device)
            val_mask[val_indices] = True
            
            with torch.no_grad():
                out = model(data.x, data.edge_index, data.edge_attr if hasattr(data, 'edge_attr') else None)
                
                if hasattr(data, 'y') and data.y is not None:
                    # 分类任务验证
                    val_loss_value = criterion(out[val_mask], data.y[val_mask]).item()
                    val_loss += val_loss_value
                    
                    # 计算验证准确率
                    _, predicted = torch.max(out[val_mask], 1)
                    val_total += data.y[val_mask].size(0)
                    val_correct += (predicted == data.y[val_mask]).sum().item()
                    
                    # 计算平均损失和准确率
                    val_loss /= val_total if val_total > 0 else 1
                    val_acc = val_correct / val_total if val_total > 0 else 0
                else:
                    # 非分类任务验证
                    val_loss_value = custom_gnn_loss(out[val_mask], data).item()
                    val_loss += val_loss_value
                    val_acc = 0.0  # 非分类任务无准确率
        
        # 记录损失和准确率
        losses['train'].append(train_loss)
        losses['val'].append(val_loss)
        metrics['train_acc'].append(train_acc)
        metrics['val_acc'].append(val_acc)
        
        # 打印训练信息
        if epoch % 10 == 0 or epoch == 1:
            print(f"轮次 {epoch}/{epochs}, 训练损失: {train_loss:.4f}, 验证损失: {val_loss:.4f}, 验证准确率: {val_acc:.4f}")
        
        # 早停检查
        if early_stopping_enabled:
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_model_state = copy.deepcopy(model.state_dict())
                patience_counter = 0
            else:
                patience_counter += 1
                
            if patience_counter >= patience:
                print(f"早停激活于第 {epoch} 轮，最佳验证损失: {best_val_loss:.4f}")
                if best_model_state is not None:
                    model.load_state_dict(best_model_state)
                break  # 这里的break在循环内部，正确使用
    
    # 如果已经完成所有轮次，检查是否应该恢复最佳模型
    if early_stopping_enabled and best_model_state is not None:
        if val_loss > best_val_loss:
            model.load_state_dict(best_model_state)
            print(f"训练完成，已恢复最佳模型 (验证损失: {best_val_loss:.4f})")
    
    return model, losses, metrics

def plot_gnn_results(losses, save_path):
    """
    绘制GNN训练结果
    
    参数:
        losses: 损失字典
        save_path: 保存路径
    """
    plt.figure(figsize=(10, 6))
    plt.plot(losses['train'], label='Training Loss')
    plt.plot(losses['val'], label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('GNN Training Progress')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(save_path)
    plt.close()

def visualize_node_embeddings(model, data, save_path, title='Node Embeddings'):
    """
    可视化GNN模型学习的节点嵌入
    
    参数:
        model: 训练好的GNN模型
        data: 图数据
        save_path: 保存图像的路径
        title: 图表标题
    """
    # 确保目录存在
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    # 获取节点嵌入
    model.eval()
    with torch.no_grad():
        try:
            embeddings = model.get_embeddings(data)
            if torch.is_tensor(embeddings):
                embeddings = embeddings.cpu().detach().numpy()
        except Exception as e:
            print(f"获取节点嵌入出错: {e}")
            print("使用替代方法...")
            # 使用前向传播输出作为备选
            out = model(data)  # 前向传播
            if hasattr(model, 'last_hidden'):
                embeddings = model.last_hidden.cpu().detach().numpy()
            else:
                # 如果都不行，使用随机值（仅用于演示）
                print("无法获取嵌入，使用随机值...")
                embeddings = np.random.randn(data.num_nodes, 64)
    
    # 使用t-SNE降维到2D
    if embeddings.shape[1] > 2:
        tsne = TSNE(n_components=2, random_state=42)
        embeddings_2d = tsne.fit_transform(embeddings)
    else:
        embeddings_2d = embeddings
    
    # 获取标签
    labels = data.y.cpu().numpy() if hasattr(data, 'y') else np.zeros(len(embeddings))
    
    # 绘制嵌入可视化
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], 
                         c=labels, cmap='tab10', s=80, alpha=0.8)
    plt.colorbar(scatter, label='Class')
    
    # 添加节点标签
    n_nodes = min(30, len(embeddings_2d))  # 限制标签数量，防止过度拥挤
    for i in range(n_nodes):
        plt.annotate(f'N{i+1}', 
                    (embeddings_2d[i, 0], embeddings_2d[i, 1]),
                    xytext=(5, 5),
                    textcoords='offset points')
    
    plt.title(title)
    plt.xlabel('t-SNE Component 1')
    plt.ylabel('t-SNE Component 2')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    
    print(f"节点嵌入可视化已保存至: {save_path}") 