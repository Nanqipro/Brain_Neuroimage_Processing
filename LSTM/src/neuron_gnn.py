import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv, GATConv, MessagePassing
import torch.nn as nn
import networkx as nx
from typing import List, Dict, Tuple, Optional, Union, Any
import os
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from torch_geometric.utils import from_networkx, to_networkx, add_self_loops, degree
from torch_geometric.nn import global_mean_pool, global_add_pool
import copy
from torch.nn import Sequential, Linear, BatchNorm1d, Dropout, ReLU, LeakyReLU, ELU, GELU
from sklearn.manifold import TSNE

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
    
    def convert_network_to_gnn_format(self, G, X_scaled, y=None):
        """
        将NetworkX神经元网络转换为PyTorch Geometric格式
        
        参数:
            G: NetworkX图对象
            X_scaled: 标准化后的神经元活动数据
            y: 可选的行为标签
        
        返回:
            data: PyTorch Geometric Data对象
        """
        print("将神经元网络转换为GNN数据格式...")
        
        # 创建节点ID映射（神经元名称到索引）
        node_map = {node: i for i, node in enumerate(G.nodes())}
        
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
    2. 残差连接
    3. 批归一化
    4. 多头注意力
    5. 跳跃连接
    6. 可调节的激活函数
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
            num_layers: GCN层数
            heads: 注意力头数
            use_batch_norm: 是否使用批归一化
            activation: 激活函数类型 ('relu', 'leaky_relu', 'elu', 'gelu')
            alpha: LeakyReLU的alpha参数
            residual: 是否使用残差连接
        """
        super(NeuronGCN, self).__init__()
        
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels
        self.dropout = dropout
        self.num_layers = num_layers
        self.use_batch_norm = use_batch_norm
        self.activation = activation
        self.alpha = alpha
        self.residual = residual
        
        # 激活函数映射
        self.act_fn = self._get_activation(activation, alpha)
        
        # 初始投影层
        self.initial_proj = Sequential(
            Linear(in_channels, hidden_channels),
            self.act_fn,
            Dropout(dropout)
        )
        
        # 图卷积层列表
        self.conv_layers = nn.ModuleList()
        
        # 批归一化层列表
        self.batch_norms = nn.ModuleList() if use_batch_norm else None
        
        # 添加多个图卷积层
        for i in range(num_layers):
            self.conv_layers.append(GCNConv(hidden_channels, hidden_channels))
            if use_batch_norm:
                self.batch_norms.append(BatchNorm1d(hidden_channels))
                
        # 注意力层
        self.attention = nn.ModuleList([
            MultiHeadSelfAttention(hidden_channels, num_heads=heads, dropout=dropout)
            for _ in range(num_layers // 2)  # 在部分层后使用注意力
        ])
        
        # 输出层
        self.output_layer = Sequential(
            Linear(hidden_channels, hidden_channels // 2),
            self.act_fn,
            Dropout(dropout),
            Linear(hidden_channels // 2, out_channels)
        )
        
    def _get_activation(self, activation, alpha=0.2):
        """返回指定的激活函数"""
        if activation == 'relu':
            return ReLU()
        elif activation == 'leaky_relu':
            return LeakyReLU(alpha)
        elif activation == 'elu':
            return ELU()
        elif activation == 'gelu':
            return GELU()
        else:
            return ReLU()  # 默认
            
    def forward(self, data):
        """
        前向传播
        
        参数:
            data: PyTorch Geometric 数据对象
            
        返回:
            x: 节点分类概率
        """
        x, edge_index = data.x, data.edge_index
        
        # 处理边权重（如果存在）
        edge_weight = data.edge_attr if hasattr(data, 'edge_attr') else None
        
        # 初始特征投影
        x = self.initial_proj(x)
        
        # 保存初始特征用于残差连接
        x_res = x
        
        # 应用多层图卷积
        for i in range(self.num_layers):
            # 图卷积
            if edge_weight is not None:
                x_conv = self.conv_layers[i](x, edge_index, edge_weight)
            else:
                x_conv = self.conv_layers[i](x, edge_index)
            
            # 批归一化
            if self.use_batch_norm:
                x_conv = self.batch_norms[i](x_conv)
                
            # 激活函数
            x_conv = self.act_fn(x_conv)
            
            # Dropout
            x_conv = F.dropout(x_conv, p=self.dropout, training=self.training)
            
            # 残差连接
            if self.residual and x_conv.size() == x.size():
                x = x_conv + x
            else:
                x = x_conv
                
            # 在一些层后应用注意力机制
            if i < len(self.attention) and (i + 1) % 2 == 0:
                att_idx = i // 2
                # 应用注意力前确保维度正确
                x = self.attention[att_idx](x)
                
            # 更新残差连接的基准
            if i % 2 == 1:
                x_res = x
        
        # 输出层
        x = self.output_layer(x)
        
        return F.log_softmax(x, dim=1)
        
    def get_embeddings(self, data):
        """
        获取节点嵌入
        
        参数:
            data: PyTorch Geometric 数据对象
            
        返回:
            embeddings: 节点嵌入
        """
        x, edge_index = data.x, data.edge_index
        
        # 处理边权重（如果存在）
        edge_weight = data.edge_attr if hasattr(data, 'edge_attr') else None
        
        # 初始特征投影
        x = self.initial_proj(x)
        
        # 应用图卷积层获取嵌入
        for i in range(self.num_layers):
            # 图卷积
            if edge_weight is not None:
                x_conv = self.conv_layers[i](x, edge_index, edge_weight)
            else:
                x_conv = self.conv_layers[i](x, edge_index)
            
            # 批归一化
            if self.use_batch_norm:
                x_conv = self.batch_norms[i](x_conv)
                
            # 激活函数
            x_conv = self.act_fn(x_conv)
            
            # Dropout (训练中使用，评估时不使用)
            if self.training:
                x_conv = F.dropout(x_conv, p=self.dropout, training=self.training)
                
            # 残差连接
            if self.residual and x_conv.size() == x.size():
                x = x_conv + x
            else:
                x = x_conv
                
            # 在一些层后应用注意力机制
            if i < len(self.attention) and (i + 1) % 2 == 0:
                att_idx = i // 2
                x = self.attention[att_idx](x)
        
        return x

# 添加多头自注意力机制
class MultiHeadSelfAttention(nn.Module):
    def __init__(self, in_channels, num_heads=4, dropout=0.1):
        super(MultiHeadSelfAttention, self).__init__()
        self.num_heads = num_heads
        self.in_channels = in_channels
        self.head_dim = in_channels // num_heads
        assert in_channels % num_heads == 0, "in_channels must be divisible by num_heads"
        
        # 定义查询、键、值投影
        self.query = Linear(in_channels, in_channels)
        self.key = Linear(in_channels, in_channels)
        self.value = Linear(in_channels, in_channels)
        
        # 输出投影
        self.out_proj = Linear(in_channels, in_channels)
        
        # Dropout
        self.dropout = Dropout(dropout)
        
        # 缩放因子
        self.scale = self.head_dim ** -0.5
        
    def forward(self, x):
        original_dim = x.dim()
        original_shape = x.shape
        
        # 处理输入数据形状 - 适应PyG中节点特征的形状
        if original_dim == 2:
            # 如果是2D输入 [num_nodes, channels]
            batch_size = 1
            seq_len = x.size(0)
            # 转换为 [1, num_nodes, channels]
            x = x.unsqueeze(0)
        else:
            # 如果已经是3D输入 [batch_size, num_nodes, channels]
            batch_size, seq_len, _ = x.size()
        
        # 投影查询、键、值
        q = self.query(x)
        k = self.key(x)
        v = self.value(x)
        
        # 重塑为多头形式 [batch_size, num_heads, seq_len, head_dim]
        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        k = k.view(batch_size, seq_len, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        v = v.view(batch_size, seq_len, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        
        # 计算注意力分数
        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        
        # 应用softmax
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # 应用注意力权重
        context = torch.matmul(attn_weights, v)
        
        # 重塑并合并多头结果 [batch_size, seq_len, in_channels]
        context = context.permute(0, 2, 1, 3).contiguous().view(batch_size, seq_len, self.in_channels)
        
        # 输出投影
        output = self.out_proj(context)
        
        # 恢复原始维度
        if original_dim == 2:
            output = output.squeeze(0)
            
        return output

class NeuronGAT(torch.nn.Module):
    """使用图注意力网络进行神经元功能模块识别"""
    
    def __init__(self, in_channels, hidden_channels, out_channels, heads=4, dropout=0.3):
        """
        初始化神经元GAT模型
        
        参数:
            in_channels: 输入特征维度
            hidden_channels: 隐藏层维度
            out_channels: 输出类别数
            heads: 注意力头数量
            dropout: Dropout比率
        """
        super(NeuronGAT, self).__init__()
        # 减小隐藏层维度，增加正则化
        self.conv1 = GATConv(in_channels, hidden_channels, heads=heads, dropout=dropout)
        self.bn1 = torch.nn.BatchNorm1d(hidden_channels * heads)
        self.conv2 = GATConv(hidden_channels * heads, hidden_channels, heads=1, dropout=dropout)
        self.bn2 = torch.nn.BatchNorm1d(hidden_channels)
        self.classifier = torch.nn.Linear(hidden_channels, out_channels)
        self.dropout = dropout
        
    def forward(self, data):
        """
        前向传播
        
        参数:
            data: PyTorch Geometric数据对象
        
        返回:
            x: 模型输出
        """
        x, edge_index = data.x, data.edge_index
        
        # 第一层图注意力卷积
        x = self.conv1(x, edge_index)
        x = self.bn1(x)
        x = F.elu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        
        # 第二层图注意力卷积
        x = self.conv2(x, edge_index)
        x = self.bn2(x)
        x = F.elu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        
        # 池化
        # x = self.pooling(x, edge_index)

        # 均值池化
        # x = self.mean_pooling(x, edge_index)
        
        # 基于自注意力机制的池化
        # x = self.self_attention_pooling(x, edge_index)
        
        # SAGPool (Self-Attention Graph Pooling)：基于自注意力机制的池化
        # x = self.sag_pooling(x, edge_index)
        
        # DiffPool：可微分的图池化层，学习将节点聚类到更高层次的结构
        # x = self.diff_pooling(x, edge_index)
        
        # TopKPool：保留最重要的k个节点，特别适合找出关键神经元
        # x = self.topk_pooling(x, edge_index)
        # 分类层
        x = self.classifier(x)
        
        return x
        
    def get_embeddings(self, data):
        """
        获取节点嵌入向量和注意力权重
        
        参数:
            data: PyTorch Geometric数据对象
            
        返回:
            embeddings: 节点嵌入向量
            attention_weights: 注意力权重（如果可用）
        """
        x, edge_index = data.x, data.edge_index
        
        # 第一层图注意力卷积
        x = self.conv1(x, edge_index)
        x = self.bn1(x)
        x = F.elu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        
        # 第二层图注意力卷积（最终嵌入）
        embeddings = self.conv2(x, edge_index)
        embeddings = self.bn2(embeddings)
        embeddings = F.elu(embeddings)
        
        return embeddings

class TemporalGNN(nn.Module):
    """用于分析神经元时间序列活动的时间图神经网络"""
    
    def __init__(self, node_features, hidden_dim, output_dim, lstm_layers=1):
        """
        初始化时间图神经网络
        
        参数:
            node_features: 节点特征维度
            hidden_dim: 隐藏层维度
            output_dim: 输出维度
            lstm_layers: LSTM层数
        """
        super(TemporalGNN, self).__init__()
        
        # 图卷积层
        self.gconv = GCNConv(node_features, hidden_dim)
        
        # LSTM层
        self.lstm = nn.LSTM(hidden_dim, hidden_dim, 
                          num_layers=lstm_layers,
                          batch_first=True)
        
        # 输出层
        self.fc = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, x, edge_index, edge_weight=None):
        """
        前向传播
        
        参数:
            x: 输入特征序列 [batch_size, sequence_length, num_nodes, node_features]
            edge_index: 边索引
            edge_weight: 边权重
            
        返回:
            out: 模型输出
        """
        batch_size = x.size(0)
        seq_length = x.size(1)
        num_nodes = x.size(2)
        
        # 对每个时间步应用GCN
        conv_out = []
        for t in range(seq_length):
            x_t = x[:, t, :, :]  # [batch_size, num_nodes, node_features]
            
            # 对每个样本应用GCN
            batch_out = []
            for b in range(batch_size):
                out = self.gconv(x_t[b], edge_index, edge_weight)
                batch_out.append(out)
            
            # 合并批次结果
            batch_out = torch.stack(batch_out)  # [batch_size, num_nodes, hidden_dim]
            conv_out.append(batch_out)
        
        # 合并GCN输出序列
        conv_out = torch.stack(conv_out, dim=1)  # [batch_size, seq_length, num_nodes, hidden_dim]
        
        # 对每个节点应用LSTM
        lstm_out = []
        for n in range(num_nodes):
            node_seq = conv_out[:, :, n, :]  # [batch_size, seq_length, hidden_dim]
            out, _ = self.lstm(node_seq)
            lstm_out.append(out[:, -1, :])  # 取最后一个时间步的输出
        
        # 合并所有节点的LSTM输出
        lstm_out = torch.stack(lstm_out, dim=1)  # [batch_size, num_nodes, hidden_dim]
        
        # 应用全连接层
        out = self.fc(lstm_out)  # [batch_size, num_nodes, output_dim]
        
        return out

class ModuleGNN(MessagePassing):
    """使用自定义消息传递进行神经元模块识别的GNN"""
    
    def __init__(self, in_channels, out_channels):
        super(ModuleGNN, self).__init__(aggr='mean')  # 使用平均聚合
        self.lin = torch.nn.Linear(in_channels, out_channels)
        
    def forward(self, x, edge_index, edge_weight=None):
        # 添加自环
        edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))
        
        # 线性变换
        x = self.lin(x)
        
        # 启动消息传递
        return self.propagate(edge_index, x=x, edge_weight=edge_weight)
        
    def message(self, x_j, edge_weight=None):
        # x_j 包含源节点的特征
        msg = x_j
        
        # 如果提供了边权重，应用权重
        if edge_weight is not None:
            msg = msg * edge_weight.view(-1, 1)
            
        return msg
        
    def update(self, aggr_out):
        # aggr_out 包含聚合后的消息
        return aggr_out

def train_gnn_model(model, data, epochs, lr=0.01, weight_decay=1e-3, device='cpu', patience=15, early_stopping_enabled=False):
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
        early_stopping_enabled: 是否启用早停机制，默认为True
    
    返回:
        model: 训练好的模型
        losses: 损失历史
        metrics: 其他指标历史（如准确率）
    """
    # 将数据移至设备
    data = data.to(device)
    model = model.to(device)
    
    # 准备优化器
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    
    # 学习率调度器 - 使用更温和的衰减
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.7, patience=5, verbose=True, min_lr=1e-5
    )
    
    criterion = torch.nn.CrossEntropyLoss()
    
    # 创建训练/验证索引 - 使用更小的验证集
    indices = list(range(data.x.size(0)))
    train_indices, val_indices = train_test_split(indices, test_size=0.18, random_state=42)
    
    # 记录损失和准确率
    losses = {'train': [], 'val': []}
    accuracies = {'train': [], 'val': []}
    
    # 早停设置 - 使用更温和的早停策略
    best_val_loss = float('inf')
    best_model_state = None
    patience_counter = 0
    min_delta = 1e-5  # 降低最小改进阈值，更容易继续训练
    
    # 训练循环
    for epoch in range(epochs):
        # 训练模式
        model.train()
        optimizer.zero_grad()
        
        # 前向传播
        out = model(data)
        
        # 计算训练损失 - 仅使用训练索引
        train_loss = criterion(out[train_indices], data.y[train_indices])
        losses['train'].append(train_loss.item())
        
        # 计算训练准确率
        _, train_predicted = torch.max(out[train_indices], 1)
        train_acc = (train_predicted == data.y[train_indices]).sum().item() / len(train_indices)
        accuracies['train'].append(train_acc)
        
        # 反向传播和优化
        train_loss.backward()
        optimizer.step()
        
        # 评估模式
        model.eval()
        with torch.no_grad():
            # 前向传播
            out = model(data)
            
            # 计算验证损失 - 仅使用验证索引
            val_loss = criterion(out[val_indices], data.y[val_indices])
            losses['val'].append(val_loss.item())
            
            # 计算验证准确率
            _, predicted = torch.max(out[val_indices], 1)
            val_acc = (predicted == data.y[val_indices]).sum().item() / len(val_indices)
            accuracies['val'].append(val_acc)
        
        # 学习率调整
        scheduler.step(val_loss)
        
        # 每10轮打印一次进度
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{epochs}, Train Loss: {train_loss.item():.4f}, Val Loss: {val_loss.item():.4f}, Val Acc: {val_acc:.4f}, LR: {optimizer.param_groups[0]['lr']:.6f}")
        
        # 早停检查
        if early_stopping_enabled:
            if val_loss < best_val_loss - min_delta:
                best_val_loss = val_loss
                best_model_state = copy.deepcopy(model.state_dict())
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"Early stopping at epoch {epoch+1}, best val loss: {best_val_loss:.4f}")
                    break
        else:
            # 如果禁用早停，仍然保存最佳模型
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_model_state = copy.deepcopy(model.state_dict())
                
    # 如果禁用早停且完成所有训练，打印最终结果
    if not early_stopping_enabled:
        print(f"Training completed for all {epochs} epochs, best val loss: {best_val_loss:.4f}")
            
    # 加载最佳模型
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        print(f"已恢复最佳模型 (验证损失: {best_val_loss:.4f})")
    
    return model, losses, accuracies

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