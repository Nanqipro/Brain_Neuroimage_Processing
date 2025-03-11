import torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import copy
from sklearn.model_selection import train_test_split
from torch_geometric.nn import GCNConv, GATConv, global_mean_pool, global_add_pool, global_max_pool
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
    
    def convert_network_to_gnn_format(self, G, X_scaled, y=None):
        """
        将NetworkX图转换为PyTorch Geometric数据格式
        
        参数:
            G: NetworkX图对象
            X_scaled: 神经元活动数据，形状为(时间点数, 神经元数)
            y: 可选的标签数据
            
        返回:
            data: PyTorch Geometric数据对象
        """
        print("\n将神经元网络转换为GNN格式...")
        
        # 检查输入数据维度
        print(f"输入数据形状: X_scaled={X_scaled.shape}, 图节点数={len(G.nodes())}")
        
        # 处理数据维度不匹配的情况
        if X_scaled.shape[1] != len(G.nodes()):
            print(f"警告: 特征维度({X_scaled.shape[1]})与图节点数({len(G.nodes())})不匹配")
            
            # 情况1: 特征维度大于节点数 - 截断特征
            if X_scaled.shape[1] > len(G.nodes()):
                print(f"特征维度大于节点数，截断特征至前{len(G.nodes())}个")
                X_scaled = X_scaled[:, :len(G.nodes())]
            
            # 情况2: 特征维度小于节点数 - 填充零向量
            else:
                print(f"特征维度小于节点数，填充零向量")
                padding = np.zeros((X_scaled.shape[0], len(G.nodes()) - X_scaled.shape[1]))
                X_scaled = np.hstack((X_scaled, padding))
        
        # 将NetworkX图转换为PyTorch Geometric数据
        try:
            # 获取节点索引映射
            node_idx = {node: i for i, node in enumerate(G.nodes())}
            
            # 创建边索引和边属性
            edge_index = []
            edge_attr = []
            
            for u, v, data in G.edges(data=True):
                # 添加双向边
                edge_index.append([node_idx[u], node_idx[v]])
                edge_index.append([node_idx[v], node_idx[u]])  # 添加反向边
                
                # 添加边属性
                weight = data.get('weight', 0.5)
                edge_attr.append([weight])
                edge_attr.append([weight])  # 反向边相同权重
            
            # 如果没有边，创建自环
            if len(edge_index) == 0:
                print("警告: 图中没有边，为每个节点添加自环")
                for i in range(len(G.nodes())):
                    edge_index.append([i, i])
                    edge_attr.append([1.0])  # 自环权重为1
            
            # 转换为张量
            if len(edge_index) > 0:
                edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
                edge_attr = torch.tensor(edge_attr, dtype=torch.float)
            else:
                # 创建空边索引和属性
                edge_index = torch.zeros((2, 0), dtype=torch.long)
                edge_attr = torch.zeros((0, 1), dtype=torch.float)
            
            # 添加节点特征
            node_features = torch.tensor(X_scaled.T, dtype=torch.float)
            
            # 创建数据对象
            data = Data(x=node_features, edge_index=edge_index, edge_attr=edge_attr)
            
            # 验证边索引和边属性的大小是否匹配
            if edge_index.shape[1] != edge_attr.shape[0]:
                print(f"警告: 边索引({edge_index.shape[1]})和边属性({edge_attr.shape[0]})大小不匹配！")
                print("修正边属性大小使其与边索引匹配...")
                # 如果不匹配，将边属性扩展到与边索引相同的大小
                new_edge_attr = torch.ones((edge_index.shape[1], 1), dtype=torch.float)
                if edge_attr.shape[0] > 0:
                    # 复制现有属性直到填满
                    for i in range(edge_index.shape[1]):
                        new_edge_attr[i] = edge_attr[i % edge_attr.shape[0]]
                data.edge_attr = new_edge_attr
                print(f"边属性已调整，大小现在为: {data.edge_attr.shape}")
            
        except Exception as e:
            print(f"创建GNN数据出错: {str(e)}，尝试备用方法")
            
            # 备用方法: 使用from_networkx但处理边属性
            try:
                data = from_networkx(G)
                
                # 添加节点特征
                node_features = torch.tensor(X_scaled.T, dtype=torch.float)
                data.x = node_features
                
                # 确保边属性与边索引匹配
                if hasattr(data, 'edge_index'):
                    data.weight = torch.ones(data.edge_index.shape[1], dtype=torch.float)
                    data.edge_attr = data.weight.view(-1, 1)
            except Exception as e2:
                print(f"备用方法也失败: {str(e2)}，创建默认数据对象")
                
                # 创建默认数据对象
                num_nodes = len(G.nodes())
                node_features = torch.tensor(X_scaled.T, dtype=torch.float)
                
                # 创建自环边
                edge_index = torch.zeros((2, num_nodes), dtype=torch.long)
                for i in range(num_nodes):
                    edge_index[0, i] = i
                    edge_index[1, i] = i
                    
                edge_attr = torch.ones((num_nodes, 1), dtype=torch.float)
                
                data = Data(x=node_features, edge_index=edge_index, edge_attr=edge_attr)
        
        # 添加标签（如果提供）
        if y is not None:
            # 确保y的长度与数据点数量匹配
            if len(y) != X_scaled.shape[0]:
                print(f"警告: 标签长度({len(y)})与数据点数({X_scaled.shape[0]})不匹配")
                
                # 如果标签太多，截断
                if len(y) > X_scaled.shape[0]:
                    print(f"标签太多，截断至前{X_scaled.shape[0]}个")
                    y = y[:X_scaled.shape[0]]
                
                # 如果标签太少，填充最后一个标签
                else:
                    print(f"标签太少，填充至{X_scaled.shape[0]}个")
                    last_label = y[-1]
                    padding = np.array([last_label] * (X_scaled.shape[0] - len(y)))
                    y = np.concatenate([y, padding])
            
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
    
    def __init__(self, in_channels, hidden_channels, out_channels, heads=4, dropout=0.3, 
                 residual=True, num_layers=3, edge_dim=None, alpha=0.2):
        """
        初始化神经元GAT模型
        
        参数:
            in_channels: 输入特征维度
            hidden_channels: 隐藏层维度
            out_channels: 输出类别数
            heads: 注意力头数量
            dropout: Dropout比率
            residual: 是否使用残差连接
            num_layers: GAT层数
            edge_dim: 边特征维度，默认为None
            alpha: LeakyReLU的alpha参数
        """
        super(NeuronGAT, self).__init__()
        self.num_layers = num_layers
        self.residual = residual
        self.dropout_rate = dropout
        
        # 初始层
        self.convs = nn.ModuleList()
        self.batch_norms = nn.ModuleList()
        
        # 第一层
        self.convs.append(GATConv(in_channels, hidden_channels, heads=heads, 
                           dropout=dropout, edge_dim=edge_dim, add_self_loops=True))
        self.batch_norms.append(torch.nn.BatchNorm1d(hidden_channels * heads))
        
        # 中间层
        for i in range(num_layers - 2):
            self.convs.append(
                GATConv(hidden_channels * heads, hidden_channels, heads=heads, 
                       dropout=dropout, edge_dim=edge_dim, add_self_loops=True)
            )
            self.batch_norms.append(torch.nn.BatchNorm1d(hidden_channels * heads))
        
        # 最后一个GAT层
        if num_layers > 1:
            self.convs.append(
                GATConv(hidden_channels * heads, hidden_channels, heads=1, 
                       dropout=dropout, edge_dim=edge_dim, add_self_loops=True)
            )
            self.batch_norms.append(torch.nn.BatchNorm1d(hidden_channels))
        
        # JK连接层 - 跳跃连接以聚合多层特征
        self.jk = JumpingKnowledge(mode='max')
        
        # 全局注意力池化层
        self.glob_attn = GlobalAttention(
            nn.Sequential(
                nn.Linear(hidden_channels, hidden_channels), 
                nn.LeakyReLU(alpha), 
                nn.Linear(hidden_channels, 1)
            )
        )
        
        # 分类层加入多层感知机结构
        self.mlp = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels),
            nn.LeakyReLU(alpha),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels, hidden_channels // 2),
            nn.LeakyReLU(alpha),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels // 2, out_channels)
        )
        
        # 残差连接的投影层
        if residual:
            self.res_projs = nn.ModuleList()
            in_dim = in_channels
            for i in range(num_layers):
                out_dim = hidden_channels if i == num_layers - 1 else hidden_channels * heads
                self.res_projs.append(nn.Linear(in_dim, out_dim, bias=False))
                in_dim = hidden_channels * heads
        
        # 层间维度调整投影层
        self.dim_projections = nn.ModuleList()
        # 只有当有多个层且最后一层头数为1时才需要进行维度调整
        if num_layers > 1 and (heads > 1 or edge_dim is not None):
            for i in range(num_layers - 1):
                # 第i层输出维度
                in_dim = hidden_channels * heads
                # 最后一层输出维度
                out_dim = hidden_channels
                self.dim_projections.append(nn.Linear(in_dim, out_dim, bias=False))
        
        # 额外的增强网络
        self.attention_enhancement = MultiHeadSelfAttention(hidden_channels, num_heads=2, dropout=dropout)
        
    def forward(self, data):
        """
        前向传播
        
        参数:
            data: PyTorch Geometric数据对象
        
        返回:
            x: 模型输出
        """
        x, edge_index = data.x, data.edge_index
        edge_weight = data.edge_weight if hasattr(data, 'edge_weight') else None
        
        # 存储每层的输出用于跳跃连接
        layer_outputs = []
        
        # 通过所有GAT层
        for i in range(self.num_layers):
            # 残差连接
            res = self.res_projs[i](x) if self.residual else None
            
            # 应用GAT卷积
            if edge_weight is not None:
                x = self.convs[i](x, edge_index, edge_weight)
            else:
                x = self.convs[i](x, edge_index)
            
            # 批量归一化
            x = self.batch_norms[i](x)
            
            # 添加残差连接
            if self.residual and res is not None and x.size() == res.size():
                x = x + res
            
            # 非线性激活
            x = F.leaky_relu(x, negative_slope=0.2)
            
            # Dropout
            x = F.dropout(x, p=self.dropout_rate, training=self.training)
            
            # 存储层输出
            layer_outputs.append(x)
        
        # 应用跳跃连接前确保所有层输出维度一致
        if len(layer_outputs) > 1:
            # 检查是否需要维度调整
            first_shape = layer_outputs[0].shape
            last_shape = layer_outputs[-1].shape
            
            if first_shape[-1] != last_shape[-1] and len(self.dim_projections) > 0:
                # 需要维度调整
                normalized_outputs = []
                for i, layer_output in enumerate(layer_outputs):
                    if i < len(layer_outputs) - 1:  # 对除最后一层外的所有层应用投影
                        # 使用预先创建的投影层
                        normalized_outputs.append(self.dim_projections[i](layer_output))
                    else:
                        normalized_outputs.append(layer_output)
                
                # 应用跳跃连接
                x = self.jk(normalized_outputs)
            else:
                # 维度已经一致，直接应用跳跃连接
                x = self.jk(layer_outputs)
        
        # 应用自注意力增强
        x = self.attention_enhancement(x)
        
        # 应用全局注意力池化（如果需要）
        batch = data.batch if hasattr(data, 'batch') else None
        if batch is not None:
            x = self.glob_attn(x, batch)
        
        # 使用MLP分类器
        x = self.mlp(x)
        
        return x
        
    def get_embeddings(self, data):
        """
        获取节点嵌入向量
        
        参数:
            data: PyTorch Geometric数据对象
            
        返回:
            embeddings: 节点嵌入向量
        """
        x, edge_index = data.x, data.edge_index
        edge_weight = data.edge_weight if hasattr(data, 'edge_weight') else None
        
        # 存储每层的输出用于跳跃连接
        layer_outputs = []
        
        # 通过所有GAT层
        for i in range(self.num_layers):
            # 残差连接
            res = self.res_projs[i](x) if self.residual else None
            
            # 应用GAT卷积
            if edge_weight is not None:
                x = self.convs[i](x, edge_index, edge_weight)
            else:
                x = self.convs[i](x, edge_index)
            
            # 批量归一化
            x = self.batch_norms[i](x)
            
            # 添加残差连接
            if self.residual and res is not None and x.size() == res.size():
                x = x + res
            
            # 非线性激活
            x = F.leaky_relu(x, negative_slope=0.2)
            
            # 存储层输出
            layer_outputs.append(x)
        
        # 应用跳跃连接前确保所有层输出维度一致
        if len(layer_outputs) > 1:
            # 检查是否需要维度调整
            first_shape = layer_outputs[0].shape
            last_shape = layer_outputs[-1].shape
            
            if first_shape[-1] != last_shape[-1] and len(self.dim_projections) > 0:
                # 需要维度调整
                normalized_outputs = []
                for i, layer_output in enumerate(layer_outputs):
                    if i < len(layer_outputs) - 1:  # 对除最后一层外的所有层应用投影
                        # 使用预先创建的投影层
                        normalized_outputs.append(self.dim_projections[i](layer_output))
                    else:
                        normalized_outputs.append(layer_output)
                
                # 应用跳跃连接
                embeddings = self.jk(normalized_outputs)
            else:
                # 维度已经一致，直接应用跳跃连接
                embeddings = self.jk(layer_outputs)
        else:
            embeddings = layer_outputs[0]
        
        # 应用自注意力增强
        embeddings = self.attention_enhancement(embeddings)
        
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

def train_gnn_model(model, data, epochs, lr=0.01, weight_decay=1e-3, device='cpu', patience=15, early_stopping_enabled=False):
    """
    训练GNN模型
    
    参数:
        model: GNN模型
        data: PyTorch Geometric数据对象
        epochs: 训练轮数
        lr: 学习率
        weight_decay: 权重衰减
        device: 训练设备
        patience: 早停耐心值
        early_stopping_enabled: 是否启用早停
        
    返回:
        model: 训练后的模型
        losses: 损失记录
        accuracies: 准确率记录
    """
    # 数据维度检查和修复
    print(f"GNN数据对象: {data}")
    
    # 特殊情况处理：检查节点特征维度是否与样本数不匹配
    node_features_transpose = False
    if hasattr(data, 'y') and data.y is not None:
        if len(data.y) != data.x.size(0):
            print(f"警告: 标签数量({len(data.y)})与节点数量({data.x.size(0)})不匹配")
            if len(data.y) == data.x.size(1):
                print("节点特征形状为[节点数, 样本数]，将特征矩阵转置")
                # 转置特征矩阵使维度匹配
                data.x = data.x.t()
                node_features_transpose = True
            else:
                print("无法自动修复维度不匹配问题，请检查数据格式")
                print(f"特征形状: {data.x.shape}, 标签形状: {data.y.shape}")
    
    # 移动模型到指定设备
    model = model.to(device)
    data = data.to(device)
    
    # 优化器设置
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
    
    # 使用标签平滑交叉熵损失
    criterion = LabelSmoothingCrossEntropy(smoothing=0.1)
    
    # 记录训练过程
    losses = {'train': [], 'val': []}
    accuracies = {'train': [], 'val': []}
    
    # 划分训练集和验证集
    try:
        # 如果特征矩阵已转置（[样本数, 节点数]），则按样本划分
        if node_features_transpose:
            num_samples = data.x.size(0)
            indices = list(range(num_samples))
            
            # 如果数据包含标签且在GPU上，需要将其移至CPU进行stratify
            y_cpu = None
            if data.y is not None:
                y_cpu = data.y.cpu().numpy() if data.y.is_cuda else data.y.numpy()
            
            train_indices, val_indices = train_test_split(
                indices, test_size=0.2, random_state=42, stratify=y_cpu
            )
            
            # 创建训练集和验证集的掩码
            train_mask = torch.zeros(num_samples, dtype=torch.bool, device=device)
            val_mask = torch.zeros(num_samples, dtype=torch.bool, device=device)
            
            train_mask[train_indices] = True
            val_mask[val_indices] = True
            
        # 常规情况，按节点划分
        else:
            num_nodes = data.x.size(0)
            indices = list(range(num_nodes))
            
            # 如果数据包含标签且在GPU上，需要将其移至CPU进行stratify
            y_cpu = None
            if data.y is not None:
                y_cpu = data.y.cpu().numpy() if data.y.is_cuda else data.y.numpy()
            
            train_indices, val_indices = train_test_split(
                indices, test_size=0.2, random_state=42, stratify=y_cpu
            )
            
            # 创建训练集和验证集的掩码
            train_mask = torch.zeros(num_nodes, dtype=torch.bool, device=device)
            val_mask = torch.zeros(num_nodes, dtype=torch.bool, device=device)
            
            train_mask[train_indices] = True
            val_mask[val_indices] = True
    except Exception as e:
        print(f"训练集验证集划分失败，使用默认划分: {str(e)}")
        # 简单的百分比拆分
        num_samples = data.x.size(0)
        train_size = int(0.8 * num_samples)
        
        train_mask = torch.zeros(num_samples, dtype=torch.bool)
        val_mask = torch.zeros(num_samples, dtype=torch.bool)
        
        train_mask[:train_size] = True
        val_mask[train_size:] = True
    
    # 将掩码移动到设备
    train_mask = train_mask.to(device)
    val_mask = val_mask.to(device)
    
    # 添加掩码到数据对象
    data.train_mask = train_mask
    data.val_mask = val_mask
    
    # 初始化最佳验证损失和早停计数器
    best_val_loss = float('inf')
    early_stopping_counter = 0
    
    # 保存最佳模型
    best_model = None
    
    # 训练循环
    for epoch in range(1, epochs + 1):
        # 训练阶段
        model.train()
        optimizer.zero_grad()
        out = model(data)
        
        # 获取训练集预测和标签
        if node_features_transpose:
            # 如果特征被转置，目标是预测样本标签
            loss = criterion(out[train_mask], data.y[train_mask])
        else:
            # 常规情况，目标是预测节点标签
            loss = criterion(out[train_mask], data.y[train_mask])
        
        loss.backward()
        optimizer.step()
        
        # 评估阶段
        model.eval()
        with torch.no_grad():
            out = model(data)
            
            # 计算验证损失
            val_loss = criterion(out[val_mask], data.y[val_mask])
            
            # 计算训练和验证准确率
            _, train_pred = out[train_mask].max(dim=1)
            train_correct = train_pred.eq(data.y[train_mask]).sum().item()
            train_acc = train_correct / train_mask.sum().item()
            
            _, val_pred = out[val_mask].max(dim=1)
            val_correct = val_pred.eq(data.y[val_mask]).sum().item()
            val_acc = val_correct / val_mask.sum().item()
        
        # 更新学习率
        current_lr = scheduler.get_last_lr()[0]
        scheduler.step()
        
        # 记录损失和准确率
        losses['train'].append(loss.item())
        losses['val'].append(val_loss.item())
        accuracies['train'].append(train_acc)
        accuracies['val'].append(val_acc)
        
        # 每10个epochs打印一次进度
        if epoch % 10 == 0 or epoch == 1 or epoch == epochs:
            print(f"Epoch {epoch}/{epochs}, "
                  f"Train Loss: {loss.item():.4f}, "
                  f"Val Loss: {val_loss.item():.4f}, "
                  f"Val Acc: {val_acc:.4f}, "
                  f"LR: {current_lr:.6f}")
        
        # 早停检查
        if early_stopping_enabled:
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                early_stopping_counter = 0
                # 保存最佳模型
                best_model = copy.deepcopy(model)
            else:
                early_stopping_counter += 1
                if early_stopping_counter >= patience:
                    print(f"早停激活: {patience}个epoch没有提升")
                    break
    
    # 如果启用了早停并且找到了最佳模型，返回最佳模型
    if early_stopping_enabled and best_model is not None:
        return best_model, losses, accuracies
    
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