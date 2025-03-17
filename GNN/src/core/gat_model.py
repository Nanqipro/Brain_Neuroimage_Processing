"""
GAT模型模块

该模块定义了用于神经元拓扑结构分析的图注意力网络模型
"""

import torch
import torch.nn.functional as F
from torch_geometric.nn import GATConv, global_mean_pool
from torch.nn import Linear, BatchNorm1d, Dropout
from typing import Dict, List, Tuple, Optional, Union, Any

class GATEncoder(torch.nn.Module):
    """
    GAT编码器
    
    使用图注意力网络提取神经元节点的表示
    """
    
    def __init__(self, in_channels: int, hidden_channels: int, out_channels: int, 
                 num_layers: int, heads: int = 4, dropout: float = 0.2):
        """
        初始化GAT编码器
        
        参数:
            in_channels: 输入特征维度
            hidden_channels: 隐藏层维度
            out_channels: 输出特征维度
            num_layers: GAT层数
            heads: 注意力头数
            dropout: Dropout比例
        """
        super(GATEncoder, self).__init__()
        
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels
        self.num_layers = num_layers
        self.heads = heads
        self.dropout = dropout
        
        # 创建GAT层
        self.convs = torch.nn.ModuleList()
        self.batch_norms = torch.nn.ModuleList()
        
        # 第一层
        self.convs.append(GATConv(in_channels, hidden_channels // heads, heads=heads))
        self.batch_norms.append(BatchNorm1d(hidden_channels))
        
        # 中间层
        for _ in range(num_layers - 2):
            self.convs.append(GATConv(hidden_channels, hidden_channels // heads, heads=heads))
            self.batch_norms.append(BatchNorm1d(hidden_channels))
        
        # 最后一层
        if num_layers > 1:
            # 最后一层使用单头注意力
            self.convs.append(GATConv(hidden_channels, out_channels, heads=1, concat=False))
            self.batch_norms.append(BatchNorm1d(out_channels))
        
        # 记录中间表示以便可视化
        self.last_hidden = None
        # 记录注意力权重以便分析
        self.attention_weights = None
        
    def forward(self, x, edge_index):
        """
        前向传播
        
        参数:
            x: 节点特征矩阵
            edge_index: 边索引
            
        返回:
            节点嵌入
        """
        # 应用GAT层
        for i, conv in enumerate(self.convs[:-1]):
            x = conv(x, edge_index)
            x = self.batch_norms[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        
        # 最后一层
        if self.num_layers > 0:
            # 获取最后一层的注意力权重
            x, attention_weights = self.convs[-1](x, edge_index, return_attention_weights=True)
            self.attention_weights = attention_weights
            x = self.batch_norms[-1](x)
        
        # 保存最后的隐藏表示
        self.last_hidden = x.detach()
        
        return x
    
    def get_embeddings(self, data):
        """
        获取节点嵌入
        
        参数:
            data: PyTorch Geometric数据对象
            
        返回:
            节点嵌入
        """
        x, edge_index = data.x, data.edge_index            
        return self.forward(x, edge_index)
    
    def get_attention_weights(self):
        """
        获取注意力权重
        
        返回:
            注意力权重
        """
        return self.attention_weights


class NeuronGAT(torch.nn.Module):
    """
    神经元GAT模型
    
    用于分析神经元网络拓扑结构的GAT模型
    """
    
    def __init__(self, in_channels: int, hidden_channels: int, out_channels: int, 
                 num_layers: int, heads: int = 4, dropout: float = 0.2, 
                 task_type: str = 'embedding'):
        """
        初始化神经元GAT模型
        
        参数:
            in_channels: 输入特征维度
            hidden_channels: 隐藏层维度
            out_channels: 输出特征维度
            num_layers: GAT层数
            heads: 注意力头数
            dropout: Dropout比例
            task_type: 任务类型，可选 'embedding'、'classification'、'regression'
        """
        super(NeuronGAT, self).__init__()
        
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels
        self.task_type = task_type
        
        # GAT编码器
        self.encoder = GATEncoder(
            in_channels=in_channels,
            hidden_channels=hidden_channels,
            out_channels=out_channels,
            num_layers=num_layers,
            heads=heads,
            dropout=dropout
        )
        
        # 如果是分类或回归任务，添加预测层
        if task_type in ['classification', 'regression']:
            self.predictor = torch.nn.Sequential(
                Linear(out_channels, hidden_channels),
                BatchNorm1d(hidden_channels),
                Dropout(dropout),
                Linear(hidden_channels, 1 if task_type == 'regression' else 4)  # 假设有4种行为类别
            )
    
    def forward(self, data):
        """
        前向传播
        
        参数:
            data: PyTorch Geometric数据对象
            
        返回:
            根据任务类型返回节点嵌入或预测结果
        """
        # 获取特征和图结构
        x, edge_index = data.x, data.edge_index
        
        # 获取节点嵌入
        embeddings = self.encoder(x, edge_index)
        
        # 记录最后的隐藏表示以便可视化
        self.last_hidden = self.encoder.last_hidden
        
        # 如果只需要嵌入，直接返回
        if self.task_type == 'embedding':
            return embeddings
        
        # 否则进行预测
        if hasattr(data, 'batch'):
            # 图级别任务
            pooled = global_mean_pool(embeddings, data.batch)
            predictions = self.predictor(pooled)
        else:
            # 节点级别任务
            predictions = self.predictor(embeddings)
            
        return predictions
    
    def get_embeddings(self, data):
        """
        获取节点嵌入
        
        参数:
            data: PyTorch Geometric数据对象
            
        返回:
            节点嵌入
        """
        return self.encoder.get_embeddings(data)
    
    def get_attention_weights(self):
        """
        获取注意力权重
        
        返回:
            注意力权重矩阵
        """
        return self.encoder.get_attention_weights() 