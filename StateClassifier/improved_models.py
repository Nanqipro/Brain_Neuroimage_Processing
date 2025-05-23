#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
改进的模型架构
==============

实现多种先进的图神经网络架构用于脑网络状态分类

Author: AI Assistant
Date: 2024
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv, TransformerConv, global_mean_pool, global_max_pool, global_add_pool
from torch_geometric.nn import BatchNorm, LayerNorm
import numpy as np
import logging

logger = logging.getLogger(__name__)


class MultiHeadGraphAttention(nn.Module):
    """
    多头图注意力网络
    """
    
    def __init__(self, in_channels: int, out_channels: int, heads: int = 4, dropout: float = 0.2):
        """
        初始化多头图注意力层
        
        Parameters
        ----------
        in_channels : int
            输入特征维度
        out_channels : int
            输出特征维度
        heads : int
            注意力头数
        dropout : float
            Dropout比率
        """
        super().__init__()
        
        self.gat = GATConv(
            in_channels=in_channels,
            out_channels=out_channels,
            heads=heads,
            dropout=dropout,
            concat=False  # 使用平均而不是拼接
        )
        self.bn = BatchNorm(out_channels)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, edge_index):
        """
        前向传播
        
        Parameters
        ----------
        x : torch.Tensor
            节点特征
        edge_index : torch.Tensor
            边索引
            
        Returns
        -------
        torch.Tensor
            输出特征
        """
        x = self.gat(x, edge_index)
        x = self.bn(x)
        x = F.elu(x)
        x = self.dropout(x)
        return x


class ImprovedGCNLayer(nn.Module):
    """
    改进的GCN层，添加残差连接和层归一化
    """
    
    def __init__(self, in_channels: int, out_channels: int, dropout: float = 0.2):
        """
        初始化改进的GCN层
        
        Parameters
        ----------
        in_channels : int
            输入特征维度
        out_channels : int
            输出特征维度
        dropout : float
            Dropout比率
        """
        super().__init__()
        
        self.gcn = GCNConv(in_channels, out_channels)
        self.bn = BatchNorm(out_channels)
        self.dropout = nn.Dropout(dropout)
        
        # 残差连接的投影层
        if in_channels != out_channels:
            self.residual_proj = nn.Linear(in_channels, out_channels)
        else:
            self.residual_proj = nn.Identity()
            
    def forward(self, x, edge_index):
        """
        前向传播
        
        Parameters
        ----------
        x : torch.Tensor
            节点特征
        edge_index : torch.Tensor
            边索引
            
        Returns
        -------
        torch.Tensor
            输出特征
        """
        residual = self.residual_proj(x)
        
        x = self.gcn(x, edge_index)
        x = self.bn(x)
        x = F.elu(x)
        x = self.dropout(x)
        
        # 残差连接
        x = x + residual
        
        return x


class GraphTransformerLayer(nn.Module):
    """
    图Transformer层
    """
    
    def __init__(self, in_channels: int, out_channels: int, heads: int = 4, dropout: float = 0.2):
        """
        初始化图Transformer层
        
        Parameters
        ----------
        in_channels : int
            输入特征维度
        out_channels : int
            输出特征维度
        heads : int
            注意力头数
        dropout : float
            Dropout比率
        """
        super().__init__()
        
        self.transformer = TransformerConv(
            in_channels=in_channels,
            out_channels=out_channels,
            heads=heads,
            dropout=dropout,
            concat=False
        )
        self.bn = BatchNorm(out_channels)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, edge_index):
        """
        前向传播
        
        Parameters
        ----------
        x : torch.Tensor
            节点特征
        edge_index : torch.Tensor
            边索引
            
        Returns
        -------
        torch.Tensor
            输出特征
        """
        x = self.transformer(x, edge_index)
        x = self.bn(x)
        x = F.gelu(x)
        x = self.dropout(x)
        return x


class AdvancedGraphClassifier(nn.Module):
    """
    高级图分类器 - 结合多种图神经网络架构
    """
    
    def __init__(self, 
                 input_dim: int = 3,
                 hidden_dims: list = [64, 128, 64],
                 num_classes: int = 6,
                 dropout: float = 0.3,
                 architecture: str = 'hybrid'):
        """
        初始化高级图分类器
        
        Parameters
        ----------
        input_dim : int
            输入特征维度
        hidden_dims : list
            隐藏层维度列表
        num_classes : int
            分类类别数
        dropout : float
            Dropout比率
        architecture : str
            模型架构类型: 'gat', 'gcn', 'transformer', 'hybrid'
        """
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.num_classes = num_classes
        self.dropout = dropout
        self.architecture = architecture
        
        # 输入投影层
        self.input_proj = nn.Linear(input_dim, hidden_dims[0])
        self.input_bn = BatchNorm(hidden_dims[0])
        
        # 构建图卷积层
        self.graph_layers = nn.ModuleList()
        
        if architecture == 'gat':
            self._build_gat_layers()
        elif architecture == 'gcn':
            self._build_gcn_layers()
        elif architecture == 'transformer':
            self._build_transformer_layers()
        elif architecture == 'hybrid':
            self._build_hybrid_layers()
        else:
            raise ValueError(f"未知的架构类型: {architecture}")
        
        # 全局池化层
        self.global_pooling = nn.ModuleDict({
            'mean': lambda x, batch: global_mean_pool(x, batch),
            'max': lambda x, batch: global_max_pool(x, batch),
            'sum': lambda x, batch: global_add_pool(x, batch)
        })
        
        # 分类头
        final_dim = hidden_dims[-1] * 3  # 3种池化方式的拼接
        self.classifier = nn.Sequential(
            nn.Linear(final_dim, final_dim // 2),
            nn.BatchNorm1d(final_dim // 2),
            nn.ELU(),
            nn.Dropout(dropout),
            nn.Linear(final_dim // 2, final_dim // 4),
            nn.BatchNorm1d(final_dim // 4),
            nn.ELU(),
            nn.Dropout(dropout),
            nn.Linear(final_dim // 4, num_classes)
        )
        
        # 初始化权重
        self._initialize_weights()
        
    def _build_gat_layers(self):
        """构建GAT层"""
        for i in range(len(self.hidden_dims)):
            in_dim = self.hidden_dims[i-1] if i > 0 else self.hidden_dims[0]
            out_dim = self.hidden_dims[i]
            
            self.graph_layers.append(
                MultiHeadGraphAttention(in_dim, out_dim, heads=4, dropout=self.dropout)
            )
    
    def _build_gcn_layers(self):
        """构建改进的GCN层"""
        for i in range(len(self.hidden_dims)):
            in_dim = self.hidden_dims[i-1] if i > 0 else self.hidden_dims[0]
            out_dim = self.hidden_dims[i]
            
            self.graph_layers.append(
                ImprovedGCNLayer(in_dim, out_dim, dropout=self.dropout)
            )
    
    def _build_transformer_layers(self):
        """构建Transformer层"""
        for i in range(len(self.hidden_dims)):
            in_dim = self.hidden_dims[i-1] if i > 0 else self.hidden_dims[0]
            out_dim = self.hidden_dims[i]
            
            self.graph_layers.append(
                GraphTransformerLayer(in_dim, out_dim, heads=4, dropout=self.dropout)
            )
    
    def _build_hybrid_layers(self):
        """构建混合架构层"""
        layer_types = ['gat', 'gcn', 'transformer']
        
        for i in range(len(self.hidden_dims)):
            in_dim = self.hidden_dims[i-1] if i > 0 else self.hidden_dims[0]
            out_dim = self.hidden_dims[i]
            
            # 循环使用不同类型的层
            layer_type = layer_types[i % len(layer_types)]
            
            if layer_type == 'gat':
                self.graph_layers.append(
                    MultiHeadGraphAttention(in_dim, out_dim, heads=4, dropout=self.dropout)
                )
            elif layer_type == 'gcn':
                self.graph_layers.append(
                    ImprovedGCNLayer(in_dim, out_dim, dropout=self.dropout)
                )
            else:  # transformer
                self.graph_layers.append(
                    GraphTransformerLayer(in_dim, out_dim, heads=4, dropout=self.dropout)
                )
    
    def _initialize_weights(self):
        """初始化模型权重"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, (nn.BatchNorm1d, BatchNorm)):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
    
    def forward(self, edge_index, x, batch=None):
        """
        前向传播
        
        Parameters
        ----------
        edge_index : torch.Tensor
            边索引
        x : torch.Tensor
            节点特征
        batch : torch.Tensor, optional
            批次信息
            
        Returns
        -------
        torch.Tensor
            分类预测结果
        """
        # 输入投影
        x = self.input_proj(x)
        x = self.input_bn(x)
        x = F.elu(x)
        
        # 图卷积层
        for layer in self.graph_layers:
            x = layer(x, edge_index)
        
        # 全局池化
        if batch is None:
            # 如果没有batch信息，假设是单个图
            batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)
        
        # 多种池化方式的组合
        pooled_features = []
        for pool_name, pool_func in self.global_pooling.items():
            pooled = pool_func(x, batch)
            pooled_features.append(pooled)
        
        # 拼接所有池化结果
        x = torch.cat(pooled_features, dim=1)
        
        # 分类
        x = self.classifier(x)
        
        return x


class EnsembleGraphClassifier(nn.Module):
    """
    集成图分类器 - 结合多个不同架构的模型
    """
    
    def __init__(self, 
                 input_dim: int = 3,
                 hidden_dims: list = [64, 128, 64],
                 num_classes: int = 6,
                 dropout: float = 0.3):
        """
        初始化集成图分类器
        
        Parameters
        ----------
        input_dim : int
            输入特征维度
        hidden_dims : list
            隐藏层维度列表
        num_classes : int
            分类类别数
        dropout : float
            Dropout比率
        """
        super().__init__()
        
        # 创建多个不同架构的分类器
        self.models = nn.ModuleDict({
            'gat': AdvancedGraphClassifier(input_dim, hidden_dims, num_classes, dropout, 'gat'),
            'gcn': AdvancedGraphClassifier(input_dim, hidden_dims, num_classes, dropout, 'gcn'),
            'transformer': AdvancedGraphClassifier(input_dim, hidden_dims, num_classes, dropout, 'transformer')
        })
        
        # 集成权重
        self.ensemble_weights = nn.Parameter(torch.ones(len(self.models)) / len(self.models))
        
    def forward(self, edge_index, x, batch=None):
        """
        前向传播
        
        Parameters
        ----------
        edge_index : torch.Tensor
            边索引
        x : torch.Tensor
            节点特征
        batch : torch.Tensor, optional
            批次信息
            
        Returns
        -------
        torch.Tensor
            集成预测结果
        """
        # 获取每个模型的预测
        predictions = []
        for model in self.models.values():
            pred = model(edge_index, x, batch)
            predictions.append(pred)
        
        # 加权平均
        weights = F.softmax(self.ensemble_weights, dim=0)
        ensemble_pred = sum(w * pred for w, pred in zip(weights, predictions))
        
        return ensemble_pred


class AdaptiveGraphClassifier(nn.Module):
    """
    自适应图分类器 - 根据输入数据动态调整架构
    """
    
    def __init__(self, 
                 input_dim: int = 3,
                 hidden_dims: list = [64, 128, 64],
                 num_classes: int = 6,
                 dropout: float = 0.3):
        """
        初始化自适应图分类器
        
        Parameters
        ----------
        input_dim : int
            输入特征维度
        hidden_dims : list
            隐藏层维度列表
        num_classes : int
            分类类别数
        dropout : float
            Dropout比率
        """
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.num_classes = num_classes
        self.dropout = dropout
        
        # 图特征分析器
        self.graph_analyzer = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 3)  # 输出3个分数，对应3种架构
        )
        
        # 多种架构的分类器
        self.classifiers = nn.ModuleDict({
            'gat': AdvancedGraphClassifier(input_dim, hidden_dims, num_classes, dropout, 'gat'),
            'gcn': AdvancedGraphClassifier(input_dim, hidden_dims, num_classes, dropout, 'gcn'),
            'transformer': AdvancedGraphClassifier(input_dim, hidden_dims, num_classes, dropout, 'transformer')
        })
        
    def forward(self, edge_index, x, batch=None):
        """
        前向传播
        
        Parameters
        ----------
        edge_index : torch.Tensor
            边索引
        x : torch.Tensor
            节点特征
        batch : torch.Tensor, optional
            批次信息
            
        Returns
        -------
        torch.Tensor
            自适应预测结果
        """
        # 分析图特征，决定使用哪种架构
        graph_features = torch.mean(x, dim=0)  # 简单的图级特征
        architecture_scores = self.graph_analyzer(graph_features)
        architecture_weights = F.softmax(architecture_scores, dim=0)
        
        # 获取每个架构的预测
        predictions = []
        for name, classifier in self.classifiers.items():
            pred = classifier(edge_index, x, batch)
            predictions.append(pred)
        
        # 根据架构权重进行加权平均
        adaptive_pred = sum(w * pred for w, pred in zip(architecture_weights, predictions))
        
        return adaptive_pred


def create_improved_model(model_type: str = 'advanced', **kwargs):
    """
    创建改进的模型
    
    Parameters
    ----------
    model_type : str
        模型类型: 'advanced', 'ensemble', 'adaptive'
    **kwargs
        模型参数
        
    Returns
    -------
    nn.Module
        创建的模型实例
    """
    if model_type == 'advanced':
        return AdvancedGraphClassifier(**kwargs)
    elif model_type == 'ensemble':
        return EnsembleGraphClassifier(**kwargs)
    elif model_type == 'adaptive':
        return AdaptiveGraphClassifier(**kwargs)
    else:
        raise ValueError(f"未知的模型类型: {model_type}")


if __name__ == "__main__":
    # 测试代码
    import torch
    from torch_geometric.data import Data, Batch
    
    # 创建测试数据
    num_nodes = 50
    num_features = 3
    num_edges = 200
    
    x = torch.randn(num_nodes, num_features)
    edge_index = torch.randint(0, num_nodes, (2, num_edges))
    
    # 创建数据对象
    data = Data(x=x, edge_index=edge_index)
    batch = Batch.from_data_list([data, data])  # 批次大小为2
    
    # 测试不同模型
    for model_type in ['advanced', 'ensemble', 'adaptive']:
        print(f"\n测试 {model_type} 模型:")
        
        model = create_improved_model(
            model_type=model_type,
            input_dim=num_features,
            hidden_dims=[32, 64, 32],
            num_classes=6,
            dropout=0.2
        )
        
        model.eval()
        with torch.no_grad():
            output = model(batch.edge_index, batch.x, batch.batch)
            
        print(f"  输入形状: {batch.x.shape}")
        print(f"  输出形状: {output.shape}")
        print(f"  参数数量: {sum(p.numel() for p in model.parameters()):,}")
        print(f"  可训练参数: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}") 