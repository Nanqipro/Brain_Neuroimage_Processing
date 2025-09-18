"""
SOTA (State-of-the-Art) 图神经网络模型实现
包含近年来表现优秀的图神经网络架构
"""

import torch
import torch.nn.functional as F
from torch_geometric.nn import GINConv, TransformerConv, ChebConv, global_mean_pool, global_add_pool, global_max_pool
from torch_geometric.nn import BatchNorm, LayerNorm
import torch.nn as nn


class GIN(torch.nn.Module):
    """
    Graph Isomorphism Network (GIN)
    论文: How Powerful are Graph Neural Networks?
    特点: 理论上最强大的GNN架构之一
    """
    def __init__(self, num_features, hidden_dim, num_classes, dropout=0.3):
        super(GIN, self).__init__()
        
        # GIN层使用MLP作为聚合函数
        nn1 = nn.Sequential(
            nn.Linear(num_features, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        self.conv1 = GINConv(nn1)
        self.bn1 = BatchNorm(hidden_dim)
        
        nn2 = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        self.conv2 = GINConv(nn2)
        self.bn2 = BatchNorm(hidden_dim)
        
        nn3 = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        self.conv3 = GINConv(nn3)
        self.bn3 = BatchNorm(hidden_dim)
        
        # 分类器
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(hidden_dim*3, hidden_dim),
            torch.nn.BatchNorm1d(hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(hidden_dim, num_classes)
        )
        
        self.dropout = torch.nn.Dropout(dropout)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        
        # 第一层
        x1 = self.conv1(x, edge_index)
        x1 = self.bn1(x1)
        x1 = F.relu(x1)
        x1 = self.dropout(x1)
        
        # 第二层
        x2 = self.conv2(x1, edge_index)
        x2 = self.bn2(x2)
        x2 = F.relu(x2)
        x2 = self.dropout(x2)
        
        # 第三层
        x3 = self.conv3(x2, edge_index)
        x3 = self.bn3(x3)
        x3 = F.relu(x3)
        
        # 多尺度池化 (结合不同层的表示)
        x1_pool = global_add_pool(x1, batch)
        x2_pool = global_add_pool(x2, batch)
        x3_pool = global_add_pool(x3, batch)
        
        # 拼接多层表示
        x_combined = torch.cat([x1_pool, x2_pool, x3_pool], dim=1)
        
        # 分类
        out = self.mlp(x_combined)
        
        return F.log_softmax(out, dim=1)


class GraphTransformer(torch.nn.Module):
    """
    Graph Transformer
    使用注意力机制的图神经网络
    特点: 能够捕获全局信息，适合处理长程依赖
    """
    def __init__(self, num_features, hidden_dim, num_classes, dropout=0.3, heads=4):
        super(GraphTransformer, self).__init__()
        
        # Transformer卷积层
        self.conv1 = TransformerConv(num_features, hidden_dim, heads=heads, concat=False)
        self.bn1 = BatchNorm(hidden_dim)
        
        self.conv2 = TransformerConv(hidden_dim, hidden_dim, heads=heads, concat=False)
        self.bn2 = BatchNorm(hidden_dim)
        
        self.conv3 = TransformerConv(hidden_dim, hidden_dim, heads=heads, concat=False)
        self.bn3 = BatchNorm(hidden_dim)
        
        # 分类器
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(hidden_dim*3, hidden_dim),
            torch.nn.BatchNorm1d(hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(hidden_dim, num_classes)
        )
        
        self.dropout = torch.nn.Dropout(dropout)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        
        # 第一层
        x = self.conv1(x, edge_index)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.dropout(x)
        
        # 第二层
        x = self.conv2(x, edge_index)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.dropout(x)
        
        # 第三层
        x = self.conv3(x, edge_index)
        x = self.bn3(x)
        x = F.relu(x)
        
        # 三种池化方式的组合
        x_mean = global_mean_pool(x, batch)
        x_max = global_max_pool(x, batch)
        x_add = global_add_pool(x, batch)
        
        x_combined = torch.cat([x_mean, x_max, x_add], dim=1)
        
        # 分类
        out = self.mlp(x_combined)
        
        return F.log_softmax(out, dim=1)


class ChebNet(torch.nn.Module):
    """
    Chebyshev Network
    使用切比雪夫多项式的图卷积网络
    特点: 能够学习局部谱滤波器，计算效率高
    """
    def __init__(self, num_features, hidden_dim, num_classes, dropout=0.3, K=3):
        super(ChebNet, self).__init__()
        
        # ChebConv层，K是切比雪夫多项式的阶数
        self.conv1 = ChebConv(num_features, hidden_dim, K=K)
        self.bn1 = BatchNorm(hidden_dim)
        
        self.conv2 = ChebConv(hidden_dim, hidden_dim, K=K)
        self.bn2 = BatchNorm(hidden_dim)
        
        self.conv3 = ChebConv(hidden_dim, hidden_dim, K=K)
        self.bn3 = BatchNorm(hidden_dim)
        
        # 分类器
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(hidden_dim*2, hidden_dim),
            torch.nn.BatchNorm1d(hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(hidden_dim, num_classes)
        )
        
        self.dropout = torch.nn.Dropout(dropout)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        
        # 第一层
        x = self.conv1(x, edge_index)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.dropout(x)
        
        # 第二层
        x = self.conv2(x, edge_index)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.dropout(x)
        
        # 第三层
        x = self.conv3(x, edge_index)
        x = self.bn3(x)
        x = F.relu(x)
        
        # 池化
        x_mean = global_mean_pool(x, batch)
        x_add = global_add_pool(x, batch)
        
        x_combined = torch.cat([x_mean, x_add], dim=1)
        
        # 分类
        out = self.mlp(x_combined)
        
        return F.log_softmax(out, dim=1)


class EnsembleGNN(torch.nn.Module):
    """
    集成图神经网络
    结合多种GNN架构的优势
    """
    def __init__(self, num_features, hidden_dim, num_classes, dropout=0.3):
        super(EnsembleGNN, self).__init__()
        
        # 不同类型的图卷积层
        self.gin_conv = GINConv(
            nn.Sequential(
                nn.Linear(num_features, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim)
            )
        )
        
        self.transformer_conv = TransformerConv(num_features, hidden_dim, heads=4, concat=False)
        
        self.cheb_conv = ChebConv(num_features, hidden_dim, K=3)
        
        # 归一化层
        self.bn_gin = BatchNorm(hidden_dim)
        self.bn_transformer = BatchNorm(hidden_dim)
        self.bn_cheb = BatchNorm(hidden_dim)
        
        # 融合层
        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # 第二层（共享）
        self.conv2 = GINConv(
            nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim)
            )
        )
        self.bn2 = BatchNorm(hidden_dim)
        
        # 分类器
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(hidden_dim*2, hidden_dim),
            torch.nn.BatchNorm1d(hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(hidden_dim, num_classes)
        )
        
        self.dropout = torch.nn.Dropout(dropout)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        
        # 第一层：三种不同的卷积
        x_gin = self.gin_conv(x, edge_index)
        x_gin = self.bn_gin(x_gin)
        x_gin = F.relu(x_gin)
        
        x_transformer = self.transformer_conv(x, edge_index)
        x_transformer = self.bn_transformer(x_transformer)
        x_transformer = F.relu(x_transformer)
        
        x_cheb = self.cheb_conv(x, edge_index)
        x_cheb = self.bn_cheb(x_cheb)
        x_cheb = F.relu(x_cheb)
        
        # 融合不同卷积的结果
        x_fused = torch.cat([x_gin, x_transformer, x_cheb], dim=1)
        x_fused = self.fusion(x_fused)
        x_fused = self.dropout(x_fused)
        
        # 第二层
        x = self.conv2(x_fused, edge_index)
        x = self.bn2(x)
        x = F.relu(x)
        
        # 池化
        x_mean = global_mean_pool(x, batch)
        x_add = global_add_pool(x, batch)
        
        x_combined = torch.cat([x_mean, x_add], dim=1)
        
        # 分类
        out = self.mlp(x_combined)
        
        return F.log_softmax(out, dim=1)
