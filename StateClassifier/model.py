"""
脑网络状态分类器模型定义模块

该模块定义了用于脑网络状态分类的图卷积神经网络(GCN)模型架构，
包括全局池化层和多层GCN分类器的实现。

作者: Clade 4
日期: 2025年5月23日
改进版本: 增加BatchNorm、残差连接、注意力机制等现代技术
"""

from torch_geometric.nn import GCNConv, SGConv, BatchNorm
import torch
import torch.nn as nn
import numpy as np
import torch_geometric.nn as pyg_nn
from collections import OrderedDict
import torch.nn.functional as F

class ImprovedGlobalPooling(torch.nn.Module):
    """
    改进的全局池化层 - 结合多种池化方式
    
    该层将输入特征通过最大池化、平均池化和求和池化处理后拼接，
    提供更丰富的图级别特征表示。
    
    属性
    ----------
    max_pool : torch.nn.AdaptiveMaxPool1d
        自适应最大池化层
    avg_pool : torch.nn.AdaptiveAvgPool1d
        自适应平均池化层
    """
    def __init__(self):
        super(ImprovedGlobalPooling, self).__init__()
        self.max_pool = torch.nn.AdaptiveMaxPool1d(output_size=1)
        self.avg_pool = torch.nn.AdaptiveAvgPool1d(output_size=1)

    def forward(self, x):
        """
        前向传播函数
        
        参数
        ----------
        x : torch.Tensor
            输入特征张量，形状为 (num_nodes, num_features)
            
        返回
        -------
        torch.Tensor
            池化后的特征张量，包含最大池化、平均池化和求和池化结果的拼接
        """
        # 转置以适应1D池化层的输入要求 (features, nodes)
        x = x.T.unsqueeze(0)  # 形状: (1, num_features, num_nodes)
        batch_size = x.size(0)
        
        # 最大池化
        x_max = self.max_pool(x).view(batch_size, -1)
        
        # 平均池化
        x_avg = self.avg_pool(x).view(batch_size, -1)
        
        # 求和池化（手动实现）
        x_sum = torch.sum(x, dim=2, keepdim=False)
        
        # 拼接三种池化结果
        x = torch.cat((x_max, x_avg, x_sum), dim=-1)
        
        return x

class GlobalPooling(torch.nn.Module):
    """
    全局池化层，结合最大池化和平均池化的特征
    
    该层将输入特征通过最大池化和平均池化处理后拼接，
    用于提取图结构的全局特征表示。
    
    属性
    ----------
    max_pool : torch.nn.AdaptiveMaxPool1d
        自适应最大池化层
    avg_pool : torch.nn.AdaptiveAvgPool1d
        自适应平均池化层
    """
    def __init__(self):
        super(GlobalPooling, self).__init__()
        self.max_pool = torch.nn.AdaptiveMaxPool1d(output_size=1)
        self.avg_pool = torch.nn.AdaptiveAvgPool1d(output_size=1)

    def forward(self, x):
        """
        前向传播函数
        
        参数
        ----------
        x : torch.Tensor
            输入特征张量
            
        返回
        -------
        torch.Tensor
            池化后的特征张量，包含最大池化和平均池化结果的拼接
        """
        x = x.T.unsqueeze(0)  # 转置并增加批次维度
        batch_size = x.size(0)
        x0 = self.max_pool(x).view(batch_size, -1)  # 最大池化
        x1 = self.avg_pool(x).view(batch_size, -1)  # 平均池化
        x = torch.cat((x0, x1), dim=-1)  # 拼接两种池化结果
        # x = x.squeeze(0)
        return x

class GlobalMaxPooling(torch.nn.Module):
    """
    全局最大池化层
    
    仅使用最大池化操作提取图结构的全局特征表示。
    
    属性
    ----------
    max_pool : torch.nn.AdaptiveMaxPool1d
        自适应最大池化层
    """
    def __init__(self):
        super(GlobalMaxPooling, self).__init__()
        self.max_pool = torch.nn.AdaptiveMaxPool1d(output_size=1)

    def forward(self, x):
        """
        前向传播函数
        
        参数
        ----------
        x : torch.Tensor
            输入特征张量
            
        返回
        -------
        torch.Tensor
            经过最大池化后的特征张量
        """
        batch_size = x.size(0)
        x = self.max_pool(x).view(batch_size, -1)
        return x


class AttentionLayer(nn.Module):
    """
    简单的注意力机制层
    用于加权不同GCN层的特征
    """
    
    def __init__(self, input_dim):
        super(AttentionLayer, self).__init__()
        self.attention_weights = nn.Parameter(torch.randn(input_dim, 1))
        
    def forward(self, x):
        """
        计算注意力权重并应用到特征上
        
        参数
        ----------
        x : torch.Tensor
            输入特征
            
        返回
        -------
        torch.Tensor
            加权后的特征
        """
        # 计算注意力分数
        attention_scores = torch.matmul(x, self.attention_weights)
        attention_weights = F.softmax(attention_scores, dim=0)
        
        # 应用注意力权重
        attended_features = x * attention_weights
        
        return attended_features


class MultiLayerGCN(nn.Module):
    """
    改进的多层图卷积神经网络模型
    
    用于脑网络状态分类的主要模型，由多个图卷积层和全连接分类层组成。
    增加了BatchNorm、残差连接、注意力机制等现代深度学习技术。
    
    参数
    ----------
    dropout : float, 可选
        Dropout比率，用于防止过拟合，默认为0.5
    num_classes : int, 可选
        输出类别数量，默认为6（修改为6分类）
        
    属性
    ----------
    conv_layers : nn.ModuleList
        图卷积层列表
    batch_norms : nn.ModuleList
        批归一化层列表
    pool : ImprovedGlobalPooling
        改进的全局池化层
    attention : AttentionLayer
        注意力层
    classifier : nn.Sequential
        分类器，由全连接层、激活函数和dropout层组成
    """
    def __init__(self, dropout=0.3, num_classes=6):
        super(MultiLayerGCN, self).__init__()
        
        # 增大网络容量，添加更多层
        hidden_dims = [64, 128, 128, 64]  # 4层GCN，逐渐增大再减小
        
        # 图卷积层
        self.conv_layers = nn.ModuleList()
        self.batch_norms = nn.ModuleList()
        
        # 第一层：从输入特征到第一个隐藏层
        self.conv_layers.append(GCNConv(3, hidden_dims[0]))
        self.batch_norms.append(BatchNorm(hidden_dims[0]))
        
        # 中间层
        for i in range(len(hidden_dims) - 1):
            self.conv_layers.append(GCNConv(hidden_dims[i], hidden_dims[i + 1]))
            self.batch_norms.append(BatchNorm(hidden_dims[i + 1]))
        
        # 改进的全局池化层
        self.pool = ImprovedGlobalPooling()
        
        # 注意力机制
        total_features = sum(hidden_dims)  # 所有层特征拼接后的维度
        self.attention = AttentionLayer(total_features)
        
        # 计算分类器输入维度（三种池化方式 × 总特征数）
        classifier_input_dim = total_features * 3
        
        # 改进的分类器结构 - 更深更宽
        self.classifier = nn.Sequential(OrderedDict([
            # 第一层
            ('fc1', nn.Linear(classifier_input_dim, 512, bias=False)),
            ('bn1', nn.BatchNorm1d(512)),
            ('relu1', nn.ELU()),  # ELU激活函数
            ('drop1', nn.Dropout(p=dropout)),
            
            # 第二层
            ('fc2', nn.Linear(512, 256, bias=False)),
            ('bn2', nn.BatchNorm1d(256)),
            ('relu2', nn.ELU()),
            ('drop2', nn.Dropout(p=dropout)),
            
            # 第三层
            ('fc3', nn.Linear(256, 128, bias=False)),
            ('bn3', nn.BatchNorm1d(128)),
            ('relu3', nn.ELU()),
            ('drop3', nn.Dropout(p=dropout)),
            
            # 输出层
            ('fc_out', nn.Linear(128, num_classes)),
        ]))
        
        # 权重初始化
        self._initialize_weights()

    def _initialize_weights(self):
        """
        初始化模型权重
        使用Xavier初始化提高训练稳定性
        """
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.BatchNorm1d):
                # PyTorch标准BatchNorm1d
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
            elif isinstance(module, BatchNorm):
                # PyTorch Geometric BatchNorm - 检查是否有这些参数
                if hasattr(module, 'weight') and module.weight is not None:
                    nn.init.ones_(module.weight)
                if hasattr(module, 'bias') and module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, adj, x):
        """
        模型前向传播函数（改进版本）
        
        参数
        ----------
        adj : torch.Tensor
            邻接矩阵，表示图的边连接关系
        x : torch.Tensor
            节点特征矩阵
            
        返回
        -------
        torch.Tensor
            分类预测结果
        """
        # 存储每层的特征用于后续拼接
        layer_features = []
        
        # 依次通过所有图卷积层
        for i, (conv, bn) in enumerate(zip(self.conv_layers, self.batch_norms)):
            if i == 0:
                # 第一层
                x = conv(x, adj)
                x = bn(x)
                x = F.elu(x)  # 使用ELU激活函数
            else:
                # 后续层：添加残差连接（如果维度匹配）
                if x.shape[1] == conv.out_channels:
                    # 残差连接
                    residual = x
                    x = conv(x, adj)
                    x = bn(x)
                    x = F.elu(x + residual)  # 残差连接
                else:
                    # 普通连接
                    x = conv(x, adj)
                    x = bn(x)
                    x = F.elu(x)
            
            layer_features.append(x)
        
        # 拼接所有层的特征
        x = torch.cat(layer_features, dim=1)
        
        # 应用注意力机制
        x = self.attention(x)
        
        # 全局池化
        x = self.pool(x)
        
        # 通过分类器得到最终输出
        x = self.classifier(x)
        
        return x


class LightweightGCN(nn.Module):
    """
    轻量级GCN模型（备用选择）
    适用于计算资源有限的情况
    """
    def __init__(self, dropout=0.3, num_classes=6):
        super(LightweightGCN, self).__init__()
        
        # 较小的网络结构
        self.conv1 = GCNConv(3, 32)
        self.conv2 = GCNConv(32, 64)
        self.conv3 = GCNConv(64, 32)
        
        self.bn1 = BatchNorm(32)
        self.bn2 = BatchNorm(64)
        self.bn3 = BatchNorm(32)
        
        self.pool = GlobalPooling()
        
        # 简化的分类器
        self.classifier = nn.Sequential(
            nn.Linear(32 * 3 * 2, 64),  # 3层×2种池化
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, num_classes)
        )
        
    def forward(self, adj, x):
        # 第一层
        x1 = F.relu(self.bn1(self.conv1(x, adj)))
        
        # 第二层
        x2 = F.relu(self.bn2(self.conv2(x1, adj)))
        
        # 第三层
        x3 = F.relu(self.bn3(self.conv3(x2, adj)))
        
        # 拼接特征
        x = torch.cat([x1, x2, x3], dim=1)
        
        # 池化和分类
        x = self.pool(x)
        x = self.classifier(x)
        
        return x