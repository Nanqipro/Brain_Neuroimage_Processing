"""
脑网络状态分类器模型定义模块

该模块定义了用于脑网络状态分类的图卷积神经网络(GCN)模型架构，
包括全局池化层和多层GCN分类器的实现。

作者: Clade 4
日期: 2025年5月23日
"""

from torch_geometric.nn import GCNConv, SGConv
import torch
import torch.nn as nn
import numpy as np
import torch_geometric.nn as pyg_nn
from collections import OrderedDict
import torch.nn.functional as F

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

class MultiLayerGCN(nn.Module):
    """
    多层图卷积神经网络模型
    
    用于脑网络状态分类的主要模型，由多个图卷积层和全连接分类层组成。
    
    参数
    ----------
    dropout : float, 可选
        Dropout比率，用于防止过拟合，默认为0.5
    num_classes : int, 可选
        输出类别数量，默认为40
        
    属性
    ----------
    conv0, conv1, conv2 : GCNConv
        三层图卷积层
    pool : GlobalPooling
        全局池化层
    classifier : nn.Sequential
        分类器，由全连接层、激活函数和dropout层组成
    """
    def __init__(self, dropout=0.5, num_classes=40):
        super(MultiLayerGCN, self).__init__()
        # 定义三层图卷积层
        self.conv0 = GCNConv(3, 32)  # 输入特征维度为3，输出为32
        self.conv1 = GCNConv(32, 32)
        self.conv2 = GCNConv(32, 32)

        # 全局池化层
        self.pool = GlobalPooling()
        
        # 分类器结构
        self.classifier = nn.Sequential(OrderedDict([
            ('fc0', nn.Linear(32 * 3 * 2, 16, bias=False)),  # 输入维度为32*3*2(三层GCN输出拼接后经过双池化)
            ('relu0', nn.LeakyReLU(negative_slope=0.2)),  # LeakyReLU激活函数
            ('drop0', nn.Dropout(p=dropout)),  # Dropout层
            ('fc2', nn.Linear(16, num_classes)),  # 最终分类层
        ]))

    def forward(self, adj, x):
        """
        模型前向传播函数
        
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
        # 依次通过三层图卷积，每层后应用LeakyReLU激活函数
        x0 = F.leaky_relu(self.conv0(x, adj), negative_slope=0.2)
        x1 = F.leaky_relu(self.conv1(x0, adj), negative_slope=0.2)
        x2 = F.leaky_relu(self.conv2(x1, adj), negative_slope=0.2)
        
        # 拼接三层卷积的输出特征
        x = torch.cat((x0, x1, x2), dim=1)
        
        # 全局池化
        x = self.pool(x)
        
        # 通过分类器得到最终输出
        x = self.classifier(x)
        return x