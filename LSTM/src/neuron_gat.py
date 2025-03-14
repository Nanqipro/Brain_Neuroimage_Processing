"""
图注意力网络(GAT)模块

该模块包含用于神经元分析的图注意力网络相关组件，包括:
1. MultiHeadSelfAttention: 多头自注意力机制
2. NeuronGAT: 神经元图注意力网络模型
3. 相关的辅助函数

此模块设计为可独立使用，也可以与其他神经网络模块集成。
"""

import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import json
import seaborn as sns
from typing import Dict, List, Any, Optional, Tuple, Union

import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Linear, Dropout, LayerNorm, BatchNorm1d
from torch_geometric.nn import GATConv, JumpingKnowledge, GlobalAttention, GATv2Conv
from torch_geometric.utils import to_networkx, remove_self_loops, add_self_loops
from sklearn.manifold import TSNE


class SELayer(nn.Module):
    """
    Squeeze-and-Excitation注意力模块
    
    用于动态调整特征通道重要性
    """
    def __init__(self, channel, reduction=4):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        # 输入x形状: [num_nodes, channels]
        b, c = x.size()
        # 重塑以适应池化层
        x_reshaped = x.unsqueeze(0).transpose(1, 2)  # [1, channels, num_nodes]
        
        # 安全处理: 确保维度正确
        try:
            y = self.avg_pool(x_reshaped).view(c)  # [channels]
            y = self.fc(y).view(1, c, 1)  # [1, channels, 1]
            # 重塑回原始形状并应用注意力
            y = y.expand_as(x_reshaped)  # [1, channels, num_nodes]
            x_se = x_reshaped * y  # 应用通道注意力
            return x_se.transpose(1, 2).squeeze(0)  # [num_nodes, channels]
        except RuntimeError as e:
            # 如果遇到维度不匹配问题，打印警告并返回原始输入
            print(f"SELayer维度错误: {e}，跳过SE层并返回原始输入")
            return x  # 如有错误，直接返回原始输入


class MultiHeadSelfAttention(nn.Module):
    """
    增强型多头自注意力机制
    
    用于增强节点特征表示的自注意力模块，包含残差连接和层归一化
    """
    def __init__(self, in_channels, num_heads=8, dropout=0.1, use_layer_norm=True):
        """
        初始化多头自注意力层
        
        参数:
            in_channels: 输入特征维度
            num_heads: 注意力头数量
            dropout: Dropout比率
            use_layer_norm: 是否使用层归一化
        """
        super(MultiHeadSelfAttention, self).__init__()
        self.num_heads = num_heads
        self.in_channels = in_channels
        self.head_dim = in_channels // num_heads
        assert in_channels % num_heads == 0, "in_channels must be divisible by num_heads"
        self.use_layer_norm = use_layer_norm
        
        # 定义查询、键、值投影
        self.query = Linear(in_channels, in_channels)
        self.key = Linear(in_channels, in_channels)
        self.value = Linear(in_channels, in_channels)
        
        # 输出投影
        self.out_proj = Linear(in_channels, in_channels)
        
        # Dropout
        self.dropout = Dropout(dropout)
        
        # 层归一化
        if use_layer_norm:
            self.layer_norm1 = LayerNorm(in_channels)
            self.layer_norm2 = LayerNorm(in_channels)
        
        # 前馈网络
        self.ffn = nn.Sequential(
            Linear(in_channels, in_channels * 2),
            nn.GELU(),
            Dropout(dropout),
            Linear(in_channels * 2, in_channels)
        )
        
        # 缩放因子
        self.scale = self.head_dim ** -0.5
        
    def forward(self, x):
        """
        前向传播
        
        参数:
            x: 输入特征 [batch_size, seq_len, channels] 或 [num_nodes, channels]
            
        返回:
            output: 注意力增强的特征
        """
        original_dim = x.dim()
        
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
        
        # 第一个残差块: 多头自注意力
        if self.use_layer_norm:
            residual = x
            x = self.layer_norm1(x)
        
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
        x = self.out_proj(context)
        
        # 应用残差连接
        if self.use_layer_norm:
            x = x + residual
        
        # 第二个残差块: 前馈网络
        if self.use_layer_norm:
            residual = x
            x = self.layer_norm2(x)
        
        # 应用前馈网络
        x_ffn = self.ffn(x)
        
        # 应用残差连接
        if self.use_layer_norm:
            x = x_ffn + residual
        else:
            x = x_ffn + x
        
        # 恢复原始维度
        if original_dim == 2:
            x = x.squeeze(0)
            
        return x


class GNNBlock(nn.Module):
    """增强型GAT块，包含注意力、归一化和残差连接"""
    
    def __init__(self, in_channels, out_channels, heads=8, dropout=0.1, 
                 edge_dim=None, use_gatv2=True, use_layer_norm=True,
                 use_se=True, reduction=8, alpha=0.2):
        super(GNNBlock, self).__init__()
        
        # 选择GAT版本
        if use_gatv2:
            self.gat = GATv2Conv(
                in_channels, out_channels // heads, heads=heads,
                dropout=dropout, edge_dim=edge_dim, add_self_loops=True
            )
        else:
            self.gat = GATConv(
                in_channels, out_channels // heads, heads=heads,
                dropout=dropout, edge_dim=edge_dim, add_self_loops=True
            )
        
        # 层归一化
        self.use_layer_norm = use_layer_norm
        if use_layer_norm:
            self.layer_norm = LayerNorm(out_channels)
        
        # Squeeze-and-Excitation模块
        self.use_se = use_se
        if use_se:
            self.se = SELayer(out_channels, reduction=reduction)
        
        # 激活函数
        self.act = nn.LeakyReLU(alpha)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, edge_index, edge_weight=None, residual=None):
        # 应用GAT
        if edge_weight is not None:
            out = self.gat(x, edge_index, edge_weight)
        else:
            out = self.gat(x, edge_index)
        
        # 应用Squeeze-and-Excitation
        if self.use_se:
            try:
                out = self.se(out)
            except RuntimeError as e:
                print(f"SE层错误: {e}，跳过SE层")
                # 出错时跳过SE层
        
        # 应用残差连接
        if residual is not None and residual.size() == out.size():
            out = out + residual
        
        # 应用层归一化
        if self.use_layer_norm:
            try:
                # 检查形状是否匹配
                if hasattr(self.layer_norm, 'normalized_shape'):
                    expected_shape = self.layer_norm.normalized_shape[0]
                    actual_shape = out.size(-1)
                    if expected_shape != actual_shape:
                        print(f"层归一化维度不匹配: 期望 {expected_shape}, 实际 {actual_shape}，跳过层归一化")
                    else:
                        out = self.layer_norm(out)
                else:
                    out = self.layer_norm(out)
            except RuntimeError as e:
                print(f"层归一化错误: {e}，跳过层归一化")
                # 出错时跳过层归一化
        
        # 应用激活函数和dropout
        out = self.act(out)
        out = self.dropout(out)
        
        return out


class NeuronGAT(torch.nn.Module):
    """
    增强型神经元图注意力网络
    
    GAT模型通过学习节点之间的注意力权重来捕获神经元之间的功能关系，
    包含多种先进技术增强性能：
    1. GATv2卷积层 - 使用动态注意力机制
    2. 通道注意力 - 使用SE模块
    3. 残差连接和层归一化 - 稳定训练过程
    4. 跳跃连接 - 结合多层特征
    5. 辅助任务 - 通过多任务学习增强泛化能力
    """
    
    def __init__(self, in_channels, hidden_channels, out_channels, heads=8, dropout=0.2, 
                 residual=True, num_layers=4, edge_dim=None, alpha=0.2, use_gatv2=True,
                 jk_mode='cat', use_layer_norm=True, use_se=True, se_reduction=8, 
                 use_auxiliary_tasks=True):
        """
        初始化增强型神经元GAT模型
        
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
            use_gatv2: 是否使用GATv2卷积
            jk_mode: 跳跃连接模式 ('max', 'lstm', 'cat')
            use_layer_norm: 是否使用层归一化
            use_se: 是否使用Squeeze-and-Excitation模块
            se_reduction: SE模块的压缩比例
            use_auxiliary_tasks: 是否使用辅助任务
        """
        super(NeuronGAT, self).__init__()
        self.num_layers = num_layers
        self.residual = residual
        self.dropout_rate = dropout
        self.jk_mode = jk_mode
        self.use_auxiliary_tasks = use_auxiliary_tasks
        
        # 特征初始处理
        self.input_proj = nn.Sequential(
            Linear(in_channels, hidden_channels),
            nn.LayerNorm(hidden_channels) if use_layer_norm else nn.Identity(),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        
        # GAT块
        self.gnn_blocks = nn.ModuleList()
        
        # 第一层使用输入维度
        self.gnn_blocks.append(
            GNNBlock(
                hidden_channels, hidden_channels, heads=heads, 
                dropout=dropout, edge_dim=edge_dim, use_gatv2=use_gatv2,
                use_layer_norm=use_layer_norm, use_se=use_se, 
                reduction=se_reduction, alpha=alpha
            )
        )
        
        # 中间层
        for _ in range(num_layers - 1):
            self.gnn_blocks.append(
                GNNBlock(
                    hidden_channels, hidden_channels, heads=heads, 
                    dropout=dropout, edge_dim=edge_dim, use_gatv2=use_gatv2,
                    use_layer_norm=use_layer_norm, use_se=use_se, 
                    reduction=se_reduction, alpha=alpha
                )
            )
        
        # 跳跃连接
        if jk_mode == 'cat':
            final_dim = hidden_channels * num_layers
        else:
            final_dim = hidden_channels
            
        self.jk = JumpingKnowledge(mode=jk_mode)
        
        # 全局注意力池化
        self.glob_attn = GlobalAttention(
            nn.Sequential(
                nn.Linear(hidden_channels, hidden_channels // 2), 
                nn.GELU(), 
                nn.Linear(hidden_channels // 2, 1)
            )
        )
        
        # 多层感知机分类器
        self.mlp = nn.Sequential(
            nn.Linear(final_dim, hidden_channels),
            nn.LayerNorm(hidden_channels) if use_layer_norm else nn.Identity(),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels, hidden_channels // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels // 2, out_channels)
        )
        
        # 辅助任务
        if use_auxiliary_tasks:
            # 边预测任务
            self.edge_predictor = nn.Sequential(
                nn.Linear(hidden_channels * 2, hidden_channels),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_channels, 1),
                nn.Sigmoid()
            )
            
            # 节点特征重构任务
            self.node_reconstructor = nn.Sequential(
                nn.Linear(hidden_channels, hidden_channels),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_channels, in_channels)
            )
        
        # 残差连接投影层
        if residual:
            self.res_projs = nn.ModuleList()
            for _ in range(num_layers):
                self.res_projs.append(nn.Linear(hidden_channels, hidden_channels, bias=False))
        
        # 自注意力增强
        self.self_attention = MultiHeadSelfAttention(
            hidden_channels, num_heads=4, dropout=dropout, 
            use_layer_norm=use_layer_norm
        )
        
        # 权重初始化
        self.apply(self._init_weights)
        
    def _init_weights(self, module):
        """初始化模型权重"""
        if isinstance(module, nn.Linear):
            # 使用Kaiming初始化线性层
            nn.init.kaiming_normal_(module.weight, a=0.01, nonlinearity='leaky_relu')
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        
    def forward(self, data):
        """
        前向传播
        
        参数:
            data: PyTorch Geometric数据对象
        
        返回:
            out: 模型主要输出
            aux_outputs: 辅助任务输出字典（如启用）
        """
        x, edge_index = data.x, data.edge_index
        edge_weight = data.edge_attr if hasattr(data, 'edge_attr') else None
        
        # 特征初始处理
        x = self.input_proj(x)
        
        # 存储每层的输出用于跳跃连接
        layer_outputs = []
        
        # 通过所有GAT层
        for i in range(self.num_layers):
            # 残差连接
            res = self.res_projs[i](x) if self.residual else None
            
            # 应用GAT块
            x = self.gnn_blocks[i](x, edge_index, edge_weight, res)
            
            # 存储层输出
            layer_outputs.append(x)
        
        # 应用跳跃连接
        try:
            if self.jk_mode == 'cat':
                x = self.jk(layer_outputs)  # 输出形状: [num_nodes, hidden_channels * num_layers]
            else:
                x = self.jk(layer_outputs)  # 输出形状: [num_nodes, hidden_channels]
        except RuntimeError as e:
            print(f"跳跃连接错误: {e}，使用最后一层输出替代")
            # 出错时直接使用最后一层的输出
            x = layer_outputs[-1]
        
        # 存储节点嵌入用于辅助任务
        node_embeddings = x
        
        # 应用自注意力增强
        try:
            x = self.self_attention(x)
        except RuntimeError as e:
            print(f"自注意力模块错误: {e}，跳过自注意力处理")
            # 出错时跳过自注意力处理
        
        # 应用全局注意力池化（如果需要）
        batch = data.batch if hasattr(data, 'batch') else None
        if batch is not None:
            try:
                x = self.glob_attn(x, batch)
            except RuntimeError as e:
                print(f"全局注意力池化错误: {e}，跳过全局注意力池化")
                # 出错时跳过全局注意力池化
        
        # 使用MLP分类器生成主要输出
        out = self.mlp(x)
        
        # 如果不使用辅助任务，直接返回主要输出
        if not self.use_auxiliary_tasks:
            return out
            
        # 辅助任务
        aux_outputs = {}
        
        # 边预测任务
        if hasattr(data, 'edge_index'):
            # 采样一些边和非边用于预测
            pos_edge_index, neg_edge_index = self._sample_edges(data.edge_index, data.num_nodes)
            
            # 计算边预测输出
            edge_pred_loss = self._compute_edge_pred_loss(
                node_embeddings, pos_edge_index, neg_edge_index
            )
            aux_outputs['edge_pred_loss'] = edge_pred_loss
        
        # 节点特征重构任务
        node_recon = self.node_reconstructor(node_embeddings)
        node_recon_loss = F.mse_loss(node_recon, data.x)
        aux_outputs['node_recon_loss'] = node_recon_loss
        
        return out, aux_outputs
        
    def _sample_edges(self, edge_index, num_nodes, neg_ratio=1.0):
        """采样正边和负边用于边预测任务"""
        # 移除自环
        edge_index, _ = remove_self_loops(edge_index)
        
        # 正边
        pos_edge_index = edge_index
        
        # 负采样 - 实际实现中可能需要更高效的实现
        num_neg_samples = int(pos_edge_index.size(1) * neg_ratio)
        neg_edge_index = self._negative_sampling(edge_index, num_nodes, num_neg_samples)
        
        return pos_edge_index, neg_edge_index
    
    def _negative_sampling(self, edge_index, num_nodes, num_neg_samples):
        """负采样 - 生成不在图中的边"""
        # 简单实现 - 实际应用中可能需要更高效的方法
        num_neg_samples = min(num_neg_samples, num_nodes * num_nodes - edge_index.size(1))
        
        # 创建邻接矩阵
        adj = torch.zeros((num_nodes, num_nodes), device=edge_index.device)
        adj[edge_index[0], edge_index[1]] = 1
        
        # 找到所有不在图中的边
        mask = adj == 0
        neg_row, neg_col = mask.nonzero(as_tuple=True)
        
        # 随机选择负样本
        perm = torch.randperm(neg_row.size(0))[:num_neg_samples]
        neg_edge_index = torch.stack([neg_row[perm], neg_col[perm]], dim=0)
        
        return neg_edge_index
    
    def _compute_edge_pred_loss(self, x, pos_edge_index, neg_edge_index):
        """计算边预测任务的损失"""
        # 正边预测
        pos_out = self._get_edge_scores(x, pos_edge_index)
        neg_out = self._get_edge_scores(x, neg_edge_index)
        
        # 使用BCE损失
        pos_loss = F.binary_cross_entropy(
            pos_out, torch.ones_like(pos_out)
        )
        neg_loss = F.binary_cross_entropy(
            neg_out, torch.zeros_like(neg_out)
        )
        
        return pos_loss + neg_loss
    
    def _get_edge_scores(self, x, edge_index):
        """计算边的分数"""
        # 获取边的首尾节点特征
        src, dst = edge_index
        src_x = x[src]
        dst_x = x[dst]
        
        # 拼接特征
        edge_features = torch.cat([src_x, dst_x], dim=1)
        
        # 使用边预测器计算分数
        return self.edge_predictor(edge_features).view(-1)
        
    def get_embeddings(self, data):
        """
        获取节点嵌入向量
        
        参数:
            data: PyTorch Geometric数据对象
            
        返回:
            embeddings: 节点嵌入向量
        """
        x, edge_index = data.x, data.edge_index
        edge_weight = data.edge_attr if hasattr(data, 'edge_attr') else None
        
        # 特征初始处理
        x = self.input_proj(x)
        
        # 存储每层的输出用于跳跃连接
        layer_outputs = []
        
        # 通过所有GAT层
        for i in range(self.num_layers):
            # 残差连接
            res = self.res_projs[i](x) if self.residual else None
            
            # 应用GAT块
            x = self.gnn_blocks[i](x, edge_index, edge_weight, res)
            
            # 存储层输出
            layer_outputs.append(x)
        
        # 应用跳跃连接
        if self.jk_mode == 'cat':
            embeddings = self.jk(layer_outputs)  # 输出形状: [num_nodes, hidden_channels * num_layers]
        else:
            embeddings = self.jk(layer_outputs)  # 输出形状: [num_nodes, hidden_channels]
        
        # 应用自注意力增强
        embeddings = self.self_attention(embeddings)
        
        return embeddings


def visualize_gat_attention_weights(attention_weights, neuron_names, output_dir, title='Neuron GAT Attention Weights'):
    """
    可视化GAT注意力权重
    
    参数:
        attention_weights: 注意力权重矩阵
        neuron_names: 神经元名称列表
        output_dir: 输出目录
        title: 图表标题
        
    返回:
        viz_path: 可视化图像保存路径
    """
    plt.figure(figsize=(14, 12))
    sns.heatmap(attention_weights, annot=False, cmap='viridis', 
               xticklabels=neuron_names, yticklabels=neuron_names)
    plt.title(title)
    plt.tight_layout()
    
    # 保存图像
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    viz_path = os.path.join(output_dir, 'gat_attention_weights.png')
    plt.savefig(viz_path, dpi=300)
    plt.close()
    return viz_path


def visualize_gat_topology(topology_data_path):
    """
    根据GAT拓扑数据文件生成静态拓扑结构图
    
    参数:
        topology_data_path: GAT拓扑数据JSON文件路径
        
    返回:
        output_path: 生成的可视化图像路径
    """
    print("\n开始生成GAT静态拓扑结构图...")
    
    try:
        # 读取拓扑数据
        with open(topology_data_path, 'r') as f:
            data = json.load(f)
        
        # 创建NetworkX图对象
        G = nx.Graph()
        
        # 添加节点
        for node in data['nodes']:
            G.add_node(node['id'])
        
        # 添加边
        for edge in data['edges']:
            G.add_edge(edge['source'], edge['target'], weight=edge['similarity'])
        
        # 检测社区结构
        import community as community_louvain
        communities = community_louvain.best_partition(G)
        
        # 为不同社区分配不同颜色
        community_colors = [
            "#FF6347", "#4682B4", "#32CD32", "#FFD700", "#9370DB", 
            "#20B2AA", "#FF69B4", "#8A2BE2", "#00CED1", "#FF8C00",
            "#1E90FF", "#FF1493", "#00FA9A", "#DC143C", "#BA55D3"
        ]
        
        # 获取社区数量
        num_communities = len(set(communities.values()))
        if num_communities > len(community_colors):
            community_colors = community_colors * (num_communities // len(community_colors) + 1)
        
        # 创建节点颜色映射
        node_colors = [community_colors[communities[node]] for node in G.nodes()]
        
        # 设置绘图参数
        plt.figure(figsize=(15, 15))
        
        # 使用spring布局，增加k值和迭代次数使节点更分散
        pos = nx.spring_layout(G, 
                             k=2.0,           # 增加节点间理想距离
                             iterations=100,   # 增加迭代次数
                             seed=42)         # 设置随机种子保证结果可重现
        
        # 绘制边（使用权重确定边的宽度，降低alpha值使边更透明）
        edge_weights = [G[u][v]['weight'] * 1.5 for u, v in G.edges()]  # 稍微减小边的宽度系数
        nx.draw_networkx_edges(G, pos, width=edge_weights, alpha=0.2, edge_color='gray')  # 降低边的不透明度
        
        # 绘制节点（稍微增大节点大小）
        node_sizes = [400 for _ in G.nodes()]  # 增加节点大小
        nx.draw_networkx_nodes(G, pos, node_size=node_sizes, node_color=node_colors, alpha=0.9)  # 增加节点不透明度
        
        # 添加节点标签（调整字体大小）
        labels = {node: f'N{node+1}' for node in G.nodes()}
        nx.draw_networkx_labels(G, pos, labels, font_size=9, font_weight='bold')  # 稍微增加字体大小
        
        plt.title('GAT-Based Neuron Network Topology', fontsize=16)
        plt.axis('off')
        
        # 添加社区图例
        legend_elements = [plt.Line2D([0], [0], marker='o', color='w', 
                                    markerfacecolor=community_colors[i],
                                    markersize=10, label=f'Community {i+1}')
                         for i in range(num_communities)]
        plt.legend(handles=legend_elements, loc='center left', bbox_to_anchor=(1, 0.5))
        
        # 保存图像
        output_path = os.path.join(os.path.dirname(topology_data_path), 'gat_topology_static.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"GAT静态拓扑结构图已保存至: {output_path}")
        return output_path
        
    except Exception as e:
        print(f"生成GAT静态拓扑结构图时出错: {str(e)}")
        import traceback
        print(f"错误详情:\n{traceback.format_exc()}")
        return None 