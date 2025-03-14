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
from torch.nn import Linear, Dropout
from torch_geometric.nn import GATConv, JumpingKnowledge, GlobalAttention
from torch_geometric.utils import to_networkx
from sklearn.manifold import TSNE


class MultiHeadSelfAttention(nn.Module):
    """
    多头自注意力机制
    
    用于增强节点特征表示的自注意力模块
    """
    def __init__(self, in_channels, num_heads=4, dropout=0.1):
        """
        初始化多头自注意力层
        
        参数:
            in_channels: 输入特征维度
            num_heads: 注意力头数量
            dropout: Dropout比率
        """
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
        """
        前向传播
        
        参数:
            x: 输入特征 [batch_size, seq_len, channels] 或 [num_nodes, channels]
            
        返回:
            output: 注意力增强的特征
        """
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
    """
    使用图注意力网络进行神经元功能模块识别
    
    GAT模型通过学习节点之间的注意力权重来捕获神经元之间的功能关系
    """
    
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
        self.hidden_channels = hidden_channels
        self.heads = heads
        
        # 输入特征的归一化层
        self.input_norm = torch.nn.LayerNorm(in_channels)
        
        # 初始层
        self.convs = nn.ModuleList()
        self.batch_norms = nn.ModuleList()
        self.layer_norms = nn.ModuleList()
        
        # 第一层
        self.convs.append(GATConv(in_channels, hidden_channels, heads=heads, 
                           dropout=dropout, edge_dim=edge_dim, add_self_loops=True))
        self.batch_norms.append(torch.nn.BatchNorm1d(hidden_channels * heads))
        self.layer_norms.append(torch.nn.LayerNorm(hidden_channels * heads))
        
        # 中间层
        for i in range(num_layers - 2):
            self.convs.append(
                GATConv(hidden_channels * heads, hidden_channels, heads=heads, 
                       dropout=dropout, edge_dim=edge_dim, add_self_loops=True)
            )
            self.batch_norms.append(torch.nn.BatchNorm1d(hidden_channels * heads))
            self.layer_norms.append(torch.nn.LayerNorm(hidden_channels * heads))
        
        # 最后一个GAT层
        if num_layers > 1:
            self.convs.append(
                GATConv(hidden_channels * heads, hidden_channels, heads=1, 
                       dropout=dropout, edge_dim=edge_dim, add_self_loops=True)
            )
            self.batch_norms.append(torch.nn.BatchNorm1d(hidden_channels))
            self.layer_norms.append(torch.nn.LayerNorm(hidden_channels))
        
        # 改进型跳跃连接，使用加权模式而非简单的max模式
        self.jk = JumpingKnowledge(mode='cat')
        self.jk_linear = nn.Linear(hidden_channels * num_layers, hidden_channels)
        
        # 全局注意力池化层 - 使用更复杂的门控注意力机制
        self.glob_attn = GlobalAttention(
            nn.Sequential(
                nn.Linear(hidden_channels, hidden_channels), 
                nn.LayerNorm(hidden_channels),  # 添加层归一化
                nn.LeakyReLU(alpha), 
                nn.Dropout(dropout),
                nn.Linear(hidden_channels, 1)
            )
        )
        
        # 改进的分类器，使用更宽的隐藏层和更深的结构
        self.mlp = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels * 2),
            nn.LayerNorm(hidden_channels * 2),
            nn.GELU(),  # 使用GELU激活函数替代LeakyReLU
            nn.Dropout(dropout),
            nn.Linear(hidden_channels * 2, hidden_channels),
            nn.LayerNorm(hidden_channels),
            nn.GELU(),
            nn.Dropout(dropout * 0.8),  # 降低最后一层dropout率
            nn.Linear(hidden_channels, out_channels)
        )
        
        # 残差连接的投影层
        if residual:
            self.res_projs = nn.ModuleList()
            in_dim = in_channels
            for i in range(num_layers):
                out_dim = hidden_channels if i == num_layers - 1 else hidden_channels * heads
                # 使用正交初始化
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
        
        # 增强型自注意力模块
        self.attention_enhancement = MultiHeadSelfAttention(hidden_channels, num_heads=heads, dropout=dropout * 0.7)
        
        # 动态特征缩放层 - 使用ModuleDict以适应不同的特征维度
        self.feature_gates = nn.ModuleDict({
            'hidden': nn.Sequential(
                nn.Linear(hidden_channels, hidden_channels),
                nn.Sigmoid()
            ),
            'hidden_heads': nn.Sequential(
                nn.Linear(hidden_channels * heads, hidden_channels * heads),
                nn.Sigmoid()
            )
        })
        
        # 初始化权重
        self._init_weights()
        
    def _init_weights(self):
        """初始化模型权重以获得更好的训练效果"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
    
    def _apply_feature_gate(self, x):
        """动态应用特征门控，适应不同的特征维度"""
        # 获取特征维度
        feat_dim = x.size(-1)
        
        # 根据特征维度选择合适的特征门控
        if feat_dim == self.hidden_channels:
            return x * self.feature_gates['hidden'](x)
        elif feat_dim == self.hidden_channels * self.heads:
            return x * self.feature_gates['hidden_heads'](x)
        else:
            # 如果维度不匹配，创建临时特征门控层
            temp_gate = nn.Sequential(
                nn.Linear(feat_dim, feat_dim),
                nn.Sigmoid()
            ).to(x.device)
            # 初始化权重
            for m in temp_gate.modules():
                if isinstance(m, nn.Linear):
                    nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)
            
            # 应用门控并返回
            return x * temp_gate(x)
                
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
        
        # 打印输入数据的形状，帮助调试
        # print(f"GAT输入形状: x={x.shape}, edge_index={edge_index.shape}")
        # if edge_weight is not None:
        #     print(f"edge_weight形状: {edge_weight.shape}")
        
        # 输入特征归一化
        x = self.input_norm(x)
        
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
            
            # 打印每层卷积后的特征形状
            # print(f"第{i+1}层GAT卷积后的特征形状: {x.shape}")
            
            # 混合使用批归一化和层归一化
            x = self.batch_norms[i](x)
            x = self.layer_norms[i](x)
            
            # 添加残差连接 - 改进的缩放残差机制
            if self.residual and res is not None:
                if x.size() == res.size():
                    x = x + res
                else:
                    # 使用自适应调整
                    try:
                        x = x + F.adaptive_avg_pool1d(res.unsqueeze(1), x.size(-1)).squeeze(1)
                    except:
                        # 如果自适应池化失败，使用线性投影
                        print(f"自适应池化失败，尝试线性投影。res形状: {res.shape}, x形状: {x.shape}")
                        proj = nn.Linear(res.size(-1), x.size(-1)).to(x.device)
                        nn.init.xavier_normal_(proj.weight)
                        x = x + proj(res)
            
            # 非线性激活 - 使用GELU替代LeakyReLU提高性能
            x = F.gelu(x)
            
            # 逐步减少Dropout率，使后面层的特征更稳定
            effective_dropout = self.dropout_rate * (1.0 - 0.1 * i)
            x = F.dropout(x, p=effective_dropout, training=self.training)
            
            # 应用动态特征门控机制，安全地选择重要特征
            if i < self.num_layers - 1:  # 在非最后一层应用
                try:
                    x = self._apply_feature_gate(x)
                except Exception as e:
                    print(f"应用特征门控失败: {e}, 形状: {x.shape}")
                    # 出错时跳过特征门控
                    pass
            
            # 存储层输出
            layer_outputs.append(x)
        
        # 改进的跳跃连接机制
        if len(layer_outputs) > 1:
            try:
                # 检查是否需要维度调整
                first_shape = layer_outputs[0].shape
                last_shape = layer_outputs[-1].shape
                
                # 打印各层输出形状以便调试
                # for i, lo in enumerate(layer_outputs):
                    # print(f"第{i+1}层输出形状: {lo.shape}")
                
                # 检查维度不匹配
                if first_shape[-1] != last_shape[-1]:
                    # 需要维度调整
                    if len(self.dim_projections) > 0:
                        normalized_outputs = []
                        for i, layer_output in enumerate(layer_outputs):
                            if i < len(layer_outputs) - 1 and i < len(self.dim_projections):  
                                # 对除最后一层外的所有层应用投影
                                normalized_outputs.append(self.dim_projections[i](layer_output))
                            else:
                                normalized_outputs.append(layer_output)
                        
                        # 改进的跳跃连接 - 使用cat然后通过线性层降维
                        x = self.jk(normalized_outputs)
                    else:
                        # 如果没有预定义的投影层，创建临时投影
                        print("创建临时投影层进行维度调整")
                        normalized_outputs = []
                        out_dim = last_shape[-1]
                        
                        for i, layer_output in enumerate(layer_outputs):
                            in_dim = layer_output.size(-1)
                            if in_dim != out_dim:
                                proj = nn.Linear(in_dim, out_dim).to(layer_output.device)
                                nn.init.xavier_normal_(proj.weight)
                                normalized_outputs.append(proj(layer_output))
                            else:
                                normalized_outputs.append(layer_output)
                        
                        x = self.jk(normalized_outputs)
                else:
                    # 维度已经一致
                    x = self.jk(layer_outputs)
                
                # 应用JK线性层
                x = self.jk_linear(x)
            except Exception as e:
                print(f"跳跃连接处理失败: {e}")
                # 出错时使用最后一层输出
                x = layer_outputs[-1]
        else:
            x = layer_outputs[0]
        
        # 使用增强型自注意力机制
        try:
            x = self.attention_enhancement(x)
        except Exception as e:
            print(f"自注意力增强失败: {e}, 形状: {x.shape}")
            # 出错时跳过自注意力
            pass
        
        # 应用全局注意力池化（如果需要）
        batch = data.batch if hasattr(data, 'batch') else None
        if batch is not None:
            try:
                x = self.glob_attn(x, batch)
            except Exception as e:
                print(f"全局注意力池化失败: {e}, 使用平均池化")
                # 使用平均池化作为备选
                from torch_geometric.nn import global_mean_pool
                x = global_mean_pool(x, batch)
        
        # 使用改进的MLP分类器
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
        
        # 输入特征归一化
        x = self.input_norm(x)
        
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
            
            # 混合使用批归一化和层归一化
            x = self.batch_norms[i](x)
            x = self.layer_norms[i](x)
            
            # 添加残差连接 - 改进的缩放残差机制
            if self.residual and res is not None:
                if x.size() == res.size():
                    x = x + res
                else:
                    # 使用自适应调整
                    try:
                        x = x + F.adaptive_avg_pool1d(res.unsqueeze(1), x.size(-1)).squeeze(1)
                    except:
                        # 如果自适应池化失败，使用线性投影
                        proj = nn.Linear(res.size(-1), x.size(-1)).to(x.device)
                        nn.init.xavier_normal_(proj.weight)
                        x = x + proj(res)
            
            # 非线性激活
            x = F.gelu(x)
            
            # 存储层输出
            layer_outputs.append(x)
        
        # 应用跳跃连接
        if len(layer_outputs) > 1:
            try:
                # 检查是否需要维度调整
                first_shape = layer_outputs[0].shape
                last_shape = layer_outputs[-1].shape
                
                # 检查维度不匹配
                if first_shape[-1] != last_shape[-1]:
                    # 需要维度调整
                    if len(self.dim_projections) > 0:
                        normalized_outputs = []
                        for i, layer_output in enumerate(layer_outputs):
                            if i < len(layer_outputs) - 1 and i < len(self.dim_projections):
                                normalized_outputs.append(self.dim_projections[i](layer_output))
                            else:
                                normalized_outputs.append(layer_output)
                        
                        # 改进的跳跃连接 - 使用cat然后通过线性层降维
                        embeddings = self.jk(normalized_outputs)
                    else:
                        # 如果没有预定义的投影层，创建临时投影
                        normalized_outputs = []
                        out_dim = last_shape[-1]
                        
                        for i, layer_output in enumerate(layer_outputs):
                            in_dim = layer_output.size(-1)
                            if in_dim != out_dim:
                                proj = nn.Linear(in_dim, out_dim).to(layer_output.device)
                                nn.init.xavier_normal_(proj.weight)
                                normalized_outputs.append(proj(layer_output))
                            else:
                                normalized_outputs.append(layer_output)
                        
                        embeddings = self.jk(normalized_outputs)
                else:
                    # 维度已经一致
                    embeddings = self.jk(layer_outputs)
                
                # 应用JK线性层
                embeddings = self.jk_linear(embeddings)
            except Exception as e:
                # 出错时使用最后一层输出
                embeddings = layer_outputs[-1]
        else:
            embeddings = layer_outputs[0]
        
        # 应用自注意力增强
        try:
            embeddings = self.attention_enhancement(embeddings)
        except Exception as e:
            # 出错时跳过自注意力
            pass
        
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