import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import (
    GCNConv, SAGEConv, GATConv, GINConv, ChebConv, EdgeConv,
    global_mean_pool, global_add_pool, global_max_pool, global_sort_pool,
    BatchNorm, TopKPooling
)

# ============================================================================
# 原有的基础模型
# ============================================================================

class ImprovedGCN(torch.nn.Module):
    """混合模型：GCN + GraphSAGE + GAT"""
    def __init__(self, num_features, hidden_dim, num_classes, dropout=0.3):
        super(ImprovedGCN, self).__init__()
        
        self.conv1 = GCNConv(num_features, hidden_dim)
        self.conv2 = SAGEConv(hidden_dim, hidden_dim)
        self.conv3 = GATConv(hidden_dim, hidden_dim, heads=2, concat=False)
        
        self.bn1 = BatchNorm(hidden_dim)
        self.bn2 = BatchNorm(hidden_dim)
        self.bn3 = BatchNorm(hidden_dim)
        
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
        
        x1 = self.conv1(x, edge_index)
        x1 = self.bn1(x1)
        x1 = F.relu(x1)
        x1 = self.dropout(x1)
        
        x2 = self.conv2(x1, edge_index)
        x2 = self.bn2(x2)
        x2 = F.relu(x2)
        x2 = self.dropout(x2)
        
        x3 = self.conv3(x2, edge_index)
        x3 = self.bn3(x3)
        x3 = F.relu(x3)
        
        x_mean = global_mean_pool(x3, batch)
        x_sum = global_add_pool(x3, batch)
        x_combined = torch.cat([x_mean, x_sum], dim=1)
        
        out = self.mlp(x_combined)
        return F.log_softmax(out, dim=1)
    

class PureGCN(torch.nn.Module):
    """纯GCN模型"""
    def __init__(self, num_features, hidden_dim, num_classes, dropout=0.3):
        super(PureGCN, self).__init__()

        self.conv1 = GCNConv(num_features, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.conv3 = GCNConv(hidden_dim, hidden_dim)

        self.bn1 = BatchNorm(hidden_dim)
        self.bn2 = BatchNorm(hidden_dim)
        self.bn3 = BatchNorm(hidden_dim)

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
        
        x = self.conv1(x, edge_index)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.dropout(x)
        
        x = self.conv2(x, edge_index)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.dropout(x)
        
        x = self.conv3(x, edge_index)
        x = self.bn3(x)
        x = F.relu(x)
        
        x_mean = global_mean_pool(x, batch)
        x_sum = global_add_pool(x, batch)
        x_combined = torch.cat([x_mean, x_sum], dim=1)
        
        out = self.mlp(x_combined)
        return F.log_softmax(out, dim=1)
    

class PureGAT(torch.nn.Module):
    """纯GAT模型"""
    def __init__(self, num_features, hidden_dim, num_classes, dropout=0.3):
        super(PureGAT, self).__init__()

        self.conv1 = GATConv(num_features, hidden_dim, heads=2, concat=False)
        self.conv2 = GATConv(hidden_dim, hidden_dim, heads=2, concat=False)
        self.conv3 = GATConv(hidden_dim, hidden_dim, heads=2, concat=False)

        self.bn1 = BatchNorm(hidden_dim)
        self.bn2 = BatchNorm(hidden_dim)
        self.bn3 = BatchNorm(hidden_dim)

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
        
        x = self.conv1(x, edge_index)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.dropout(x)
        
        x = self.conv2(x, edge_index)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.dropout(x)
        
        x = self.conv3(x, edge_index)
        x = self.bn3(x)
        x = F.relu(x)
        
        x_mean = global_mean_pool(x, batch)
        x_sum = global_add_pool(x, batch)
        x_combined = torch.cat([x_mean, x_sum], dim=1)
        
        out = self.mlp(x_combined)
        return F.log_softmax(out, dim=1)


class PureGraphSAGE(torch.nn.Module):
    """纯GraphSAGE模型"""
    def __init__(self, num_features, hidden_dim, num_classes, dropout=0.3):
        super(PureGraphSAGE, self).__init__()

        self.conv1 = SAGEConv(num_features, hidden_dim)
        self.conv2 = SAGEConv(hidden_dim, hidden_dim)
        self.conv3 = SAGEConv(hidden_dim, hidden_dim)

        self.bn1 = BatchNorm(hidden_dim)
        self.bn2 = BatchNorm(hidden_dim)
        self.bn3 = BatchNorm(hidden_dim)

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
        
        x = self.conv1(x, edge_index)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.dropout(x)
        
        x = self.conv2(x, edge_index)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.dropout(x)
        
        x = self.conv3(x, edge_index)
        x = self.bn3(x)
        x = F.relu(x)
        
        x_mean = global_mean_pool(x, batch)
        x_sum = global_add_pool(x, batch)
        x_combined = torch.cat([x_mean, x_sum], dim=1)
        
        out = self.mlp(x_combined)
        return F.log_softmax(out, dim=1)


# ============================================================================
# 近年来的先进GNN模型
# ============================================================================

class GIN(torch.nn.Module):
    """
    Graph Isomorphism Network (GIN)
    
    论文: How Powerful are Graph Neural Networks? (ICLR 2019)
    作者: Xu et al.
    链接: https://arxiv.org/abs/1810.00826
    
    特点:
    - 理论上与WL测试等价，具有最强的图结构区分能力
    - 使用MLP作为聚合函数
    - 在图分类任务上表现优异
    - 适合需要精确区分图结构的任务
    """
    def __init__(self, num_features, hidden_dim, num_classes, dropout=0.5, num_layers=3):
        super(GIN, self).__init__()
        
        self.num_layers = num_layers
        self.dropout = dropout
        
        # GIN卷积层
        self.convs = nn.ModuleList()
        self.batch_norms = nn.ModuleList()
        
        # 第一层
        nn1 = nn.Sequential(
            nn.Linear(num_features, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        self.convs.append(GINConv(nn1))
        self.batch_norms.append(BatchNorm(hidden_dim))
        
        # 中间层
        for _ in range(num_layers - 1):
            nn_i = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim)
            )
            self.convs.append(GINConv(nn_i))
            self.batch_norms.append(BatchNorm(hidden_dim))
        
        # 分类器
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes)
        )
    
    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        
        # GIN层
        for i in range(self.num_layers):
            x = self.convs[i](x, edge_index)
            x = self.batch_norms[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        
        # 全局池化
        x_mean = global_mean_pool(x, batch)
        x_sum = global_add_pool(x, batch)
        x = torch.cat([x_mean, x_sum], dim=1)
        
        # 分类
        x = self.mlp(x)
        return F.log_softmax(x, dim=1)


class ChebNet(torch.nn.Module):
    """
    Chebyshev Graph Convolutional Network
    
    论文: Convolutional Neural Networks on Graphs with Fast Localized Spectral Filtering (NIPS 2016)
    作者: Defferrard et al.
    链接: https://arxiv.org/abs/1606.09375
    
    特点:
    - 使用Chebyshev多项式近似图卷积
    - 计算效率高，O(K|E|) 复杂度
    - 适合大规模图
    - 基于谱图理论
    """
    def __init__(self, num_features, hidden_dim, num_classes, dropout=0.5, K=3):
        super(ChebNet, self).__init__()
        
        # Chebyshev卷积层（K是多项式阶数）
        self.conv1 = ChebConv(num_features, hidden_dim, K=K)
        self.conv2 = ChebConv(hidden_dim, hidden_dim, K=K)
        self.conv3 = ChebConv(hidden_dim, hidden_dim, K=K)
        
        # BatchNorm
        self.bn1 = BatchNorm(hidden_dim)
        self.bn2 = BatchNorm(hidden_dim)
        self.bn3 = BatchNorm(hidden_dim)
        
        # 分类器
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes)
        )
        
        self.dropout = dropout
    
    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        
        # Chebyshev卷积层
        x = self.conv1(x, edge_index)
        x = self.bn1(x)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        
        x = self.conv2(x, edge_index)
        x = self.bn2(x)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        
        x = self.conv3(x, edge_index)
        x = self.bn3(x)
        x = F.relu(x)
        
        # 全局池化
        x_mean = global_mean_pool(x, batch)
        x_sum = global_add_pool(x, batch)
        x = torch.cat([x_mean, x_sum], dim=1)
        
        # 分类
        x = self.mlp(x)
        return F.log_softmax(x, dim=1)


class EdgeConvNet(torch.nn.Module):
    """
    EdgeConv Network (Dynamic Graph CNN)
    
    论文: Dynamic Graph CNN for Learning on Point Clouds (TOG 2019)
    作者: Wang et al.
    链接: https://arxiv.org/abs/1801.07829
    
    特点:
    - 动态更新图结构
    - 边卷积操作，同时考虑节点和边的信息
    - 多尺度特征融合
    - 在点云和图数据上表现优异
    """
    def __init__(self, num_features, hidden_dim, num_classes, dropout=0.5):
        super(EdgeConvNet, self).__init__()
        
        # EdgeConv层
        self.conv1 = EdgeConv(
            nn.Sequential(
                nn.Linear(2 * num_features, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim)
            )
        )
        
        self.conv2 = EdgeConv(
            nn.Sequential(
                nn.Linear(2 * hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim)
            )
        )
        
        self.conv3 = EdgeConv(
            nn.Sequential(
                nn.Linear(2 * hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim)
            )
        )
        
        # BatchNorm
        self.bn1 = BatchNorm(hidden_dim)
        self.bn2 = BatchNorm(hidden_dim)
        self.bn3 = BatchNorm(hidden_dim)
        
        # 分类器
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim * 2),
            nn.BatchNorm1d(hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes)
        )
        
        self.dropout = dropout
    
    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        
        # EdgeConv层
        x1 = self.conv1(x, edge_index)
        x1 = self.bn1(x1)
        x1 = F.relu(x1)
        
        x2 = self.conv2(x1, edge_index)
        x2 = self.bn2(x2)
        x2 = F.relu(x2)
        
        x3 = self.conv3(x2, edge_index)
        x3 = self.bn3(x3)
        x3 = F.relu(x3)
        
        # 多尺度特征融合
        x1_pooled = global_mean_pool(x1, batch)
        x2_pooled = global_mean_pool(x2, batch)
        x3_pooled = global_mean_pool(x3, batch)
        
        x = torch.cat([x1_pooled, x2_pooled, x3_pooled], dim=1)
        
        # 分类
        x = self.mlp(x)
        return F.log_softmax(x, dim=1)


class GraphUNet(torch.nn.Module):
    """
    Graph U-Net
    
    论文: Graph U-Nets (ICML 2019)
    作者: Gao et al.
    链接: https://arxiv.org/abs/1905.05178
    
    特点:
    - U-Net架构应用于图
    - 使用TopK池化进行下采样
    - 编码器-解码器结构，保留多尺度信息
    - 适合需要多尺度特征的任务
    """
    def __init__(self, num_features, hidden_dim, num_classes, dropout=0.5, pool_ratio=0.5):
        super(GraphUNet, self).__init__()
        
        self.pool_ratio = pool_ratio
        
        # 编码器
        self.conv1 = GCNConv(num_features, hidden_dim)
        self.pool1 = TopKPooling(hidden_dim, ratio=pool_ratio)
        
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.pool2 = TopKPooling(hidden_dim, ratio=pool_ratio)
        
        self.conv3 = GCNConv(hidden_dim, hidden_dim)
        
        # BatchNorm
        self.bn1 = BatchNorm(hidden_dim)
        self.bn2 = BatchNorm(hidden_dim)
        self.bn3 = BatchNorm(hidden_dim)
        
        # 分类器
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes)
        )
        
        self.dropout = dropout
    
    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        
        # 编码器路径
        x = self.conv1(x, edge_index)
        x = self.bn1(x)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x, edge_index, _, batch, _, _ = self.pool1(x, edge_index, None, batch)
        
        x = self.conv2(x, edge_index)
        x = self.bn2(x)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x, edge_index, _, batch, _, _ = self.pool2(x, edge_index, None, batch)
        
        x = self.conv3(x, edge_index)
        x = self.bn3(x)
        x = F.relu(x)
        
        # 全局池化
        x_mean = global_mean_pool(x, batch)
        x_max = global_max_pool(x, batch)
        x = torch.cat([x_mean, x_max], dim=1)
        
        # 分类
        x = self.mlp(x)
        return F.log_softmax(x, dim=1)


# ============================================================================
# 2020年后的最新先进GNN模型
# ============================================================================

class PNA(torch.nn.Module):
    """
    Principal Neighbourhood Aggregation (PNA)
    
    论文: Principal Neighbourhood Aggregation for Graph Nets (NeurIPS 2020)
    作者: Corso et al.
    链接: https://arxiv.org/abs/2004.05718
    
    特点:
    - 多种聚合器组合（mean, max, sum, std）
    - 度数缩放器（Degree Scaler）
    - 在OGB等多个基准上达到SOTA
    - 表达能力强，泛化性好
    """
    def __init__(self, num_features, hidden_dim, num_classes, dropout=0.5):
        super(PNA, self).__init__()
        
        # PNA使用多种聚合方式，这里简化实现
        # 实际PNA需要PNAConv，这里用多路径模拟
        self.conv1 = GCNConv(num_features, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.conv3 = GCNConv(hidden_dim, hidden_dim)
        
        # SAGE路径（不同聚合）
        self.sage1 = SAGEConv(num_features, hidden_dim)
        self.sage2 = SAGEConv(hidden_dim, hidden_dim)
        
        # BatchNorm
        self.bn1 = BatchNorm(hidden_dim)
        self.bn2 = BatchNorm(hidden_dim)
        self.bn3 = BatchNorm(hidden_dim)
        
        # 分类器
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim * 4, hidden_dim * 2),
            nn.BatchNorm1d(hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes)
        )
        
        self.dropout = dropout
    
    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        
        # GCN路径
        x1 = self.conv1(x, edge_index)
        x1 = self.bn1(x1)
        x1 = F.relu(x1)
        x1 = F.dropout(x1, p=self.dropout, training=self.training)
        
        x1 = self.conv2(x1, edge_index)
        x1 = F.relu(x1)
        
        # SAGE路径
        x2 = self.sage1(x, edge_index)
        x2 = F.relu(x2)
        x2 = F.dropout(x2, p=self.dropout, training=self.training)
        
        x2 = self.sage2(x2, edge_index)
        x2 = F.relu(x2)
        
        # 多种池化聚合
        x1_mean = global_mean_pool(x1, batch)
        x1_max = global_max_pool(x1, batch)
        x2_mean = global_mean_pool(x2, batch)
        x2_sum = global_add_pool(x2, batch)
        
        x = torch.cat([x1_mean, x1_max, x2_mean, x2_sum], dim=1)
        
        # 分类
        x = self.mlp(x)
        return F.log_softmax(x, dim=1)


class GATv2(torch.nn.Module):
    """
    Graph Attention Networks v2 (GATv2)
    
    论文: How Attentive are Graph Attention Networks? (ICLR 2022)
    作者: Brody et al.
    链接: https://arxiv.org/abs/2105.14491
    
    特点:
    - 改进了原始GAT的注意力计算方式
    - 动态注意力机制，更强的表达能力
    - 解决了GAT的静态注意力问题
    - 在节点分类和图分类上都有提升
    """
    def __init__(self, num_features, hidden_dim, num_classes, dropout=0.5, heads=4):
        super(GATv2, self).__init__()
        
        from torch_geometric.nn import GATv2Conv
        
        # GATv2卷积层
        self.conv1 = GATv2Conv(num_features, hidden_dim, heads=heads, concat=True, dropout=dropout)
        self.conv2 = GATv2Conv(hidden_dim * heads, hidden_dim, heads=heads, concat=True, dropout=dropout)
        self.conv3 = GATv2Conv(hidden_dim * heads, hidden_dim, heads=1, concat=False, dropout=dropout)
        
        # BatchNorm
        self.bn1 = BatchNorm(hidden_dim * heads)
        self.bn2 = BatchNorm(hidden_dim * heads)
        self.bn3 = BatchNorm(hidden_dim)
        
        # 分类器
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes)
        )
        
        self.dropout_p = dropout
    
    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        
        # GATv2层
        x = F.dropout(x, p=self.dropout_p, training=self.training)
        x = self.conv1(x, edge_index)
        x = self.bn1(x)
        x = F.elu(x)
        
        x = F.dropout(x, p=self.dropout_p, training=self.training)
        x = self.conv2(x, edge_index)
        x = self.bn2(x)
        x = F.elu(x)
        
        x = F.dropout(x, p=self.dropout_p, training=self.training)
        x = self.conv3(x, edge_index)
        x = self.bn3(x)
        x = F.elu(x)
        
        # 全局池化
        x_mean = global_mean_pool(x, batch)
        x_max = global_max_pool(x, batch)
        x = torch.cat([x_mean, x_max], dim=1)
        
        # 分类
        x = self.mlp(x)
        return F.log_softmax(x, dim=1)


class DeeperGCN(torch.nn.Module):
    """
    DeeperGCN
    
    论文: DeeperGCN: All You Need to Train Deeper GCNs (ICLR 2020)
    作者: Li et al.
    链接: https://arxiv.org/abs/2006.07739
    
    特点:
    - 可以训练非常深的GCN（14-28层）
    - 使用残差连接和层归一化防止过平滑
    - 在OGB大规模图上表现优异
    - 适合需要大感受野的任务
    """
    def __init__(self, num_features, hidden_dim, num_classes, dropout=0.5, num_layers=7):
        super(DeeperGCN, self).__init__()
        
        self.num_layers = num_layers
        
        # 输入投影
        self.node_encoder = nn.Linear(num_features, hidden_dim)
        
        # 深层GCN（带残差连接）
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        
        for _ in range(num_layers):
            self.convs.append(GCNConv(hidden_dim, hidden_dim))
            self.norms.append(nn.LayerNorm(hidden_dim))
        
        # 分类器
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes)
        )
        
        self.dropout = dropout
    
    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        
        # 输入编码
        x = self.node_encoder(x)
        x = F.relu(x)
        
        # 深层GCN with residual connections
        for i in range(self.num_layers):
            x_res = x  # 保存残差
            
            x = self.convs[i](x, edge_index)
            x = self.norms[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
            
            # 残差连接（解决过平滑问题）
            x = x + x_res
        
        # 全局池化
        x_mean = global_mean_pool(x, batch)
        x_max = global_max_pool(x, batch)
        x = torch.cat([x_mean, x_max], dim=1)
        
        # 分类
        x = self.mlp(x)
        return F.log_softmax(x, dim=1)

