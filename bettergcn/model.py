import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, SAGEConv, GATConv, global_mean_pool, global_add_pool
from torch_geometric.nn import BatchNorm

class ImprovedGCN(torch.nn.Module):
    def __init__(self, num_features, hidden_dim, num_classes, dropout=0.3):
        super(ImprovedGCN, self).__init__()
        
        self.conv1 = GCNConv(num_features, hidden_dim)
        self.conv2 = SAGEConv(hidden_dim, hidden_dim)
        self.conv3 = GATConv(hidden_dim, hidden_dim, heads=2, concat=False)
        
        # BatchNorm层
        self.bn1 = BatchNorm(hidden_dim)
        self.bn2 = BatchNorm(hidden_dim)
        self.bn3 = BatchNorm(hidden_dim)
        
        # multiple layer perceptron (MLP) for classification
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(hidden_dim*2, hidden_dim),
            torch.nn.BatchNorm1d(hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(hidden_dim, num_classes)
        )
    
    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        
        # 处理边权重（如果存在）
        edge_weight = data.edge_attr if hasattr(data, 'edge_attr') else None
        
        # 检查批次大小，如果为1则临时切换BN层为eval模式
        batch_size = 1
        if batch is not None:
            unique_batches = torch.unique(batch)
            batch_size = len(unique_batches)
        
        # 批次大小为1时的特殊处理
        bn_training = self.training
        if batch_size == 1 and self.training:
            # 暂时将所有BN层设为评估模式
            self._set_bn_eval(True)
        
        # 第一层GCN
        x1 = self.conv1(x, edge_index, edge_weight)
        x1 = self.bn1(x1)
        x1 = F.relu(x1)
        x1 = F.dropout(x1, p=0.2, training=self.training)
        
        # 第二层SAGE
        x2 = self.conv2(x1, edge_index)
        x2 = self.bn2(x2)
        x2 = F.relu(x2)
        x2 = F.dropout(x2, p=0.2, training=self.training)
        
        # 第三层GAT
        x3 = self.conv3(x2, edge_index)
        x3 = self.bn3(x3)
        x3 = F.relu(x3)
        x3 = F.dropout(x3, p=0.2, training=self.training)
        
        # 跳跃连接
        x_combined = torch.cat([x1, x3], dim=1)
        
        # 全局池化
        if batch is None:
            # 如果没有batch信息，认为所有节点是单个图
            batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)
        
        # 全局池化 - 将节点特征池化为图特征
        x_pool = global_mean_pool(x_combined, batch)
        
        # 对池化结果应用MLP
        out = None
        try:
            # 检查池化后的批次大小
            if x_pool.size(0) == 1 and self.training:
                # 暂时切换到评估模式处理单批次
                training_state = {}
                for name, module in self.mlp.named_modules():
                    if isinstance(module, torch.nn.BatchNorm1d):
                        training_state[name] = module.training
                        module.eval()
                        
                out = self.mlp(x_pool)
                
                # 恢复原始训练状态
                for name, module in self.mlp.named_modules():
                    if isinstance(module, torch.nn.BatchNorm1d) and name in training_state:
                        if training_state[name]:
                            module.train()
            else:
                out = self.mlp(x_pool)
        except Exception as e:
            print(f"前向传播错误: {e}")
            print(f"池化输出形状: {x_pool.shape}, 批次大小: {batch_size}")
            raise e
            
        # 恢复批次大小为1时的BN层状态
        if batch_size == 1 and bn_training:
            self._set_bn_eval(False)
            
        return out

    def _set_bn_eval(self, is_eval):
        """设置所有BN层的评估/训练模式"""
        for module in self.modules():
            if isinstance(module, (torch.nn.BatchNorm1d, torch.nn.BatchNorm2d, BatchNorm)):
                if is_eval:
                    module.eval()
                else:
                    module.train()
    
    def get_embeddings(self, data):
        """
        获取节点嵌入向量，用于可视化和分析
        
        参数:
            data: PyTorch Geometric数据对象
            
        返回:
            embeddings: 节点嵌入向量
        """
        x, edge_index = data.x, data.edge_index
        batch = data.batch if hasattr(data, 'batch') else None
        
        # 处理边权重（如果存在）
        edge_weight = data.edge_attr if hasattr(data, 'edge_attr') else None
        
        # 只获取嵌入，不应用分类器
        # 第一层GCN
        x1 = self.conv1(x, edge_index, edge_weight)
        x1 = self.bn1(x1)
        x1 = F.relu(x1)
        
        # 第二层SAGE
        x2 = self.conv2(x1, edge_index)
        x2 = self.bn2(x2)
        x2 = F.relu(x2)
        
        # 第三层GAT
        x3 = self.conv3(x2, edge_index)
        x3 = self.bn3(x3)
        x3 = F.relu(x3)
        
        # 返回组合后的节点嵌入
        return torch.cat([x1, x3], dim=1)