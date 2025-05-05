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
        
        # BatchNorm layer
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
        
        self.dropout = torch.nn.Dropout(dropout)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        
        # GCN
        x1 = self.conv1(x, edge_index)
        x1 = self.bn1(x1)
        x1 = F.relu(x1)
        x1 = self.dropout(x1)
        
        # GraphSAGE
        x2 = self.conv2(x1, edge_index)
        x2 = self.bn2(x2)
        x2 = F.relu(x2)
        x2 = self.dropout(x2)
        
        # GAT
        x3 = self.conv3(x2, edge_index)
        x3 = self.bn3(x3)
        x3 = F.relu(x3)
        
        # pooling
        x_mean = global_mean_pool(x3, batch)
        x_sum = global_add_pool(x3, batch)
        
        x_combined = torch.cat([x_mean, x_sum], dim=1)
        
        out = self.mlp(x_combined)
        
        return F.log_softmax(out, dim=1)