import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool, BatchNorm

class GCN(torch.nn.Module):
    def __init__(self, num_features, hidden_dim, num_classes, dropout=0.4, use_batch_norm=False):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(num_features, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.conv3 = GCNConv(hidden_dim, hidden_dim)
        self.dropout = torch.nn.Dropout(dropout)
        self.classifier = torch.nn.Linear(hidden_dim, num_classes)
        self.batch_norm = torch.nn.ModuleList(
            [BatchNorm(hidden_dim) for _ in range(3)]
        ) if use_batch_norm else None

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch

        x = self.conv1(x, edge_index)
        if self.batch_norm:
            x = self.batch_norm[0](x)
        x = F.relu(x)
        x = self.dropout(x)

        x = self.conv2(x, edge_index)
        if self.batch_norm:
            x = self.batch_norm[1](x)
        x = F.relu(x)
        x = self.dropout(x)

        x = self.conv3(x, edge_index)
        if self.batch_norm:
            x = self.batch_norm[2](x)
        x = F.relu(x)
        x = self.dropout(x)

        
        x = global_mean_pool(x, batch)

        x = self.classifier(x)

        return F.log_softmax(x, dim=1)