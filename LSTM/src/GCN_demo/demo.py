import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool
from torch_geometric.data import Data, DataLoader
import numpy as np

graph_nums = 100
node_nums = 53 # the number of neures
node_feature_dim = 3 
num_classes = 5


# 1. processing the data

# generate graph data

def generate_graph(node_nums, feature_dim, *args, **kwargs):
    """
    Generate a graph data according to the nodes and edges
    Args:
        node_nums (_type_): _description_
        feature_dim (_type_): _description_

    Returns:
        _type_: _description_
    """
    
    # demo here
    x = torch.randn(node_nums, feature_dim)
    edge_index = torch.randint(0, node_nums, (2, node_nums))
    
    return x, edge_index

# generate dataset
data_list = []
for i in range(graph_nums):
    x, edge_index = generate_graph(node_nums, node_feature_dim, edge_prob=0,1)
    y = torch.tensor(np.random.randint(0, num_classes, dtype=torch.long))
    data = Data(x=x, edge_index=edge_index)
    data_list.append(data)
    
# divide the dataset into train and test
# need to modify to make it more perfect
split_idx = int(graph_nums * 0.8)
train_data = data_list[:split_idx]
test_data = data_list[split_idx:]

train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
test_loader = DataLoader(test_data, batch_size=32, shuffle=False)

# 2. GCN model

class GCN(torch.nn.Module):
    def __init__(self, feature_nums, hidden_dim, num_classes):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(feature_nums, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.conv3 = GCNConv(hidden_dim, hidden_dim)
        self.fc = nn.Linear(hidden_dim, num_classes)
        
    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        # the first layer
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        # the second layer
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        # the third layer
        x = self.conv3(x, edge_index)
        x = F.relu(x)
        
        x = global_mean_pool(x, batch)
        # linear classifier
        x = self.fc(x)
        return F.log_softmax(x, dim=1)
    
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = GCN(node_feature_dim, 64, num_classes).to(device)
optimizer = optim.Adam(model.parameters(), lr=0.01)
criterion = nn.NLLLoss() # negative log likelihood loss

# 3. training
epochs = 200
for epoch in range(epochs):
    model.train()
    total_loss = 0
    for data in train_loader:
        data = data.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, data.y) # the label is stored in data.y
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * data.num_graphs
    total_loss /= len(train_loader.dataset)
    print('Epoch: {}, Loss: {:.4f}'.format(epoch, total_loss))
    
# 4. testing and evaluation
model.eval()
correct = 0
total = 0
with torch.no_grad():
    for data in test_loader:
        data = data.to(device)
        output = model(data)
        pred = output.max(dim=1)[1]
        correct += pred.eq(data.y).sum().item()
        total += data.num_graphs
print('Accuracy: {:.4f}'.format(correct / total))