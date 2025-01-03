import pandas as pd
import torch
from torch_geometric.data import Data, Dataset, DataLoader
from sklearn.preprocessing import LabelEncoder
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
import torch.optim as optim
import os

# ----------------------------
# 1. 数据导入和预处理
# ----------------------------

# 定义文件路径
NODES_FILE = 'network_structures_for_GNN.xlsx'
EDGES_FILE = 'network_structures_for_GNN.xlsx'
LABELS_FILE = 'network_structures_for_GNN.xlsx'
BEHAVIOR_DATA_FILE = 'behavior_data.xlsx'  # 另一个行为数据的Excel文件

# 读取Excel文件
nodes_df = pd.read_excel(NODES_FILE, sheet_name='Nodes')
edges_df = pd.read_excel(EDGES_FILE, sheet_name='Edges')
labels_df = pd.read_excel(LABELS_FILE, sheet_name='Labels')
behavior_data_df = pd.read_excel(BEHAVIOR_DATA_FILE, sheet_name='Sheet1')  # 请根据实际Sheet名称修改

# 查看数据结构（可选）
print("Nodes DataFrame:")
print(nodes_df.head())
print("\nEdges DataFrame:")
print(edges_df.head())
print("\nLabels DataFrame:")
print(labels_df.head())
print("\nBehavior Data DataFrame:")
print(behavior_data_df.head())

# 标签映射（将行为标签转换为数值）
le = LabelEncoder()
labels_df['behavior_num'] = le.fit_transform(labels_df['behavior'])
label_mapping = {label: idx for idx, label in enumerate(le.classes_)}
print("\nLabel Mapping:", label_mapping)

# 确保行为数据和Labels数据对齐
# 假设behavior_data_df中的'time'与labels_df中的'time'对应
# 如果不对应，需要根据具体情况调整
# 这里假设behavior_data_df的第三列是行为标签，与labels_df相同

# 合并行为标签
# 假设behavior_data_df有 'time' 和 'behavior' 两列，且与labels_df对应
# 如果结构不同，请根据实际情况调整
merged_labels_df = labels_df.copy()
# 如果需要从behavior_data_df获取标签，可以执行如下操作
# merged_labels_df = pd.merge(labels_df, behavior_data_df[['time', 'behavior']], on='time', how='left')
# merged_labels_df['behavior_num'] = le.transform(merged_labels_df['behavior'])

# 获取唯一时间点
unique_times = nodes_df['time'].unique()
print(f"\nUnique time points: {unique_times}")

# 创建图列表
graph_list = []

for t in unique_times:
    # 筛选当前时间点的节点和边
    nodes_t = nodes_df[nodes_df['time'] == t]
    edges_t = edges_df[edges_df['time'] == t]
    label_t = merged_labels_df[merged_labels_df['time'] == t]['behavior_num'].values[0]

    # 创建节点特征
    # 使用 activity_value 和 state 作为节点特征
    # 将 state 转换为数值（ON=1, OFF=0）
    state_mapping = {'ON': 1, 'OFF': 0, None: 0}  # 根据实际数据调整
    state_numeric = nodes_t['state'].map(state_mapping).fillna(0).values
    activity_value = nodes_t['activity_value'].values
    # 组合多个特征
    x = torch.tensor(list(zip(activity_value, state_numeric)), dtype=torch.float)  # [num_nodes, 2]

    # 创建节点名称到索引的映射
    neuron_names = nodes_t['Neuron'].unique()
    node_mapping = {node: idx for idx, node in enumerate(neuron_names)}

    # 替换边中的节点名称为索引
    edges_t_mapped = edges_t[['source', 'target']].replace(node_mapping)
    edge_index = torch.tensor(edges_t_mapped.values, dtype=torch.long).t().contiguous()  # [2, num_edges]

    # 创建图标签
    y = torch.tensor([label_t], dtype=torch.long)

    # 创建 Data 对象
    data = Data(x=x, edge_index=edge_index, y=y)
    data.time = t  # 可选属性，便于后续分析
    graph_list.append(data)

print(f"\nNumber of graphs in dataset: {len(graph_list)}")


# 定义自定义数据集
class NeuronDataset(Dataset):
    def __init__(self, graphs):
        super(NeuronDataset, self).__init__()
        self.graphs = graphs

    def len(self):
        return len(self.graphs)

    def get(self, idx):
        return self.graphs[idx]


# 创建数据集和数据加载器
dataset = NeuronDataset(graph_list)
batch_size = 32
loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
print(f"Batch size: {batch_size}")


# ----------------------------
# 2. 定义图神经网络模型
# ----------------------------

class GCN(torch.nn.Module):
    def __init__(self, num_features, hidden_dim, num_classes):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(num_features, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.fc = torch.nn.Linear(hidden_dim, num_classes)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        # 第一层图卷积 + ReLU
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        # 第二层图卷积 + ReLU
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        # 全局平均池化
        x = torch.mean(x, dim=0).unsqueeze(0)
        # 全连接层
        out = self.fc(x)
        return F.log_softmax(out, dim=1)


# ----------------------------
# 3. 训练和评估
# ----------------------------

# 设置设备
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"\nUsing device: {device}")

# 初始化模型、优化器和损失函数
num_features = dataset[0].x.shape[1]  # 2
num_classes = len(label_mapping)
hidden_dim = 16

model = GCN(num_features=num_features, hidden_dim=hidden_dim, num_classes=num_classes).to(device)
optimizer = optim.Adam(model.parameters(), lr=0.01)
criterion = torch.nn.CrossEntropyLoss()


# 训练函数
def train(model, loader, optimizer, criterion, device, epochs=100):
    model.train()
    for epoch in range(1, epochs + 1):
        total_loss = 0
        for batch in loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            out = model(batch)
            loss = criterion(out, batch.y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        avg_loss = total_loss / len(loader)
        if epoch % 10 == 0 or epoch == 1:
            print(f"Epoch {epoch}, Loss: {avg_loss:.4f}")


# 评估函数
def evaluate(model, loader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            out = model(batch)
            _, pred = torch.max(out, dim=1)
            correct += (pred == batch.y).sum().item()
            total += batch.y.size(0)
    accuracy = correct / total
    print(f"Accuracy: {accuracy * 100:.2f}%")
    return accuracy


# 开始训练
epochs = 100
print("\n开始训练模型...")
train(model, loader, optimizer, criterion, device, epochs)

# 评估模型
print("\n评估模型性能...")
evaluate(model, loader, device)

# ----------------------------
# 4. 额外功能（可选）
# ----------------------------

# 如果你希望记录损失并绘制损失曲线，可以修改训练函数如下：

import matplotlib.pyplot as plt


def train_with_logging(model, loader, optimizer, criterion, device, epochs=100):
    model.train()
    loss_values = []
    for epoch in range(1, epochs + 1):
        total_loss = 0
        for batch in loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            out = model(batch)
            loss = criterion(out, batch.y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        avg_loss = total_loss / len(loader)
        loss_values.append(avg_loss)
        if epoch % 10 == 0 or epoch == 1:
            print(f"Epoch {epoch}, Loss: {avg_loss:.4f}")
    return loss_values


# 重新初始化模型和优化器
model = GCN(num_features=num_features, hidden_dim=hidden_dim, num_classes=num_classes).to(device)
optimizer = optim.Adam(model.parameters(), lr=0.01)
criterion = torch.nn.CrossEntropyLoss()

# 训练并记录损失
print("\n开始训练模型并记录损失...")
loss_values = train_with_logging(model, loader, optimizer, criterion, device, epochs)

# 绘制损失曲线
plt.figure(figsize=(10, 6))
plt.plot(range(1, epochs + 1), loss_values, label='Training Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Loss over Epochs')
plt.legend()
plt.show()

# ----------------------------
# 5. 完整代码总结
# ----------------------------

# 为了方便，你可以将以上所有代码保存为一个 Python 脚本（例如 `train_gnn.py`），然后在命令行中运行：
# python train_gnn.py

# 确保将 Excel 文件路径和 Sheet 名称根据你的实际数据进行调整。

