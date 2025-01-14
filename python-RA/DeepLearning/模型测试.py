import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool
from torch_geometric.data import Data

# ================== 1. 定义与训练时相同的网络结构 ==================
class GCN(torch.nn.Module):
    def __init__(self, num_features, hidden_dim, num_classes):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(num_features, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.fc = torch.nn.Linear(hidden_dim, num_classes)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        # 使用全局平均池化，将同一图的节点合并成一个图级向量
        x = global_mean_pool(x, batch)  # shape: [batch_size, hidden_dim]
        out = self.fc(x)                # shape: [batch_size, num_classes]
        return F.log_softmax(out, dim=1)

# ================== 2. 加载模型并进行推理的函数 ==================
def load_model_and_infer(model_path, data_list,
                         num_features=2, hidden_dim=16, num_classes=10):
    """
    :param model_path: 训练时保存的模型权重文件，如 'gnn_model_weights.pt'
    :param data_list:  待预测的 PyG Data 对象列表（可以单个，也可以多个）
    :param num_features: 同训练时一致的输入特征维度
    :param hidden_dim:   同训练时一致的 GCN 隐藏层大小
    :param num_classes:  分类类别数
    :return: 对每个图的预测结果列表
    """
    # 初始化模型(结构要和训练时相同)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = GCN(num_features, hidden_dim, num_classes).to(device)

    # 加载已保存的权重
    state_dict = torch.load(r"C:\Users\PAN\PycharmProjects\GitHub\python-RA\DeepLearning\gnn_model_weights.pt", map_location=device)
    model.load_state_dict(state_dict)
    model.eval()

    # 对每个待预测的图进行推理
    preds = []
    with torch.no_grad():
        for data in data_list:
            data = data.to(device)
            out = model(data)       # shape: [1, num_classes] (单图) 或 [batch_size, num_classes]
            _, pred = torch.max(out, dim=1)
            preds.append(pred.cpu().item() if pred.numel() == 1 else pred.cpu().tolist())
    return preds


# ================== 3. 示例：如何使用上面的函数 ==================
if __name__ == "__main__":
    # 假设你已经有一个或多个 PyG 的 Data 对象，以下仅示例
    # 如果你要预测多个图，就把它们放到一个列表中
    # 下面只是一个演示的 Data，实际请根据你的节点特征/边信息来构造
    demo_x = torch.tensor([[1.2, 0.3], [0.4, -0.1]], dtype=torch.float)  # 2个节点，每节点2维特征
    demo_edge_index = torch.tensor([[0, 1], [1, 0]], dtype=torch.long)   # 简单的双向边
    demo_data = Data(x=demo_x, edge_index=demo_edge_index)
    demo_data.num_nodes = 2  # 一般PyG会自动识别，这里只是显式指定
    # 批信息 batch，如果你只有一个图，可以在构造 Data 时给 data.batch=0
    # 或在 forward 里处理单图逻辑(不用 batch)
    demo_data.batch = torch.zeros(demo_data.num_nodes, dtype=torch.long)  # 所有节点属于同一图

    # 准备一个列表，里面有一到多个 Data
    test_dataset = [demo_data]

    # 调用推理函数
    # 注意：num_features=2, num_classes=10 等需与训练时一致
    model_path = "gnn_model_weights.pt"
    predictions = load_model_and_infer(model_path, test_dataset,
                                       num_features=2, hidden_dim=16, num_classes=10)

    print("推理结果:", predictions)
    # 如果是多个图，就会返回多个预测。对于单图，这里是一个长度为1的列表，比如 [3]
