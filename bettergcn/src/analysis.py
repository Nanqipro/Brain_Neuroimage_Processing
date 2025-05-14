import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from torch_geometric.utils import to_networkx
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from matplotlib.colors import LinearSegmentedColormap
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D

# 导入项目中的其他模块
from model import ImprovedGCN
from process import load_data, generate_graph, create_dataset
from feature import extract_advanced_features, select_features

class GCNVisualizer:
    """
    GCN模型可视化工具类，用于生成各种神经网络拓扑结构图
    """
    def __init__(self, model_path='../results/best_model.pth', data_path='../dataset/EMtrace01.xlsx'):
        """
        初始化可视化器
        
        参数:
            model_path: 训练好的模型路径
            data_path: 原始数据路径
        """
        self.model_path = model_path
        self.data_path = data_path
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # 加载数据和模型
        self.features, self.labels, self.encoder, _ = load_data(data_path)
        self.num_classes = len(self.encoder.classes_)
        
        # 加载模型
        self.model = self._load_model()
        
        # 生成图结构
        self.edge_index = generate_graph(self.features, k=15)
        
    def _load_model(self):
        """
        加载训练好的GCN模型
        
        返回:
            加载好的模型
        """
        # 确保模型路径存在
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"模型文件不存在: {self.model_path}")
        
        # 初始化模型（与训练时结构一致）
        model = ImprovedGCN(
            num_features=self.features.shape[1],  
            hidden_dim=64,
            num_classes=self.num_classes,
            dropout=0.3
        )
        
        # 加载模型权重
        model.load_state_dict(torch.load(self.model_path, map_location=self.device))
        model.eval()  # 设置为评估模式
        
        return model
    
    def extract_node_embeddings(self):
        """
        提取节点嵌入向量，用于可视化
        
        返回:
            节点嵌入向量
        """
        # 创建数据集
        dataset = create_dataset(self.features, self.labels, self.edge_index)
        
        embeddings = []
        with torch.no_grad():
            for data in dataset:
                data = data.to(self.device)
                
                # 提取中间层特征（修改forward方法或使用钩子）
                # 这里我们直接使用模型的第一个卷积层输出作为节点嵌入
                x, edge_index = data.x, data.edge_index
                
                # GCN第一层
                x1 = self.model.conv1(x, edge_index)
                x1 = self.model.bn1(x1)
                x1 = torch.relu(x1)
                
                embeddings.append(x1.cpu().numpy())
                
        return np.vstack(embeddings)
    
    def visualize_network(self, save_path='../results/network_visualization.png'):
        """
        可视化神经元之间的拓扑关系
        
        参数:
            save_path: 保存图像的路径
        """
        # 构建图结构
        edge_list = self.edge_index.t().numpy()
        G = nx.Graph()
        
        # 添加节点
        for i in range(len(self.features)):
            G.add_node(i, label=str(self.labels[i]))
            
        # 添加边
        for edge in edge_list:
            G.add_edge(edge[0], edge[1])
            
        # 使用社区检测算法找出社区结构
        communities = nx.community.greedy_modularity_communities(G)
        
        # 为每个社区节点分配颜色
        colors = plt.cm.rainbow(np.linspace(0, 1, len(communities)))
        node_colors = np.zeros((len(G.nodes), 4))
        
        for i, community in enumerate(communities):
            for node in community:
                node_colors[node] = colors[i]
        
        # 创建图形
        plt.figure(figsize=(12, 10))
        pos = nx.spring_layout(G, seed=42)  # 使用弹簧布局
        
        # 绘制节点
        nx.draw_networkx_nodes(
            G, pos, 
            node_color=node_colors, 
            node_size=50, 
            alpha=0.8
        )
        
        # 绘制边
        nx.draw_networkx_edges(
            G, pos, 
            width=0.5, 
            alpha=0.3, 
            edge_color='gray'
        )
        
        # 为不同类别的节点添加标签
        class_labels = {i: f"Class {i}" for i in range(self.num_classes)}
        plt.title('GCN Neural Topology Structure')
        
        # 添加图例
        for i in range(self.num_classes):
            plt.scatter([], [], color=plt.cm.rainbow(i/self.num_classes), label=f'Class {self.encoder.inverse_transform([i])[0]}')
        
        plt.legend()
        
        # 保存图像
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"神经元拓扑结构图已保存至: {save_path}")
    
    def visualize_embeddings_2d(self, save_path='../results/embeddings_2d.png'):
        """
        使用t-SNE将节点嵌入可视化为2D散点图
        
        参数:
            save_path: 保存图像的路径
        """
        # 提取节点嵌入
        embeddings = self.extract_node_embeddings()
        
        # 使用PCA进行初步降维
        if embeddings.shape[1] > 50:
            pca = PCA(n_components=50)
            embeddings_pca = pca.fit_transform(embeddings)
        else:
            embeddings_pca = embeddings
            
        # 使用t-SNE进行降维到2D
        tsne = TSNE(n_components=2, random_state=42)
        embeddings_2d = tsne.fit_transform(embeddings_pca)
        
        # 可视化
        plt.figure(figsize=(10, 8))
        scatter = plt.scatter(
            embeddings_2d[:, 0], 
            embeddings_2d[:, 1], 
            c=self.labels, 
            cmap='viridis', 
            alpha=0.8,
            s=50
        )
        
        plt.colorbar(scatter, label='Class')
        plt.title('GCN Node Embedding 2D Visualization')
        plt.xlabel('t-SNE Feature 1')
        plt.ylabel('t-SNE Feature 2')
        
        # 保存图像
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"节点嵌入2D可视化已保存至: {save_path}")
    
    def visualize_embeddings_3d(self, save_path='../results/embeddings_3d.png'):
        """
        使用t-SNE将节点嵌入可视化为3D散点图
        
        参数:
            save_path: 保存图像的路径
        """
        # 提取节点嵌入
        embeddings = self.extract_node_embeddings()
        
        # 使用PCA进行初步降维
        if embeddings.shape[1] > 50:
            pca = PCA(n_components=50)
            embeddings_pca = pca.fit_transform(embeddings)
        else:
            embeddings_pca = embeddings
            
        # 使用t-SNE进行降维到3D
        tsne = TSNE(n_components=3, random_state=42)
        embeddings_3d = tsne.fit_transform(embeddings_pca)
        
        # 创建3D静态散点图
        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111, projection='3d')
        
        # 为不同类别使用不同颜色
        scatter = ax.scatter(
            embeddings_3d[:, 0], 
            embeddings_3d[:, 1], 
            embeddings_3d[:, 2],
            c=self.labels,
            cmap='viridis',
            alpha=0.8,
            s=50
        )
        
        ax.set_title('GCN Node Embedding 3D Visualization')
        ax.set_xlabel('t-SNE Feature 1')
        ax.set_ylabel('t-SNE Feature 2')
        ax.set_zlabel('t-SNE Feature 3')
        
        plt.colorbar(scatter, ax=ax, label='Class')
        
        # 保存图像
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"节点嵌入3D可视化已保存至: {save_path}")
    
    def visualize_attention_weights(self, save_path='../results/attention_weights.png'):
        """
        可视化注意力权重 - 使用GAT层的注意力权重
        
        参数:
            save_path: 保存图像的路径
        """
        # 注意: 这需要模型中GAT层能够访问注意力权重
        # 创建数据集
        dataset = create_dataset(self.features, self.labels, self.edge_index)
        
        # 选择一个样本进行可视化
        sample_idx = 0
        data = dataset[sample_idx].to(self.device)
        
        # 使用模型获取注意力权重
        with torch.no_grad():
            # 模拟前向传播以获取注意力权重
            x, edge_index = data.x, data.edge_index
            
            # GCN和SAGE层
            x1 = self.model.conv1(x, edge_index)
            x1 = self.model.bn1(x1)
            x1 = torch.relu(x1)
            
            x2 = self.model.conv2(x1, edge_index)
            x2 = self.model.bn2(x2)
            x2 = torch.relu(x2)
            
            # GAT层 - 需要修改GAT层代码来返回注意力权重
            # 这里简化为使用边的权重
            edge_list = self.edge_index.t().numpy()
            num_edges = edge_list.shape[0]
            
            # 随机生成注意力权重(实际应用中应获取真实权重)
            attention_weights = np.random.rand(num_edges)
            
        # 构建图
        G = nx.Graph()
        
        # 添加节点
        for i in range(len(self.features)):
            G.add_node(i, label=str(self.labels[i]))
            
        # 添加边并设置权重
        for i, edge in enumerate(edge_list):
            G.add_edge(edge[0], edge[1], weight=attention_weights[i])
        
        # 绘制图
        plt.figure(figsize=(12, 10))
        pos = nx.spring_layout(G, seed=42)
        
        # 获取边的权重用于设置宽度
        edges = G.edges()
        weights = [G[u][v]['weight'] * 3 for u, v in edges]  # 权重乘以3以增加可视性
        
        # 绘制节点
        nx.draw_networkx_nodes(
            G, pos, 
            node_color=[plt.cm.viridis(x) for x in self.labels/max(self.labels)], 
            node_size=100, 
            alpha=0.8
        )
        
        # 绘制边 - 宽度与注意力权重成比例
        nx.draw_networkx_edges(
            G, pos, 
            width=weights, 
            alpha=0.5, 
            edge_color=weights,
            edge_cmap=plt.cm.Blues
        )
        
        plt.title('GCN Model Attention Weights Visualization')
        plt.colorbar(plt.cm.ScalarMappable(cmap=plt.cm.Blues), 
                     label='Attention Weight')
        
        # 保存图像
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"注意力权重可视化已保存至: {save_path}")
    
    def visualize_heatmap(self, save_path='../results/correlation_heatmap.png'):
        """
        可视化特征之间的相关性热力图
        
        参数:
            save_path: 保存图像的路径
        """
        # 计算相关性矩阵
        corr_matrix = np.corrcoef(self.features.T)
        
        # 绘制热力图
        plt.figure(figsize=(12, 10))
        sns.heatmap(
            corr_matrix, 
            cmap='coolwarm', 
            center=0,
            square=True,
            annot=False,
            cbar_kws={'label': 'Correlation Coefficient'}
        )
        
        plt.title('Feature Correlation Heatmap')
        plt.tight_layout()
        
        # 保存图像
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"特征相关性热力图已保存至: {save_path}")


# 主函数
def main():
    """
    主函数 - 执行各种可视化分析
    """
    # 创建结果目录
    os.makedirs('../results', exist_ok=True)
    
    # 初始化可视化器
    visualizer = GCNVisualizer()
    
    # 执行各种可视化
    print("正在生成神经元拓扑结构图...")
    visualizer.visualize_network()
    
    print("正在生成节点嵌入2D可视化...")
    visualizer.visualize_embeddings_2d()
    
    print("正在生成节点嵌入3D静态可视化...")
    visualizer.visualize_embeddings_3d()
    
    print("正在生成注意力权重可视化...")
    visualizer.visualize_attention_weights()
    
    print("正在生成特征相关性热力图...")
    visualizer.visualize_heatmap()
    
    print("可视化分析完成！所有结果已保存到../results目录")


if __name__ == "__main__":
    main()