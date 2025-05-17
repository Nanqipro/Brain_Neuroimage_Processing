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

class GCNVisualizer:
    """
    GCN模型可视化工具类，用于生成各种神经网络拓扑结构图
    """
    def __init__(self, base_path='../datasets/EMtrace01', model_file_name='_best_model.pth', data_file_name='.xlsx'):
        """
        初始化可视化器
        
        参数:
            base_path: 基础路径，所有输入输出路径都基于此路径生成
            model_file_name: 模型文件名后缀
            data_file_name: 数据文件名后缀
        """
        # 提取基础文件名（不含路径和扩展名）
        self.dataset_name = os.path.basename(base_path)
        
        # 设置路径
        self.base_path = base_path
        self.model_path = f"../results/{self.dataset_name}{model_file_name}"
        self.data_path = f"{base_path}{data_file_name}"
        self.output_dir = f"../results/{self.dataset_name}"
        
        # 确保输出目录存在
        os.makedirs(self.output_dir, exist_ok=True)
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # 加载数据和模型
        self.features, self.labels, self.encoder, _ = load_data(self.data_path)
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
        
        # 确保模型在正确的设备上
        self.model = self.model.to(self.device)
        
        embeddings = []
        with torch.no_grad():
            for data in dataset:
                # 将整个数据移动到设备上
                data = data.to(self.device)
                
                # 提取中间层特征（修改forward方法或使用钩子）
                # 这里我们直接使用模型的第一个卷积层输出作为节点嵌入
                # 明确确保x和edge_index在同一设备上
                x, edge_index = data.x.to(self.device), data.edge_index.to(self.device)
                
                # GCN第一层
                x1 = self.model.conv1(x, edge_index)
                x1 = self.model.bn1(x1)
                x1 = torch.relu(x1)
                
                embeddings.append(x1.cpu().numpy())
                
        return np.vstack(embeddings)
    
    def visualize_network(self, filename='network_visualization.png'):
        """
        可视化神经元之间的拓扑关系
        
        参数:
            filename: 保存文件名
        """
        save_path = os.path.join(self.output_dir, filename)
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
    
    def visualize_embeddings_2d(self, filename='embeddings_2d.png'):
        """
        使用t-SNE将节点嵌入可视化为2D散点图
        
        参数:
            filename: 保存文件名
        """
        save_path = os.path.join(self.output_dir, filename)
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
    
    def visualize_embeddings_3d(self, filename='embeddings_3d.png'):
        """
        使用t-SNE将节点嵌入可视化为3D散点图
        
        参数:
            filename: 保存文件名
        """
        save_path = os.path.join(self.output_dir, filename)
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
    
    def visualize_heatmap(self, filename='correlation_heatmap.png'):
        """
        可视化特征之间的相关性热力图
        
        参数:
            filename: 保存文件名
        """
        save_path = os.path.join(self.output_dir, filename)
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
    
    # 设置数据集基础路径 - 只需修改此处即可更改所有相关路径
    dataset_base_path = '../datasets/EMtrace01_plus'
    
    # 初始化可视化器
    visualizer = GCNVisualizer(base_path=dataset_base_path)
    
    # 执行各种可视化
    print("正在生成神经元拓扑结构图...")
    visualizer.visualize_network()
    
    print("正在生成节点嵌入2D可视化...")
    visualizer.visualize_embeddings_2d()
    
    print("正在生成节点嵌入3D静态可视化...")
    visualizer.visualize_embeddings_3d()
     
    print("正在生成特征相关性热力图...")
    visualizer.visualize_heatmap()
    
    print(f"可视化分析完成！所有结果已保存到{visualizer.output_dir}目录")


if __name__ == "__main__":
    main()