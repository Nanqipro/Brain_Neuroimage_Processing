import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import seaborn as sns
import community as community_louvain
from collections import Counter, defaultdict
import pandas as pd
from scipy.stats import chi2_contingency
import matplotlib.patheffects as PathEffects

# 导入项目中的其他模块
from model import ImprovedGCN
from process import load_data, generate_graph, create_pyg_dataset, compute_correlation_matrix, visualize_graph

class GCNVisualizer:
    """
    GCN模型可视化工具类，用于生成特定的拓扑结构图
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
        self.features, self.labels, _, self.class_weights, self.class_names = load_data(self.data_path, return_names=True)
        self.num_classes = len(self.class_names)
        
        # 计算相关性矩阵用于构建神经元拓扑图
        self.correlation_matrix = compute_correlation_matrix(self.features)
        
        # 生成图结构
        self.G = self._create_graph_from_correlation()
        
        # 加载模型
        self.model = self._load_model()
        
    def _create_graph_from_correlation(self, threshold=0.4):
        """
        基于神经元相关性创建图结构，使用process.py中的方法
        
        参数:
            threshold: 相关性阈值，大于该值的连接将被保留
            
        返回:
            networkx图对象
        """
        # 创建空图
        G = nx.Graph()
        
        # 添加节点（神经元）
        num_neurons = self.correlation_matrix.shape[0]
        for i in range(num_neurons):
            G.add_node(i, neuron_id=i)
        
        # 使用process.py中的generate_graph函数生成边
        # 为了与self.correlation_matrix一致，我们传入一个样本特征（可以是任意的，只需要长度一致）
        sample_features = np.zeros(num_neurons)  # 虚拟样本，仅用于获取神经元数量
        edge_index, edge_attr = generate_graph(sample_features, self.correlation_matrix, threshold)
        
        # 将边添加到图中
        for i in range(edge_index.shape[1]):
            src = int(edge_index[0, i])
            dst = int(edge_index[1, i])
            weight = float(edge_attr[i]) if edge_attr is not None else 1.0
            # 由于添加边是双向的，我们只添加一次（防止重复）
            if not G.has_edge(src, dst):
                G.add_edge(src, dst, weight=weight)
        
        print(f"创建的神经元图: {num_neurons}个节点, {G.number_of_edges()}条边")
        return G
    
    def _load_model(self):
        """
        加载训练好的GCN模型
        
        返回:
            加载好的模型
        """
        # 确保模型路径存在
        if not os.path.exists(self.model_path):
            print(f"警告: 模型文件不存在: {self.model_path}，跳过模型加载")
            return None
        
        # 初始化模型（与训练时结构一致）
        model = ImprovedGCN(
            num_features=1,  # 神经元特征是1维的（钙离子浓度）
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
        if self.model is None:
            print("模型未加载，无法提取节点嵌入")
            return None
            
        # 创建数据集
        dataset = create_pyg_dataset(self.features, self.labels, self.correlation_matrix)
        
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
    
    def visualize_heatmap(self, filename='correlation_heatmap.png'):
        """
        可视化特征之间的相关性热力图
        
        参数:
            filename: 保存文件名
        """
        save_path = os.path.join(self.output_dir, filename)
        # 计算相关性矩阵
        corr_matrix = self.correlation_matrix
        
        # 绘制热力图
        plt.figure(figsize=(12, 10))
        sns.heatmap(
            corr_matrix, 
            cmap='coolwarm', 
            center=0,
            square=True,
            annot=False,
            cbar_kws={'label': '相关系数'}
        )
        
        plt.title('神经元间相关性热力图')
        plt.tight_layout()
        
        # 保存图像
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"特征相关性热力图已保存至: {save_path}")
        
    def detect_communities(self, method='louvain', resolution=1.0):
        """
        检测神经元网络中的社区
        
        参数:
            method: 社区检测算法，可选 'louvain', 'girvan_newman', 'label_propagation'
            resolution: Louvain算法的分辨率参数，值越大社区越小
            
        返回:
            社区划分字典，键为节点ID，值为社区ID
        """
        if method == 'louvain':
            # Louvain方法
            print("使用Louvain算法检测社区...")
            partition = community_louvain.best_partition(self.G, resolution=resolution)
            
        elif method == 'girvan_newman':
            # Girvan-Newman方法
            print("使用Girvan-Newman算法检测社区...")
            comp = nx.community.girvan_newman(self.G)
            # 取第一个划分结果
            communities = tuple(sorted(c) for c in next(comp))
            partition = {}
            for i, community in enumerate(communities):
                for node in community:
                    partition[node] = i
                    
        elif method == 'label_propagation':
            # 标签传播算法
            print("使用标签传播算法检测社区...")
            communities = nx.community.label_propagation_communities(self.G)
            partition = {}
            for i, community in enumerate(communities):
                for node in community:
                    partition[node] = i
        else:
            print(f"未知社区检测方法: {method}，使用默认的Louvain方法")
            partition = community_louvain.best_partition(self.G, resolution=resolution)
        
        # 计算社区数量
        num_communities = len(set(partition.values()))
        print(f"检测到{num_communities}个社区")
        
        # 将社区信息添加到图中
        for node, community in partition.items():
            self.G.nodes[node]['community'] = community
            
        return partition
        
    def analyze_community_behavior_correlation(self, partition, behavior_data=None):
        """
        分析社区与行为标签之间的关系
        
        参数:
            partition: 社区划分字典
            behavior_data: 行为数据，默认使用self.labels
            
        返回:
            社区-行为对应关系DataFrame
        """
        if behavior_data is None:
            behavior_data = self.labels
            
        # 为每个社区创建一个标签分布计数器
        community_behavior = defaultdict(Counter)
        
        # 遍历每个神经元及其社区
        for node, community in partition.items():
            # 统计不同行为在该社区的分布
            for i, behavior in enumerate(behavior_data):
                community_behavior[community][behavior] += 1
        
        # 转换为DataFrame以便可视化
        data = []
        for community, behavior_counter in community_behavior.items():
            for behavior, count in behavior_counter.items():
                data.append({
                    '社区ID': community,
                    '行为标签': self.class_names[behavior] if self.class_names is not None else behavior,
                    '数量': count
                })
        
        df = pd.DataFrame(data)
        
        # 创建社区-行为交叉表
        cross_tab = pd.crosstab(df['社区ID'], df['行为标签'])
        
        # 计算卡方检验，分析社区与行为之间的关联
        chi2, p, dof, expected = chi2_contingency(cross_tab)
        
        print(f"社区与行为标签关联性分析:")
        print(f"卡方值: {chi2:.2f}")
        print(f"p值: {p:.6f}")
        print(f"自由度: {dof}")
        
        if p < 0.05:
            print("社区划分与行为标签存在显著关联")
        else:
            print("社区划分与行为标签关联性不显著")
            
        return cross_tab
    
    def visualize_communities(self, partition, filename='neuron_communities.png', 
                             with_labels=True, node_size=100, edge_alpha=0.3):
        """
        可视化神经元网络社区结构
        
        参数:
            partition: 社区划分字典
            filename: 保存文件名
            with_labels: 是否显示节点标签
            node_size: 节点大小
            edge_alpha: 边的透明度
        """
        save_path = os.path.join(self.output_dir, filename)
        
        # 获取社区数量
        num_communities = len(set(partition.values()))
        
        # 创建颜色映射
        if num_communities <= 10:
            # 使用定性颜色映射，如果社区数量较少
            cmap = plt.cm.get_cmap('tab10', num_communities)
        else:
            # 使用连续颜色映射，如果社区数量较多
            cmap = plt.cm.get_cmap('viridis', num_communities)
            
        # 为每个节点分配颜色
        node_colors = [cmap(partition[node]) for node in self.G.nodes()]
        
        # 绘制图
        plt.figure(figsize=(14, 12))
        
        # 使用spring布局算法获取节点位置
        try:
            pos = nx.kamada_kawai_layout(self.G)
        except:
            pos = nx.spring_layout(self.G, seed=42, k=0.3)
        
        # 绘制边
        nx.draw_networkx_edges(
            self.G, pos, 
            alpha=edge_alpha,
            width=[self.G[u][v].get('weight', 1) * 2 for u, v in self.G.edges()]
        )
        
        # 绘制节点
        nx.draw_networkx_nodes(
            self.G, pos,
            node_color=node_colors,
            node_size=node_size,
            cmap=cmap
        )
        
        # 绘制节点标签
        if with_labels:
            nx.draw_networkx_labels(self.G, pos, font_size=8, font_family='sans-serif')
        
        # 添加标题和图例
        community_sizes = Counter(partition.values())
        legend_labels = [f"社区 {i} ({community_sizes[i]}个神经元)" 
                        for i in sorted(set(partition.values()))]
        
        # 创建图例代理
        legend_handles = [plt.Line2D([0], [0], marker='o', color='w', 
                                    markerfacecolor=cmap(i), markersize=10)
                        for i in range(num_communities)]
        
        plt.legend(legend_handles, legend_labels, title="社区", loc='upper right')
        plt.title("神经元网络社区结构可视化", fontsize=16)
        plt.axis('off')
        plt.tight_layout()
        
        # 保存图像
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"社区可视化图已保存至: {save_path}")
    
    def visualize_community_behavior_heatmap(self, cross_tab, filename='community_behavior_heatmap.png'):
        """
        可视化社区与行为标签的关系热力图
        
        参数:
            cross_tab: 社区-行为交叉表
            filename: 保存文件名
        """
        save_path = os.path.join(self.output_dir, filename)
        
        # 归一化交叉表以显示比例
        # 按行归一化（每个社区内部的行为分布）
        norm_tab = cross_tab.div(cross_tab.sum(axis=1), axis=0)
        
        # 绘制热力图
        plt.figure(figsize=(14, 10))
        sns.heatmap(
            norm_tab, 
            cmap='YlOrRd', 
            annot=True, 
            fmt='.2f', 
            cbar_kws={'label': '比例'}
        )
        
        plt.title('社区与行为标签关系热力图', fontsize=16)
        plt.xlabel('行为标签', fontsize=14)
        plt.ylabel('社区ID', fontsize=14)
        plt.tight_layout()
        
        # 保存图像
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"社区-行为关系热力图已保存至: {save_path}")
    
    def visualize_community_network(self, partition, filename='community_network.png'):
        """
        可视化社区间的连接网络
        
        参数:
            partition: 社区划分字典
            filename: 保存文件名
        """
        save_path = os.path.join(self.output_dir, filename)
        
        # 构建社区图 - 节点是社区，边表示社区间连接
        C = nx.Graph()
        
        # 添加社区节点
        community_sizes = Counter(partition.values())
        for comm_id, size in community_sizes.items():
            C.add_node(comm_id, size=size)
        
        # 计算社区间连接
        for u, v in self.G.edges():
            comm_u = partition[u]
            comm_v = partition[v]
            if comm_u != comm_v:
                # 如果社区间已有连接，增加权重；否则创建新连接
                if C.has_edge(comm_u, comm_v):
                    C[comm_u][comm_v]['weight'] += 1
                else:
                    C.add_edge(comm_u, comm_v, weight=1)
        
        # 创建颜色映射
        num_communities = len(C.nodes())
        if num_communities <= 10:
            cmap = plt.cm.get_cmap('tab10', num_communities)
        else:
            cmap = plt.cm.get_cmap('viridis', num_communities)
        
        # 绘制社区网络
        plt.figure(figsize=(14, 12))
        
        # 使用spring布局，节点大小根据社区规模
        pos = nx.spring_layout(C, seed=42, k=0.3)
        
        # 节点大小根据社区规模调整
        node_sizes = [C.nodes[node]['size'] * 100 for node in C.nodes()]
        
        # 边宽度根据权重调整
        edge_widths = [C[u][v]['weight'] * 0.5 for u, v in C.edges()]
        
        # 绘制边
        nx.draw_networkx_edges(
            C, pos, 
            width=edge_widths,
            alpha=0.7,
            edge_color='gray'
        )
        
        # 绘制节点
        nx.draw_networkx_nodes(
            C, pos,
            node_color=[cmap(i) for i in C.nodes()],
            node_size=node_sizes,
            alpha=0.9
        )
        
        # 绘制节点标签
        nx.draw_networkx_labels(
            C, pos, 
            font_size=10, 
            font_weight='bold'
        )
        
        # 边标签（显示权重）
        edge_labels = {(u, v): f'{d["weight"]}' for u, v, d in C.edges(data=True)}
        nx.draw_networkx_edge_labels(C, pos, edge_labels=edge_labels, font_size=8)
        
        # 添加标题和图例
        plt.title("社区间连接网络", fontsize=16)
        
        # 为图例创建代理对象
        legend_handles = [plt.Line2D([0], [0], marker='o', color='w', 
                                     markerfacecolor=cmap(i), markersize=10)
                         for i in C.nodes()]
        legend_labels = [f"社区 {i} ({C.nodes[i]['size']}个神经元)" for i in C.nodes()]
        
        plt.legend(legend_handles, legend_labels, title="社区", loc='upper right')
        plt.axis('off')
        plt.tight_layout()
        
        # 保存图像
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"社区间连接网络图已保存至: {save_path}")
    
    def visualize_neuron_embedding(self, method='tsne', partition=None, filename='neuron_embedding.png'):
        """
        可视化神经元嵌入空间
        
        参数:
            method: 降维方法，'tsne'或'pca'
            partition: 社区划分字典，如果提供则按社区着色
            filename: 保存文件名
        """
        save_path = os.path.join(self.output_dir, filename)
        
        # 获取节点嵌入
        node_embeddings = self.extract_node_embeddings()
        
        if node_embeddings is None:
            print("节点嵌入获取失败，无法可视化")
            return
        
        # 降维
        if method == 'tsne':
            print("使用t-SNE进行降维...")
            reducer = TSNE(n_components=2, random_state=42)
        else:
            print("使用PCA进行降维...")
            reducer = PCA(n_components=2)
            
        reduced_embeddings = reducer.fit_transform(node_embeddings)
        
        # 可视化
        plt.figure(figsize=(14, 10))
        
        if partition is not None:
            # 按社区着色
            num_communities = len(set(partition.values()))
            if num_communities <= 10:
                cmap = plt.cm.get_cmap('tab10', num_communities)
            else:
                cmap = plt.cm.get_cmap('viridis', num_communities)
                
            # 为每个点分配颜色
            node_colors = [cmap(partition[i]) for i in range(len(reduced_embeddings))]
            
            # 绘制散点图
            scatter = plt.scatter(
                reduced_embeddings[:, 0], 
                reduced_embeddings[:, 1],
                c=node_colors,
                s=50,
                alpha=0.8
            )
            
            # 添加图例
            legend_handles = [plt.Line2D([0], [0], marker='o', color='w', 
                                        markerfacecolor=cmap(i), markersize=10)
                            for i in range(num_communities)]
            legend_labels = [f"社区 {i}" for i in range(num_communities)]
            
            plt.legend(legend_handles, legend_labels, title="社区", loc='best')
            
        else:
            # 普通散点图
            plt.scatter(
                reduced_embeddings[:, 0], 
                reduced_embeddings[:, 1],
                c='blue',
                s=50,
                alpha=0.8
            )
        
        # 添加一些标签
        for i in range(0, len(reduced_embeddings), max(1, len(reduced_embeddings) // 20)):
            txt = plt.annotate(
                f"{i}", 
                (reduced_embeddings[i, 0], reduced_embeddings[i, 1]),
                fontsize=9
            )
            txt.set_path_effects([PathEffects.withStroke(linewidth=3, foreground='w')])
        
        plt.title(f"神经元嵌入空间可视化 ({method.upper()})", fontsize=16)
        plt.xlabel(f"{method.upper()} 维度 1", fontsize=12)
        plt.ylabel(f"{method.upper()} 维度 2", fontsize=12)
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.tight_layout()
        
        # 保存图像
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"神经元嵌入空间可视化已保存至: {save_path}")
    
    def run_full_analysis(self, community_method='louvain', resolution=1.0, embedding_method='tsne'):
        """
        运行完整的社区分析和可视化流程
        
        参数:
            community_method: 社区检测方法
            resolution: 分辨率参数
            embedding_method: 嵌入空间可视化方法
        """
        print("开始进行完整的神经元网络分析...")
        
        # 1. 可视化神经元相关性热力图
        self.visualize_heatmap(filename='neuron_correlation_heatmap.png')
        
        # 2. 检测社区
        partition = self.detect_communities(method=community_method, resolution=resolution)
        
        # 3. 可视化社区结构
        self.visualize_communities(partition, filename=f'neuron_communities_{community_method}.png')
        
        # 4. 分析社区与行为的关系
        cross_tab = self.analyze_community_behavior_correlation(partition)
        
        # 5. 可视化社区与行为的关系热力图
        self.visualize_community_behavior_heatmap(cross_tab, filename='community_behavior_heatmap.png')
        
        # 6. 可视化社区间网络
        self.visualize_community_network(partition, filename='community_network.png')
        
        # 7. 可视化神经元嵌入空间
        if self.model is not None:
            self.visualize_neuron_embedding(
                method=embedding_method, 
                partition=partition,
                filename=f'neuron_embedding_{embedding_method}.png'
            )
        
        print("神经元网络分析完成！所有结果已保存到目录:", self.output_dir)


# 修改load_data函数，添加return_names参数以返回类名
def load_data(data_path, min_samples=50, return_names=False):
    """
    加载数据并进行预处理
    
    参数:
        data_path: 数据文件路径
        min_samples: 最小样本数
        return_names: 是否返回类名
        
    返回:
        如果return_names为True:
            (features, labels, class_weights, class_names)
        否则:
            (features, labels, class_weights)
    """
    # 这里应该调用process.py中的load_data函数，但可能需要修改其返回类名
    # 为简化代码，这里添加一个兼容层
    from process import load_data as original_load_data
    features, labels, class_weights, class_names = original_load_data(data_path, min_samples=min_samples)
    
    if return_names:
        return features, labels, class_weights, class_names
    else:
        return features, labels, class_weights


# 主函数
def main():
    """
    主函数 - 执行各种可视化分析
    """
    # 创建结果目录
    os.makedirs('../results', exist_ok=True)
    
    # 设置数据集基础路径 - 只需修改此处即可更改所有相关路径
    dataset_base_path = '../datasets/EMtrace01'
    
    # 初始化可视化器
    visualizer = GCNVisualizer(base_path=dataset_base_path)
    
    # 执行完整分析
    visualizer.run_full_analysis(
        community_method='louvain',  # 可选: 'louvain', 'girvan_newman', 'label_propagation'
        resolution=1.0,  # Louvain算法的分辨率参数，值越大社区越小
        embedding_method='tsne'  # 可选: 'tsne', 'pca'
    )
    
    print(f"可视化分析完成！所有结果已保存到{visualizer.output_dir}目录")


if __name__ == "__main__":
    main()