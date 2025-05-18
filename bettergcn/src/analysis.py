import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import seaborn as sns
import community as community_louvain
from collections import Counter, defaultdict
import pandas as pd
from scipy.stats import chi2_contingency
from sklearn.manifold import TSNE
from sklearn.metrics.pairwise import cosine_similarity
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform

# 导入项目中的其他模块
from model import ImprovedGCN
from process import load_data, generate_graph, create_pyg_dataset, compute_correlation_matrix, visualize_graph

class GCNVisualizer:
    """
    GCN模型可视化工具类，用于生成特定的拓扑结构图
    """
    def __init__(self, base_path='../datasets/EMtrace01', model_file_name='best_model.pth', data_file_name='.xlsx'):
        """
        初始化可视化器
        
        参数:
            base_path: 基础路径，所有输入输出路径都基于此路径生成
            model_file_name: 模型文件名
            data_file_name: 数据文件名后缀
        """
        # 提取基础文件名（不含路径和扩展名）
        self.dataset_name = os.path.basename(base_path)
        
        # 设置路径
        self.base_path = base_path
        self.model_path = f"../results/{self.dataset_name}/{model_file_name}"
        self.data_path = f"{base_path}{data_file_name}"
        self.output_dir = f"../results/{self.dataset_name}"
        
        # 确保输出目录存在
        os.makedirs(self.output_dir, exist_ok=True)
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # 加载数据和模型
        self.features, self.labels, self.class_weights, self.class_names = load_data(self.data_path, return_names=True)
        self.num_classes = len(self.class_names)
        
        # 计算相关性矩阵用于构建神经元拓扑图
        self.correlation_matrix = compute_correlation_matrix(self.features)
        
        # 生成图结构
        self.G = self._create_graph_from_correlation()
        
        # 加载模型
        self.model = self._load_model()
        
        # 社区检测结果
        self.communities = None
        self.community_colors = None
        
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
    
    def detect_communities(self, resolution=1.0):
        """
        使用Louvain算法检测图中的社区结构
        
        参数:
            resolution: 分辨率参数，值越大社区数量越多，越小社区数量越少
            
        返回:
            communities: 节点-社区映射字典，键为节点ID，值为社区ID
        """
        print(f"\n开始进行社区检测 (分辨率: {resolution})...")
        
        try:
            # 使用Louvain方法检测社区
            communities = community_louvain.best_partition(self.G, resolution=resolution)
            
            # 统计社区信息
            community_count = Counter(communities.values())
            community_sizes = sorted([(comm, count) for comm, count in community_count.items()], 
                                    key=lambda x: x[1], reverse=True)
            
            print(f"社区检测完成，共发现{len(community_sizes)}个社区")
            print("社区大小分布（前5个）:")
            for i, (comm_id, size) in enumerate(community_sizes[:5]):
                print(f"  社区 {comm_id}: {size}个节点")
            
            # 为不同社区分配不同颜色
            community_colors = [
                "#FF6347", "#4682B4", "#32CD32", "#FFD700", "#9370DB", 
                "#20B2AA", "#FF69B4", "#8A2BE2", "#00CED1", "#FF8C00",
                "#1E90FF", "#FF1493", "#00FA9A", "#DC143C", "#BA55D3"
            ]
            
            # 如果社区数量超过颜色列表长度，则循环使用
            num_communities = len(community_sizes)
            if num_communities > len(community_colors):
                community_colors = community_colors * (num_communities // len(community_colors) + 1)
                
            # 存储检测结果
            self.communities = communities
            self.community_colors = community_colors
            
            return communities
            
        except Exception as e:
            print(f"社区检测失败: {str(e)}")
            import traceback
            print(f"错误详情:\n{traceback.format_exc()}")
            return {}
    
    def hierarchical_community_detection(self, n_communities=10):
        """
        使用层次聚类方法得到指定数量的社区
        
        参数:
            n_communities: 目标社区数量
            
        返回:
            communities: 节点-社区映射字典
        """
        print(f"\n开始使用层次聚类进行社区检测 (目标社区数: {n_communities})...")
        
        try:
            # 获取邻接矩阵
            adj_matrix = nx.to_numpy_array(self.G)
            
            # 将权重转换为距离矩阵：高权重(高相似度)=低距离
            # 对于没有连接的节点对，距离设为最大值1.0
            dist_matrix = 1.0 - adj_matrix
            
            # 确保对角线为0，这是scipy.spatial.distance.squareform的要求
            np.fill_diagonal(dist_matrix, 0)
            
            # 压缩距离矩阵为压缩距离向量（层次聚类算法需要）
            condensed_dist = squareform(dist_matrix)
            
            # 进行层次聚类 - 使用Ward方法，最小化聚类内方差
            print("正在进行层次聚类...")
            Z = linkage(condensed_dist, method='ward')
            
            # 切割树状图得到指定数量的社区
            print(f"截断聚类树以得到{n_communities}个社区...")
            labels = fcluster(Z, n_communities, criterion='maxclust')
            
            # 创建节点-社区映射
            communities = {node: int(label-1) for node, label in zip(self.G.nodes(), labels)}
            
            # 统计社区信息
            community_count = Counter(communities.values())
            community_sizes = sorted([(comm, count) for comm, count in community_count.items()], 
                                    key=lambda x: x[1], reverse=True)
            
            actual_n_communities = len(set(communities.values()))
            print(f"层次聚类完成，共生成了{actual_n_communities}个社区")
            print("社区大小分布:")
            for i, (comm_id, size) in enumerate(community_sizes):
                print(f"  社区 {comm_id}: {size}个节点")
            
            # 为不同社区分配不同颜色
            community_colors = [
                "#FF6347", "#4682B4", "#32CD32", "#FFD700", "#9370DB", 
                "#20B2AA", "#FF69B4", "#8A2BE2", "#00CED1", "#FF8C00",
                "#1E90FF", "#FF1493", "#00FA9A", "#DC143C", "#BA55D3"
            ]
            
            # 确保颜色足够
            if actual_n_communities > len(community_colors):
                community_colors = community_colors * (actual_n_communities // len(community_colors) + 1)
                
            # 存储检测结果
            self.communities = communities
            self.community_colors = community_colors
            
            # 可视化层次聚类树状图
            self._visualize_dendrogram(Z, n_communities)
            
            return communities
            
        except Exception as e:
            print(f"层次聚类社区检测失败: {str(e)}")
            import traceback
            print(f"错误详情:\n{traceback.format_exc()}")
            return {}
    
    def _visualize_dendrogram(self, Z, n_clusters, max_d=None):
        """
        可视化层次聚类的树状图
        
        参数:
            Z: 层次聚类的结果
            n_clusters: 社区数量
            max_d: 最大距离阈值，用于在图上显示截断线
        """
        from scipy.cluster.hierarchy import dendrogram
        
        plt.figure(figsize=(12, 8))
        plt.title('Neural Network Hierarchical Clustering Dendrogram', fontsize=16)
        plt.xlabel('Neuron Index', fontsize=12)
        plt.ylabel('Distance', fontsize=12)
        
        # 计算截断距离，如果未提供
        if max_d is None:
            # 找到产生n_clusters的距离
            from scipy.cluster.hierarchy import fcluster
            distances = Z[:, 2]
            distances_sorted = np.sort(distances)
            if len(distances) >= n_clusters:
                # 由于Python索引从0开始，我们需要使用len(distances) - n_clusters + 1来获取正确的距离
                threshold_idx = len(distances) - n_clusters + 1
                if threshold_idx < len(distances):
                    max_d = distances_sorted[threshold_idx]
                else:
                    max_d = distances_sorted[-1]
            else:
                max_d = distances_sorted[-1] if len(distances_sorted) > 0 else None
        
        # 绘制树状图
        dendrogram(
            Z,
            truncate_mode='lastp',  # 显示最后的p个聚类
            p=n_clusters * 2,       # 显示的节点数
            leaf_rotation=90.,      # 旋转叶节点标签
            leaf_font_size=10.,     # 叶节点标签的字体大小
            show_contracted=True,   # 压缩非叶节点
            color_threshold=max_d   # 颜色阈值
        )
        
        # 添加截断线
        if max_d:
            plt.axhline(y=max_d, c='k', ls='--', lw=1)
            plt.text(plt.xlim()[1] * 0.98, max_d, f'Cutoff Line (n_clusters={n_clusters})', 
                    va='center', ha='right', bbox=dict(facecolor='white', alpha=0.7))
        
        # 保存图像
        output_path = os.path.join(self.output_dir, 'hierarchy_dendrogram.png')
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"层次聚类树状图已保存至: {output_path}")
        return output_path
    
    def merge_small_communities(self, min_size=5):
        """
        合并小社区到最相近的大社区
        
        参数:
            min_size: 小于此大小的社区将被合并
            
        返回:
            new_communities: 合并后的社区映射
        """
        if self.communities is None:
            print("请先运行社区检测再合并小社区")
            return None
            
        print(f"\n开始合并小于{min_size}节点的社区...")
        
        # 统计每个社区的大小
        community_count = Counter(self.communities.values())
        community_sizes = {comm: count for comm, count in community_count.items()}
        
        # 找出小社区
        small_communities = {c for c, size in community_sizes.items() if size < min_size}
        
        if not small_communities:
            print("没有发现小于阈值的社区，无需合并")
            return self.communities
            
        print(f"发现{len(small_communities)}个小于{min_size}节点的社区，开始合并...")
        
        # 计算社区间连接强度
        community_connections = defaultdict(lambda: defaultdict(float))
        
        for u, v, data in self.G.edges(data=True):
            if u in self.communities and v in self.communities:
                c1, c2 = self.communities[u], self.communities[v]
                if c1 != c2:  # 不同社区之间的连接
                    weight = data.get('weight', 1.0)
                    community_connections[c1][c2] += weight
                    community_connections[c2][c1] += weight
        
        # 复制原始社区结果
        new_communities = dict(self.communities)
        
        # 处理每个小社区
        for small_comm in small_communities:
            # 找到连接最强的大社区
            best_target = None
            max_strength = -1
            
            for target_comm, strength in community_connections[small_comm].items():
                if target_comm not in small_communities and strength > max_strength:
                    best_target = target_comm
                    max_strength = strength
            
            # 如果找不到连接的大社区，则找最大的社区
            if best_target is None:
                large_comms = [(c, size) for c, size in community_sizes.items() if c not in small_communities]
                if large_comms:
                    best_target = max(large_comms, key=lambda x: x[1])[0]
                    
            # 合并社区
            if best_target is not None:
                for node, comm in self.communities.items():
                    if comm == small_comm:
                        new_communities[node] = best_target
                print(f"社区 {small_comm} (大小: {community_sizes[small_comm]}) 已合并到社区 {best_target} (大小: {community_sizes[best_target]})")
        
        # 重新映射社区ID为连续数字
        unique_communities = sorted(set(new_communities.values()))
        comm_mapping = {old: new for new, old in enumerate(unique_communities)}
        new_communities = {node: comm_mapping[comm] for node, comm in new_communities.items()}
        
        # 更新类属性
        self.communities = new_communities
        
        # 重新统计并打印社区信息
        community_count = Counter(new_communities.values())
        community_sizes = sorted([(comm, count) for comm, count in community_count.items()], 
                                key=lambda x: x[1], reverse=True)
        
        print(f"合并后共有{len(community_sizes)}个社区")
        print("合并后社区大小分布（前5个）:")
        for i, (comm_id, size) in enumerate(community_sizes[:5]):
            print(f"  社区 {comm_id}: {size}个节点")
            
        return new_communities
    
    def visualize_communities(self, output_path=None, title="Neural Network Community Structure", position_file=None):
        """
        可视化社区结构
        
        参数:
            output_path: 输出文件路径，默认为None，将根据title生成
            title: 图表标题
            position_file: 神经元位置坐标CSV文件路径，如果提供则使用真实空间位置
            
        返回:
            output_path: 输出图像路径
        """
        if self.communities is None:
            print("请先运行社区检测(detect_communities)再进行可视化")
            return None
            
        if output_path is None:
            output_path = os.path.join(self.output_dir, "community_structure.png")
            
        print(f"\n开始生成社区结构可视化: {output_path}")
        print(f"位置数据文件: {position_file if position_file else '未提供，将使用布局算法'}")
        
        # 创建图形
        plt.figure(figsize=(15, 15))
        
        # 根据社区分配节点颜色
        node_colors = [self.community_colors[self.communities[node] % len(self.community_colors)] 
                      for node in self.G.nodes()]
        
        # 获取边权重用于设置边的宽度
        edge_weights = [self.G[u][v]['weight'] * 1.5 for u, v in self.G.edges()]
        
        # 确定节点位置
        if position_file and os.path.exists(position_file):
            try:
                # 读取位置数据
                import pandas as pd
                position_data = pd.read_csv(position_file)
                print(f"位置数据文件包含 {len(position_data)} 个神经元的位置信息")
                
                # 创建一个从神经元ID到位置的映射
                position_map = {}
                
                # 检查CSV文件中的列名
                columns = position_data.columns.tolist()
                x_col = 'relative_x' if 'relative_x' in columns else columns[1]  # 默认第二列为x坐标
                y_col = 'relative_y' if 'relative_y' in columns else columns[2]  # 默认第三列为y坐标
                id_col = 'number' if 'number' in columns else columns[0]        # 默认第一列为ID
                
                print(f"使用列: ID={id_col}, X={x_col}, Y={y_col}")
                
                # 检查是否存在神经元编号与索引不一致的问题
                neuron_ids_in_csv = set([int(row[id_col]) for _, row in position_data.iterrows()])
                nodes_in_graph = set(self.G.nodes())
                
                # 查看最小和最大神经元ID，可能的偏移量
                min_neuron_id = min(neuron_ids_in_csv) if neuron_ids_in_csv else -1
                max_neuron_id = max(neuron_ids_in_csv) if neuron_ids_in_csv else -1
                min_node_id = min(nodes_in_graph) if nodes_in_graph else -1
                max_node_id = max(nodes_in_graph) if nodes_in_graph else -1
                print(f"CSV中神经元ID范围: {min_neuron_id} 到 {max_neuron_id}")
                print(f"图中节点ID范围: {min_node_id} 到 {max_node_id}")
                
                # 计算可能的偏移量，检查多种情况
                offset = 0
                
                # 情况1: CSV从1开始，图从0开始 - 常见情况
                if min_neuron_id == 1 and min_node_id == 0 and max_neuron_id - min_neuron_id == max_node_id - min_node_id:
                    offset = -1
                    print(f"检测到固定偏移: CSV从1开始，图从0开始，应用偏移量 {offset}")
                
                # 情况2: 检查ID集合的平均差值
                elif len(neuron_ids_in_csv) > 0 and len(nodes_in_graph) > 0:
                    # 如果数据集大小相同且有固定偏移，计算平均偏移
                    if len(neuron_ids_in_csv) == len(nodes_in_graph):
                        sorted_csv_ids = sorted(neuron_ids_in_csv)
                        sorted_node_ids = sorted(nodes_in_graph)
                        # 计算前10个ID的平均偏移（或全部如果少于10个）
                        sample_size = min(10, len(sorted_csv_ids))
                        offsets = [sorted_csv_ids[i] - sorted_node_ids[i] for i in range(sample_size)]
                        avg_offset = sum(offsets) / len(offsets)
                        # 如果所有偏移都相同，则认为是有效偏移
                        if all(o == offsets[0] for o in offsets):
                            offset = -offsets[0]  # 取负值因为我们需要从CSV ID到节点ID的映射
                            print(f"检测到固定偏移模式: 平均偏移 {offsets[0]}，应用偏移量 {offset}")
                
                # 寻找每个节点的位置
                missing_nodes = []
                
                for node in self.G.nodes():
                    csv_id = node - offset  # 计算CSV中的对应ID
                    if csv_id < 0:  # 防止负ID
                        missing_nodes.append(node)
                        continue
                        
                    row = position_data[position_data[id_col] == csv_id]
                    if len(row) > 0:
                        position_map[node] = (float(row[x_col].values[0]), float(row[y_col].values[0]))
                    else:
                        missing_nodes.append(node)
                
                # 如果我们有足够的位置信息，使用真实的空间位置
                if len(position_map) == len(self.G.nodes()):
                    pos = position_map
                    print(f"使用真实空间位置可视化 {len(pos)} 个神经元")
                else:
                    print(f"警告：位置数据不完整 ({len(position_map)}/{len(self.G.nodes())})")
                    if missing_nodes:
                        print(f"缺失节点数量: {len(missing_nodes)}")
                        if len(missing_nodes) < 10:
                            print(f"缺失节点ID: {missing_nodes}")
                        
                    # 尝试使用可用的位置数据，对缺失的使用布局算法补充
                    if len(position_map) > len(self.G.nodes()) * 0.8:  # 如果有超过80%的节点有位置数据
                        print("使用部分真实位置数据，缺失部分使用布局算法补充")
                        # 为缺失的节点生成位置
                        try:
                            # 使用布局算法为缺失节点生成位置
                            temp_g = self.G.copy()
                            for node in position_map:
                                if node in temp_g:
                                    temp_g.remove_node(node)
                            
                            if temp_g.number_of_nodes() > 0:
                                temp_pos = nx.spring_layout(temp_g, seed=42)
                                # 合并两个位置字典
                                pos = {**position_map, **temp_pos}
                            else:
                                pos = position_map
                        except Exception as e:
                            print(f"补充缺失节点位置时出错: {e}")
                            pos = position_map
                    else:  # 如果缺失太多，使用布局算法
                        print("缺失位置数据过多，使用布局算法")
                        try:
                            pos = nx.kamada_kawai_layout(self.G)
                        except:
                            pos = nx.spring_layout(self.G, seed=42, k=2.0, iterations=100)
            except Exception as e:
                print(f"读取位置数据时出错：{e}，将使用布局算法")
                import traceback
                print(f"错误详情:\n{traceback.format_exc()}")
                try:
                    pos = nx.kamada_kawai_layout(self.G)
                except:
                    pos = nx.spring_layout(self.G, seed=42, k=2.0, iterations=100)
        else:
            # 如果没有位置文件，使用布局算法
            try:
                pos = nx.kamada_kawai_layout(self.G)
                print("使用Kamada-Kawai布局算法")
            except:
                pos = nx.spring_layout(self.G, seed=42, k=2.0, iterations=100)
                print("使用Spring布局算法")
        
        # 绘制边 - 根据不同社区边的透明度不同
        same_community_edges = []
        diff_community_edges = []
        
        for u, v in self.G.edges():
            if self.communities[u] == self.communities[v]:
                same_community_edges.append((u, v))
            else:
                diff_community_edges.append((u, v))
        
        # 先绘制社区间边（低透明度）
        if diff_community_edges:
            nx.draw_networkx_edges(self.G, pos, edgelist=diff_community_edges,
                                width=[self.G[u][v]['weight'] * 1.0 for u, v in diff_community_edges],
                                alpha=0.2, edge_color='lightgray')
        
        # 再绘制社区内边（高透明度）
        if same_community_edges:
            nx.draw_networkx_edges(self.G, pos, edgelist=same_community_edges,
                                width=[self.G[u][v]['weight'] * 1.5 for u, v in same_community_edges],
                                alpha=0.5, edge_color='gray')
        
        # 绘制节点
        node_sizes = [400 for _ in self.G.nodes()]
        nx.draw_networkx_nodes(self.G, pos, node_size=node_sizes, node_color=node_colors, alpha=0.9,
                            edgecolors='black', linewidths=0.5)
        
        # 绘制节点标签
        labels = {node: f'N{node+1}' for node in self.G.nodes()}
        nx.draw_networkx_labels(self.G, pos, labels, font_size=9, font_weight='bold')
        
        plt.title(title, fontsize=16)
        plt.axis('off')
        
        # 添加社区图例
        num_communities = len(set(self.communities.values()))
        legend_elements = [plt.Line2D([0], [0], marker='o', color='w', 
                                    markerfacecolor=self.community_colors[i % len(self.community_colors)],
                                    markersize=10, label=f'Community {i+1}')
                         for i in range(num_communities)]
        plt.legend(handles=legend_elements, loc='center left', bbox_to_anchor=(1, 0.5))
        
        # 添加位置信息来源说明
        position_source = "Real Space Position" if position_file and os.path.exists(position_file) else "Layout Algorithm Generated"
        plt.figtext(0.02, 0.02, f"Position Source: {position_source}", fontsize=10)
        plt.figtext(0.02, 0.04, f"Nodes: {len(self.G.nodes())}, Edges: {len(self.G.edges())}, Communities: {num_communities}", fontsize=10)
        
        # 保存图像
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"社区结构可视化已保存至: {output_path}")
        return output_path
        
    def analyze_community_behavior_association(self, output_path=None):
        """
        分析社区与行为标签的关联
        
        参数:
            output_path: 输出文件路径，默认为None，将自动生成
            
        返回:
            community_behavior_mapping: 社区-行为映射字典
        """
        if self.communities is None:
            print("请先运行社区检测(detect_communities)再进行行为关联分析")
            return None
            
        if output_path is None:
            output_path = os.path.join(self.output_dir, "community_behavior_mapping.png")
            
        print("\n分析社区与行为的关联...")
        
        # 初始化结果字典
        community_behavior_mapping = {}
        
        # 创建节点-社区映射
        node_to_community = {}
        for node, comm_id in self.communities.items():
            if comm_id not in node_to_community:
                node_to_community[comm_id] = []
            node_to_community[comm_id].append(node)
        
        # 分析每个社区与行为的关联
        for comm_id, nodes in node_to_community.items():
            # 过滤可能的无效节点
            valid_nodes = [node for node in nodes if node < self.features.shape[1]]
            
            if not valid_nodes:
                continue
                
            # 计算每种行为与该社区神经元的关联强度
            behavior_associations = {}
            
            # 对每个行为标签计算
            for behavior_idx, behavior_name in enumerate(self.class_names):
                # 该行为的样本掩码
                behavior_mask = (self.labels == behavior_idx)
                
                if np.sum(behavior_mask) == 0:
                    continue
                    
                # 计算该行为下社区神经元的平均活动
                try:
                    community_activity_in_behavior = np.mean(self.features[behavior_mask][:, valid_nodes], axis=0)
                    
                    # 计算其他行为下社区神经元的平均活动
                    other_activity = np.mean(self.features[~behavior_mask][:, valid_nodes], axis=0)
                    
                    # 计算效应量（Cohen's d）
                    behavior_std = np.std(self.features[behavior_mask][:, valid_nodes], axis=0)
                    other_std = np.std(self.features[~behavior_mask][:, valid_nodes], axis=0)
                    pooled_std = np.sqrt((behavior_std**2 + other_std**2) / 2)
                    effect_size = np.mean(np.abs(community_activity_in_behavior - other_activity) / (pooled_std + 1e-10))
                    
                    # 保存该行为的关联强度
                    behavior_associations[behavior_name] = {
                        'effect_size': float(effect_size),
                        'mean_activity': float(np.mean(community_activity_in_behavior)),
                        'mean_activity_diff': float(np.mean(community_activity_in_behavior - other_activity))
                    }
                except Exception as e:
                    print(f"计算社区{comm_id}与行为{behavior_name}的关联时出错: {str(e)}")
                    continue
            
            # 找出与该社区关联最强的行为
            if behavior_associations:
                strongest_behavior = max(behavior_associations.items(), key=lambda x: x[1]['effect_size'])
                
                # 保存社区-行为映射
                community_behavior_mapping[f'Community_{comm_id}'] = {
                    'behavior': strongest_behavior[0],
                    'effect_size': strongest_behavior[1]['effect_size'],
                    'mean_activity': strongest_behavior[1]['mean_activity'],
                    'mean_activity_diff': strongest_behavior[1]['mean_activity_diff'],
                    'neurons': [str(node) for node in nodes],
                    'size': len(nodes),
                    'behavior_associations': behavior_associations
                }
                
                print(f"社区 {comm_id} ({len(nodes)} 个神经元) 与行为 '{strongest_behavior[0]}' 最相关 (效应量: {strongest_behavior[1]['effect_size']:.3f})")
        
        print(f"分析完成，共发现{len(community_behavior_mapping)}个社区与行为的关联")
        
        # 可视化社区-行为关联
        self._visualize_community_behavior_mapping(community_behavior_mapping, output_path)
        
        return community_behavior_mapping
    
    def _visualize_community_behavior_mapping(self, community_behavior_mapping, output_path):
        """
        可视化社区与行为之间的关联
        
        参数:
            community_behavior_mapping: 社区-行为映射字典
            output_path: 输出文件路径
        """
        if not community_behavior_mapping:
            print("警告: 没有社区-行为映射数据可视化")
            return None
        
        # 确保输出目录存在
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
        # 准备数据
        communities = []
        behaviors = []
        effect_sizes = []
        community_sizes = []
        
        for comm_id, data in community_behavior_mapping.items():
            communities.append(comm_id)
            behaviors.append(data['behavior'])
            effect_sizes.append(data['effect_size'])
            community_sizes.append(data['size'])
        
        # 创建图表
        plt.figure(figsize=(14, 8))
        
        # 创建颜色映射
        unique_behaviors = list(set(behaviors))
        behavior_colors = plt.cm.tab10(np.linspace(0, 1, len(unique_behaviors)))
        behavior_color_map = {b: behavior_colors[i] for i, b in enumerate(unique_behaviors)}
        
        # 将社区按大小排序
        sorted_indices = np.argsort(community_sizes)[::-1]  # 从大到小排序
        
        # 创建柱状图
        bar_positions = np.arange(len(communities))
        bars = plt.bar(bar_positions, 
                       [effect_sizes[i] for i in sorted_indices], 
                       width=0.8,
                       color=[behavior_color_map[behaviors[i]] for i in sorted_indices],
                       alpha=0.8)
        
        # 添加社区大小标注
        for i, bar in enumerate(bars):
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + 0.05,
                    f'{community_sizes[sorted_indices[i]]} neurons',
                    ha='center', va='bottom', rotation=0,
                    fontsize=9)
        # 添加标签和标题
        plt.xlabel('Neuron Communities', fontsize=12)
        plt.ylabel('Behavior Association Strength (Effect Size)', fontsize=12)
        plt.title('Neuron Community-Behavior Association Analysis', fontsize=14)
        # 设置x轴刻度
        plt.xticks(bar_positions, [communities[i] for i in sorted_indices], rotation=45)
        
        # 添加行为图例
        legend_elements = [plt.Rectangle((0,0), 1, 1, color=behavior_color_map[b], alpha=0.8, label=b) 
                          for b in unique_behaviors]
        plt.legend(handles=legend_elements, title='Behavior Types', loc='upper right')
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"社区-行为关联可视化已保存到: {output_path}")
        
        # 创建网络可视化
        network_output_path = os.path.join(os.path.dirname(output_path), 'community_behavior_network.png')
        self._visualize_community_behavior_network(community_behavior_mapping, network_output_path)
        
        return output_path
    
    def _visualize_community_behavior_network(self, community_behavior_mapping, output_path):
        """
        可视化基于社区划分的神经元网络，不同社区使用不同颜色，并根据行为关联标注
        
        参数:
            community_behavior_mapping: 社区-行为映射字典
            output_path: 输出文件路径
        """
        plt.figure(figsize=(16, 16))
        
        # 创建颜色映射
        unique_behaviors = list(set([data['behavior'] for data in community_behavior_mapping.values()]))
        behavior_colors = plt.cm.tab10(np.linspace(0, 1, len(unique_behaviors)))
        behavior_color_map = {b: behavior_colors[i] for i, b in enumerate(unique_behaviors)}
        
        # 创建节点颜色映射
        node_colors = {}
        node_communities = {}
        
        # 解析社区ID
        for comm_id, data in community_behavior_mapping.items():
            # 从'社区_X'格式中提取数字
            comm_num = int(comm_id.split('_')[1]) if '_' in comm_id else int(comm_id)
            behavior = data['behavior']
            color = behavior_color_map[behavior]
            
            for node_str in data['neurons']:
                # 将节点字符串转换为合适的节点ID
                try:
                    if node_str.isdigit():
                        node = int(node_str)
                    else:
                        node = node_str
                        
                    node_colors[node] = color
                    node_communities[node] = comm_id
                except Exception as e:
                    print(f"处理节点{node_str}时出错: {str(e)}")
                    continue
        
        # 使用spring布局 - 控制随机种子以确保每次可视化结果一致
        pos = nx.spring_layout(self.G, seed=42, k=0.3)
        
        # 绘制节点
        for node in self.G.nodes():
            if node in node_colors:
                nx.draw_networkx_nodes(self.G, pos, nodelist=[node], 
                                     node_color=[node_colors[node]], 
                                     node_size=100, alpha=0.8)
            else:
                # 对于没有社区的节点，使用灰色
                nx.draw_networkx_nodes(self.G, pos, nodelist=[node], 
                                     node_color=['lightgray'], 
                                     node_size=50, alpha=0.5)
        
        # 绘制边 - 根据是否连接同一社区的节点使用不同透明度
        same_community_edges = []
        diff_community_edges = []
        
        for u, v in self.G.edges():
            if u in node_communities and v in node_communities:
                if node_communities[u] == node_communities[v]:
                    same_community_edges.append((u, v))
                else:
                    diff_community_edges.append((u, v))
            else:
                diff_community_edges.append((u, v))
        
        # 同一社区内的边使用高透明度
        nx.draw_networkx_edges(self.G, pos, edgelist=same_community_edges, 
                             width=1.5, alpha=0.7, edge_color='gray')
        # 不同社区之间的边使用低透明度
        nx.draw_networkx_edges(self.G, pos, edgelist=diff_community_edges, 
                             width=0.5, alpha=0.2, edge_color='lightgray')
        
        # 添加社区标签
        # 计算每个社区的中心位置
        community_centers = {}
        for comm_id, data in community_behavior_mapping.items():
            nodes = []
            for node_str in data['neurons']:
                try:
                    if node_str.isdigit():
                        node = int(node_str)
                    else:
                        node = node_str
                    nodes.append(node)
                except:
                    continue
                    
            if not nodes:
                continue
                
            # 计算社区节点的平均位置
            centers_x = []
            centers_y = []
            
            for node in nodes:
                if node in pos:
                    centers_x.append(pos[node][0])
                    centers_y.append(pos[node][1])
            
            if centers_x and centers_y:
                center_x = sum(centers_x) / len(centers_x)
                center_y = sum(centers_y) / len(centers_y)
                community_centers[comm_id] = (center_x, center_y)
        
        # 绘制社区标签
        for comm_id, center in community_centers.items():
            behavior = community_behavior_mapping[comm_id]['behavior']
            plt.text(center[0], center[1], 
                    f"{comm_id}\n({behavior})", 
                    fontsize=12, fontweight='bold', 
                    ha='center', va='center',
                    bbox=dict(facecolor='white', alpha=0.7, edgecolor='none'))
        
        # 添加图例
        legend_elements = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=color, 
                                    markersize=10, label=behavior) 
                          for behavior, color in behavior_color_map.items()]
        plt.legend(handles=legend_elements, loc='upper right', title='Behavior Types')
        
        plt.title('Neural Network Community-Behavior Association Network', fontsize=16)
        plt.axis('off')
        plt.tight_layout()
        
        # 确保输出目录存在
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"社区-行为网络可视化已保存到: {output_path}")
        return output_path


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
    position_file = '../datasets/EMtrace01_Max_position.csv'
    
    # 初始化可视化器
    visualizer = GCNVisualizer(base_path=dataset_base_path)
    
    print("\n=== 方法1: 使用Louvain算法检测社区（默认分辨率）===")
    # 执行Louvain社区检测
    communities = visualizer.detect_communities()
    # 可视化社区结构
    community_viz_path = visualizer.visualize_communities(output_path="../results/EMtrace01/louvain_communities.png", 
                                                         title="Neural Network Community Structure (Louvain Algorithm)",
                                                         position_file=position_file)
    # 分析社区与行为的关联
    community_behavior_mapping = visualizer.analyze_community_behavior_association(
                                                         output_path="../results/EMtrace01/louvain_community_behaviors.png")
    
    print("\n=== 方法2: 使用Louvain算法，降低分辨率控制社区数量 ===")
    # 执行社区检测，降低分辨率
    communities = visualizer.detect_communities(resolution=0.3)
    # 可视化社区结构
    community_viz_path = visualizer.visualize_communities(output_path="../results/EMtrace01/louvain_communities_low_res.png", 
                                                         title="Neural Network Community Structure (Louvain Algorithm, Low Resolution)",
                                                         position_file=position_file)
    # 分析社区与行为的关联
    community_behavior_mapping = visualizer.analyze_community_behavior_association(
                                                         output_path="../results/EMtrace01/louvain_low_res_behaviors.png")
    
    print("\n=== 方法3: 使用层次聚类截断法控制社区数量 ===")
    # 执行层次聚类，指定社区数量
    communities = visualizer.hierarchical_community_detection(n_communities=10)
    # 可视化社区结构
    community_viz_path = visualizer.visualize_communities(output_path="../results/EMtrace01/hierarchical_communities.png", 
                                                         title="Neural Network Community Structure (Hierarchical Clustering, 10 Communities)",
                                                         position_file=position_file)
    # 分析社区与行为的关联
    community_behavior_mapping = visualizer.analyze_community_behavior_association(
                                                         output_path="../results/EMtrace01/hierarchical_behaviors.png")
    
    print("\n=== 方法4: 合并小社区 ===")
    # 使用Louvain算法检测社区
    communities = visualizer.detect_communities()
    # 合并小社区
    merged_communities = visualizer.merge_small_communities(min_size=5)
    # 可视化合并后的社区结构
    community_viz_path = visualizer.visualize_communities(output_path="../results/EMtrace01/merged_communities.png", 
                                                         title="Neural Network Community Structure (Merged Small Communities)",
                                                         position_file=position_file)
    # 分析社区与行为的关联
    community_behavior_mapping = visualizer.analyze_community_behavior_association(
                                                         output_path="../results/EMtrace01/merged_behaviors.png")
    
    # 创建一个专门用真实位置的版本进行对比
    print("\n=== 额外输出: 使用真实位置的层次聚类结果 ===")
    # 执行层次聚类，指定社区数量为更少的数字
    communities = visualizer.hierarchical_community_detection(n_communities=6)
    # 可视化社区结构 - 使用真实位置
    community_viz_path = visualizer.visualize_communities(
        output_path="../results/EMtrace01/hierarchical_real_positions.png", 
        title="Neural Network Community Structure (Hierarchical, Real Positions, 6 Communities)",
        position_file=position_file
    )
    
    print(f"\n可视化分析完成！所有结果已保存到{visualizer.output_dir}目录")
    print(f"所有可视化均使用了真实神经元位置数据：{position_file}")


if __name__ == "__main__":
    main()