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
    
    def detect_communities(self):
        """
        使用Louvain算法检测图中的社区结构
        
        返回:
            communities: 节点-社区映射字典，键为节点ID，值为社区ID
        """
        print("\n开始进行社区检测...")
        
        try:
            # 使用Louvain方法检测社区
            communities = community_louvain.best_partition(self.G)
            
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
    
    def visualize_communities(self, output_path=None, title="Neural Network Community Structure"):
        """
        可视化社区结构
        
        参数:
            output_path: 输出文件路径，默认为None，将根据title生成
            title: 图表标题
            
        返回:
            output_path: 输出图像路径
        """
        if self.communities is None:
            print("请先运行社区检测(detect_communities)再进行可视化")
            return None
            
        if output_path is None:
            output_path = os.path.join(self.output_dir, "community_structure.png")
            
        print(f"\n开始生成社区结构可视化: {output_path}")
        
        # 创建图形
        plt.figure(figsize=(15, 15))
        
        # 根据社区分配节点颜色
        node_colors = [self.community_colors[self.communities[node] % len(self.community_colors)] 
                      for node in self.G.nodes()]
        
        # 获取边权重用于设置边的宽度
        edge_weights = [self.G[u][v]['weight'] * 1.5 for u, v in self.G.edges()]
        
        # 使用spring布局
        pos = nx.spring_layout(self.G, k=2.0, iterations=100, seed=42)
        
        # 绘制边
        nx.draw_networkx_edges(self.G, pos, width=edge_weights, alpha=0.2, edge_color='gray')
        
        # 绘制节点
        node_sizes = [400 for _ in self.G.nodes()]
        nx.draw_networkx_nodes(self.G, pos, node_size=node_sizes, node_color=node_colors, alpha=0.9)
        
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
    
    # 初始化可视化器
    visualizer = GCNVisualizer(base_path=dataset_base_path)
    
    # 执行社区检测
    communities = visualizer.detect_communities()
    
    # 可视化社区结构
    community_viz_path = visualizer.visualize_communities()
    
    # 分析社区与行为的关联
    community_behavior_mapping = visualizer.analyze_community_behavior_association()
    
    print(f"可视化分析完成！所有结果已保存到{visualizer.output_dir}目录")


if __name__ == "__main__":
    main()