"""
数据处理模块

该模块用于加载和预处理神经元钙离子浓度波动数据，构建图结构
"""

import pandas as pd
import numpy as np
import torch
from torch_geometric.data import Data
import networkx as nx
from typing import Dict, List, Tuple, Optional, Union, Any
from pathlib import Path
import matplotlib.pyplot as plt
from scipy.stats import pearsonr

class NeuronDataProcessor:
    """
    神经元数据处理器
    
    用于加载和预处理神经元数据，构建图结构用于GNN模型
    """
    
    def __init__(self, config):
        """
        初始化数据处理器
        
        参数:
            config: 配置对象，包含数据路径和处理参数
        """
        self.config = config
        self.data = None
        self.neuron_features = None
        self.behavior_labels = None
        self.timestamp = None
        self.neuron_columns = []
        self.node_mapping = {}  # 节点名称到索引的映射
        self.graph = None
        
    def load_data(self, file_path: Optional[Path] = None) -> pd.DataFrame:
        """
        加载神经元数据
        
        参数:
            file_path: 数据文件路径，如果为None则使用配置中的默认路径
            
        返回:
            加载的数据DataFrame
        """
        if file_path is None:
            file_path = self.config.day6_data_path
            
        print(f"正在加载数据: {file_path}")
        self.data = pd.read_excel(file_path)
        
        # 提取神经元列
        self.neuron_columns = [col for col in self.data.columns if col.startswith('n') and col[1:].isdigit()]
        
        # 创建节点映射
        self.node_mapping = {name: i for i, name in enumerate(self.neuron_columns)}
        
        # 提取特征和标签
        self.neuron_features = self.data[self.neuron_columns].values
        self.behavior_labels = self.data['behavior'].values
        self.timestamp = self.data['stamp'].values
        
        print(f"数据加载完成，共 {len(self.data)} 个时间点，{len(self.neuron_columns)} 个神经元节点")
        return self.data
    
    def compute_correlation_matrix(self) -> np.ndarray:
        """
        计算神经元之间的相关性矩阵
        
        返回:
            神经元之间的相关性矩阵
        """
        if self.neuron_features is None:
            raise ValueError("请先加载数据")
            
        num_neurons = len(self.neuron_columns)
        corr_matrix = np.zeros((num_neurons, num_neurons))
        
        print("计算神经元之间的相关性...")
        for i in range(num_neurons):
            for j in range(i, num_neurons):
                if i == j:
                    corr_matrix[i, j] = 1.0
                else:
                    corr, _ = pearsonr(self.neuron_features[:, i], self.neuron_features[:, j])
                    corr_matrix[i, j] = corr
                    corr_matrix[j, i] = corr
        
        return corr_matrix
    
    def build_graph(self, threshold: Optional[float] = None) -> nx.Graph:
        """
        基于相关性构建神经元网络图
        
        参数:
            threshold: 添加边的相关性阈值，如果为None则使用配置中的默认值
            
        返回:
            构建的NetworkX图
        """
        if threshold is None:
            threshold = self.config.correlation_threshold
            
        # 计算相关性矩阵
        corr_matrix = self.compute_correlation_matrix()
        
        # 创建图
        G = nx.Graph()
        
        # 添加节点
        for neuron_name in self.neuron_columns:
            G.add_node(neuron_name)
            
        # 添加边
        edges = []
        weights = []
        for i in range(len(self.neuron_columns)):
            for j in range(i+1, len(self.neuron_columns)):
                if abs(corr_matrix[i, j]) > threshold:
                    neuron_i = self.neuron_columns[i]
                    neuron_j = self.neuron_columns[j]
                    G.add_edge(neuron_i, neuron_j, weight=abs(corr_matrix[i, j]))
                    edges.append((i, j))
                    weights.append(abs(corr_matrix[i, j]))
        
        self.graph = G
        print(f"图构建完成，共 {len(G.nodes())} 个节点，{len(G.edges())} 条边")
        
        return G
    
    def visualize_graph(self, output_path: Optional[Path] = None) -> None:
        """
        可视化神经元网络图
        
        参数:
            output_path: 图像保存路径
        """
        if self.graph is None:
            raise ValueError("请先构建图")
            
        if output_path is None:
            output_path = self.config.get_results_path("neuron_graph.png")
            
        plt.figure(figsize=(12, 12))
        
        # 使用spring布局
        pos = nx.spring_layout(self.graph, seed=42)
        
        # 获取边权重用于边的宽度
        edge_weights = [self.graph[u][v]['weight'] * self.config.edge_width for u, v in self.graph.edges()]
        
        # 绘制图
        nx.draw_networkx(
            self.graph, 
            pos=pos,
            with_labels=True,
            node_size=self.config.node_size,
            node_color='skyblue',
            font_size=10,
            width=edge_weights,
            edge_color='gray',
            alpha=0.8
        )
        
        plt.title("神经元相关性网络")
        plt.axis('off')
        
        # 保存图像
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"图像已保存至 {output_path}")
    
    def to_pytorch_geometric(self) -> Data:
        """
        将NetworkX图转换为PyTorch Geometric数据对象
        
        返回:
            PyTorch Geometric数据对象
        """
        if self.graph is None:
            raise ValueError("请先构建图")
            
        # 获取边列表
        edge_index = []
        edge_attr = []
        
        for u, v, data in self.graph.edges(data=True):
            # 从节点名称获取索引
            u_idx = self.node_mapping[u]
            v_idx = self.node_mapping[v]
            
            # 添加双向边
            edge_index.append([u_idx, v_idx])
            edge_index.append([v_idx, u_idx])
            
            # 添加边属性（权重）
            weight = data['weight']
            edge_attr.append([weight])
            edge_attr.append([weight])
        
        if not edge_index:
            print("警告：图中没有边，使用全连接图")
            num_nodes = len(self.node_mapping)
            edge_index = []
            edge_attr = []
            for i in range(num_nodes):
                for j in range(num_nodes):
                    if i != j:
                        edge_index.append([i, j])
                        edge_attr.append([0.1])  # 默认权重
        
        # 转换为PyTorch tensors
        edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
        edge_attr = torch.tensor(edge_attr, dtype=torch.float)
        x = torch.tensor(self.neuron_features, dtype=torch.float)
        
        # 创建PyG数据对象
        data = Data(
            x=x, 
            edge_index=edge_index,
            edge_attr=edge_attr
        )
        
        print(f"PyTorch Geometric数据对象创建完成：{data}")
        return data 