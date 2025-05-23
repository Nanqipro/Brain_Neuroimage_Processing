"""
脑网络状态分类器工具模块

该模块提供了数据加载和预处理的工具函数，包括从CSV文件读取图数据结构，
以及数据集的分割与批处理等功能。

作者: SCN研究小组
日期: 2023
改进版本: 增加PyTorch Geometric数据格式支持
"""

import csv
import numpy as np
import torch
import torch.utils.data as utils
from torch_geometric.data import Data, DataLoader as GeometricDataLoader
from config import config

def load_data(path):
    """
    从CSV文件加载图结构数据
    
    读取路径下的边、节点特征和标签数据，将其转换为适合模型输入的格式。
    
    参数
    ----------
    path : str
        数据文件所在的目录路径
        
    返回
    -------
    tuple
        (edges, node_features, labels)，其中：
        - edges: [num_graphs, 2, num_edges] 形状的边连接数据
        - node_features: [num_graphs, num_nodes, 3] 形状的节点特征数据
        - labels: [num_graphs] 形状的图标签数据
    """
    # 读取边数据
    edges = []
    tmp_edges = []
    with open(path+'/edges.csv') as edgefile:
        cur_graph = '1'  # 当前图的ID，初始为1
        edge_reader = csv.reader(edgefile)
        next(edge_reader)  # 跳过表头行
        for row in edge_reader:
            graph_id, src, dst, feat = row
            # 如果到了新的图，保存当前图的边并重置临时列表
            if graph_id != cur_graph:
                edges.append(tmp_edges)
                cur_graph = graph_id
                tmp_edges = []
            
            tmp_edges.append([int(src)-1, int(dst)-1])  # 节点ID从0开始，所以减1
        edges.append(tmp_edges)  # 添加最后一个图的边
    edges = np.array(edges).transpose(0, 2, 1)  # 转置为[num_graphs, 2, num_edges]格式


    # 读取节点特征数据
    node_features = []
    tmp_nodes = []
    with open(path+'/nodes.csv') as nodefile:
        cur_graph = '1'  # 当前图的ID，初始为1
        node_reader = csv.reader(nodefile)
        next(node_reader)  # 跳过表头行
        for row in node_reader:
            graph_id, node_id, feat = row
            # 如果到了新的图，保存当前图的节点特征并重置临时列表
            if graph_id != cur_graph:
                node_features.append(tmp_nodes)
                cur_graph = graph_id
                tmp_nodes = []
            
            # 解析特征字符串为浮点数列表
            feat = feat.split(',')
            feat = [float(x) for x in feat]
            tmp_nodes.append(feat)
        node_features.append(tmp_nodes)  # 添加最后一个图的节点特征
    node_features = np.array(node_features)

    # 读取图标签数据
    labels = []
    with open(path+'/graphs.csv') as graphfile:
        graph_reader = csv.reader(graphfile)
        next(graph_reader)  # 跳过表头行
        for row in graph_reader:
            labels.append(int(row[2]))  # 标签直接使用，无需减1

    labels = np.array(labels)

    return edges, node_features, labels

def create_geometric_data(node_features, edges, labels):
    """
    将数据转换为PyTorch Geometric Data格式
    
    Parameters
    ----------
    node_features : np.ndarray
        节点特征数组 [num_graphs, num_nodes, num_features]
    edges : np.ndarray
        边连接数组 [num_graphs, 2, num_edges]
    labels : np.ndarray
        图标签数组 [num_graphs]
        
    Returns
    -------
    list
        PyTorch Geometric Data对象列表
    """
    data_list = []
    
    for i in range(len(labels)):
        # 节点特征
        x = torch.tensor(node_features[i], dtype=torch.float)
        
        # 边索引 - 转换为PyTorch Geometric格式 [2, num_edges]
        edge_index = torch.tensor(edges[i], dtype=torch.long)
        
        # 图标签
        y = torch.tensor(labels[i], dtype=torch.long)
        
        # 创建Data对象
        data = Data(x=x, edge_index=edge_index, y=y)
        data_list.append(data)
    
    return data_list


def get_geometric_dataset(data_path=None, train_proportion=0.6, valid_proportion=0.2, batch_size=1):
    """
    准备PyTorch Geometric格式的训练、验证和测试数据集
    
    专门为先进模型设计的数据加载函数，支持PyTorch Geometric的Data格式
    
    Parameters
    ----------
    data_path : str, optional
        数据文件所在的目录路径，默认使用config中的路径
    train_proportion : float, optional
        训练集所占比例，默认为0.6
    valid_proportion : float, optional
        验证集所占比例，默认为0.2
    batch_size : int, optional
        批次大小，默认为1
        
    Returns
    -------
    tuple
        (train_loader, val_loader, test_loader) - PyTorch Geometric数据加载器
    """
    # 使用默认数据路径
    if data_path is None:
        data_path = str(config.DATA_DIR)
    
    # 加载原始数据
    edges, node_features, labels = load_data(data_path)
    
    # 准备数据集分割索引
    sample_size = labels.shape[0]
    index = np.arange(sample_size, dtype=int)
    np.random.seed(42)  # 使用固定种子确保可复现性
    np.random.shuffle(index)
    
    # 按比例分割数据集
    train_end = int(np.floor(sample_size * train_proportion))
    valid_end = int(np.floor(sample_size * (train_proportion + valid_proportion)))
    
    train_index = index[:train_end]
    valid_index = index[train_end:valid_end]
    test_index = index[valid_end:]
    
    # 通过索引获取对应数据集
    train_nodes = node_features[train_index]
    train_edges = edges[train_index]
    train_labels = labels[train_index]
    
    valid_nodes = node_features[valid_index]
    valid_edges = edges[valid_index]
    valid_labels = labels[valid_index]
    
    test_nodes = node_features[test_index]
    test_edges = edges[test_index]
    test_labels = labels[test_index]
    
    # 转换为PyTorch Geometric格式
    train_data_list = create_geometric_data(train_nodes, train_edges, train_labels)
    valid_data_list = create_geometric_data(valid_nodes, valid_edges, valid_labels)
    test_data_list = create_geometric_data(test_nodes, test_edges, test_labels)
    
    # 创建PyTorch Geometric DataLoader
    train_loader = GeometricDataLoader(train_data_list, batch_size=batch_size, shuffle=True)
    valid_loader = GeometricDataLoader(valid_data_list, batch_size=batch_size, shuffle=False)
    test_loader = GeometricDataLoader(test_data_list, batch_size=batch_size, shuffle=False)
    
    return train_loader, valid_loader, test_loader


def get_dataset(data_path=None, train_propotion=0.6, valid_propotion=0.2, BATCH_SIZE=1):
    """
    智能数据集加载器 - 自动选择合适的数据格式
    
    这个函数会根据是否安装了torch_geometric来决定返回哪种格式的数据
    为了保持向后兼容性，默认返回PyTorch Geometric格式（如果可用）
    """
    try:
        # 尝试使用PyTorch Geometric格式（新的先进模型需要）
        return get_geometric_dataset(
            data_path=data_path or str(config.DATA_DIR),
            train_proportion=train_propotion,
            valid_proportion=valid_propotion,
            batch_size=BATCH_SIZE
        )
    except ImportError:
        # 如果PyTorch Geometric不可用，则使用传统格式
        print("PyTorch Geometric不可用，使用传统数据格式")
        return get_traditional_dataset(data_path, train_propotion, valid_propotion, BATCH_SIZE)


def get_traditional_dataset(data_path, train_propotion=0.6, valid_propotion=0.2, BATCH_SIZE=1):
    """
    准备训练、验证和测试数据集（传统PyTorch格式）
    
    加载数据并按照指定比例分割为训练集、验证集和测试集，同时进行数据增强和批处理。
    
    参数
    ----------
    data_path : str
        数据文件所在的目录路径
    train_propotion : float, 可选
        训练集所占比例，默认为0.6
    valid_propotion : float, 可选
        验证集所占比例，默认为0.2
    BATCH_SIZE : int, 可选
        批次大小，默认为1
        
    返回
    -------
    tuple
        (train_dataloader, valid_dataloader, test_dataloader)
        三个数据加载器，用于模型的训练、验证和测试
    """
    # 加载原始数据
    edges, node_features, labels = load_data(data_path)
    
    # 准备数据集分割索引
    sample_size = labels.shape[0]
    index = np.arange(sample_size-1, dtype=int)
    np.random.seed(0)  # 设置随机种子确保可复现性
    np.random.shuffle(index)  # 随机打乱索引
    
    # 按比例分割数据集
    train_index = index[0 : int(np.floor(sample_size * train_propotion))]
    valid_index = index[int(np.floor(sample_size * train_propotion)) : int(np.floor(sample_size * (train_propotion + valid_propotion)))]
    test_index = index[int(np.floor(sample_size * (train_propotion + valid_propotion))):]

    # 通过索引获取对应数据集
    train_data, train_label, train_edge = node_features[train_index,:,:], labels[train_index], edges[train_index]
    valid_data, valid_label, valid_edge = node_features[valid_index,:,:], labels[valid_index], edges[valid_index]
    test_data, test_label, test_edge = node_features[test_index,:,:], labels[test_index], edges[test_index]

    # 对训练集进行数据增强 - 重复样本并添加噪声
    num_graphs = train_label.shape[0]
    train_edge = np.repeat(train_edge, 4, axis=0)  # 每个样本重复4次
    train_label = np.repeat(train_label, 4, axis=0)
    train_data = np.repeat(train_data, 4, axis=0)
    # 对重复的样本添加随机噪声
    train_data[num_graphs:,:,:] = np.random.normal(1, 0.5, (num_graphs*3, 1, 1)) * train_data[num_graphs:,:,:]

    # 转换为PyTorch张量
    train_data, train_label, train_edge = torch.Tensor(train_data), torch.LongTensor(train_label), torch.LongTensor(train_edge)
    valid_data, valid_label, valid_edge = torch.Tensor(valid_data), torch.LongTensor(valid_label), torch.LongTensor(valid_edge)
    test_data, test_label, test_edge = torch.Tensor(test_data), torch.LongTensor(test_label), torch.LongTensor(test_edge)

    # 创建TensorDataset
    train_dataset = utils.TensorDataset(train_data, train_label, train_edge)
    valid_dataset = utils.TensorDataset(valid_data, valid_label, valid_edge)
    test_dataset = utils.TensorDataset(test_data, test_label, test_edge)
    
    # 创建DataLoader用于批处理
    train_dataloader = utils.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
    valid_dataloader = utils.DataLoader(valid_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
    test_dataloader = utils.DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)

    return train_dataloader, valid_dataloader, test_dataloader


