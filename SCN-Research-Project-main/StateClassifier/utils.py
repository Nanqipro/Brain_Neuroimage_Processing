"""
脑网络状态分类器工具模块

该模块提供了数据加载和预处理的工具函数，包括从CSV文件读取图数据结构，
以及数据集的分割与批处理等功能。

作者: SCN研究小组
日期: 2023
"""

import csv
import numpy as np
import torch
import torch.utils.data as utils

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
            labels.append(int(row[2])-1)  # 标签从0开始，所以减1

    labels = np.array(labels)

    return edges, node_features, labels

def get_dataset(data_path, train_propotion=0.6, valid_propotion=0.2, BATCH_SIZE=1):
    """
    准备训练、验证和测试数据集
    
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


