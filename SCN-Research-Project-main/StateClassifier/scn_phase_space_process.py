#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
3D SCN数据处理流程

该脚本用于处理脑神经元钙成像数据，将时间序列转换为相空间流形，并构建图数据集
主要步骤包括：
1. 加载原始钙信号数据
2. 将钙成像时间序列转换为相空间流形(phase-space manifolds)
3. 生成图数据集的节点、边和图属性文件

作者: SCN研究小组（Python版本）
日期: 2024
"""

import numpy as np
import pandas as pd
import scipy.io as sio
from scipy import stats
from scipy.signal import find_peaks
import os
import warnings
from tqdm import tqdm
from typing import List, Tuple, Optional, Union
import matplotlib.pyplot as plt

# 忽略警告
warnings.filterwarnings('ignore')


def mutual_information(signal: np.ndarray, partitions: int = 16, max_tau: int = 20) -> np.ndarray:
    """
    计算时间延迟互信息函数
    
    该函数计算时间序列的时间延迟互信息，用于确定相空间重构的最佳时间延迟。
    互信息是衡量两个变量之间相互依赖性的度量，第一个局部最小值通常被选为最佳时间延迟。
    
    Parameters
    ----------
    signal : np.ndarray
        输入时间序列，一维数组
    partitions : int, optional
        分区框数，用于概率估计的离散化，默认为16
    max_tau : int, optional
        最大时间延迟，默认为20
        
    Returns
    -------
    np.ndarray
        从延迟0到max_tau的互信息值数组
    """
    signal = np.array(signal).flatten()
    length = len(signal)
    
    # 确保延迟不超过信号长度
    if max_tau >= length:
        max_tau = length - 1
    
    # 将信号标准化到[0,1]区间
    signal_min = np.min(signal)
    signal_max = np.max(signal)
    interval = signal_max - signal_min
    
    if interval == 0:
        return np.zeros(max_tau + 1)
    
    signal_norm = (signal - signal_min) / interval
    
    # 将标准化信号离散化为整数值(0到partitions-1)
    signal_discrete = np.ceil(signal_norm * partitions).astype(int)
    signal_discrete[signal_discrete == 0] = 1
    signal_discrete[signal_discrete > partitions] = partitions
    signal_discrete -= 1  # 转换为0-based索引
    
    # 计算延迟从0到max_tau的互信息
    mi_values = np.zeros(max_tau + 1)
    
    for tau in range(max_tau + 1):
        mi_values[tau] = _compute_mutual_info(signal_discrete, tau, partitions)
    
    return mi_values


def _compute_mutual_info(signal_discrete: np.ndarray, tau: int, partitions: int) -> float:
    """
    计算给定时间延迟tau的互信息
    
    Parameters
    ----------
    signal_discrete : np.ndarray
        离散化后的信号数组
    tau : int
        时间延迟
    partitions : int
        分区数量
        
    Returns
    -------
    float
        计算得到的互信息值
    """
    length = len(signal_discrete)
    
    # 初始化联合概率和边缘概率的计数数组
    joint_count = np.zeros((partitions, partitions))
    marginal_count_i = np.zeros(partitions)
    marginal_count_j = np.zeros(partitions)
    
    count = 0
    
    # 统计联合出现频率和边缘频率
    for i in range(tau, length):
        val_i = signal_discrete[i - tau]
        val_j = signal_discrete[i]
        
        marginal_count_i[val_i] += 1
        marginal_count_j[val_j] += 1
        joint_count[val_i, val_j] += 1
        count += 1
    
    if count == 0:
        return 0.0
    
    # 计算互信息
    mutual_info = 0.0
    norm = 1.0 / count
    
    for i in range(partitions):
        p_i = marginal_count_i[i] * norm
        if p_i > 0:
            for j in range(partitions):
                p_j = marginal_count_j[j] * norm
                if p_j > 0:
                    p_ij = joint_count[i, j] * norm
                    if p_ij > 0:
                        mutual_info += p_ij * np.log(p_ij / (p_i * p_j))
    
    return mutual_info


def phase_space_reconstruction(signal: np.ndarray, dim: int, tau: int) -> np.ndarray:
    """
    相空间重构函数
    
    该函数将一维时间序列重构为高维相空间轨迹，用于非线性动力学分析。
    利用时间延迟嵌入方法，将一维信号映射到多维空间，以揭示系统的动力学特性。
    
    Parameters
    ----------
    signal : np.ndarray
        输入时间序列，一维数组
    dim : int
        嵌入维度，表示重构相空间的维数
    tau : int
        时间延迟，用于确定相空间点的构建方式
        
    Returns
    -------
    np.ndarray
        重构的相空间轨迹矩阵，大小为 T×dim，每行代表相空间中的一个点
    """
    signal = np.array(signal).flatten()
    N = len(signal)
    
    # 计算相空间中的总点数
    T = N - (dim - 1) * tau
    
    if T <= 0:
        raise ValueError(f"信号长度不足以进行相空间重构。需要至少 {(dim-1)*tau + 1} 个点")
    
    # 初始化相空间矩阵
    Y = np.zeros((T, dim))
    
    # 构建相空间轨迹
    for i in range(T):
        # 使用降序排列的延迟索引构建相空间点
        indices = i + (dim - 1) * tau - np.arange(dim) * tau
        Y[i, :] = signal[indices]
    
    return Y


def cellset_trim(dataset: List[List], trim_len: int) -> List[List]:
    """
    细胞数组裁剪函数
    
    该函数将列表中的每个非空元素裁剪到指定长度，
    主要用于统一相空间轨迹的长度，便于后续处理和分析。
    
    Parameters
    ----------
    dataset : List[List]
        输入的二维列表，包含相空间轨迹数据
    trim_len : int
        裁剪后的目标长度
        
    Returns
    -------
    List[List]
        裁剪后的二维列表，每个非空元素都被裁剪到相同的长度
    """
    cell_num = len(dataset)
    timeline = len(dataset[0]) if dataset else 0
    
    # 初始化结果列表
    data_trim = [[None for _ in range(timeline)] for _ in range(cell_num)]
    
    # 遍历所有细胞和时间点
    for i in range(cell_num):
        for j in range(timeline):
            temp = dataset[i][j]
            # 如果数据非空，则裁剪到指定长度
            if temp is not None and len(temp) >= trim_len:
                data_trim[i][j] = temp[:trim_len]
    
    return data_trim


def format_convert(x: np.ndarray) -> str:
    """
    数值数组转字符串格式化函数
    
    该函数将数值数组转换为以逗号分隔的字符串格式，
    主要用于将节点特征数据转换为CSV文件可接受的字符串格式。
    
    Parameters
    ----------
    x : np.ndarray
        包含要转换的数据的数值数组
        
    Returns
    -------
    str
        转换后的字符串，以逗号分隔各元素
    """
    return ','.join(map(str, x.flatten()))


def main():
    """
    主函数：执行3D SCN数据处理流程
    """
    # 配置参数
    file_path = './SCNData/Dataset1_SCNProject.mat'  # 输入文件路径
    frame_rate = 0.67  # 帧率，单位Hz
    out_path = './data'  # 输出目录路径
    
    # 创建输出目录
    os.makedirs(out_path, exist_ok=True)
    
    print("开始加载数据...")
    
    # 加载MAT格式数据
    try:
        data = sio.loadmat(file_path)
        F_set = data['F_set']  # 假设数据变量名为F_set
        print(f"数据加载成功！数据维度: {F_set.shape}")
    except Exception as e:
        print(f"数据加载失败: {e}")
        return
    
    # 将钙离子时间序列转换为相空间流形
    print("开始相空间重构...")
    
    cell_num, timeline = F_set.shape
    trace_zs_set = [[None for _ in range(timeline)] for _ in range(cell_num)]
    xyz = [[None for _ in range(timeline)] for _ in range(cell_num)]
    
    # 使用进度条显示处理进度
    total_iterations = min(10, cell_num) * timeline
    with tqdm(total=total_iterations, desc="相空间重构进度") as pbar:
        for tt in range(timeline):
            for ii in range(min(10, cell_num)):  # 只处理前10个细胞
                try:
                    # 获取钙信号数据
                    dat = F_set[ii, tt]
                    if dat is None or len(dat) == 0:
                        pbar.update(1)
                        continue
                    
                    # 确保数据是一维数组
                    if hasattr(dat, 'flatten'):
                        dat = dat.flatten()
                    else:
                        dat = np.array(dat).flatten()
                    
                    # Z-score标准化
                    trace_zs = stats.zscore(dat)
                    trace_zs_set[ii][tt] = trace_zs
                    
                    # 计算互信息以确定最佳时间延迟tau
                    mi = mutual_information(trace_zs)
                    peaks, _ = find_peaks(-mi)  # 寻找互信息的局部最小值
                    
                    if len(peaks) == 0:
                        tau = 8  # 如果没有找到局部最小值，使用经验值8
                    else:
                        tau = peaks[0]  # 使用第一个局部最小值
                    
                    # 设置相空间参数
                    dim = 3  # 嵌入维度(3D相空间)
                    
                    # 进行相空间重构
                    try:
                        y = phase_space_reconstruction(trace_zs, dim, tau)
                        xyz[ii][tt] = y
                    except ValueError as ve:
                        print(f"相空间重构失败 (细胞{ii}, 时间{tt}): {ve}")
                        xyz[ii][tt] = None
                    
                except Exception as e:
                    print(f"处理失败 (细胞{ii}, 时间{tt}): {e}")
                    xyz[ii][tt] = None
                
                pbar.update(1)
    
    print("开始裁剪相空间轨迹...")
    
    # 裁剪相空间轨迹到统一长度
    xyz_len = 170  # 经验确定的统一长度
    xyz_trim = cellset_trim(xyz, xyz_len)
    
    print("开始构建图数据集...")
    
    # 构建图数据集
    # 将相空间轨迹转换为图数据集格式，包括节点、边和图属性
    for_pred = []
    for row in xyz_trim:
        for item in row:
            if item is not None:
                for_pred.append(item)
    
    pred_num = len(for_pred)
    print(f"图数据总数: {pred_num}")
    
    if pred_num == 0:
        print("没有有效的图数据，程序终止")
        return
    
    # 生成nodes.csv文件
    print("生成nodes.csv文件...")
    
    # 构建节点ID和特征
    graph_ids = []
    node_ids = []
    
    for graph_idx in range(pred_num):
        for node_idx in range(xyz_len):
            graph_ids.append(graph_idx + 1)  # 图ID从1开始
            node_ids.append(node_idx + 1)    # 节点ID从1开始
    
    # 准备节点特征
    feat1 = np.vstack(for_pred)  # 将列表转换为矩阵
    features = []
    
    print("格式化节点特征...")
    for i in tqdm(range(len(feat1)), desc="格式化进度"):
        features.append(format_convert(feat1[i]))
    
    # 保存节点数据到CSV文件
    nodes_df = pd.DataFrame({
        'graph_id': graph_ids,
        'node_id': node_ids,
        'feat': features
    })
    nodes_file = os.path.join(out_path, 'nodes.csv')
    nodes_df.to_csv(nodes_file, index=False)
    print(f"nodes.csv已保存到: {nodes_file}")
    
    # 生成edges.csv文件
    print("生成edges.csv文件...")
    
    # 构建边的源节点和目标节点
    edge_graph_ids = []
    src_ids = []
    dst_ids = []
    edge_features = []
    
    for graph_idx in range(pred_num):
        for node_idx in range(xyz_len - 1):
            edge_graph_ids.append(graph_idx + 1)
            src_ids.append(node_idx + 1)
            dst_ids.append(node_idx + 2)
            edge_features.append(1)  # 边特征设为1表示连接存在
    
    # 保存边数据到CSV文件
    edges_df = pd.DataFrame({
        'graph_id': edge_graph_ids,
        'src_id': src_ids,
        'dst_id': dst_ids,
        'feat': edge_features
    })
    edges_file = os.path.join(out_path, 'edges.csv')
    edges_df.to_csv(edges_file, index=False)
    print(f"edges.csv已保存到: {edges_file}")
    
    # 生成graphs.csv文件
    print("生成graphs.csv文件...")
    
    # 构建图的属性和标签
    graph_ids = list(range(1, pred_num + 1))
    graph_features = ['1,0,0,0,0,0'] * pred_num  # 图特征
    labels = [0] * pred_num  # 初始标签为0
    
    # 保存图数据到CSV文件
    graphs_df = pd.DataFrame({
        'graph_id': graph_ids,
        'feat': graph_features,
        'label': labels
    })
    graphs_file = os.path.join(out_path, 'graphs.csv')
    graphs_df.to_csv(graphs_file, index=False)
    print(f"graphs.csv已保存到: {graphs_file}")
    
    print("全部完成！")
    print(f"输出文件位置: {out_path}")
    print(f"- 节点文件: {nodes_file}")
    print(f"- 边文件: {edges_file}")
    print(f"- 图文件: {graphs_file}")


if __name__ == "__main__":
    main() 