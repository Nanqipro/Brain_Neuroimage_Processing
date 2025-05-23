#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
3D SCN数据处理流程 (版本2 - 使用模块化src_python)

该脚本用于处理脑神经元钙成像数据，将时间序列转换为相空间流形，并构建图数据集
完全复制MATLAB版本scn_phase_space_process.m的功能和行为

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

# 导入我们自己的src_python模块
from src_python.mutual import mutual
from src_python.phasespace import phasespace  
from src_python.cellset2trim import cellset2trim
from src_python.formatConvert import formatConvert

# 忽略警告
warnings.filterwarnings('ignore')


def main():
    """
    主函数：执行3D SCN数据处理流程
    完全复制MATLAB版本的行为和逻辑
    """
    # 清理环境（模拟MATLAB的clearvars; clc; close all; warning off; dbstop if error）
    plt.close('all')
    warnings.filterwarnings('ignore')
    
    print("=== 3D SCN数据处理流程 ===")
    print("Python版本 - 等效于MATLAB scn_phase_space_process.m")
    print("添加src_python目录到Python路径... 完成")
    print()
    
    # 配置参数（与MATLAB版本完全一致）
    file_path = './SCNData/Dataset1_SCNProject.mat'  # 输入文件路径
    frame_rate = 0.67  # 帧率，单位Hz
    out_path = './data'  # 输出目录路径
    
    # 创建输出目录
    os.makedirs(out_path, exist_ok=True)
    print(f"输出目录已创建: {out_path}")
    
    print("\n== 加载数据 ==")
    
    # 加载MAT格式数据
    try:
        data = sio.loadmat(file_path)
        F_set = data['F_set']  # 加载F_set变量
        print(f"数据加载成功！")
        print(f"F_set数据维度: {F_set.shape}")
    except Exception as e:
        print(f"数据加载失败: {e}")
        return
    
    print("\n== 将钙离子时间序列转换为相空间流形 ==")
    print("相空间重构是分析动态系统的一种方法，可以揭示时间序列中的非线性动力学特性")
    
    # 获取数据尺寸：细胞数量和时间点
    cell_num, timeline = F_set.shape
    print(f"细胞数量: {cell_num}, 时间点数量: {timeline}")
    
    # 存储标准化的时间序列和相空间坐标
    trace_zs_set = [[None for _ in range(timeline)] for _ in range(cell_num)]
    xyz = [[None for _ in range(timeline)] for _ in range(cell_num)]
    
    # 使用进度条显示处理进度（处理前10个细胞，与MATLAB版本一致）
    total_iterations = min(10, cell_num) * timeline
    print(f"开始处理前{min(10, cell_num)}个细胞...")
    
    with tqdm(total=total_iterations, desc="相空间重构进度", 
              bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]') as pbar:
        
        for tt in range(timeline):
            for ii in range(min(10, cell_num)):  # 只处理前10个细胞（与MATLAB一致）
                try:
                    # 获取第ii个细胞在第tt个时间点的钙信号
                    dat = F_set[ii, tt]
                    
                    if dat is None:
                        pbar.update(1)
                        continue
                    
                    # 确保数据是一维数组
                    if hasattr(dat, 'flatten'):
                        dat = dat.flatten()
                    else:
                        dat = np.array(dat).flatten()
                    
                    if len(dat) == 0:
                        pbar.update(1)
                        continue
                    
                    # Z-score标准化
                    trace_zs = stats.zscore(dat)
                    
                    # 时间向量（虽然在后续处理中没有使用，但为了与MATLAB一致）
                    x = (1 / frame_rate) * np.linspace(1, len(trace_zs), len(trace_zs))
                    
                    # 保存标准化数据
                    trace_zs_set[ii][tt] = trace_zs
                    
                    # 计算互信息以确定最佳时间延迟tau
                    mi = mutual(trace_zs)  # 使用我们的mutual函数
                    
                    # 寻找互信息的第一个局部最小值
                    peaks, _ = find_peaks(-mi)
                    
                    if len(peaks) == 0:
                        mini = 8  # 如果没有找到局部最小值，使用经验值8
                    else:
                        mini = peaks[0]  # 使用第一个局部最小值
                    
                    # 设置相空间参数
                    dim = 3  # 嵌入维度(3D相空间)
                    tau = mini  # 使用互信息第一个局部最小值作为时间延迟
                    
                    # 进行相空间重构
                    try:
                        y = phasespace(trace_zs, dim, tau)  # 使用我们的phasespace函数
                        xyz[ii][tt] = y  # 保存相空间坐标
                    except ValueError as ve:
                        if "信号长度不足" in str(ve):
                            # 如果信号太短，跳过这个样本
                            xyz[ii][tt] = None
                        else:
                            raise ve
                    
                except Exception as e:
                    print(f"\n处理失败 (细胞{ii}, 时间{tt}): {e}")
                    xyz[ii][tt] = None
                
                pbar.update(1)
    
    print("\n裁剪相空间轨迹到统一长度，用于后续处理")
    
    # 裁剪相空间轨迹到统一长度
    xyz_len = 170  # 经验确定的统一长度
    xyz_trim = cellset2trim(xyz, xyz_len)  # 使用我们的cellset2trim函数
    print(f"轨迹统一长度: {xyz_len}")
    
    print("\n== 构建图数据集 ==")
    print("将相空间轨迹转换为图数据集格式，包括节点、边和图属性")
    
    # 将相空间轨迹转换为图数据集格式
    for_pred = []
    for row in xyz_trim:
        for item in row:
            if item is not None:
                for_pred.append(item)
    
    # 重塑为列向量（模拟MATLAB的reshape(xyz_trim, [], 1)）
    pred_num = len(for_pred)  # 样本总数
    print(f"有效图数据总数: {pred_num}")
    
    if pred_num == 0:
        print("错误: 没有有效的图数据，程序终止")
        return
    
    print("\n=== 生成nodes.csv文件 ===")
    print("构建节点ID和特征...")
    
    # 构建节点ID和特征
    # 图ID从1开始
    graph_id = list(range(1, pred_num + 1))
    graph_id = np.repeat(graph_id, xyz_len)  # 重复xyz_len次
    
    # 节点ID从1开始
    node_id = list(range(1, xyz_len + 1))
    node_id = np.tile(node_id, pred_num)  # 为每个图重复
    
    # 准备节点特征
    feat1 = np.vstack(for_pred)  # 将细胞数组转换为矩阵（模拟MATLAB的cell2mat）
    feat = []  # 初始化特征单元格数组
    
    print("正在格式化节点特征（请稍候...）")
    
    # 使用进度条处理特征格式化
    for ii in tqdm(range(len(feat1)), desc="特征格式化进度"):
        feat.append(formatConvert(feat1[ii, :]))  # 使用我们的formatConvert函数
    
    # 保存节点数据到CSV文件
    nodes_df = pd.DataFrame({
        'graph_id': graph_id,
        'node_id': node_id,
        'feat': feat
    })
    
    outfile = 'nodes.csv'
    nodes_file = os.path.join(out_path, outfile)
    nodes_df.to_csv(nodes_file, index=False)
    print(f"nodes.csv已保存: {nodes_file}")
    
    print("\n=== 生成edges.csv文件 ===")
    print("构建边的源节点和目标节点...")
    
    # 构建边的源节点和目标节点
    # 图ID从1开始
    edge_graph_id = list(range(1, pred_num + 1))
    edge_graph_id = np.repeat(edge_graph_id, xyz_len - 1)  # 重复(xyz_len-1)次
    
    # 源节点ID，从1到xyz_len-1
    src_id = list(range(1, xyz_len))
    src_id = np.tile(src_id, pred_num)  # 为每个图重复
    
    # 目标节点ID等于源节点ID加1，形成连续连接
    dst_id = src_id + 1
    
    # 边特征，这里设为1表示连接存在
    edge_feat = np.ones(len(dst_id), dtype=int)
    
    # 保存边数据到CSV文件
    edges_df = pd.DataFrame({
        'graph_id': edge_graph_id,
        'src_id': src_id,
        'dst_id': dst_id,
        'feat': edge_feat
    })
    
    outfile = 'edges.csv'
    edges_file = os.path.join(out_path, outfile)
    edges_df.to_csv(edges_file, index=False)
    print(f"edges.csv已保存: {edges_file}")
    
    print("\n=== 生成graphs.csv文件 ===")
    print("构建图的属性和标签...")
    
    # 构建图的属性和标签
    # 图ID从1开始
    graphs_graph_id = list(range(1, pred_num + 1))
    
    # 图特征和标签
    graph_feat = ['1,0,0,0,0,0'] * pred_num  # 图特征
    label = [0] * pred_num  # 初始标签为0
    
    # 保存图数据到CSV文件
    graphs_df = pd.DataFrame({
        'graph_id': graphs_graph_id,
        'feat': graph_feat,
        'label': label
    })
    
    outfile = 'graphs.csv'
    graphs_file = os.path.join(out_path, outfile)
    graphs_df.to_csv(graphs_file, index=False)
    print(f"graphs.csv已保存: {graphs_file}")
    
    print("\n=== 全部完成! ===")
    print(f"输出文件位置: {out_path}")
    print(f"- 节点文件: {nodes_file}")
    print(f"- 边文件: {edges_file}")
    print(f"- 图文件: {graphs_file}")
    
    # 显示统计信息
    print(f"\n处理统计:")
    print(f"- 处理的细胞数: {min(10, cell_num)}")
    print(f"- 时间点数: {timeline}")
    print(f"- 生成的图数量: {pred_num}")
    print(f"- 每个图的节点数: {xyz_len}")
    print(f"- 每个图的边数: {xyz_len - 1}")


if __name__ == "__main__":
    main() 