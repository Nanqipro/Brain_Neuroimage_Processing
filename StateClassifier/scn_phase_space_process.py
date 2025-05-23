#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
3D SCN数据处理流程 - Python版本

该脚本用于处理脑神经元钙成像数据，将时间序列转换为相空间流形，并构建图数据集
主要步骤包括：
1. 加载原始钙信号数据
2. 将钙成像时间序列转换为相空间流形(phase-space manifolds)
3. 生成图数据集的节点、边和图属性文件

作者: SCN研究小组 (原MATLAB版本)
转换: Python版本
日期: 2023
"""

import numpy as np
import pandas as pd
import scipy.io
from scipy.stats import zscore
from scipy.signal import find_peaks
import os
import sys
from pathlib import Path
from tqdm import tqdm
import warnings

# 导入本地模块
sys.path.append('./src')
from src.mutual import mutual, find_optimal_delay
from src.phasespace import phasespace
from src.cellset2trim import cellset2trim
from src.format_convert import format_convert

# 抑制警告
warnings.filterwarnings('ignore')


def load_scn_data(file_path: str) -> dict:
    """
    加载SCN数据文件
    
    Parameters
    ----------
    file_path : str
        MAT文件路径
        
    Returns
    -------
    dict
        加载的数据字典
    """
    try:
        data = scipy.io.loadmat(file_path)
        print(f"✓ 成功加载数据文件: {file_path}")
        return data
    except Exception as e:
        print(f"✗ 加载数据文件失败: {e}")
        sys.exit(1)


def find_first_local_minimum(mi_values: np.ndarray, default_value: int = 8) -> int:
    """
    寻找互信息的第一个局部最小值
    
    Parameters
    ----------
    mi_values : np.ndarray
        互信息值数组
    default_value : int, default=8
        如果找不到局部最小值时的默认值
        
    Returns
    -------
    int
        第一个局部最小值的索引
    """
    # 使用scipy的find_peaks来寻找负值的峰值（即原函数的谷值）
    peaks, _ = find_peaks(-mi_values)
    
    if len(peaks) > 0:
        return peaks[0]  # 返回第一个局部最小值
    else:
        return default_value  # 使用默认值


def process_calcium_signals_to_phasespace(F_set: list, frame_rate: float) -> tuple:
    """
    将钙离子时间序列转换为相空间流形
    
    Parameters
    ----------
    F_set : list
        细胞钙信号数据集，格式为 [cell_num, timeline]
    frame_rate : float
        帧率，单位Hz
        
    Returns
    -------
    tuple
        (trace_zs_set, xyz) - 标准化数据和相空间坐标
    """
    cell_num, timeline = len(F_set), len(F_set[0])
    print(f"数据维度: {cell_num} 个细胞, {timeline} 个时间线")
    
    # 初始化存储结构
    trace_zs_set = [[None for _ in range(timeline)] for _ in range(cell_num)]
    xyz = [[None for _ in range(timeline)] for _ in range(cell_num)]
    
    print("开始处理钙信号数据...")
    
    # 使用进度条显示处理进度
    total_operations = min(10, cell_num) * timeline  # 原脚本只处理前10个细胞
    pbar = tqdm(total=total_operations, desc="处理相空间重构")
    
    for tt in range(timeline):
        for ii in range(min(10, cell_num)):  # 只处理前10个细胞（保持与原脚本一致）
            # 获取第ii个细胞在第tt个时间点的钙信号
            dat = F_set[ii][tt]
            
            if dat is not None and len(dat) > 0:
                # Z-score标准化
                trace_zs = zscore(dat)
                
                # 时间向量（虽然在相空间重构中不直接使用，但保持与原脚本一致）
                x = np.linspace(1, len(trace_zs), len(trace_zs)) / frame_rate
                
                # 保存标准化数据
                trace_zs_set[ii][tt] = trace_zs
                
                # 计算互信息以确定最佳时间延迟tau
                mi = mutual(trace_zs)
                
                # 寻找互信息的第一个局部最小值
                mini = find_first_local_minimum(mi, default_value=8)
                
                # 设置相空间参数
                dim = 3  # 嵌入维度(3D相空间)
                tau = mini  # 使用互信息第一个局部最小值作为时间延迟
                
                # 进行相空间重构
                try:
                    y = phasespace(trace_zs, dim, tau)
                    xyz[ii][tt] = y
                except ValueError as e:
                    print(f"\n警告: 细胞{ii+1},时间线{tt+1} 相空间重构失败: {e}")
                    xyz[ii][tt] = None
            
            pbar.update(1)
    
    pbar.close()
    print("✓ 相空间重构完成")
    
    return trace_zs_set, xyz


def trim_phasespace_data(xyz: list, xyz_len: int = 170) -> list:
    """
    裁剪相空间轨迹到统一长度
    
    Parameters
    ----------
    xyz : list
        相空间坐标数据
    xyz_len : int, default=170
        统一的轨迹长度
        
    Returns
    -------
    list
        裁剪后的相空间数据
    """
    print(f"开始裁剪数据到统一长度: {xyz_len}")
    
    # 使用已转换的cellset2trim函数
    xyz_trim = cellset2trim(xyz, xyz_len)
    
    print("✓ 数据裁剪完成")
    return xyz_trim


def generate_nodes_csv(xyz_trim: list, output_path: str) -> None:
    """
    生成nodes.csv文件
    
    Parameters
    ----------
    xyz_trim : list
        裁剪后的相空间数据
    output_path : str
        输出目录路径
    """
    print("生成 nodes.csv 文件...")
    
    # 重塑数据为列向量
    forPred = [item for sublist in xyz_trim for item in sublist if item is not None]
    pred_num = len(forPred)
    
    if pred_num == 0:
        raise ValueError("没有有效的相空间数据用于生成节点文件")
    
    # 获取轨迹长度（假设所有非空轨迹长度相同）
    xyz_len = len(forPred[0]) if forPred[0] is not None else 170
    
    # 构建节点ID和特征
    graph_id = np.repeat(np.arange(1, pred_num + 1), xyz_len)
    node_id = np.tile(np.arange(1, xyz_len + 1), pred_num)
    
    # 准备节点特征
    feat1 = np.vstack(forPred)  # 将所有轨迹垂直堆叠
    
    # 进度条
    feat = []
    print("转换节点特征格式...")
    for i in tqdm(range(len(feat1)), desc="格式转换"):
        feat.append(format_convert(feat1[i, :]))
    
    # 创建DataFrame并保存
    df = pd.DataFrame({
        'graph_id': graph_id,
        'node_id': node_id,
        'feat': feat
    })
    
    output_file = os.path.join(output_path, 'nodes.csv')
    df.to_csv(output_file, index=False)
    print(f"✓ nodes.csv 已保存到: {output_file}")


def generate_edges_csv(pred_num: int, xyz_len: int, output_path: str) -> None:
    """
    生成edges.csv文件
    
    Parameters
    ----------
    pred_num : int
        图的数量
    xyz_len : int
        每个图中的节点数量
    output_path : str
        输出目录路径
    """
    print("生成 edges.csv 文件...")
    
    # 构建边的源节点和目标节点
    graph_id = np.repeat(np.arange(1, pred_num + 1), xyz_len - 1)
    src_id = np.tile(np.arange(1, xyz_len), pred_num)  # 源节点ID，从1到xyz_len-1
    dst_id = src_id + 1  # 目标节点ID等于源节点ID加1，形成连续连接
    feat = np.ones(len(dst_id))  # 边特征，设为1表示连接存在
    
    # 创建DataFrame并保存
    df = pd.DataFrame({
        'graph_id': graph_id,
        'src_id': src_id,
        'dst_id': dst_id,
        'feat': feat.astype(int)
    })
    
    output_file = os.path.join(output_path, 'edges.csv')
    df.to_csv(output_file, index=False)
    print(f"✓ edges.csv 已保存到: {output_file}")


def generate_graphs_csv(pred_num: int, output_path: str) -> None:
    """
    生成graphs.csv文件
    
    Parameters
    ----------
    pred_num : int
        图的数量
    output_path : str
        输出目录路径
    """
    print("生成 graphs.csv 文件...")
    
    # 构建图的属性和标签
    graph_id = np.arange(1, pred_num + 1)
    feat = ['1,0,0,0,0,0'] * pred_num  # 图特征
    label = np.zeros(pred_num, dtype=int)  # 初始标签为0
    
    # 创建DataFrame并保存
    df = pd.DataFrame({
        'graph_id': graph_id,
        'feat': feat,
        'label': label
    })
    
    output_file = os.path.join(output_path, 'graphs.csv')
    df.to_csv(output_file, index=False)
    print(f"✓ graphs.csv 已保存到: {output_file}")


def main():
    """
    主处理函数
    """
    print("=" * 60)
    print("3D SCN数据处理流程 - Python版本")
    print("=" * 60)
    
    # 配置参数
    file_path = '../SCNData/Dataset1_SCNProject.mat'  # 输入文件路径，请根据实际情况修改
    frame_rate = 0.67  # 帧率，单位Hz
    output_path = './data'  # 输出目录路径
    xyz_len = 170  # 经验确定的统一长度
    
    # 创建输出目录
    Path(output_path).mkdir(parents=True, exist_ok=True)
    print(f"输出目录: {output_path}")
    
    # 步骤1: 加载数据
    print("\n步骤1: 加载数据")
    print("-" * 30)
    
    if not os.path.exists(file_path):
        print(f"警告: 数据文件不存在: {file_path}")
        print("生成模拟数据进行演示...")
        
        # 生成模拟数据
        np.random.seed(42)
        cell_num, timeline = 10, 3
        F_set = []
        for i in range(cell_num):
            cell_data = []
            for j in range(timeline):
                # 生成模拟的钙信号数据
                signal_length = np.random.randint(200, 400)
                calcium_signal = np.sin(np.linspace(0, 4*np.pi, signal_length)) + \
                               0.1 * np.random.randn(signal_length)
                cell_data.append(calcium_signal)
            F_set.append(cell_data)
    else:
        data = load_scn_data(file_path)
        F_set = data['F_set']  # 假设数据中有F_set字段
    
    # 步骤2: 将钙离子时间序列转换为相空间流形
    print("\n步骤2: 相空间重构")
    print("-" * 30)
    
    trace_zs_set, xyz = process_calcium_signals_to_phasespace(F_set, frame_rate)
    
    # 步骤3: 裁剪相空间轨迹到统一长度
    print("\n步骤3: 数据裁剪")
    print("-" * 30)
    
    xyz_trim = trim_phasespace_data(xyz, xyz_len)
    
    # 步骤4: 构建图数据集
    print("\n步骤4: 构建图数据集")
    print("-" * 30)
    
    # 计算有效样本数量
    forPred = [item for sublist in xyz_trim for item in sublist if item is not None]
    pred_num = len(forPred)
    
    print(f"有效样本数量: {pred_num}")
    
    if pred_num == 0:
        print("✗ 没有有效的相空间数据，无法生成图数据集")
        return
    
    # 生成CSV文件
    try:
        generate_nodes_csv(xyz_trim, output_path)
        generate_edges_csv(pred_num, xyz_len, output_path)
        generate_graphs_csv(pred_num, output_path)
        
        print("\n" + "=" * 60)
        print("全部完成!")
        print("=" * 60)
        print(f"生成的文件:")
        print(f"  - {output_path}/nodes.csv")
        print(f"  - {output_path}/edges.csv") 
        print(f"  - {output_path}/graphs.csv")
        print(f"总共处理了 {pred_num} 个图样本")
        
    except Exception as e:
        print(f"✗ 生成CSV文件时出错: {e}")
        return


def validate_environment():
    """
    验证运行环境和依赖
    """
    required_modules = ['numpy', 'pandas', 'scipy', 'tqdm']
    missing_modules = []
    
    for module in required_modules:
        try:
            __import__(module)
        except ImportError:
            missing_modules.append(module)
    
    if missing_modules:
        print(f"✗ 缺少依赖模块: {missing_modules}")
        print("请安装缺少的模块:")
        print(f"pip install {' '.join(missing_modules)}")
        return False
    
    return True


if __name__ == "__main__":
    # 验证环境
    if not validate_environment():
        sys.exit(1)
    
    try:
        main()
    except KeyboardInterrupt:
        print("\n用户中断程序")
    except Exception as e:
        print(f"\n程序执行出错: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1) 