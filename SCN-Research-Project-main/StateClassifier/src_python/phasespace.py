#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
相空间重构模块

该模块实现了与MATLAB phasespace.m等效的功能，用于将一维时间序列重构为
高维相空间轨迹，用于非线性动力学分析。

基于Hui Yang的原始MATLAB实现：
- 宾夕法尼亚州立大学
- Email: yanghui@gmail.com

参考文献:
[1] H. Yang, Multiscale Recurrence Quantification Analysis of Spatial Vectorcardiogram (VCG) 
    Signals, IEEE Transactions on Biomedical Engineering, Vol. 58, No. 2, p339-347, 2011
[2] Y. Chen and H. Yang, "Multiscale recurrence analysis of long-term nonlinear and 
    nonstationary time series," Chaos, Solitons and Fractals, Vol. 45, No. 7, p978-987, 2012

作者: SCN研究小组（Python版本）
日期: 2024
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from typing import Optional


def phase_space_reconstruction(signal: np.ndarray, dim: int, tau: int, plot: bool = False) -> np.ndarray:
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
    plot : bool, optional
        是否绘制相空间轨迹图，默认为False
        
    Returns
    -------
    np.ndarray
        重构的相空间轨迹矩阵，大小为 T×dim，每行代表相空间中的一个点
    """
    signal = np.array(signal).flatten()
    N = len(signal)
    
    # 计算相空间中的总点数
    # 考虑时间延迟和嵌入维度，相空间中的点数会减少
    T = N - (dim - 1) * tau
    
    if T <= 0:
        raise ValueError(f"信号长度不足以进行相空间重构。需要至少 {(dim-1)*tau + 1} 个点，当前信号长度为 {N}")
    
    # 初始化相空间矩阵
    Y = np.zeros((T, dim))
    
    # 构建相空间轨迹
    # 对每个时间点，基于时间延迟tau和嵌入维度dim构建相空间中的对应点
    for i in range(T):
        # 使用降序排列的延迟索引构建相空间点
        # 每个点由当前时刻及其过去的值组成
        indices = i + (dim - 1) * tau - np.arange(dim) * tau
        Y[i, :] = signal[indices]
    
    # 获取相空间维度
    size_Y = Y.shape[1]
    
    # 如果需要绘图且没有返回值（模拟MATLAB的nargout == 0）
    if plot:
        if size_Y == 2:
            # 2D相空间可视化
            plt.figure(figsize=(10, 8))
            plt.plot(Y[:, 0], Y[:, 1])
            plt.xlabel('y1', fontsize=12, fontweight='bold')
            plt.ylabel('y2', fontsize=12, fontweight='bold')
            plt.title('2D相空间轨迹', fontsize=14, fontweight='bold')
            plt.grid(True)
            plt.show()
        else:
            # 3D相空间可视化
            fig = plt.figure(figsize=(12, 10))
            ax = fig.add_subplot(111, projection='3d')
            ax.plot(Y[:, 0], Y[:, 1], Y[:, 2])
            ax.set_xlabel('y1', fontsize=12, fontweight='bold')
            ax.set_ylabel('y2', fontsize=12, fontweight='bold')
            ax.set_zlabel('y3', fontsize=12, fontweight='bold')
            ax.set_title('3D相空间轨迹', fontsize=14, fontweight='bold')
            plt.show()
    
    return Y


# 为了兼容性，提供与MATLAB函数名一致的别名
def phasespace(signal: np.ndarray, dim: int, tau: int) -> np.ndarray:
    """
    与MATLAB phasespace函数完全兼容的接口
    
    这是phase_space_reconstruction函数的别名，提供与原MATLAB代码相同的调用方式。
    
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
    return phase_space_reconstruction(signal, dim, tau, plot=False) 