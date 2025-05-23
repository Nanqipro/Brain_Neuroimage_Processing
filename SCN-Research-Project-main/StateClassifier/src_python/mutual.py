#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
时间延迟互信息计算模块

该模块实现了与MATLAB mutual.m等效的功能，用于计算时间序列的时间延迟互信息，
确定相空间重构的最佳时间延迟参数。

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
from typing import Optional
import matplotlib.pyplot as plt


def mutual_information(signal: np.ndarray, 
                      partitions: Optional[int] = None, 
                      tau: Optional[int] = None, 
                      plot: bool = False) -> np.ndarray:
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
    tau : int, optional
        最大时间延迟，默认为20
    plot : bool, optional
        是否绘制互信息图，默认为False
        
    Returns
    -------
    np.ndarray
        从延迟0到tau的互信息值数组
    """
    # 参数默认值设置（与MATLAB版本一致）
    if partitions is None:
        partitions = 16
    if tau is None:
        tau = 20
    
    signal = np.array(signal).flatten()
    length = len(signal)
    
    # 计算信号的基本统计量
    av = np.mean(signal)
    variance = np.var(signal)
    minimum = np.min(signal)
    maximum = np.max(signal)
    interval = maximum - minimum
    
    # 确保延迟不超过信号长度
    if tau >= length:
        tau = length - 1
    
    # 将信号标准化到[0,1]区间
    if interval == 0:
        return np.zeros(tau + 1)
    
    signal_norm = (signal - minimum) / interval
    
    # 将标准化信号离散化为整数值(1到partitions)
    array = np.zeros(length, dtype=int)
    for i in range(length):
        if signal_norm[i] > 0:
            array[i] = int(np.ceil(signal_norm[i] * partitions))
        else:
            array[i] = 1
    
    # 计算延迟为0时的香农熵(作为基准)
    shannon = _make_cond_entropy(0, array, length, partitions)
    
    # 计算延迟从0到tau的互信息
    mi = np.zeros(tau + 1)
    for i in range(tau + 1):
        mi[i] = _make_cond_entropy(i, array, length, partitions)
    
    # 如果需要绘图且没有返回值（模拟MATLAB的nargout == 0）
    if plot:
        plt.figure(figsize=(10, 8))
        plt.plot(range(tau + 1), mi, 'o-', markersize=5)
        plt.title('互信息测试（寻找第一个局部最小值）', fontsize=12, fontweight='bold')
        plt.xlabel('延迟（采样时间）', fontsize=12, fontweight='bold')
        plt.ylabel('互信息值', fontsize=12, fontweight='bold')
        plt.grid(True)
        plt.show()
    
    return mi


def _make_cond_entropy(t: int, array: np.ndarray, length: int, partitions: int) -> float:
    """
    计算条件熵的内部函数
    
    该内部函数计算给定时间延迟t的条件熵（互信息）
    
    Parameters
    ----------
    t : int
        时间延迟
    array : np.ndarray
        离散化后的信号数组
    length : int
        信号长度
    partitions : int
        分区数量
        
    Returns
    -------
    float
        计算得到的互信息值
    """
    # 初始化变量
    count = 0
    cond_ent = 0.0
    
    # 初始化联合概率和边缘概率的计数数组
    h2 = np.zeros((partitions, partitions))  # 联合概率的计数
    h1 = np.zeros(partitions)                # 第一个变量的边缘概率计数
    h11 = np.zeros(partitions)               # 第二个变量的边缘概率计数
    
    # 统计不同延迟时的联合出现频率和边缘频率
    for i in range(length):
        if i > t:  # 确保有足够的延迟
            hii = array[i] - 1      # 当前时间点的值（转为0-based索引）
            hi = array[i - t] - 1   # 延迟t后的值（转为0-based索引）
            
            # 确保索引在有效范围内
            if 0 <= hi < partitions and 0 <= hii < partitions:
                h1[hi] += 1                    # 累加边缘频率
                h11[hii] += 1
                h2[hi, hii] += 1               # 累加联合频率
                count += 1
    
    if count == 0:
        return 0.0
    
    # 计算归一化因子
    norm = 1.0 / count
    
    # 计算互信息（基于条件熵）
    for i in range(partitions):
        hpi = h1[i] * norm  # 第一个变量的边缘概率
        if hpi > 0.0:
            for j in range(partitions):
                hpj = h11[j] * norm  # 第二个变量的边缘概率
                if hpj > 0.0:
                    pij = h2[i, j] * norm  # 联合概率
                    if pij > 0.0:
                        # 基于联合概率和边缘概率计算互信息
                        cond_ent += pij * np.log(pij / (hpj * hpi))
    
    return cond_ent


# 为了兼容性，提供与MATLAB函数名一致的别名
def mutual(signal: np.ndarray, 
          partitions: Optional[int] = None, 
          tau: Optional[int] = None) -> np.ndarray:
    """
    与MATLAB mutual函数完全兼容的接口
    
    这是mutual_information函数的别名，提供与原MATLAB代码相同的调用方式。
    """
    return mutual_information(signal, partitions, tau, plot=False) 