#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
时间延迟互信息计算模块

该模块实现时间序列的时间延迟互信息计算，用于确定相空间重构的最佳时间延迟。
互信息是衡量两个变量之间相互依赖性的度量，第一个局部最小值通常被选为最佳时间延迟。

Author: Hui Yang (Original MATLAB), Converted to Python
Affiliation: 
    The Pennsylvania State University
    310 Leohard Building, University Park, PA
    Email: yanghui@gmail.com

References:
[1] H. Yang, Multiscale Recurrence Quantification Analysis of Spatial Vectorcardiogram (VCG) 
    Signals, IEEE Transactions on Biomedical Engineering, Vol. 58, No. 2, p339-347, 2011
    DOI: 10.1109/TBME.2010.2063704
[2] Y. Chen and H. Yang, "Multiscale recurrence analysis of long-term nonlinear and 
    nonstationary time series," Chaos, Solitons and Fractals, Vol. 45, No. 7, p978-987, 2012 
    DOI: 10.1016/j.chaos.2012.03.013
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Union, List, Optional
import warnings


def mutual(signal: Union[List[float], np.ndarray], 
          partitions: Optional[int] = None, 
          tau: Optional[int] = None,
          plot_result: bool = False) -> np.ndarray:
    """
    时间延迟互信息计算函数
    
    该函数计算时间序列的时间延迟互信息，用于确定相空间重构的最佳时间延迟。
    互信息是衡量两个变量之间相互依赖性的度量，第一个局部最小值通常被选为最佳时间延迟。
    
    Parameters
    ----------
    signal : Union[List[float], np.ndarray]
        输入时间序列，一维数组
    partitions : Optional[int], default=16
        分区框数，用于概率估计的离散化
    tau : Optional[int], default=20
        最大时间延迟
    plot_result : bool, default=False
        是否绘制互信息图
        
    Returns
    -------
    np.ndarray
        从延迟0到tau的互信息值数组
        
    Examples
    --------
    >>> import numpy as np
    >>> signal = np.sin(np.linspace(0, 10*np.pi, 1000))
    >>> mi_values = mutual(signal, partitions=16, tau=20)
    >>> print(f"互信息值: {mi_values}")
    """
    # 确保输入为numpy数组
    if not isinstance(signal, np.ndarray):
        signal = np.array(signal)
    
    # 参数默认值设置
    if partitions is None:
        partitions = 16  # 默认分区数为16
    if tau is None:
        tau = 20  # 默认最大延迟为20
    
    # 计算信号的基本统计量
    minimum = np.min(signal)
    maximum = np.max(signal)
    interval = maximum - minimum
    signal_len = len(signal)
    
    # 避免除零错误
    if interval == 0:
        warnings.warn("信号为常数，无法计算互信息")
        return np.zeros(tau + 1)
    
    # 将信号标准化到[0,1]区间
    normalized_signal = (signal - minimum) / interval
    
    # 将标准化信号离散化为整数值(1到partitions)
    array = np.zeros(signal_len, dtype=int)
    for i in range(signal_len):
        if normalized_signal[i] > 0:
            array[i] = int(np.ceil(normalized_signal[i] * partitions))
        else:
            array[i] = 1
    
    # 确保延迟不超过信号长度
    if tau >= signal_len:
        tau = signal_len - 1
        warnings.warn(f"延迟值过大，已调整为 {tau}")
    
    # 计算延迟从0到tau的互信息
    mi = np.zeros(tau + 1)
    for i in range(tau + 1):
        mi[i] = _make_cond_entropy(i, array, signal_len, partitions)
    
    # 如果需要绘图
    if plot_result:
        plt.figure(figsize=(10, 6))
        plt.plot(range(tau + 1), mi, 'o-', markersize=5)
        plt.title('互信息测试（寻找第一个局部最小值）', fontsize=12, fontweight='bold')
        plt.xlabel('延迟（采样时间）', fontsize=12, fontweight='bold')
        plt.ylabel('互信息值', fontsize=12, fontweight='bold')
        plt.grid(True)
        plt.show()
    
    return mi


def _make_cond_entropy(t: int, array: np.ndarray, signal_len: int, partitions: int) -> float:
    """
    计算条件熵的内部函数
    
    该内部函数计算给定时间延迟t的条件熵（互信息）
    
    Parameters
    ----------
    t : int
        时间延迟
    array : np.ndarray
        离散化后的信号数组
    signal_len : int
        信号长度
    partitions : int
        分区数量
        
    Returns
    -------
    float
        计算得到的互信息值
    """
    # 初始化联合概率和边缘概率的计数数组
    h2 = np.zeros((partitions, partitions), dtype=int)  # 联合概率的计数
    h1 = np.zeros(partitions, dtype=int)                # 第一个变量的边缘概率计数
    h11 = np.zeros(partitions, dtype=int)               # 第二个变量的边缘概率计数
    
    count = 0
    
    # 统计不同延迟时的联合出现频率和边缘频率
    for i in range(signal_len):
        if i > t:  # 确保有足够的延迟
            hii = array[i] - 1          # 当前时间点的值（转为0索引）
            hi = array[i - t] - 1       # 延迟t后的值（转为0索引）
            
            # 确保索引在有效范围内
            if 0 <= hi < partitions and 0 <= hii < partitions:
                h1[hi] += 1                 # 累加边缘频率
                h11[hii] += 1
                h2[hi, hii] += 1           # 累加联合频率
                count += 1
    
    # 避免除零错误
    if count == 0:
        return 0.0
    
    # 计算归一化因子
    norm = 1.0 / count
    cond_ent = 0.0
    
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


def find_optimal_delay(signal: Union[List[float], np.ndarray], 
                      partitions: Optional[int] = None,
                      max_tau: Optional[int] = None) -> int:
    """
    寻找最佳时间延迟
    
    寻找互信息函数的第一个局部最小值作为最佳时间延迟
    
    Parameters
    ----------
    signal : Union[List[float], np.ndarray]
        输入时间序列
    partitions : Optional[int], default=16
        分区框数
    max_tau : Optional[int], default=20
        最大搜索延迟
        
    Returns
    -------
    int
        最佳时间延迟值
    """
    mi_values = mutual(signal, partitions, max_tau)
    
    # 寻找第一个局部最小值
    for i in range(1, len(mi_values) - 1):
        if mi_values[i] < mi_values[i-1] and mi_values[i] < mi_values[i+1]:
            return i
    
    # 如果没有找到局部最小值，返回最小值的索引
    return np.argmin(mi_values)


if __name__ == "__main__":
    # 测试示例
    # 生成测试信号（混沌时间序列）
    np.random.seed(42)
    t = np.linspace(0, 20*np.pi, 1000)
    test_signal = np.sin(t) + 0.1 * np.random.randn(len(t))
    
    # 计算互信息
    mi_values = mutual(test_signal, partitions=16, tau=30, plot_result=True)
    
    # 找到最佳延迟
    optimal_delay = find_optimal_delay(test_signal)
    
    print(f"计算完成！")
    print(f"最佳时间延迟: {optimal_delay}")
    print(f"对应的互信息值: {mi_values[optimal_delay]:.4f}") 