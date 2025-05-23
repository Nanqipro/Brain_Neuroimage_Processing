#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
相空间重构模块

该模块将一维时间序列重构为高维相空间轨迹，用于非线性动力学分析。
利用时间延迟嵌入方法，将一维信号映射到多维空间，以揭示系统的动力学特性。

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
from mpl_toolkits.mplot3d import Axes3D
from typing import Union, List, Tuple
import warnings


def phasespace(signal: Union[List[float], np.ndarray], 
              dim: int, 
              tau: int,
              plot_result: bool = False) -> np.ndarray:
    """
    相空间重构函数
    
    该函数将一维时间序列重构为高维相空间轨迹，用于非线性动力学分析。
    利用时间延迟嵌入方法，将一维信号映射到多维空间，以揭示系统的动力学特性。
    
    Parameters
    ----------
    signal : Union[List[float], np.ndarray]
        输入时间序列，一维数组
    dim : int
        嵌入维度，表示重构相空间的维数
    tau : int
        时间延迟，用于确定相空间点的构建方式
    plot_result : bool, default=False
        是否绘制相空间轨迹图
        
    Returns
    -------
    np.ndarray
        重构的相空间轨迹矩阵，大小为 T×dim，每行代表相空间中的一个点
        
    Examples
    --------
    >>> import numpy as np
    >>> t = np.linspace(0, 4*np.pi, 1000)
    >>> signal = np.sin(t)
    >>> Y = phasespace(signal, dim=3, tau=10)
    >>> print(f"相空间轨迹形状: {Y.shape}")
    """
    # 确保输入为numpy数组
    if not isinstance(signal, np.ndarray):
        signal = np.array(signal)
    
    # 参数验证
    if dim < 1:
        raise ValueError("嵌入维度必须大于等于1")
    if tau < 1:
        raise ValueError("时间延迟必须大于等于1")
    
    # 获取信号长度
    N = len(signal)
    
    # 计算相空间中的总点数
    # 考虑时间延迟和嵌入维度，相空间中的点数会减少
    T = N - (dim - 1) * tau
    
    if T <= 0:
        raise ValueError(f"信号长度不足以构建相空间。需要至少 {(dim-1)*tau + 1} 个数据点")
    
    # 初始化相空间矩阵
    Y = np.zeros((T, dim))
    
    # 构建相空间轨迹
    # 对每个时间点，基于时间延迟tau和嵌入维度dim构建相空间中的对应点
    for i in range(T):
        # 使用降序排列的延迟索引构建相空间点
        # 每个点由当前时刻及其过去的值组成
        delay_indices = np.arange(dim)[::-1] * tau  # 降序延迟索引
        Y[i, :] = signal[i + (dim-1)*tau - delay_indices]
    
    # 如果需要绘图
    if plot_result:
        _plot_phasespace(Y)
    
    return Y


def _plot_phasespace(Y: np.ndarray) -> None:
    """
    绘制相空间轨迹图的内部函数
    
    Parameters
    ----------
    Y : np.ndarray
        相空间轨迹矩阵
    """
    dim = Y.shape[1]
    
    plt.figure(figsize=(10, 8))
    
    if dim == 2:
        # 2D相空间可视化
        plt.plot(Y[:, 0], Y[:, 1], 'b-', linewidth=0.8, alpha=0.7)
        plt.scatter(Y[0, 0], Y[0, 1], color='green', s=50, label='起点', zorder=5)
        plt.scatter(Y[-1, 0], Y[-1, 1], color='red', s=50, label='终点', zorder=5)
        plt.xlabel('y1', fontsize=12, fontweight='bold')
        plt.ylabel('y2', fontsize=12, fontweight='bold')
        plt.title('2D相空间重构', fontsize=14, fontweight='bold')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
    elif dim >= 3:
        # 3D相空间可视化
        ax = plt.gca(projection='3d')
        ax.plot(Y[:, 0], Y[:, 1], Y[:, 2], 'b-', linewidth=0.8, alpha=0.7)
        ax.scatter(Y[0, 0], Y[0, 1], Y[0, 2], color='green', s=50, label='起点')
        ax.scatter(Y[-1, 0], Y[-1, 1], Y[-1, 2], color='red', s=50, label='终点')
        ax.set_xlabel('y1', fontsize=12, fontweight='bold')
        ax.set_ylabel('y2', fontsize=12, fontweight='bold')
        ax.set_zlabel('y3', fontsize=12, fontweight='bold')
        ax.set_title('3D相空间重构', fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(True)
        
    else:
        # 1D情况，绘制时间序列
        plt.plot(Y[:, 0], 'b-', linewidth=1)
        plt.xlabel('时间索引', fontsize=12, fontweight='bold')
        plt.ylabel('信号值', fontsize=12, fontweight='bold')
        plt.title('1D时间序列', fontsize=14, fontweight='bold')
        plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()


def estimate_embedding_params(signal: Union[List[float], np.ndarray],
                            max_dim: int = 10,
                            max_tau: int = 20) -> Tuple[int, int]:
    """
    估计最佳嵌入参数（维度和时间延迟）
    
    使用互信息方法估计最佳时间延迟，使用虚假最近邻方法估计最佳嵌入维度
    
    Parameters
    ----------
    signal : Union[List[float], np.ndarray]
        输入时间序列
    max_dim : int, default=10
        最大搜索维度
    max_tau : int, default=20
        最大搜索时间延迟
        
    Returns
    -------
    Tuple[int, int]
        (最佳嵌入维度, 最佳时间延迟)
    """
    # 导入mutual模块来计算最佳延迟
    try:
        from .mutual import find_optimal_delay
    except ImportError:
        # 如果无法导入，使用简单的自相关方法
        optimal_tau = _estimate_tau_autocorr(signal, max_tau)
    else:
        optimal_tau = find_optimal_delay(signal, max_tau=max_tau)
    
    # 使用简化的维度估计方法
    optimal_dim = _estimate_dim_simple(signal, optimal_tau, max_dim)
    
    return optimal_dim, optimal_tau


def _estimate_tau_autocorr(signal: np.ndarray, max_tau: int) -> int:
    """
    使用自相关函数估计时间延迟
    
    Parameters
    ----------
    signal : np.ndarray
        输入信号
    max_tau : int
        最大搜索延迟
        
    Returns
    -------
    int
        估计的最佳时间延迟
    """
    # 计算自相关函数
    autocorr = np.correlate(signal, signal, mode='full')
    autocorr = autocorr[autocorr.size // 2:]
    autocorr = autocorr / autocorr[0]  # 归一化
    
    # 寻找第一个零交叉点
    for tau in range(1, min(max_tau, len(autocorr))):
        if autocorr[tau] <= 1/np.e:  # 约等于自相关时间
            return tau
    
    return max_tau // 2  # 默认返回


def _estimate_dim_simple(signal: np.ndarray, tau: int, max_dim: int) -> int:
    """
    简化的嵌入维度估计
    
    Parameters
    ----------
    signal : np.ndarray
        输入信号
    tau : int
        时间延迟
    max_dim : int
        最大搜索维度
        
    Returns
    -------
    int
        估计的最佳嵌入维度
    """
    # 简单启发式：基于信号复杂度估计
    if len(signal) < 100:
        return 2
    elif len(signal) < 500:
        return 3
    else:
        return min(max_dim, 5)  # 大多数情况下，3-5维就足够了


if __name__ == "__main__":
    # 测试示例
    # 生成洛伦兹吸引子数据
    def lorenz_attractor(dt: float = 0.01, num_steps: int = 5000) -> np.ndarray:
        """生成洛伦兹吸引子时间序列"""
        # 洛伦兹系统参数
        sigma, rho, beta = 10.0, 28.0, 8.0/3.0
        
        # 初始条件
        x, y, z = 1.0, 1.0, 1.0
        
        # 存储轨迹
        trajectory = np.zeros((num_steps, 3))
        
        for i in range(num_steps):
            # 洛伦兹方程
            dx = sigma * (y - x)
            dy = x * (rho - z) - y
            dz = x * y - beta * z
            
            # 更新状态
            x += dx * dt
            y += dy * dt
            z += dz * dt
            
            trajectory[i] = [x, y, z]
        
        return trajectory[:, 0]  # 返回x分量
    
    # 生成测试数据
    print("生成洛伦兹吸引子数据...")
    test_signal = lorenz_attractor()
    
    # 估计最佳参数
    print("估计最佳嵌入参数...")
    optimal_dim, optimal_tau = estimate_embedding_params(test_signal)
    print(f"估计的最佳维度: {optimal_dim}")
    print(f"估计的最佳延迟: {optimal_tau}")
    
    # 进行相空间重构
    print("进行相空间重构...")
    Y = phasespace(test_signal, dim=3, tau=optimal_tau, plot_result=True)
    
    print(f"相空间重构完成！")
    print(f"原始信号长度: {len(test_signal)}")
    print(f"相空间轨迹形状: {Y.shape}")
    print(f"相空间维度: {Y.shape[1]}")
    print(f"轨迹点数: {Y.shape[0]}") 