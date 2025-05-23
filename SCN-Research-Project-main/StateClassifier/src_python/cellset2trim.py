#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
细胞数组裁剪模块

该模块实现了与MATLAB cellset2trim.m等效的功能，用于将细胞数组中的每个
非空元素裁剪到指定长度，主要用于统一相空间轨迹的长度。

作者: SCN研究小组（Python版本）
日期: 2024
"""

import numpy as np
from typing import List, Union, Optional, Any


def cellset_trim(dataset: List[List[Optional[np.ndarray]]], trim_len: int) -> List[List[Optional[np.ndarray]]]:
    """
    细胞数组裁剪函数
    
    该函数将细胞数组中的每个非空元素裁剪到指定长度，
    主要用于统一相空间轨迹的长度，便于后续处理和分析。
    
    Parameters
    ----------
    dataset : List[List[Optional[np.ndarray]]]
        输入的二维列表，包含相空间轨迹数据。每个元素可以是numpy数组或None
    trim_len : int
        裁剪后的目标长度
        
    Returns
    -------
    List[List[Optional[np.ndarray]]]
        裁剪后的二维列表，每个非空元素都被裁剪到相同的长度
        
    Examples
    --------
    >>> import numpy as np
    >>> # 创建测试数据
    >>> data = [[np.random.rand(200, 3), np.random.rand(180, 3)],
    ...         [np.random.rand(250, 3), None]]
    >>> # 裁剪到170个点
    >>> trimmed_data = cellset_trim(data, 170)
    >>> print(trimmed_data[0][0].shape)  # (170, 3)
    """
    # 获取数据集的维度（细胞数量和时间线）
    cell_num = len(dataset)
    timeline = len(dataset[0]) if dataset and len(dataset) > 0 else 0
    
    # 初始化结果列表，与输入列表大小相同
    data_trim = [[None for _ in range(timeline)] for _ in range(cell_num)]
    
    # 遍历所有细胞和时间点
    for ii in range(cell_num):
        for jj in range(timeline):
            # 获取当前细胞和时间点的数据
            temp = dataset[ii][jj]
            
            # 如果数据非空且长度足够，则裁剪到指定长度
            if temp is not None and len(temp) >= trim_len:
                # 确保temp是numpy数组
                if not isinstance(temp, np.ndarray):
                    temp = np.array(temp)
                
                # 裁剪数据到指定长度
                data_trim[ii][jj] = temp[:trim_len]
            elif temp is not None and len(temp) < trim_len:
                # 如果数据长度不足，给出警告但仍保存原数据
                print(f"警告: 细胞({ii}, {jj})的数据长度({len(temp)})小于目标长度({trim_len})，保持原长度")
                data_trim[ii][jj] = temp if isinstance(temp, np.ndarray) else np.array(temp)
            else:
                # 如果数据为空，保持为None
                data_trim[ii][jj] = None
    
    return data_trim


# 为了兼容性，提供与MATLAB函数名一致的别名
def cellset2trim(dataset: List[List[Optional[np.ndarray]]], trim_len: int) -> List[List[Optional[np.ndarray]]]:
    """
    与MATLAB cellset2trim函数完全兼容的接口
    
    这是cellset_trim函数的别名，提供与原MATLAB代码相同的调用方式。
    
    Parameters
    ----------
    dataset : List[List[Optional[np.ndarray]]]
        输入的二维列表，包含相空间轨迹数据
    trim_len : int
        裁剪后的目标长度
        
    Returns
    -------
    List[List[Optional[np.ndarray]]]
        裁剪后的二维列表，每个非空元素都被裁剪到相同的长度
    """
    return cellset_trim(dataset, trim_len) 