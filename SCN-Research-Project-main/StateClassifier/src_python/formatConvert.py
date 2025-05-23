#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
格式转换模块

该模块实现了与MATLAB formatConvert.m等效的功能，用于将数值数组转换为
以逗号分隔的字符串格式，主要用于将节点特征数据转换为CSV文件可接受的格式。

作者: SCN研究小组（Python版本）
日期: 2024
"""

import numpy as np
from typing import Union, List


def format_convert(x: Union[np.ndarray, List, float, int]) -> str:
    """
    数值数组转字符串格式化函数
    
    该函数将数值数组转换为以逗号分隔的字符串格式，
    主要用于将节点特征数据转换为CSV文件可接受的字符串格式。
    与MATLAB版本完全一致的行为。
    
    Parameters
    ----------
    x : Union[np.ndarray, List, float, int]
        包含要转换的数据的数值数组、列表或单个数值
        
    Returns
    -------
    str
        转换后的字符串，以逗号分隔各元素，不包含末尾逗号
        
    Examples
    --------
    >>> format_convert([1.5, 2.3, 3.7])
    '1.5,2.3,3.7'
    
    >>> format_convert(np.array([1, 2, 3]))
    '1,2,3'
    
    >>> format_convert(42)
    '42'
    """
    # 确保输入是numpy数组格式
    if isinstance(x, (int, float)):
        # 单个数值直接转换
        return str(x)
    elif isinstance(x, list):
        x = np.array(x)
    elif not isinstance(x, np.ndarray):
        x = np.array(x)
    
    # 将数组展平为一维
    x_flat = x.flatten()
    
    # 初始化空字符串（与MATLAB版本一致）
    result = ''
    
    # 遍历数组中的每个元素，将其转换为字符串并用逗号连接
    for i in range(len(x_flat)):
        result += str(x_flat[i])
        if i < len(x_flat) - 1:  # 不是最后一个元素时添加逗号
            result += ','
    
    return result


# 为了兼容性，提供与MATLAB函数名一致的别名
def formatConvert(x: Union[np.ndarray, List, float, int]) -> str:
    """
    与MATLAB formatConvert函数完全兼容的接口
    
    这是format_convert函数的别名，提供与原MATLAB代码相同的调用方式。
    
    Parameters
    ----------
    x : Union[np.ndarray, List, float, int]
        包含要转换的数据的数值数组、列表或单个数值
        
    Returns
    -------
    str
        转换后的字符串，以逗号分隔各元素
        
    Notes
    -----
    此函数完全复制了MATLAB版本的行为：
    1. 遍历数组中的每个元素
    2. 将每个元素转换为字符串
    3. 用逗号连接所有元素
    4. 确保最后不包含多余的逗号
    """
    return format_convert(x) 