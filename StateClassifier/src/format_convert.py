#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
数值数组格式转换模块

该模块提供将数值数组转换为CSV格式字符串的功能，
主要用于神经图像处理中的数据格式化输出。

Author: Converted from MATLAB
"""

from typing import Union, List
import numpy as np


def format_convert(x: Union[List[float], np.ndarray]) -> str:
    """
    数值数组转字符串格式化函数
    
    该函数将数值数组转换为以逗号分隔的字符串格式，
    主要用于将节点特征数据转换为CSV文件可接受的字符串格式。
    
    Parameters
    ----------
    x : Union[List[float], np.ndarray]
        数值数组，包含要转换的数据
        
    Returns
    -------
    str
        转换后的字符串，以逗号分隔各元素
        
    Examples
    --------
    >>> data = [1.2, 3.4, 5.6]
    >>> result = format_convert(data)
    >>> print(result)
    '1.2,3.4,5.6'
    """
    # 确保输入为numpy数组格式
    if not isinstance(x, np.ndarray):
        x = np.array(x)
    
    # 将数组元素转换为字符串并用逗号连接
    return ','.join(str(element) for element in x.flatten())


if __name__ == "__main__":
    # 测试示例
    test_data = [1.23, 4.56, 7.89, 0.12]
    result = format_convert(test_data)
    print(f"原始数据: {test_data}")
    print(f"转换结果: {result}") 