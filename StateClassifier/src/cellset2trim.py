#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
细胞数组裁剪模块

该模块提供细胞数组裁剪功能，主要用于统一相空间轨迹的长度，
便于神经图像处理中的后续处理和分析。

Author: Converted from MATLAB
"""

import numpy as np
from typing import List, Optional, Union, Any
import warnings


def cellset2trim(dataset: List[List[Optional[np.ndarray]]], 
                trim_len: int) -> List[List[Optional[np.ndarray]]]:
    """
    细胞数组裁剪函数
    
    该函数将细胞数组中的每个非空元素裁剪到指定长度，
    主要用于统一相空间轨迹的长度，便于后续处理和分析。
    
    Parameters
    ----------
    dataset : List[List[Optional[np.ndarray]]]
        输入的细胞数组，包含相空间轨迹数据
        每个元素可以是numpy数组或None
    trim_len : int
        裁剪后的目标长度
        
    Returns
    -------
    List[List[Optional[np.ndarray]]]
        裁剪后的细胞数组，每个非空元素都被裁剪到相同的长度
        
    Raises
    ------
    ValueError
        当trim_len小于等于0时抛出异常
        
    Examples
    --------
    >>> import numpy as np
    >>> # 创建测试数据
    >>> data1 = np.random.randn(100, 3)
    >>> data2 = np.random.randn(150, 3)
    >>> dataset = [[data1, None], [data2, data1]]
    >>> trimmed = cellset2trim(dataset, trim_len=50)
    >>> print(f"裁剪后形状: {trimmed[0][0].shape}")
    (50, 3)
    """
    # 参数验证
    if trim_len <= 0:
        raise ValueError("裁剪长度必须大于0")
    
    if not dataset:
        return dataset
    
    # 获取数据集的维度（细胞数量和时间线）
    cell_num = len(dataset)
    timeline = len(dataset[0]) if dataset else 0
    
    # 初始化结果细胞数组，与输入数组大小相同
    data_trim = [[None for _ in range(timeline)] for _ in range(cell_num)]
    
    # 遍历所有细胞和时间点
    for ii in range(cell_num):
        for jj in range(timeline):
            # 获取当前细胞和时间点的数据
            temp = dataset[ii][jj]
            
            # 如果数据非空，则裁剪到指定长度
            if temp is not None:
                # 确保输入为numpy数组
                if not isinstance(temp, np.ndarray):
                    temp = np.array(temp)
                
                # 检查数据长度是否足够
                if len(temp) < trim_len:
                    warnings.warn(f"位置[{ii},{jj}]的数据长度({len(temp)})小于裁剪长度({trim_len})，将保持原样")
                    data_trim[ii][jj] = temp.copy()
                else:
                    # 裁剪到指定长度
                    data_trim[ii][jj] = temp[:trim_len, :].copy()
    
    return data_trim


def cellset2trim_dict(dataset: dict, 
                     trim_len: int) -> dict:
    """
    字典格式的细胞数组裁剪函数
    
    该函数处理以字典形式存储的细胞数组数据
    
    Parameters
    ----------
    dataset : dict
        输入的字典数据，值为numpy数组或None
    trim_len : int
        裁剪后的目标长度
        
    Returns
    -------
    dict
        裁剪后的字典数据
        
    Examples
    --------
    >>> import numpy as np
    >>> dataset = {
    ...     'cell1_t1': np.random.randn(100, 3),
    ...     'cell1_t2': None,
    ...     'cell2_t1': np.random.randn(80, 3)
    ... }
    >>> trimmed = cellset2trim_dict(dataset, trim_len=50)
    """
    # 参数验证
    if trim_len <= 0:
        raise ValueError("裁剪长度必须大于0")
    
    data_trim = {}
    
    for key, data in dataset.items():
        if data is not None:
            # 确保输入为numpy数组
            if not isinstance(data, np.ndarray):
                data = np.array(data)
            
            # 检查数据长度是否足够
            if len(data) < trim_len:
                warnings.warn(f"键'{key}'的数据长度({len(data)})小于裁剪长度({trim_len})，将保持原样")
                data_trim[key] = data.copy()
            else:
                # 裁剪到指定长度
                data_trim[key] = data[:trim_len, :].copy()
        else:
            data_trim[key] = None
    
    return data_trim


def get_dataset_stats(dataset: List[List[Optional[np.ndarray]]]) -> dict:
    """
    获取数据集的统计信息
    
    Parameters
    ----------
    dataset : List[List[Optional[np.ndarray]]]
        输入的细胞数组
        
    Returns
    -------
    dict
        包含统计信息的字典
    """
    stats = {
        'cell_count': len(dataset),
        'timeline_count': len(dataset[0]) if dataset else 0,
        'non_empty_count': 0,
        'empty_count': 0,
        'min_length': float('inf'),
        'max_length': 0,
        'avg_length': 0,
        'total_elements': 0
    }
    
    lengths = []
    
    for ii in range(len(dataset)):
        for jj in range(len(dataset[0]) if dataset else 0):
            data = dataset[ii][jj]
            if data is not None:
                stats['non_empty_count'] += 1
                length = len(data)
                lengths.append(length)
                stats['min_length'] = min(stats['min_length'], length)
                stats['max_length'] = max(stats['max_length'], length)
                stats['total_elements'] += length
            else:
                stats['empty_count'] += 1
    
    if lengths:
        stats['avg_length'] = np.mean(lengths)
        stats['std_length'] = np.std(lengths)
    else:
        stats['min_length'] = 0
        stats['avg_length'] = 0
        stats['std_length'] = 0
    
    return stats


def validate_trim_length(dataset: List[List[Optional[np.ndarray]]], 
                        trim_len: int) -> bool:
    """
    验证裁剪长度是否合理
    
    Parameters
    ----------
    dataset : List[List[Optional[np.ndarray]]]
        输入的细胞数组
    trim_len : int
        目标裁剪长度
        
    Returns
    -------
    bool
        如果裁剪长度合理返回True，否则返回False
    """
    stats = get_dataset_stats(dataset)
    
    if trim_len <= 0:
        print(f"错误: 裁剪长度必须大于0，当前值: {trim_len}")
        return False
    
    if stats['non_empty_count'] == 0:
        print("警告: 数据集中没有非空元素")
        return True
    
    if trim_len > stats['min_length']:
        print(f"警告: 裁剪长度({trim_len})大于最小数据长度({stats['min_length']})")
        return False
    
    return True


if __name__ == "__main__":
    # 测试示例
    import numpy as np
    
    print("创建测试数据集...")
    
    # 创建模拟的相空间轨迹数据
    np.random.seed(42)
    
    # 生成不同长度的轨迹数据
    data1 = np.random.randn(100, 3)  # 100个时间点，3维相空间
    data2 = np.random.randn(150, 3)  # 150个时间点，3维相空间
    data3 = np.random.randn(80, 3)   # 80个时间点，3维相空间
    data4 = np.random.randn(120, 3)  # 120个时间点，3维相空间
    
    # 创建细胞数组（模拟多个细胞在不同时间线的数据）
    dataset = [
        [data1, None, data3],      # 细胞1在3个时间线的数据
        [data2, data4, None],      # 细胞2在3个时间线的数据
        [None, data1, data2]       # 细胞3在3个时间线的数据
    ]
    
    print("原始数据集统计:")
    stats = get_dataset_stats(dataset)
    for key, value in stats.items():
        print(f"  {key}: {value}")
    
    # 设定裁剪长度
    trim_length = 75
    
    # 验证裁剪长度
    if validate_trim_length(dataset, trim_length):
        print(f"\n开始裁剪到长度: {trim_length}")
        
        # 执行裁剪
        trimmed_dataset = cellset2trim(dataset, trim_length)
        
        print("裁剪后数据集统计:")
        trimmed_stats = get_dataset_stats(trimmed_dataset)
        for key, value in trimmed_stats.items():
            print(f"  {key}: {value}")
        
        # 验证裁剪结果
        print("\n验证裁剪结果:")
        for ii in range(len(trimmed_dataset)):
            for jj in range(len(trimmed_dataset[0])):
                data = trimmed_dataset[ii][jj]
                if data is not None:
                    print(f"  位置[{ii},{jj}]: 形状 {data.shape}")
                else:
                    print(f"  位置[{ii},{jj}]: None")
    
    else:
        print("裁剪长度验证失败！")
    
    # 测试字典格式
    print("\n\n测试字典格式:")
    dict_dataset = {
        'cell1_t1': np.random.randn(100, 3),
        'cell1_t2': None,
        'cell2_t1': np.random.randn(80, 3),
        'cell2_t2': np.random.randn(120, 3)
    }
    
    trimmed_dict = cellset2trim_dict(dict_dataset, 60)
    
    print("字典格式裁剪结果:")
    for key, data in trimmed_dict.items():
        if data is not None:
            print(f"  {key}: 形状 {data.shape}")
        else:
            print(f"  {key}: None") 