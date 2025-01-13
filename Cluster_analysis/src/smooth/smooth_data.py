#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
神经信号数据处理模块

该模块提供了一系列用于处理神经信号数据的函数，包括：
- 移动平均滤波
- Butterworth低通滤波
- 数据标准化
"""

import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt
from sklearn.preprocessing import StandardScaler
from typing import List, Optional, Union
import logging

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def moving_average(data: Union[pd.Series, np.ndarray], window_size: int = 3) -> pd.Series:
    """
    应用移动平均滤波平滑数据。

    参数:
        data: 输入的时间序列数据
        window_size: 滑动窗口大小，必须为奇数（默认为3）

    返回:
        平滑后的数据序列
    """
    if window_size % 2 == 0:
        logging.warning("窗口大小应为奇数，已自动加1")
        window_size += 1
    return pd.Series(data).rolling(window=window_size, center=True).mean().bfill().ffill()

def butterworth_filter(
    data: np.ndarray,
    cutoff_freq: float = 20,
    fs: float = 10.0,
    order: int = 2,
    strength: float = 0.05
) -> np.ndarray:
    """
    应用Butterworth低通滤波器去除高频噪声。

    参数:
        data: 输入信号数据
        cutoff_freq: 截止频率，值越小滤波效果越强（默认0.1）
        fs: 采样频率（默认10.0）
        order: 滤波器阶数，阶数越高滤波效果越陡峭（默认4）
        strength: 滤波强度系数，范围0-1，值越大滤波效果越强（默认0.5）

    返回:
        滤波后的数据
    """
    nyquist = fs * 0.5
    normal_cutoff = (cutoff_freq * strength) / nyquist
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return filtfilt(b, a, data)

def check_data_distribution(data: pd.DataFrame, columns: List[str], stage: str = ""):
    """
    检查数据分布情况。

    参数:
        data: 输入的DataFrame
        columns: 要检查的列名列表
        stage: 处理阶段说明
    """
    stats = {
        'mean': data[columns].mean().mean(),
        'std': data[columns].std().mean(),
        'min': data[columns].min().min(),
        'max': data[columns].max().max(),
        'range': data[columns].max().max() - data[columns].min().min()
    }
    
    logging.info(f"{stage} 数据分布统计:")
    for metric, value in stats.items():
        logging.info(f"- {metric}: {value:.4f}")

def process_signal(
    data: pd.DataFrame,
    moving_avg_window: int = 3,
    apply_butterworth: bool = False,
    apply_normalization: bool = True
) -> pd.DataFrame:
    """
    处理神经信号数据，仅处理n1到n62的神经元列。

    参数:
        data: 输入的DataFrame，包含神经元数据
        moving_avg_window: 移动平均窗口大小（默认3）
        apply_butterworth: 是否应用Butterworth滤波（默认True）
        apply_normalization: 是否应用Z-score归一化（默认True）

    返回:
        处理后的DataFrame
    """
    processed_data = data.copy()
    neuron_columns = [f'n{i}' for i in range(1, 63)]
    existing_neuron_cols = [col for col in neuron_columns if col in processed_data.columns]

    if not existing_neuron_cols:
        logging.warning("未找到任何神经元数据列(n1-n62)")
        return processed_data

    # 检查原始数据分布
    check_data_distribution(processed_data, existing_neuron_cols, "原始数据")

    for column in existing_neuron_cols:
        # 步骤1：移动平均滤波
        signal = moving_average(processed_data[column], window_size=moving_avg_window)
        
        # 步骤2：Butterworth滤波
        if apply_butterworth:
            signal = butterworth_filter(signal)
            
        processed_data[column] = signal
    
    # 检查滤波后的数据分布
    check_data_distribution(processed_data, existing_neuron_cols, "滤波后数据")
    
    # 步骤3：Z-score归一化
    if apply_normalization:
        scaler = StandardScaler()
        processed_data[existing_neuron_cols] = scaler.fit_transform(processed_data[existing_neuron_cols])
        # 检查归一化后的数据分布
        check_data_distribution(processed_data, existing_neuron_cols, "归一化后数据")
    
    return processed_data

def main():
    """主函数入口"""
    # 配置参数
    input_file = '../../datasets/Day6_with_behavior_labels_filled.xlsx'
    output_file = '../../datasets/processed_Day6.xlsx'
    params = {
        'moving_avg_window': 3,
        'apply_butterworth': True,
        'apply_normalization': True
    }

    try:
        # 加载数据
        logging.info(f"正在读取数据: {input_file}")
        data = pd.read_excel(input_file)
        
        # 处理数据
        logging.info("正在处理数据...")
        processed_data = process_signal(data, **params)
        
        # 保存结果
        logging.info(f"正在保存处理后的数据: {output_file}")
        processed_data.to_excel(output_file, index=False)
        
        logging.info("数据处理完成！")
        
    except Exception as e:
        logging.error(f"处理过程中出现错误: {str(e)}")
        raise

if __name__ == '__main__':
    main()
