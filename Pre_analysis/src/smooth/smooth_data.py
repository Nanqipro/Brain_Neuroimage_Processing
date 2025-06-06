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
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from typing import List, Optional, Union, Literal
import logging
import matplotlib.pyplot as plt

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

def normalize_data(
    data: pd.DataFrame,
    columns: List[str],
    method: Literal['standard', 'minmax', 'robust', 'log_standard', 'log_minmax'] = 'standard',
    feature_range: tuple = (0, 1)
) -> pd.DataFrame:
    """
    使用不同方法对数据进行归一化。

    参数:
        data: 输入的DataFrame
        columns: 要归一化的列名列表
        method: 归一化方法，可选：
               'standard' - 标准化(Z-score)
               'minmax' - 最小最大值归一化
               'robust' - 稳健归一化（基于分位数）
               'log_standard' - 对数变换后的标准化
               'log_minmax' - 对数变换后的最小最大值归一化
        feature_range: 用于minmax归一化的目标范围

    返回:
        归一化后的DataFrame
    """
    df = data.copy()
    
    # 对数变换预处理
    if method.startswith('log_'):
        # 将数据平移到正数区间
        min_vals = df[columns].min()
        shift = abs(min_vals.min()) + 1 if min_vals.min() <= 0 else 0
        df[columns] = df[columns] + shift
        # 应用对数变换
        df[columns] = np.log1p(df[columns])
    
    # 选择归一化方法
    if method in ['standard', 'log_standard']:
        scaler = StandardScaler()
    elif method in ['minmax', 'log_minmax']:
        scaler = MinMaxScaler(feature_range=feature_range)
    elif method == 'robust':
        scaler = RobustScaler(quantile_range=(25, 75))
    
    # 应用归一化
    df[columns] = scaler.fit_transform(df[columns])
    return df

def process_signal(
    data: pd.DataFrame,
    moving_avg_window: int = 3,
    apply_butterworth: bool = False,
    apply_normalization: bool = True,
    normalization_method: str = 'standard',
    feature_range: tuple = (0, 1)
) -> pd.DataFrame:
    """
    处理神经信号数据，仅处理n1到n62的神经元列。

    参数:
        data: 输入的DataFrame，包含神经元数据
        moving_avg_window: 移动平均窗口大小（默认3）
        apply_butterworth: 是否应用Butterworth滤波（默认True）
        apply_normalization: 是否应用归一化（默认True）
        normalization_method: 归一化方法（默认'standard'）
            可选：'standard', 'minmax', 'robust', 'log_standard', 'log_minmax'
        feature_range: 用于minmax归一化的目标范围（默认(0,1)）

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
    
    # 步骤3：数据归一化
    if apply_normalization:
        processed_data = normalize_data(
            processed_data,
            existing_neuron_cols,
            method=normalization_method,
            feature_range=feature_range
        )
        # 检查归一化后的数据分布
        check_data_distribution(processed_data, existing_neuron_cols, f"归一化后数据 (方法: {normalization_method})")
    
    return processed_data

def visualize_traces(
    df: pd.DataFrame,
    output_path: str,
    scaling_factor: float = 40,
    time_range: tuple = (0, 3000)
):
    """
    可视化神经元信号轨迹。

    参数:
        df: 包含神经元数据的DataFrame
        output_path: 图像保存路径
        scaling_factor: 信号放大系数（默认40）
        time_range: 时间范围元组 (start, end)（默认(0, 3000)）
    """
    # Set up the plot
    plt.figure(figsize=(50, 15))

    # Find behavior change points if 'behavior' column exists
    if 'behavior' in df.columns:
        # Get indices where behavior changes
        behavior_changes = df.index[df['behavior'] != df['behavior'].shift()].tolist()
        # Get corresponding stamp values and behaviors
        change_stamps = df.loc[behavior_changes, 'stamp']
        behaviors = df.loc[behavior_changes, 'behavior']

    # Plot each line at a different vertical level, with scaling applied
    for i in range(1, 64):
        column_name = f'n{i}'
        if column_name in df.columns:  # Only plot if the column exists
            plt.plot(df['stamp'], df[column_name] * scaling_factor + i * 50, label=column_name)

    # Set x-axis limits
    plt.xlim(time_range)

    # Add vertical lines at behavior changes if behavior column exists
    if 'behavior' in df.columns:
        for stamp, behavior in zip(change_stamps, behaviors):
            plt.axvline(x=stamp, color='red', linestyle='--', linewidth=0.8, alpha=0.5)
            # Add behavior label
            plt.text(stamp, plt.ylim()[1], str(behavior), rotation=90, verticalalignment='top')

    # Adding labels and title
    plt.xlabel('Stamp')
    plt.ylabel('Traces (n1 ~ n63)')
    plt.title('Traces with Increased Amplitude')

    # Adjust layout and save the plot
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    logging.info(f"图像已保存至: {output_path}")

def main():
    """主函数入口"""
    # 配置参数No.29790930NORhabitituteCellVedio22024-12-04202145trace
    input_file = '../../datasets/No.297920240925homecagefamilarmice.xlsx'
    output_file = '../../datasets/processed_297920240925homecagefamilarmice.xlsx'
    output_graph = '../../graph/smooth_traces_amplitude_297920240925homecagefamilarmice.png'
    
    params = {
        'moving_avg_window': 3,
        'apply_butterworth': True,
        'apply_normalization': True,
        'normalization_method': 'log_standard',  # 可选: 'standard', 'minmax', 'robust', 'log_standard', 'log_minmax'
        'feature_range': (0, 1)  # 仅用于minmax方法
    }

    try:
        # 加载数据
        logging.info(f"正在读取数据: {input_file}")
        data = pd.read_excel(input_file)
        
        # 处理数据
        logging.info(f"正在处理数据... (归一化方法: {params['normalization_method']})")
        processed_data = process_signal(data, **params)
        
        # 保存结果
        logging.info(f"正在保存处理后的数据: {output_file}")
        processed_data.to_excel(output_file, index=False)
        
        # 可视化结果
        logging.info("正在生成可视化图像...")
        visualize_traces(processed_data, output_graph)
        
        logging.info("数据处理完成！")
        
    except Exception as e:
        logging.error(f"处理过程中出现错误: {str(e)}")
        raise

if __name__ == '__main__':
    main()
