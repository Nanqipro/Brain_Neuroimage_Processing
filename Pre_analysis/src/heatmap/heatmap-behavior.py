"""
特定行为前后时间窗口的神经元活动热力图分析工具

功能：
- 检测指定行为的发生时间点
- 提取行为前后指定时间窗口的神经元活动数据
- 生成标准化热力图显示钙离子浓度变化
- 支持多个行为事件的批量分析
"""

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import argparse
import os
from typing import List, Tuple, Dict, Optional
from scipy.signal import find_peaks
from datetime import datetime

class BehaviorHeatmapConfig:
    """
    行为热力图配置类
    
    Attributes
    ----------
    INPUT_FILE : str
        输入数据文件路径
    OUTPUT_DIR : str
        输出目录路径
    TARGET_BEHAVIOR : str
        目标行为类型
    TIME_WINDOW : float
        时间窗口大小（秒）
    SAMPLING_RATE : float
        采样率（Hz）
    MIN_BEHAVIOR_DURATION : float
        最小行为持续时间（秒）
    """
    
    def __init__(self):
        # 输入文件路径
        self.INPUT_FILE = '../../datasets/Day3_with_behavior_labels_filled.xlsx'
        # 输出目录
        self.OUTPUT_DIR = '../../graph/behavior_heatmaps'
        # 目标行为类型
        self.TARGET_BEHAVIOR = 'CD1'
        # 时间窗口（秒）
        self.TIME_WINDOW = 10.0
        # 采样率（钙离子数据采样频率：4.8Hz）
        self.SAMPLING_RATE = 4.8
        # 最小行为持续时间（秒），用于过滤短暂的误标记
        self.MIN_BEHAVIOR_DURATION = 1.0
        # 热力图颜色范围
        self.VMIN = -3.0
        self.VMAX = 3.0

def parse_arguments() -> argparse.Namespace:
    """
    解析命令行参数
    
    Returns
    -------
    argparse.Namespace
        解析后的参数对象
    """
    parser = argparse.ArgumentParser(
        description='生成特定行为前后时间窗口的神经元活动热力图'
    )
    
    parser.add_argument(
        '--input', 
        type=str, 
        help='输入数据文件路径'
    )
    
    parser.add_argument(
        '--output-dir', 
        type=str, 
        help='输出目录路径'
    )
    
    parser.add_argument(
        '--behavior', 
        type=str, 
        default='CD1',
        help='目标行为类型（默认：CD1）'
    )
    
    parser.add_argument(
        '--time-window', 
        type=float, 
        default=10.0,
        help='时间窗口大小，秒（默认：10.0）'
    )
    
    parser.add_argument(
        '--min-duration', 
        type=float, 
        default=1.0,
        help='最小行为持续时间，秒（默认：1.0）'
    )
    
    parser.add_argument(
        '--sampling-rate', 
        type=float, 
        default=4.8,
        help='数据采样率，Hz（默认：4.8）'
    )
    
    return parser.parse_args()

def convert_timestamps_to_seconds(data: pd.DataFrame, sampling_rate: float) -> pd.DataFrame:
    """
    将时间戳转换为秒为单位的时间序列
    
    Parameters
    ----------
    data : pd.DataFrame
        原始数据，索引为时间戳
    sampling_rate : float
        采样率（Hz）
        
    Returns
    -------
    pd.DataFrame
        转换后的数据，索引为秒
    """
    # 检查时间戳是否为连续的数值序列
    if isinstance(data.index[0], (int, float)):
        # 计算时间间隔来判断是否需要转换
        time_intervals = np.diff(data.index[:10])  # 检查前10个时间间隔
        avg_interval = np.mean(time_intervals)
        
        # 如果平均时间间隔接近1/采样率，说明时间戳已经是秒
        expected_interval = 1.0 / sampling_rate
        
        if abs(avg_interval - expected_interval) < expected_interval * 0.1:
            # 时间戳已经是秒，直接返回
            print(f"检测到时间戳已为秒格式（平均间隔: {avg_interval:.3f}s）")
            return data
        else:
            # 时间戳可能是数据点索引，需要转换为秒
            print(f"检测到时间戳为数据点格式（平均间隔: {avg_interval:.1f}），转换为秒")
            time_seconds = data.index / sampling_rate
            data_converted = data.copy()
            data_converted.index = time_seconds
            return data_converted
    else:
        # 如果时间戳已经是时间格式，直接返回
        print("检测到时间戳为时间格式")
        return data

def load_and_validate_data(file_path: str) -> pd.DataFrame:
    """
    加载和验证数据文件
    
    Parameters
    ----------
    file_path : str
        数据文件路径
        
    Returns
    -------
    pd.DataFrame
        加载的数据框
        
    Raises
    ------
    FileNotFoundError
        文件不存在时抛出
    ValueError
        数据格式不正确时抛出
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"数据文件不存在: {file_path}")
    
    # 加载数据
    data = pd.read_excel(file_path)
    
    # 验证必要列是否存在
    if 'stamp' not in data.columns:
        raise ValueError("数据文件缺少 'stamp' 列")
    
    if 'behavior' not in data.columns:
        raise ValueError("数据文件缺少 'behavior' 列")
    
    # 设置时间戳为索引
    data = data.set_index('stamp')
    
    return data

def detect_behavior_events(data: pd.DataFrame, 
                          behavior_type: str, 
                          min_duration: float,
                          sampling_rate: float) -> List[Tuple[float, float]]:
    """
    检测行为事件的开始和结束时间
    
    Parameters
    ----------
    data : pd.DataFrame
        包含行为标签的数据
    behavior_type : str
        目标行为类型
    min_duration : float
        最小行为持续时间（秒）
    sampling_rate : float
        采样率（Hz）
        
    Returns
    -------
    List[Tuple[float, float]]
        行为事件列表，每个元素为(开始时间, 结束时间)
    """
    behavior_column = data['behavior']
    events = []
    
    # 找到行为变化点
    behavior_changes = behavior_column != behavior_column.shift(1)
    change_indices = behavior_changes[behavior_changes].index
    
    current_behavior = None
    start_time = None
    
    for timestamp in change_indices:
        new_behavior = behavior_column.loc[timestamp]
        
        # 如果当前行为结束
        if current_behavior == behavior_type and new_behavior != behavior_type:
            if start_time is not None:
                duration = timestamp - start_time
                # 只保留持续时间足够长的行为事件
                if duration >= min_duration:
                    events.append((start_time, timestamp))
        
        # 如果目标行为开始
        if new_behavior == behavior_type and current_behavior != behavior_type:
            start_time = timestamp
        
        current_behavior = new_behavior
    
    # 处理最后一个行为事件（如果数据在行为期间结束）
    if current_behavior == behavior_type and start_time is not None:
        end_time = data.index[-1]
        duration = end_time - start_time
        if duration >= min_duration:
            events.append((start_time, end_time))
    
    return events

def extract_time_window_data(data: pd.DataFrame, 
                           center_time: float, 
                           window_size: float) -> Optional[pd.DataFrame]:
    """
    提取指定时间窗口的数据
    
    Parameters
    ----------
    data : pd.DataFrame
        完整的神经元数据
    center_time : float
        中心时间点
    window_size : float
        时间窗口大小（秒）
        
    Returns
    -------
    Optional[pd.DataFrame]
        提取的时间窗口数据，如果窗口超出数据范围则返回None
    """
    start_time = center_time - window_size
    end_time = center_time + window_size
    
    # 检查时间窗口是否在数据范围内
    if start_time < data.index.min() or end_time > data.index.max():
        return None
    
    # 提取时间窗口内的数据
    window_data = data.loc[start_time:end_time].copy()
    
    # 移除行为列（如果存在）
    if 'behavior' in window_data.columns:
        window_data = window_data.drop(columns=['behavior'])
    
    return window_data

def standardize_neural_data(data: pd.DataFrame) -> pd.DataFrame:
    """
    对神经元数据进行Z-score标准化
    
    Parameters
    ----------
    data : pd.DataFrame
        原始神经元数据
        
    Returns
    -------
    pd.DataFrame
        标准化后的数据
    """
    # 只对数值列进行标准化
    numeric_columns = data.select_dtypes(include=[np.number]).columns
    standardized_data = data.copy()
    
    for col in numeric_columns:
        mean_val = data[col].mean()
        std_val = data[col].std()
        if std_val != 0:  # 避免除零错误
            standardized_data[col] = (data[col] - mean_val) / std_val
        else:
            standardized_data[col] = 0
    
    return standardized_data

def sort_neurons_by_peak_time(data: pd.DataFrame) -> pd.DataFrame:
    """
    按神经元峰值时间排序
    
    Parameters
    ----------
    data : pd.DataFrame
        标准化的神经元数据
        
    Returns
    -------
    pd.DataFrame
        按峰值时间排序的数据
    """
    # 计算每个神经元的峰值时间
    peak_times = data.idxmax()
    
    # 按峰值时间排序
    sorted_neurons = peak_times.sort_values().index
    
    return data[sorted_neurons]

def create_behavior_heatmap(data: pd.DataFrame, 
                          behavior_start_time: float,
                          behavior_type: str,
                          window_size: float,
                          config: BehaviorHeatmapConfig,
                          event_index: int) -> plt.Figure:
    """
    创建行为前后的热力图
    
    Parameters
    ----------
    data : pd.DataFrame
        时间窗口内的神经元数据
    behavior_start_time : float
        行为开始时间
    behavior_type : str
        行为类型
    window_size : float
        时间窗口大小
    config : BehaviorHeatmapConfig
        配置对象
    event_index : int
        事件索引
        
    Returns
    -------
    plt.Figure
        生成的图形对象
    """
    # 创建图形
    fig, ax = plt.subplots(figsize=(15, 10))
    
    # 绘制热力图
    sns.heatmap(
        data.T,  # 转置：行为神经元，列为时间
        cmap='RdYlBu_r',
        center=0,
        vmin=config.VMIN,
        vmax=config.VMAX,
        cbar_kws={'label': 'Standardized Calcium Signal'},
        ax=ax
    )
    
    # 在行为开始时间处画垂直线
    behavior_position = len(data.index) // 2  # 行为开始时间在窗口中心
    ax.axvline(x=behavior_position, color='white', linestyle='--', linewidth=3)
    
    # 添加文本标注
    ax.text(behavior_position + 1, -2, f'{behavior_type} Start', 
           color='black', fontweight='bold', fontsize=12)
    
    # 设置标题和标签
    ax.set_title(f'{behavior_type} Behavior Event #{event_index + 1}\n'
                f'Neural Activity ±{window_size}s Window', 
                fontsize=16, fontweight='bold')
    ax.set_xlabel('Time (seconds)', fontsize=14)
    ax.set_ylabel('Neurons', fontsize=14)
    
    # 设置X轴刻度标签（显示相对时间，以秒为单位）
    time_points = data.index - behavior_start_time
    tick_positions = np.linspace(0, len(data.index)-1, 5)
    tick_labels = [f'{time_points.iloc[int(pos)]:.1f}' for pos in tick_positions]
    ax.set_xticks(tick_positions)
    ax.set_xticklabels(tick_labels)
    
    # 调整Y轴标签
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=8)
    
    plt.tight_layout()
    return fig

def create_average_heatmap(all_window_data: List[pd.DataFrame],
                          behavior_type: str,
                          window_size: float,
                          config: BehaviorHeatmapConfig) -> plt.Figure:
    """
    创建所有行为事件的平均热力图
    
    Parameters
    ----------
    all_window_data : List[pd.DataFrame]
        所有时间窗口数据的列表
    behavior_type : str
        行为类型
    window_size : float
        时间窗口大小
    config : BehaviorHeatmapConfig
        配置对象
        
    Returns
    -------
    plt.Figure
        平均热力图
    """
    if not all_window_data:
        raise ValueError("没有有效的时间窗口数据")
    
    # 找到所有数据的公共神经元
    common_neurons = set(all_window_data[0].columns)
    for data in all_window_data[1:]:
        common_neurons &= set(data.columns)
    
    common_neurons = sorted(list(common_neurons))
    
    # 确保所有窗口数据长度相同（插值到统一长度）
    target_length = len(all_window_data[0])
    aligned_data = []
    
    for data in all_window_data:
        # 只保留公共神经元
        data_subset = data[common_neurons]
        
        # 如果长度不同，进行插值
        if len(data_subset) != target_length:
            # 创建新的时间索引
            new_index = np.linspace(data_subset.index[0], data_subset.index[-1], target_length)
            data_subset = data_subset.reindex(new_index).interpolate()
        
        aligned_data.append(data_subset.values)
    
    # 计算平均值
    average_data = np.mean(aligned_data, axis=0)
    
    # 创建平均数据的DataFrame
    time_relative = np.linspace(-window_size, window_size, target_length)
    average_df = pd.DataFrame(average_data, 
                             index=time_relative, 
                             columns=common_neurons)
    
    # 按峰值时间排序
    average_df_sorted = sort_neurons_by_peak_time(average_df)
    
    # 创建图形
    fig, ax = plt.subplots(figsize=(15, 10))
    
    # 绘制热力图
    sns.heatmap(
        average_df_sorted.T,
        cmap='RdYlBu_r',
        center=0,
        vmin=config.VMIN,
        vmax=config.VMAX,
        cbar_kws={'label': 'Average Standardized Calcium Signal'},
        ax=ax
    )
    
    # 在时间零点画垂直线
    zero_position = len(average_df_sorted) // 2
    ax.axvline(x=zero_position, color='white', linestyle='--', linewidth=3)
    
    # 添加文本标注
    ax.text(zero_position + 1, -2, f'{behavior_type} Start', 
           color='black', fontweight='bold', fontsize=12)
    
    # 设置标题和标签
    ax.set_title(f'Average {behavior_type} Behavior Response\n'
                f'Neural Activity ±{window_size}s Window (n={len(all_window_data)} events)', 
                fontsize=16, fontweight='bold')
    ax.set_xlabel('Time (seconds)', fontsize=14)
    ax.set_ylabel('Neurons (sorted by peak time)', fontsize=14)
    
    # 设置X轴刻度标签
    tick_positions = np.linspace(0, len(average_df_sorted)-1, 5)
    tick_labels = [f'{time_relative[int(pos)]:.1f}' for pos in tick_positions]
    ax.set_xticks(tick_positions)
    ax.set_xticklabels(tick_labels)
    
    # 调整Y轴标签
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=8)
    
    plt.tight_layout()
    return fig

def main():
    """
    主函数：执行行为热力图分析流程
    """
    # 解析命令行参数
    args = parse_arguments()
    
    # 创建配置对象
    config = BehaviorHeatmapConfig()
    
    # 更新配置
    if args.input:
        config.INPUT_FILE = args.input
    if args.output_dir:
        config.OUTPUT_DIR = args.output_dir
    if args.behavior:
        config.TARGET_BEHAVIOR = args.behavior
    if args.time_window:
        config.TIME_WINDOW = args.time_window
    if args.min_duration:
        config.MIN_BEHAVIOR_DURATION = args.min_duration
    if args.sampling_rate:
        config.SAMPLING_RATE = args.sampling_rate
    
    # 创建输出目录
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)
    
    try:
        # 加载数据
        print(f"正在加载数据: {config.INPUT_FILE}")
        data = load_and_validate_data(config.INPUT_FILE)
        
        # 转换时间戳为秒
        print(f"转换时间戳为秒（采样率: {config.SAMPLING_RATE}Hz）")
        data = convert_timestamps_to_seconds(data, config.SAMPLING_RATE)
        print(f"数据加载成功，包含 {len(data)} 个时间点，时间范围: {data.index.min():.2f}s - {data.index.max():.2f}s")
        
        # 检测行为事件
        print(f"正在检测 {config.TARGET_BEHAVIOR} 行为事件...")
        behavior_events = detect_behavior_events(
            data, 
            config.TARGET_BEHAVIOR, 
            config.MIN_BEHAVIOR_DURATION,
            config.SAMPLING_RATE
        )
        
        if not behavior_events:
            print(f"未检测到符合条件的 {config.TARGET_BEHAVIOR} 行为事件")
            return
        
        print(f"检测到 {len(behavior_events)} 个 {config.TARGET_BEHAVIOR} 行为事件")
        
        # 分析每个行为事件
        all_window_data = []
        
        for i, (start_time, end_time) in enumerate(behavior_events):
            print(f"正在分析事件 {i+1}: {start_time:.1f}s - {end_time:.1f}s")
            
            # 提取时间窗口数据
            window_data = extract_time_window_data(
                data, 
                start_time, 
                config.TIME_WINDOW
            )
            
            if window_data is None:
                print(f"事件 {i+1} 的时间窗口超出数据范围，跳过")
                continue
            
            # 标准化数据
            standardized_data = standardize_neural_data(window_data)
            
            # 按峰值时间排序
            sorted_data = sort_neurons_by_peak_time(standardized_data)
            
            # 保存用于平均计算
            all_window_data.append(sorted_data)
            
            # 创建单个事件的热力图
            fig = create_behavior_heatmap(
                sorted_data,
                start_time,
                config.TARGET_BEHAVIOR,
                config.TIME_WINDOW,
                config,
                i
            )
            
            # 保存图形
            output_path = os.path.join(
                config.OUTPUT_DIR,
                f'{config.TARGET_BEHAVIOR}_event_{i+1}_heatmap.png'
            )
            fig.savefig(output_path, dpi=300, bbox_inches='tight')
            plt.close(fig)
            print(f"已保存: {output_path}")
        
        # 创建平均热力图
        if all_window_data:
            print("正在创建平均热力图...")
            avg_fig = create_average_heatmap(
                all_window_data,
                config.TARGET_BEHAVIOR,
                config.TIME_WINDOW,
                config
            )
            
            # 保存平均热力图
            avg_output_path = os.path.join(
                config.OUTPUT_DIR,
                f'{config.TARGET_BEHAVIOR}_average_heatmap.png'
            )
            avg_fig.savefig(avg_output_path, dpi=300, bbox_inches='tight')
            plt.close(avg_fig)
            print(f"已保存平均热力图: {avg_output_path}")
        
        print(f"分析完成！共处理 {len(all_window_data)} 个有效事件")
        
    except Exception as e:
        print(f"错误: {e}")
        raise

if __name__ == "__main__":
    main()
