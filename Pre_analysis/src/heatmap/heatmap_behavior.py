"""
特定行为前后时间窗口的神经元活动热力图分析工具

功能：
- 检测指定行为的发生时间点
- 提取从第一个行为开始前指定时间到第二个行为结束的神经元活动数据
- 生成标准化热力图显示钙离子浓度变化
- 支持同一行为或不同行为的组合分析
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
    START_BEHAVIOR : str
        起始行为类型
    END_BEHAVIOR : str
        结束行为类型
    PRE_BEHAVIOR_TIME : float
        行为开始前的时间（秒）
    SAMPLING_RATE : float
        采样率（Hz）
    MIN_BEHAVIOR_DURATION : float
        最小行为持续时间（秒）
    """
    
    def __init__(self):
        # 输入文件路径
        self.INPUT_FILE = '../../datasets/29790930糖水铁网糖水trace2.xlsx'
        # 输出目录
        self.OUTPUT_DIR = '../../graph/behavior_heatmaps'
        # 起始行为类型（分析从此行为开始前的时间）
        self.START_BEHAVIOR = 'Crack-seeds-shells'
        # 结束行为类型（分析到此行为结束时刻）
        self.END_BEHAVIOR = 'Eat-seed-kernels'
        # 行为开始前的时间（秒）
        self.PRE_BEHAVIOR_TIME = 10.0
        # 采样率（钙离子数据采样频率：4.8Hz）
        self.SAMPLING_RATE = 4.8
        # 最小行为持续时间（秒），用于过滤短暂的误标记
        self.MIN_BEHAVIOR_DURATION = 1.0
        # 热力图颜色范围
        self.VMIN = -2
        self.VMAX = 2
        # 配置优先级控制：True表示__init__中的设定优先级最高，False表示命令行参数优先
        self.INIT_CONFIG_PRIORITY = True

def parse_arguments() -> argparse.Namespace:
    """
    解析命令行参数
    
    Returns
    -------
    argparse.Namespace
        解析后的参数对象
    """
    parser = argparse.ArgumentParser(
        description='生成从指定行为开始前到另一行为结束的神经元活动热力图'
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
        '--start-behavior', 
        type=str, 
        default='Eat-seed-kernels',
        help='起始行为类型（默认：Eat-seed-kernels）'
    )
    
    parser.add_argument(
        '--end-behavior', 
        type=str, 
        default='Eat-seed-kernels',
        help='结束行为类型（默认：Eat-seed-kernels）'
    )
    
    parser.add_argument(
        '--pre-behavior-time', 
        type=float, 
        default=10.0,
        help='行为开始前的时间，秒（默认：10.0）'
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
    # 调试信息：查看数据索引的具体情况
    print(f"调试信息：数据索引类型: {type(data.index[0])}")
    print(f"调试信息：前5个索引值: {data.index[:5].tolist()}")
    print(f"调试信息：索引范围: {data.index.min()} - {data.index.max()}")
    
    # 检查时间戳是否为连续的数值序列
    import numbers
    is_numeric = isinstance(data.index[0], (int, float, numbers.Integral, numbers.Real)) or np.issubdtype(data.index.dtype, np.number)
    print(f"调试信息：isinstance检查结果: {is_numeric}")
    
    if is_numeric:
        # 计算时间间隔来判断是否需要转换
        time_intervals = np.diff(data.index[:10])  # 检查前10个时间间隔
        avg_interval = np.mean(time_intervals)
        
        # 如果平均时间间隔接近1/采样率，说明时间戳已经是秒
        expected_interval = 1.0 / sampling_rate
        
        print(f"调试信息：平均时间间隔: {avg_interval:.3f}, 期望秒间隔: {expected_interval:.3f}")
        print(f"调试信息：差值: {abs(avg_interval - expected_interval):.3f}, 阈值: {expected_interval * 0.1:.3f}")
        
        if abs(avg_interval - expected_interval) < expected_interval * 0.1:
            # 时间戳已经是秒，直接返回
            print(f"检测到时间戳已为秒格式（平均间隔: {avg_interval:.3f}s）")
            return data
        else:
            # 时间戳是数据点索引，需要转换为秒
            print(f"检测到时间戳为采样点索引（间隔: {avg_interval:.1f}点），转换为秒")
            
            # 正确的转换：将采样点索引转换为时间（秒）
            # 假设第一个采样点（索引1）对应时间0秒
            time_seconds = (data.index - data.index.min()) / sampling_rate
            
            data_converted = data.copy()
            data_converted.index = time_seconds
            
            print(f"时间转换完成: {data.index.min()}-{data.index.max()}(采样点) → {time_seconds.min():.2f}-{time_seconds.max():.2f}(秒)")
            return data_converted
    else:
        # 如果时间戳已经是时间格式，直接返回
        print(f"检测到时间戳为非数值格式，类型: {type(data.index[0])}")
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

def find_behavior_pairs(data: pd.DataFrame,
                       start_behavior: str,
                       end_behavior: str,
                       min_duration: float,
                       sampling_rate: float) -> List[Tuple[float, float, float, float]]:
    """
    找到起始行为和结束行为的配对
    
    Parameters
    ----------
    data : pd.DataFrame
        包含行为标签的数据
    start_behavior : str
        起始行为类型
    end_behavior : str
        结束行为类型
    min_duration : float
        最小行为持续时间（秒）
    sampling_rate : float
        采样率（Hz）
        
    Returns
    -------
    List[Tuple[float, float, float, float]]
        行为配对列表，每个元素为(起始行为开始时间, 起始行为结束时间, 结束行为开始时间, 结束行为结束时间)
    """
    # 检测起始行为事件
    start_events = detect_behavior_events(data, start_behavior, min_duration, sampling_rate)
    
    # 检测结束行为事件
    end_events = detect_behavior_events(data, end_behavior, min_duration, sampling_rate)
    
    if not start_events:
        print(f"未找到 {start_behavior} 行为事件")
        return []
    
    if not end_events:
        print(f"未找到 {end_behavior} 行为事件")
        return []
    
    behavior_pairs = []
    
    # 如果是同一行为，直接使用每个事件的开始和结束时间
    if start_behavior == end_behavior:
        for start_time, end_time in start_events:
            behavior_pairs.append((start_time, start_time, end_time, end_time))
    else:
        # 如果是不同行为，找到时间上匹配的配对
        for start_begin, start_end in start_events:
            # 找到在起始行为之后开始的第一个结束行为
            for end_begin, end_end in end_events:
                if end_begin >= start_begin:  # 结束行为必须在起始行为开始之后
                    behavior_pairs.append((start_begin, start_end, end_begin, end_end))
                    break  # 找到第一个匹配的就停止
    
    return behavior_pairs

def extract_behavior_sequence_data(data: pd.DataFrame,
                                 start_time: float,
                                 end_time: float,
                                 pre_behavior_time: float) -> Optional[pd.DataFrame]:
    """
    提取从行为开始前到行为结束的数据序列
    
    Parameters
    ----------
    data : pd.DataFrame
        完整的神经元数据
    start_time : float
        起始行为开始时间
    end_time : float
        结束行为结束时间
    pre_behavior_time : float
        行为开始前的时间（秒）
        
    Returns
    -------
    Optional[pd.DataFrame]
        提取的数据序列，如果时间范围超出数据范围则返回None
    """
    sequence_start = start_time - pre_behavior_time
    sequence_end = end_time
    
    # 检查时间范围是否在数据范围内
    if sequence_start < data.index.min() or sequence_end > data.index.max():
        print(f"时间范围 {sequence_start:.1f}s - {sequence_end:.1f}s 超出数据范围 {data.index.min():.1f}s - {data.index.max():.1f}s")
        return None
    
    # 提取时间序列内的数据
    sequence_data = data.loc[sequence_start:sequence_end].copy()
    
    # 移除行为列（如果存在）
    if 'behavior' in sequence_data.columns:
        sequence_data = sequence_data.drop(columns=['behavior'])
    
    return sequence_data

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

def create_behavior_sequence_heatmap(data: pd.DataFrame,
                                   start_behavior_time: float,
                                   end_behavior_time: float,
                                   start_behavior: str,
                                   end_behavior: str,
                                   pre_behavior_time: float,
                                   config: BehaviorHeatmapConfig,
                                   pair_index: int) -> plt.Figure:
    """
    创建行为序列的热力图
    
    Parameters
    ----------
    data : pd.DataFrame
        行为序列内的神经元数据
    start_behavior_time : float
        起始行为开始时间
    end_behavior_time : float
        结束行为结束时间
    start_behavior : str
        起始行为类型
    end_behavior : str
        结束行为类型
    pre_behavior_time : float
        行为开始前的时间
    config : BehaviorHeatmapConfig
        配置对象
    pair_index : int
        配对索引
        
    Returns
    -------
    plt.Figure
        生成的图形对象
    """
    # 创建图形
    fig, ax = plt.subplots(figsize=(18, 10))
    
    # 绘制热力图
    sns.heatmap(
        data.T,  # 转置：行为神经元，列为时间
        cmap='viridis',
        center=0,
        vmin=config.VMIN,
        vmax=config.VMAX,
        cbar_kws={'label': 'Standardized Calcium Signal'},
        ax=ax
    )
    
    # 计算关键时间点的位置
    sequence_start_time = start_behavior_time - pre_behavior_time
    total_duration = end_behavior_time - sequence_start_time
    
    # 起始行为开始位置
    start_position = len(data.index) * pre_behavior_time / total_duration
    # 结束行为结束位置（序列结束）
    end_position = len(data.index) - 1
    
    # 在关键时间点画垂直线
    ax.axvline(x=start_position, color='white', linestyle='--', linewidth=3, alpha=0.8)
    ax.axvline(x=end_position, color='white', linestyle='-', linewidth=3, alpha=0.8)
    
    # 添加文本标注
    ax.text(start_position + 1, -3, f'{start_behavior} Start', 
           color='black', fontweight='bold', fontsize=12, 
           bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
    
    ax.text(end_position - 10, -3, f'{end_behavior} End', 
           color='black', fontweight='bold', fontsize=12,
           bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
    
    # 设置标题和标签
    if start_behavior == end_behavior:
        title = f'{start_behavior} Behavior Sequence #{pair_index + 1}\n'
        title += f'Neural Activity: -{pre_behavior_time}s to End'
    else:
        title = f'{start_behavior} → {end_behavior} Behavior Sequence #{pair_index + 1}\n'
        title += f'Neural Activity: {start_behavior} -{pre_behavior_time}s to {end_behavior} End'
    
    ax.set_title(title, fontsize=16, fontweight='bold')
    ax.set_xlabel('Time (seconds)', fontsize=14)
    ax.set_ylabel('Neurons', fontsize=14)
    
    # 设置X轴刻度标签（显示相对于起始行为开始的时间）
    time_points = data.index - start_behavior_time
    tick_positions = np.linspace(0, len(data.index)-1, 8)
    tick_labels = [f'{time_points[int(pos)]:.1f}' for pos in tick_positions]
    ax.set_xticks(tick_positions)
    ax.set_xticklabels(tick_labels)
    
    # 调整Y轴标签
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=8)
    
    plt.tight_layout()
    return fig

def create_average_sequence_heatmap(all_sequence_data: List[pd.DataFrame],
                                  start_behavior: str,
                                  end_behavior: str,
                                  pre_behavior_time: float,
                                  config: BehaviorHeatmapConfig) -> plt.Figure:
    """
    创建所有行为序列的平均热力图
    
    Parameters
    ----------
    all_sequence_data : List[pd.DataFrame]
        所有行为序列数据的列表
    start_behavior : str
        起始行为类型
    end_behavior : str
        结束行为类型
    pre_behavior_time : float
        行为开始前的时间
    config : BehaviorHeatmapConfig
        配置对象
        
    Returns
    -------
    plt.Figure
        平均热力图
    """
    if not all_sequence_data:
        raise ValueError("没有有效的行为序列数据")
    
    # 找到所有数据的公共神经元
    common_neurons = set(all_sequence_data[0].columns)
    for data in all_sequence_data[1:]:
        common_neurons &= set(data.columns)
    
    common_neurons = sorted(list(common_neurons))
    
    # 确定目标长度（使用最短序列的长度以确保所有数据都有效）
    min_length = min(len(data) for data in all_sequence_data)
    aligned_data = []
    
    for data in all_sequence_data:
        # 只保留公共神经元
        data_subset = data[common_neurons]
        
        # 重采样到统一长度
        if len(data_subset) != min_length:
            # 创建新的索引
            new_index = np.linspace(0, len(data_subset)-1, min_length)
            original_index = np.arange(len(data_subset))
            
            # 对每个神经元进行插值
            resampled_data = np.zeros((min_length, len(common_neurons)))
            for j, neuron in enumerate(common_neurons):
                resampled_data[:, j] = np.interp(new_index, original_index, data_subset[neuron].values)
            
            aligned_data.append(resampled_data)
        else:
            aligned_data.append(data_subset.values)
    
    # 计算平均值
    average_data = np.mean(aligned_data, axis=0)
    
    # 创建平均数据的DataFrame
    time_relative = np.linspace(-pre_behavior_time, 0, min_length)  # 相对于起始行为开始的时间
    average_df = pd.DataFrame(average_data, 
                             index=time_relative, 
                             columns=common_neurons)
    
    # 按峰值时间排序
    average_df_sorted = sort_neurons_by_peak_time(average_df)
    
    # 创建图形
    fig, ax = plt.subplots(figsize=(18, 10))
    
    # 绘制热力图
    sns.heatmap(
        average_df_sorted.T,
        cmap='viridis',
        center=0,
        vmin=config.VMIN,
        vmax=config.VMAX,
        cbar_kws={'label': 'Average Standardized Calcium Signal'},
        ax=ax
    )
    
    # 在起始行为开始时间点画垂直线
    start_position = len(average_df_sorted) * pre_behavior_time / (pre_behavior_time + 0)  # 在序列中的相对位置
    ax.axvline(x=start_position, color='white', linestyle='--', linewidth=3)
    
    # 添加文本标注
    ax.text(start_position + 1, -3, f'{start_behavior} Start', 
           color='black', fontweight='bold', fontsize=12,
           bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
    
    # 设置标题和标签
    if start_behavior == end_behavior:
        title = f'Average {start_behavior} Behavior Sequence\n'
        title += f'Neural Activity: -{pre_behavior_time}s to End (n={len(all_sequence_data)} sequences)'
    else:
        title = f'Average {start_behavior} → {end_behavior} Behavior Sequence\n'
        title += f'Neural Activity: {start_behavior} -{pre_behavior_time}s to {end_behavior} End (n={len(all_sequence_data)} sequences)'
    
    ax.set_title(title, fontsize=16, fontweight='bold')
    ax.set_xlabel('Time (seconds)', fontsize=14)
    ax.set_ylabel('Neurons (sorted by peak time)', fontsize=14)
    
    # 设置X轴刻度标签
    tick_positions = np.linspace(0, len(average_df_sorted)-1, 8)
    tick_labels = [f'{time_relative[int(pos)]:.1f}' for pos in tick_positions]
    ax.set_xticks(tick_positions)
    ax.set_xticklabels(tick_labels)
    
    # 调整Y轴标签
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=8)
    
    plt.tight_layout()
    return fig

def main():
    """
    主函数：执行行为序列热力图分析流程
    """
    # 解析命令行参数
    args = parse_arguments()
    
    # 创建配置对象
    config = BehaviorHeatmapConfig()
    
    # 保存__init__中的行为设定
    init_start_behavior = config.START_BEHAVIOR
    init_end_behavior = config.END_BEHAVIOR
    init_priority = config.INIT_CONFIG_PRIORITY
    
    # 更新其他配置项
    if args.input:
        config.INPUT_FILE = args.input
    if args.output_dir:
        config.OUTPUT_DIR = args.output_dir
    if args.pre_behavior_time:
        config.PRE_BEHAVIOR_TIME = args.pre_behavior_time
    if args.min_duration:
        config.MIN_BEHAVIOR_DURATION = args.min_duration
    if args.sampling_rate:
        config.SAMPLING_RATE = args.sampling_rate
    
    # 行为设定优先级控制
    if init_priority:
        # __init__中的设定具有最高优先级
        config.START_BEHAVIOR = init_start_behavior
        config.END_BEHAVIOR = init_end_behavior
        print(f"使用起始行为: {config.START_BEHAVIOR}, 结束行为: {config.END_BEHAVIOR} (来源: __init__配置，优先级最高)")
    else:
        # 命令行参数优先
        if args.start_behavior:
            config.START_BEHAVIOR = args.start_behavior
        if args.end_behavior:
            config.END_BEHAVIOR = args.end_behavior
        print(f"使用起始行为: {config.START_BEHAVIOR}, 结束行为: {config.END_BEHAVIOR} (来源: 命令行参数)")
    
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
        
        # 查找行为配对
        print(f"正在查找 {config.START_BEHAVIOR} → {config.END_BEHAVIOR} 行为配对...")
        behavior_pairs = find_behavior_pairs(
            data,
            config.START_BEHAVIOR,
            config.END_BEHAVIOR,
            config.MIN_BEHAVIOR_DURATION,
            config.SAMPLING_RATE
        )
        
        if not behavior_pairs:
            print(f"未找到符合条件的行为配对")
            return
        
        print(f"找到 {len(behavior_pairs)} 个行为配对")
        
        # 分析每个行为配对
        all_sequence_data = []
        
        for i, (start_begin, start_end, end_begin, end_end) in enumerate(behavior_pairs):
            if config.START_BEHAVIOR == config.END_BEHAVIOR:
                print(f"正在分析序列 {i+1}: {config.START_BEHAVIOR} ({start_begin:.1f}s - {end_end:.1f}s)")
                sequence_start_time = start_begin
                sequence_end_time = end_end
            else:
                print(f"正在分析序列 {i+1}: {config.START_BEHAVIOR} ({start_begin:.1f}s - {start_end:.1f}s) → {config.END_BEHAVIOR} ({end_begin:.1f}s - {end_end:.1f}s)")
                sequence_start_time = start_begin
                sequence_end_time = end_end
            
            # 提取行为序列数据
            sequence_data = extract_behavior_sequence_data(
                data,
                sequence_start_time,
                sequence_end_time,
                config.PRE_BEHAVIOR_TIME
            )
            
            if sequence_data is None:
                print(f"序列 {i+1} 的时间范围超出数据范围，跳过")
                continue
            
            # 标准化数据
            standardized_data = standardize_neural_data(sequence_data)
            
            # 按峰值时间排序
            sorted_data = sort_neurons_by_peak_time(standardized_data)
            
            # 保存用于平均计算
            all_sequence_data.append(sorted_data)
            
            # 创建单个序列的热力图
            fig = create_behavior_sequence_heatmap(
                sorted_data,
                sequence_start_time,
                sequence_end_time,
                config.START_BEHAVIOR,
                config.END_BEHAVIOR,
                config.PRE_BEHAVIOR_TIME,
                config,
                i
            )
            
            # 保存图形
            if config.START_BEHAVIOR == config.END_BEHAVIOR:
                output_path = os.path.join(
                    config.OUTPUT_DIR,
                    f'{config.START_BEHAVIOR}_sequence_{i+1}_heatmap.png'
                )
            else:
                output_path = os.path.join(
                    config.OUTPUT_DIR,
                    f'{config.START_BEHAVIOR}_to_{config.END_BEHAVIOR}_sequence_{i+1}_heatmap.png'
                )
            
            fig.savefig(output_path, dpi=300, bbox_inches='tight')
            plt.close(fig)
            print(f"已保存: {output_path}")
        
        # 创建平均热力图
        if all_sequence_data:
            print("正在创建平均热力图...")
            avg_fig = create_average_sequence_heatmap(
                all_sequence_data,
                config.START_BEHAVIOR,
                config.END_BEHAVIOR,
                config.PRE_BEHAVIOR_TIME,
                config
            )
            
            # 保存平均热力图
            if config.START_BEHAVIOR == config.END_BEHAVIOR:
                avg_output_path = os.path.join(
                    config.OUTPUT_DIR,
                    f'{config.START_BEHAVIOR}_average_sequence_heatmap.png'
                )
            else:
                avg_output_path = os.path.join(
                    config.OUTPUT_DIR,
                    f'{config.START_BEHAVIOR}_to_{config.END_BEHAVIOR}_average_sequence_heatmap.png'
                )
            
            avg_fig.savefig(avg_output_path, dpi=300, bbox_inches='tight')
            plt.close(avg_fig)
            print(f"已保存平均热力图: {avg_output_path}")
        
        print(f"分析完成！共处理 {len(all_sequence_data)} 个有效序列")
        
    except Exception as e:
        print(f"错误: {e}")
        raise

if __name__ == "__main__":
    main()

"""
使用示例和配置说明：

1. 分析同一行为（从开始前10秒到结束）：
   ```python
   class BehaviorHeatmapConfig:
       def __init__(self):
           self.START_BEHAVIOR = 'Eat-seed-kernels'  # 起始行为
           self.END_BEHAVIOR = 'Eat-seed-kernels'    # 结束行为（同一行为）
           self.PRE_BEHAVIOR_TIME = 10.0             # 行为开始前10秒
           self.INIT_CONFIG_PRIORITY = True
   ```

2. 分析不同行为序列：
   ```python
   class BehaviorHeatmapConfig:
       def __init__(self):
           self.START_BEHAVIOR = 'Groom'           # 从梳理行为开始前10秒
           self.END_BEHAVIOR = 'Water'             # 到饮水行为结束
           self.PRE_BEHAVIOR_TIME = 10.0           # 行为开始前10秒
           self.INIT_CONFIG_PRIORITY = True
   ```

3. 命令行使用示例：
   ```bash
   # 分析同一行为
   python heatmap_behavior.py --start-behavior "Crack-seeds-shells" --end-behavior "Crack-seeds-shells" --pre-behavior-time 10

   # 分析不同行为序列  
   python heatmap_behavior.py --start-behavior "Find-seeds" --end-behavior "Eat-seed-kernels" --pre-behavior-time 15
   ```

4. 功能特点：
   - 支持同一行为的完整序列分析（开始前N秒到行为结束）
   - 支持不同行为的连续序列分析（行为A开始前N秒到行为B结束）
   - 自动匹配时间上相关的行为配对
   - 生成个体序列和平均序列的热力图
   - 在热力图上标注关键时间点（行为开始和结束）

5. 可用的行为类型：
   - 'Crack-seeds-shells', 'Eat-feed', 'Eat-seed-kernels', 'Explore'
   - 'Explore-search-seeds', 'Find-seeds', 'Get-feed', 'Get-seeds'
   - 'Grab-seeds', 'Groom', 'Smell-feed', 'Smell-Get-seeds'
   - 'Store-seeds', 'Water'
"""
