#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
神经元钙离子波动可视化模块

该模块用于将每条神经元的钙离子波动数据可视化展现为图表，
横坐标为时间戳（stamp），纵坐标为钙离子浓度。
支持显示行为标签和行为时间段标记。

新增功能：
- 滤波降噪预处理（与smooth_data.py保持一致）
- 移动平均滤波
- Butterworth低通滤波
- 数据归一化
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.gridspec import GridSpec
import matplotlib.patches as mpatches
import seaborn as sns
from typing import Union, List, Dict, Optional, Tuple, Literal
from scipy.signal import butter, filtfilt
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
import logging

def moving_average(data: Union[pd.Series, np.ndarray], window_size: int = 3) -> pd.Series:
    """
    应用移动平均滤波平滑数据（与smooth_data.py保持一致）

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
    fs: float = 4.8,
    order: int = 2,
    strength: float = 0.05
) -> np.ndarray:
    """
    应用Butterworth低通滤波器去除高频噪声（与smooth_data.py保持一致）

    参数:
        data: 输入信号数据
        cutoff_freq: 截止频率，值越小滤波效果越强（默认20）
        fs: 采样频率（默认4.8Hz）
        order: 滤波器阶数，阶数越高滤波效果越陡峭（默认2）
        strength: 滤波强度系数，范围0-1，值越大滤波效果越强（默认0.05）

    返回:
        滤波后的数据
    """
    try:
        nyquist = fs * 0.5
        normal_cutoff = (cutoff_freq * strength) / nyquist
        
        # 确保截止频率在合理范围内
        if normal_cutoff >= 1.0:
            normal_cutoff = 0.99  # 防止截止频率超过奈奎斯特频率
        elif normal_cutoff <= 0:
            logging.warning(f"截止频率过低 ({normal_cutoff})，跳过Butterworth滤波")
            return data
        
        b, a = butter(order, normal_cutoff, btype='low', analog=False)
        return filtfilt(b, a, data)
    except Exception as e:
        logging.warning(f"Butterworth滤波失败: {e}，返回原始数据")
        return data

def normalize_data(
    data: pd.DataFrame,
    columns: List[str],
    method: Literal['standard', 'minmax', 'robust', 'log_standard', 'log_minmax'] = 'standard',
    feature_range: tuple = (0, 1)
) -> pd.DataFrame:
    """
    使用不同方法对数据进行归一化（与smooth_data.py保持一致）

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

def preprocess_signal(
    data: pd.DataFrame,
    neuron_columns: List[str],
    apply_moving_average: bool = True,
    moving_avg_window: int = 3,
    apply_butterworth: bool = True,
    butterworth_cutoff: float = 20,
    butterworth_strength: float = 0.05,
    apply_normalization: bool = False,
    normalization_method: str = 'standard',
    feature_range: tuple = (0, 1),
    fs: float = 4.8
) -> pd.DataFrame:
    """
    处理神经信号数据，与smooth_data.py保持一致的预处理流程

    参数:
        data: 输入的DataFrame，包含神经元数据
        neuron_columns: 要处理的神经元列名列表
        apply_moving_average: 是否应用移动平均滤波（默认True）
        moving_avg_window: 移动平均窗口大小（默认3）
        apply_butterworth: 是否应用Butterworth滤波（默认True）
        butterworth_cutoff: Butterworth滤波器截止频率（默认20）
        butterworth_strength: Butterworth滤波强度（默认0.05）
        apply_normalization: 是否应用归一化（默认False）
        normalization_method: 归一化方法（默认'standard'）
        feature_range: 用于minmax归一化的目标范围（默认(0,1)）
        fs: 采样频率（默认4.8Hz）

    返回:
        处理后的DataFrame
    """
    processed_data = data.copy()
    existing_neuron_cols = [col for col in neuron_columns if col in processed_data.columns]

    if not existing_neuron_cols:
        logging.warning(f"未找到任何指定的神经元数据列: {neuron_columns}")
        return processed_data

    for column in existing_neuron_cols:
        signal = processed_data[column].values
        
        # 步骤1：移动平均滤波
        if apply_moving_average:
            signal = moving_average(signal, window_size=moving_avg_window).values
        
        # 步骤2：Butterworth滤波
        if apply_butterworth:
            signal = butterworth_filter(
                signal, 
                cutoff_freq=butterworth_cutoff,
                fs=fs,
                strength=butterworth_strength
            )
            
        processed_data[column] = signal
    
    # 步骤3：数据归一化
    if apply_normalization:
        processed_data = normalize_data(
            processed_data,
            existing_neuron_cols,
            method=normalization_method,
            feature_range=feature_range
        )
    
    return processed_data

def load_data(file_path: str) -> pd.DataFrame:
    """
    加载神经元钙离子数据
    
    Parameters
    ----------
    file_path : str
        数据文件路径，支持.md, .csv, .xlsx格式
    
    Returns
    -------
    pd.DataFrame
        包含神经元钙离子数据的DataFrame
    """
    if file_path.endswith('.md'):
        # 尝试从markdown文件中解析表格数据
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
            # 提取markdown表格内容
            lines = content.strip().split('\n')
            # 跳过表头和分隔线
            data_lines = [line.strip().strip('|').split('|') for line in lines[2:]]
            columns = [col.strip() for col in lines[0].strip().strip('|').split('|')]
            
            # 将数据转换为DataFrame
            df = pd.DataFrame(data_lines, columns=columns)
            # 将数值列转换为浮点数
            for col in df.columns:
                if col != 'behavior':
                    df[col] = pd.to_numeric(df[col], errors='coerce')
            return df
    elif file_path.endswith('.csv'):
        return pd.read_csv(file_path)
    elif file_path.endswith('.xlsx'):
        return pd.read_excel(file_path)
    else:
        raise ValueError(f"不支持的文件格式: {file_path}")

def plot_neuron_calcium(df: pd.DataFrame, neuron_id: str, save_dir: str = None, 
                        sampling_rate: float = 4.8,
                        apply_preprocessing: bool = True,
                        apply_moving_average: bool = True,
                        moving_avg_window: int = 3,
                        apply_butterworth: bool = True,
                        butterworth_cutoff: float = 20,
                        butterworth_strength: float = 0.05,
                        apply_normalization: bool = False,
                        normalization_method: str = 'standard',
                        feature_range: tuple = (0, 1)) -> Figure:
    """
    绘制单个神经元的钙离子波动图，支持行为标签显示和预处理
    
    Parameters
    ----------
    df : pd.DataFrame
        包含神经元数据的DataFrame
    neuron_id : str
        神经元ID，对应DataFrame中的列名
    save_dir : str, optional
        图像保存目录，默认为None（不保存）
    sampling_rate : float, optional
        采样频率，默认为4.8Hz（保留参数以保持兼容性，但现在横坐标使用时间戳）
    apply_preprocessing : bool, optional
        是否应用预处理，默认为False
    apply_moving_average : bool, optional
        是否应用移动平均滤波，默认为True
    moving_avg_window : int, optional
        移动平均窗口大小，默认为3
    apply_butterworth : bool, optional
        是否应用Butterworth滤波，默认为True
    butterworth_cutoff : float, optional
        Butterworth滤波器截止频率，默认为20（与smooth_data.py保持一致）
    butterworth_strength : float, optional
        Butterworth滤波强度，默认为0.05（与smooth_data.py保持一致）
    apply_normalization : bool, optional
        是否应用归一化，默认为False
    normalization_method : str, optional
        归一化方法，默认为'standard'
    feature_range : tuple, optional
        用于minmax归一化的目标范围，默认为(0, 1)
    
    Returns
    -------
    Figure  
        matplotlib图像对象
    """
    if neuron_id not in df.columns:
        raise ValueError(f"神经元ID {neuron_id} 在数据中不存在")
    
    # 应用预处理（如果启用）
    if apply_preprocessing:
        print(f"对神经元 {neuron_id} 应用预处理: 移动平均({apply_moving_average}), Butterworth({apply_butterworth}), 归一化({apply_normalization})")
        df_processed = preprocess_signal(
            df, [neuron_id],
            apply_moving_average=apply_moving_average,
            moving_avg_window=moving_avg_window,
            apply_butterworth=apply_butterworth,
            butterworth_cutoff=butterworth_cutoff,
            butterworth_strength=butterworth_strength,
            apply_normalization=apply_normalization,
            normalization_method=normalization_method,
            feature_range=feature_range,
            fs=sampling_rate
        )
    else:
        df_processed = df.copy()
    
    # 检查是否存在行为数据
    has_behavior = 'behavior' in df_processed.columns
    behavior_data = None
    
    if has_behavior:
        behavior_data = df_processed['behavior']
        print(f"检测到行为数据，将在图表中显示行为区间")
    
    # 创建图像布局
    if has_behavior and behavior_data.dropna().unique().size > 0:
        # 如果有行为数据，使用两行一列的布局
        fig = plt.figure(figsize=(60, 20))
        grid = GridSpec(2, 1, height_ratios=[1, 4], hspace=0.08, figure=fig)
        ax_behavior = fig.add_subplot(grid[0])
        ax_main = fig.add_subplot(grid[1])
    else:
        # 没有行为数据，只创建一个图表
        fig, ax_main = plt.subplots(figsize=(60, 15))
    
    # 使用时间戳作为横坐标
    time_stamps = df_processed['stamp']
    
    # 绘制钙离子浓度变化曲线
    ax_main.plot(time_stamps, df_processed[neuron_id], linewidth=2, color='#1f77b4', 
                 label='Calcium Signal')
    
    # 添加平滑曲线（移动平均）
    window_size = min(10, len(df_processed) // 10)  # 根据数据长度调整窗口大小
    if window_size > 1:
        smooth_data = df_processed[neuron_id].rolling(window=window_size, center=True).mean()
        ax_main.plot(time_stamps, smooth_data, linewidth=2.5, color='#ff7f0e', alpha=0.7, 
                    label=f'Smooth Curve (Window={window_size})')
    
    # 设置图表样式和标题（包含预处理信息）
    title = f'Neuron {neuron_id} Calcium Concentration Fluctuation'
    if apply_preprocessing:
        preprocessing_info = []
        if apply_moving_average:
            preprocessing_info.append(f"MovAvg({moving_avg_window})")
        if apply_butterworth:
            preprocessing_info.append(f"Butterworth({butterworth_strength})")
        if apply_normalization:
            preprocessing_info.append(f"Norm({normalization_method})")
        if preprocessing_info:
            title += f" [Preprocessed: {', '.join(preprocessing_info)}]"
    ax_main.set_title(title, fontsize=50)
    ax_main.set_xlabel('Time (stamps)', fontsize=48)
    ax_main.set_ylabel('Calcium Concentration', fontsize=48)
    ax_main.grid(True, linestyle='--', alpha=0.7)
    
    # 设置坐标轴刻度标签的字体大小
    ax_main.tick_params(axis='both', which='major', labelsize=40)
    
    # 设置横坐标刻度，根据时间戳范围自动调整刻度间隔
    max_stamp = time_stamps.max()
    min_stamp = time_stamps.min()
    stamp_range = max_stamp - min_stamp
    
    # 根据时间戳范围动态设置刻度间隔
    if stamp_range > 10000:
        step = 500  # 大范围时每500个时间戳显示一个刻度
    elif stamp_range > 5000:
        step = 200  # 中等范围时每200个时间戳显示一个刻度
    else:
        step = 100  # 小范围时每100个时间戳显示一个刻度
    
    tick_positions = np.arange(min_stamp, max_stamp + step, step)
    ax_main.set_xticks(tick_positions)
    ax_main.tick_params(axis='x', rotation=30)  # 旋转刻度标签以防重叠
    
    # 突出显示极值点
    max_val = df_processed[neuron_id].max()
    max_idx = df_processed[neuron_id].idxmax()
    min_val = df_processed[neuron_id].min()
    min_idx = df_processed[neuron_id].idxmin()
    
    ax_main.scatter(time_stamps.iloc[max_idx], max_val, color='red', s=100, zorder=5, 
                   label=f'Maximum: {max_val:.2f}')
    ax_main.scatter(time_stamps.iloc[min_idx], min_val, color='green', s=100, zorder=5, 
                   label=f'Minimum: {min_val:.2f}')
    
    # 添加统计信息
    mean_val = df_processed[neuron_id].mean()
    std_val = df_processed[neuron_id].std()
    stats_text = f'Mean: {mean_val:.2f}\nStandard Deviation: {std_val:.2f}'
    ax_main.text(0.02, 0.97, stats_text, transform=ax_main.transAxes, fontsize=50, 
                verticalalignment='top', bbox=dict(boxstyle='round', alpha=0.1))
    
    # 添加图例
    ax_main.legend(loc='best', fontsize=30)
    
    # 处理行为数据（如果存在）
    if has_behavior and behavior_data.dropna().unique().size > 0:
        # 预定义颜色映射，与show_trace.py保持一致
        fixed_color_map = {
            'Crack-seeds-shells': '#FF9500',    # 明亮橙色
            'Eat-feed': '#0066CC',              # 深蓝色
            'Eat-seed-kernels': '#00CC00',      # 亮绿色
            'Explore': '#FF0000',               # 鲜红色
            'Explore-search-seeds': '#9900FF',  # 亮紫色
            'Find-seeds': '#994C00',            # 深棕色
            'Get-feed': '#FF00CC',              # 亮粉色
            'Get-seeds': '#000000',             # 黑色
            'Grab-seeds': '#AACC00',            # 亮黄绿色
            'Groom': '#00CCFF',                 # 亮蓝绿色
            'Smell-feed': '#66B3FF',            # 亮蓝色
            'Smell-Get-seeds': '#33FF33',       # 鲜绿色
            'Store-seeds': '#FF6666',           # 亮红色
            'Water': '#CC99FF'                  # 亮紫色
        }
        
        # 获取所有不同的行为标签
        unique_behaviors = behavior_data.dropna().unique()
        
        # 处理行为区间数据
        behavior_intervals = {}
        for behavior in unique_behaviors:
            behavior_intervals[behavior] = []
        
        # 找出每种行为的连续区间
        current_behavior = None
        start_stamp = None
        
        # 为了确保最后一个区间也被记录，将索引列表扩展一个元素
        extended_index = list(df_processed.index) + [None]
        extended_stamps = list(df_processed['stamp'].values) + [None]
        extended_behaviors = list(behavior_data.values) + [None]
        
        for i, (idx, stamp, behavior) in enumerate(zip(extended_index, extended_stamps, extended_behaviors)):
            # 最后一个元素特殊处理
            if i == len(df_processed):
                if start_stamp is not None and current_behavior is not None:
                    end_stamp = extended_stamps[i-1]
                    behavior_intervals[current_behavior].append((start_stamp, end_stamp))
                break
            
            # 跳过空值
            if pd.isna(behavior):
                # 如果之前有行为，则结束当前区间
                if start_stamp is not None and current_behavior is not None:
                    end_stamp = stamp
                    behavior_intervals[current_behavior].append((start_stamp, end_stamp))
                    start_stamp = None
                    current_behavior = None
                continue
            
            # 如果是新的行为类型或第一个行为
            if behavior != current_behavior:
                # 如果之前有行为，先结束当前区间
                if start_stamp is not None and current_behavior is not None:
                    end_stamp = stamp
                    behavior_intervals[current_behavior].append((start_stamp, end_stamp))
                
                # 开始新的行为区间
                start_stamp = stamp
                current_behavior = behavior
        
        # 绘制行为标记
        if len(unique_behaviors) > 0:
            # 创建图例补丁列表
            legend_patches = []
            
            # 为每种行为分配Y轴位置
            y_positions = {}
            max_position = len(unique_behaviors)
            
            for i, behavior in enumerate(unique_behaviors):
                y_positions[behavior] = max_position - i
            
            # 设置行为子图的Y轴范围和刻度
            ax_behavior.set_ylim(0, max_position + 1)
            ax_behavior.set_yticks([y_positions[b] for b in unique_behaviors])
            ax_behavior.set_yticklabels(unique_behaviors, fontsize=25, fontweight='bold')
            
            # 移除X轴刻度，让它只在主图上显示
            ax_behavior.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
            ax_behavior.set_title('Behavior Intervals', fontsize=35, pad=10)
            ax_behavior.set_xlabel('')
            
            # 确保行为图和主图水平对齐
            ax_behavior.set_xlim(ax_main.get_xlim())
            ax_behavior.set_anchor('SW')
            
            # 去除行为子图边框
            ax_behavior.spines['top'].set_visible(False)
            ax_behavior.spines['right'].set_visible(False)
            ax_behavior.spines['bottom'].set_visible(False)
            ax_behavior.spines['left'].set_visible(False)
            
            # 为每种行为绘制区间
            for behavior, intervals in behavior_intervals.items():
                behavior_color = fixed_color_map.get(behavior, plt.cm.tab10(list(unique_behaviors).index(behavior) % 10))
                
                for start_stamp, end_stamp in intervals:
                    # 如果区间有宽度
                    if end_stamp - start_stamp > 0:  
                        # 在行为标记子图中绘制区间
                        rect = plt.Rectangle(
                            (start_stamp, y_positions[behavior] - 0.4), 
                            end_stamp - start_stamp, 0.8, 
                            color=behavior_color, alpha=0.9, 
                            ec='black'  # 添加黑色边框以增强可见度
                        )
                        ax_behavior.add_patch(rect)
                        
                        # 在主图中添加区间边界垂直线
                        ax_main.axvline(x=start_stamp, color='gray', linestyle='--', linewidth=0.8, alpha=0.6)
                        ax_main.axvline(x=end_stamp, color='gray', linestyle='--', linewidth=0.8, alpha=0.6)
                
                # 添加到图例
                legend_patches.append(plt.Rectangle((0, 0), 1, 1, color=behavior_color, alpha=0.9, label=behavior))
            
            # 添加图例
            legend = ax_behavior.legend(
                handles=legend_patches, 
                loc='upper right', 
                fontsize=20, 
                title='Behavior Types', 
                title_fontsize=25,
                bbox_to_anchor=(1.0, 1.3)
            )
    
    # 调整布局
    if has_behavior and behavior_data.dropna().unique().size > 0:
        # 有行为图时的布局调整
        plt.subplots_adjust(top=0.88, bottom=0.1, left=0.08, right=0.95, hspace=0.12)
    else:
        # 无行为图时的布局调整
        plt.tight_layout()
    
    # 保存图像（如果指定了保存路径）
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        # 构建文件名，包含预处理信息
        filename = f'neuron_{neuron_id}_calcium_with_behavior'
        if apply_preprocessing:
            preprocessing_suffix = []
            if apply_moving_average:
                preprocessing_suffix.append(f"mavg{moving_avg_window}")
            if apply_butterworth:
                preprocessing_suffix.append(f"butter{butterworth_strength}")
            if apply_normalization:
                preprocessing_suffix.append(f"norm_{normalization_method}")
            if preprocessing_suffix:
                filename += f"_processed_{'_'.join(preprocessing_suffix)}"
        filename += ".png"
        
        save_path = os.path.join(save_dir, filename)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"图像已保存至: {save_path}")
    
    return fig

def plot_all_neurons(df: pd.DataFrame, save_dir: str = None, 
                     exclude_cols: List[str] = ['stamp', 'behavior'],
                     sampling_rate: float = 4.8,
                     apply_preprocessing: bool = True,
                     apply_moving_average: bool = True,
                     moving_avg_window: int = 3,
                     apply_butterworth: bool = True,
                     butterworth_cutoff: float = 20,
                     butterworth_strength: float = 0.05,
                     apply_normalization: bool = False,
                     normalization_method: str = 'standard',
                     feature_range: tuple = (0, 1)) -> List[Figure]:
    """
    绘制所有神经元的钙离子波动图，支持行为标签显示和预处理
    
    Parameters
    ----------
    df : pd.DataFrame
        包含神经元数据的DataFrame
    save_dir : str, optional
        图像保存目录，默认为None（由调用函数指定）
    exclude_cols : List[str], optional
        需要排除的列名列表，默认排除'stamp'和'behavior'列
    sampling_rate : float, optional
        采样频率，默认为4.8Hz（用于将时间戳转换为秒）
    apply_preprocessing : bool, optional
        是否应用预处理，默认为False
    apply_moving_average : bool, optional
        是否应用移动平均滤波，默认为True
    moving_avg_window : int, optional
        移动平均窗口大小，默认为3
    apply_butterworth : bool, optional
        是否应用Butterworth滤波，默认为True
    butterworth_cutoff : float, optional
        Butterworth滤波器截止频率，默认为20（与smooth_data.py保持一致）
    butterworth_strength : float, optional
        Butterworth滤波强度，默认为0.05（与smooth_data.py保持一致）
    apply_normalization : bool, optional
        是否应用归一化，默认为False
    normalization_method : str, optional
        归一化方法，默认为'standard'
    feature_range : tuple, optional
        用于minmax归一化的目标范围，默认为(0, 1)
    
    Returns
    -------
    List[Figure]
        所有图像对象的列表
    """
    # 确保保存目录存在
    if save_dir is None:
        save_dir = '../results/calcium_waves'
    os.makedirs(save_dir, exist_ok=True)
    
    # 使用element_extraction.py中的方法获取所有神经元列
    neuron_cols = [col for col in df.columns if col.startswith('n') and col[1:].isdigit()]
    print(f"检测到 {len(neuron_cols)} 个神经元数据列")
    
    if apply_preprocessing:
        print(f"启用预处理模式: 移动平均({apply_moving_average}), Butterworth({apply_butterworth}), 归一化({apply_normalization})")
    
    # 绘制并保存每个神经元的图像
    figures = []
    for neuron_id in neuron_cols:
        try:
            fig = plot_neuron_calcium(
                df, neuron_id, save_dir, sampling_rate,
                apply_preprocessing=apply_preprocessing,
                apply_moving_average=apply_moving_average,
                moving_avg_window=moving_avg_window,
                apply_butterworth=apply_butterworth,
                butterworth_cutoff=butterworth_cutoff,
                butterworth_strength=butterworth_strength,
                apply_normalization=apply_normalization,
                normalization_method=normalization_method,
                feature_range=feature_range
            )
            figures.append(fig)
            plt.close(fig)  # 关闭图像以释放内存
        except Exception as e:
            print(f"处理神经元 {neuron_id} 时出错: {e}")
    
    print(f"已为 {len(figures)} 个神经元生成钙离子波动图，保存至 {save_dir}")
    return figures

def generate_summary_report(df: pd.DataFrame, save_path: str = None) -> None:
    """
    生成神经元钙离子数据的摘要报告
    
    Parameters
    ----------
    df : pd.DataFrame
        包含神经元数据的DataFrame
    save_path : str, optional
        报告保存路径，默认为None（由调用函数指定）
    """
    # 确保目录存在
    if save_path is None:
        save_path = '../results/neuron_summary.html'
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    # 使用element_extraction.py中的方法获取所有神经元列
    neuron_cols = [col for col in df.columns if col.startswith('n') and col[1:].isdigit()]
    
    # 计算每个神经元的统计量
    stats = []
    for col in neuron_cols:
        stats.append({
            '神经元ID': col,
            '最大值': df[col].max(),
            '最小值': df[col].min(),
            '均值': df[col].mean(),
            '标准差': df[col].std(),
            '变异系数': df[col].std() / df[col].mean() if df[col].mean() != 0 else np.nan,
            '波动范围': df[col].max() - df[col].min()
        })
    
    # 创建统计摘要DataFrame
    stats_df = pd.DataFrame(stats)
    
    # 生成HTML报告
    with open(save_path, 'w', encoding='utf-8') as f:
        f.write('<html><head><title>神经元钙离子波动分析报告</title>')
        f.write('<style>body{font-family:Arial;margin:20px;} table{border-collapse:collapse;width:100%;}')
        f.write('th,td{border:1px solid #ddd;padding:8px;text-align:left;}')
        f.write('th{background-color:#f2f2f2;} tr:nth-child(even){background-color:#f9f9f9;}')
        f.write('</style></head><body>')
        f.write('<h1>神经元钙离子波动分析报告</h1>')
        f.write(f'<p>分析时间: {pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")}</p>')
        f.write(f'<p>数据样本数: {len(df)}</p>')
        f.write(f'<p>神经元数量: {len(neuron_cols)}</p>')
        f.write('<h2>神经元统计摘要</h2>')
        f.write(stats_df.to_html(index=False, float_format='%.4f'))
        f.write('</body></html>')
    
    print(f"分析报告已保存至: {save_path}")

def main():
    """
    主函数，用于处理命令行参数并执行可视化
    支持批量处理多个文件，支持行为标签显示
    """
    import argparse
    
    parser = argparse.ArgumentParser(description='神经元钙离子波动可视化工具（支持行为标签显示和预处理）')
    parser.add_argument('--data', type=str, nargs='+', 
                        default=['../datasets/EMtrace01.xlsx'],
                        help='数据文件路径列表，支持.md, .csv, .xlsx格式，可提供多个文件路径')
    parser.add_argument('--output', type=str, default=None,
                        help='图像保存根目录，不指定则使用../results/')
    parser.add_argument('--neuron', type=str, default=None,
                        help='指定要可视化的神经元ID，不指定则处理所有神经元')
    parser.add_argument('--report', action='store_true',
                        help='生成神经元统计摘要报告')
    parser.add_argument('--sampling-rate', type=float, default=4.8,
                        help='采样频率（Hz），现在仅用于保持兼容性，横坐标已改为时间戳，默认为4.8Hz')
    
    # 预处理相关参数（与smooth_data.py保持一致）
    parser.add_argument('--disable-preprocess', action='store_true',
                        help='禁用预处理模式（默认启用预处理，对数据进行滤波降噪处理）')
    parser.add_argument('--disable-moving-average', action='store_true',
                        help='禁用移动平均滤波（默认启用）')
    parser.add_argument('--moving-avg-window', type=int, default=3,
                        help='移动平均窗口大小，默认为3')
    parser.add_argument('--disable-butterworth', action='store_true',
                        help='禁用Butterworth滤波（默认启用）')
    parser.add_argument('--butterworth-cutoff', type=float, default=20,
                        help='Butterworth滤波器截止频率，默认为20（与smooth_data.py保持一致）')
    parser.add_argument('--butterworth-strength', type=float, default=0.05,
                        help='Butterworth滤波强度，默认为0.05（与smooth_data.py保持一致）')
    parser.add_argument('--enable-normalization', action='store_true',
                        help='启用数据归一化（默认禁用）')
    parser.add_argument('--normalization-method', type=str, default='standard',
                        choices=['standard', 'minmax', 'robust', 'log_standard', 'log_minmax'],
                        help='归一化方法，默认为standard')
    parser.add_argument('--feature-range', type=float, nargs=2, default=[0, 1],
                        help='用于minmax归一化的目标范围，默认为0 1')
    
    args = parser.parse_args()
    
    # 设置输出根目录
    output_root = args.output if args.output else "../results"
    
    # 对每个数据文件进行处理
    for data_file in args.data:
        print(f"\n开始处理文件: {data_file}")
        
        try:
            # 加载数据
            print(f"正在加载数据文件: {data_file}")
            df = load_data(data_file)
            print(f"数据加载完成，共有 {len(df)} 条记录和 {len(df.columns)} 个列")
            
            # 根据数据文件名生成输出目录
            # 提取数据文件名（不含扩展名）
            data_basename = os.path.basename(data_file)
            dataset_name = os.path.splitext(data_basename)[0]
            output_dir = os.path.join(output_root, dataset_name, "calcium_waves")
            
            print(f"输出目录设置为: {output_dir}")
            
            # 根据参数执行可视化
            if args.neuron:
                print(f"正在为神经元 {args.neuron} 生成图像...")
                plot_neuron_calcium(
                    df, args.neuron, output_dir, args.sampling_rate,
                    apply_preprocessing=not args.disable_preprocess,
                    apply_moving_average=not args.disable_moving_average,
                    moving_avg_window=args.moving_avg_window,
                    apply_butterworth=not args.disable_butterworth,
                    butterworth_cutoff=args.butterworth_cutoff,
                    butterworth_strength=args.butterworth_strength,
                    apply_normalization=args.enable_normalization,
                    normalization_method=args.normalization_method,
                    feature_range=tuple(args.feature_range)
                )
            else:
                print("正在为所有神经元生成图像...")
                plot_all_neurons(
                    df, output_dir, sampling_rate=args.sampling_rate,
                    apply_preprocessing=not args.disable_preprocess,
                    apply_moving_average=not args.disable_moving_average,
                    moving_avg_window=args.moving_avg_window,
                    apply_butterworth=not args.disable_butterworth,
                    butterworth_cutoff=args.butterworth_cutoff,
                    butterworth_strength=args.butterworth_strength,
                    apply_normalization=args.enable_normalization,
                    normalization_method=args.normalization_method,
                    feature_range=tuple(args.feature_range)
                )
            
            # 生成报告（如果需要）
            if args.report:
                print("正在生成统计摘要报告...")
                report_dir = os.path.dirname(output_dir)
                report_path = os.path.join(report_dir, f'{dataset_name}_summary.html')
                generate_summary_report(df, report_path)
                
            print(f"文件 {data_file} 处理完成!")
            
        except Exception as e:
            print(f"处理文件 {data_file} 时发生错误: {e}")
            print("继续处理下一个文件...")
    
    print("\n所有文件处理完成!")

if __name__ == "__main__":
    main()
