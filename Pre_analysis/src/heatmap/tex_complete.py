"""
完整增强版：整合所有功能的精简版本

功能特性：
1. 4种神经元排序：global/local/first/custom
2. 支持同一行为或不同行为的序列分析
3. 对比度增强（adaptive/percentile/power）
4. 保持opt1的简洁绘制风格
"""

import pandas as pd
from scipy import interpolate
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import os
from typing import List, Tuple, Optional
from matplotlib.colors import PowerNorm

class Config:
    """配置类"""
    def __init__(self):
        # === 基础配置 ===
        self.INPUT_FILE = './29790930糖水铁网糖水trace2.xlsx'
        self.OUTPUT_DIR = './behavior_heatmaps_complete'
        self.START_BEHAVIOR = 'Close'
        self.END_BEHAVIOR = 'Close-to-Middle'  # 可设置为与START_BEHAVIOR相同
        self.PRE_BEHAVIOR_TIME = 10.0
        self.SAMPLING_RATE = 4.8
        self.MIN_BEHAVIOR_DURATION = 1.0
        
        # === 排序配置 ===
        # 'peak' 或 'global' - 全局排序（基于全局数据峰值时间，所有热图使用相同顺序，便于比较）
        # 'local' - 局部排序（每个热图独立排序，突出各自特征）
        # 'first' - 首图排序（以第一个热图为基准）
        # 'custom' - 自定义排序（指定神经元顺序）
        self.SORT_METHOD = 'peak'  # 'peak'和'global'等价，保持向后兼容
        self.CUSTOM_NEURON_ORDER = ['n53', 'n40', 'n29', 'n34', 'n4', 'n32', 'n25']
        
        # === 对比度增强配置 ===
        self.COLORMAP = 'viridis'  # 'viridis', 'plasma', 'inferno', 'magma'
        self.CONTRAST_MODE = 'adaptive'  # 'adaptive'/'percentile'/'power'/'standard'
        self.VMIN_PERCENTILE = 5
        self.VMAX_PERCENTILE = 95
        self.POWER_GAMMA = 0.7
        self.APPLY_CLIPPING = True
        self.CLIP_PERCENTILE = 2


def load_data(file_path: str) -> pd.DataFrame:
    """加载数据"""
    print(f"加载数据: {file_path}")
    data = pd.read_excel(file_path)
    data = data.set_index('stamp')
    print(f"数据加载完成: {len(data)} 个时间点")
    return data


def find_behavior_pairs(data: pd.DataFrame, 
                       start_behavior: str,
                       end_behavior: str,
                       min_duration_stamps: float) -> List[Tuple[float, float, float, float]]:
    """
    查找行为配对（支持同一行为或不同行为）
    
    Returns:
        List of (start_begin, start_end, end_begin, end_end) tuples
    """
    behavior_data = data['behavior']
    change_mask = behavior_data != behavior_data.shift(1)
    merged_beh = behavior_data[change_mask]
    merged_idx = behavior_data.index[change_mask]
    
    pairs = []
    
    # 情况1: 同一行为（start == end）
    if start_behavior == end_behavior:
        for i in range(len(merged_beh)):
            if merged_beh.iloc[i] == start_behavior:
                start_begin = merged_idx[i]
                end_end = merged_idx[i + 1] if i + 1 < len(merged_idx) else data.index[-1]
                duration = end_end - start_begin
                
                if duration >= min_duration_stamps:
                    pairs.append((start_begin, end_end, start_begin, end_end))
                    print(f"找到行为 #{len(pairs)}: {start_behavior}({start_begin:.0f}-{end_end:.0f}, 时长:{duration:.0f})")
        
        print(f'共找到 {len(pairs)} 段 {start_behavior} 行为')
        return pairs
    
    # 情况2: 不同行为（start != end）- 查找连续配对
    i = 0
    while i < len(merged_beh) - 1:
        current_behavior = merged_beh.iloc[i]
        next_behavior = merged_beh.iloc[i + 1]
        
        if current_behavior == start_behavior and next_behavior == end_behavior:
            start_begin = merged_idx[i]
            start_end = merged_idx[i + 1]
            end_begin = merged_idx[i + 1]
            end_end = merged_idx[i + 2] if i + 2 < len(merged_idx) else data.index[-1]
            
            start_duration = start_end - start_begin
            end_duration = end_end - end_begin
            
            if start_duration >= min_duration_stamps and end_duration >= min_duration_stamps:
                pairs.append((start_begin, start_end, end_begin, end_end))
                print(f"找到配对 #{len(pairs)}: {start_behavior}({start_begin:.0f}-{start_end:.0f}) → {end_behavior}({end_begin:.0f}-{end_end:.0f})")
            
            i += 2
        else:
            i += 1
    
    print(f'共找到 {len(pairs)} 段 {start_behavior}→{end_behavior} 连续配对')
    return pairs


def sort_neurons_by_peak_time(data: pd.DataFrame) -> pd.DataFrame:
    """按峰值时间排序神经元（局部排序）"""
    peak_times = data.idxmax()
    sorted_neurons = peak_times.sort_values().index
    return data[sorted_neurons]


def apply_custom_neuron_order(data: pd.DataFrame, custom_order: List[str]) -> pd.DataFrame:
    """应用自定义神经元顺序"""
    available_neurons = set(data.columns)
    ordered_neurons = [n for n in custom_order if n in available_neurons]
    remaining_neurons = sorted(list(available_neurons - set(ordered_neurons)))
    final_order = ordered_neurons + remaining_neurons
    return data[final_order]


def extract_and_process_sequence(data: pd.DataFrame,
                                neural_data_standardized: pd.DataFrame,
                                start_time: float,
                                end_time: float,
                                pre_behavior_time: float,
                                sampling_rate: float,
                                config: Config,
                                global_neuron_order: Optional[pd.Index] = None,
                                first_neuron_order: Optional[pd.Index] = None) -> Optional[pd.DataFrame]:
    """提取并处理序列数据，应用排序"""
    # 计算实际时间范围
    actual_start = start_time - pre_behavior_time * sampling_rate
    actual_end = end_time
    
    if actual_start < data.index.min() or actual_end > data.index.max():
        print(f"  警告: 时间范围超出数据范围")
        return None
    
    # 提取数据
    mask = (neural_data_standardized.index >= actual_start) & (neural_data_standardized.index <= actual_end)
    sequence_data = neural_data_standardized.loc[mask].copy()
    
    # 应用排序
    if config.SORT_METHOD in ['global', 'peak'] and global_neuron_order is not None:
        # 全局排序（peak即为全局排序）
        sequence_data = sequence_data[global_neuron_order]
    elif config.SORT_METHOD == 'first' and first_neuron_order is not None:
        # 首图排序
        sequence_data = sequence_data[first_neuron_order]
    elif config.SORT_METHOD == 'local':
        # 局部排序
        sequence_data = sort_neurons_by_peak_time(sequence_data)
    elif config.SORT_METHOD == 'custom':
        # 自定义排序
        sequence_data = apply_custom_neuron_order(sequence_data, config.CUSTOM_NEURON_ORDER)
    
    return sequence_data


def enhance_contrast(data: np.ndarray, config: Config) -> Tuple[np.ndarray, float, float]:
    """增强数据对比度"""
    if config.CONTRAST_MODE == 'adaptive':
        vmin = np.percentile(data, config.VMIN_PERCENTILE)
        vmax = np.percentile(data, config.VMAX_PERCENTILE)
        if vmax - vmin < 0.5:
            vmin, vmax = -2, 2
        enhanced_data = data.copy()
        
    elif config.CONTRAST_MODE == 'percentile':
        vmin = np.percentile(data, config.CLIP_PERCENTILE)
        vmax = np.percentile(data, 100 - config.CLIP_PERCENTILE)
        enhanced_data = np.clip(data, vmin, vmax)
        
    elif config.CONTRAST_MODE == 'power':
        data_min, data_max = data.min(), data.max()
        data_norm = (data - data_min) / (data_max - data_min + 1e-10)
        enhanced_norm = np.power(data_norm, config.POWER_GAMMA)
        enhanced_data = enhanced_norm * (data_max - data_min) + data_min
        vmin, vmax = -2, 2
        
    else:  # 'standard'
        enhanced_data = data.copy()
        vmin, vmax = -2, 2
    
    # 可选剪裁
    if config.APPLY_CLIPPING and config.CONTRAST_MODE != 'percentile':
        clip_min = np.percentile(enhanced_data, config.CLIP_PERCENTILE)
        clip_max = np.percentile(enhanced_data, 100 - config.CLIP_PERCENTILE)
        enhanced_data = np.clip(enhanced_data, clip_min, clip_max)
    
    return enhanced_data, vmin, vmax


def create_heatmap(data: pd.DataFrame,
                  start_time: float,
                  end_time: float,
                  pre_behavior_time: float,
                  start_behavior: str,
                  end_behavior: str,
                  sampling_rate: float,
                  sequence_idx: int,
                  sort_method: str,
                  config: Config) -> plt.Figure:
    """创建单个序列的热图（支持对比度增强）"""
    # 应用对比度增强
    enhanced_data, vmin, vmax = enhance_contrast(data.values, config)
    enhanced_df = pd.DataFrame(enhanced_data, index=data.index, columns=data.columns)
    
    fig, ax = plt.subplots(figsize=(16, 60))
    
    sns.heatmap(
        enhanced_df.T,
        cmap=config.COLORMAP,
        cbar=False,
        vmin=vmin,
        vmax=vmax,
        ax=ax,
        xticklabels=False,
        yticklabels=True
    )
    
    # 设置标题
    if start_behavior == end_behavior:
        title = f'{start_behavior} #{sequence_idx + 1}'
    else:
        title = f'{start_behavior} → {end_behavior} #{sequence_idx + 1}'
    
    ax.set_title(title, fontsize=25, fontweight='bold')
    ax.set_xlabel('Time (seconds)', fontsize=30, fontweight='bold')
    ax.set_ylabel(f'Neurons (sorted by {sort_method})', fontsize=30, fontweight='bold')
    ax.set_yticklabels(ax.get_yticklabels(), fontsize=30, fontweight='bold', rotation=0)
    
    # 添加时间刻度
    tick_positions, tick_labels = calculate_5second_ticks(
        data.index,
        start_time - pre_behavior_time * sampling_rate,
        sampling_rate
    )
    ax.set_xticks(tick_positions)
    ax.set_xticklabels(tick_labels, fontsize=30, fontweight='bold', rotation=90)
    
    # 标记起始行为时间点
    start_behavior_pos = (start_time - (start_time - pre_behavior_time * sampling_rate)) / sampling_rate
    if start_behavior_pos < len(data):
        ax.axvline(x=start_behavior_pos, color='black', linestyle='--', linewidth=5, alpha=0.9)
    
    ax.text(start_behavior_pos + 1, -3, f'{start_behavior} Start',
           color='black', fontweight='bold', fontsize=12,
           bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
    
    plt.tight_layout()
    return fig


def calculate_5second_ticks(data_index: pd.Index,
                           reference_time: float,
                           sampling_rate: float) -> Tuple[List[float], List[str]]:
    """计算5秒间隔的时间刻度"""
    time_points_seconds = (data_index - reference_time) / sampling_rate
    min_time = time_points_seconds.min()
    max_time = time_points_seconds.max()
    
    tick_times_seconds = np.arange(
        np.floor(min_time / 5) * 5,
        np.ceil(max_time / 5) * 5 + 5,
        5
    )
    
    tick_positions = []
    tick_labels = []
    
    for tick_time in tick_times_seconds:
        if min_time <= tick_time <= max_time:
            relative_position = (tick_time - min_time) / (max_time - min_time)
            pixel_position = relative_position * (len(data_index) - 1)
            tick_positions.append(pixel_position)
            tick_labels.append(f'{tick_time:.0f}')
    
    return tick_positions, tick_labels


def create_average_heatmap(all_sequences: List[pd.DataFrame],
                          start_behavior: str,
                          end_behavior: str,
                          sort_method: str,
                          sampling_rate: float,
                          pre_behavior_time: float,
                          config: Config) -> plt.Figure:
    """创建平均热图（参照part.py思路，使用opt1风格）"""
    if not all_sequences:
        raise ValueError("没有有效的序列数据")
    
    print(f"创建平均热图，共 {len(all_sequences)} 个序列")
    
    # 找到公共神经元
    common_neurons = set(all_sequences[0].columns)
    for data in all_sequences[1:]:
        common_neurons &= set(data.columns)
    common_neurons = sorted(list(common_neurons))
    print(f"公共神经元: {len(common_neurons)}")
    
    # 重采样对齐
    min_length = min(len(data) for data in all_sequences)
    aligned_data = []
    
    for data in all_sequences:
        data_subset = data[common_neurons]
        if len(data_subset) != min_length:
            new_index = np.linspace(0, len(data_subset)-1, min_length)
            original_index = np.arange(len(data_subset))
            resampled_data = np.zeros((min_length, len(common_neurons)))
            for j, neuron in enumerate(common_neurons):
                resampled_data[:, j] = np.interp(new_index, original_index, data_subset[neuron].values)
            aligned_data.append(resampled_data)
        else:
            aligned_data.append(data_subset.values)
    
    # 计算平均
    average_data = np.mean(aligned_data, axis=0)
    
    # 增强对比度
    enhanced_data, vmin, vmax = enhance_contrast(average_data, config)
    
    # 创建DataFrame
    time_relative_timestamps = np.linspace(-pre_behavior_time * sampling_rate, 0, min_length)
    average_df = pd.DataFrame(enhanced_data, 
                             index=time_relative_timestamps, 
                             columns=common_neurons)
    
    # 创建热图
    fig, ax = plt.subplots(figsize=(16, 60))
    
    sns.heatmap(
        average_df.T,
        cmap=config.COLORMAP,
        cbar=False,
        vmin=vmin,
        vmax=vmax,
        ax=ax,
        xticklabels=False,
        yticklabels=True
    )
    
    # 设置标题
    if start_behavior == end_behavior:
        title = f'Average {start_behavior} (n={len(all_sequences)})'
    else:
        title = f'Average {start_behavior} → {end_behavior} (n={len(all_sequences)})'
    
    ax.set_title(title, fontsize=25, fontweight='bold')
    ax.set_xlabel('Time (seconds)', fontsize=30, fontweight='bold')
    ax.set_ylabel('Neurons', fontsize=30, fontweight='bold')
    ax.set_yticklabels(ax.get_yticklabels(), fontsize=30, fontweight='bold', rotation=0)
    
    # 添加时间刻度
    tick_positions, tick_labels = calculate_5second_ticks(
        average_df.index, 
        -pre_behavior_time * sampling_rate,
        sampling_rate
    )
    ax.set_xticks(tick_positions)
    ax.set_xticklabels(tick_labels, fontsize=30, fontweight='bold', rotation=90)
    
    # 标记起始行为时间点
    start_behavior_pos = (0 - (-pre_behavior_time * sampling_rate)) / sampling_rate
    if start_behavior_pos < len(average_df):
        ax.axvline(x=start_behavior_pos, color='black', linestyle='--', linewidth=5, alpha=0.9)
    
    ax.text(start_behavior_pos + 1, -3, f'{start_behavior} Start', 
           color='black', fontweight='bold', fontsize=12, 
           bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
    
    plt.tight_layout()
    return fig


def main():
    """主函数"""
    config = Config()
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)
    
    try:
        # 1. 加载数据
        data = load_data(config.INPUT_FILE)
        behavior_data_full = data['behavior']
        neural_data_full = data.drop(columns=['behavior'])
        
        # 2. 全局标准化
        print("计算全局标准化...")
        neural_data_full_standardized = (neural_data_full - neural_data_full.mean()) / neural_data_full.std()
        
        # 3. 计算全局神经元排序（用于global/peak模式）
        global_neuron_order = None
        if config.SORT_METHOD in ['global', 'peak']:  # peak和global等价
            print("计算全局神经元排序（基于峰值时间）...")
            peak_times = neural_data_full_standardized.idxmax()
            global_neuron_order = peak_times.sort_values().index
            print(f"全局排序完成，共 {len(global_neuron_order)} 个神经元")
        
        # 4. 查找行为配对
        if config.START_BEHAVIOR == config.END_BEHAVIOR:
            print(f"\n查找行为: {config.START_BEHAVIOR}")
        else:
            print(f"\n查找连续行为配对: {config.START_BEHAVIOR} → {config.END_BEHAVIOR}")
        
        min_duration_stamps = config.MIN_BEHAVIOR_DURATION * config.SAMPLING_RATE
        pairs = find_behavior_pairs(
            data,
            config.START_BEHAVIOR,
            config.END_BEHAVIOR,
            min_duration_stamps
        )
        
        if not pairs:
            print("未找到符合条件的行为配对")
            return
        
        print(f"找到 {len(pairs)} 个配对\n")
        
        # 5. 处理每个序列
        all_sequences = []
        first_neuron_order = None
        
        for i, (start_begin, start_end, end_begin, end_end) in enumerate(pairs):
            print(f"处理序列 {i+1}:")
            if config.START_BEHAVIOR == config.END_BEHAVIOR:
                print(f"  {config.START_BEHAVIOR}: {start_begin:.0f} - {end_end:.0f}")
            else:
                print(f"  {config.START_BEHAVIOR}: {start_begin:.0f} - {start_end:.0f}")
                print(f"  {config.END_BEHAVIOR}: {end_begin:.0f} - {end_end:.0f}")
            
            # 提取序列
            sequence_data = extract_and_process_sequence(
                data,
                neural_data_full_standardized,
                start_begin,
                end_end,
                config.PRE_BEHAVIOR_TIME,
                config.SAMPLING_RATE,
                config,
                global_neuron_order,
                first_neuron_order
            )
            
            if sequence_data is None:
                continue
            
            # 保存首图排序顺序
            if config.SORT_METHOD == 'first' and i == 0:
                first_neuron_order = sequence_data.columns
                print(f"  首图排序已确定")
            
            print(f"  提取数据: {len(sequence_data)} 个时间点, {len(sequence_data.columns)} 个神经元")
            all_sequences.append(sequence_data)
            
            # 创建热图
            fig = create_heatmap(
                sequence_data,
                start_begin,
                end_end,
                config.PRE_BEHAVIOR_TIME,
                config.START_BEHAVIOR,
                config.END_BEHAVIOR,
                config.SAMPLING_RATE,
                i,
                config.SORT_METHOD,
                config
            )
            
            # 保存
            if config.START_BEHAVIOR == config.END_BEHAVIOR:
                output_path = os.path.join(
                    config.OUTPUT_DIR,
                    f'{config.START_BEHAVIOR}_sequence_{i+1}.png'
                )
            else:
                output_path = os.path.join(
                    config.OUTPUT_DIR,
                    f'{config.START_BEHAVIOR}_to_{config.END_BEHAVIOR}_sequence_{i+1}.png'
                )
            
            fig.savefig(output_path, bbox_inches='tight', dpi=100)
            plt.close(fig)
            print(f"  已保存: {output_path}\n")
        
        # 6. 创建平均热图
        if all_sequences:
            print("="*60)
            print("创建平均热图...")
            print("="*60)
            
            avg_fig = create_average_heatmap(
                all_sequences,
                config.START_BEHAVIOR,
                config.END_BEHAVIOR,
                config.SORT_METHOD,
                config.SAMPLING_RATE,
                config.PRE_BEHAVIOR_TIME,
                config
            )
            
            if config.START_BEHAVIOR == config.END_BEHAVIOR:
                avg_path = os.path.join(
                    config.OUTPUT_DIR,
                    f'{config.START_BEHAVIOR}_average.png'
                )
            else:
                avg_path = os.path.join(
                    config.OUTPUT_DIR,
                    f'{config.START_BEHAVIOR}_to_{config.END_BEHAVIOR}_average.png'
                )
            
            avg_fig.savefig(avg_path, bbox_inches='tight', dpi=100)
            plt.close(avg_fig)
            print(f"已保存平均热图: {avg_path}")
        
        print(f"\n完成！共处理 {len(all_sequences)} 个序列")
        print(f"输出目录: {config.OUTPUT_DIR}")
        
    except Exception as e:
        print(f"错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()