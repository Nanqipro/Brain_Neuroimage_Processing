#colorful
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import argparse
from scipy.signal import find_peaks
from scipy import stats
import matplotlib.cm as cm
from matplotlib.colors import ListedColormap

# 添加命令行参数解析
parser = argparse.ArgumentParser(description='神经元Trace图生成工具，支持不同排序方式')
parser.add_argument('--sort-method', type=str, choices=['none', 'peak', 'calcium_wave'], default='none',
                    help='排序方式：none（无排序）、peak（按峰值时间排序）或calcium_wave（按第一次真实钙波时间排序）')
parser.add_argument('--ca-threshold', type=float, default=1.5, help='钙波检测阈值（标准差的倍数）')
parser.add_argument('--min-prominence', type=float, default=1.0, help='最小峰值突出度')
parser.add_argument('--min-rise-rate', type=float, default=0.1, help='最小上升速率')
parser.add_argument('--max-fall-rate', type=float, default=0.05, help='最大下降速率')
args = parser.parse_args()

# 设置排序方式和钙波检测参数
SORT_METHOD = args.sort_method
CALCIUM_WAVE_THRESHOLD = args.ca_threshold
MIN_PROMINENCE = args.min_prominence
MIN_RISE_RATE = args.min_rise_rate
MAX_FALL_RATE = args.max_fall_rate

# Trace图绘制参数配置
class TraceConfig:
    """
    Trace图绘制参数配置类
    """
    TRACE_OFFSET = 60  # 不同神经元trace之间的垂直偏移量
    SCALING_FACTOR = 80  # 信号振幅缩放因子
    TRACE_ALPHA = 0.8   # trace线的透明度
    LINE_WIDTH = 2.0    # trace线的宽度 (增加线宽以匹配 show_trace.py)
    SAMPLING_RATE = 4.8  # 采样频率，用于将时间戳转换为秒

def generate_distinct_colors(n_colors):
    """
    生成n个尽可能不同的颜色
    
    Parameters
    ----------
    n_colors : int
        需要生成的颜色数量
        
    Returns
    -------
    list
        包含颜色的列表
    """
    if n_colors <= 20:
        # 使用预定义的20种明显不同的颜色
        colors = [
            '#FF0000', '#00FF00', '#0000FF', '#FFFF00', '#FF00FF',
            '#00FFFF', '#800000', '#008000', '#000080', '#808000',
            '#800080', '#008080', '#FFA500', '#A52A2A', '#DDA0DD',
            '#90EE90', '#FFB6C1', '#87CEEB', '#F0E68C', '#D2691E'
        ]
        return colors[:n_colors]
    else:
        # 对于更多颜色，使用colormap生成
        try:
            colormap = plt.cm.get_cmap('tab20')
        except:
            colormap = cm.get_cmap('tab20')
        base_colors = [colormap(i) for i in range(20)]
        
        # 如果需要更多颜色，使用其他colormap补充
        if n_colors > 20:
            try:
                colormap2 = plt.cm.get_cmap('Set3')
            except:
                colormap2 = cm.get_cmap('Set3')
            additional_colors = [colormap2(i % 12) for i in range(n_colors - 20)]
            return base_colors + additional_colors
        
        return base_colors

def assign_neuron_colors(neuron_list, correspondence_table):
    """
    为神经元分配颜色，确保同一组神经元颜色相同，相邻神经元颜色不同
    
    Parameters
    ----------
    neuron_list : list
        神经元ID列表
    correspondence_table : pd.DataFrame
        神经元对应表
        
    Returns
    -------
    dict
        神经元ID到颜色的映射字典
    """
    # 创建神经元组的映射（每一行是一组对应的神经元）
    neuron_groups = {}
    group_colors = {}
    
    # 生成足够的颜色
    n_groups = len(correspondence_table)
    colors = generate_distinct_colors(n_groups)
    
    # 为每个组分配颜色
    for group_idx, (_, row) in enumerate(correspondence_table.iterrows()):
        group_color = colors[group_idx % len(colors)]
        
        # 将这一组的所有神经元都映射到相同颜色
        for col in correspondence_table.columns:
            neuron_id = row[col]
            if pd.notna(neuron_id):
                neuron_groups[neuron_id] = group_idx
                group_colors[neuron_id] = group_color
    
    # 为神经元列表中的每个神经元分配颜色
    neuron_color_map = {}
    default_color = '#000000'  # 默认黑色
    
    for neuron in neuron_list:
        neuron_color_map[neuron] = group_colors.get(neuron, default_color)
    
    return neuron_color_map

# 函数：检测神经元第一次真实钙波发生的时间点
def detect_first_calcium_wave(neuron_data):
    """
    检测神经元第一次真实钙波发生的时间点
    
    Parameters
    ----------
    neuron_data : pd.Series
        包含神经元活动的时间序列数据（标准化后）
    
    Returns
    -------
    int
        第一次真实钙波发生的时间点，如果没有检测到则返回数据最后一个时间点
    """
    # 计算阈值（基于数据的标准差）
    threshold = CALCIUM_WAVE_THRESHOLD
    
    # 使用find_peaks函数检测峰值
    peaks, properties = find_peaks(neuron_data, 
                                 height=threshold, 
                                 prominence=MIN_PROMINENCE,
                                 distance=5)  # 要求峰值之间至少间隔5个时间点
    
    if len(peaks) == 0:
        # 如果没有检测到峰值，返回时间序列的最后一个点
        return neuron_data.index[-1]
    
    # 对每个峰值进行验证，确认是否为真实钙波（上升快，下降慢）
    for peak_idx in peaks:
        # 确保峰值不在时间序列的开始或结束处
        if peak_idx <= 1 or peak_idx >= len(neuron_data) - 2:
            continue
            
        # 计算峰值前的上升速率（取峰值前5个点或更少）
        pre_peak_idx = max(0, peak_idx - 5)
        rise_rate = (neuron_data.iloc[peak_idx] - neuron_data.iloc[pre_peak_idx]) / (peak_idx - pre_peak_idx)
        
        # 计算峰值后的下降速率（取峰值后10个点或更少）
        post_peak_idx = min(len(neuron_data) - 1, peak_idx + 10)
        if post_peak_idx <= peak_idx:
            continue
        
        fall_rate = (neuron_data.iloc[peak_idx] - neuron_data.iloc[post_peak_idx]) / (post_peak_idx - peak_idx)
        
        # 确认是否符合钙波特征：上升快，下降慢
        if rise_rate > MIN_RISE_RATE and 0 < fall_rate < MAX_FALL_RATE:
            # 找到第一个真实钙波，返回时间点
            return neuron_data.index[peak_idx]
    
    # 如果没有满足条件的钙波，返回时间序列的最后一个点
    return neuron_data.index[-1]

def plot_trace_subplot(ax, data_df, neuron_color_map, title, sort_method_str, max_neurons=60):
    """
    绘制单个trace子图
    
    Parameters
    ----------
    ax : matplotlib.axes.Axes
        绘图轴对象
    data_df : pd.DataFrame
        标准化后的数据
    neuron_color_map : dict
        神经元到颜色的映射
    title : str
        子图标题
    sort_method_str : str
        排序方法描述
    max_neurons : int
        最大显示神经元数量
    """
    # 绘制每个神经元的trace
    for i, column in enumerate(data_df.columns):
        if i >= max_neurons:
            break
        
        # 获取神经元颜色
        neuron_color = neuron_color_map.get(column, '#000000')
        
        # 计算当前神经元trace的垂直偏移量
        ax.plot(
            data_df.index / TraceConfig.SAMPLING_RATE,  # x轴是时间(秒)
            data_df[column] * TraceConfig.SCALING_FACTOR + (i+1) * TraceConfig.TRACE_OFFSET,
            linewidth=TraceConfig.LINE_WIDTH,
            alpha=TraceConfig.TRACE_ALPHA,
            color=neuron_color,
            label=column
        )
    
    # 设置Y轴标签
    yticks = np.arange(TraceConfig.TRACE_OFFSET, 
                      (min(max_neurons, len(data_df.columns))+1) * TraceConfig.TRACE_OFFSET, 
                      TraceConfig.TRACE_OFFSET)
    ytick_labels = [str(column) for column in data_df.columns[:max_neurons]]
    ax.set_yticks(yticks[:len(ytick_labels)])
    ax.set_yticklabels(ytick_labels, fontsize=12)  # 增大Y轴标签字体
    
    # 设置轴标签和标题 (增大字体以匹配 show_trace.py 风格)
    ax.set_xlabel('Time (seconds)', fontsize=25)
    ax.set_ylabel('Neuron ID', fontsize=25)
    ax.set_title(f'{title} ({sort_method_str})', fontsize=20)  # 稍微调整标题字体大小
    
    # 设置刻度标签字体大小 (增大以匹配 show_trace.py)
    ax.tick_params(axis='both', labelsize=15)
    
    # 取消网格线以匹配 show_trace.py 风格
    ax.grid(False)

# 加载数据
day3_data = pd.read_excel('../../datasets/Day3_with_behavior_labels_filled.xlsx')
day6_data = pd.read_excel('../../datasets/Day6_with_behavior_labels_filled.xlsx')
day9_data = pd.read_excel('../../datasets/Day9_with_behavior_labels_filled.xlsx')
correspondence_table = pd.read_excel('../../datasets/神经元对应表2979.xlsx')

# 根据对应表准备数据
aligned_day3, aligned_day6, aligned_day9 = [], [], []
neuron_labels_day3, neuron_labels_day6, neuron_labels_day9 = [], [], []

# 筛选三天都有数据的神经元
for _, row in correspondence_table.iterrows():
    day3_neuron, day6_neuron, day9_neuron = row['Day3_with_behavior_labels_filled'], row['Day6_with_behavior_labels_filled'], row['Day9_with_behavior_labels_filled']
    
    # 检查三天是否都有数据 (都不为null且对应列存在于数据中)
    if (pd.notna(day3_neuron) and day3_neuron in day3_data.columns and
        pd.notna(day6_neuron) and day6_neuron in day6_data.columns and
        pd.notna(day9_neuron) and day9_neuron in day9_data.columns):
        
        # 三天都有数据，添加到相应列表
        aligned_day3.append(day3_data[day3_neuron])
        neuron_labels_day3.append(day3_neuron)
        
        aligned_day6.append(day6_data[day6_neuron])
        neuron_labels_day6.append(day6_neuron)
        
        aligned_day9.append(day9_data[day9_neuron])
        neuron_labels_day9.append(day9_neuron)

# 将列表转换为 DataFrame，并设置索引为时间戳 (stamp)，再转置以交换横纵坐标
aligned_day3_df = pd.DataFrame(aligned_day3, index=neuron_labels_day3).T
aligned_day6_df = pd.DataFrame(aligned_day6, index=neuron_labels_day6).T
aligned_day9_df = pd.DataFrame(aligned_day9, index=neuron_labels_day9).T

# 打印保留的神经元数量
print(f"保留的神经元数量: {len(neuron_labels_day3)}")

# 标准化数据
aligned_day3_df_std = (aligned_day3_df - aligned_day3_df.mean()) / aligned_day3_df.std()
aligned_day6_df_std = (aligned_day6_df - aligned_day6_df.mean()) / aligned_day6_df.std()
aligned_day9_df_std = (aligned_day9_df - aligned_day9_df.mean()) / aligned_day9_df.std()

# 生成颜色映射
neuron_color_map_day3 = assign_neuron_colors(neuron_labels_day3, correspondence_table)
neuron_color_map_day6 = assign_neuron_colors(neuron_labels_day6, correspondence_table)
neuron_color_map_day9 = assign_neuron_colors(neuron_labels_day9, correspondence_table)

# 对神经元进行排序（如果需要）
sort_method_str = "No sorting"
if SORT_METHOD != 'none':
    # 获取排序索引
    if SORT_METHOD == 'peak':
        # 按峰值时间排序
        peak_times = aligned_day3_df_std.idxmax(axis=0)  # 找到每列的最大值所在的索引
        sorting_indices = peak_times.sort_values().index
        sort_method_str = "Sorted by peak time"
    else:  # calcium_wave
        # 按第一次钙波时间排序
        first_wave_times = {}
        for neuron in aligned_day3_df_std.columns:
            neuron_data = aligned_day3_df_std[neuron]
            first_wave_times[neuron] = detect_first_calcium_wave(neuron_data)
        
        first_wave_times_series = pd.Series(first_wave_times)
        sorting_indices = first_wave_times_series.sort_values().index
        sort_method_str = "Sorted by first calcium wave time"
    
    # 根据排序索引重新排列数据
    aligned_day3_df_std = aligned_day3_df_std[sorting_indices]
    aligned_day6_df_std = aligned_day6_df_std[sorting_indices]
    aligned_day9_df_std = aligned_day9_df_std[sorting_indices]
    
    # 重新生成排序后的颜色映射
    neuron_color_map_day3 = assign_neuron_colors(list(sorting_indices), correspondence_table)
    neuron_color_map_day6 = assign_neuron_colors(list(sorting_indices), correspondence_table)
    neuron_color_map_day9 = assign_neuron_colors(list(sorting_indices), correspondence_table)

# 绘制Trace图
plt.figure(figsize=(60, 20))

# Day3 Trace图
plt.subplot(1, 3, 1)
ax1 = plt.gca()
plot_trace_subplot(ax1, aligned_day3_df_std, neuron_color_map_day3, 'Day3', sort_method_str)

# Day6 Trace图
plt.subplot(1, 3, 2)
ax2 = plt.gca()
plot_trace_subplot(ax2, aligned_day6_df_std, neuron_color_map_day6, 'Day6', sort_method_str)

# Day9 Trace图
plt.subplot(1, 3, 3)
ax3 = plt.gca()
plot_trace_subplot(ax3, aligned_day9_df_std, neuron_color_map_day9, 'Day9', sort_method_str)

# 添加总标题
plt.suptitle(f'Combined Neural Traces Across Three Days ({sort_method_str})', fontsize=25, y=0.98)

# 调整布局 (使用 subplots_adjust 代替 tight_layout 以匹配 show_trace.py 风格)
plt.subplots_adjust(top=0.90, bottom=0.1, left=0.1, right=0.95, wspace=0.3)

# 保存图像
import os
output_dir = '../../graph/'
os.makedirs(output_dir, exist_ok=True)
output_filename = f'{output_dir}traces_combined_complete_neurons_{SORT_METHOD}.png'
plt.savefig(output_filename, dpi=300, bbox_inches='tight')
print(f"Trace图已保存到: {output_filename}")

plt.close()
print("程序执行完成")
