#colorful
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import argparse
from scipy.signal import find_peaks
from scipy import stats

# 添加命令行参数解析
parser = argparse.ArgumentParser(description='神经元热图生成工具，支持不同排序方式')
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

# 函数：检测神经元第一次真实钙波发生的时间点
def detect_first_calcium_wave(neuron_data):
    """
    检测神经元第一次真实钙波发生的时间点
    
    参数:
    neuron_data -- 包含神经元活动的时间序列数据（标准化后）
    
    返回:
    first_wave_time -- 第一次真实钙波发生的时间点，如果没有检测到则返回数据最后一个时间点
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

# 设置绘图颜色范围
vmin, vmax = -2, 2  # 控制颜色对比度

# 绘制热图
plt.figure(figsize=(40, 10))

# Day3 热图
plt.subplot(1, 3, 1)
sns.heatmap(aligned_day3_df_std.T, cmap='RdYlBu', cbar=True, vmin=vmin, vmax=vmax)  # 转置 DataFrame 以调换横纵坐标
plt.title(f'Day3 ({sort_method_str})')
plt.xlabel('stamp')
plt.ylabel('neural')

# Day6 热图
plt.subplot(1, 3, 2)
sns.heatmap(aligned_day6_df_std.T, cmap='RdYlBu', cbar=True, vmin=vmin, vmax=vmax)  # 转置 DataFrame 以调换横纵坐标
plt.title(f'Day6 ({sort_method_str})')
plt.xlabel('stamp')
plt.ylabel('')

# Day9 热图
plt.subplot(1, 3, 3)
sns.heatmap(aligned_day9_df_std.T, cmap='RdYlBu', cbar=True, vmin=vmin, vmax=vmax)  # 转置 DataFrame 以调换横纵坐标
plt.title(f'Day9 ({sort_method_str})')
plt.xlabel('stamp')
plt.ylabel('')

plt.tight_layout()
plt.savefig(f'../../graph/heatmap_combined_complete_neurons_{SORT_METHOD}.png')
plt.close()
