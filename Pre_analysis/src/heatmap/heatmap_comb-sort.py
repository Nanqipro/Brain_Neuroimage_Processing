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
parser.add_argument('--sort-method', type=str, choices=['peak', 'calcium_wave'], default='peak',
                    help='排序方式：peak（按峰值时间排序）或calcium_wave（按第一次真实钙波时间排序）')
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
day0_data = pd.read_excel('../../datasets/No.297920240925homecagefamilarmice.xlsx')
correspondence_table = pd.read_excel('../../datasets/神经元对应表2979.xlsx')

# 首先创建Day3数据的标准化副本，用于计算峰值时间或钙波时间
day3_standardized = {}
for col in day3_data.columns:
    if col != 'stamp' and col != 'behavior':
        # 确保只处理神经元数据列
        series = day3_data[col]
        # 标准化数据
        day3_standardized[col] = (series - series.mean()) / series.std()

# 确定排序方式
sort_method_str = ""
if SORT_METHOD == 'peak':
    # 计算每个神经元在Day3中的峰值时间
    day3_peak_times = {}
    for neuron, data in day3_standardized.items():
        # 找出最大值所在的索引位置
        peak_idx = data.idxmax()
        day3_peak_times[neuron] = peak_idx
    sort_method_str = "Sorted by peak time"
else:  # calcium_wave
    # 计算每个神经元在Day3中的第一次钙波时间
    day3_peak_times = {}
    for neuron, data in day3_standardized.items():
        # 重建Series以便使用detect_first_calcium_wave函数
        series = pd.Series(data.values, index=data.index, name=neuron)
        day3_peak_times[neuron] = detect_first_calcium_wave(series)
    sort_method_str = "Sorted by first calcium wave time"

# 根据对应表准备数据
aligned_day0, aligned_day3, aligned_day6, aligned_day9 = [], [], [], []
neuron_labels_day0, neuron_labels_day3, neuron_labels_day6, neuron_labels_day9 = [], [], [], []
peak_times_order = []  # 用于记录对应表中每行神经元的峰值时间或钙波时间

# 筛选四天都有数据的神经元
valid_indices = []
for idx, row in correspondence_table.iterrows():
    day0_neuron = row.get('0925homecagefamilarmice', None)  # 使用get方法避免KeyError
    day3_neuron, day6_neuron, day9_neuron = row['Day3_with_behavior_labels_filled'], row['Day6_with_behavior_labels_filled'], row['Day9_with_behavior_labels_filled']
    
    # 检查四天是否都有数据 (都不为null且对应列存在于数据中)
    if (pd.notna(day0_neuron) and day0_neuron in day0_data.columns and
        pd.notna(day3_neuron) and day3_neuron in day3_data.columns and
        pd.notna(day6_neuron) and day6_neuron in day6_data.columns and
        pd.notna(day9_neuron) and day9_neuron in day9_data.columns):
        
        # 记录Day3神经元的峰值时间或钙波时间
        sort_time = None
        if day3_neuron in day3_peak_times:
            sort_time = day3_peak_times[day3_neuron]
        peak_times_order.append((sort_time, len(valid_indices)))  # 保存排序时间和新的索引
        valid_indices.append(idx)
        
        # 四天都有数据，添加到相应列表
        aligned_day0.append(day0_data[day0_neuron])
        neuron_labels_day0.append(day0_neuron)
        
        aligned_day3.append(day3_data[day3_neuron])
        neuron_labels_day3.append(day3_neuron)
        
        aligned_day6.append(day6_data[day6_neuron])
        neuron_labels_day6.append(day6_neuron)
        
        aligned_day9.append(day9_data[day9_neuron])
        neuron_labels_day9.append(day9_neuron)

# 打印保留的神经元数量
print(f"保留的神经元数量: {len(neuron_labels_day3)}")

# 按照Day3的峰值或钙波时间排序（先去除None值的行再排序）
valid_peak_times = [(t, i) for t, i in peak_times_order if t is not None]
sorted_indices = [i for _, i in sorted(valid_peak_times, key=lambda x: x[0])]
# 将None值的行添加到最后
none_indices = [i for t, i in peak_times_order if t is None]
sorted_indices.extend(none_indices)

# 根据排序后的索引重新排列数据
aligned_day0 = [aligned_day0[i] for i in sorted_indices]
aligned_day3 = [aligned_day3[i] for i in sorted_indices]
aligned_day6 = [aligned_day6[i] for i in sorted_indices]
aligned_day9 = [aligned_day9[i] for i in sorted_indices]
neuron_labels_day0 = [neuron_labels_day0[i] for i in sorted_indices]
neuron_labels_day3 = [neuron_labels_day3[i] for i in sorted_indices]
neuron_labels_day6 = [neuron_labels_day6[i] for i in sorted_indices]
neuron_labels_day9 = [neuron_labels_day9[i] for i in sorted_indices]

# 将列表转换为 DataFrame，并设置索引为时间戳 (stamp)，再转置以交换横纵坐标
aligned_day0_df = pd.DataFrame(aligned_day0, index=neuron_labels_day0).T
aligned_day3_df = pd.DataFrame(aligned_day3, index=neuron_labels_day3).T
aligned_day6_df = pd.DataFrame(aligned_day6, index=neuron_labels_day6).T
aligned_day9_df = pd.DataFrame(aligned_day9, index=neuron_labels_day9).T

# 标准化数据
aligned_day0_df = (aligned_day0_df - aligned_day0_df.mean()) / aligned_day0_df.std()
aligned_day3_df = (aligned_day3_df - aligned_day3_df.mean()) / aligned_day3_df.std()
aligned_day6_df = (aligned_day6_df - aligned_day6_df.mean()) / aligned_day6_df.std()
aligned_day9_df = (aligned_day9_df - aligned_day9_df.mean()) / aligned_day9_df.std()

# 设置绘图颜色范围
vmin, vmax = -2, 2  # 控制颜色对比度

# 检查数据集中是否包含行为列
has_behavior_day0 = 'behavior' in day0_data.columns
has_behavior_day3 = 'behavior' in day3_data.columns
has_behavior_day6 = 'behavior' in day6_data.columns
has_behavior_day9 = 'behavior' in day9_data.columns

# 找出CD1行为的时间点
cd1_indices_day0 = []
cd1_indices_day3 = []
cd1_indices_day6 = []
cd1_indices_day9 = []

if has_behavior_day0:
    cd1_indices_day0 = day0_data[day0_data['behavior'] == 'CD1'].index.tolist()
if has_behavior_day3:
    cd1_indices_day3 = day3_data[day3_data['behavior'] == 'CD1'].index.tolist()
if has_behavior_day6:
    cd1_indices_day6 = day6_data[day6_data['behavior'] == 'CD1'].index.tolist()
if has_behavior_day9:
    cd1_indices_day9 = day9_data[day9_data['behavior'] == 'CD1'].index.tolist()

# 绘制热图
plt.figure(figsize=(53, 10))

# Day0 热图
plt.subplot(1, 4, 1)
ax0 = sns.heatmap(aligned_day0_df.T, cmap='viridis', cbar=True, vmin=vmin, vmax=vmax)  # 转置 DataFrame 以调换横纵坐标
plt.title(f'Day0 (Using Day3 {sort_method_str})', fontsize=25)
plt.xlabel('Stamp', fontsize=25)
plt.ylabel('Neuron', fontsize=25)

# 标记Day0的CD1行为
if has_behavior_day0 and len(cd1_indices_day0) > 0:
    for cd1_time in cd1_indices_day0:
        if cd1_time in aligned_day0_df.index:
            position = aligned_day0_df.index.get_loc(cd1_time)
            ax0.axvline(x=position, color='white', linestyle='--', linewidth=2)
            plt.text(position + 0.5, -5, 'CD1', 
                    color='black', rotation=90, verticalalignment='top', fontsize=20, fontweight='bold')

# Day3 热图
plt.subplot(1, 4, 2)
ax1 = sns.heatmap(aligned_day3_df.T, cmap='viridis', cbar=True, vmin=vmin, vmax=vmax)  # 转置 DataFrame 以调换横纵坐标
plt.title(f'Day3 ({sort_method_str})', fontsize=25)
plt.xlabel('Stamp', fontsize=25)
plt.ylabel('')

# 标记Day3的CD1行为
if has_behavior_day3 and len(cd1_indices_day3) > 0:
    for cd1_time in cd1_indices_day3:
        if cd1_time in aligned_day3_df.index:
            position = aligned_day3_df.index.get_loc(cd1_time)
            ax1.axvline(x=position, color='white', linestyle='--', linewidth=2)
            plt.text(position + 0.5, -5, 'CD1', 
                    color='black', rotation=90, verticalalignment='top', fontsize=20, fontweight='bold')

# Day6 热图
plt.subplot(1, 4, 3)
ax2 = sns.heatmap(aligned_day6_df.T, cmap='viridis', cbar=True, vmin=vmin, vmax=vmax)  # 转置 DataFrame 以调换横纵坐标
plt.title(f'Day6 (Using Day3 {sort_method_str})', fontsize=25)
plt.xlabel('Stamp', fontsize=25)
plt.ylabel('')

# 标记Day6的CD1行为
if has_behavior_day6 and len(cd1_indices_day6) > 0:
    for cd1_time in cd1_indices_day6:
        if cd1_time in aligned_day6_df.index:
            position = aligned_day6_df.index.get_loc(cd1_time)
            ax2.axvline(x=position, color='white', linestyle='--', linewidth=2)
            plt.text(position + 0.5, -5, 'CD1', 
                    color='black', rotation=90, verticalalignment='top', fontsize=20, fontweight='bold')

# Day9 热图
plt.subplot(1, 4, 4)
ax3 = sns.heatmap(aligned_day9_df.T, cmap='viridis', cbar=True, vmin=vmin, vmax=vmax)  # 转置 DataFrame 以调换横纵坐标
plt.title(f'Day9 (Using Day3 {sort_method_str})', fontsize=25)
plt.xlabel('Stamp', fontsize=25)
plt.ylabel('')

# 标记Day9的CD1行为
if has_behavior_day9 and len(cd1_indices_day9) > 0:
    for cd1_time in cd1_indices_day9:
        if cd1_time in aligned_day9_df.index:
            position = aligned_day9_df.index.get_loc(cd1_time)
            ax3.axvline(x=position, color='white', linestyle='--', linewidth=2)
            plt.text(position + 0.5, -5, 'CD1', 
                    color='black', rotation=90, verticalalignment='top', fontsize=20, fontweight='bold')

plt.tight_layout()
plt.savefig(f'../../graph/heatmap_combined_sorted_complete_neurons_{SORT_METHOD}.png')
plt.close()
