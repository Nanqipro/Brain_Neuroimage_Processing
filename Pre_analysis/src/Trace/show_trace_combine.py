#colorful
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import argparse
from scipy.signal import find_peaks
from scipy import stats

# 添加命令行参数解析
parser = argparse.ArgumentParser(description='神经元trace图生成工具，支持不同排序方式')
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

# Trace图参数配置（严格按照show_trace.py的设置）
TRACE_OFFSET = 70       # 不同神经元trace之间的垂直偏移量（增加间隔以改善可读性）
SCALING_FACTOR = 40     # 信号振幅缩放因子（增加振幅以提高信号可见性）
MAX_NEURONS = 60        # 最大显示神经元数量（避免图表过于拥挤）
TRACE_ALPHA = 0.8       # trace线的透明度
LINE_WIDTH = 2.0        # trace线的宽度
SAMPLING_RATE = 4.8     # 采样频率，用于将时间戳转换为秒

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

# 按照show_trace.py的方法进行数据处理和排序
# 将'stamp'列设置为索引
day3_trace_data = day3_data.set_index('stamp')

# 检查是否存在'behavior'列并分离
if 'behavior' in day3_trace_data.columns:
    day3_trace_data = day3_trace_data.drop(columns=['behavior'])

print(f"Day3数据处理完成，神经元数量: {len(day3_trace_data.columns)}")

# 按照show_trace.py的方法：对整个数据框进行标准化，用于钙事件检测
day3_trace_data_standardized = (day3_trace_data - day3_trace_data.mean()) / day3_trace_data.std()

# 确定排序方式（严格按照show_trace.py的方法）
sort_method_str = ""
if SORT_METHOD == 'peak':
    # 使用与show_trace.py完全相同的排序方法
    # 对于每个神经元，找到其信号达到全局最大值的时间戳
    peak_times = day3_trace_data_standardized.idxmax()
    
    # 将神经元按照峰值时间从早到晚排序
    sorted_neurons_by_first_peak = peak_times.sort_values().index
    
    print(f"按峰值时间排序完成，早期峰值神经元: {list(sorted_neurons_by_first_peak[:5])}")
    
    # 创建字典存储峰值时间
    day3_peak_times = peak_times.to_dict()
    sort_method_str = "Sorted by peak time"
else:  # calcium_wave
    # 计算每个神经元在Day3中的第一次钙波时间
    first_wave_times = {}
    
    # 对每个神经元进行钙波检测
    for neuron in day3_trace_data_standardized.columns:
        neuron_data = day3_trace_data_standardized[neuron]
        first_wave_times[neuron] = detect_first_calcium_wave(neuron_data)
    
    # 转换为Series以便排序
    first_wave_times_series = pd.Series(first_wave_times)
    
    # 按第一次钙波时间排序
    sorted_neurons_by_first_peak = first_wave_times_series.sort_values().index
    
    print(f"按钙波时间排序完成，早期钙波神经元: {list(sorted_neurons_by_first_peak[:5])}")
    
    # 创建字典存储钙波时间
    day3_peak_times = first_wave_times
    sort_method_str = "Sorted by first calcium wave time"

# 根据对应表和排序结果准备数据
aligned_day3, aligned_day6, aligned_day9 = [], [], []
neuron_labels_day3, neuron_labels_day6, neuron_labels_day9 = [], [], []

# 基于已经排序的神经元顺序，按对应表提取三天的数据
for day3_neuron in sorted_neurons_by_first_peak:
    # 在对应表中查找这个Day3神经元对应的Day6和Day9神经元
    matching_row = correspondence_table[correspondence_table['Day3_with_behavior_labels_filled'] == day3_neuron]
    
    if not matching_row.empty:
        row = matching_row.iloc[0]
        day6_neuron = row['Day6_with_behavior_labels_filled']
        day9_neuron = row['Day9_with_behavior_labels_filled']
        
        # 检查三天是否都有数据
        if (pd.notna(day6_neuron) and day6_neuron in day6_data.columns and
            pd.notna(day9_neuron) and day9_neuron in day9_data.columns):
            
            # 三天都有数据，添加到相应列表（保持排序顺序）
            aligned_day3.append(day3_data[day3_neuron])
            neuron_labels_day3.append(day3_neuron)
            
            aligned_day6.append(day6_data[day6_neuron])
            neuron_labels_day6.append(day6_neuron)
            
            aligned_day9.append(day9_data[day9_neuron])
            neuron_labels_day9.append(day9_neuron)

# 打印保留的神经元数量
print(f"保留的神经元数量: {len(neuron_labels_day3)}")
print(f"排序方式: {sort_method_str}")

# 验证排序是否正确 - 打印最终列表中前5个神经元的排序时间
if len(neuron_labels_day3) >= 5:
    print("最终保留的前5个神经元及其排序时间:")
    for i in range(5):
        neuron = neuron_labels_day3[i]
        sort_time = day3_peak_times.get(neuron, 'N/A')
        print(f"  {i+1}. {neuron}: {sort_time}")

# 检查数据集中是否包含行为列
has_behavior_day3 = 'behavior' in day3_data.columns
has_behavior_day6 = 'behavior' in day6_data.columns
has_behavior_day9 = 'behavior' in day9_data.columns

# 找出CD1行为的时间点
cd1_indices_day3 = []
cd1_indices_day6 = []
cd1_indices_day9 = []

if has_behavior_day3:
    cd1_indices_day3 = day3_data[day3_data['behavior'] == 'CD1'].index.tolist()
if has_behavior_day6:
    cd1_indices_day6 = day6_data[day6_data['behavior'] == 'CD1'].index.tolist()
if has_behavior_day9:
    cd1_indices_day9 = day9_data[day9_data['behavior'] == 'CD1'].index.tolist()

# 绘制trace图（调整尺寸以适应三个子图的布局）
plt.figure(figsize=(60, 25))

# 函数：绘制单个trace图（严格按照show_trace.py的方法，三个子图Y轴对齐）
def plot_trace_subplot(ax, aligned_data, neuron_labels, day_data, cd1_indices, title, ylabel_visible=True, total_neurons=None):
    """
    绘制单个trace图子图，严格按照show_trace.py的方法，确保三个子图Y轴位置对齐
    
    Parameters
    ----------
    ax : matplotlib轴对象
        绘图轴
    aligned_data : list
        对齐的神经元数据列表
    neuron_labels : list
        神经元标签列表
    day_data : DataFrame
        原始数据（用于获取时间戳）
    cd1_indices : list
        CD1行为的时间点索引
    title : str
        子图标题
    ylabel_visible : bool
        是否显示Y轴标签
    total_neurons : int
        所有子图的统一神经元总数，用于对齐Y轴
    """
    
    # 使用统一的神经元总数来确保三个子图Y轴对齐
    if total_neurons is None:
        total_neurons = len(aligned_data)
    
    # 绘制每个神经元的trace（严格按照show_trace.py的方法）
    for i, (neuron_data, neuron_label) in enumerate(zip(aligned_data, neuron_labels)):
        if i >= MAX_NEURONS:  # 限制显示的神经元数量，避免图表过于拥挤
            break
        
        # 使用统一的基准计算位置，确保三个子图对齐
        position = min(total_neurons, MAX_NEURONS) - i
        
        # 按照show_trace.py的方法：使用原始数据而不是标准化数据，直接缩放并偏移
        ax.plot(
            day_data['stamp'] / SAMPLING_RATE,  # x轴是时间(秒) = 时间戳 / 采样率
            neuron_data * SCALING_FACTOR + position * TRACE_OFFSET,  # 应用缩放并偏移
            linewidth=LINE_WIDTH,
            alpha=TRACE_ALPHA,
            label=neuron_label
        )
    
    # 设置Y轴标签（使用统一的基准，确保三个子图对齐）
    if len(aligned_data) > 0:
        yticks = []
        ytick_labels = []
        
        # 为每个神经元计算正确的Y轴位置和标签，使用统一基准
        for i, neuron_label in enumerate(neuron_labels[:MAX_NEURONS]):
            position = min(total_neurons, MAX_NEURONS) - i
            yticks.append(position * TRACE_OFFSET)
            ytick_labels.append(str(neuron_label))
        
        ax.set_yticks(yticks)
        if ylabel_visible:
            ax.set_yticklabels(ytick_labels, fontsize=20)
        else:
            ax.set_yticklabels([])
        
        # 设置统一的Y轴范围，确保三个子图对齐，并为最上面的trace留出振幅空间
        y_max = min(total_neurons, MAX_NEURONS) * TRACE_OFFSET + TRACE_OFFSET
        # 为最上面的trace信号振幅预留额外空间（大约是缩放因子的一半）
        y_max += SCALING_FACTOR * 0.6  
        ax.set_ylim(0, y_max)
    
    # 标记CD1行为
    if len(cd1_indices) > 0:
        for cd1_time in cd1_indices:
            cd1_time_seconds = cd1_time / SAMPLING_RATE
            ax.axvline(x=cd1_time_seconds, color='red', linestyle='--', linewidth=2, alpha=0.8)
            ax.text(cd1_time_seconds, ax.get_ylim()[1] * 0.95, 'CD1', 
                   color='red', rotation=90, verticalalignment='top', fontsize=12, fontweight='bold')
    
    # 设置标题和标签（放大字体以提高可读性）
    ax.set_title(title, fontsize=30, pad=20)
    ax.set_xlabel('Time (seconds)', fontsize=35)
    if ylabel_visible:
        ax.set_ylabel('Neuron ID', fontsize=35)
    
    # 设置刻度标签字体大小（放大以提高可读性）
    ax.tick_params(axis='both', labelsize=35)
    
    # 添加网格
    ax.grid(False)

# 计算统一的神经元总数，用于确保三个子图Y轴对齐
total_neurons_count = len(neuron_labels_day3)  # 使用Day3的神经元数量作为基准

# Day3 trace图
plt.subplot(1, 3, 1)
ax1 = plt.gca()
plot_trace_subplot(ax1, aligned_day3, neuron_labels_day3, day3_data, cd1_indices_day3, 
                  f'Day3 ({sort_method_str})', ylabel_visible=True, total_neurons=total_neurons_count)

# Day6 trace图
plt.subplot(1, 3, 2)
ax2 = plt.gca()
plot_trace_subplot(ax2, aligned_day6, neuron_labels_day6, day6_data, cd1_indices_day6, 
                  f'Day6 (Using Day3 {sort_method_str})', ylabel_visible=True, total_neurons=total_neurons_count)

# Day9 trace图
plt.subplot(1, 3, 3)
ax3 = plt.gca()
plot_trace_subplot(ax3, aligned_day9, neuron_labels_day9, day9_data, cd1_indices_day9, 
                  f'Day9 (Using Day3 {sort_method_str})', ylabel_visible=True, total_neurons=total_neurons_count)

plt.tight_layout()
plt.savefig(f'../../graph/traces_combined_sorted_complete_neurons_{SORT_METHOD}.png', dpi=300)
plt.close()

print(f"Trace图已保存到: ../../graph/traces_combined_sorted_complete_neurons_{SORT_METHOD}.png")
