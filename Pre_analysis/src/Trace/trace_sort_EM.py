# 神经元活动 Trace 图绘制，基于热图代码修改而来
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import argparse
from matplotlib.gridspec import GridSpec
import matplotlib.patches as mpatches
import os

# 确保matplotlib能够正确处理大线宽
plt.rcParams['lines.linewidth'] = 2.0  # 设置默认线宽

# 简化后的参数配置类
class Config:
    # 输入文件路径
    INPUT_FILE = '../../datasets/NO5355homecage1111.xlsx'
    # 输出目录
    OUTPUT_DIR = '../../graph/'
    # 时间戳区间默认值（None表示不限制）
    STAMP_MIN = 18000  # 最小时间戳
    STAMP_MAX = 19998  # 最大时间戳
    # 排序方式：'original'（原始顺序）、'peak'（按峰值时间排序）、'calcium_wave'（按第一次真实钙波发生时间排序）或'custom'（按自定义顺序排序）
    SORT_METHOD = 'peak'
    # 自定义神经元排序顺序（仅在SORT_METHOD='custom'时使用）
    CUSTOM_NEURON_ORDER = ['n53', 'n40', 'n29', 'n34', 'n4', 'n32', 'n25', 'n27', 'n22', 'n55', 'n21', 'n5', 'n19']
    # Trace图的显示参数
    TRACE_OFFSET = 60  # 不同神经元trace之间的垂直偏移量（增加间隔以改善可读性）
    SCALING_FACTOR = 80  # 信号振幅缩放因子（增加振幅以提高信号可见性）
    MAX_NEURONS = 60    # 最大显示神经元数量（避免图表过于拥挤）
    TRACE_ALPHA = 0.8   # trace线的透明度
    LINE_WIDTH = 2.0    # trace线的宽度
    # 采样率 (Hz)
    SAMPLING_RATE = 9.02  # 采样频率，用于将时间戳转换为秒
    # 横坐标单位：'stamp'（默认，使用原始stamp）或 'seconds'（stamp/采样率）
    X_AXIS_UNIT = 'stamp'
    # 钙爆发检测参数
    CALCIUM_THRESHOLD = 2.0  # 标准差的倍数，超过此阈值视为钙爆发
    # 钙波检测参数（用于calcium_wave排序）
    CALCIUM_WAVE_THRESHOLD = 1.5  # 钙波阈值（标准差的倍数）
    MIN_PROMINENCE = 1.0  # 最小峰值突出度
    MIN_RISE_RATE = 0.1  # 最小上升速率
    MAX_FALL_RATE = 0.05  # 最大下降速率（下降应当比上升慢）

# 解析命令行参数
def parse_args():
    parser = argparse.ArgumentParser(description='神经元活动 Trace 图生成工具，支持自定义时间区间和排序方式')
    parser.add_argument('--input', type=str, help='输入数据文件路径')
    parser.add_argument('--output-dir', type=str, help='输出目录')
    parser.add_argument('--stamp-min', type=float, help='最小时间戳值')
    parser.add_argument('--stamp-max', type=float, help='最大时间戳值')
    parser.add_argument('--x-unit', type=str, choices=['stamp', 'seconds'], default=Config.X_AXIS_UNIT,
                        help="横坐标单位：stamp（原始时间戳）或 seconds（stamp/采样率）")
    parser.add_argument('--max-neurons', type=int, help='最大显示神经元数量')
    parser.add_argument('--scaling', type=float, help='信号振幅缩放因子')
    parser.add_argument('--sort-method', type=str, choices=['original', 'peak', 'calcium_wave', 'custom'], 
                        help='排序方式：original（原始顺序）、peak（按峰值时间排序）、calcium_wave（按第一次真实钙波时间排序）或custom（按自定义顺序排序）')
    parser.add_argument('--ca-threshold', type=float, help='钙波检测阈值（标准差的倍数）')
    parser.add_argument('--min-prominence', type=float, help='最小峰值突出度')
    parser.add_argument('--line-width', type=float, help='trace线条宽度')
    return parser.parse_args()

# 解析命令行参数并更新配置
args = parse_args()
if args.input:
    Config.INPUT_FILE = args.input
if args.output_dir:
    Config.OUTPUT_DIR = args.output_dir
if args.stamp_min is not None:
    Config.STAMP_MIN = args.stamp_min
if args.stamp_max is not None:
    Config.STAMP_MAX = args.stamp_max
if args.x_unit:
    Config.X_AXIS_UNIT = args.x_unit
if args.max_neurons is not None:
    Config.MAX_NEURONS = args.max_neurons
if args.scaling is not None:
    Config.SCALING_FACTOR = args.scaling
if args.sort_method:
    Config.SORT_METHOD = args.sort_method
if args.ca_threshold is not None:
    Config.CALCIUM_WAVE_THRESHOLD = args.ca_threshold
if args.min_prominence is not None:
    Config.MIN_PROMINENCE = args.min_prominence
if args.line_width is not None:
    Config.LINE_WIDTH = args.line_width

# # 加载数据
# print(f"正在从 {Config.INPUT_FILE} 加载数据...")
# trace_data = pd.read_excel(Config.INPUT_FILE)

# # 将 'stamp' 列设置为索引
# stamp_column = trace_data['stamp'].copy()  # 保存原始时间戳
# # 创建秒为单位的时间索引
# seconds_index = stamp_column / Config.SAMPLING_RATE
# trace_data = trace_data.set_index('stamp')

# # 根据配置的时间戳区间筛选数据
# if Config.STAMP_MIN is not None or Config.STAMP_MAX is not None:
#     # 确定实际的最小值和最大值
#     min_stamp = Config.STAMP_MIN if Config.STAMP_MIN is not None else trace_data.index.min()
#     max_stamp = Config.STAMP_MAX if Config.STAMP_MAX is not None else trace_data.index.max()
    
#     # 筛选数据，保留指定区间内的数据
#     trace_data = trace_data.loc[min_stamp:max_stamp]
#     print(f"已筛选时间戳区间: {min_stamp} 到 {max_stamp}")

# # 检查是否存在 'behavior' 列
# has_behavior = 'behavior' in trace_data.columns

# # 分离 'behavior' 列（如果存在）
# if has_behavior:
#     behavior_data = trace_data['behavior']
#     trace_data = trace_data.drop(columns=['behavior'])
#     print("检测到行为数据，将在图表中显示行为区间")

# # 数据处理（简化处理流程，保留原始信号特性）
# # 不进行Z-score标准化，仿照init_show.py的做法直接使用原始信号并缩放
# print("已准备处理数据")

# # ========= 添加排序功能 =========
# # 函数：按自定义神经元顺序排序
# def sort_neurons_by_custom_order(data_columns, custom_order):
#     """
#     按自定义神经元顺序排序
    
#     指定神经元按给定顺序排在前面，剩余神经元按字符串排序排在后面
    
#     参数:
#     data_columns -- 数据中的神经元列名
#     custom_order -- 自定义的神经元顺序列表
    
#     返回:
#     sorted_neurons -- 按自定义顺序排列的神经元列表
#     """
#     available_neurons = set(data_columns)
    
#     # 首先按照自定义顺序排列存在的神经元
#     ordered_neurons = []
#     for neuron in custom_order:
#         if neuron in available_neurons:
#             ordered_neurons.append(neuron)
    
#     # 找出剩余的神经元，按字符串大小顺序排列
#     remaining_neurons = sorted(list(available_neurons - set(ordered_neurons)))
    
#     # 合并两部分：自定义顺序 + 剩余神经元（按大小排序）
#     final_order = ordered_neurons + remaining_neurons
    
#     return final_order

# # 函数：检测神经元第一次真实钙波发生的时间点
# def detect_first_calcium_wave(neuron_data):
#     """
#     检测神经元第一次真实钙波发生的时间点
    
#     参数:
#     neuron_data -- 包含神经元活动的时间序列数据（标准化后）
    
#     返回:
#     first_wave_time -- 第一次真实钙波发生的时间点，如果没有检测到则返回数据最后一个时间点
#     """
#     from scipy.signal import find_peaks
    
#     # 计算阈值（基于数据的标准差）
#     threshold = Config.CALCIUM_WAVE_THRESHOLD
    
#     # 使用find_peaks函数检测峰值
#     peaks, properties = find_peaks(neuron_data, 
#                                  height=threshold, 
#                                  prominence=Config.MIN_PROMINENCE,
#                                  distance=5)  # 要求峰值之间至少间隔5个时间点
    
#     if len(peaks) == 0:
#         # 如果没有检测到峰值，返回时间序列的最后一个点
#         return neuron_data.index[-1]
    
#     # 对每个峰值进行验证，确认是否为真实钙波（上升快，下降慢）
#     for peak_idx in peaks:
#         # 确保峰值不在时间序列的开始或结束处
#         if peak_idx <= 1 or peak_idx >= len(neuron_data) - 2:
#             continue
            
#         # 计算峰值前的上升速率（取峰值前5个点或更少）
#         pre_peak_idx = max(0, peak_idx - 5)
#         rise_rate = (neuron_data.iloc[peak_idx] - neuron_data.iloc[pre_peak_idx]) / (peak_idx - pre_peak_idx)
        
#         # 计算峰值后的下降速率（取峰值后10个点或更少）
#         post_peak_idx = min(len(neuron_data) - 1, peak_idx + 10)
#         if post_peak_idx <= peak_idx:
#             continue
        
#         fall_rate = (neuron_data.iloc[peak_idx] - neuron_data.iloc[post_peak_idx]) / (post_peak_idx - peak_idx)
        
#         # 确认是否符合钙波特征：上升快，下降慢
#         if rise_rate > Config.MIN_RISE_RATE and 0 < fall_rate < Config.MAX_FALL_RATE:
#             # 找到第一个真实钙波，返回时间点
#             return neuron_data.index[peak_idx]
    
#     # 如果没有满足条件的钙波，返回时间序列的最后一个点
#     return neuron_data.index[-1]

# print(f"使用排序方式: {Config.SORT_METHOD}")
# print(f"使用线条宽度: {Config.LINE_WIDTH}")

# # 根据排序方式选择相应的排序算法
# if Config.SORT_METHOD == 'original':
#     # 原始顺序：保持数据原有的神经元顺序
#     sorted_neurons = trace_data.columns
#     sort_method_str = "Original Order"
#     peak_times_dict = None  # 原始顺序不需要峰值时间
# elif Config.SORT_METHOD == 'peak':
#     # 按峰值时间排序
#     trace_data_standardized = (trace_data - trace_data.mean()) / trace_data.std()
#     peak_times = trace_data_standardized.idxmax()
#     sorted_neurons = peak_times.sort_values().index
#     sort_method_str = "Sorted by Peak Time"
#     peak_times_dict = peak_times.to_dict()
# elif Config.SORT_METHOD == 'custom':
#     # 自定义顺序排序
#     sorted_neurons = sort_neurons_by_custom_order(trace_data.columns, Config.CUSTOM_NEURON_ORDER)
#     sort_method_str = "Sorted by Custom Order"
#     peak_times_dict = None
# else:  # 'calcium_wave'
#     # 按第一次真实钙波发生时间排序
#     trace_data_standardized = (trace_data - trace_data.mean()) / trace_data.std()
#     first_wave_times = {}
    
#     # 对每个神经元进行钙波检测
#     for neuron in trace_data_standardized.columns:
#         neuron_data = trace_data_standardized[neuron]
#         first_wave_times[neuron] = detect_first_calcium_wave(neuron_data)
    
#     # 转换为Series以便排序
#     first_wave_times_series = pd.Series(first_wave_times)
    
#     # 按第一次钙波时间排序
#     sorted_neurons = first_wave_times_series.sort_values().index
#     sort_method_str = "Sorted by First Calcium Wave Time"
#     peak_times_dict = first_wave_times

# # 根据排序重新排列数据
# if Config.SORT_METHOD != 'original':
#     sorted_trace_data = trace_data[sorted_neurons]
# else:
#     sorted_trace_data = trace_data
# # ========= 排序功能添加完成 =========

# ================= 替换开始 =================
# 加载全量数据 (为了实现全局排序，必须先加载所有数据)
print(f"正在从 {Config.INPUT_FILE} 加载数据...")
full_data = pd.read_excel(Config.INPUT_FILE)

# 将 'stamp' 列设置为索引
stamp_column = full_data['stamp'].copy()
seconds_index = stamp_column / Config.SAMPLING_RATE
full_data = full_data.set_index('stamp')

# 检查并分离 'behavior' 列（基于全量数据）
has_behavior = 'behavior' in full_data.columns
if has_behavior:
    behavior_data_full = full_data['behavior']
    neural_data_full = full_data.drop(columns=['behavior'])
    print("检测到行为数据，将在图表中显示行为区间")
else:
    neural_data_full = full_data
    behavior_data_full = None

# 数据预处理：标准化全量数据（用于计算排序）
print("正在基于全局数据计算排序...")
neural_data_full_std = (neural_data_full - neural_data_full.mean()) / neural_data_full.std()

# --- 定义辅助函数 ---
def sort_neurons_by_custom_order(data_columns, custom_order):
    available_neurons = set(data_columns)
    ordered_neurons = []
    for neuron in custom_order:
        if neuron in available_neurons:
            ordered_neurons.append(neuron)
    remaining_neurons = sorted(list(available_neurons - set(ordered_neurons)))
    return ordered_neurons + remaining_neurons

def detect_first_calcium_wave(neuron_data):
    from scipy.signal import find_peaks
    threshold = Config.CALCIUM_WAVE_THRESHOLD
    peaks, _ = find_peaks(neuron_data, height=threshold, prominence=Config.MIN_PROMINENCE, distance=5)
    if len(peaks) == 0: return neuron_data.index[-1]
    
    for peak_idx in peaks:
        if peak_idx <= 1 or peak_idx >= len(neuron_data) - 2: continue
        pre_peak_idx = max(0, peak_idx - 5)
        rise_rate = (neuron_data.iloc[peak_idx] - neuron_data.iloc[pre_peak_idx]) / (peak_idx - pre_peak_idx)
        post_peak_idx = min(len(neuron_data) - 1, peak_idx + 10)
        if post_peak_idx <= peak_idx: continue
        fall_rate = (neuron_data.iloc[peak_idx] - neuron_data.iloc[post_peak_idx]) / (post_peak_idx - peak_idx)
        if rise_rate > Config.MIN_RISE_RATE and 0 < fall_rate < Config.MAX_FALL_RATE:
            return neuron_data.index[peak_idx]
    return neuron_data.index[-1]
# -------------------

# 核心步骤：基于全量数据确定神经元顺序 (Sorted Neurons)
peak_times_dict = None

if Config.SORT_METHOD == 'original':
    sorted_neurons = neural_data_full.columns
    sort_method_str = "Original Order"
elif Config.SORT_METHOD == 'peak':
    # 全局峰值排序
    peak_times = neural_data_full_std.idxmax()
    sorted_neurons = peak_times.sort_values().index
    sort_method_str = "Sorted by Peak Time (Global)"
    peak_times_dict = peak_times.to_dict()
elif Config.SORT_METHOD == 'custom':
    sorted_neurons = sort_neurons_by_custom_order(neural_data_full.columns, Config.CUSTOM_NEURON_ORDER)
    sort_method_str = "Sorted by Custom Order"
else:  # calcium_wave
    # 全局钙波排序
    first_wave_times = {}
    for neuron in neural_data_full_std.columns:
        first_wave_times[neuron] = detect_first_calcium_wave(neural_data_full_std[neuron])
    first_wave_times_series = pd.Series(first_wave_times)
    sorted_neurons = first_wave_times_series.sort_values().index
    sort_method_str = "Sorted by First Calcium Wave (Global)"
    peak_times_dict = first_wave_times

print(f"神经元排序方式: {sort_method_str}")

# 最后步骤：根据时间区间筛选用于绘图的数据 (Slicing)
if Config.STAMP_MIN is not None or Config.STAMP_MAX is not None:
    min_stamp = Config.STAMP_MIN if Config.STAMP_MIN is not None else neural_data_full.index.min()
    max_stamp = Config.STAMP_MAX if Config.STAMP_MAX is not None else neural_data_full.index.max()
    
    # 筛选神经元数据并应用排序
    sorted_trace_data = neural_data_full.loc[min_stamp:max_stamp]
    sorted_trace_data = sorted_trace_data[sorted_neurons]
    
    # 筛选行为数据
    if has_behavior:
        behavior_data = behavior_data_full.loc[min_stamp:max_stamp]
    
    print(f"已筛选时间戳区间: {min_stamp} 到 {max_stamp}")
else:
    sorted_trace_data = neural_data_full[sorted_neurons]
    if has_behavior:
        behavior_data = behavior_data_full
# ================= 替换结束 =================

# ===== 开始绘制Trace图 =====
print(f"开始绘制Trace图，排序方式: {sort_method_str}...")
if has_behavior:
    # 如果有行为数据，使用2行2列的布局，与热图保持一致
    fig = plt.figure(figsize=(80, 30))
    # 使用GridSpec，与heatmap_sort-EM.py保持一致的布局
    grid = GridSpec(2, 2, height_ratios=[0.5, 6], width_ratios=[6, 0.5], hspace=0.05, wspace=0.02, figure=fig)
    ax_behavior = fig.add_subplot(grid[0, 0])
    ax_trace = fig.add_subplot(grid[1, 0])
    ax_legend = fig.add_subplot(grid[1, 1])
else:
    # 没有行为数据，只创建一个图表
    fig = plt.figure(figsize=(80, 30))
    ax_trace = fig.add_subplot(111)

# 预定义颜色映射，与热图保持一致
fixed_color_map = {
    # === 原有配色（保持不变）===
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
    'Water': '#CC99FF',                 # 亮紫色
    
    # === 通用行为库 ===
    'Rest': '#8B4513',                  # 深褐色
    'Sleep': '#2F4F4F',                 # 深灰绿色
    'Social': '#FF1493',                # 深粉色
    'Climbing': '#32CD32',              # 酸橙绿
    'Digging': '#8B008B',               # 深洋红色
    'Running': '#FF4500',               # 橙红色
    'Swimming': '#1E90FF',              # 道奇蓝
    'Freezing': '#708090',              # 石板灰
    'Hiding': '#556B2F',                # 暗橄榄绿
    'Aggressive': '#DC143C',            # 深红色
    'Defensive': '#9932CC',             # 深兰花紫
    'Play': '#FFD700',                  # 金色
    'Sniffing': '#20B2AA',              # 浅海绿色
    'Licking': '#FF69B4',               # 热粉色
    'Scratching': '#CD853F',            # 秘鲁色 (对应图片 Image 3)
    'Stretching': '#4169E1',            # 皇家蓝
    'Turning': '#DA70D6',               # 兰花紫
    'Jumping': '#FF6347',               # 番茄色
    'Rearing': '#40E0D0',               # 青绿色
    'Grooming-self': '#9370DB',         # 中紫色
    'Grooming-other': '#3CB371',        # 中海绿色
    'Feeding-young': '#F0E68C',         # 卡其色
    'Nesting': '#DDA0DD',               # 李子色
    'Mating': '#FA8072',                # 鲑鱼色
    'Territory-marking': '#87CEEB',     # 天蓝色
    'Escape': '#B22222',                # 火砖色
    'Approach': '#228B22',              # 森林绿
    'Avoid': '#4B0082',                 # 靛蓝色
    'Investigate': '#FF8C00',           # 深橙色
    'Vocalization': '#6A5ACD',          # 石蓝色
    
    # === 高架十字迷宫行为标签 ===
    'Close-arm': '#8B0000',             # 深红色
    'Close-armed-Exp': '#CD5C5C',       # 印度红
    'Closed-arm-freezing': '#2F2F2F',   # 深灰色
    'Middle-zone': '#FFD700',           # 金色
    'Middle-zone-freezing': '#B8860B',  # 深金黄色
    'Open-arm': '#32CD32',              # 酸橙绿
    'Open-arm-exp': '#00FF7F',          # 春绿色
    'open-arm-freezing': '#006400',     # 深绿色
    'Open-arm-head dipping': '#7CFC00', # 草绿色

    # === [UPDATED] 图片中的特定行为标签 (严格匹配字符串) ===
    # 饮水/饮食类
    'Drinking': '#1E90FF',              # 道奇蓝 - 通用
    'Drinking Water': '#4169E1',        # 皇家蓝 - (Image 2)
    'Drinking Quinine': '#5F9EA0',      # 军校蓝 - (Image 2) 冷色调表示厌恶/苦味
    'Drinking Sucrose': '#FF1493',      # 深粉色 - (Image 2) 暖色调表示糖水/奖励
    'Eating Food': '#228B22',           # 森林绿 - (Image 2)
    'Eating Seeds': '#32CD32',          # 酸橙绿 - (Image 2) 与 Grab-seeds 同色系
    'Feeding': '#006400',               # 深绿色 - (Image 2)
    
    # 垫料/探索类
    'Digging Bedding': '#800080',       # 紫色 - (Image 2)
    'Digging Bedding/Finding Food': '#8B4513', # 鞍褐色 - (Image 2) 混合了挖掘和寻找
    'Cotton Interaction': '#FFB6C1',    # 浅粉色 - (Image 2) 玩棉花/筑巢，柔和色
    'Dropped Seed': '#778899',          # 亮石板灰 - (Image 2) 意外事件，使用中性色
    
    # 清洁/休息类
    'Grooming': '#00CED1',              # 暗青色 - (Image 2)
    'Grooming/Face Wash': '#48D1CC',    # 中绿松石 - (Image 2) 与 Grooming 相近
    'Resting': '#A0522D',               # 赭色 - (Image 2) 与 Rest 相近
    
    # 站立/运动类
    'Stand': '#DAA520',                 # 金麒麟色 - (Image 3)
    'Stand/Climb': '#B8860B',           # 暗金黄色 - (Image 3) 垂直运动
    
    # 刺激/惊吓类 (Aversive Events)
    'Sprayed by Water': '#FF4500',      # 橙红色 - (Image 3) 突发刺激，警示色
    'Startled by Water': '#800000',     #栗色 - (Image 3) 惊吓反应，深警示色
    'Quinine_Spray': '#FF6347',         # 番茄色 - 补充：奎宁喷射
    'Sniffing_Quinine': '#9ACD32',      # 黄绿色 - 补充：嗅探刺激物
    'Tremble': '#8A2BE2'                # 蓝紫色 - 战栗
}

def _stamp_to_x(stamp_value: float):
    return stamp_value / Config.SAMPLING_RATE if Config.X_AXIS_UNIT == 'seconds' else stamp_value

x_values = sorted_trace_data.index.to_numpy(dtype=float)
if Config.X_AXIS_UNIT == 'seconds':
    x_values = x_values / Config.SAMPLING_RATE

# 绘制Trace图
for i, column in enumerate(sorted_neurons):
    if i >= Config.MAX_NEURONS:
        break
    
    # 计算当前神经元trace的垂直偏移量，并应用缩放因子
    # 根据排序方式调整位置
    if Config.SORT_METHOD in ['peak', 'calcium_wave']:
        # 对于峰值排序和钙波排序，使用反向位置（早期的在上方）
        position = Config.MAX_NEURONS - i if i < Config.MAX_NEURONS else 1
    else:
        # 对于原始顺序和自定义顺序，使用正向位置
        position = i + 1
    
    ax_trace.plot(
        x_values,
        sorted_trace_data[column] * Config.SCALING_FACTOR + position * Config.TRACE_OFFSET,  # 应用缩放并偏移
        linewidth=Config.LINE_WIDTH,
        alpha=Config.TRACE_ALPHA,
        label=column
    )
    
    # # 如果是峰值排序或钙波排序，标记峰值点
    # if peak_times_dict is not None and column in peak_times_dict:
    #     peak_time = peak_times_dict[column] / Config.SAMPLING_RATE
    #     ax_trace.scatter(
    #         peak_time, 
    #         sorted_trace_data.loc[peak_times_dict[column], column] * Config.SCALING_FACTOR + position * Config.TRACE_OFFSET,
    #         color='red', s=30, zorder=3  # zorder确保点在线的上方
    #     )

    # 如果是峰值排序或钙波排序，标记峰值点
    if peak_times_dict is not None and column in peak_times_dict:
        peak_ts = peak_times_dict[column]
        # 【关键修改】只有当峰值时间在当前显示的区间内时，才绘制红点
        if peak_ts in sorted_trace_data.index:
            peak_time_x = _stamp_to_x(peak_ts)
            ax_trace.scatter(
                peak_time_x, 
                sorted_trace_data.loc[peak_ts, column] * Config.SCALING_FACTOR + position * Config.TRACE_OFFSET,
                color='red', s=30, zorder=3
            )

# 设置Y轴标签，简化显示格式
total_positions = min(Config.MAX_NEURONS, len(sorted_neurons))
yticks = []
ytick_labels = []

# 为每个trace计算正确的位置和标签
for i, column in enumerate(sorted_neurons):
    if i >= Config.MAX_NEURONS:
        break
    
    # 计算位置 - 这需要与上面绘制时的position计算完全一致
    if Config.SORT_METHOD in ['peak', 'calcium_wave']:
        position = Config.MAX_NEURONS - i if i < Config.MAX_NEURONS else 1
    else:
        position = i + 1
    
    # 将位置和对应的标签添加到列表
    yticks.append(position * Config.TRACE_OFFSET)
    ytick_labels.append(str(column))

# 设置Y轴刻度和标签
ax_trace.set_yticks(yticks)
ax_trace.set_yticklabels(ytick_labels)

# 设置X轴范围
if Config.STAMP_MIN is not None and Config.STAMP_MAX is not None:
    min_x = _stamp_to_x(Config.STAMP_MIN)
    max_x = _stamp_to_x(Config.STAMP_MAX)
    min_x = max(0, min_x)
    ax_trace.set_xlim(min_x, max_x)
else:
    # 如果没有指定时间范围，从0开始显示
    min_x = max(0, _stamp_to_x(float(sorted_trace_data.index.min())))
    max_x = _stamp_to_x(float(sorted_trace_data.index.max()))
    ax_trace.set_xlim(min_x, max_x)

# 设置轴标签和标题
ax_trace.set_xlabel('stamp' if Config.X_AXIS_UNIT == 'stamp' else 'Time (seconds)', fontsize=50, fontweight='bold')
# 根据排序方式设置不同的Y轴标签
if Config.SORT_METHOD == 'peak':
    ylabel = 'Neuron ID (Sorted by Peak Time)'
elif Config.SORT_METHOD == 'calcium_wave':
    ylabel = 'Neuron ID (Sorted by First Calcium Wave)'
elif Config.SORT_METHOD == 'custom':
    ylabel = 'Neuron ID (Custom Order)'
else:
    ylabel = 'Neuron ID'
ax_trace.set_ylabel(ylabel, fontsize=50, fontweight='bold')

# 设置刻度标签字体大小
ax_trace.tick_params(axis='x', labelsize=30, rotation=45)  # X轴刻度旋转45度
ax_trace.tick_params(axis='y', labelsize=23)  # Y轴刻度字体大小改为23

# 设置X轴刻度间隔为10秒
import matplotlib.ticker as ticker
tick_step = 10 if Config.X_AXIS_UNIT == 'seconds' else max(1, int(round(10 * Config.SAMPLING_RATE)))
ax_trace.xaxis.set_major_locator(ticker.MultipleLocator(tick_step))

# 添加网格线，使trace更容易阅读
ax_trace.grid(False)

# 处理行为区间数据
behavior_intervals = {}
unique_behaviors = []
behavior_change_times = []

# 只有当behavior列存在时才处理行为标签
if has_behavior:
    behavior_data = behavior_data.where(behavior_data.astype(str).str.strip().ne(''), np.nan)

    # 获取所有不同的行为标签
    unique_behaviors = behavior_data.dropna().unique()
    
    # 初始化所有行为的区间字典
    for behavior in unique_behaviors:
        behavior_intervals[behavior] = []
    
    # 对behavior_data进行处理，找出每种行为的连续区间
    current_behavior = None
    start_time = None
    
    # 为了确保最后一个区间也被记录，将索引列表扩展一个元素
    extended_index = list(behavior_data.index) + [None]
    extended_values = list(behavior_data.values) + [None]
    
    for i, (timestamp, behavior) in enumerate(zip(extended_index, extended_values)):
        # 最后一个元素特殊处理
        if i == len(behavior_data):
            if start_time is not None and current_behavior is not None:
                behavior_intervals[current_behavior].append((_stamp_to_x(start_time), _stamp_to_x(extended_index[i-1])))
            break
        
        # 跳过空值
        if pd.isna(behavior):
            # 如果之前有行为，则结束当前区间
            if start_time is not None and current_behavior is not None:
                behavior_intervals[current_behavior].append((_stamp_to_x(start_time), _stamp_to_x(timestamp)))
                start_time = None
                current_behavior = None
            continue
        
        # 如果是新的行为类型或第一个行为
        if behavior != current_behavior:
            # 如果之前有行为，先结束当前区间
            if start_time is not None and current_behavior is not None:
                behavior_intervals[current_behavior].append((_stamp_to_x(start_time), _stamp_to_x(timestamp)))
            
            # 开始新的行为区间
            start_time = timestamp
            current_behavior = behavior

    aligned_behavior = behavior_data.reindex(sorted_trace_data.index)
    filled_behavior = aligned_behavior.fillna('__MISSING__')
    change_mask = filled_behavior.ne(filled_behavior.shift(1))
    change_positions = [int(i) for i in np.where(change_mask.values)[0] if i != 0]
    behavior_change_times = [_stamp_to_x(float(sorted_trace_data.index[i])) for i in change_positions]

if has_behavior:
    ax_behavior.set_ylim(0, 1)
    ax_behavior.set_yticks([])
    ax_behavior.set_yticklabels([])
    ax_behavior.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
    ax_behavior.set_title('Behavior Timeline', fontsize=40, pad=10)
    ax_behavior.set_xlabel('')
    ax_behavior.set_xlim(ax_trace.get_xlim())
    ax_behavior.set_anchor('SW')
    ax_behavior.spines['top'].set_visible(False)
    ax_behavior.spines['right'].set_visible(False)
    ax_behavior.spines['bottom'].set_visible(False)
    ax_behavior.spines['left'].set_visible(False)

    for x in behavior_change_times:
        ax_trace.axvline(x=x, color='black', linestyle='--', linewidth=2, alpha=0.8)
        ax_behavior.axvline(x=x, color='black', linestyle='--', linewidth=2, alpha=0.8)

    ax_legend.axis('off')

    if len(unique_behaviors) > 0:
        legend_patches = []
        y_position = 0.5
        line_height = 0.8

        for behavior, intervals in behavior_intervals.items():
            behavior_color = fixed_color_map.get(behavior, plt.cm.tab10(list(unique_behaviors).index(behavior) % 10))

            for start_time, end_time in intervals:
                if end_time - start_time > 0:
                    rect = plt.Rectangle(
                        (start_time, y_position - line_height/2),
                        end_time - start_time,
                        line_height,
                        color=behavior_color,
                        alpha=0.9,
                        ec='black',
                        linewidth=0.5,
                    )
                    ax_behavior.add_patch(rect)

            legend_patches.append(plt.Rectangle((0, 0), 1, 1, color=behavior_color, alpha=0.9, label=behavior))

        legend_fontsize = 40
        title_fontsize = 40

        legend = ax_legend.legend(
            handles=legend_patches,
            loc='center left',
            fontsize=legend_fontsize,
            title='Behavior Types',
            title_fontsize=title_fontsize,
            ncol=1,
            frameon=True,
            fancybox=True,
            shadow=True,
            bbox_to_anchor=(0, 0.5),
        )

# # 生成标题，包含排序方式和时间区间信息
# title_text = f'Traces with Increased Amplitude ({sort_method_str})'
# if Config.STAMP_MIN is not None or Config.STAMP_MAX is not None:
#     min_stamp = Config.STAMP_MIN if Config.STAMP_MIN is not None else sorted_trace_data.index.min()
#     max_stamp = Config.STAMP_MAX if Config.STAMP_MAX is not None else sorted_trace_data.index.max()
#     min_seconds = min_stamp / Config.SAMPLING_RATE
#     max_seconds = max_stamp / Config.SAMPLING_RATE
#     title_text += f' (Time Range: {min_seconds:.2f}s - {max_seconds:.2f}s)'

# # 添加标题
# plt.suptitle(title_text, fontsize=40, y=0.98)

# 调整布局（与heatmap_sort-EM.py保持一致）
# 不使用tight_layout()，因为它与GridSpec布局不兼容
# 使用精确的位置对齐方法
if has_behavior:
    # 强制更新布局
    fig.canvas.draw()
    
    # 获取热图的实际边界位置
    trace_bbox = ax_trace.get_position()
    behavior_bbox = ax_behavior.get_position()
    
    # 使用Bbox的坐标创建新的位置
    from matplotlib.transforms import Bbox
    new_behavior_pos = Bbox([[trace_bbox.x0, behavior_bbox.y0], 
                            [trace_bbox.x0 + trace_bbox.width, behavior_bbox.y0 + behavior_bbox.height]])
    
    # 设置新的位置
    ax_behavior.set_position(new_behavior_pos)

# 在布局调整完成后设置刻度标签加粗
# 强制更新图形以确保所有刻度标签都已生成
fig.canvas.draw()
# 设置X轴刻度标签加粗
for label in ax_trace.get_xticklabels():
    label.set_fontweight('bold')
# 设置Y轴刻度标签加粗  
for label in ax_trace.get_yticklabels():
    label.set_fontweight('bold')
else:
    # 没有行为数据的情况，也需要设置刻度标签加粗
    # 强制更新图形以确保所有刻度标签都已生成
    fig.canvas.draw()
    # 设置X轴刻度标签加粗
    for label in ax_trace.get_xticklabels():
        label.set_fontweight('bold')
    # 设置Y轴刻度标签加粗  
    for label in ax_trace.get_yticklabels():
        label.set_fontweight('bold')

# 从输入文件路径中提取文件名（不包括路径和扩展名）
input_filename = os.path.basename(Config.INPUT_FILE)
input_filename = os.path.splitext(input_filename)[0]  # 去除扩展名

# 构建输出文件名：目录 + 前缀 + 排序方式 + 输入文件名 + 时间戳信息
stamp_info = ''
if Config.STAMP_MIN is not None or Config.STAMP_MAX is not None:
    min_stamp = Config.STAMP_MIN if Config.STAMP_MIN is not None else sorted_trace_data.index.min()
    max_stamp = Config.STAMP_MAX if Config.STAMP_MAX is not None else sorted_trace_data.index.max()
    min_seconds = min_stamp / Config.SAMPLING_RATE
    max_seconds = max_stamp / Config.SAMPLING_RATE
    stamp_info = f'_{min_seconds:.2f}s_{max_seconds:.2f}s'

# output_file = f"{Config.OUTPUT_DIR}traces_{Config.SORT_METHOD}_{input_filename}{stamp_info}.png"
# print(f"正在保存图像到 {output_file}")

# # 保存图像
# plt.savefig(output_file, dpi=150)
# print(f"图像已保存")

# # 显示图像（可选，可以根据需要取消注释）
# # plt.show()
# print("程序执行完成")

output_file = f"{Config.OUTPUT_DIR}traces_{Config.SORT_METHOD}_{input_filename}{stamp_info}.png"

# === 【修改点 1】自动创建目录，防止报错 ===
output_dir = os.path.dirname(output_file)
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
# =======================================

print(f"正在保存图像到 {output_file}")

# === 【修改点 2】关键修改：添加 bbox_inches='tight' 去除白边 ===
# pad_inches=0.1 留一点点缝隙防止文字被切，dpi=150 保证清晰度且不超限
plt.savefig(output_file, dpi=150, bbox_inches='tight', pad_inches=0.1)
# ==========================================================

print(f"图像已保存")

# 显示图像（可选，可以根据需要取消注释）
# plt.show()
print("程序执行完成")
