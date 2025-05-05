# 神经元活动 Trace 图绘制，基于热图代码修改而来
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import argparse
from matplotlib.gridspec import GridSpec
import matplotlib.patches as mpatches
import os

# 简化后的参数配置类
class Config:
    # 输入文件路径
    INPUT_FILE = '../../datasets/EM2Trace_plus.xlsx'
    # 输出目录
    OUTPUT_DIR = '../../graph/'
    # 时间戳区间默认值（None表示不限制）
    STAMP_MIN = None  # 最小时间戳
    STAMP_MAX = None  # 最大时间戳
    # Trace图的显示参数
    TRACE_OFFSET = 35  # 不同神经元trace之间的垂直偏移量（仿照init_show.py中的值）
    SCALING_FACTOR = 40  # 信号振幅缩放因子（仿照init_show.py）
    MAX_NEURONS = 60    # 最大显示神经元数量（避免图表过于拥挤）
    TRACE_ALPHA = 0.8   # trace线的透明度
    LINE_WIDTH = 1.0    # trace线的宽度
    # 采样率 (Hz)
    SAMPLING_RATE = 4.8  # 采样频率，用于将时间戳转换为秒
    # 钙爆发检测参数
    CALCIUM_THRESHOLD = 2.0  # 标准差的倍数，超过此阈值视为钙爆发

# 解析命令行参数
def parse_args():
    parser = argparse.ArgumentParser(description='神经元活动 Trace 图生成工具，支持自定义时间区间')
    parser.add_argument('--input', type=str, help='输入数据文件路径')
    parser.add_argument('--output-dir', type=str, help='输出目录')
    parser.add_argument('--stamp-min', type=float, help='最小时间戳值')
    parser.add_argument('--stamp-max', type=float, help='最大时间戳值')
    parser.add_argument('--max-neurons', type=int, help='最大显示神经元数量')
    parser.add_argument('--scaling', type=float, help='信号振幅缩放因子')
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
if args.max_neurons is not None:
    Config.MAX_NEURONS = args.max_neurons
if args.scaling is not None:
    Config.SCALING_FACTOR = args.scaling

# 加载数据
print(f"正在从 {Config.INPUT_FILE} 加载数据...")
trace_data = pd.read_excel(Config.INPUT_FILE)

# 将 'stamp' 列设置为索引
stamp_column = trace_data['stamp'].copy()  # 保存原始时间戳
# 创建秒为单位的时间索引
seconds_index = stamp_column / Config.SAMPLING_RATE
trace_data = trace_data.set_index('stamp')

# 根据配置的时间戳区间筛选数据
if Config.STAMP_MIN is not None or Config.STAMP_MAX is not None:
    # 确定实际的最小值和最大值
    min_stamp = Config.STAMP_MIN if Config.STAMP_MIN is not None else trace_data.index.min()
    max_stamp = Config.STAMP_MAX if Config.STAMP_MAX is not None else trace_data.index.max()
    
    # 筛选数据，保留指定区间内的数据
    trace_data = trace_data.loc[min_stamp:max_stamp]
    print(f"已筛选时间戳区间: {min_stamp} 到 {max_stamp}")

# 检查是否存在 'behavior' 列
has_behavior = 'behavior' in trace_data.columns

# 分离 'behavior' 列（如果存在）
if has_behavior:
    behavior_data = trace_data['behavior']
    trace_data = trace_data.drop(columns=['behavior'])
    print("检测到行为数据，将在图表中显示行为区间")

# 数据处理（简化处理流程，保留原始信号特性）
# 不进行Z-score标准化，仿照init_show.py的做法直接使用原始信号并缩放
print("已准备处理数据")

# ========= 开始添加钙爆发排序功能 =========
# 创建第二个图，按神经元钙爆发时间排序
print("开始计算神经元钙爆发时间排序...")

# 对数据进行标准化，用于钙事件检测
trace_data_standardized = (trace_data - trace_data.mean()) / trace_data.std()

# 定义函数：找到神经元第一次超过阈值的时间点
def find_first_peak_time(neuron_data, threshold=Config.CALCIUM_THRESHOLD):
    """找到神经元信号第一次超过阈值的时间点"""
    # 找到所有超过阈值的时间点
    above_threshold = neuron_data[neuron_data > threshold]
    if len(above_threshold) > 0:
        # 返回第一次超过阈值的时间点
        return above_threshold.index[0]
    else:
        # 如果没有超过阈值的点，返回最大值的时间点
        return neuron_data.idxmax()

# 找到每个神经元第一次钙爆发的时间点
first_peak_times = {}
for column in trace_data_standardized.columns:
    first_peak_times[column] = find_first_peak_time(trace_data_standardized[column])

# 转换为Series并排序
first_peak_times_series = pd.Series(first_peak_times)
sorted_neurons_by_first_peak = first_peak_times_series.sort_values().index

# 根据排序创建排序后的数据框
sorted_trace_data = trace_data[sorted_neurons_by_first_peak]
# ========= 钙爆发排序功能添加完成 =========

# ===== 开始绘制两个图：原始顺序和按钙爆发排序 =====
# 1. 先绘制原始顺序的图（保持原有代码）
print("开始绘制原始顺序的Trace图...")
if has_behavior and behavior_data.dropna().unique().size > 0:
    # 如果有行为数据，使用两行一列的布局
    fig = plt.figure(figsize=(40, 15))
    # 使用GridSpec，并增加间距，解决tight_layout警告
    grid = GridSpec(2, 1, height_ratios=[1, 5], hspace=0.05, figure=fig)
    ax_behavior = fig.add_subplot(grid[0])
    ax_trace = fig.add_subplot(grid[1])
else:
    # 没有行为数据，只创建一个图表
    fig = plt.figure(figsize=(40, 15))
    ax_trace = fig.add_subplot(111)

# 预定义颜色映射，与热图保持一致
fixed_color_map = {
    'Open': '#ff7f0e',  # 橙色
    'Close': '#1f77b4',  # 蓝色
    'Middle': '#2ca02c',   # 绿色
    'WI': '#d62728',   # 红色
    'LVR': '#9467bd',  # 紫色
    'FM': '#8c564b',   # 棕色
    'GR': '#e377c2',   # 粉色
    'ELT': '#7f7f7f',  # 灰色
    'CM': '#bcbd22',   # 黄绿色
    'CT': '#17becf',   # 海蓝色
    'NI': '#aec7e8',   # 浅蓝色
    'EM': '#98df8a',   # 浅绿色
    'RF': '#ff9896',   # 浅红色
    'RN': '#c5b0d5'    # 浅紫色
}

# 绘制Trace图
for i, column in enumerate(trace_data.columns):
    if i >= Config.MAX_NEURONS:
        break
    
    # 计算当前神经元trace的垂直偏移量，并应用缩放因子
    ax_trace.plot(
        trace_data.index / Config.SAMPLING_RATE,  # x轴是时间(秒) = 时间戳 / 采样率
        trace_data[column] * Config.SCALING_FACTOR + (i+1) * Config.TRACE_OFFSET,  # 应用缩放并偏移
        linewidth=Config.LINE_WIDTH,
        alpha=Config.TRACE_ALPHA,
        label=column
    )

# 原始图中的Y轴设置修复
# 设置Y轴标签，简化显示格式
yticks = np.arange(Config.TRACE_OFFSET, (len(trace_data.columns)+1) * Config.TRACE_OFFSET, Config.TRACE_OFFSET)
ytick_labels = [str(column) for column in trace_data.columns[:Config.MAX_NEURONS]]
ax_trace.set_yticks(yticks[:len(ytick_labels)])
ax_trace.set_yticklabels(ytick_labels)

# 设置X轴范围
if Config.STAMP_MIN is not None and Config.STAMP_MAX is not None:
    ax_trace.set_xlim(Config.STAMP_MIN / Config.SAMPLING_RATE, Config.STAMP_MAX / Config.SAMPLING_RATE)

# 设置轴标签和标题
ax_trace.set_xlabel('Time (seconds)', fontsize=12)
ax_trace.set_ylabel('Neuron ID', fontsize=12)

# 添加网格线，使trace更容易阅读
ax_trace.grid(False)

# 处理行为区间数据
behavior_intervals = {}
unique_behaviors = []

# 只有当behavior列存在时才处理行为标签
if has_behavior:
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
                behavior_intervals[current_behavior].append((start_time / Config.SAMPLING_RATE, extended_index[i-1] / Config.SAMPLING_RATE))
            break
        
        # 跳过空值
        if pd.isna(behavior):
            # 如果之前有行为，则结束当前区间
            if start_time is not None and current_behavior is not None:
                behavior_intervals[current_behavior].append((start_time / Config.SAMPLING_RATE, timestamp / Config.SAMPLING_RATE))
                start_time = None
                current_behavior = None
            continue
        
        # 如果是新的行为类型或第一个行为
        if behavior != current_behavior:
            # 如果之前有行为，先结束当前区间
            if start_time is not None and current_behavior is not None:
                behavior_intervals[current_behavior].append((start_time / Config.SAMPLING_RATE, timestamp / Config.SAMPLING_RATE))
            
            # 开始新的行为区间
            start_time = timestamp
            current_behavior = behavior

# 绘制行为标记（如果存在）
if has_behavior and len(unique_behaviors) > 0:
    # 创建图例补丁列表
    legend_patches = []
    
    # 设置行为区域的展示参数
    ax_behavior.set_ylim(0, 1)  # 固定高度
    ax_behavior.set_xlim(ax_trace.get_xlim())  # 确保与主图x轴范围一致
    
    # 隐藏行为区域的y轴刻度和标签
    ax_behavior.set_yticks([])
    
    # 为每种行为分配颜色
    color_map = {}
    for behavior in unique_behaviors:
        if behavior in fixed_color_map:
            color_map[behavior] = fixed_color_map[behavior]
        else:
            # 对于未预定义的行为，使用动态颜色
            colors = plt.cm.Set3(np.linspace(0, 1, len(unique_behaviors)*2))
            color_map[behavior] = colors[list(unique_behaviors).index(behavior)]
    
    # 收集所有行为变化点用于绘制垂直分隔线
    behavior_change_points = set()
    
    # 按行为堆叠显示区间
    y_positions = {}
    total_behaviors = len(unique_behaviors)
    for i, behavior in enumerate(unique_behaviors):
        # 计算行为区间的垂直位置，将不同行为错开显示
        y_pos = 1.0 - (i + 1) / (total_behaviors + 1)
        y_positions[behavior] = y_pos
        
        # 为此行为创建图例项
        legend_patches.append(mpatches.Patch(color=color_map[behavior], label=behavior))
        
        # 绘制此行为的所有区间
        for start, end in behavior_intervals[behavior]:
            # 添加起点和终点到变化点集合
            behavior_change_points.add(start)
            behavior_change_points.add(end)
            
            # 创建一个矩形区域表示行为区间
            rect = mpatches.Rectangle(
                (start, y_pos - 0.1),  # 左下角坐标
                end - start,           # 宽度
                0.2,                   # 高度
                edgecolor='none',      # 边框颜色
                facecolor=color_map[behavior],  # 填充颜色
                alpha=0.7,             # 透明度
            )
            ax_behavior.add_patch(rect)
            
            # 仿照init_show.py添加行为标签文本
            # ax_behavior.text(start, 1, str(behavior), rotation=90, verticalalignment='top', fontsize=10)
    
    # 在每个行为变化点处添加垂直虚线
    for change_point in behavior_change_points:
        # 在行为图中添加虚线
        ax_behavior.axvline(x=change_point, color='gray', linestyle='--', linewidth=0.8, alpha=0.5)
        # 在主图中也添加虚线
        ax_trace.axvline(x=change_point, color='gray', linestyle='--', linewidth=0.8, alpha=0.5)
    
    # 添加行为图例，修改位置避免布局问题
    legend = ax_behavior.legend(
        handles=legend_patches, 
        loc='upper right', 
        fontsize=10, 
        title='Behavior Types', 
        title_fontsize=12
    )
    
    # 仅在行为轴上方显示x轴刻度
    ax_trace.xaxis.set_tick_params(labelbottom=True)
    ax_behavior.xaxis.set_tick_params(labelbottom=False)

# 生成标题，包含时间区间信息
title_text = 'Traces with Increased Amplitude'
if Config.STAMP_MIN is not None or Config.STAMP_MAX is not None:
    min_stamp = Config.STAMP_MIN if Config.STAMP_MIN is not None else trace_data.index.min()
    max_stamp = Config.STAMP_MAX if Config.STAMP_MAX is not None else trace_data.index.max()
    min_seconds = min_stamp / Config.SAMPLING_RATE
    max_seconds = max_stamp / Config.SAMPLING_RATE
    title_text += f' (Time Range: {min_seconds:.2f}s - {max_seconds:.2f}s)'

# 添加标题
plt.suptitle(title_text, fontsize=16, y=0.98)

# 调整布局
# plt.tight_layout()  # 这会导致警告，因为GridSpec布局与tight_layout不兼容
# 使用subplots_adjust代替tight_layout
if has_behavior:
    # 有行为图时的布局调整
    plt.subplots_adjust(top=0.90, bottom=0.1, left=0.1, right=0.95, hspace=0.15)
else:
    # 无行为图时的布局调整
    plt.subplots_adjust(top=0.90, bottom=0.1, left=0.1, right=0.95)

# 从输入文件路径中提取文件名（不包括路径和扩展名）
input_filename = os.path.basename(Config.INPUT_FILE)
input_filename = os.path.splitext(input_filename)[0]  # 去除扩展名

# 构建输出文件名：目录 + 前缀 + 输入文件名 + 时间戳信息
stamp_info = ''
if Config.STAMP_MIN is not None or Config.STAMP_MAX is not None:
    min_stamp = Config.STAMP_MIN if Config.STAMP_MIN is not None else trace_data.index.min()
    max_stamp = Config.STAMP_MAX if Config.STAMP_MAX is not None else trace_data.index.max()
    min_seconds = min_stamp / Config.SAMPLING_RATE
    max_seconds = max_stamp / Config.SAMPLING_RATE
    stamp_info = f'_{min_seconds:.2f}s_{max_seconds:.2f}s'

output_file = f"{Config.OUTPUT_DIR}traces_amplitude_{input_filename}{stamp_info}.png"
print(f"正在保存图像到 {output_file}")

# 保存图像
plt.savefig(output_file, dpi=300)
print(f"图像已保存")

# ===== 开始绘制第二个图：按钙爆发排序的图 =====
print("开始绘制按钙爆发排序的Trace图...")
# 创建新的图形
plt.close()  # 关闭前一个图形

if has_behavior and behavior_data.dropna().unique().size > 0:
    # 如果有行为数据，使用两行一列的布局
    fig_sorted = plt.figure(figsize=(40, 15))
    grid_sorted = GridSpec(2, 1, height_ratios=[1, 5], hspace=0.05, figure=fig_sorted)
    ax_behavior_sorted = fig_sorted.add_subplot(grid_sorted[0])
    ax_trace_sorted = fig_sorted.add_subplot(grid_sorted[1])
else:
    # 没有行为数据，只创建一个图表
    fig_sorted = plt.figure(figsize=(40, 15))
    ax_trace_sorted = fig_sorted.add_subplot(111)

# 绘制按钙爆发排序的Trace图
for i, column in enumerate(sorted_neurons_by_first_peak):
    if i >= Config.MAX_NEURONS:
        break
    
    # 计算当前神经元trace的垂直偏移量，并应用缩放因子
    # 将索引反过来，使早期钙爆发的神经元显示在Y轴上方
    position = Config.MAX_NEURONS - i if i < Config.MAX_NEURONS else 1
    
    ax_trace_sorted.plot(
        trace_data.index / Config.SAMPLING_RATE,  # x轴是时间(秒)
        trace_data[column] * Config.SCALING_FACTOR + position * Config.TRACE_OFFSET,  # 应用缩放并偏移
        linewidth=Config.LINE_WIDTH,
        alpha=Config.TRACE_ALPHA,
        label=column
    )
    
    # 标记第一次钙爆发的位置
    first_peak_time = first_peak_times[column] / Config.SAMPLING_RATE
    ax_trace_sorted.scatter(
        first_peak_time, 
        trace_data.loc[first_peak_times[column], column] * Config.SCALING_FACTOR + position * Config.TRACE_OFFSET,
        color='red', s=30, zorder=3  # zorder确保点在线的上方
    )

# 排序图中的Y轴设置修复
# 设置Y轴标签
total_positions = min(Config.MAX_NEURONS, len(sorted_neurons_by_first_peak))
yticks_sorted = []
ytick_labels_sorted = []

# 为每个trace计算正确的位置和标签
for i, column in enumerate(sorted_neurons_by_first_peak):
    if i >= Config.MAX_NEURONS:
        break
    
    # 计算位置 - 这需要与上面绘制时的position计算完全一致
    position = Config.MAX_NEURONS - i if i < Config.MAX_NEURONS else 1
    
    # 将位置和对应的标签添加到列表
    yticks_sorted.append(position * Config.TRACE_OFFSET)
    ytick_labels_sorted.append(str(column))

# 设置Y轴刻度和标签
ax_trace_sorted.set_yticks(yticks_sorted)
ax_trace_sorted.set_yticklabels(ytick_labels_sorted)

# 设置X轴范围
if Config.STAMP_MIN is not None and Config.STAMP_MAX is not None:
    ax_trace_sorted.set_xlim(Config.STAMP_MIN / Config.SAMPLING_RATE, Config.STAMP_MAX / Config.SAMPLING_RATE)

# 设置轴标签和标题
ax_trace_sorted.set_xlabel('Time (seconds)', fontsize=12)
ax_trace_sorted.set_ylabel('Neuron ID (Sorted by First Calcium Event)', fontsize=12)

# 添加网格线
ax_trace_sorted.grid(False)

# 如果有行为数据，再次绘制行为区间
if has_behavior and len(unique_behaviors) > 0:
    # 设置行为区域参数
    ax_behavior_sorted.set_ylim(0, 1)
    ax_behavior_sorted.set_xlim(ax_trace_sorted.get_xlim())
    ax_behavior_sorted.set_yticks([])
    
    # 再次收集行为变化点
    behavior_change_points_sorted = set()
    
    # 按行为堆叠显示区间
    for i, behavior in enumerate(unique_behaviors):
        y_pos = 1.0 - (i + 1) / (total_behaviors + 1)
        
        # 绘制此行为的所有区间
        for start, end in behavior_intervals[behavior]:
            # 添加起点和终点到变化点集合
            behavior_change_points_sorted.add(start)
            behavior_change_points_sorted.add(end)
            
            # 创建矩形区域
            rect = mpatches.Rectangle(
                (start, y_pos - 0.1), 
                end - start, 0.2,
                edgecolor='none',
                facecolor=color_map[behavior],
                alpha=0.7
            )
            ax_behavior_sorted.add_patch(rect)
    
    # 在每个行为变化点处添加垂直虚线
    for change_point in behavior_change_points_sorted:
        # 在行为图中添加虚线
        ax_behavior_sorted.axvline(x=change_point, color='gray', linestyle='--', linewidth=0.8, alpha=0.5)
        # 在主图中也添加虚线
        ax_trace_sorted.axvline(x=change_point, color='gray', linestyle='--', linewidth=0.8, alpha=0.5)
    
    # 添加行为图例
    ax_behavior_sorted.legend(
        handles=legend_patches,
        loc='upper right',
        fontsize=10,
        title='Behavior Types',
        title_fontsize=12
    )
    
    # 设置轴参数
    ax_trace_sorted.xaxis.set_tick_params(labelbottom=True)
    ax_behavior_sorted.xaxis.set_tick_params(labelbottom=False)

# 生成标题
sorted_title_text = 'Traces Sorted by First Calcium Event'
if Config.STAMP_MIN is not None or Config.STAMP_MAX is not None:
    min_seconds = min_stamp / Config.SAMPLING_RATE
    max_seconds = max_stamp / Config.SAMPLING_RATE
    sorted_title_text += f' (Time Range: {min_seconds:.2f}s - {max_seconds:.2f}s)'

# 添加标题
plt.suptitle(sorted_title_text, fontsize=16, y=0.98)

# 调整布局
if has_behavior:
    plt.subplots_adjust(top=0.90, bottom=0.1, left=0.1, right=0.95, hspace=0.15)
else:
    plt.subplots_adjust(top=0.90, bottom=0.1, left=0.1, right=0.95)

# 构建输出文件名
sorted_output_file = f"{Config.OUTPUT_DIR}traces_sorted_by_calcium_{input_filename}{stamp_info}.png"
print(f"正在保存排序图像到 {sorted_output_file}")

# 保存排序后的图像
plt.savefig(sorted_output_file, dpi=300)
print(f"排序图像已保存")

# 显示图像（可选，可以根据需要取消注释）
# plt.show()
print("程序执行完成")
