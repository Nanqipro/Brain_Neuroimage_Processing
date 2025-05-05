# 神经元活动 Trace 图绘制，基于热图代码修改而来
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import argparse
from matplotlib.gridspec import GridSpec
import matplotlib.patches as mpatches
import os

# 自定义参数配置
# 可以根据需要修改默认值
class Config:
    # 输入文件路径
    INPUT_FILE = '../../datasets/EMtrace_plus.xlsx'
    # 输出文件名前缀
    OUTPUT_PREFIX = '../../graph/trace_plot_'
    # 时间戳区间默认值（None表示不限制）
    STAMP_MIN = None  # 最小时间戳
    STAMP_MAX = None  # 最大时间戳
    # Trace图的显示参数
    TRACE_OFFSET = 1.5  # 不同神经元trace之间的垂直偏移量
    MAX_NEURONS = 60    # 最大显示神经元数量（避免图表过于拥挤）
    TRACE_ALPHA = 0.8   # trace线的透明度
    LINE_WIDTH = 1.0    # trace线的宽度

# 解析命令行参数（如果需要从命令行指定参数）
def parse_args():
    parser = argparse.ArgumentParser(description='神经元活动 Trace 图生成工具，支持自定义时间区间')
    parser.add_argument('--input', type=str, help='输入数据文件路径')
    parser.add_argument('--output-prefix', type=str, help='输出文件名前缀')
    parser.add_argument('--stamp-min', type=float, help='最小时间戳值')
    parser.add_argument('--stamp-max', type=float, help='最大时间戳值')
    parser.add_argument('--max-neurons', type=int, help='最大显示神经元数量')
    return parser.parse_args()

# 解析命令行参数并更新配置
args = parse_args()
if args.input:
    Config.INPUT_FILE = args.input
if args.output_prefix:
    Config.OUTPUT_PREFIX = args.output_prefix
if args.stamp_min is not None:
    Config.STAMP_MIN = args.stamp_min
if args.stamp_max is not None:
    Config.STAMP_MAX = args.stamp_max
if args.max_neurons is not None:
    Config.MAX_NEURONS = args.max_neurons

# 加载数据
print(f"正在从 {Config.INPUT_FILE} 加载数据...")
trace_data = pd.read_excel(Config.INPUT_FILE)

# 将 'stamp' 列设置为索引
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

# 数据标准化（Z-score 标准化）
trace_data_standardized = (trace_data - trace_data.mean()) / trace_data.std()
print("已完成数据标准化处理")

# 计算每个神经元信号峰值出现的时间点
peak_times = trace_data_standardized.idxmax()

# 将神经元按照峰值时间从早到晚排序
sorted_neurons = peak_times.sort_values().index

# 根据排序后的神经元顺序重新排列 DataFrame 的列
sorted_trace_data = trace_data_standardized[sorted_neurons]

# 如果神经元数量超过最大设定值，只保留前N个
if len(sorted_neurons) > Config.MAX_NEURONS:
    print(f"神经元数量({len(sorted_neurons)})超过最大显示限制({Config.MAX_NEURONS})，将只显示前{Config.MAX_NEURONS}个")
    sorted_neurons = sorted_neurons[:Config.MAX_NEURONS]
    sorted_trace_data = sorted_trace_data[sorted_neurons]

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
                behavior_intervals[current_behavior].append((start_time, extended_index[i-1]))
            break
        
        # 跳过空值
        if pd.isna(behavior):
            # 如果之前有行为，则结束当前区间
            if start_time is not None and current_behavior is not None:
                behavior_intervals[current_behavior].append((start_time, timestamp))
                start_time = None
                current_behavior = None
            continue
        
        # 如果是新的行为类型或第一个行为
        if behavior != current_behavior:
            # 如果之前有行为，先结束当前区间
            if start_time is not None and current_behavior is not None:
                behavior_intervals[current_behavior].append((start_time, timestamp))
            
            # 开始新的行为区间
            start_time = timestamp
            current_behavior = behavior

# 创建图形
print("开始绘制Trace图...")
if has_behavior and len(unique_behaviors) > 0:
    # 如果有行为数据，使用两行一列的布局
    fig = plt.figure(figsize=(100, 15))
    # 使用GridSpec，并增加间距，解决tight_layout警告
    grid = GridSpec(2, 1, height_ratios=[1, 5], hspace=0.05, figure=fig)
    ax_behavior = fig.add_subplot(grid[0])
    ax_trace = fig.add_subplot(grid[1])
else:
    # 没有行为数据，只创建一个图表
    fig = plt.figure(figsize=(100, 15))
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
for i, neuron in enumerate(sorted_neurons):
    # 计算当前神经元trace的垂直偏移量
    offset = i * Config.TRACE_OFFSET
    
    # 绘制神经元活动trace线
    ax_trace.plot(
        sorted_trace_data.index,  # x轴是时间戳
        sorted_trace_data[neuron] + offset,  # y轴是信号值加上偏移量
        linewidth=Config.LINE_WIDTH,
        alpha=Config.TRACE_ALPHA,
        label=f'Neuron {neuron}'
    )

# 设置Y轴标签，显示神经元ID
yticks = np.arange(0, len(sorted_neurons) * Config.TRACE_OFFSET, Config.TRACE_OFFSET)
ytick_labels = [f'{i}' for i in sorted_neurons]
ax_trace.set_yticks(yticks)
ax_trace.set_yticklabels(ytick_labels)

# 设置X轴为时间戳，使用合适数量的刻度点
timestamps = sorted_trace_data.index
num_ticks = min(60, len(timestamps))  # 最多显示60个时间刻度
tick_indices = np.linspace(0, len(timestamps) - 1, num_ticks, dtype=int)
tick_positions = [timestamps[i] for i in tick_indices]
ax_trace.set_xticks(tick_positions)
ax_trace.set_xticklabels([f'{t:.2f}' for t in tick_positions], rotation=45)

# 设置轴标签和标题
ax_trace.set_xlabel('Timestamp (seconds)', fontsize=12)
ax_trace.set_ylabel('Neuron ID', fontsize=12)

# 添加网格线，使trace更容易阅读
ax_trace.grid(True, linestyle='--', alpha=0.6)

# 绘制行为标记（如果存在）
if has_behavior and len(unique_behaviors) > 0:
    # 创建图例补丁列表
    legend_patches = []
    
    # 设置行为区域的展示参数
    ax_behavior.set_ylim(0, 1)  # 固定高度
    ax_behavior.set_xlim(sorted_trace_data.index.min(), sorted_trace_data.index.max())
    
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
    
    # 添加行为图例，修改位置避免布局问题
    legend = ax_behavior.legend(
        handles=legend_patches, 
        loc='upper right', 
        fontsize=10, 
        title='Behavior Types', 
        title_fontsize=12
    )
    
    # 确保行为轴和trace轴的x轴对齐
    ax_behavior.set_xlim(ax_trace.get_xlim())
    
    # 仅在行为轴上方显示x轴刻度
    ax_trace.xaxis.set_tick_params(labelbottom=True)
    ax_behavior.xaxis.set_tick_params(labelbottom=False)

# 生成标题，包含时间区间信息
title_text = 'EMtrace - Neuron Activity Trace Plot'
if Config.STAMP_MIN is not None or Config.STAMP_MAX is not None:
    min_stamp = Config.STAMP_MIN if Config.STAMP_MIN is not None else trace_data.index.min()
    max_stamp = Config.STAMP_MAX if Config.STAMP_MAX is not None else trace_data.index.max()
    title_text += f' (Time Range: {min_stamp:.2f}s - {max_stamp:.2f}s)'

# 添加标题，调整位置以避免与图例重叠
fig.suptitle(title_text, fontsize=16, y=0.98)

# 调整布局 - 使用更可靠的方法替代tight_layout
# 不使用tight_layout，手动调整子图参数
# 避免使用tight_layout，因为在复杂布局中可能不兼容
if has_behavior:
    # 有行为图时的布局调整
    plt.subplots_adjust(top=0.90, bottom=0.1, left=0.1, right=0.95, hspace=0.15)
else:
    # 无行为图时的布局调整
    plt.subplots_adjust(top=0.90, bottom=0.1, left=0.1, right=0.95)

# 生成输出文件名
stamp_info = ''
if Config.STAMP_MIN is not None or Config.STAMP_MAX is not None:
    min_stamp = Config.STAMP_MIN if Config.STAMP_MIN is not None else trace_data.index.min()
    max_stamp = Config.STAMP_MAX if Config.STAMP_MAX is not None else trace_data.index.max()
    stamp_info = f'{min_stamp:.2f}_{max_stamp:.2f}'

# 从输入文件路径中提取文件名（不包括路径和扩展名）
input_filename = os.path.basename(Config.INPUT_FILE)
input_filename = os.path.splitext(input_filename)[0]  # 去除扩展名

# 构建输出文件名：前缀 + 输入文件名 + 时间戳信息
output_file = f"{Config.OUTPUT_PREFIX}{input_filename}{'_' + stamp_info if stamp_info else ''}.png"
print(f"正在保存图像到 {output_file}")

# 保存图像 - 使用bbox_inches='tight'以保证保存的图像包含所有元素
plt.savefig(output_file, dpi=300)
print(f"图像已保存")

# 显示图像
# plt.show()
print("程序执行完成")
