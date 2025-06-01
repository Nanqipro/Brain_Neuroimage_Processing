# 有放入CD1的数据进行热图绘制
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import argparse
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvas
from scipy.signal import find_peaks
from scipy import stats

# 自定义参数配置
# 可以根据需要修改默认值
class Config:
    # 输入文件路径
    INPUT_FILE = '../../datasets/BLA62500627homecagecelltrace.xlsx'
    # 输出文件名前缀
    OUTPUT_PREFIX = '../../graph/heatmap_sort_'
    # 时间戳区间默认值（None表示不限制）
    STAMP_MIN = None  # 最小时间戳
    STAMP_MAX = None  # 最大时间戳
    # 排序方式：'peak'（默认，按峰值时间排序）或'calcium_wave'（按第一次真实钙波发生时间排序）
    SORT_METHOD = 'peak'
    # 钙波检测参数
    CALCIUM_WAVE_THRESHOLD = 1.5  # 钙波阈值（标准差的倍数）
    MIN_PROMINENCE = 1.0  # 最小峰值突出度
    MIN_RISE_RATE = 0.1  # 最小上升速率
    MAX_FALL_RATE = 0.05  # 最大下降速率（下降应当比上升慢）

# 解析命令行参数（如果需要从命令行指定参数）
def parse_args():
    parser = argparse.ArgumentParser(description='神经元活动热图生成工具，支持自定义时间区间和排序方式')
    parser.add_argument('--input', type=str, help='输入数据文件路径')
    parser.add_argument('--output-prefix', type=str, help='输出文件名前缀')
    parser.add_argument('--stamp-min', type=float, help='最小时间戳值')
    parser.add_argument('--stamp-max', type=float, help='最大时间戳值')
    parser.add_argument('--sort-method', type=str, choices=['peak', 'calcium_wave'], 
                        help='排序方式：peak（按峰值时间排序）或calcium_wave（按第一次真实钙波时间排序）')
    parser.add_argument('--ca-threshold', type=float, help='钙波检测阈值（标准差的倍数）')
    parser.add_argument('--min-prominence', type=float, help='最小峰值突出度')
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
if args.sort_method:
    Config.SORT_METHOD = args.sort_method
if args.ca_threshold is not None:
    Config.CALCIUM_WAVE_THRESHOLD = args.ca_threshold
if args.min_prominence is not None:
    Config.MIN_PROMINENCE = args.min_prominence

# 加载数据
day6_data = pd.read_excel(Config.INPUT_FILE)

# 将 'stamp' 列设置为索引
day6_data = day6_data.set_index('stamp')

# 根据配置的时间戳区间筛选数据
if Config.STAMP_MIN is not None or Config.STAMP_MAX is not None:
    # 确定实际的最小值和最大值
    min_stamp = Config.STAMP_MIN if Config.STAMP_MIN is not None else day6_data.index.min()
    max_stamp = Config.STAMP_MAX if Config.STAMP_MAX is not None else day6_data.index.max()
    
    # 筛选数据，保留指定区间内的数据
    day6_data = day6_data.loc[min_stamp:max_stamp]

# 检查是否存在 'behavior' 列
has_behavior = 'behavior' in day6_data.columns

# 分离 'behavior' 列（如果存在）
if has_behavior:
    frame_lost = day6_data['behavior']
    day6_data = day6_data.drop(columns=['behavior'])

# 数据标准化（Z-score 标准化）
day6_data_standardized = (day6_data - day6_data.mean()) / day6_data.std()

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
    threshold = Config.CALCIUM_WAVE_THRESHOLD
    
    # 使用find_peaks函数检测峰值
    peaks, properties = find_peaks(neuron_data, 
                                 height=threshold, 
                                 prominence=Config.MIN_PROMINENCE,
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
        if rise_rate > Config.MIN_RISE_RATE and 0 < fall_rate < Config.MAX_FALL_RATE:
            # 找到第一个真实钙波，返回时间点
            return neuron_data.index[peak_idx]
    
    # 如果没有满足条件的钙波，返回时间序列的最后一个点
    return neuron_data.index[-1]

# 根据排序方式选择相应的排序算法
if Config.SORT_METHOD == 'peak':
    # 原始方法：按峰值时间排序
    # 对于每个神经元，找到其信号达到最大值的时间戳
    peak_times = day6_data_standardized.idxmax()
    
    # 将神经元按照峰值时间从早到晚排序
    sorted_neurons = peak_times.sort_values().index
    
    sort_method_str = "Sorted by peak time"
else:  # 'calcium_wave'
    # 新方法：按第一次真实钙波发生时间排序
    first_wave_times = {}
    
    # 对每个神经元进行钙波检测
    for neuron in day6_data_standardized.columns:
        neuron_data = day6_data_standardized[neuron]
        first_wave_times[neuron] = detect_first_calcium_wave(neuron_data)
    
    # 转换为Series以便排序
    first_wave_times_series = pd.Series(first_wave_times)
    
    # 按第一次钙波时间排序
    sorted_neurons = first_wave_times_series.sort_values().index
    
    sort_method_str = "Sorted by first calcium wave time"

# 根据排序后的神经元顺序重新排列 DataFrame 的列
sorted_day6_data = day6_data_standardized[sorted_neurons]

# **步骤4：找到所有行为标签的区间**

# 初始化行为区间变量
behavior_intervals = {}
unique_behaviors = []

# 只有当behavior列存在时才处理行为标签
if has_behavior:
    # 获取所有不同的行为标签
    unique_behaviors = frame_lost.dropna().unique()
    
    # 初始化所有行为的区间字典
    for behavior in unique_behaviors:
        behavior_intervals[behavior] = []
    
    # 对frame_lost进行处理，找出每种行为的连续区间
    current_behavior = None
    start_time = None
    
    # 为了确保最后一个区间也被记录，将索引列表扩展一个元素
    extended_index = list(frame_lost.index) + [None]
    extended_values = list(frame_lost.values) + [None]
    
    for i, (timestamp, behavior) in enumerate(zip(extended_index, extended_values)):
        # 最后一个元素特殊处理
        if i == len(frame_lost):
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

# **步骤5：绘制热图并标注所有行为区间**

# 设置绘图颜色范围
vmin, vmax = -2, 2  # 控制颜色对比度

# 创建图形和轴，使用更高的高度比例和精确调整来确保对齐
fig = plt.figure(figsize=(200, 15))

# 使用更精确的GridSpec布局系统
# 通过更大的底部空间留出足够的标签空间
from matplotlib.gridspec import GridSpec
grid = GridSpec(2, 1, height_ratios=[1, 5], hspace=0.0, figure=fig)

# 先创建热图子图
ax_heatmap = fig.add_subplot(grid[1])

# 为了确保对齐，我们首先定义X轴范围
# 创建一个X轴数据点数组，这个数组将用于两个图表
data_x_points = np.arange(len(sorted_day6_data.index))

# 绘制热图
heatmap = sns.heatmap(sorted_day6_data.T, cmap='viridis', cbar=True, vmin=vmin, vmax=vmax, ax=ax_heatmap)

# 置空刻度位置，稍后设置
# 设置热图的x轴范围为精确数据点范围
ax_heatmap.set_xlim(-0.5, len(sorted_day6_data.index) - 0.5)

# 创建行为标记子图作为单独的子图
ax_behavior = fig.add_subplot(grid[0])

# 手动设置行为子图的X轴范围与热图完全相匹配
ax_behavior.set_xlim(-0.5, len(sorted_day6_data.index) - 0.5)

# 只有当behavior列存在时才添加行为标记
if has_behavior and len(unique_behaviors) > 0:
    # 创建固定的行为颜色映射，确保相同行为始终使用相同颜色
    # 预定义所有可能的行为及其固定颜色，使用更加鲜明和对比度更高的颜色
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
    
    # 为当前数据集中的行为创建颜色映射
    color_map = {}
    for i, behavior in enumerate(unique_behaviors):
        if behavior in fixed_color_map:
            # 使用预定义的颜色
            color_map[behavior] = fixed_color_map[behavior]
        else:
            # 对于未预定义的行为，使用更鲜明的调色板
            # 从tab10调色板获取颜色，它提供更高的对比度
            colors = plt.cm.tab10(np.linspace(0, 1, len(unique_behaviors)))
            color_map[behavior] = colors[i % 10]
    
    # 创建图例补丁列表
    legend_patches = []
    
    # 为每种行为绘制区间
    y_positions = {}
    max_position = len(unique_behaviors)
    # 为每种行为分配Y轴位置
    for i, behavior in enumerate(unique_behaviors):
        y_positions[behavior] = max_position - i
    
    # 为每种行为绘制区间
    for behavior, intervals in behavior_intervals.items():
        behavior_color = color_map[behavior]
        
        for start_time, end_time in intervals:
            # 检查时间点是否在排序后的数据索引中
            if start_time in sorted_day6_data.index and end_time in sorted_day6_data.index:
                # 获取对应的绘图位置
                start_idx = sorted_day6_data.index.get_loc(start_time)  # 获取索引位置
                end_idx = sorted_day6_data.index.get_loc(end_time)
                
                # 在使用前转换为实际坐标位置 (对应网格)
                start_pos = start_idx # 网格的中心是整数位置
                end_pos = end_idx
                
                # 如果区间有宽度
                if end_pos - start_pos > 0:  
                    # 在行为标记子图中绘制区间，使用精确的数据点对齐方式
                    # 提高alpha值从0.7到0.9，使颜色更加明显
                    rect = plt.Rectangle((start_pos - 0.5, y_positions[behavior] - 0.4), 
                                        end_pos - start_pos, 0.8, 
                                        color=behavior_color, alpha=0.9, 
                                        ec='black')  # 添加黑色边框以增强可见度
                    ax_behavior.add_patch(rect)
                    
                    # 在热图中添加区间边界垂直线
                    # 使用与热图网格线一致的位置，确保对齐
                    ax_heatmap.axvline(x=start_pos - 0.5, color='white', linestyle='--', linewidth=2, alpha=0.5)
                    ax_heatmap.axvline(x=end_pos - 0.5, color='white', linestyle='--', linewidth=2, alpha=0.5)
            
        # 添加到图例，同样提高alpha值
        legend_patches.append(plt.Rectangle((0, 0), 1, 1, color=behavior_color, alpha=0.9, label=behavior))
    
    # 再次确认两个坐标轴的对齐情况
    # 唯一正确的做法是设置完全相同的范围
    ax_heatmap.set_xlim(-0.5, len(sorted_day6_data.index) - 0.5)
    ax_behavior.set_xlim(-0.5, len(sorted_day6_data.index) - 0.5)
    
    # 设置Y轴范围和刻度位置
    ax_behavior.set_ylim(0, max_position + 1)
    # 先设置刻度位置，再设置刻度标签，避免警告
    ax_behavior.set_yticks([y_positions[b] for b in unique_behaviors])
    ax_behavior.set_yticklabels(unique_behaviors, fontsize=12, fontweight='bold')
    
    # 特别重要：移除X轴刻度，让它只在热图上显示
    ax_behavior.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
    ax_behavior.set_title('Behavior Intervals', fontsize=16, pad=10)
    ax_behavior.set_xlabel('')  # Remove x-axis label, shared with the heatmap below

    # 更强制地隐藏X轴刻度和标签
    ax_behavior.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
    # 确保行为图和热图水平对齐
    ax_behavior.set_anchor('SW')
    # 去除行为子图边框
    ax_behavior.spines['top'].set_visible(False)
    ax_behavior.spines['right'].set_visible(False)
    ax_behavior.spines['bottom'].set_visible(False)
    ax_behavior.spines['left'].set_visible(False)
    
    # 添加图例
    legend = ax_behavior.legend(handles=legend_patches, loc='upper right', fontsize=12, 
                           title='Behavior Types', title_fontsize=14, bbox_to_anchor=(1.0, 1.3))

# 在第426帧处添加白色虚线
# 检查数据中是否有足够的时间戳
if len(sorted_day6_data.index) > 426:
    # 绘制垂直线，白色虚线
    ax_heatmap.axvline(x=426 - 0.5, color='white', linestyle='--', linewidth=4)

# 生成标题，如果设置了时间区间，则在标题中显示区间信息
title_text = f'BLA62500627homecagecelltrace-heatmap ({sort_method_str})'
if Config.STAMP_MIN is not None or Config.STAMP_MAX is not None:
    min_stamp = Config.STAMP_MIN if Config.STAMP_MIN is not None else day6_data.index.min()
    max_stamp = Config.STAMP_MAX if Config.STAMP_MAX is not None else day6_data.index.max()
    title_text += f' (Time: {min_stamp:.2f} to {max_stamp:.2f})'

# 为热图添加标题和标签
ax_heatmap.set_title(title_text, fontsize=25)
ax_heatmap.set_xlabel('stamp', fontsize=25)
ax_heatmap.set_ylabel('neuron', fontsize=25)

# 修改Y轴标签（神经元标签）的字体大小和粗细，设置为水平方向
ax_heatmap.set_yticklabels(ax_heatmap.get_yticklabels(), fontsize=14, fontweight='bold', rotation=0)

# 修改X轴标签（时间戳）的字体大小和粗细
ax_heatmap.set_xticklabels(ax_heatmap.get_xticklabels(), fontsize=14, fontweight='bold')

# 应用紧凑布局
# 不使用tight_layout()，因为它与GridSpec布局不兼容
# 而是使用之前设置的subplots_adjust()已经足够调整布局

# 构建输出文件名，包含排序方式和时间区间信息（如果有）
output_filename = f"{Config.OUTPUT_PREFIX}BLA62500627homecagecelltrace_{Config.SORT_METHOD}"
if Config.STAMP_MIN is not None or Config.STAMP_MAX is not None:
    min_stamp = Config.STAMP_MIN if Config.STAMP_MIN is not None else day6_data.index.min()
    max_stamp = Config.STAMP_MAX if Config.STAMP_MAX is not None else day6_data.index.max()
    output_filename += f"_time_{min_stamp:.2f}_to_{max_stamp:.2f}"
output_filename += ".png"

# 在保存前使用多重方法确保对齐

# 1. 强制更新布局
fig.canvas.draw()

# 2. 再次确认轴范围一致
ax_heatmap.set_xlim(-0.5, len(sorted_day6_data.index) - 0.5)
ax_behavior.set_xlim(-0.5, len(sorted_day6_data.index) - 0.5)

# 3. 达到更精确的对齐
# 获取热图的实际边界位置（像素单位）
heatmap_bbox = ax_heatmap.get_position()
behavior_bbox = ax_behavior.get_position()

# 不能直接修改Bbox对象，需要创建新的
# 使用Bbox的坐标创建新的位置，保持高度不变，但使用热图的宽度和水平位置
from matplotlib.transforms import Bbox
new_behavior_pos = Bbox([[heatmap_bbox.x0, behavior_bbox.y0], 
                        [heatmap_bbox.x0 + heatmap_bbox.width, behavior_bbox.y0 + behavior_bbox.height]])

# 设置新的位置
ax_behavior.set_position(new_behavior_pos)

# 4. 保存图像
fig.savefig(output_filename, bbox_inches='tight', pad_inches=0.1, dpi=100)
plt.close()

# 输出保存信息
print(f"热图已保存至: {output_filename}")
