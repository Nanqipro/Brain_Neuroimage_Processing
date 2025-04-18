# 有放入CD1的数据进行热图绘制
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import argparse
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvas
from matplotlib.colors import Normalize

# 自定义参数配置
# 可以根据需要修改默认值
class Config:
    # 输入文件路径
    INPUT_FILE = '../../datasets/EMtrace_plus.xlsx'
    # 输出文件名前缀
    OUTPUT_PREFIX = '../../graph/heatmap_sort_'
    # 时间戳区间默认值（None表示不限制）
    STAMP_MIN = None  # 最小时间戳
    STAMP_MAX = None  # 最大时间戳

# 解析命令行参数（如果需要从命令行指定参数）
def parse_args():
    parser = argparse.ArgumentParser(description='神经元活动热图生成工具，支持自定义时间区间')
    parser.add_argument('--input', type=str, help='输入数据文件路径')
    parser.add_argument('--output-prefix', type=str, help='输出文件名前缀')
    parser.add_argument('--stamp-min', type=float, help='最小时间戳值')
    parser.add_argument('--stamp-max', type=float, help='最大时间戳值')
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

# **步骤1：计算每个神经元信号峰值出现的时间点**

# 对于每个神经元，找到其信号达到最大值的时间戳
peak_times = day6_data_standardized.idxmax()

# **步骤2：按照峰值出现的时间对神经元进行排序**

# 将神经元按照峰值时间从早到晚排序
sorted_neurons = peak_times.sort_values().index

# **步骤3：重新排列数据**

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

# 创建新的全景布局，采用垂直堆叠的可视化方案
fig = plt.figure(figsize=(100, 15))

# 创建网格布局，指定高度比例和间距
# 关键点：将hspace设置为0，确保子图之间无间隔
grid = plt.GridSpec(2, 1, height_ratios=[1, 8], hspace=0.0, wspace=0.0)

# 采用完全新的方法绘制热图和行为区间，确保完美对齐

# 先创建下方热图
ax_heatmap = fig.add_subplot(grid[1])

# 计算数据点总数
ndata = len(sorted_day6_data)

# 使用原生的matplotlib imshow函数，而非seaborn heatmap
# 关键点：设置extent参数为初始数据的域范围，对应到精确的像素坐标
img = ax_heatmap.imshow(sorted_day6_data.T, 
                       aspect='auto', 
                       cmap='viridis', 
                       vmin=vmin, vmax=vmax,
                       interpolation='nearest',
                       extent=[-0.5, ndata-0.5, -0.5, len(sorted_day6_data.columns)-0.5],
                       origin='lower')

# 为热图添加颜色条
cbar = fig.colorbar(img, ax=ax_heatmap, shrink=0.8)

# 设置x轴刻度 - 对于长度较大的数据，选择间隔显示
xtick_step = max(len(sorted_day6_data) // 20, 1)  # 确保不过于拍数
xtick_indices = range(0, len(sorted_day6_data), xtick_step)
xtick_positions = np.array(list(xtick_indices))
xtick_labels = [f"{sorted_day6_data.index[i]:.1f}" for i in xtick_indices]
ax_heatmap.set_xticks(xtick_positions)
ax_heatmap.set_xticklabels(xtick_labels, fontsize=10, rotation=45)

# 设置y轴刻度 - 每个神经元一个刻度
ax_heatmap.set_yticks(range(len(sorted_day6_data.columns)))
ax_heatmap.set_yticklabels(sorted_day6_data.columns, fontsize=10)

# 创建上方行为区间标记子图
# 关键参数：sharex=ax_heatmap确保共享X轴
ax_behavior = fig.add_subplot(grid[0], sharex=ax_heatmap)

# 创建空白背景作为行为标记区域
ax_behavior.imshow(np.zeros((1, ndata)), 
                  aspect='auto', 
                  cmap='Greys', 
                  alpha=0,
                  extent=[-0.5, ndata-0.5, 0, len(unique_behaviors)],
                  origin='lower')

# 调整全局布局，确保紧凑并有足够空间显示全部内容
# 关键参数：hspace=0确保上下子图无间隔
fig.subplots_adjust(left=0.06, right=0.94, top=0.9, bottom=0.15, hspace=0)

# 只有当behavior列存在时才添加行为标记
if has_behavior and len(unique_behaviors) > 0:
    # 颜色映射，为每种行为分配不同的颜色
    colors = plt.cm.tab20(np.linspace(0, 1, len(unique_behaviors)))
    color_map = {behavior: colors[i] for i, behavior in enumerate(unique_behaviors)}
    
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
                # 获取对应的索引位置
                # 获取这个时间点在数据中的索引位置
                start_idx = sorted_day6_data.index.get_loc(start_time)
                end_idx = sorted_day6_data.index.get_loc(end_time)
                
                # 关键是使用完全相同的坐标系统在两个子图中绘制元素
                
                # 在行为区域绘制矩形
                rect = patches.Rectangle((start_idx, y_positions[behavior] - 0.4), 
                                       end_idx - start_idx, 0.8, 
                                       color=behavior_color, alpha=0.7,
                                       edgecolor=None, linewidth=0)
                ax_behavior.add_patch(rect)
                
                # 在热图中添加对应的垂直线
                ax_heatmap.axvline(x=start_idx, color='white', linestyle='--', linewidth=1, alpha=0.5)
                ax_heatmap.axvline(x=end_idx, color='white', linestyle='--', linewidth=1, alpha=0.5)
        
        # 添加到图例
        legend_patches.append(plt.Rectangle((0, 0), 1, 1, color=behavior_color, alpha=0.7, label=behavior))
    
    # 设置行为标记子图的属性
    ax_behavior.set_ylim(0, max_position + 1)
    ax_behavior.set_yticks([y_positions[b] for b in unique_behaviors])
    ax_behavior.set_yticklabels(unique_behaviors, fontsize=12, fontweight='bold')
    ax_behavior.set_title('Behavior Intervals', fontsize=16, pad=10)

    # 关键设置：隐藏行为图的x轴制作无缝过渡
    ax_behavior.set_xticks([])
    ax_behavior.set_xticklabels([])
    ax_behavior.xaxis.set_visible(False)

    # 去除行为图所有边框以创建清晰的布局
    for spine in ax_behavior.spines.values():
        spine.set_visible(False)

    # 关键设置：强制关闭所有刻度线
    ax_behavior.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
    ax_behavior.tick_params(axis='y', which='both', left=True, right=False, labelleft=True, pad=5)
    
    # 添加图例
    legend = ax_behavior.legend(handles=legend_patches, loc='upper right', fontsize=12, 
                           title='Behavior Types', title_fontsize=14, bbox_to_anchor=(1.0, 1.3))

# 生成标题，如果设置了时间区间，则在标题中显示区间信息
title_text = 'EMtrace-heatmap'
if Config.STAMP_MIN is not None or Config.STAMP_MAX is not None:
    min_stamp = Config.STAMP_MIN if Config.STAMP_MIN is not None else day6_data.index.min()
    max_stamp = Config.STAMP_MAX if Config.STAMP_MAX is not None else day6_data.index.max()
    title_text += f' (Time: {min_stamp:.2f} to {max_stamp:.2f})'

# 为热图添加标题和标签
ax_heatmap.set_title(title_text, fontsize=16)
ax_heatmap.set_xlabel('stamp', fontsize=20)
ax_heatmap.set_ylabel('neuron', fontsize=20)

# 注意：已在绘制imshow时设置了标签格式，这里不需要重复设置

# 应用紧凑布局
# 不使用tight_layout()，因为它与GridSpec布局不兼容
# 使用的subplots_adjust()已经足够调整布局
# 对于所有图元素，增加以下调用以确保尺寸一致
fig.canvas.draw()
# 确保热图的水平范围和行为区间图的水平范围完全匹配
fig.align_xlabels([ax_heatmap, ax_behavior])

# 构建输出文件名，包含时间区间信息（如果有）
output_filename = f"{Config.OUTPUT_PREFIX}EMtrace"
if Config.STAMP_MIN is not None or Config.STAMP_MAX is not None:
    min_stamp = Config.STAMP_MIN if Config.STAMP_MIN is not None else day6_data.index.min()
    max_stamp = Config.STAMP_MAX if Config.STAMP_MAX is not None else day6_data.index.max()
    output_filename += f"_time_{min_stamp:.2f}_to_{max_stamp:.2f}"
output_filename += ".png"

# 保存前的关键处理步骤

# 1. 强制绘制以应用所有设置
fig.canvas.draw()

# 2. 确保两个子图使用完全相同的x轴范围
# 这是实现完美对齐的关键
ax_behavior.set_xlim(ax_heatmap.get_xlim())

# 3. 再次确认对齐
# 确保热图和行为图的轴对齐
fig.align_xlabels([ax_heatmap])

# 4. 保存图像
# 使用紧凑边界模式保存，减少冗余空白
plt.savefig(output_filename, bbox_inches='tight', pad_inches=0.1, dpi=100)
plt.close()

# 输出保存信息
print(f"热图已保存至: {output_filename}")
