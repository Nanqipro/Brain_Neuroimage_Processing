# 神经元活动热图生成工具（修复版本）
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
class Config:
    """
    配置类，包含所有可调整的参数
    
    Attributes
    ----------
    INPUT_FILE : str
        输入数据文件路径
    OUTPUT_PREFIX : str
        输出文件名前缀
    STAMP_MIN : float or None
        最小时间戳
    STAMP_MAX : float or None
        最大时间戳
    SORT_METHOD : str
        排序方式：'peak'或'calcium_wave'
    CALCIUM_WAVE_THRESHOLD : float
        钙波检测阈值
    MIN_PROMINENCE : float
        最小峰值突出度
    MIN_RISE_RATE : float
        最小上升速率
    MAX_FALL_RATE : float
        最大下降速率
    """
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

def parse_args():
    """
    解析命令行参数
    
    Returns
    -------
    argparse.Namespace
        解析后的命令行参数
    """
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

def detect_first_calcium_wave(neuron_data: pd.Series) -> float:
    """
    检测神经元第一次真实钙波发生的时间点
    
    Parameters
    ----------
    neuron_data : pd.Series
        包含神经元活动的时间序列数据（标准化后）
    
    Returns
    -------
    float
        第一次真实钙波发生的时间点，如果没有检测到则返回数据最后一个时间点
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

def main():
    """
    主函数：加载数据、处理、绘图并保存
    """
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

    # 根据排序方式选择相应的排序算法
    if Config.SORT_METHOD == 'peak':
        # 原始方法：按峰值时间排序
        peak_times = day6_data_standardized.idxmax()
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

    # 找到所有行为标签的区间
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

    # 绘制热图并标注所有行为区间
    # 修复：使用合理的图形尺寸
    fig_width = min(20, max(12, len(sorted_day6_data.index) * 0.05))  # 根据数据点数量动态调整宽度
    fig_height = min(15, max(8, len(sorted_day6_data.columns) * 0.3))  # 根据神经元数量动态调整高度
    
    fig = plt.figure(figsize=(fig_width, fig_height))

    # 使用GridSpec布局，增加适当的边距
    from matplotlib.gridspec import GridSpec
    
    if has_behavior and len(unique_behaviors) > 0:
        # 有行为数据时使用两个子图
        grid = GridSpec(2, 1, height_ratios=[1, 4], hspace=0.05, 
                       left=0.1, right=0.95, top=0.95, bottom=0.1, figure=fig)
        ax_behavior = fig.add_subplot(grid[0])
        ax_heatmap = fig.add_subplot(grid[1])
    else:
        # 没有行为数据时只用一个子图
        grid = GridSpec(1, 1, left=0.1, right=0.95, top=0.95, bottom=0.15, figure=fig)
        ax_heatmap = fig.add_subplot(grid[0])

    # 设置绘图颜色范围
    vmin, vmax = -2, 2

    # 绘制热图
    heatmap = sns.heatmap(sorted_day6_data.T, cmap='viridis', cbar=True, 
                         vmin=vmin, vmax=vmax, ax=ax_heatmap)

    # 处理行为标记（如果存在）
    if has_behavior and len(unique_behaviors) > 0:
        # 创建固定的行为颜色映射
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
                color_map[behavior] = fixed_color_map[behavior]
            else:
                colors = plt.cm.tab10(np.linspace(0, 1, len(unique_behaviors)))
                color_map[behavior] = colors[i % 10]
        
        # 创建图例补丁列表
        legend_patches = []
        
        # 为每种行为分配Y轴位置
        y_positions = {}
        max_position = len(unique_behaviors)
        for i, behavior in enumerate(unique_behaviors):
            y_positions[behavior] = max_position - i
        
        # 为每种行为绘制区间
        for behavior, intervals in behavior_intervals.items():
            behavior_color = color_map[behavior]
            
            for start_time, end_time in intervals:
                if start_time in sorted_day6_data.index and end_time in sorted_day6_data.index:
                    start_idx = sorted_day6_data.index.get_loc(start_time)
                    end_idx = sorted_day6_data.index.get_loc(end_time)
                    
                    start_pos = start_idx
                    end_pos = end_idx
                    
                    if end_pos - start_pos > 0:
                        # 在行为标记子图中绘制区间
                        rect = plt.Rectangle((start_pos - 0.5, y_positions[behavior] - 0.4), 
                                           end_pos - start_pos, 0.8, 
                                           color=behavior_color, alpha=0.8, 
                                           ec='black', linewidth=0.5)
                        ax_behavior.add_patch(rect)
                        
                        # 在热图中添加区间边界垂直线
                        ax_heatmap.axvline(x=start_pos - 0.5, color='white', 
                                         linestyle='--', linewidth=1, alpha=0.7)
                        ax_heatmap.axvline(x=end_pos - 0.5, color='white', 
                                         linestyle='--', linewidth=1, alpha=0.7)
            
            # 添加到图例
            legend_patches.append(plt.Rectangle((0, 0), 1, 1, color=behavior_color, 
                                              alpha=0.8, label=behavior))
        
        # 设置行为子图
        ax_behavior.set_xlim(ax_heatmap.get_xlim())
        ax_behavior.set_ylim(0, max_position + 1)
        ax_behavior.set_yticks([y_positions[b] for b in unique_behaviors])
        ax_behavior.set_yticklabels(unique_behaviors, fontsize=10)
        ax_behavior.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
        ax_behavior.set_title('Behavior Intervals', fontsize=14, pad=10)
        
        # 修复：调整图例位置，避免溢出
        legend = ax_behavior.legend(handles=legend_patches, loc='center left', 
                                  fontsize=9, title='Behavior Types', title_fontsize=10, 
                                  bbox_to_anchor=(1.02, 0.5))

    # 在第426帧处添加白色虚线（如果数据中有足够的时间戳）
    if len(sorted_day6_data.index) > 426:
        ax_heatmap.axvline(x=426 - 0.5, color='white', linestyle='--', linewidth=2)

    # 生成标题
    title_text = f'BLA62500627homecagecelltrace-heatmap ({sort_method_str})'
    if Config.STAMP_MIN is not None or Config.STAMP_MAX is not None:
        min_stamp = Config.STAMP_MIN if Config.STAMP_MIN is not None else day6_data.index.min()
        max_stamp = Config.STAMP_MAX if Config.STAMP_MAX is not None else day6_data.index.max()
        title_text += f' (Time: {min_stamp:.2f} to {max_stamp:.2f})'

    # 修复：使用适当的字体大小
    title_fontsize = min(16, max(12, fig_width * 0.8))
    label_fontsize = min(14, max(10, fig_width * 0.6))
    tick_fontsize = min(12, max(8, fig_width * 0.5))

    # 为热图添加标题和标签
    ax_heatmap.set_title(title_text, fontsize=title_fontsize)
    ax_heatmap.set_xlabel('Timestamp', fontsize=label_fontsize)
    ax_heatmap.set_ylabel('Neuron ID', fontsize=label_fontsize)

    # 设置刻度标签字体大小
    ax_heatmap.set_yticklabels(ax_heatmap.get_yticklabels(), 
                              fontsize=tick_fontsize, rotation=0)
    ax_heatmap.set_xticklabels(ax_heatmap.get_xticklabels(), 
                              fontsize=tick_fontsize, rotation=45)

    # 构建输出文件名
    output_filename = f"{Config.OUTPUT_PREFIX}BLA62500627homecagecelltrace_{Config.SORT_METHOD}"
    if Config.STAMP_MIN is not None or Config.STAMP_MAX is not None:
        min_stamp = Config.STAMP_MIN if Config.STAMP_MIN is not None else day6_data.index.min()
        max_stamp = Config.STAMP_MAX if Config.STAMP_MAX is not None else day6_data.index.max()
        output_filename += f"_time_{min_stamp:.2f}_to_{max_stamp:.2f}"
    output_filename += "_fixed.png"

    # 保存图像
    fig.savefig(output_filename, bbox_inches='tight', pad_inches=0.3, dpi=150)
    plt.close()

    # 输出保存信息
    print(f"热图已保存至: {output_filename}")
    print(f"图形尺寸: {fig_width:.1f} x {fig_height:.1f} 英寸")

if __name__ == "__main__":
    main() 