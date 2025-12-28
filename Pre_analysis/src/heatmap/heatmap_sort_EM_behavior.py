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
    INPUT_FILE = '../../datasets/5355homecage1107merge.xlsx'
    # 输出文件名前缀
    OUTPUT_PREFIX = '../../graph/heatmap_sort-'
    # 时间戳区间默认值（None表示不限制）
    STAMP_MIN = 0  # 最小时间戳
    STAMP_MAX = 12495  # 最大时间戳
    # 排序方式：'peak'（默认，按峰值时间排序）、'calcium_wave'（按第一次真实钙波发生时间排序）或'custom'（按自定义顺序排序）
    SORT_METHOD = 'peak'
    # 自定义神经元排序顺序（仅在SORT_METHOD='custom'时使用）
    CUSTOM_NEURON_ORDER = ['n53', 'n40', 'n29', 'n34', 'n4', 'n32', 'n25', 'n27', 'n22', 'n55', 'n21', 'n5', 'n19']
    # 采样率 (Hz)
    SAMPLING_RATE = 9.02  # 采样频率，用于将时间戳转换为秒
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
    parser.add_argument('--sort-method', type=str, choices=['peak', 'calcium_wave', 'custom'], 
                        help='排序方式：peak（按峰值时间排序）、calcium_wave（按第一次真实钙波时间排序）或custom（按自定义顺序排序）')
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
print(f"正在从 {Config.INPUT_FILE} 加载数据...")
day6_data_full = pd.read_excel(Config.INPUT_FILE)

# 将 'stamp' 列设置为索引
stamp_column = day6_data_full['stamp'].copy()  # 保存原始时间戳
# 创建秒为单位的时间索引
seconds_index = stamp_column / Config.SAMPLING_RATE
day6_data_full = day6_data_full.set_index('stamp')

# 检查是否存在 'behavior' 列
has_behavior = 'behavior' in day6_data_full.columns

# 分离 'behavior' 列（如果存在）
if has_behavior:
    behavior_data_full = day6_data_full['behavior']
    neural_data_full = day6_data_full.drop(columns=['behavior'])
else:
    neural_data_full = day6_data_full.copy()

# 基于全局数据进行神经元排序（重要：确保不同时间区间的神经元排序一致）
print("正在基于全局数据计算神经元排序...")
neural_data_full_standardized = (neural_data_full - neural_data_full.mean()) / neural_data_full.std()

# 函数：按自定义神经元顺序排序
def sort_neurons_by_custom_order(data_columns, custom_order):
    """
    按自定义神经元顺序排序
    
    指定神经元按给定顺序排在前面，剩余神经元按字符串排序排在后面
    
    参数:
    data_columns -- 数据中的神经元列名
    custom_order -- 自定义的神经元顺序列表
    
    返回:
    sorted_neurons -- 按自定义顺序排列的神经元列表
    """
    available_neurons = set(data_columns)
    
    # 首先按照自定义顺序排列存在的神经元
    ordered_neurons = []
    for neuron in custom_order:
        if neuron in available_neurons:
            ordered_neurons.append(neuron)
    
    # 找出剩余的神经元，按字符串大小顺序排列
    remaining_neurons = sorted(list(available_neurons - set(ordered_neurons)))
    
    # 合并两部分：自定义顺序 + 剩余神经元（按大小排序）
    final_order = ordered_neurons + remaining_neurons
    
    return final_order

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

# 根据排序方式选择相应的排序算法（基于全局数据）
if Config.SORT_METHOD == 'peak':
    # 原始方法：按峰值时间排序
    # 对于每个神经元，找到其信号达到最大值的时间戳（基于全局数据）
    peak_times = neural_data_full_standardized.idxmax()
    
    # 将神经元按照峰值时间从早到晚排序
    sorted_neurons = peak_times.sort_values().index
    
    sort_method_str = "Sorted by peak time (global)"
elif Config.SORT_METHOD == 'custom':
    # 自定义方法：按指定的神经元顺序排序
    sorted_neurons = sort_neurons_by_custom_order(neural_data_full_standardized.columns, Config.CUSTOM_NEURON_ORDER)
    
    sort_method_str = "Sorted by custom order"
    print(f"使用自定义神经元排序")
    print(f"指定顺序: {Config.CUSTOM_NEURON_ORDER}")
    print("剩余神经元将按字符串大小顺序排列在指定神经元下方")
else:  # 'calcium_wave'
    # 新方法：按第一次真实钙波发生时间排序（基于全局数据）
    first_wave_times = {}
    
    # 对每个神经元进行钙波检测
    for neuron in neural_data_full_standardized.columns:
        neuron_data = neural_data_full_standardized[neuron]
        first_wave_times[neuron] = detect_first_calcium_wave(neuron_data)
    
    # 转换为Series以便排序
    first_wave_times_series = pd.Series(first_wave_times)
    
    # 按第一次钙波时间排序
    sorted_neurons = first_wave_times_series.sort_values().index
    
    sort_method_str = "Sorted by first calcium wave time (global)"

print(f"神经元排序方式: {sort_method_str}")
print(f"神经元排序基于全局数据，确保不同时间区间的热图具有一致的纵坐标排序")

# 现在根据配置的时间戳区间筛选数据
if Config.STAMP_MIN is not None or Config.STAMP_MAX is not None:
    # 确定实际的最小值和最大值
    min_stamp = Config.STAMP_MIN if Config.STAMP_MIN is not None else neural_data_full.index.min()
    max_stamp = Config.STAMP_MAX if Config.STAMP_MAX is not None else neural_data_full.index.max()
    
    # 筛选神经元数据，保留指定区间内的数据
    neural_data_interval = neural_data_full.loc[min_stamp:max_stamp]
    
    # 如果有行为数据，也进行筛选
    if has_behavior:
        behavior_data_interval = behavior_data_full.loc[min_stamp:max_stamp]
        frame_lost = behavior_data_interval
    
    print(f"已筛选时间戳区间: {min_stamp} 到 {max_stamp}")
    # 对应的秒数区间
    min_seconds = min_stamp / Config.SAMPLING_RATE
    max_seconds = max_stamp / Config.SAMPLING_RATE
    print(f"对应的时间区间: {min_seconds:.2f}s 到 {max_seconds:.2f}s")
else:
    # 如果没有指定时间区间，使用全部数据
    neural_data_interval = neural_data_full
    if has_behavior:
        frame_lost = behavior_data_full

# 对筛选后的区间数据进行标准化
neural_data_interval_standardized = (neural_data_interval - neural_data_interval.mean()) / neural_data_interval.std()

# 根据全局排序后的神经元顺序重新排列筛选后数据的列
sorted_day6_data = neural_data_interval_standardized[sorted_neurons]

# **步骤4：找到所有行为标签的区间**

# 初始化行为区间变量
behavior_intervals = {}
unique_behaviors = []
global_unique_behaviors = []  # 全局行为类型

# 只有当behavior列存在时才处理行为标签
if has_behavior:
    # 获取全局数据中所有不同的行为标签（用于创建一致的图例）
    global_unique_behaviors = behavior_data_full.dropna().unique()
    print(f"全局行为类型: {list(global_unique_behaviors)}")
    
    # 获取当前时间区间内的行为标签
    unique_behaviors = frame_lost.dropna().unique()
    print(f"当前时间区间内的行为类型: {list(unique_behaviors)}")
    
    # 初始化当前时间区间内行为的区间字典
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

def extract_behavior_intervals_precise(behavior_series: pd.Series):
    intervals = {}
    current_behavior = None
    start_stamp = None
    prev_stamp = None

    for stamp, behavior in behavior_series.items():
        if pd.isna(behavior):
            if current_behavior is not None and start_stamp is not None and prev_stamp is not None:
                intervals.setdefault(current_behavior, []).append((start_stamp, prev_stamp))
            current_behavior = None
            start_stamp = None
            prev_stamp = stamp
            continue

        if behavior != current_behavior:
            if current_behavior is not None and start_stamp is not None and prev_stamp is not None:
                intervals.setdefault(current_behavior, []).append((start_stamp, prev_stamp))
            current_behavior = behavior
            start_stamp = stamp

        prev_stamp = stamp

    if current_behavior is not None and start_stamp is not None and prev_stamp is not None:
        intervals.setdefault(current_behavior, []).append((start_stamp, prev_stamp))

    return intervals


def build_concatenated_matrix(time_by_neuron_df: pd.DataFrame, intervals, gap_stamps: int):
    pieces = []
    mapped_stamps = []

    for start_stamp, end_stamp in intervals:
        seg = time_by_neuron_df.loc[start_stamp:end_stamp]
        if seg.empty:
            continue

        pieces.append(seg)
        mapped_stamps.append(seg.index.to_numpy())

        if gap_stamps > 0:
            pieces.append(pd.DataFrame(np.nan, index=np.arange(gap_stamps), columns=time_by_neuron_df.columns))
            mapped_stamps.append(np.full(gap_stamps, np.nan))

    if pieces and pieces[-1].isna().all(axis=None):
        pieces = pieces[:-1]
        if mapped_stamps:
            mapped_stamps = mapped_stamps[:-1]

    if not pieces:
        return pd.DataFrame(columns=time_by_neuron_df.columns), np.array([])

    concat_df = pd.concat(pieces, ignore_index=True)
    mapped_stamps_arr = np.concatenate(mapped_stamps) if mapped_stamps else np.array([])
    return concat_df, mapped_stamps_arr


def safe_filename(text: str):
    return ''.join(c if (c.isalnum() or c in ['-', '_', '.']) else '_' for c in str(text)).strip('_')


def plot_heatmap(time_by_neuron_df: pd.DataFrame, title_text: str, mapped_stamps=None, vmin=-2, vmax=2):
    import matplotlib.ticker as ticker
    from matplotlib.ticker import FuncFormatter

    fig = plt.figure(figsize=(80, 30))
    ax_heatmap = fig.add_subplot(111)

    cmap = plt.get_cmap('viridis').copy()
    cmap.set_bad(color='white')

    mask = time_by_neuron_df.T.isna()
    sns.heatmap(time_by_neuron_df.T, cmap=cmap, cbar=False, vmin=vmin, vmax=vmax, ax=ax_heatmap, mask=mask)

    n = time_by_neuron_df.shape[0]
    ax_heatmap.set_xlim(-0.5, n - 0.5)

    ax_heatmap.set_xlabel('Time (s)', fontsize=40)
    ax_heatmap.set_ylabel('neuron', fontsize=40)
    ax_heatmap.set_yticklabels(ax_heatmap.get_yticklabels(), fontsize=23, fontweight='bold', rotation=0)

    ax_heatmap.set_title(title_text, fontsize=40, pad=20, fontweight='bold')

    if mapped_stamps is not None and len(mapped_stamps) == n:
        mapped_seconds = mapped_stamps / Config.SAMPLING_RATE

        def _fmt(x, pos):
            if n <= 0:
                return ''
            idx = int(np.clip(np.round(x), 0, n - 1))
            v = mapped_seconds[idx]
            if np.isnan(v):
                return ''
            return f'{v:.1f}'

        ax_heatmap.xaxis.set_major_locator(ticker.MaxNLocator(nbins=12, prune=None))
        ax_heatmap.xaxis.set_major_formatter(FuncFormatter(_fmt))
        ax_heatmap.tick_params(axis='x', labelsize=30, rotation=45)
        for label in ax_heatmap.get_xticklabels():
            label.set_fontweight('bold')
    else:
        ax_heatmap.tick_params(axis='x', labelsize=30, rotation=45)
        for label in ax_heatmap.get_xticklabels():
            label.set_fontweight('bold')

    fig.canvas.draw()
    return fig, ax_heatmap


vmin, vmax = -2, 2
gap_stamps = 3

import os
input_filename = os.path.basename(Config.INPUT_FILE)
input_filename = os.path.splitext(input_filename)[0]

stamp_info = ''
if Config.STAMP_MIN is not None or Config.STAMP_MAX is not None:
    min_stamp = Config.STAMP_MIN if Config.STAMP_MIN is not None else neural_data_full.index.min()
    max_stamp = Config.STAMP_MAX if Config.STAMP_MAX is not None else neural_data_full.index.max()
    min_seconds = min_stamp / Config.SAMPLING_RATE
    max_seconds = max_stamp / Config.SAMPLING_RATE
    stamp_info = f'_{min_seconds:.2f}s_{max_seconds:.2f}s'

output_dir = os.path.dirname(Config.OUTPUT_PREFIX)
if output_dir and not os.path.exists(output_dir):
    os.makedirs(output_dir)

if has_behavior and 'behavior_data_interval' in globals() and behavior_data_interval.dropna().unique().size > 0:
    intervals_by_behavior = extract_behavior_intervals_precise(behavior_data_interval)
    for behavior_label, intervals in intervals_by_behavior.items():
        concat_data, mapped_stamps = build_concatenated_matrix(sorted_day6_data, intervals, gap_stamps=gap_stamps)
        if concat_data.empty:
            continue

        title_text = f'{behavior_label} | {sort_method_str} | gap={gap_stamps} stamps'
        fig, _ = plot_heatmap(concat_data, title_text=title_text, mapped_stamps=mapped_stamps, vmin=vmin, vmax=vmax)

        behavior_slug = safe_filename(behavior_label)
        output_filename = f"{Config.OUTPUT_PREFIX}{input_filename}_{Config.SORT_METHOD}_{behavior_slug}{stamp_info}.png"
        print(f"正在保存图像到 {output_filename}")
        fig.savefig(output_filename, bbox_inches='tight', pad_inches=0.1, dpi=100)
        plt.close(fig)
else:
    title_text = f'All | {sort_method_str}'
    fig, _ = plot_heatmap(sorted_day6_data.reset_index(drop=True), title_text=title_text, mapped_stamps=sorted_day6_data.index.to_numpy(), vmin=vmin, vmax=vmax)
    output_filename = f"{Config.OUTPUT_PREFIX}{input_filename}_{Config.SORT_METHOD}{stamp_info}.png"
    print(f"正在保存图像到 {output_filename}")
    fig.savefig(output_filename, bbox_inches='tight', pad_inches=0.1, dpi=100)
    plt.close(fig)

# 输出保存信息
print(f"热图已保存至: {output_filename}")
print("程序执行完成")
