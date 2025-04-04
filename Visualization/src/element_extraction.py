import numpy as np
import pandas as pd
from scipy import signal
from scipy.signal import find_peaks, peak_widths
from numpy import trapezoid
import matplotlib.pyplot as plt
import os
import argparse

def detect_calcium_transients(data, fs=1.0, min_snr=8.0, min_duration=20, smooth_window=50, 
                             peak_distance=30, baseline_percentile=20, max_duration=350,
                             smooth_method='savgol'):
    """
    检测钙离子浓度数据中的钙爆发(calcium transients)
    
    参数
    ----------
    data : numpy.ndarray
        钙离子浓度时间序列数据
    fs : float, 可选
        采样频率，默认为1.0Hz
    min_snr : float, 可选
        最小信噪比阈值，默认为3.0
    min_duration : int, 可选
        最小持续时间（采样点数），默认为3
    smooth_window : int, 可选
        平滑窗口大小，默认为5
    peak_distance : int, 可选
        峰值间最小距离，默认为5
    baseline_percentile : int, 可选
        用于估计基线的百分位数，默认为20
    max_duration : int, 可选
        钙爆发最大持续时间（采样点数），默认为200
    smooth_method : str, 可选
        平滑方法，可选值：'savgol'（默认，Savitzky-Golay滤波器）, 'sma'（简单移动平均）, 
        'ema'（指数移动平均）, 'none'（不进行平滑）
        
    返回
    -------
    transients : list of dict
        每个钙爆发的特征参数字典列表
    smoothed_data : numpy.ndarray
        平滑后的数据
    """
    # 原始数据备份
    raw_data = data.copy()
    
    # 1. 应用平滑处理
    if smooth_method == 'none' or smooth_window <= 1:
        # 不进行平滑处理
        smoothed_data = raw_data.copy()
        print("未应用平滑处理")
    elif smooth_method == 'savgol':
        # 使用Savitzky-Golay滤波器平滑数据
        # 确保窗口大小是奇数
        if smooth_window % 2 == 0:
            smooth_window += 1
        smoothed_data = signal.savgol_filter(raw_data, smooth_window, 3)
        print(f"应用Savitzky-Golay滤波器平滑处理（窗口大小={smooth_window}）")
    elif smooth_method == 'sma':
        # 使用简单移动平均(SMA)平滑数据
        # pandas rolling方法需要Series类型输入
        series = pd.Series(raw_data)
        # center=True确保窗口以当前点为中心
        smoothed_series = series.rolling(window=smooth_window, center=True).mean()
        # 填充NaN值（窗口两端）
        smoothed_series = smoothed_series.fillna(method='bfill').fillna(method='ffill')
        smoothed_data = smoothed_series.values
        print(f"应用简单移动平均(SMA)平滑处理（窗口大小={smooth_window}）")
    elif smooth_method == 'ema':
        # 使用指数移动平均(EMA)平滑数据
        series = pd.Series(raw_data)
        # 计算α值（平滑因子）
        alpha = 2 / (smooth_window + 1)
        # 应用EMA
        smoothed_series = series.ewm(alpha=alpha, adjust=False).mean()
        smoothed_data = smoothed_series.values
        print(f"应用指数移动平均(EMA)平滑处理（窗口大小={smooth_window}，α={alpha:.4f}）")
    else:
        raise ValueError(f"不支持的平滑方法: {smooth_method}，支持的方法有：'savgol', 'sma', 'ema', 'none'")
    
    # 2. 估计基线和噪声水平
    # 使用平滑后的数据估计基线
    baseline = np.percentile(smoothed_data, baseline_percentile)
    
    # 对于噪声水平估计, 使用平滑后数据低于中位数的部分
    # 这样可以避免将信号成分视为噪声
    noise_data = smoothed_data[smoothed_data < np.median(smoothed_data)]
    if len(noise_data) > 0:
        noise_level = np.std(noise_data)
    else:
        # 如果没有数据点小于中位数，使用较小的噪声估计
        noise_level = np.std(smoothed_data) * 0.1
    
    # 如果噪声水平过低（可能是因为过度平滑），设置一个合理的最小值
    min_noise_threshold = np.std(raw_data) * 0.001
    if noise_level < min_noise_threshold:
        noise_level = min_noise_threshold
    
    # 计算信噪比
    signal_range = np.max(smoothed_data) - baseline
    actual_snr = signal_range / noise_level
    print(f"数据基线: {baseline:.4f}, 噪声水平: {noise_level:.4f}, 估计信噪比: {actual_snr:.1f}")
    
    # 3. 检测峰值
    # 调整检测阈值 - 重要: 不同平滑方法会改变噪声特性，因此需要调整
    threshold = baseline + min_snr * noise_level
    
    # 根据平滑方法进行调整
    if smooth_method == 'savgol':
        # Savgol可能会增强峰值
        peaks, peak_props = find_peaks(smoothed_data, height=threshold, distance=peak_distance)
    elif smooth_method == 'sma':
        # SMA可能会降低峰值，小幅降低阈值
        adjusted_threshold = baseline + min_snr * noise_level * 0.9
        peaks, peak_props = find_peaks(smoothed_data, height=adjusted_threshold, distance=peak_distance)
    elif smooth_method == 'ema':
        # EMA对最近数据敏感
        peaks, peak_props = find_peaks(smoothed_data, height=threshold, distance=peak_distance)
    else:
        # 不平滑时需要更高阈值以过滤噪声峰
        adjusted_threshold = baseline + min_snr * noise_level * 1.2
        peaks, peak_props = find_peaks(smoothed_data, height=adjusted_threshold, distance=peak_distance)
    
    print(f"检测阈值: {threshold:.4f}, 检测到 {len(peaks)} 个峰值")
    
    # 如果没有检测到峰值，返回空列表
    if len(peaks) == 0:
        return [], smoothed_data
    
    # 4. 分析每个钙爆发
    transients = []
    for i, peak_idx in enumerate(peaks):
        # 寻找左侧边界（从峰值向左搜索）
        start_idx = peak_idx
        # 向左搜索到信号低于基线或达到最大距离或达到前一个峰值的右边界
        left_limit = 0 if i == 0 else peaks[i-1]
        while start_idx > left_limit and smoothed_data[start_idx] > baseline:
            start_idx -= 1
            # 如果搜索范围过大，在局部最小值处停止
            if peak_idx - start_idx > max_duration:
                # 找到从peak_idx向左max_duration点范围内的局部最小值
                local_min_idx = start_idx + np.argmin(smoothed_data[start_idx:start_idx+max_duration])
                start_idx = local_min_idx
                break
        
        # 寻找右侧边界（从峰值向右搜索）
        end_idx = peak_idx
        # 向右搜索到信号低于基线或达到最大距离或达到下一个峰值的左边界
        right_limit = len(smoothed_data) - 1 if i == len(peaks) - 1 else peaks[i+1]
        while end_idx < right_limit and smoothed_data[end_idx] > baseline:
            end_idx += 1
            # 如果搜索范围过大，在局部最小值处停止
            if end_idx - peak_idx > max_duration:
                # 找到从peak_idx向右max_duration点范围内的局部最小值
                search_end = min(end_idx + max_duration, len(smoothed_data))
                if peak_idx < search_end - 1:
                    local_min_idx = peak_idx + np.argmin(smoothed_data[peak_idx:search_end])
                    end_idx = local_min_idx
                break
        
        # 如果峰值之间的信号始终高于基线，则使用峰值之间的最低点作为分界
        if i < len(peaks) - 1 and end_idx >= peaks[i+1]:
            # 寻找两个峰值之间的最低点作为分界
            valley_idx = peak_idx + np.argmin(smoothed_data[peak_idx:peaks[i+1]])
            end_idx = valley_idx
        
        if i > 0 and start_idx <= peaks[i-1]:
            # 寻找两个峰值之间的最低点作为分界
            valley_idx = peaks[i-1] + np.argmin(smoothed_data[peaks[i-1]:peak_idx])
            start_idx = valley_idx
            
        # 计算持续时间
        duration = (end_idx - start_idx) / fs
        
        # 如果持续时间太短，跳过此峰值
        if (end_idx - start_idx) < min_duration:
            continue
            
        # 计算特征
        peak_value = smoothed_data[peak_idx]
        amplitude = peak_value - baseline
        
        # 计算半高宽 (FWHM)
        half_max = baseline + amplitude / 2
        widths, width_heights, left_ips, right_ips = peak_widths(smoothed_data, [peak_idx], rel_height=0.5)
        fwhm = widths[0] / fs
        
        # 上升和衰减时间
        rise_time = (peak_idx - start_idx) / fs
        decay_time = (end_idx - peak_idx) / fs
        
        # 计算峰面积 (AUC)
        segment = smoothed_data[start_idx:end_idx+1] - baseline
        auc = trapezoid(segment, dx=1.0/fs)
        
        # 存储此次钙爆发的特征
        transient = {
            'start_idx': start_idx,
            'peak_idx': peak_idx,
            'end_idx': end_idx,
            'amplitude': amplitude,
            'peak_value': peak_value,
            'baseline': baseline,
            'duration': duration,
            'fwhm': fwhm,
            'rise_time': rise_time,
            'decay_time': decay_time,
            'auc': auc,
            'snr': amplitude / noise_level
        }
        
        transients.append(transient)
    
    print(f"过滤后保留 {len(transients)} 个有效钙爆发")
    return transients, smoothed_data

def extract_calcium_features(neuron_data, fs=1.0, visualize=False, smooth_method='savgol', smooth_window=50):
    """
    从钙离子浓度数据中提取关键特征
    
    参数
    ----------
    neuron_data : numpy.ndarray 或 pandas.Series
        神经元钙离子浓度时间序列数据
    fs : float, 可选
        采样频率，默认为1.0Hz
    visualize : bool, 可选
        是否可视化结果，默认为False
    smooth_method : str, 可选
        平滑方法，可选值：'savgol'（默认，Savitzky-Golay滤波器）, 'sma'（简单移动平均）, 
        'ema'（指数移动平均）, 'none'（不进行平滑）
    smooth_window : int, 可选
        平滑窗口大小，默认为50
        
    返回
    -------
    features : dict
        计算得到的特征统计数据
    transients : list of dict
        每个钙爆发的特征参数字典列表
    """
    if isinstance(neuron_data, pd.Series):
        data = neuron_data.values
    else:
        data = neuron_data
    
    # 检测钙爆发
    transients, smoothed_data = detect_calcium_transients(data, fs=fs, 
                                                         smooth_method=smooth_method,
                                                         smooth_window=smooth_window)
    
    # 如果没有检测到钙爆发，返回空特征
    if len(transients) == 0:
        return {
            'num_transients': 0,
            'mean_amplitude': np.nan,
            'mean_duration': np.nan,
            'mean_fwhm': np.nan,
            'mean_rise_time': np.nan,
            'mean_decay_time': np.nan,
            'mean_auc': np.nan,
            'frequency': 0
        }, []
    
    # 计算特征统计值
    amplitudes = [t['amplitude'] for t in transients]
    durations = [t['duration'] for t in transients]
    fwhms = [t['fwhm'] for t in transients]
    rise_times = [t['rise_time'] for t in transients]
    decay_times = [t['decay_time'] for t in transients]
    aucs = [t['auc'] for t in transients]
    
    # 记录总体特征
    total_time = len(data) / fs  # 总时间（秒）
    features = {
        'num_transients': len(transients),
        'mean_amplitude': np.mean(amplitudes),
        'mean_duration': np.mean(durations),
        'mean_fwhm': np.mean(fwhms),
        'mean_rise_time': np.mean(rise_times),
        'mean_decay_time': np.mean(decay_times),
        'mean_auc': np.mean(aucs),
        'frequency': len(transients) / total_time  # 每秒事件数
    }
    
    # 可视化（如果需要）
    if visualize:
        visualize_calcium_transients(data, smoothed_data, transients, fs)
    
    return features, transients

def visualize_calcium_transients(raw_data, smoothed_data, transients, fs=1.0):
    """
    可视化钙离子浓度数据和检测到的钙爆发
    
    参数
    ----------
    raw_data : numpy.ndarray
        原始钙离子浓度数据
    smoothed_data : numpy.ndarray
        平滑后的数据
    transients : list of dict
        检测到的钙爆发特征列表
    fs : float, 可选
        采样频率，默认为1.0Hz
    """
    time = np.arange(len(raw_data)) / fs
    plt.figure(figsize=(12, 6))
    plt.plot(time, raw_data, 'k-', alpha=0.4, label='Raw data')
    plt.plot(time, smoothed_data, 'b-', label='Smoothed data')
    
    # 标记钙爆发
    for i, t in enumerate(transients):
        # 标记峰值
        plt.plot(t['peak_idx']/fs, t['peak_value'], 'ro', markersize=8)
        
        # 标记开始和结束
        plt.axvline(x=t['start_idx']/fs, color='g', linestyle='--', alpha=0.5)
        plt.axvline(x=t['end_idx']/fs, color='r', linestyle='--', alpha=0.5)
        
        # 标记半高宽
        half_max = t['baseline'] + t['amplitude'] / 2
        plt.plot([t['start_idx']/fs, t['end_idx']/fs], [half_max, half_max], 'y-', linewidth=2)
        
        # 添加编号
        plt.text(t['peak_idx']/fs, t['peak_value']*1.05, f"{i+1}", 
                 horizontalalignment='center', verticalalignment='bottom')
    
    plt.xlabel('Time (seconds)')
    plt.ylabel('Calcium Signal Intensity')
    plt.title(f'Detected {len(transients)} calcium transient events')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

def process_multiple_neurons(data_df, neuron_columns, fs=1.0, smooth_method='savgol', smooth_window=50):
    """
    处理多个神经元的钙离子数据并提取特征
    
    参数
    ----------
    data_df : pandas.DataFrame
        包含多个神经元数据的DataFrame
    neuron_columns : list of str
        要处理的神经元列名列表
    fs : float, 可选
        采样频率，默认为1.0Hz
    smooth_method : str, 可选
        平滑方法，可选值：'savgol'（默认）, 'sma', 'ema', 'none'
    smooth_window : int, 可选
        平滑窗口大小，默认为50
        
    返回
    -------
    results : pandas.DataFrame
        每个神经元的特征统计结果
    """
    results = []
    
    for neuron in neuron_columns:
        print(f"处理神经元 {neuron}...")
        features, _ = extract_calcium_features(data_df[neuron], fs=fs, 
                                             smooth_method=smooth_method,
                                             smooth_window=smooth_window)
        features['neuron'] = neuron
        features['smooth_method'] = smooth_method
        features['smooth_window'] = smooth_window
        results.append(features)
    
    return pd.DataFrame(results)

def analyze_behavior_specific_features(data_df, neuron_columns, behavior_col='behavior', fs=1.0,
                                      smooth_method='savgol', smooth_window=50):
    """
    分析不同行为条件下的钙离子特征
    
    参数
    ----------
    data_df : pandas.DataFrame
        包含神经元数据和行为标签的DataFrame
    neuron_columns : list of str
        要处理的神经元列名列表
    behavior_col : str, 可选
        行为标签列名，默认为'behavior'
    fs : float, 可选
        采样频率，默认为1.0Hz
    smooth_method : str, 可选
        平滑方法，可选值：'savgol'（默认）, 'sma', 'ema', 'none'
    smooth_window : int, 可选
        平滑窗口大小，默认为50
        
    返回
    -------
    results : pandas.DataFrame
        不同行为条件下每个神经元的特征统计结果
    """
    # 获取所有行为类型
    behaviors = data_df[behavior_col].unique()
    results = []
    
    for neuron in neuron_columns:
        for behavior in behaviors:
            # 获取特定行为下的数据
            behavior_data = data_df[data_df[behavior_col] == behavior][neuron]
            
            if len(behavior_data) > 0:
                print(f"分析神经元 {neuron} 在行为 '{behavior}' 下的特征...")
                features, _ = extract_calcium_features(behavior_data, fs=fs, 
                                                    smooth_method=smooth_method,
                                                    smooth_window=smooth_window)
                features['neuron'] = neuron
                features['behavior'] = behavior
                features['smooth_method'] = smooth_method
                features['smooth_window'] = smooth_window
                results.append(features)
    
    return pd.DataFrame(results)

def analyze_all_neurons_transients(data_df, neuron_columns, fs=1.0, save_path=None,
                               smooth_method='savgol', smooth_window=50):
    """
    分析所有神经元的钙爆发并为每个爆发分配唯一ID
    
    参数
    ----------
    data_df : pandas.DataFrame
        包含多个神经元数据的DataFrame
    neuron_columns : list of str
        要处理的神经元列名列表
    fs : float, 可选
        采样频率，默认为1.0Hz
    save_path : str, 可选
        Excel文件保存路径，默认为None（不保存）
    smooth_method : str, 可选
        平滑方法，可选值：'savgol'（默认）, 'sma', 'ema', 'none'
    smooth_window : int, 可选
        平滑窗口大小，默认为50
        
    返回
    -------
    all_transients_df : pandas.DataFrame
        包含所有神经元所有钙爆发特征的DataFrame
    """
    all_transients = []
    transient_id = 1  # 起始ID
    
    for neuron in neuron_columns:
        print(f"处理神经元 {neuron} 的钙爆发...")
        neuron_data = data_df[neuron].values
        
        # 检测钙爆发
        transients, smoothed_data = detect_calcium_transients(neuron_data, fs=fs,
                                                           smooth_method=smooth_method,
                                                           smooth_window=smooth_window)
        
        # 为该神经元的每个钙爆发分配ID并添加到列表
        for t in transients:
            t['neuron'] = neuron
            t['transient_id'] = transient_id
            t['smooth_method'] = smooth_method
            t['smooth_window'] = smooth_window
            all_transients.append(t)
            transient_id += 1
    
    # 如果没有检测到钙爆发，返回空DataFrame
    if len(all_transients) == 0:
        print("未检测到任何钙爆发")
        return pd.DataFrame()
    
    # 创建DataFrame
    all_transients_df = pd.DataFrame(all_transients)
    
    # 如果指定了保存路径，则保存到Excel
    if save_path:
        all_transients_df.to_excel(save_path, index=False)
        print(f"成功将所有钙爆发数据保存到: {save_path}")
    
    return all_transients_df

def compare_smoothing_methods(data_df, neuron_columns=None, fs=1.0):
    """
    比较不同平滑方法对钙爆发检测的影响
    
    参数
    ----------
    data_df : pandas.DataFrame
        包含多个神经元数据的DataFrame
    neuron_columns : list of str, optional
        要处理的神经元列名列表，默认为None（自动检测）
    fs : float, 可选
        采样频率，默认为1.0Hz
        
    返回
    -------
    comparison_df : pandas.DataFrame
        比较不同平滑方法对钙爆发检测影响的结果
    """
    if neuron_columns is None:
        # 自动检测神经元列
        neuron_columns = [col for col in data_df.columns if col.startswith('n') and col[1:].isdigit()]
    
    # 要比较的平滑方法和窗口大小
    smooth_methods = ['none', 'savgol', 'sma', 'ema']
    window_sizes = [5, 20, 50, 100]
    
    results = []
    
    for neuron in neuron_columns:
        print(f"\n分析神经元 {neuron} 的不同平滑方法效果:")
        neuron_data = data_df[neuron].values
        
        # 首先检测不平滑的情况作为基准
        print("未平滑处理:")
        base_transients, _ = detect_calcium_transients(neuron_data, fs=fs, smooth_method='none')
        base_count = len(base_transients)
        print(f"  检测到 {base_count} 个钙爆发")
        
        result_row = {
            'neuron': neuron,
            'none_count': base_count,
        }
        
        # 对每种平滑方法和窗口大小进行测试
        for method in ['savgol', 'sma', 'ema']:
            for window in window_sizes:
                if method == 'savgol' and window % 2 == 0:
                    # Savitzky-Golay 需要奇数窗口大小
                    window += 1
                
                print(f"{method.upper()} 平滑 (窗口={window}):")
                transients, _ = detect_calcium_transients(neuron_data, fs=fs, 
                                                     smooth_method=method,
                                                     smooth_window=window)
                count = len(transients)
                diff = count - base_count
                diff_percent = (diff / base_count * 100) if base_count > 0 else float('inf')
                
                print(f"  检测到 {count} 个钙爆发 (相比无平滑变化: {diff:+d}, {diff_percent:+.1f}%)")
                
                # 添加到结果中
                result_row[f'{method}_{window}_count'] = count
                result_row[f'{method}_{window}_diff'] = diff
                result_row[f'{method}_{window}_diff_percent'] = diff_percent
        
        results.append(result_row)
    
    # 创建比较结果DataFrame
    comparison_df = pd.DataFrame(results)
    
    # 输出汇总统计
    print("\n===== 总体平滑效果统计 =====")
    for method in ['savgol', 'sma', 'ema']:
        for window in window_sizes:
            if method == 'savgol' and window % 2 == 0:
                window += 1
            
            # 计算平均变化百分比
            diff_percent_col = f'{method}_{window}_diff_percent'
            if diff_percent_col in comparison_df.columns:
                mean_diff_percent = comparison_df[diff_percent_col].mean()
                print(f"{method.upper()} (窗口={window}): 平均变化 {mean_diff_percent:+.1f}%")
    
    return comparison_df

if __name__ == "__main__":
    """
    从Excel文件加载神经元数据并进行特征提取的示例
    """
    
    # 创建命令行参数解析器
    parser = argparse.ArgumentParser(description='神经元钙离子特征提取工具')
    parser.add_argument('--data', type=str, default='../datasets/Day6_with_behavior_labels_filled.xlsx',
                      help='数据文件路径')
    parser.add_argument('--output', type=str, default='../results/all_neurons_transients.xlsx',
                      help='结果保存路径')
    parser.add_argument('--smooth', type=str, default='savgol', 
                      choices=['savgol', 'sma', 'ema', 'none'],
                      help='平滑方法: savgol (Savitzky-Golay滤波器), sma (简单移动平均), ema (指数移动平均), none (不平滑)')
    parser.add_argument('--window', type=int, default=50,
                      help='平滑窗口大小')
    parser.add_argument('--compare', action='store_true',
                      help='比较不同平滑方法的效果')
    parser.add_argument('--neurons', type=str, default=None,
                      help='要处理的神经元列表，用逗号分隔（如 "n1,n5,n10"）。不指定则处理所有神经元')
    
    args = parser.parse_args()
    
    # 定义数据文件路径
    data_path = args.data
    
    # 检查文件是否存在
    if os.path.exists(data_path):
        try:
            # 从指定路径加载Excel数据
            print(f"正在从 {data_path} 加载数据...")
            df = pd.read_excel(data_path)
            # 清理列名（去除可能的空格）
            df.columns = [col.strip() for col in df.columns]
            print(f"成功加载数据，共 {len(df)} 行")
            
            # 提取神经元列
            if args.neurons:
                neuron_columns = [n.strip() for n in args.neurons.split(',')]
                print(f"将处理指定的 {len(neuron_columns)} 个神经元")
            else:
                neuron_columns = [col for col in df.columns if col.startswith('n') and col[1:].isdigit()]
                print(f"检测到 {len(neuron_columns)} 个神经元数据列")
            
            # 检测指定的神经元是否存在于数据中
            for neuron in neuron_columns:
                if neuron not in df.columns:
                    print(f"警告: 神经元 {neuron} 不在数据中！")
            
            # 过滤出存在的神经元
            neuron_columns = [n for n in neuron_columns if n in df.columns]
            
            if args.compare:
                # 比较不同平滑方法
                print("\n正在比较不同平滑方法的效果...")
                comparison_df = compare_smoothing_methods(df, neuron_columns)
                
                # 保存比较结果
                compare_path = os.path.join(os.path.dirname(args.output), 'smoothing_comparison.xlsx')
                comparison_df.to_excel(compare_path, index=False)
                print(f"平滑方法比较结果已保存至: {compare_path}")
            else:
                # 分析所有神经元的钙爆发并保存到Excel
                save_path = args.output
                
                # 确保保存目录存在
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                
                # 显示平滑方法信息
                print(f"使用 {args.smooth} 平滑方法，窗口大小 = {args.window}")
                
                # 分析并保存所有钙爆发数据
                all_transients = analyze_all_neurons_transients(df, neuron_columns, 
                                                             save_path=save_path,
                                                             smooth_method=args.smooth,
                                                             smooth_window=args.window)
                print(f"共检测到 {len(all_transients)} 个钙爆发")
            
        except Exception as e:
            import traceback
            print(f"加载或处理数据时出错:")
            traceback.print_exc()
    else:
        print(f"错误: 找不到数据文件 '{data_path}'，请检查文件路径")
