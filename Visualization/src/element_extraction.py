import numpy as np
import pandas as pd
from scipy import signal
from scipy.signal import find_peaks, peak_widths
from numpy import trapezoid
import matplotlib.pyplot as plt
import os
import argparse

def detect_calcium_transients(data, fs=1.0, min_snr=10.0, min_duration=20, smooth_window=50, 
                             peak_distance=30, baseline_percentile=20, max_duration=350):
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
        
    返回
    -------
    transients : list of dict
        每个钙爆发的特征参数字典列表
    smoothed_data : numpy.ndarray
        平滑后的数据
    """
    # 1. 应用平滑滤波器
    if smooth_window > 1:
        smoothed_data = signal.savgol_filter(data, smooth_window, 3)
    else:
        smoothed_data = data.copy()
    
    # 2. 估计基线和噪声水平
    baseline = np.percentile(smoothed_data, baseline_percentile)
    noise_level = np.std(smoothed_data[smoothed_data < np.percentile(smoothed_data, 50)])
    
    # 3. 检测峰值
    threshold = baseline + min_snr * noise_level
    peaks, peak_props = find_peaks(smoothed_data, height=threshold, distance=peak_distance)
    
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
    
    return transients, smoothed_data

def extract_calcium_features(neuron_data, fs=1.0, visualize=False):
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
    transients, smoothed_data = detect_calcium_transients(data, fs=fs)
    
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

def process_multiple_neurons(data_df, neuron_columns, fs=1.0):
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
        
    返回
    -------
    results : pandas.DataFrame
        每个神经元的特征统计结果
    """
    results = []
    
    for neuron in neuron_columns:
        print(f"处理神经元 {neuron}...")
        features, _ = extract_calcium_features(data_df[neuron], fs=fs)
        features['neuron'] = neuron
        results.append(features)
    
    return pd.DataFrame(results)

def analyze_behavior_specific_features(data_df, neuron_columns, behavior_col='behavior', fs=1.0):
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
                features, _ = extract_calcium_features(behavior_data, fs=fs)
                features['neuron'] = neuron
                features['behavior'] = behavior
                results.append(features)
    
    return pd.DataFrame(results)

def analyze_all_neurons_transients(data_df, neuron_columns, fs=1.0, save_path=None):
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
        transients, smoothed_data = detect_calcium_transients(neuron_data, fs=fs)
        
        # 为该神经元的每个钙爆发分配ID并添加到列表
        for t in transients:
            t['neuron'] = neuron
            t['transient_id'] = transient_id
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
        # 确保保存目录存在
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        all_transients_df.to_excel(save_path, index=False)
        print(f"成功将所有钙爆发数据保存到: {save_path}")
    
    return all_transients_df

if __name__ == "__main__":
    """
    从Excel文件加载神经元数据并进行特征提取的示例
    """
    
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='神经元钙离子特征提取工具')
    parser.add_argument('--data', type=str, default='../datasets/processed_Day6.xlsx',
                        help='数据文件路径，支持.xlsx格式')
    parser.add_argument('--output', type=str, default=None,
                        help='输出目录，不指定则根据数据集名称自动生成')
    args = parser.parse_args()
    
    # 检查文件是否存在
    if os.path.exists(args.data):
        try:
            # 从指定路径加载Excel数据
            print(f"正在从 {args.data} 加载数据...")
            df = pd.read_excel(args.data)
            # 清理列名（去除可能的空格）
            df.columns = [col.strip() for col in df.columns]
            print(f"成功加载数据，共 {len(df)} 行")
            
            # 提取神经元列
            neuron_columns = [col for col in df.columns if col.startswith('n') and col[1:].isdigit()]
            print(f"检测到 {len(neuron_columns)} 个神经元数据列")
            
            # 根据数据文件名生成输出目录
            if args.output is None:
                # 提取数据文件名（不含扩展名）
                data_basename = os.path.basename(args.data)
                dataset_name = os.path.splitext(data_basename)[0]
                output_dir = f"../results/{dataset_name}"
                save_path = f"{output_dir}/all_neurons_transients.xlsx"
            else:
                output_dir = args.output
                save_path = f"{output_dir}/all_neurons_transients.xlsx"
            
            print(f"输出目录设置为: {output_dir}")
            
            # 确保保存目录存在
            os.makedirs(output_dir, exist_ok=True)
            
            # 分析并保存所有钙爆发数据
            all_transients = analyze_all_neurons_transients(df, neuron_columns, save_path=save_path)
            print(f"共检测到 {len(all_transients)} 个钙爆发")
            
        except Exception as e:
            print(f"加载或处理数据时出错: {str(e)}")
    else:
        print(f"错误: 找不到数据文件 '{args.data}'，请检查文件路径")
