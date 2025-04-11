import numpy as np
import pandas as pd
from scipy import signal
from scipy.signal import find_peaks, peak_widths, savgol_filter
from numpy import trapezoid
import matplotlib.pyplot as plt
import os
import argparse
import glob

def detect_calcium_transients(data, fs=1.0, min_snr=8.0, min_duration=20, smooth_window=50, 
                             peak_distance=30, baseline_percentile=20, max_duration=350,
                             detect_subpeaks=True, subpeak_prominence=0.25, 
                             subpeak_width=10, subpeak_distance=15, params=None):
    """
    检测钙离子浓度数据中的钙爆发(calcium transients)，包括大波中的小波动
    
    参数
    ----------
    data : numpy.ndarray
        钙离子浓度时间序列数据
    fs : float, 可选
        采样频率，默认为1.0Hz
    min_snr : float, 可选
        最小信噪比阈值，默认为8.0
    min_duration : int, 可选
        最小持续时间（采样点数），默认为20
    smooth_window : int, 可选
        平滑窗口大小，默认为50
    peak_distance : int, 可选
        峰值间最小距离，默认为30
    baseline_percentile : int, 可选
        用于估计基线的百分位数，默认为20
    max_duration : int, 可选
        钙爆发最大持续时间（采样点数），默认为350
    detect_subpeaks : bool, 可选
        是否检测大波中的小波峰，默认为True
    subpeak_prominence : float, 可选
        子峰的相对突出度（主峰振幅的比例），默认为0.25
    subpeak_width : int, 可选
        子峰的最小宽度，默认为10
    subpeak_distance : int, 可选
        子峰间的最小距离，默认为15
        
    返回
    -------
    transients : list of dict
        每个钙爆发的特征参数字典列表
    smoothed_data : numpy.ndarray
        平滑后的数据
    """
    # 如果提供了自定义参数，覆盖默认参数
    if params is not None:
        min_snr = params.get('min_snr', min_snr)
        min_duration = params.get('min_duration', min_duration)
        smooth_window = params.get('smooth_window', smooth_window)
        peak_distance = params.get('peak_distance', peak_distance)
        baseline_percentile = params.get('baseline_percentile', baseline_percentile)
        max_duration = params.get('max_duration', max_duration)
        subpeak_prominence = params.get('subpeak_prominence', subpeak_prominence)
        subpeak_width = params.get('subpeak_width', subpeak_width)
        subpeak_distance = params.get('subpeak_distance', subpeak_distance)
    
    # 1. 应用平滑滤波器
    if smooth_window > 1:
        # 确保smooth_window是奇数
        if smooth_window % 2 == 0:
            smooth_window += 1
        smoothed_data = signal.savgol_filter(data, smooth_window, 3)
    else:
        smoothed_data = data.copy()
    
    # 2. 估计基线和噪声水平
    baseline = np.percentile(smoothed_data, baseline_percentile)
    noise_level = np.std(smoothed_data[smoothed_data < np.percentile(smoothed_data, 50)])
    
    # 3. 检测主要峰值
    threshold = baseline + min_snr * noise_level
    peaks, peak_props = find_peaks(smoothed_data, height=threshold, distance=peak_distance)
    
    # 如果没有检测到峰值，返回空列表
    if len(peaks) == 0:
        return [], smoothed_data
    
    # 4. 分析每个钙爆发
    transients = []
    
    # 第一遍检测：主要钙爆发
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
        
        # 检测波形的子峰值（如果启用）
        subpeaks = []
        if detect_subpeaks and (end_idx - start_idx) > 3 * subpeak_width:
            # 计算当前波形区间
            wave_segment = smoothed_data[start_idx:end_idx+1]
            
            # 计算相对突出度阈值（基于主峰的振幅）
            abs_prominence = subpeak_prominence * amplitude
            
            # 在此波形内找到所有局部峰值
            sub_peaks, sub_properties = find_peaks(
                wave_segment,
                prominence=abs_prominence,
                width=subpeak_width,
                distance=subpeak_distance
            )
            
            # 转换为原始数据索引
            sub_peaks = sub_peaks + start_idx
            
            # 排除与主峰相同的峰
            sub_peaks = [sp for sp in sub_peaks if abs(sp - peak_idx) > subpeak_distance]
            
            # 记录子峰特征
            for sp_idx in sub_peaks:
                # 计算子峰特征
                sp_value = smoothed_data[sp_idx]
                sp_amplitude = sp_value - baseline
                
                # 子峰的半高宽特性
                try:
                    sp_widths, _, sp_left_ips, sp_right_ips = peak_widths(
                        smoothed_data, [sp_idx], rel_height=0.5
                    )
                    sp_fwhm = sp_widths[0] / fs
                    
                    # 找出子峰的边界（局部最小值点）
                    # 向左寻找局部最小值
                    sp_start = sp_idx
                    left_search_limit = max(start_idx, sp_idx - max_duration//2)
                    while sp_start > left_search_limit:
                        if sp_start == left_search_limit + 1 or smoothed_data[sp_start] <= smoothed_data[sp_start-1]:
                            break
                        sp_start -= 1
                    
                    # 向右寻找局部最小值
                    sp_end = sp_idx
                    right_search_limit = min(end_idx, sp_idx + max_duration//2)
                    while sp_end < right_search_limit:
                        if sp_end == right_search_limit - 1 or smoothed_data[sp_end] <= smoothed_data[sp_end+1]:
                            break
                        sp_end += 1
                    
                    # 子峰持续时间
                    sp_duration = (sp_end - sp_start) / fs
                    
                    # 上升和衰减时间
                    sp_rise_time = (sp_idx - sp_start) / fs
                    sp_decay_time = (sp_end - sp_idx) / fs
                    
                    # 计算子峰面积
                    sp_segment = smoothed_data[sp_start:sp_end+1] - baseline
                    sp_auc = trapezoid(sp_segment, dx=1.0/fs)
                    
                    # 添加子峰信息
                    subpeaks.append({
                        'index': sp_idx,
                        'value': sp_value,
                        'amplitude': sp_amplitude,
                        'start_idx': sp_start,
                        'end_idx': sp_end,
                        'duration': sp_duration,
                        'fwhm': sp_fwhm,
                        'rise_time': sp_rise_time,
                        'decay_time': sp_decay_time,
                        'auc': sp_auc
                    })
                except Exception as e:
                    # 子峰分析失败，跳过该子峰
                    pass
        
        # 收集主波形对象特征
        wave_type = "complex" if len(subpeaks) > 0 else "simple"
        subpeaks_count = len(subpeaks)
        
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
            'snr': amplitude / noise_level,
            'wave_type': wave_type,
            'subpeaks_count': subpeaks_count,
            'subpeaks': subpeaks
        }
        
        transients.append(transient)
    
    # 检测是否有复杂波形或组合波形（不同特征的波）
    wave_types = {t['wave_type'] for t in transients}
    complex_waves = [t for t in transients if t['wave_type'] == 'complex']
    
    # 输出波形分类统计
    if len(transients) > 0:
        print(f"总共检测到 {len(transients)} 个钙爆发，其中：")
        print(f"  - 简单波形: {len(transients) - len(complex_waves)} 个")
        print(f"  - 复合波形: {len(complex_waves)} 个 (含有子峰)")
        
        # 计算子峰总数
        total_subpeaks = sum(t['subpeaks_count'] for t in transients)
        if total_subpeaks > 0:
            print(f"  - 子峰总数: {total_subpeaks} 个")
    
    return transients, smoothed_data

def extract_calcium_features(neuron_data, fs=1.0, visualize=False, detect_subpeaks=True, params=None):
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
    detect_subpeaks : bool, 可选
        是否检测大波中的小波峰，默认为True
        
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
    transients, smoothed_data = detect_calcium_transients(data, fs=fs, detect_subpeaks=detect_subpeaks, params=params)
    
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
            'frequency': 0,
            'complex_waves_ratio': 0,
            'subpeaks_per_wave': 0
        }, []
    
    # 计算特征统计值
    amplitudes = [t['amplitude'] for t in transients]
    durations = [t['duration'] for t in transients]
    fwhms = [t['fwhm'] for t in transients]
    rise_times = [t['rise_time'] for t in transients]
    decay_times = [t['decay_time'] for t in transients]
    aucs = [t['auc'] for t in transients]
    
    # 统计复杂波形比例
    complex_waves = [t for t in transients if t['wave_type'] == 'complex']
    complex_waves_ratio = len(complex_waves) / len(transients) if len(transients) > 0 else 0
    
    # 计算每个波的平均子峰数
    subpeaks_per_wave = sum(t['subpeaks_count'] for t in transients) / len(transients) if len(transients) > 0 else 0
    
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
        'frequency': len(transients) / total_time,  # 每秒事件数
        'complex_waves_ratio': complex_waves_ratio,
        'subpeaks_per_wave': subpeaks_per_wave
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
    plt.figure(figsize=(14, 8))
    
    # 创建两个子图：原始数据和放大的波峰
    ax1 = plt.subplot(2, 1, 1)
    ax1.plot(time, raw_data, 'k-', alpha=0.4, label='Raw data')
    ax1.plot(time, smoothed_data, 'b-', label='Smoothed data')
    
    # 标记钙爆发
    for i, t in enumerate(transients):
        # 使用不同的颜色标记简单波和复杂波
        color = 'green' if t['wave_type'] == 'simple' else 'red'
        marker_size = 8 if t['wave_type'] == 'simple' else 10
        
        # 标记主峰值
        ax1.plot(t['peak_idx']/fs, t['peak_value'], 'o', color=color, markersize=marker_size)
        
        # 标记开始和结束
        ax1.axvline(x=t['start_idx']/fs, color='g', linestyle='--', alpha=0.5)
        ax1.axvline(x=t['end_idx']/fs, color='r', linestyle='--', alpha=0.5)
        
        # 添加编号
        ax1.text(t['peak_idx']/fs, t['peak_value']*1.05, f"{i+1}", 
                 horizontalalignment='center', verticalalignment='bottom')
        
        # 标记子峰（如果有）
        if 'subpeaks' in t and t['subpeaks']:
            for sp in t['subpeaks']:
                ax1.plot(sp['index']/fs, sp['value'], '*', color='magenta', markersize=8)
    
    ax1.set_xlabel('Time (seconds)')
    ax1.set_ylabel('Calcium Signal Intensity')
    ax1.set_title(f'Detected {len(transients)} calcium transient events')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 选择一个复杂波形进行放大显示
    complex_waves = [t for t in transients if t['wave_type'] == 'complex']
    if complex_waves:
        ax2 = plt.subplot(2, 1, 2)
        
        # 选择子峰数最多的复杂波
        t = max(complex_waves, key=lambda x: x['subpeaks_count'])
        
        # 计算放大区域
        margin = 20  # 左右额外显示的点数
        zoom_start = max(0, t['start_idx'] - margin)
        zoom_end = min(len(raw_data), t['end_idx'] + margin)
        
        # 绘制放大区域
        zoom_time = time[zoom_start:zoom_end]
        ax2.plot(zoom_time, raw_data[zoom_start:zoom_end], 'k-', alpha=0.4, label='Raw data')
        ax2.plot(zoom_time, smoothed_data[zoom_start:zoom_end], 'b-', label='Smoothed data')
        
        # 标记主峰
        ax2.plot(t['peak_idx']/fs, t['peak_value'], 'o', color='red', markersize=10, label='Main peak')
        
        # 标记子峰
        for sp in t['subpeaks']:
            ax2.plot(sp['index']/fs, sp['value'], '*', color='magenta', markersize=8)
            # 标记子峰边界
            ax2.axvline(x=sp['start_idx']/fs, color='c', linestyle=':', alpha=0.7)
            ax2.axvline(x=sp['end_idx']/fs, color='m', linestyle=':', alpha=0.7)
        
        ax2.set_xlabel('Time (seconds)')
        ax2.set_ylabel('Calcium Signal Intensity')
        ax2.set_title(f'Complex wave with {len(t["subpeaks"])} subpeaks')
        ax2.legend(loc='upper right')
        ax2.grid(True, alpha=0.3)
    
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

def estimate_neuron_params(neuron_data):
    """
    根据神经元数据特性估计最优检测参数
    
    参数
    ----------
    neuron_data : numpy.ndarray
        神经元钙离子浓度时间序列数据
        
    返回
    -------
    params : dict
        自适应参数字典
    """
    # 计算基本统计量
    data_mean = np.mean(neuron_data)
    data_std = np.std(neuron_data)
    data_min = np.min(neuron_data)
    data_max = np.max(neuron_data)
    data_range = data_max - data_min
    
    # 更稳健的信噪比计算
    # 使用最高10%和最低10%的数据来计算信号差异
    upper_10_percentile = np.percentile(neuron_data, 90)
    lower_10_percentile = np.percentile(neuron_data, 10)
    robust_range = upper_10_percentile - lower_10_percentile
    
    # 使用中位数绝对偏差(MAD)作为噪声度量，比标准差更稳健
    median_val = np.median(neuron_data)
    mad = np.median(np.abs(neuron_data - median_val))
    
    # 更稳健的信噪比计算
    signal_noise_ratio = robust_range / (mad * 1.4826) if mad > 0 else 0  # 1.4826是使MAD与正态分布的标准差对应的系数
    
    # 评估数据的基线波动
    sorted_data = np.sort(neuron_data)
    lower_half = sorted_data[:len(sorted_data)//2]
    baseline_variability = np.std(lower_half) / np.mean(lower_half) if np.mean(lower_half) > 0 else 0
    
    # 评估峰值特性
    upper_percentile = np.percentile(neuron_data, 95)
    lower_percentile = np.percentile(neuron_data, 20)  # 使用较低百分位估计基线
    peak_intensity = (upper_percentile - lower_percentile) / data_std if data_std > 0 else 0
    
    # 进行初步峰值检测来评估信号特征
    # 使用一个保守的参数集来尝试检测峰值
    from scipy.signal import find_peaks
    peaks, _ = find_peaks(neuron_data, distance=20, prominence=data_std*1.5)
    num_tentative_peaks = len(peaks)
    
    # 自适应参数配置
    params = {}
    
    # 1. 使用更灵活的min_snr调整策略
    # 针对不同情况做特殊处理
    if num_tentative_peaks == 0:  # 使用初步检测没发现任何峰值
        # 大幅降低阈值以尝试检测微弱信号
        params['min_snr'] = 2.5
    elif num_tentative_peaks < 3:  # 只检测到很少的峰值
        # 降低阈值，以便能够检测更多可能的峰值
        params['min_snr'] = 3.0
    else:
        # 根据信噪比使用更细致的调整
        if signal_noise_ratio < 2:
            params['min_snr'] = 2.5
        elif signal_noise_ratio < 3:
            params['min_snr'] = 3.0
        elif signal_noise_ratio < 4:
            params['min_snr'] = 3.5
        elif signal_noise_ratio < 5:
            params['min_snr'] = 4.0
        elif signal_noise_ratio < 7:
            params['min_snr'] = 5.0
        elif signal_noise_ratio < 10:
            params['min_snr'] = 6.0
        elif signal_noise_ratio < 15:
            params['min_snr'] = 7.0
        else:
            params['min_snr'] = 8.0
    
    # 2. 根据基线波动调整baseline_percentile
    # 对于信号差异大的神经元，降低基线百分位以减少漏检
    if num_tentative_peaks == 0 or num_tentative_peaks < 3:
        # 如果初步检测到的峰值很少或没有，使用更低的百分位数以减少漏检
        params['baseline_percentile'] = 5
    elif baseline_variability > 0.3:
        # 基线不稳定，使用更低的百分位数
        params['baseline_percentile'] = 8
    elif baseline_variability > 0.2:
        params['baseline_percentile'] = 10
    elif baseline_variability < 0.1:
        # 基线稳定，可以使用更高的百分位数
        params['baseline_percentile'] = 25
    else:
        # 默认值
        params['baseline_percentile'] = 15
    
    # 3. 根据信号特性调整平滑窗口
    # 对于信号弱的神经元，减小平滑窗口以保留微弱信号
    if num_tentative_peaks == 0 or num_tentative_peaks < 3:
        # 对于难以检测到峰值的神经元，减小平滑窗口
        params['smooth_window'] = 21  # 确保是奇数
    elif baseline_variability > 0.3:
        # 噪声很大的信号需要更大的平滑窗口
        params['smooth_window'] = 101
    elif baseline_variability > 0.25:
        # 噪声大的信号需要更大的平滑窗口
        params['smooth_window'] = 75
    elif baseline_variability < 0.1:
        # 干净的信号可以使用更小的平滑窗口
        params['smooth_window'] = 25
    else:
        # 默认值
        params['smooth_window'] = 51  # 确保是奇数
    
    # 4. 根据峰值强度调整peak_distance
    # 减小peak_distance以检测更密集的峰值
    if num_tentative_peaks == 0 or num_tentative_peaks < 3:
        # 对于检测不到或很少峰值的神经元，使用非常小的距离阈值
        params['peak_distance'] = 15
    elif peak_intensity < 2:
        # 弱峰值可能需要更小的距离
        params['peak_distance'] = 20
    elif peak_intensity > 5:
        # 强峰值通常间隔更大
        params['peak_distance'] = 35
    else:
        # 默认值
        params['peak_distance'] = 25
    
    # 5. 根据信号特性调整min_duration
    # 减小min_duration以检测更短的事件
    if num_tentative_peaks == 0 or num_tentative_peaks < 3:
        # 对于检测难度大的神经元，减小最小持续时间阈值
        params['min_duration'] = 10
    elif baseline_variability > 0.3:
        # 非常嘈杂的信号，要求更长的持续时间以避免假阳性
        params['min_duration'] = 30
    elif baseline_variability > 0.2:
        # 更嘈杂的信号，要求更长的持续时间以避免假阳性
        params['min_duration'] = 25
    elif baseline_variability < 0.1:
        # 干净的信号可以检测更短的事件
        params['min_duration'] = 15
    else:
        # 默认值
        params['min_duration'] = 18
        
    # 6. 子峰检测参数调整
    if num_tentative_peaks == 0 or num_tentative_peaks < 3:
        # 对于难以检测的神经元，降低子峰突出度要求
        params['subpeak_prominence'] = 0.15
    elif peak_intensity > 4:
        # 强峰值信号的子峰可能更明显
        params['subpeak_prominence'] = 0.3
    else:
        # 弱峰值信号需要更低的子峰阈值
        params['subpeak_prominence'] = 0.2
        
    # 7. 调整最大持续时间参数，以便适应不同类型的信号模式
    if num_tentative_peaks < 3:
        # 对于峰值少的数据，增加最大持续时间以捕获较长的事件
        params['max_duration'] = 400
    elif signal_noise_ratio > 10:
        # 高信噪比的信号可能有较短的事件
        params['max_duration'] = 300
    else:
        # 默认值
        params['max_duration'] = 350
        
    # 8. 打印当前神经元的自适应参数，便于调试
    print(f"  - 自适应参数: SNR={signal_noise_ratio:.2f}, min_snr={params['min_snr']}, "  
          f"baseline={params['baseline_percentile']}, peaks={num_tentative_peaks}")
    
    return params

def analyze_all_neurons_transients(data_df, neuron_columns, fs=1.0, save_path=None, adaptive_params=True, start_id=1):
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
    adaptive_params : bool, 可选
        是否为每个神经元使用自适应参数，默认为True
    start_id : int, 可选
        钙爆发ID的起始编号，默认为1
        
    返回
    -------
    all_transients_df : pandas.DataFrame
        包含所有神经元所有钙爆发特征的DataFrame
    next_id : int
        下一个可用的钙爆发ID
    """
    all_transients = []
    transient_id = start_id  # 使用传入的起始ID
    
    for neuron in neuron_columns:
        print(f"处理神经元 {neuron} 的钙爆发...")
        neuron_data = data_df[neuron].values
        
        # 如果启用自适应参数，为每个神经元生成自定义参数
        custom_params = None
        if adaptive_params:
            print(f"  估计神经元 {neuron} 的最优参数...")
            custom_params = estimate_neuron_params(neuron_data)
        
        # 检测钙爆发
        transients, smoothed_data = detect_calcium_transients(neuron_data, fs=fs, params=custom_params)
        
        # 为该神经元的每个钙爆发分配ID并添加到列表
        for t in transients:
            t['neuron'] = neuron
            t['transient_id'] = transient_id
            all_transients.append(t)
            transient_id += 1
    
    # 如果没有检测到钙爆发，返回空DataFrame
    if len(all_transients) == 0:
        print("未检测到任何钙爆发")
        return pd.DataFrame(), transient_id
    
    # 创建DataFrame
    all_transients_df = pd.DataFrame(all_transients)
    
    # 如果指定了保存路径，则保存到Excel
    if save_path:
        # 确保保存目录存在
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        all_transients_df.to_excel(save_path, index=False)
        print(f"成功将所有钙爆发数据保存到: {save_path}")
    
    return all_transients_df, transient_id

if __name__ == "__main__":
    """
    从多个Excel文件加载神经元数据并进行特征提取
    """
    
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='神经元钙离子特征提取工具')
    parser.add_argument('--data', type=str, nargs='+', default=['../datasets/processed_EMtrace.xlsx'],
                        help='数据文件路径列表，支持.xlsx格式，可以指定多个文件')
    parser.add_argument('--output_dir', type=str, default='../results',
                        help='输出基础目录，结果将保存在此目录下对应的子文件夹中')
    parser.add_argument('--combine', action='store_true',
                        help='是否将所有文件的结果合并到一个总表中')
    args = parser.parse_args()
    
    # 展开输入文件路径列表（支持通配符）
    input_files = []
    for path in args.data:
        if os.path.isdir(path):
            # 如果是目录，获取目录中的所有Excel文件
            excel_files = glob.glob(os.path.join(path, "*.xlsx"))
            input_files.extend(excel_files)
        elif '*' in path:
            # 如果包含通配符，展开匹配的文件
            matched_files = glob.glob(path)
            input_files.extend(matched_files)
        elif os.path.exists(path):
            # 单个文件
            input_files.append(path)
    
    if not input_files:
        print("错误: 未找到任何匹配的输入文件")
        exit(1)
    
    print(f"共找到 {len(input_files)} 个输入文件:")
    for file in input_files:
        print(f"  - {file}")
    
    # 用于可能的合并结果
    all_datasets_transients = []
    
    # 全局钙爆发ID计数器，确保唯一性
    next_transient_id = 1
    
    # 处理每个输入文件
    for input_file in input_files:
        if not os.path.exists(input_file):
            print(f"警告: 文件 '{input_file}' 不存在，跳过处理")
            continue
            
        try:
            # 从指定路径加载Excel数据
            print(f"\n正在处理文件: {input_file}")
            df = pd.read_excel(input_file)
            # 清理列名（去除可能的空格）
            df.columns = [col.strip() for col in df.columns]
            print(f"成功加载数据，共 {len(df)} 行")
            
            # 提取神经元列
            neuron_columns = [col for col in df.columns if col.startswith('n') and col[1:].isdigit()]
            print(f"检测到 {len(neuron_columns)} 个神经元数据列")
            
            # 提取数据文件名（不含扩展名）
            data_basename = os.path.basename(input_file)
            dataset_name = os.path.splitext(data_basename)[0]
            output_dir = os.path.join(args.output_dir, dataset_name)
            save_path = os.path.join(output_dir, f"{dataset_name}_transients.xlsx")
            
            print(f"输出目录设置为: {output_dir}")
            
            # 确保保存目录存在
            os.makedirs(output_dir, exist_ok=True)
            
            # 分析并保存所有钙爆发数据，启用自适应参数，传递当前的ID计数器
            transients_df, next_transient_id = analyze_all_neurons_transients(
                df, neuron_columns, save_path=save_path, adaptive_params=True, start_id=next_transient_id
            )
            
            if len(transients_df) > 0:
                print(f"共检测到 {len(transients_df)} 个钙爆发，已保存到 {save_path}")
                
                # 添加数据集标识，用于合并结果
                if args.combine and not transients_df.empty:
                    transients_df['dataset'] = dataset_name
                    all_datasets_transients.append(transients_df)
            else:
                print("未检测到任何钙爆发")
            
        except Exception as e:
            print(f"处理文件 '{input_file}' 时出错: {str(e)}")
    
    # 如果需要合并所有结果
    if args.combine and all_datasets_transients:
        try:
            combined_df = pd.concat(all_datasets_transients, ignore_index=True)
            combined_output_path = os.path.join(args.output_dir, "all_datasets_transients.xlsx")
            combined_df.to_excel(combined_output_path, index=False)
            print(f"\n已将所有数据集的钙爆发结果合并，共 {len(combined_df)} 条记录")
            print(f"合并结果已保存到: {combined_output_path}")
        except Exception as e:
            print(f"合并结果时出错: {str(e)}")
