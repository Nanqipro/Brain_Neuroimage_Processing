import numpy as np
import pandas as pd
from scipy import signal
from scipy.signal import find_peaks, peak_widths, savgol_filter
from scipy.integrate import trapezoid
import matplotlib.pyplot as plt
import os
import argparse
import sys

def integrate_multiple_datasets(data_files, neuron_columns_list=None, fs=1.0, 
                               output_path=None, adaptive_params=True):
    """
    整合多个数据集的钙爆发分析结果
    
    参数
    ----------
    data_files : list of str
        数据文件路径列表（Excel或CSV文件）
    neuron_columns_list : list of list, 可选
        每个数据文件中要处理的神经元列名列表，如果为None则自动检测
    fs : float, 可选
        采样频率，默认为1.0Hz
    output_path : str, 可选
        输出Excel文件路径，默认为None（不保存）
    adaptive_params : bool, 可选
        是否使用自适应参数，默认为True
        
    返回
    -------
    integrated_df : pandas.DataFrame
        整合的钙爆发特征DataFrame
    summary_df : pandas.DataFrame
        各数据集神经元特征汇总表
    """
    all_transients_dfs = []
    all_summaries = []
    
    # 为每个数据集分配唯一标识符
    for dataset_idx, file_path in enumerate(data_files):
        # 加载数据
        if file_path.endswith('.xlsx') or file_path.endswith('.xls'):
            data_df = pd.read_excel(file_path)
        elif file_path.endswith('.csv'):
            data_df = pd.read_csv(file_path)
        else:
            print(f"不支持的文件格式: {file_path}")
            continue
            
        # 确定神经元列
        if neuron_columns_list is not None and dataset_idx < len(neuron_columns_list):
            neuron_columns = neuron_columns_list[dataset_idx]
        else:
            # 自动检测可能的神经元列（排除时间列和非数值列）
            neuron_columns = [col for col in data_df.columns if data_df[col].dtype.kind in 'fc' 
                            and not any(time_key in col.lower() for time_key in ['time', 'frame'])]
        
        if len(neuron_columns) == 0:
            print(f"在数据集 {file_path} 中未找到任何神经元列")
            continue
            
        # 分析当前数据集中的所有神经元
        dataset_name = os.path.basename(file_path).split('.')[0]
        temp_save_path = None if output_path is None else f"temp_{dataset_name}_transients.xlsx"
        
        try:
            # 分析当前数据集中的所有神经元钙爆发
            transients_df = analyze_all_neurons_transients(
                data_df, neuron_columns, fs=fs, 
                save_path=temp_save_path, 
                adaptive_params=adaptive_params
            )
            
            # 添加数据集标识
            transients_df['dataset_id'] = dataset_idx
            transients_df['dataset_name'] = dataset_name
            
            # 收集结果
            all_transients_dfs.append(transients_df)
            
            # 生成当前数据集的摘要统计信息
            summary = transients_df.groupby('neuron_id').agg({
                'amplitude': ['mean', 'std', 'count'],
                'duration': ['mean', 'std'],
                'auc': ['mean', 'std'],
                'rise_time': ['mean', 'std'],
                'decay_time': ['mean', 'std'],
                'fwhm': ['mean', 'std']
            }).reset_index()
            
            # 添加数据集信息
            summary['dataset_id'] = dataset_idx
            summary['dataset_name'] = dataset_name
            summary['total_neurons'] = len(neuron_columns)
            
            all_summaries.append(summary)
            
            print(f"成功处理数据集: {dataset_name}, 神经元数量: {len(neuron_columns)}")
            
        except Exception as e:
            print(f"处理数据集 {dataset_name} 时出错: {str(e)}")
            continue
            
    # 如果没有成功处理任何数据集，则返回空
    if len(all_transients_dfs) == 0:
        print("未能成功处理任何数据集")
        return None, None
        
    # 整合所有数据集的结果
    integrated_df = pd.concat(all_transients_dfs, ignore_index=True)
    
    # 重新分配全局唯一ID
    integrated_df['global_transient_id'] = range(1, len(integrated_df) + 1)
    
    # 整合摘要统计
    summary_df = pd.concat(all_summaries, ignore_index=True)
    
    # 保存结果
    if output_path is not None:
        try:
            # 创建包含多个sheet的Excel文件
            with pd.ExcelWriter(output_path) as writer:
                # 保存详细数据，保留索引
                integrated_df.to_excel(writer, sheet_name='All_Transients')
                
                # 保存摘要数据，保留索引
                summary_df.to_excel(writer, sheet_name='Summary_Statistics')
                
                # 每个数据集一个sheet
                for dataset_idx in range(len(data_files)):
                    dataset_data = integrated_df[integrated_df['dataset_id'] == dataset_idx]
                    if len(dataset_data) > 0:
                        dataset_name = dataset_data['dataset_name'].iloc[0]
                        # 确保sheet名称不超过31个字符（Excel限制）
                        sheet_name = f'Dataset_{dataset_name[:28]}'
                        # 重置索引以避免MultiIndex问题
                        dataset_data.reset_index(drop=True).to_excel(writer, sheet_name=sheet_name)
                        
            print(f"整合结果已保存至: {output_path}")
        except Exception as e:
            print(f"保存结果时出错: {str(e)}")
    
    return integrated_df, summary_df

def analyze_all_neurons_transients(data_df, neuron_columns, fs=1.0, save_path=None, adaptive_params=True):
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
        是否使用自适应参数，默认为True
        
    返回
    -------
    all_transients_df : pandas.DataFrame
        包含所有神经元所有钙爆发特征的DataFrame
    """
    all_transients = []
    transient_id = 1  # 起始ID
    
    for neuron_id, neuron in enumerate(neuron_columns):
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
            # 转换子峰列表为可序列化的格式
            if 'subpeaks' in t and t['subpeaks']:
                subpeaks_data = []
                for sp in t['subpeaks']:
                    subpeaks_data.append({
                        'index': int(sp['index']),
                        'value': float(sp['value']),
                        'amplitude': float(sp['amplitude']),
                        'start_idx': int(sp['start_idx']),
                        'end_idx': int(sp['end_idx']),
                        'duration': float(sp['duration']),
                        'fwhm': float(sp['fwhm']),
                        'rise_time': float(sp['rise_time']),
                        'decay_time': float(sp['decay_time']),
                        'auc': float(sp['auc'])
                    })
                t['subpeaks'] = subpeaks_data
            
            # 添加神经元信息
            t['neuron'] = neuron
            t['neuron_id'] = neuron_id + 1  # 从1开始编号
            t['transient_id'] = transient_id
            
            # 转换为DataFrame友好的格式
            transient_data = {
                'transient_id': t['transient_id'],
                'neuron': t['neuron'],
                'neuron_id': t['neuron_id'],
                'start_idx': int(t['start_idx']),
                'peak_idx': int(t['peak_idx']),
                'end_idx': int(t['end_idx']),
                'amplitude': float(t['amplitude']),
                'peak_value': float(t['peak_value']),
                'baseline': float(t['baseline']),
                'duration': float(t['duration']),
                'fwhm': float(t['fwhm']),
                'rise_time': float(t['rise_time']),
                'decay_time': float(t['decay_time']),
                'auc': float(t['auc']),
                'snr': float(t['snr']),
                'wave_type': t['wave_type'],
                'subpeaks_count': int(t['subpeaks_count'])
            }
            
            all_transients.append(transient_data)
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
        os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)
        # 重置索引以避免MultiIndex问题
        all_transients_df.reset_index(drop=True).to_excel(save_path, index=False)
        print(f"成功将所有钙爆发数据保存到: {save_path}")
    
    return all_transients_df

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

def detect_calcium_transients(data, fs=1.0, min_snr=3.0, min_duration=10, smooth_window=30, 
                             peak_distance=20, baseline_percentile=15, max_duration=350,
                             detect_subpeaks=True, subpeak_prominence=0.2, 
                             subpeak_width=8, subpeak_distance=12, params=None):
    """
    检测神经元钙离子浓度时间序列中的钙爆发事件
    
    参数
    ----------
    data : numpy.ndarray
        荧光强度时间序列数据
    fs : float, 可选
        采样频率（Hz），默认为1.0
    min_snr : float, 可选
        最小信噪比，默认为3.0
    min_duration : float, 可选
        最小持续时间（秒），默认为10秒
    smooth_window : int, 可选
        平滑窗口大小，默认为30个点
    peak_distance : int, 可选
        峰之间的最小距离，默认为20个点
    baseline_percentile : int, 可选
        用于计算基线的百分位数，默认为15
    max_duration : float, 可选
        最大持续时间（秒），默认为350秒
    detect_subpeaks : bool, 可选
        是否检测子峰，默认为True
    subpeak_prominence : float, 可选
        子峰突出度阈值，默认为0.2
    subpeak_width : int, 可选
        子峰最小宽度，默认为8个点
    subpeak_distance : int, 可选
        子峰之间的最小距离，默认为12个点
    params : dict, 可选
        覆盖默认参数的字典，默认为None
        
    返回
    -------
    transients : list of dict
        检测到的钙爆发事件列表，每个事件包含多个特征
    smoothed_data : numpy.ndarray
        平滑后的数据
    """
    # 使用提供的参数（如果有）覆盖默认参数
    if params is not None:
        min_snr = params.get('min_snr', min_snr)
        min_duration = params.get('min_duration', min_duration)
        smooth_window = params.get('smooth_window', smooth_window)
        peak_distance = params.get('peak_distance', peak_distance)
        baseline_percentile = params.get('baseline_percentile', baseline_percentile)
        max_duration = params.get('max_duration', max_duration)
        detect_subpeaks = params.get('detect_subpeaks', detect_subpeaks)
        subpeak_prominence = params.get('subpeak_prominence', subpeak_prominence)
        subpeak_width = params.get('subpeak_width', subpeak_width)
        subpeak_distance = params.get('subpeak_distance', subpeak_distance)
    
    # 数据平滑处理
    if smooth_window > 0:
        smoothed_data = savgol_filter(data, min(smooth_window, len(data) - 1 if len(data) % 2 == 0 else len(data) - 2), 3)
    else:
        smoothed_data = data.copy()
    
    # 计算基线水平（使用百分位数）
    baseline = np.percentile(smoothed_data, baseline_percentile)
    
    # 计算噪声水平（作为低频波动的标准差估计）
    noise_level = np.std(smoothed_data[smoothed_data < np.percentile(smoothed_data, 50)])
    
    # 寻找峰值
    peaks, properties = find_peaks(
        smoothed_data, 
        distance=peak_distance, 
        prominence=min_snr * noise_level, 
        height=baseline + min_snr * noise_level
    )
    
    # 如果没有检测到峰值，返回空列表
    if len(peaks) == 0:
        return [], smoothed_data
    
    # 初始化钙爆发列表
    transients = []
    
    # 分析每个检测到的峰值
    for i, peak_idx in enumerate(peaks):
        # 获取峰值属性
        peak_value = smoothed_data[peak_idx]
        prominence = properties['prominences'][i]
        
        # 确定钙爆发的开始和结束
        left_base = properties['left_bases'][i]
        right_base = properties['right_bases'][i]
        
        # 计算半峰全宽 (FWHM)
        half_height = baseline + prominence / 2
        width_data = peak_widths(smoothed_data, [peak_idx], rel_height=0.5)
        fwhm = width_data[0][0] / fs  # 转换为秒
        
        # 计算上升时间和下降时间
        rise_indices = np.arange(left_base, peak_idx + 1)
        decay_indices = np.arange(peak_idx, right_base + 1)
        
        rise_time = len(rise_indices) / fs  # 上升时间（秒）
        decay_time = len(decay_indices) / fs  # 下降时间（秒）
        
        # 计算持续时间
        duration = (right_base - left_base) / fs  # 总持续时间（秒）
        
        # 计算面积（曲线下面积，AUC）
        if left_base < right_base:
            auc = trapezoid(smoothed_data[left_base:right_base+1] - baseline) / fs
        else:
            auc = 0
        
        # 跳过持续时间太短或太长的峰
        if duration < min_duration / fs or (max_duration > 0 and duration > max_duration / fs):
            continue
        
        # 计算信噪比 (SNR)
        snr = prominence / noise_level
        
        # 初始化子峰列表
        subpeaks = []
        subpeaks_count = 1  # 默认包含主峰
        wave_type = 'simple'  # 默认为简单波形
        
        # 检测子峰（如果需要）
        if detect_subpeaks and duration > subpeak_width * 2 / fs:
            # 在钙爆发区域内寻找子峰
            wave_segment = smoothed_data[left_base:right_base+1]
            # 计算突出度阈值为主峰突出度的一定比例
            sub_prominence_threshold = prominence * subpeak_prominence
            
            sub_peaks, sub_props = find_peaks(
                wave_segment, 
                distance=subpeak_distance,
                prominence=sub_prominence_threshold,
                width=subpeak_width
            )
            
            # 如果检测到子峰
            if len(sub_peaks) > 1:  # 超过1个子峰（包括主峰）
                wave_type = 'complex'
                subpeaks_count = len(sub_peaks)
                
                # 分析每个子峰
                for j, sub_idx in enumerate(sub_peaks):
                    global_sub_idx = left_base + sub_idx
                    sub_value = smoothed_data[global_sub_idx]
                    sub_prominence = sub_props['prominences'][j]
                    
                    # 子峰的左右基点
                    sub_left_base = left_base + sub_props['left_bases'][j]
                    sub_right_base = left_base + sub_props['right_bases'][j]
                    
                    # 计算子峰的半峰全宽
                    sub_half_height = smoothed_data[sub_left_base] + sub_prominence / 2
                    sub_width_data = peak_widths(smoothed_data, [global_sub_idx], rel_height=0.5)
                    sub_fwhm = sub_width_data[0][0] / fs
                    
                    # 计算子峰的上升和下降时间
                    sub_rise_indices = np.arange(sub_left_base, global_sub_idx + 1)
                    sub_decay_indices = np.arange(global_sub_idx, sub_right_base + 1)
                    
                    sub_rise_time = len(sub_rise_indices) / fs
                    sub_decay_time = len(sub_decay_indices) / fs
                    
                    # 计算子峰的面积
                    if sub_left_base < sub_right_base:
                        sub_auc = trapezoid(smoothed_data[sub_left_base:sub_right_base+1] - baseline) / fs
                    else:
                        sub_auc = 0
                    
                    # 记录子峰信息
                    subpeaks.append({
                        'index': global_sub_idx,
                        'value': sub_value,
                        'amplitude': sub_prominence,
                        'start_idx': sub_left_base,
                        'end_idx': sub_right_base,
                        'duration': (sub_right_base - sub_left_base) / fs,
                        'fwhm': sub_fwhm,
                        'rise_time': sub_rise_time,
                        'decay_time': sub_decay_time,
                        'auc': sub_auc
                    })
        
        # 记录钙爆发的所有特征
        transient = {
            'peak_idx': peak_idx,
            'peak_value': peak_value,
            'amplitude': prominence,
            'baseline': baseline,
            'start_idx': left_base,
            'end_idx': right_base,
            'duration': duration,
            'fwhm': fwhm,
            'rise_time': rise_time,
            'decay_time': decay_time,
            'auc': auc,
            'snr': snr,
            'wave_type': wave_type,
            'subpeaks_count': subpeaks_count,
            'subpeaks': subpeaks
        }
        
        transients.append(transient)
    
    return transients, smoothed_data

def estimate_neuron_params(neuron_data):
    """
    估计神经元钙爆发检测的最优参数
    
    参数
    ----------
    neuron_data : numpy.ndarray
        神经元钙离子浓度时间序列数据
    
    返回
    -------
    params : dict
        最优参数字典
    """
    # 计算数据统计特性
    data_mean = np.mean(neuron_data)
    data_std = np.std(neuron_data)
    data_ptp = np.ptp(neuron_data)  # 峰峰值
    data_percentile_20 = np.percentile(neuron_data, 20)
    data_percentile_80 = np.percentile(neuron_data, 80)
    
    # 根据数据特性调整参数
    # 信噪比 - 如果数据波动小，减小阈值；如果波动大，增加阈值
    snr_factor = 1.0
    if data_ptp / data_mean < 0.5:  # 相对波动小
        snr_factor = 0.8
    elif data_ptp / data_mean > 2.0:  # 相对波动大
        snr_factor = 1.2
    
    min_snr = 3.0 * snr_factor
    
    # 平滑窗口 - 根据数据长度调整
    smooth_window = min(int(len(neuron_data) * 0.05), 51)
    # 确保窗口是奇数
    if smooth_window % 2 == 0:
        smooth_window += 1
    
    # 峰距离 - 根据数据长度和波动调整
    peak_factor = 1.0
    if data_ptp / data_std < 3.0:  # 峰不太明显
        peak_factor = 0.8
    else:  # 峰比较明显
        peak_factor = 1.2
    
    peak_distance = max(10, int(len(neuron_data) * 0.01 * peak_factor))
    
    # 基线百分位数 - 根据数据分布调整
    if data_percentile_20 < data_mean * 0.5:  # 数据有较多低值
        baseline_percentile = 10
    else:  # 数据分布较均匀
        baseline_percentile = 20
    
    # 持续时间参数
    min_duration = 5  # 较短的最小持续时间，单位为采样点
    max_duration = int(len(neuron_data) * 0.2)  # 最大持续时间为数据长度的20%
    
    # 子峰检测参数
    subpeak_prominence = 0.3  # 子峰突出度阈值，相对于主峰
    subpeak_width = max(3, int(peak_distance * 0.3))  # 子峰最小宽度
    subpeak_distance = max(2, int(peak_distance * 0.2))  # 子峰最小距离
    
    # 组合参数
    params = {
        'min_snr': min_snr,
        'min_duration': min_duration,
        'smooth_window': smooth_window,
        'peak_distance': peak_distance,
        'baseline_percentile': baseline_percentile,
        'max_duration': max_duration,
        'detect_subpeaks': True,
        'subpeak_prominence': subpeak_prominence,
        'subpeak_width': subpeak_width,
        'subpeak_distance': subpeak_distance
    }
    
    return params

def visualize_calcium_transients(data, smoothed_data, transients, fs=1.0):
    """
    可视化钙爆发检测结果
    
    参数
    ----------
    data : numpy.ndarray
        原始钙离子浓度数据
    smoothed_data : numpy.ndarray
        平滑后的数据
    transients : list of dict
        检测到的钙爆发事件列表
    fs : float, 可选
        采样频率（Hz），默认为1.0
    """
    # 创建时间轴
    time = np.arange(len(data)) / fs
    
    # 创建图形
    plt.figure(figsize=(14, 10))
    
    # 绘制原始数据和平滑数据
    plt.subplot(211)
    plt.plot(time, data, 'k-', alpha=0.5, label='原始数据')
    plt.plot(time, smoothed_data, 'b-', label='平滑数据')
    
    # 在图上标记检测到的钙爆发
    for i, t in enumerate(transients):
        # 提取特征
        peak_idx = t['peak_idx']
        peak_time = peak_idx / fs
        peak_value = t['peak_value']
        start_idx = t['start_idx']
        end_idx = t['end_idx']
        
        # 高亮钙爆发区域
        plt.axvspan(start_idx/fs, end_idx/fs, alpha=0.2, color='r')
        
        # 标记峰值
        plt.plot(peak_time, peak_value, 'ro', markersize=8)
        plt.text(peak_time, peak_value*1.05, f"#{i+1}", fontsize=10, ha='center')
        
        # 标记子峰（如果有）
        if 'subpeaks' in t and t['subpeaks']:
            for sp in t['subpeaks']:
                sub_idx = sp['index']
                sub_time = sub_idx / fs
                sub_value = sp['value']
                plt.plot(sub_time, sub_value, 'go', markersize=6)
    
    plt.xlabel('时间 (秒)')
    plt.ylabel('荧光强度')
    plt.title('钙爆发检测结果')
    plt.legend()
    plt.grid(True)
    
    # 显示钙爆发特征表格
    plt.subplot(212)
    plt.axis('off')
    
    if len(transients) > 0:
        # 创建表格数据
        table_data = [
            ['ID', '峰值时间 (s)', '振幅', '持续时间 (s)', 'FWHM (s)', '上升时间 (s)', '下降时间 (s)', 'AUC', 'SNR', '波形类型', '子峰数']
        ]
        
        for i, t in enumerate(transients):
            row = [
                f"#{i+1}", 
                f"{t['peak_idx']/fs:.2f}", 
                f"{t['amplitude']:.2f}", 
                f"{t['duration']:.2f}", 
                f"{t['fwhm']:.2f}", 
                f"{t['rise_time']:.2f}", 
                f"{t['decay_time']:.2f}", 
                f"{t['auc']:.2f}", 
                f"{t['snr']:.2f}", 
                t['wave_type'], 
                t['subpeaks_count']
            ]
            table_data.append(row)
        
        # 创建表格
        table = plt.table(
            cellText=table_data,
            cellLoc='center',
            loc='center',
            colWidths=[0.05, 0.1, 0.08, 0.1, 0.1, 0.1, 0.1, 0.08, 0.08, 0.1, 0.08]
        )
        
        # 设置表格样式
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 1.5)
        
        # 设置表头样式
        for (i, j), cell in table.get_celld().items():
            if i == 0:
                cell.set_text_props(weight='bold')
                cell.set_facecolor('lightgray')
    else:
        plt.text(0.5, 0.5, "未检测到钙爆发", fontsize=16, ha='center')
    
    plt.title('检测到的钙爆发特征')
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    """
    从Excel文件加载神经元数据并进行特征提取的示例
    """
    # 设置默认的数据文件路径
    default_files = [
        "../datasets/processed_Day3.xlsx",
        "../datasets/processed_Day6.xlsx",
        "../datasets/processed_Day9.xlsx"
    ]
    
    # 设置默认的输出文件路径
    default_output = "../results/integrated_transients.xlsx"
    
    parser = argparse.ArgumentParser(description='神经元钙离子浓度数据分析工具')
    parser.add_argument('--file', type=str, help='输入Excel或CSV文件路径')
    parser.add_argument('--files', nargs='+', default=default_files, help='多个数据文件路径列表（用于整合分析）')
    parser.add_argument('--fs', type=float, default=1.0, help='采样频率（Hz）')
    parser.add_argument('--output', type=str, default=default_output, help='输出文件路径')
    parser.add_argument('--visualize', action='store_true', help='可视化结果')
    parser.add_argument('--neurons', nargs='+', help='要分析的神经元列名列表（不指定则自动检测）')
    parser.add_argument('--behavior', type=str, help='行为标签列名')
    parser.add_argument('--adaptive', action='store_true', default=True, help='使用自适应参数')
    parser.add_argument('--all_transients', action='store_true', help='分析所有神经元的所有钙爆发')
    parser.add_argument('--integrate', action='store_true', default=True, help='整合多个数据集的分析结果')
    
    args = parser.parse_args()
    
    # 默认执行整合多个数据集的分析
    if args.integrate:
        print(f"正在整合 {len(args.files)} 个数据集的分析结果...")
        
        # 确保输出目录存在
        output_dir = os.path.dirname(args.output)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)
            
        integrated_df, summary_df = integrate_multiple_datasets(
            args.files,
            neuron_columns_list=None,  # 自动检测神经元列
            fs=args.fs,
            output_path=args.output,
            adaptive_params=args.adaptive
        )
        
        if integrated_df is not None:
            print(f"共分析了 {len(set(integrated_df['dataset_id']))} 个数据集")
            print(f"共检测到 {len(integrated_df)} 个钙爆发")
            print("各数据集神经元数量:")
            for idx, (name, count) in enumerate(zip(summary_df['dataset_name'].unique(), 
                                                 summary_df.groupby('dataset_name')['total_neurons'].first())):
                print(f"  数据集 {idx+1}: {name} - {count} 个神经元")
        sys.exit(0)
    
    # 以下为原有的单数据集分析逻辑 (仅在未设置整合时执行)
    if not args.file:
        print("错误: 必须提供输入文件路径 (--file)")
        sys.exit(1)
        
    # 检查文件是否存在
    file_path = args.file
    if not os.path.exists(file_path):
        print(f"错误: 文件不存在: {file_path}")
        sys.exit(1)
        
    # 加载数据
    if file_path.endswith('.xlsx') or file_path.endswith('.xls'):
        data_df = pd.read_excel(file_path)
    elif file_path.endswith('.csv'):
        data_df = pd.read_csv(file_path)
    else:
        print(f"错误: 不支持的文件格式: {file_path}")
        sys.exit(1)
        
    # 确定神经元列
    if args.neurons:
        neuron_columns = args.neurons
    else:
        # 自动检测可能的神经元列（排除时间列和非数值列）
        neuron_columns = [col for col in data_df.columns if data_df[col].dtype.kind in 'fc' 
                         and not any(time_key in col.lower() for time_key in ['time', 'frame'])]
    
    print(f"检测到 {len(neuron_columns)} 个神经元列")
    
    # 分析所有神经元的所有钙爆发
    if args.all_transients:
        output_path = args.output or "all_transients.xlsx"
        all_transients_df = analyze_all_neurons_transients(
            data_df, neuron_columns, fs=args.fs, 
            save_path=output_path,
            adaptive_params=args.adaptive
        )
        print(f"已保存所有钙爆发分析结果至: {output_path}")
        print(f"共检测到 {len(all_transients_df)} 个钙爆发")
        sys.exit(0)
        
    # 常规特征统计分析
    if args.behavior:
        # 分析不同行为下的特征
        results_df = analyze_behavior_specific_features(
            data_df, neuron_columns, 
            behavior_col=args.behavior, 
            fs=args.fs
        )
        if args.output:
            results_df.to_excel(args.output, index=False)
            print(f"已保存行为相关分析结果至: {args.output}")
    else:
        # 分析所有神经元的特征
        results_df = process_multiple_neurons(data_df, neuron_columns, fs=args.fs)
        if args.output:
            results_df.to_excel(args.output, index=False)
            print(f"已保存分析结果至: {args.output}")
            
    # 打印汇总统计信息
    print("\n神经元钙离子特征统计:")
    print(results_df.describe())
    
    # 可视化单个神经元的结果（如果选项启用）
    if args.visualize and len(neuron_columns) > 0:
        print(f"\n可视化第一个神经元的结果: {neuron_columns[0]}")
        # 获取原始数据
        neuron_data = data_df[neuron_columns[0]].values
        
        # 如果需要，估计最优参数
        params = None
        if args.adaptive:
            params = estimate_neuron_params(neuron_data)
            
        # 检测钙爆发
        transients, smoothed_data = detect_calcium_transients(
            neuron_data, fs=args.fs, 
            params=params
        )
        
        # 可视化结果
        visualize_calcium_transients(neuron_data, smoothed_data, transients, fs=args.fs)
