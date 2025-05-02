import numpy as np
import pandas as pd
from scipy import signal
from scipy.signal import find_peaks, peak_widths, savgol_filter
from numpy import trapezoid
import matplotlib.pyplot as plt
import os
import argparse
import logging
import datetime
import sys

# 初始化日志记录器
def setup_logger(output_dir=None, prefix="element_extraction", capture_all_output=True):
    """
    设置日志记录器，将日志消息输出到控制台和文件
    
    参数:
        output_dir: 日志文件输出目录，默认为输出到当前脚本所在目录的logs文件夹
        prefix: 日志文件名称前缀
        capture_all_output: 是否捕获所有标准输出到日志文件
    
    返回:
        logger: 已配置好的日志记录器
    """
    # 创建日志记录器
    logger = logging.getLogger(prefix)
    logger.setLevel(logging.INFO)
    
    # 清除现有的处理器，避免重复添加
    if logger.handlers:
        for handler in logger.handlers:
            logger.removeHandler(handler)
    
    # 创建格式化器
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    # 添加控制台处理器
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # 如果指定了输出目录，则添加文件处理器
    if output_dir is None:
        # 默认在当前脚本目录下创建 logs 文件夹
        output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs")
    
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)
    
    # 创建日志文件名，包含时间戳
    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    log_file = os.path.join(output_dir, f"{prefix}_{timestamp}.log")
    
    # 添加文件处理器
    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    
    # 创建输出重定向类，捕获所有的标准输出和错误输出
    class OutputRedirector:
        def __init__(self, original_stream, logger, level=logging.INFO):
            self.original_stream = original_stream
            self.logger = logger
            self.level = level
            self.buffer = ''
        
        def write(self, buf):
            self.original_stream.write(buf)
            self.buffer += buf
            if '\n' in buf:
                self.flush()
        
        def flush(self):
            if self.buffer.strip():
                for line in self.buffer.rstrip().split('\n'):
                    if line.strip():  # 只记录非空行
                        self.logger.log(self.level, f"OUTPUT: {line.rstrip()}")
            self.buffer = ''
    
    # 重定向标准输出和错误输出到日志文件
    if capture_all_output:
        sys.stdout = OutputRedirector(sys.stdout, logger, logging.INFO)
        sys.stderr = OutputRedirector(sys.stderr, logger, logging.ERROR)
    
    logger.info(f"日志文件创建于: {log_file}")
    return logger

def detect_calcium_transients(data, fs=1.0, min_snr=8.0, min_duration=20, smooth_window=50, 
                             peak_distance=30, baseline_percentile=20, max_duration=350,
                             detect_subpeaks=True, subpeak_prominence=0.25, 
                             subpeak_width=10, subpeak_distance=15, params=None, filter_strength=1.0):
    """
    检测钙离子浓度数据中的钙爆发(calcium transients)，包括大波中的小波动
    增强对钙爆发形态的过滤，剔除不符合典型钙波特征的信号
    
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
    params : dict, 可选
        自定义参数字典，可覆盖默认参数
    filter_strength : float, 可选
        过滤强度调节参数，值越大过滤越强，默认为1.0。
        可以调整此参数来平衡检测灵敏度和假阳性率，
        值<1.0会降低过滤强度（更多检测），值>1.0会增加过滤强度（更少检测）
        
    返回
    -------
    transients : list of dict
        每个钙爆发的特征参数字典列表
    smoothed_data : numpy.ndarray
        平滑后的数据
    """
    # 获取数据特征和信噪比信息，用于调整参数
    data_mean = np.mean(data)
    data_std = np.std(data)
    robust_range = np.percentile(data, 98) - np.percentile(data, 5)
    
    # 使用更高级的噪声评估方法
    median_val = np.median(data)
    mad = np.median(np.abs(data - median_val))
    signal_noise_ratio = robust_range / (mad * 1.4826) if mad > 0 else 0
    
    # 评估数据的基线稳定性
    sorted_data = np.sort(data)
    lower_half = sorted_data[:len(sorted_data)//3]  # 使用下三分之一作为基线估计
    baseline_variability = np.std(lower_half) / np.mean(lower_half) if np.mean(lower_half) > 0 else 0
    
    # 进行初步峰值检测来评估信号特征
    from scipy.signal import find_peaks
    peaks, _ = find_peaks(data, distance=25, prominence=data_std*2.0)
    num_tentative_peaks = len(peaks)
    
    # 自适应参数配置，在这里使用现有的参数作为默认值创建完整参数字典
    local_params = {
        'min_snr': min_snr,
        'min_duration': min_duration,
        'smooth_window': smooth_window,
        'peak_distance': peak_distance,
        'baseline_percentile': baseline_percentile,
        'max_duration': max_duration,
        'subpeak_prominence': subpeak_prominence,
        'subpeak_width': subpeak_width,
        'subpeak_distance': subpeak_distance,
        'min_morphology_score': 0.45  # 新增默认形态评分阈值
    }
    
    # 1. 根据filter_strength调整信噪比阈值，平衡过滤噪声和保留信号
    base_snr = 0.0  # 基础信噪比阈值
    
    if num_tentative_peaks == 0:  
        # 即使没有检测到峰值，也提高最低阈值，避免误检
        base_snr = 4.0
    elif num_tentative_peaks < 3: 
        base_snr = 4.5
    else:
        # 根据信噪比使用折中的调整
        if signal_noise_ratio < 2:
            base_snr = 4.5
        elif signal_noise_ratio < 3:
            base_snr = 5.0
        elif signal_noise_ratio < 4:
            base_snr = 5.5
        elif signal_noise_ratio < 5:
            base_snr = 6.0
        elif signal_noise_ratio < 7:
            base_snr = 7.0
        elif signal_noise_ratio < 10:
            base_snr = 8.0
        elif signal_noise_ratio < 15:
            base_snr = 8.5
        else:
            base_snr = 9.0
    
    # 根据filter_strength调整最终SNR阈值
    local_params['min_snr'] = base_snr * filter_strength
    
    # 2. 调整baseline_percentile，更准确地估计基线
    if baseline_variability > 0.3:
        # 对于基线不稳定的信号，使用更低的百分位以更准确地捕获真实基线
        local_params['baseline_percentile'] = 10
    elif baseline_variability > 0.2:
        local_params['baseline_percentile'] = 12
    elif baseline_variability < 0.1:
        # 基线稳定时，可以使用更高的百分位数，避免误将低振幅波动视为信号
        local_params['baseline_percentile'] = 30
    else:
        local_params['baseline_percentile'] = 20
    
    # 3. 根据filter_strength和baseline_variability调整平滑窗口
    base_window = 0  # 基础窗口大小
    
    if baseline_variability > 0.35:
        base_window = 121
    elif baseline_variability > 0.25:
        base_window = 101
    elif baseline_variability > 0.15:
        base_window = 81
    elif baseline_variability < 0.1:
        base_window = 51
    else:
        base_window = 65
    
    # 根据filter_strength调整窗口大小
    if filter_strength > 1.0:
        window_adjustment = int(base_window * (filter_strength - 1) * 0.5)
        local_params['smooth_window'] = base_window + window_adjustment
    elif filter_strength < 1.0:
        window_adjustment = int(base_window * (1 - filter_strength) * 0.3)
        local_params['smooth_window'] = max(21, base_window - window_adjustment)
    else:
        local_params['smooth_window'] = base_window
    
    # 4. 调整峰值间距离和持续时间要求
    if filter_strength > 1.0:
        local_params['peak_distance'] = int(peak_distance * (1 + (filter_strength - 1) * 0.5))
        local_params['min_duration'] = int(min_duration * (1 + (filter_strength - 1) * 0.3))
    elif filter_strength < 1.0:
        local_params['peak_distance'] = max(15, int(peak_distance * (1 - (1 - filter_strength) * 0.3)))
        local_params['min_duration'] = max(10, int(min_duration * (1 - (1 - filter_strength) * 0.2)))
    
    # 5. 调整子峰参数
    if filter_strength > 1.0:
        local_params['subpeak_prominence'] = min(0.45, subpeak_prominence * (1 + (filter_strength - 1) * 0.5))
    elif filter_strength < 1.0:
        local_params['subpeak_prominence'] = max(0.15, subpeak_prominence * (1 - (1 - filter_strength) * 0.4))
    
    # 如果提供了外部参数字典，使用它来覆盖我们计算的值
    if params is not None:
        for key, value in params.items():
            local_params[key] = value
    
    # 将自适应计算的参数提取到变量中使用
    min_snr = local_params['min_snr']
    min_duration = local_params['min_duration']
    smooth_window = local_params['smooth_window']
    peak_distance = local_params['peak_distance']
    baseline_percentile = local_params['baseline_percentile']
    max_duration = local_params['max_duration']
    subpeak_prominence = local_params['subpeak_prominence']
    subpeak_width = local_params['subpeak_width']
    subpeak_distance = local_params['subpeak_distance']
    
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
        
        # 新增：计算典型钙波形态特征评分
        # 1. 上升期陡峭、下降期缓慢的特征 - 钙波通常上升快，下降慢
        rise_decay_ratio = rise_time / decay_time if decay_time > 0 else float('inf')
        
        # 2. 计算波形对称性 - 钙波通常是非对称的（快速上升，缓慢下降）
        # 理想钙波的对称性应该较低
        left_half = smoothed_data[start_idx:peak_idx+1] - baseline
        right_half = smoothed_data[peak_idx:end_idx+1] - baseline
        
        # 对齐左右两侧长度
        min_half_len = min(len(left_half), len(right_half))
        if min_half_len > 3:  # 确保有足够的点来计算
            left_half_resampled = np.interp(
                np.linspace(0, 1, min_half_len),
                np.linspace(0, 1, len(left_half)),
                left_half
            )
            right_half_resampled = np.interp(
                np.linspace(0, 1, min_half_len),
                np.linspace(0, 1, len(right_half)),
                right_half[::-1]  # 反转右半部分
            )
            
            # 计算两侧差异作为非对称度量
            asymmetry = np.sum(np.abs(left_half_resampled - right_half_resampled)) / np.sum(left_half_resampled)
        else:
            # 如果半峰太短，则使用默认值
            asymmetry = 0
            
        # 3. 计算上升沿的平滑性和单调性
        if peak_idx > start_idx + 2:
            # 使用一阶差分评估平滑性
            rise_segment = smoothed_data[start_idx:peak_idx+1]
            rise_diff = np.diff(rise_segment)
            
            # 负值比例表示非单调上升的程度
            non_monotonic_ratio = np.sum(rise_diff < 0) / len(rise_diff) if len(rise_diff) > 0 else 1
            
            # 计算一阶差分的波动性 - 平滑的上升沿应有较小的波动
            rise_smoothness = np.std(rise_diff) / np.mean(rise_diff) if np.mean(rise_diff) > 0 else float('inf')
        else:
            non_monotonic_ratio = 1
            rise_smoothness = float('inf')
            
        # 4. 下降沿的指数衰减特性
        if end_idx > peak_idx + 3:
            # 提取衰减部分并归一化
            decay_segment = smoothed_data[peak_idx:end_idx+1] - baseline
            decay_segment = decay_segment / decay_segment[0]  # 归一化
            
            # 对数变换后应近似线性 - 测量对数变换后的线性度
            log_decay = np.log(decay_segment + 1e-10)  # 防止log(0)
            
            # 使用线性拟合，计算拟合度
            x = np.arange(len(log_decay))
            if len(x) > 1:  # 确保有足够的点拟合
                try:
                    slope, intercept = np.polyfit(x, log_decay, 1)
                    # 计算R²作为衰减指数特性指标
                    y_pred = slope * x + intercept
                    ss_tot = np.sum((log_decay - np.mean(log_decay))**2)
                    ss_res = np.sum((log_decay - y_pred)**2)
                    exp_decay_score = 1 - ss_res/ss_tot if ss_tot > 0 else 0
                except:
                    exp_decay_score = 0
            else:
                exp_decay_score = 0
        else:
            exp_decay_score = 0
            
        # 5. 钙爆发持续时间/宽度比例 - 钙爆发通常有一定的形态比例
        duration_width_ratio = duration / fwhm if fwhm > 0 else 0
        
        # 组合计算一个形态评分：0-1之间，越高表示越符合典型钙波特征
        # 理想情况下：rise_decay_ratio应该小，asymmetry应该大，
        # non_monotonic_ratio应该小，rise_smoothness应该小，
        # exp_decay_score应该大，duration_width_ratio在适当范围
        
        # 定义理想的参数范围
        ideal_rise_decay_ratio = 0.3  # 理想的上升/衰减时间比例
        min_asymmetry = 0.2  # 最小非对称度
        max_non_monotonic = 0.3  # 上升沿非单调性最大允许值
        ideal_duration_width_ratio = 2.5  # 理想的持续时间/宽度比
        
        # 计算各指标的评分
        rise_decay_score = np.exp(-2 * abs(rise_decay_ratio - ideal_rise_decay_ratio)) if rise_decay_ratio < 1 else 0
        asymmetry_score = min(asymmetry / min_asymmetry, 1) if min_asymmetry > 0 else 0
        monotonic_score = 1 - min(non_monotonic_ratio / max_non_monotonic, 1)
        duration_ratio_score = np.exp(-0.5 * abs(duration_width_ratio - ideal_duration_width_ratio))
        
        # 综合形态评分 (0-1)，权重可根据重要性调整
        morphology_score = (
            0.25 * rise_decay_score + 
            0.2 * asymmetry_score + 
            0.2 * monotonic_score + 
            0.2 * exp_decay_score + 
            0.15 * duration_ratio_score
        )
        
        # 设置形态评分阈值，过滤不符合典型钙波形态的峰值
        min_morphology_score = 0.4  # 可调整此阈值
        
        if morphology_score < min_morphology_score:
            continue  # 跳过此峰值，因为形态不符合典型钙波特征
        
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
            'subpeaks': subpeaks,
            'morphology_score': morphology_score,  # 添加形态评分
            'rise_decay_ratio': rise_decay_ratio,
            'asymmetry': asymmetry,
            'exp_decay_score': exp_decay_score
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

def extract_calcium_features(neuron_data, fs=1.0, visualize=False, detect_subpeaks=True, params=None, filter_strength=1.0):
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
    params : dict, 可选
        自定义参数字典，可覆盖默认参数
    filter_strength : float, 可选
        过滤强度调节参数，值越大过滤越强，默认为1.0
        可以调整此参数来平衡检测灵敏度和假阳性率
        
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
    transients, smoothed_data = detect_calcium_transients(data, fs=fs, detect_subpeaks=detect_subpeaks, params=params, filter_strength=filter_strength)
    
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
    可视化钙离子浓度数据和检测到的钙爆发，包括形态评分信息
    
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
    plt.figure(figsize=(14, 10))
    
    # 创建三个子图：原始数据、放大的波峰和形态评分分布
    ax1 = plt.subplot(3, 1, 1)
    ax1.plot(time, raw_data, 'k-', alpha=0.4, label='Raw data')
    ax1.plot(time, smoothed_data, 'b-', label='Smoothed data')
    
    # 用颜色梯度表示形态评分
    import matplotlib.cm as cm
    from matplotlib.colors import Normalize
    
    # 获取所有形态评分
    scores = [t.get('morphology_score', 0.5) for t in transients]
    norm = Normalize(vmin=0.4, vmax=1.0)  # 评分范围从0.4到1.0
    cmap = cm.viridis
    
    # 标记钙爆发
    for i, t in enumerate(transients):
        # 使用形态评分来确定颜色，评分越高颜色越亮
        morphology_score = t.get('morphology_score', 0.5)
        color = cmap(norm(morphology_score))
        marker_size = 8 + morphology_score * 5  # 根据评分调整标记大小
        
        # 标记主峰值
        ax1.plot(t['peak_idx']/fs, t['peak_value'], 'o', color=color, markersize=marker_size)
        
        # 标记开始和结束
        ax1.axvline(x=t['start_idx']/fs, color=color, linestyle='--', alpha=0.5)
        ax1.axvline(x=t['end_idx']/fs, color=color, linestyle='--', alpha=0.5)
        
        # 添加编号和评分
        ax1.text(t['peak_idx']/fs, t['peak_value']*1.05, 
                 f"{i+1}\n{morphology_score:.2f}", 
                 horizontalalignment='center', verticalalignment='bottom',
                 fontsize=8)
        
        # 标记子峰（如果有）
        if 'subpeaks' in t and t['subpeaks']:
            for sp in t['subpeaks']:
                ax1.plot(sp['index']/fs, sp['value'], '*', color='magenta', markersize=8)
    
    # 添加颜色条以显示形态评分范围
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax1, orientation='vertical', pad=0.01)
    cbar.set_label('形态评分')
    
    ax1.set_xlabel('Time (seconds)')
    ax1.set_ylabel('Calcium Signal Intensity')
    ax1.set_title(f'Detected {len(transients)} calcium transient events')
    ax1.legend(loc='upper right')
    ax1.grid(True, alpha=0.3)
    
    # 选择一个高形态评分的波形进行放大显示
    if transients:
        ax2 = plt.subplot(3, 1, 2)
        
        # 尝试选择形态评分最高的波形
        if 'morphology_score' in transients[0]:
            t = max(transients, key=lambda x: x.get('morphology_score', 0))
        else:
            # 如果没有形态评分，则选择最大振幅的波形
            t = max(transients, key=lambda x: x['amplitude'])
        
        # 计算放大区域
        margin = 20  # 左右额外显示的点数
        zoom_start = max(0, t['start_idx'] - margin)
        zoom_end = min(len(raw_data), t['end_idx'] + margin)
        
        # 绘制放大区域
        zoom_time = time[zoom_start:zoom_end]
        ax2.plot(zoom_time, raw_data[zoom_start:zoom_end], 'k-', alpha=0.4, label='Raw data')
        ax2.plot(zoom_time, smoothed_data[zoom_start:zoom_end], 'b-', label='Smoothed data')
        
        # 计算高度百分比标记点
        peak_val = t['peak_value']
        baseline = t['baseline']
        amplitude = peak_val - baseline
        heights = [0.25, 0.5, 0.75]  # 25%, 50%, 75%的高度
        
        # 标记主峰
        ms = t.get('morphology_score', 0.5)
        color = cmap(norm(ms))
        ax2.plot(t['peak_idx']/fs, peak_val, 'o', color=color, markersize=10, 
                label=f'Peak (Score: {ms:.2f})')
        
        # 标记不同高度百分比
        for h in heights:
            h_val = baseline + h * amplitude
            ax2.axhline(y=h_val, color='gray', linestyle=':', alpha=0.5)
            ax2.text(zoom_time[0], h_val, f'{int(h*100)}%', 
                    verticalalignment='center', fontsize=8)
        
        # 标记基线
        ax2.axhline(y=baseline, color='r', linestyle='-', alpha=0.5, label='Baseline')
        
        # 添加关键特征标注
        rise_time = t['rise_time']
        decay_time = t['decay_time']
        fwhm = t['fwhm']
        ratio = t.get('rise_decay_ratio', 0)
        asymm = t.get('asymmetry', 0)
        
        # 在图上标记这些特征
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
        feature_text = (f"Rise: {rise_time:.2f}s\nDecay: {decay_time:.2f}s\n"
                        f"FWHM: {fwhm:.2f}s\nRise/Decay: {ratio:.2f}\n"
                        f"Asymmetry: {asymm:.2f}")
        ax2.text(0.05, 0.95, feature_text, transform=ax2.transAxes, 
                fontsize=9, verticalalignment='top', bbox=props)
        
        # 标记子峰
        if 'subpeaks' in t and t['subpeaks']:
            for sp in t['subpeaks']:
                ax2.plot(sp['index']/fs, sp['value'], '*', color='magenta', markersize=8)
                # 标记子峰边界
                ax2.axvline(x=sp['start_idx']/fs, color='c', linestyle=':', alpha=0.7)
                ax2.axvline(x=sp['end_idx']/fs, color='m', linestyle=':', alpha=0.7)
        
        ax2.set_xlabel('Time (seconds)')
        ax2.set_ylabel('Calcium Signal Intensity')
        ax2.set_title(f'High-quality calcium wave (Score: {ms:.2f})')
        ax2.legend(loc='upper right')
        ax2.grid(True, alpha=0.3)
        
        # 第三个子图：形态评分分布直方图
        ax3 = plt.subplot(3, 1, 3)
        
        # 提取所有波形的形态特征
        if 'morphology_score' in transients[0]:
            morphology_scores = [t.get('morphology_score', 0) for t in transients]
            rise_decay_ratios = [t.get('rise_decay_ratio', 0) for t in transients]
            asymmetries = [t.get('asymmetry', 0) for t in transients]
            
            # 绘制形态评分直方图
            ax3.hist(morphology_scores, bins=15, alpha=0.7, color='skyblue', 
                    label='Morphology Scores')
            ax3.set_xlabel('Morphology Score')
            ax3.set_ylabel('Frequency')
            ax3.set_title('Distribution of Calcium Wave Morphology Scores')
            
            # 添加中位数标记
            median_score = np.median(morphology_scores)
            ax3.axvline(x=median_score, color='r', linestyle='--', label=f'Median: {median_score:.2f}')
            
            # 添加统计信息文本框
            stats_text = (f"Total waves: {len(transients)}\n"
                        f"Median score: {median_score:.2f}\n"
                        f"Mean rise/decay ratio: {np.mean(rise_decay_ratios):.2f}\n"
                        f"Mean asymmetry: {np.mean(asymmetries):.2f}")
            ax3.text(0.95, 0.95, stats_text, transform=ax3.transAxes, 
                    fontsize=9, horizontalalignment='right', 
                    verticalalignment='top', bbox=props)
            
            ax3.legend()
            ax3.grid(True, alpha=0.3)
        else:
            # 如果没有形态评分，则显示振幅分布
            amplitudes = [t['amplitude'] for t in transients]
            ax3.hist(amplitudes, bins=15, alpha=0.7, color='skyblue', 
                    label='Amplitude Distribution')
            ax3.set_xlabel('Amplitude')
            ax3.set_ylabel('Frequency')
            ax3.set_title('Distribution of Calcium Wave Amplitudes')
            ax3.grid(True, alpha=0.3)
    
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
    根据神经元数据特性估计最优检测参数，强化过滤条件以减少噪声
    
    参数
    ----------
    neuron_data : numpy.ndarray
        神经元钙离子浓度时间序列数据
        
    返回
    -------
    params : dict
        强化过滤的自适应参数字典
    """
    # 计算基本统计量
    data_mean = np.mean(neuron_data)
    data_std = np.std(neuron_data)
    data_min = np.min(neuron_data)
    data_max = np.max(neuron_data)
    data_range = data_max - data_min
    
    # 更稳健的信噪比计算
    # 使用最高5%和最低5%的数据来计算信号差异，更严格的信号范围判断
    upper_percentile = np.percentile(neuron_data, 95)
    lower_percentile = np.percentile(neuron_data, 5)
    robust_range = upper_percentile - lower_percentile
    
    # 使用中位数绝对偏差(MAD)作为噪声度量
    median_val = np.median(neuron_data)
    mad = np.median(np.abs(neuron_data - median_val))
    
    # 更严格的信噪比计算
    signal_noise_ratio = robust_range / (mad * 1.4826) if mad > 0 else 0
    
    # 评估数据的基线稳定性
    sorted_data = np.sort(neuron_data)
    lower_half = sorted_data[:len(sorted_data)//3]  # 使用下三分之一作为基线估计
    baseline_variability = np.std(lower_half) / np.mean(lower_half) if np.mean(lower_half) > 0 else 0
    
    # 评估峰值特性
    upper_percentile = np.percentile(neuron_data, 98)  # 更严格地只考虑最高的2%
    lower_percentile = np.percentile(neuron_data, 15)  # 使用较低百分位估计基线
    peak_intensity = (upper_percentile - lower_percentile) / data_std if data_std > 0 else 0
    
    # 进行初步峰值检测来评估信号特征
    # 使用更严格的参数进行初步检测
    from scipy.signal import find_peaks
    peaks, _ = find_peaks(neuron_data, distance=25, prominence=data_std*2.0)
    num_tentative_peaks = len(peaks)
    
    # 自适应参数配置，提高门槛以过滤更多噪声
    params = {}
    
    # 1. 显著提高信噪比阈值
    if num_tentative_peaks == 0:  
        # 即使没有检测到峰值，也提高最低阈值，避免误检
        params['min_snr'] = 3.5  # 提高最低阈值
    elif num_tentative_peaks < 3: 
        params['min_snr'] = 4.0  # 提高小峰值的检测阈值
    else:
        # 根据信噪比使用更严格的调整
        if signal_noise_ratio < 2:
            params['min_snr'] = 4.0  # 显著提高
        elif signal_noise_ratio < 3:
            params['min_snr'] = 4.5
        elif signal_noise_ratio < 4:
            params['min_snr'] = 5.0
        elif signal_noise_ratio < 5:
            params['min_snr'] = 5.5
        elif signal_noise_ratio < 7:
            params['min_snr'] = 6.5
        elif signal_noise_ratio < 10:
            params['min_snr'] = 8.0
        elif signal_noise_ratio < 15:
            params['min_snr'] = 9.0
        else:
            params['min_snr'] = 10.0  # 进一步提高高信噪比情况下的阈值
    
    # 2. 调整baseline_percentile，更准确地估计基线
    if baseline_variability > 0.3:
        # 对于基线不稳定的信号，使用更低的百分位以更准确地捕获真实基线
        params['baseline_percentile'] = 10
    elif baseline_variability > 0.2:
        params['baseline_percentile'] = 12
    elif baseline_variability < 0.1:
        # 基线稳定时，可以使用更高的百分位数，避免误将低振幅波动视为信号
        params['baseline_percentile'] = 30
    else:
        params['baseline_percentile'] = 20
    
    # 3. 增大平滑窗口以减少噪声影响
    if baseline_variability > 0.35:
        # 极度嘈杂的信号需要更大的平滑窗口
        params['smooth_window'] = 121
    elif baseline_variability > 0.25:
        params['smooth_window'] = 101
    elif baseline_variability > 0.15:
        params['smooth_window'] = 75
    elif baseline_variability < 0.1:
        # 即使对于干净的信号，也保持一定的平滑以提高一致性
        params['smooth_window'] = 41
    else:
        params['smooth_window'] = 61  # 提高默认平滑窗口大小
    
    # 4. 提高peak_distance以减少近距离的多重检测
    if peak_intensity > 8:
        # 强峰值信号通常间隔更大，增加距离以避免多次检测同一事件
        params['peak_distance'] = 45
    elif peak_intensity > 5:
        params['peak_distance'] = 40
    elif peak_intensity > 3:
        params['peak_distance'] = 35
    else:
        # 提高默认距离，减少相邻事件的虚假检测
        params['peak_distance'] = 30
    
    # 5. 提高最小持续时间，避免检测短暂噪声
    if baseline_variability > 0.3:
        # 嘈杂信号需要更长的持续时间以区分真实信号和噪声
        params['min_duration'] = 40
    elif baseline_variability > 0.2:
        params['min_duration'] = 35
    elif baseline_variability > 0.1:
        params['min_duration'] = 30
    else:
        # 即使对于干净的信号，也提高时间阈值
        params['min_duration'] = 25
        
    # 6. 提高子峰检测的突出度要求
    if peak_intensity > 6:
        # 强信号中的子峰需要更明显才能被认为是有效子峰
        params['subpeak_prominence'] = 0.4
    elif peak_intensity > 4:
        params['subpeak_prominence'] = 0.35
    else:
        # 增加默认阈值，减少误检
        params['subpeak_prominence'] = 0.3
        
    # 7. 子峰宽度和距离要求增强
    params['subpeak_width'] = 15  # 增加子峰最小宽度
    params['subpeak_distance'] = 20  # 增加子峰间最小距离
        
    # 8. 调整最大持续时间参数
    if signal_noise_ratio > 10:
        # 高信噪比的信号可能有较短的事件
        params['max_duration'] = 300
    else:
        # 保持原有的最大持续时间
        params['max_duration'] = 350
        
    # 9. 打印当前神经元的自适应参数，便于调试
    print(f"  - 强化过滤参数: SNR={signal_noise_ratio:.2f}, min_snr={params['min_snr']}, "  
          f"baseline={params['baseline_percentile']}, min_duration={params['min_duration']}, "
          f"smooth={params['smooth_window']}, peaks={num_tentative_peaks}")
    
    return params

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
    parser.add_argument('--data', type=str, default='../datasets/processed_EMtrace.xlsx',
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
            
            # 分析并保存所有钙爆发数据，启用自适应参数
            all_transients = analyze_all_neurons_transients(df, neuron_columns, save_path=save_path, adaptive_params=True)
            print(f"共检测到 {len(all_transients)} 个钙爆发")
            
        except Exception as e:
            print(f"加载或处理数据时出错: {str(e)}")
    else:
        print(f"错误: 找不到数据文件 '{args.data}'，请检查文件路径")