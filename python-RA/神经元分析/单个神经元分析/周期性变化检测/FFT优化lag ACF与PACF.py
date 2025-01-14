import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import os
import re
import numpy as np
from scipy.signal import find_peaks

# 设置Matplotlib中文显示（可选）
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用于显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用于正常显示负号


def create_directory(path):
    """
    创建目录（如果不存在）
    """
    if not os.path.exists(path):
        os.makedirs(path)


def optimize_lag_via_fft(signal, sampling_interval, lags_range=(50, 500)):
    """
    通过FFT分析信号频谱并优化滞后期范围

    参数：
    - signal: 时间序列数据（numpy数组或pandas Series）
    - sampling_interval: 采样间隔（秒）
    - lags_range: 滞后期范围（元组，如(50, 500)）

    返回：
    - selected_lags: 选择的滞后期列表
    - selected_magnitudes: 选择的滞后期对应的峰值幅度
    - freq: 所有频率分量
    - fft_magnitude: 所有频率对应的幅度
    - peak_freqs: 峰值频率
    - peak_magnitudes: 峰值幅度
    """
    # 1. 计算FFT
    n = len(signal)
    freq = np.fft.rfftfreq(n, d=sampling_interval)  # 频率
    fft_magnitude = np.abs(np.fft.rfft(signal))  # 幅度

    # 2. 找到频谱中的峰值
    peaks, _ = find_peaks(fft_magnitude, height=0.01)  # 提取峰值索引（阈值可调整）
    peak_freqs = freq[peaks]  # 对应频率
    peak_magnitudes = fft_magnitude[peaks]  # 峰值幅度

    # 3. 根据频率计算滞后期，避免除以零
    peak_freqs_nonzero = peak_freqs[peak_freqs > 0]
    peak_magnitudes_nonzero = peak_magnitudes[peak_freqs > 0]
    lag_values = (1 / peak_freqs_nonzero) / sampling_interval

    print("关键频率和对应滞后期：")
    for f, lag in zip(peak_freqs_nonzero, lag_values):
        print(f"频率: {f:.4f} Hz, 滞后期: {int(lag)}")

    # 4. 选择范围内的滞后期
    selected_lags = [int(lag) for lag in lag_values if lags_range[0] <= lag <= lags_range[1]]
    selected_magnitudes = [mag for lag, mag in zip(lag_values, peak_magnitudes_nonzero) if
                           lags_range[0] <= lag <= lags_range[1]]

    return selected_lags, selected_magnitudes, freq, fft_magnitude, peak_freqs, peak_magnitudes


def perform_fft(calcium_series, time_step):
    """
    对时间序列执行快速傅里叶变换（FFT），返回频率和幅度

    参数：
    - calcium_series: 神经元的钙离子浓度时间序列（pandas Series）
    - time_step: 采样间隔（秒）

    返回：
    - freqs: 频率分量（Hz）
    - magnitudes: 幅度
    """
    N = len(calcium_series)
    T = time_step  # 时间步长（秒）
    # 执行FFT
    fft_vals = np.fft.fft(calcium_series)
    fft_freq = np.fft.fftfreq(N, T)

    # 只保留正频率部分
    pos_mask = fft_freq >= 0
    freqs = fft_freq[pos_mask]
    magnitudes = np.abs(fft_vals[pos_mask]) * 2 / N  # 归一化幅度

    return freqs, magnitudes


def plot_fft(freqs, magnitudes, neuron, sheet_output_dir, image_filename):
    """
    绘制频谱图，并保存为图像文件

    参数：
    - freqs: 频率分量（Hz）
    - magnitudes: 幅度
    - neuron: 神经元名称
    - sheet_output_dir: 当前工作表的输出目录
    - image_filename: 图像文件名
    """
    plt.figure(figsize=(12, 6))
    plt.plot(freqs, magnitudes, color='blue', label='FFT幅度')
    plt.title(f'{neuron} 的频谱图 (FFT)')
    plt.xlabel('频率 (Hz)')
    plt.ylabel('幅度')
    plt.grid(True)

    # 识别主导频率（峰值）
    peaks, _ = find_peaks(magnitudes, height=np.max(magnitudes) * 0.1)  # 只考虑高度超过最大幅度10%的峰
    peak_freqs = freqs[peaks]
    peak_magnitudes = magnitudes[peaks]

    # 标注主导频率
    for freq, mag in zip(peak_freqs, peak_magnitudes):
        plt.annotate(f'{freq:.3f} Hz', xy=(freq, mag), xytext=(freq, mag + np.max(magnitudes) * 0.05),
                     arrowprops=dict(facecolor='red', shrink=0.05), fontsize=8, color='red')

    plt.legend()
    plt.tight_layout()

    # 保存图形
    image_path = os.path.join(sheet_output_dir, image_filename)
    plt.savefig(image_path, dpi=300)
    plt.close()


def process_sheet(df, sheet_name, output_dir, lags_range=(50, 500), time_step=0.48):
    """
    处理单个工作表的数据，计算并绘制ACF、PACF和FFT图

    参数：
    - df: 当前工作表的DataFrame
    - sheet_name: 工作表名称
    - output_dir: 输出根目录
    - lags_range: 滞后期范围（元组）
    - time_step: 采样间隔（秒）
    """
    # 检查是否有至少两列（Time和至少一个神经元）
    if df.shape[1] < 2:
        print(f"工作表 {sheet_name} 中的列数不足，跳过。")
        return

    # 假设第一列为Time
    time_col = df.columns[0]

    # 确保Time列为数值类型（步数）
    df[time_col] = pd.to_numeric(df[time_col], errors='coerce')

    # 检查Time列转换是否成功
    if df[time_col].isnull().all():
        print(f"工作表 {sheet_name} 中，时间列转换为数值失败，跳过。")
        return

    # 创建新的时间列（秒）
    df['Time_seconds'] = df[time_col] * time_step

    # 设置Time_seconds为索引
    df.set_index('Time_seconds', inplace=True)

    # 处理缺失值（使用前向填充）
    df = df.ffill().dropna()

    # 提取神经元名称（假设列名以n+数字命名，例如n1, n2, ..., n60）
    neuron_names = [col for col in df.columns if re.match(r'^n\d+$', col)]

    if not neuron_names:
        print(f"工作表 {sheet_name} 中没有找到符合条件的神经元列，跳过。")
        return

    # 为当前工作表创建输出目录
    sheet_output_dir = os.path.join(output_dir, sheet_name)
    create_directory(sheet_output_dir)

    # 遍历每个神经元
    for neuron in neuron_names:
        print(f"正在处理工作表：{sheet_name}，神经元：{neuron}")

        # 提取时间序列
        calcium_series = df[neuron]

        # 检查是否有足够的数据
        if calcium_series.empty:
            print(f"神经元 {neuron} 没有数据，跳过。")
            continue

        # 优化滞后期 via FFT
        selected_lags, selected_magnitudes, freq, fft_magnitude, peak_freqs, peak_magnitudes = optimize_lag_via_fft(
            calcium_series, sampling_interval=time_step, lags_range=lags_range
        )

        if not selected_lags:
            print(f"神经元 {neuron} 没有符合滞后期范围的lag，使用默认lag=40")
            max_lag = 40
        else:
            # 选择具有最高幅度的lag
            max_lag_index = np.argmax(selected_magnitudes)
            max_lag = selected_lags[max_lag_index]
            print(f"选择滞后期 {max_lag} 作为 {neuron} 的ACF和PACF绘图滞后期。")

        # 绘制ACF图
        fig_acf = plt.figure(figsize=(12, 6))
        plot_acf(calcium_series, lags=max_lag, ax=plt.gca(), title=f'{neuron} 的自相关函数 (ACF)')
        plt.xlabel(f'滞后期数 (每步 {time_step} 秒)')
        plt.tight_layout()
        image_filename_acf = f"{neuron}_ACF.png"
        image_path_acf = os.path.join(sheet_output_dir, image_filename_acf)
        plt.savefig(image_path_acf, dpi=300)
        plt.close()

        # 绘制PACF图
        fig_pacf = plt.figure(figsize=(12, 6))
        plot_pacf(calcium_series, lags=max_lag, ax=plt.gca(), title=f'{neuron} 的偏自相关函数 (PACF)', method='ywm')
        plt.xlabel(f'滞后期数 (每步 {time_step} 秒)')
        plt.tight_layout()
        image_filename_pacf = f"{neuron}_PACF.png"
        image_path_pacf = os.path.join(sheet_output_dir, image_filename_pacf)
        plt.savefig(image_path_pacf, dpi=300)
        plt.close()

        # 绘制频谱图并保存
        image_filename_fft = f"{neuron}_FFT.png"
        plot_fft(freq, fft_magnitude, neuron, sheet_output_dir, image_filename_fft)


def main():
    # 1. 设置输入Excel文件路径
    excel_file = r'C:\Users\PAN\PycharmProjects\GitHub\python-RA\数据\Day6\calcium_data.xlsx'  # 请替换为您的Excel文件路径

    # 2. 设置输出目录
    output_dir = r'C:\Users\PAN\PycharmProjects\GitHub\python-RA\神经元分析\单个神经元分析\周期性变化检测'
    create_directory(output_dir)

    # 3. 读取Excel文件（所有工作表）
    try:
        all_sheets = pd.read_excel(excel_file, sheet_name=None, engine='openpyxl')
    except Exception as e:
        print(f"读取Excel文件失败: {e}")
        return

    # 4. 遍历每个工作表
    for sheet_name, df in all_sheets.items():
        print(f"正在处理工作表：{sheet_name}")
        process_sheet(df, sheet_name, output_dir, lags_range=(50, 500), time_step=0.48)

    print("所有工作表的ACF、PACF和FFT图已生成并保存。")


if __name__ == "__main__":
    main()
