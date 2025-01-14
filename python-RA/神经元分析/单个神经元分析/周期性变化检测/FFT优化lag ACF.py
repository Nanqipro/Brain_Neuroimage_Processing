import numpy as np
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf
from scipy.signal import find_peaks

def optimize_lag_via_fft(signal, sampling_interval, lags_range=(50, 500)):
    """
    通过FFT分析信号频谱并优化滞后期范围
    """
    # 1. 计算FFT
    n = len(signal)
    freq = np.fft.rfftfreq(n, d=sampling_interval)  # 频率
    fft_magnitude = np.abs(np.fft.rfft(signal))     # 幅度

    # 2. 找到频谱中的峰值
    peaks, _ = find_peaks(fft_magnitude, height=0.01)  # 提取峰值索引
    peak_freqs = freq[peaks]                          # 对应频率
    peak_magnitudes = fft_magnitude[peaks]            # 峰值幅度

    # 3. 根据频率计算滞后期
    lag_values = (1 / peak_freqs) / sampling_interval

    print("关键频率和对应滞后期：")
    for f, lag in zip(peak_freqs, lag_values):
        print(f"频率: {f:.4f} Hz, 滞后期: {int(lag)}")

    # 4. 选择范围内的滞后期
    selected_lags = [int(lag) for lag in lag_values if lags_range[0] <= lag <= lags_range[1]]

    return selected_lags, freq, fft_magnitude, peak_freqs, peak_magnitudes

# 示例信号
np.random.seed(0)
signal = np.sin(2 * np.pi * 0.005 * np.arange(3000)) + np.random.normal(0, 0.1, 3000)  # 示例信号
sampling_interval = 0.48  # 采样间隔（秒）

# 优化滞后期
lags_range = (50, 500)  # 定义滞后期范围
selected_lags, freq, fft_magnitude, peak_freqs, peak_magnitudes = optimize_lag_via_fft(signal, sampling_interval, lags_range)

# 绘制FFT图
plt.figure(figsize=(10, 6))
plt.plot(freq, fft_magnitude, label='FFT幅度')
plt.scatter(peak_freqs, peak_magnitudes, color='red', label='峰值频率')
plt.xlabel("频率 (Hz)")
plt.ylabel("幅度")
plt.title("信号的频谱分析 (FFT)")
plt.legend()
plt.show()

# 绘制ACF图
plt.figure(figsize=(10, 6))
for lag in selected_lags:
    plot_acf(signal, lags=lag)
    plt.title(f"ACF 滞后期 {lag}")
plt.show()
