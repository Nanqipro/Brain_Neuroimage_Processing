import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from scipy.fft import fft, fftfreq
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap
import os
from scipy.signal import find_peaks
from scipy import signal
import time as time_module

df = pd.read_csv('day3.csv')


output_folder = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'visualization_results')
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

neuron_columns = [col for col in df.columns if col.startswith('n') and col[1:].isdigit()]
print(f"找到 {len(neuron_columns)} 个神经元数据列")

start_time = time_module.time()
for neuron_idx, neuron_name in enumerate(neuron_columns):
    print(f"处理神经元 {neuron_name} ({neuron_idx+1}/{len(neuron_columns)})...")
    
    plt.figure(figsize=(15, 12))
    gs = gridspec.GridSpec(3, 2, height_ratios=[2, 1, 1])

    ax1 = plt.subplot(gs[0, :])
    time = df['stamp'].values
    neuron_data = df[neuron_name].values
    ax1.plot(time, neuron_data, 'b-', alpha=0.5, label='Raw Data')

    # 使用Savitzky-Golay滤波器
    window_length = 51  # 必须是奇数
    polyorder = 3
    neuron_smooth = signal.savgol_filter(neuron_data, window_length, polyorder)
    ax1.plot(time, neuron_smooth, 'r-', linewidth=2, label='Smoothed Curve')

    # 添加均值线
    mean_val = np.mean(neuron_data)
    ax1.axhline(y=mean_val, color='g', linestyle='--', label=f'Mean: {mean_val:.3f}')

    # 添加标准差范围
    std_val = np.std(neuron_data)
    ax1.axhline(y=mean_val + std_val, color='g', linestyle=':', alpha=0.5, label=f'Std Dev: {std_val:.3f}')
    ax1.axhline(y=mean_val - std_val, color='g', linestyle=':', alpha=0.5)
    ax1.fill_between(time, mean_val - std_val, mean_val + std_val, color='g', alpha=0.1)

    # 标记峰值
    peaks, _ = find_peaks(neuron_data, height=mean_val + 0.5*std_val, distance=50)
    ax1.plot(time[peaks], neuron_data[peaks], 'ro', label='Peaks')

    # 看行为标签，每种颜色对应一种行为
    behaviors = df['behavior'].unique()
    colors = plt.cm.tab10(np.linspace(0, 1, len(behaviors)))
    behavior_color_map = {behavior: color for behavior, color in zip(behaviors, colors)}

    current_behavior = df['behavior'].iloc[0]
    start_idx = 0

    for i, behavior in enumerate(df['behavior']):
        if behavior != current_behavior:
            ax1.axvspan(time[start_idx], time[i-1], alpha=0.2, color=behavior_color_map[current_behavior])
            current_behavior = behavior
            start_idx = i
    ax1.axvspan(time[start_idx], time[-1], alpha=0.2, color=behavior_color_map[current_behavior])

    ax1.set_title(f'Time Series Analysis of Neuron {neuron_name}', fontsize=14)
    ax1.set_xlabel('Time')
    ax1.set_ylabel('Value')
    ax1.legend(loc='upper right')
    ax1.grid(True)

    # 2. 频谱分析
    ax2 = plt.subplot(gs[1, 0])
    # 计算FFT
    N = len(neuron_data)
    T = 1.0  # 假设采样间隔为1，这里可能需要根据Ca波调整一下
    yf = fft(neuron_data)
    xf = fftfreq(N, T)[:N//2]
    ax2.plot(xf, 2.0/N * np.abs(yf[0:N//2]))
    ax2.set_title('Frequency Spectrum Analysis')
    ax2.set_xlabel('Frequency')
    ax2.set_ylabel('Amplitude')
    ax2.grid(True)

    ax3 = plt.subplot(gs[1, 1])
    widths = np.arange(1, 31)
    cwtmatr = signal.cwt(neuron_data, signal.ricker, widths)
    ax3.imshow(cwtmatr, extent=[0, time[-1], 1, 31], cmap='viridis', aspect='auto')
    ax3.set_title('Wavelet Transform')
    ax3.set_ylabel('Scale')
    ax3.set_xlabel('Time')

    # 行为分类统计
    ax4 = plt.subplot(gs[2, 0])
    behavior_stats = df.groupby('behavior')[neuron_name].agg(['mean', 'std', 'min', 'max'])
    behavior_stats.plot(kind='bar', y='mean', yerr='std', ax=ax4, legend=False)
    ax4.set_title(f'Average {neuron_name} Values by Behavior')
    ax4.set_ylabel(f'{neuron_name} Average Value')
    ax4.set_xlabel('Behavior Type')

    # 5. 相关性热图
    ax5 = plt.subplot(gs[2, 1])
    # 选择部分神经元计算相关性（当前神经元和其他9个），到时候选关键神经元来改
    related_neurons = [neuron_name]
    other_neurons = [n for n in neuron_columns if n != neuron_name]
    related_neurons.extend(other_neurons[:min(9, len(other_neurons))])
    
    corr = df[related_neurons].corr()
    im = ax5.imshow(corr, cmap='coolwarm')
    ax5.set_title('Neuron Correlation')
    ax5.set_xticks(np.arange(len(related_neurons)))
    ax5.set_yticks(np.arange(len(related_neurons)))
    ax5.set_xticklabels(related_neurons)
    ax5.set_yticklabels(related_neurons)
    plt.colorbar(im, ax=ax5)

    plt.tight_layout()

    output_path = os.path.join(output_folder, f'neuron_{neuron_name}.png')
    plt.savefig(output_path, dpi=400)
    plt.close()
    
    if (neuron_idx + 1) % 10 == 0 or neuron_idx == len(neuron_columns) - 1:
        elapsed_time = time_module.time() - start_time
        avg_time_per_neuron = elapsed_time / (neuron_idx + 1)
        remaining_neurons = len(neuron_columns) - (neuron_idx + 1)
        estimated_time_remaining = avg_time_per_neuron * remaining_neurons
        
        print(f"已完成: {neuron_idx+1}/{len(neuron_columns)} 神经元")
        print(f"已用时间: {elapsed_time:.2f}秒")
        print(f"预计剩余时间: {estimated_time_remaining:.2f}秒")
        print("-" * 40)

print(f"所有神经元处理完成！结果保存在: {output_folder}")