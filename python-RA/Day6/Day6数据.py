import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# 读取 Excel 文件，假设文件名为 'calcium_data.xlsx'
df = pd.read_excel('calcium_data.xlsx')  # 替换为你的文件路径
calcium_data = df.to_numpy()  # 将导入的数据转换为 numpy 数组

# 1. 提取每个神经元的峰值振幅
peak_amplitudes = []

for i in range(calcium_data.shape[1]):  # 遍历每个神经元
    neuron_signal = calcium_data[:, i]  # 取出第 i 个神经元的信号
    peaks, _ = find_peaks(neuron_signal)  # 提取峰值位置
    amplitudes = neuron_signal[peaks]  # 获取峰值对应的振幅
    if len(amplitudes) > 0:
        peak_amplitudes.append(np.mean(amplitudes))  # 使用峰值振幅的平均值作为特征
    else:
        peak_amplitudes.append(0)  # 如果没有峰值，使用 0 代替

# 转换为 NumPy 数组
peak_amplitudes = np.array(peak_amplitudes).reshape(-1, 1)

# 2. 标准化峰值振幅数据
scaler = StandardScaler()
peak_amplitudes_scaled = scaler.fit_transform(peak_amplitudes)

# 3. 使用 K-means 进行聚类
n_clusters = 3  # 假设我们要分为 3 类，具体数目可以根据需求调整
kmeans = KMeans(n_clusters=n_clusters, random_state=42)
kmeans.fit(peak_amplitudes_scaled)

# 获取聚类标签
labels = kmeans.labels_

# 4. 为每个聚类组生成单独的图片
time_points = np.arange(calcium_data.shape[0])  # 时间点，2999个时间点

for cluster in range(n_clusters):
    plt.figure(figsize=(10, 6))  # 每个图像大小设定为 10x6 英寸
    cluster_neurons = np.where(labels == cluster)[0]  # 获取属于该簇的神经元索引
    for neuron_idx in cluster_neurons:
        plt.plot(time_points, calcium_data[:, neuron_idx], label=f'Neuron {neuron_idx + 1}')

    plt.title(f'Cluster {cluster + 1} Neuron Activity')
    plt.xlabel('Time Points')
    plt.ylabel('Calcium Concentration')
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))

    # 调整图像的间距
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))  # 图例放在右侧
    plt.subplots_adjust(left=0.1, right=0.85, top=0.9, bottom=0.1)  # 调整图的边距
    plt.show()


# # 去除极端值：用95%和5%的分位数值来过滤
# def remove_outliers(signal, lower_percentile=5, upper_percentile=95):
#     lower_threshold = np.percentile(signal, lower_percentile)
#     upper_threshold = np.percentile(signal, upper_percentile)
#     return np.clip(signal, lower_threshold, upper_threshold)
#
#
# # 平滑信号：使用滑动平均
# def smooth_signal(signal, window_size=10):
#     return np.convolve(signal, np.ones(window_size) / window_size, mode='valid')
#
#
# # 绘制每个类的平均信号曲线
# def plot_average_signal(calcium_data, labels, n_clusters, window_size=10):
#     plt.figure(figsize=(10, 8))
#
#     for cluster in range(n_clusters):
#         cluster_neurons = calcium_data[:, labels == cluster]  # 取出该类神经元数据
#         average_signal = np.mean(cluster_neurons, axis=1)  # 计算该类神经元的平均信号
#
#         # 去除极端值并平滑信号
#         average_signal = remove_outliers(average_signal)
#         average_signal = smooth_signal(average_signal, window_size=window_size)
#
#         plt.plot(average_signal, label=f'Cluster {cluster + 1}', linewidth=2)
#
#     plt.title('Smoothed and Outlier-Reduced Signals for Each Cluster')
#     plt.xlabel('Time (frames)')
#     plt.ylabel('Calcium Signal (smoothed)')
#     plt.legend()
#     plt.show()
#
#
# # 运行函数，假设有3个类
# plot_average_signal(calcium_data, labels, n_clusters=3, window_size=20)
