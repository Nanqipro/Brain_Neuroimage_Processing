import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from scipy.signal import find_peaks
import plotly.graph_objects as go

# 假设你的数据在 calcium_data.csv 文件中
df = pd.read_excel('calcium_data9.xlsx')
calcium_data = df.to_numpy()

# 初始化特征列表
peak_values = []
durations = []
amplitudes = []

# 1. 提取每个神经元的峰值、持续时长和幅度
for i in range(calcium_data.shape[1]):
    neuron_signal = calcium_data[:, i]

    # 峰值
    peaks, _ = find_peaks(neuron_signal)
    peak_values.append(np.mean(neuron_signal[peaks]) if len(peaks) > 0 else 0)

    # 持续时长 (峰值之间的平均时间间隔)
    if len(peaks) > 1:
        durations.append(np.mean(np.diff(peaks)))
    else:
        durations.append(0)

    # 幅度 (最大值和最小值的差)
    amplitude = np.max(neuron_signal) - np.min(neuron_signal)
    amplitudes.append(amplitude)

# 将特征转换为 NumPy 数组并进行标准化
features = np.array([peak_values, durations, amplitudes]).T
scaler = StandardScaler()
features_scaled = scaler.fit_transform(features)

# 2. 使用 DBSCAN 进行聚类
# eps：邻居之间的最大距离；min_samples：形成簇的最小邻居数量
dbscan = DBSCAN(eps=0.5, min_samples=2)
labels = dbscan.fit_predict(features_scaled)

# 使用 Plotly 绘制3D散点图
fig = go.Figure()

# 获取每个类的点（DBSCAN 标签），并为每个类单独绘制
unique_labels = set(labels)
for label in unique_labels:
    # 获取当前类的神经元索引
    indices = np.where(labels == label)

    # 生成3D散点
    fig.add_trace(go.Scatter3d(
        x=features_scaled[indices, 0].flatten(),
        y=features_scaled[indices, 1].flatten(),
        z=features_scaled[indices, 2].flatten(),
        mode='markers',
        marker=dict(size=5),
        name=f'Cluster {label}' if label != -1 else 'Noise'  # -1 表示噪声点
    ))

# 设置轴标签
fig.update_layout(
    scene=dict(
        xaxis_title='Peak Value',
        yaxis_title='Duration',
        zaxis_title='Amplitude'
    ),
    title='DBSCAN Clustering of Neurons in 3D Space',
    showlegend=True
)

# 显示交互式图形
fig.show()
