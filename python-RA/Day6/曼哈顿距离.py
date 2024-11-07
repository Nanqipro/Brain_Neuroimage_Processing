import numpy as np
from sklearn.preprocessing import StandardScaler
from scipy.spatial.distance import cdist
import plotly.graph_objects as go
import numpy as np
import pandas as pd
from scipy.signal import find_peaks

# 假设你的数据在 calcium_data.excel 文件中
df = pd.read_excel('calcium_data.xlsx')
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

features = np.array([
    peak_values, durations, amplitudes
]).T

# 1. 数据标准化
scaler = StandardScaler()
features_scaled = scaler.fit_transform(features)

# 2. 初始化聚类参数
n_clusters = 5  # 聚类的数量
max_iter = 100  # 最大迭代次数
tol = 1e-4  # 收敛容忍度

# 随机选择初始聚类中心
np.random.seed(42)
initial_centroids = features_scaled[np.random.choice(features_scaled.shape[0], n_clusters, replace=False)]

# 3. 开始聚类迭代
centroids = initial_centroids
for i in range(max_iter):
    # 计算每个点到聚类中心的曼哈顿距离
    distances = cdist(features_scaled, centroids, metric='cityblock')  # 使用曼哈顿距离（cityblock）

    # 分配每个点到最近的聚类中心
    labels = np.argmin(distances, axis=1)

    # 计算新的聚类中心
    new_centroids = np.array([features_scaled[labels == j].mean(axis=0) for j in range(n_clusters)])

    # 判断收敛条件
    if np.all(np.abs(new_centroids - centroids) < tol):
        print(f"聚类在 {i + 1} 次迭代后收敛。")
        break

    centroids = new_centroids

# 初始化一个 Plotly 3D 图形
fig = go.Figure()

# 获取每个类的点（根据 labels），并为每个类单独绘制
unique_labels = set(labels)
for label in unique_labels:
    # 获取属于当前类的点的索引
    indices = np.where(labels == label)

    # 生成3D散点图，每个类别不同颜色
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
    title='Manhattan Distance-Based Clustering of Neurons in 3D Space',
    showlegend=True
)

# 显示交互式图形
fig.show()
