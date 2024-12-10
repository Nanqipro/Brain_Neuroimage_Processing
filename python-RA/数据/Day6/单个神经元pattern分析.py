import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN
import plotly.graph_objects as go


# 数据在 calcium_data文件中
df = pd.read_excel(r'C:\Users\PAN\PycharmProjects\GitHub\python-RA\数据\Day6\calcium_data.xlsx')
calcium_data = df.to_numpy()

# 取 calcium_data 的首列作为单个神经元的数据
neuron_data = calcium_data[:, 0]  # 取第一列的数据，表示一个神经元在3000个时间点的浓度

# 1. 划分时间段，每个段包含 5 个时间戳
segment_size = 5
num_segments = len(neuron_data) // segment_size
segments = np.array_split(neuron_data[:num_segments * segment_size], num_segments)

# 2. 提取特征：幅度、波动模式和持续时长
features = []
for segment in segments:
    peak_value = np.max(segment)  # 峰值
    amplitude = peak_value - np.mean(segment)  # 幅度
    fluctuation = np.std(segment)  # 波动模式

    # 计算持续时长为高于峰值 50% 的时长
    half_peak_threshold = 0.5 * peak_value
    duration = np.sum(segment > half_peak_threshold)  # 满足条件的时间点数目作为持续时长

    features.append([amplitude, fluctuation, duration])

# 转换为 NumPy 数组并标准化特征数据
features = np.array(features)
scaler = StandardScaler()
features_scaled = scaler.fit_transform(features)

# 3. 使用 DBSCAN 聚类（曼哈顿距离）
dbscan = DBSCAN(eps=0.7, min_samples=2, metric='manhattan')
labels = dbscan.fit_predict(features_scaled)

# 4. 使用 Plotly 绘制3D散点图
fig = go.Figure()

unique_labels = set(labels)
for label in unique_labels:
    # 获取属于当前类的时间段索引
    indices = np.where(labels == label)

    # 生成3D散点图
    fig.add_trace(go.Scatter3d(
        x=features_scaled[indices, 0].flatten(),
        y=features_scaled[indices, 1].flatten(),
        z=features_scaled[indices, 2].flatten(),
        mode='markers',
        marker=dict(size=5),
        name=f'Cluster {label}' if label != -1 else 'Noise'
    ))

# 设置轴标签和图例
fig.update_layout(
    scene=dict(
        xaxis_title='Amplitude',
        yaxis_title='Fluctuation',
        zaxis_title='Duration'
    ),
    title='DBSCAN Clustering of Time Segments with Manhattan Distance',
    showlegend=True
)

fig.show()
