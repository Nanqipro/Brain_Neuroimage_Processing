import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import mpl_toolkits.mplot3d
import plotly.graph_objects as go
from sklearn.cluster import DBSCAN

# 你的数据在 calcium_data 文件中，包含 60 列表示神经元，500 行表示时间点
df = pd.read_excel('calcium_data9.xlsx')
calcium_data = df.to_numpy()

# 初始化特征列表
peak_values = []
durations = []
fluctuations = []

# 1. 提取每个神经元的峰值、持续时长和波动模式
for i in range(calcium_data.shape[1]):
    neuron_signal = calcium_data[:, i]

    # 峰值
    peaks , prominence = find_peaks(neuron_signal)
    peak_values.append(np.mean(neuron_signal[peaks]) if len(peaks) > 0 else 0)

    # 持续时长 (峰值之间的平均时间间隔)
    if len(peaks) > 1:
        durations.append(np.mean(np.diff(peaks)))
    else:
        durations.append(0)

    # 波动模式 (标准差)
    fluctuations.append(np.std(neuron_signal))

# 转换为 NumPy 数组并进行标准化
features = np.array([peak_values, durations, fluctuations]).T
scaler = StandardScaler()
features_scaled = scaler.fit_transform(features)

# 2. 使用 KMeans 聚类
n_clusters = 5
kmeans = KMeans(n_clusters=n_clusters, random_state=42)
labels = kmeans.fit_predict(features_scaled)

# #交互式三维 # 创建3D散点图
# fig = go.Figure()
#
# for cluster in range(5):
#     indices = np.where(labels == cluster)
#     fig.add_trace(go.Scatter3d(
#         x=features_scaled[indices, 0].flatten(),
#         y=features_scaled[indices, 1].flatten(),
#         z=features_scaled[indices, 2].flatten(),
#         mode='markers',
#         marker=dict(size=5),
#         name=f'Cluster {cluster+1}'
#     ))
#
# # 轴标签
# fig.update_layout(scene=dict(
#     xaxis_title='Peak Value',
#     yaxis_title='Duration',
#     zaxis_title='Fluctuation'
# ))
#
# # 显示3D交互图
# fig.show()

# #交互式NAOT # 每个聚类类别绘制一个图
# num_clusters = 5
# time_points = calcium_data.shape[0]  # 时间点数
#
# for cluster in range(num_clusters):
#     fig = go.Figure()
#
#     # 获取属于当前类别的神经元索引
#     neuron_indices = np.where(labels == cluster)[0]
#
#     # 创建每个神经元的曲线，并设置初始的可见性
#     for neuron in neuron_indices:
#         fig.add_trace(
#             go.Scatter(
#                 x=np.arange(time_points),
#                 y=calcium_data[:, neuron],
#                 mode='lines',
#                 name=f'Neuron {neuron + 1}',
#                 line=dict(width=2.5),  # 加深曲线颜色并增粗线条
#                 opacity=0.65
#             )
#         )
#
#     # 设置按钮，以便高亮显示单条曲线
#     buttons = []
#     for i, neuron in enumerate(neuron_indices):
#         # 每个按钮控制一个神经元的显示状态
#         visibility = [False] * len(neuron_indices)  # 设置所有神经元初始为不可见
#         visibility[i] = True  # 仅高亮当前选中的神经元
#
#         buttons.append(dict(
#             label=f'Neuron {neuron + 1}',
#             method='update',
#             args=[{'visible': visibility},  # 更新每条曲线的可见性
#                   {'title': f'Cluster {cluster + 1} - Highlight Neuron {neuron + 1}'}]
#         ))
#
#     # 添加“显示全部”按钮
#     buttons.append(dict(
#         label='Show All',
#         method='update',
#         args=[{'visible': [True] * len(neuron_indices)},
#               {'title': f'Cluster {cluster + 1} - All Neurons'}]
#     ))
#
#     # 更新图表布局并添加按钮
#     fig.update_layout(
#         title=f'Cluster {cluster + 1} Neuron Activity Over Time',
#         xaxis_title='Time Points',
#         yaxis_title='Calcium Ion Concentration',
#         showlegend=True,
#         updatemenus=[dict(
#             type='dropdown',
#             showactive=True,
#             buttons=buttons,
#             x=1.15,
#             y=0.5
#         )]
#     )
#
#     fig.show()

#  绘制典型曲线初始化图形对象
fig = go.Figure()

num_clusters = 5
time_points = calcium_data.shape[0]  # 时间点数
# 为每个类别计算典型曲线，并添加到图中
for cluster in range(num_clusters):
    # 获取属于当前类别的神经元索引
    neuron_indices = np.where(labels == cluster)[0]

    # 计算该类别的典型曲线（平均值）
    typical_curve = np.mean(calcium_data[:, neuron_indices], axis=1)

    # 添加典型曲线到图中
    fig.add_trace(
        go.Scatter(
            x=np.arange(time_points),
            y=typical_curve,
            mode='lines',
            name=f'Typical Curve for Cluster {cluster + 1}',
            line=dict(width=2.5),  # 加深曲线颜色并增粗线条
            opacity=0.9
        )
    )

# 添加按钮来控制每条曲线的显示状态
buttons = []
for cluster in range(num_clusters):
    # 每个按钮控制一个类别的典型曲线的显示
    visibility = [False] * num_clusters  # 默认所有曲线不可见
    visibility[cluster] = True  # 仅显示选中的典型曲线

    buttons.append(dict(
        label=f'Cluster {cluster + 1}',
        method='update',
        args=[{'visible': visibility},  # 更新曲线的可见性
              {'title': f'Typical Curve for Cluster {cluster + 1}'}]
    ))

# 添加“显示所有典型曲线”按钮
buttons.append(dict(
    label='Show All',
    method='update',
    args=[{'visible': [True] * num_clusters},
          {'title': 'All Clusters - Typical Curves'}]
))

# 更新图表布局并添加交互按钮
fig.update_layout(
    title='Typical Curves for Each Cluster',
    xaxis_title='Time Points',
    yaxis_title='Calcium Ion Concentration',
    showlegend=True,
    updatemenus=[dict(
        type='dropdown',
        showactive=True,
        buttons=buttons,
        x=1.15,
        y=0.5
    )]
)

fig.show()