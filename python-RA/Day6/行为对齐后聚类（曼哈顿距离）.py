import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from scipy.spatial.distance import cityblock
import plotly.graph_objects as go

# 读取数据
data = pd.read_excel('aligned_behavior_calcium_data.xlsx')

# 根据行为类型分组
behavior_groups = data.groupby('Behavior')

# 存储每种行为的特征和聚类结果
features = {}
n_clusters = 5  # 聚类数，可以根据需要调整
cluster_results = {}

# 特征提取和聚类
for behavior, group in behavior_groups:
    neuron_data = group.iloc[:, 2:]  # 跳过行为列，获取神经元数据

    # 特征计算
    peak_values = neuron_data.max()
    std_deviation = neuron_data.std()
    duration_above_half_peak = neuron_data.apply(lambda x: np.sum(x > 0.5 * x.max()), axis=0)

    # 组合成特征矩阵
    feature_matrix = pd.DataFrame({
        'Peak Value': peak_values,
        'Std Deviation': std_deviation,
        'Duration Above Half Peak': duration_above_half_peak
    })

    # 填充 NaN 值
    feature_matrix.fillna(feature_matrix.mean(), inplace=True)

    # 计算曼哈顿距离矩阵，并处理 NaN 值
    distance_matrix = np.array([[cityblock(a, b) for b in feature_matrix.values] for a in feature_matrix.values])
    distance_matrix = np.nan_to_num(distance_matrix)  # 替换 NaN 为 0

    # 使用 KMeans 进行聚类
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    labels = kmeans.fit_predict(distance_matrix)

    # 存储聚类结果
    feature_matrix['Cluster'] = labels
    features[behavior] = feature_matrix
    cluster_results[behavior] = labels

    # 生成 Plotly 3D 图
    fig = go.Figure()
    for cluster_label in range(n_clusters):
        cluster_data = feature_matrix[feature_matrix['Cluster'] == cluster_label]
        fig.add_trace(go.Scatter3d(
            x=cluster_data['Peak Value'],
            y=cluster_data['Std Deviation'],
            z=cluster_data['Duration Above Half Peak'],
            mode='markers',
            marker=dict(size=5),
            name=f'Cluster {cluster_label}'
        ))

    # 设置图表布局
    fig.update_layout(
        title=f'3D Clustering of Neuron Activity for {behavior}',
        scene=dict(
            xaxis_title='Peak Value',
            yaxis_title='Std Deviation',
            zaxis_title='Duration Above Half Peak'
        )
    )
    fig.show()
