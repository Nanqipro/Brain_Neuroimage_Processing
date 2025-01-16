# import pandas as pd
# from sklearn.cluster import DBSCAN
#
# # 加载数据
# file_path = './data/Day6_Neuron_Calcium_Metrics.xlsx'  # 请替换为实际文件路径
# metrics_df = pd.read_excel(file_path, sheet_name='Windows100_step10')
#
# # 定义用于聚类的特征列
# features = ['Start Time', 'Amplitude', 'Peak', 'Decay Time', 'Rise Time', 'Latency', 'Frequency']
# X = metrics_df[features].values
#
# # 设置 DBSCAN 的参数并执行聚类
# dbscan = DBSCAN(eps=0.5, min_samples=5)  # 可根据数据特性调整 eps 和 min_samples
# metrics_df['DBSCAN'] = dbscan.fit_predict(X)
#
# # 将结果保存回原文件
# metrics_df.to_excel(file_path, index=False, sheet_name='Windows100_step10')
#
# print(f'DBSCAN 聚类结果已保存至原文件的 "DBSCAN" 列中: {file_path}')

# V1 优化版本
import pandas as pd
import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

# 加载数据
file_path = './data/Day6_Neuron_Calcium_Metrics.xlsx'  # 请替换为实际文件路径
metrics_df = pd.read_excel(file_path, sheet_name='Windows100_step10')

# 定义用于聚类的特征列
features = ['Start Time', 'Amplitude', 'Peak', 'Decay Time', 'Rise Time', 'Latency', 'Frequency']

# 检查并处理缺失值
if metrics_df[features].isnull().values.any():
    print("数据包含缺失值，正在处理...")
    # 方法1：删除包含缺失值的行
    metrics_df = metrics_df.dropna(subset=features).reset_index(drop=True)
    # 方法2：填充缺失值（例如使用均值）
    # metrics_df[features] = metrics_df[features].fillna(metrics_df[features].mean())

# 准备数据
X = metrics_df[features]

# 特征标准化
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 定义特征的权重（根据需要调整权重值）
weights = {
    'Start Time': 0.1,
    'Amplitude': 2.0,
    'Peak': 2.0,
    'Decay Time': 1.0,
    'Rise Time': 1.0,
    'Latency': 1.5,
    'Frequency': 1.5
}

# 将权重应用于标准化后的特征
weight_values = np.array([weights[feature] for feature in features])
X_weighted = X_scaled * weight_values

# 使用 KNN 方法确定最佳的 eps 值
# 计算每个点的 k 距离（k 为 min_samples）
min_samples = 5  # 您可以根据需要调整 min_samples
neigh = NearestNeighbors(n_neighbors=min_samples)
nbrs = neigh.fit(X_weighted)
distances, indices = nbrs.kneighbors(X_weighted)

# 对距离排序
distances = np.sort(distances[:, min_samples - 1], axis=0)

# 绘制 k 距离图，帮助确定最佳 eps 值
plt.figure(figsize=(8, 4))
plt.plot(distances)
plt.xlabel('样本点索引')
plt.ylabel(f'{min_samples} 阶距离')
plt.title('k 距离图用于确定最佳 eps 值')
plt.show()

# 根据 k 距离图选择 eps 值（例如，通过观察曲线的“拐点”）
eps = 0.5  # 请根据您的数据和 k 距离图调整 eps 值

# 使用不同的 eps 值计算轮廓系数，确定最佳参数
eps_values = np.linspace(0.1, 1.0, 10)
silhouette_scores = []
for eps in eps_values:
    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    labels = dbscan.fit_predict(X_weighted)
    # 只计算有意义的聚类（标签不全为 -1）
    if len(set(labels)) > 1 and -1 not in set(labels):
        score = silhouette_score(X_weighted, labels)
        silhouette_scores.append(score)
    else:
        silhouette_scores.append(-1)

# 绘制 eps 值与轮廓系数的关系
plt.figure(figsize=(8, 4))
plt.plot(eps_values, silhouette_scores, 'bo-')
plt.xlabel('eps 值')
plt.ylabel('平均轮廓系数')
plt.title('选择最佳 eps 值')
plt.show()

# 根据轮廓系数选择最佳 eps 值
optimal_eps = eps_values[np.argmax(silhouette_scores)]
print(f'最佳 eps 值为: {optimal_eps}')

# 使用最佳参数进行 DBSCAN 聚类
dbscan = DBSCAN(eps=optimal_eps, min_samples=min_samples)
metrics_df['DBSCAN'] = dbscan.fit_predict(X_weighted)

# 将结果保存回原文件，避免覆盖原始数据
# output_file_path = './data/Day6_Neuron_Calcium_Metrics_with_DBSCAN_clusters.xlsx'
metrics_df.to_excel(file_path, index=False, sheet_name='Windows100_step10')

print(f'DBSCAN 聚类结果已保存至文件: {file_path}')
