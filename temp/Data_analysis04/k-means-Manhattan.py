# import pandas as pd
# import numpy as np
#
# # 加载数据
# file_path = './data/Day6_Neuron_Calcium_Metrics.xlsx'  # 请替换为实际文件路径
# metrics_df = pd.read_excel(file_path,sheet_name='Windows100_step10')
#
# # 定义用于聚类的特征列
# features = ['Start Time', 'Amplitude', 'Peak', 'Decay Time', 'Rise Time', 'Latency', 'Frequency']
# X = metrics_df[features].values
#
# # 自定义函数：计算曼哈顿距离
# def manhattan_distance(point1, point2):
#     return np.sum(np.abs(point1 - point2))
#
# # 自定义函数：基于曼哈顿距离的 K-means 算法
# def k_means_manhattan(X, n_clusters, max_iters=100, tol=1e-4, random_state=None):
#     if random_state:
#         np.random.seed(random_state)
#
#     n_samples = X.shape[0]
#     # 随机初始化质心（通过选择随机索引）
#     initial_centroids_indices = np.random.choice(n_samples, n_clusters, replace=False)
#     centroids = X[initial_centroids_indices]
#
#     # 初始化标签
#     labels = np.zeros(n_samples, dtype=int)
#     for iteration in range(max_iters):
#         # 第一步：计算每个样本到每个质心的曼哈顿距离
#         distance_matrix = np.zeros((n_samples, n_clusters))
#         for i in range(n_samples):
#             for j in range(n_clusters):
#                 distance_matrix[i, j] = manhattan_distance(X[i], centroids[j])
#
#         # 第二步：根据最近的质心分配标签
#         new_labels = np.argmin(distance_matrix, axis=1)
#
#         # 第三步：检查收敛（标签是否不再变化）
#         if np.array_equal(new_labels, labels):
#             break
#         labels = new_labels
#
#         # 第四步：更新质心（计算每个簇中点的中位数作为新的质心）
#         for k in range(n_clusters):
#             cluster_points = X[labels == k]
#             if len(cluster_points) > 0:
#                 centroids[k] = np.median(cluster_points, axis=0)
#
#     return labels
#
# # 使用自定义的曼哈顿距离 K-means 进行聚类
# n_clusters = 5
# random_state = 0
# metrics_df['k-means-Manhattan'] = k_means_manhattan(X, n_clusters, random_state=random_state)
#
# # 将聚类结果保存至原文件
# metrics_df.to_excel(file_path, index=False, sheet_name='Windows100_step10')
#
# print(f'聚类结果已保存至原文件的 "k-means-Manhattan" 列中: {file_path}')


# # V1 优化版本
# import pandas as pd
# import numpy as np
# from sklearn.preprocessing import StandardScaler
# from sklearn.metrics import silhouette_score
# import matplotlib.pyplot as plt
# import warnings
# warnings.filterwarnings("ignore")
#
# # 加载数据
# file_path = './data/Day6_Neuron_Calcium_Metrics.xlsx'  # 请替换为实际文件路径
# metrics_df = pd.read_excel(file_path, sheet_name='Windows100_step10')
#
# # 定义用于聚类的特征列
# features = ['Start Time', 'Amplitude', 'Peak', 'Decay Time', 'Rise Time', 'Latency', 'Frequency']
#
# # 检查并处理缺失值
# if metrics_df[features].isnull().values.any():
#     print("数据包含缺失值，正在处理...")
#     # 方法1：删除包含缺失值的行
#     metrics_df = metrics_df.dropna(subset=features).reset_index(drop=True)
#     # 方法2：填充缺失值（例如使用均值）
#     # metrics_df[features] = metrics_df[features].fillna(metrics_df[features].mean())
#
# # 准备数据
# X = metrics_df[features]
#
# # 特征标准化
# scaler = StandardScaler()
# X_scaled = scaler.fit_transform(X)
#
# # 定义特征的权重（根据需要调整权重值）
# weights = {
#     'Start Time': 0.1,
#     'Amplitude': 2.0,
#     'Peak': 2.0,
#     'Decay Time': 1.0,
#     'Rise Time': 1.0,
#     'Latency': 1.5,
#     'Frequency': 1.5
# }
#
# # 将权重应用于标准化后的特征
# weight_values = np.array([weights[feature] for feature in features])
# X_weighted = X_scaled * weight_values
#
# # 自定义函数：计算曼哈顿距离
# def manhattan_distance(point1, point2):
#     return np.sum(np.abs(point1 - point2))
#
# # 自定义函数：基于曼哈顿距离的 K-means 算法
# def k_means_manhattan(X, n_clusters, max_iters=100, tol=1e-4, random_state=None):
#     if random_state is not None:
#         np.random.seed(random_state)
#
#     n_samples = X.shape[0]
#     # 随机初始化质心（通过选择随机索引）
#     initial_centroids_indices = np.random.choice(n_samples, n_clusters, replace=False)
#     centroids = X[initial_centroids_indices]
#
#     # 初始化标签
#     labels = np.full(n_samples, -1)
#     for iteration in range(max_iters):
#         # 第一步：计算每个样本到每个质心的曼哈顿距离
#         distance_matrix = np.zeros((n_samples, n_clusters))
#         for i in range(n_samples):
#             for j in range(n_clusters):
#                 distance_matrix[i, j] = manhattan_distance(X[i], centroids[j])
#
#         # 第二步：根据最近的质心分配标签
#         new_labels = np.argmin(distance_matrix, axis=1)
#
#         # 第三步：检查收敛（标签是否不再变化）
#         if np.array_equal(new_labels, labels):
#             print(f"算法在第 {iteration+1} 次迭代后收敛。")
#             break
#         labels = new_labels
#
#         # 第四步：更新质心（计算每个簇中点的中位数作为新的质心）
#         for k in range(n_clusters):
#             cluster_points = X[labels == k]
#             if len(cluster_points) > 0:
#                 centroids[k] = np.median(cluster_points, axis=0)
#
#     return labels
#
# # 使用肘部法确定最佳聚类数量
# wcss = []
# K = range(1, 11)
# for k in K:
#     print(f"正在计算 {k} 个簇的聚类...")
#     labels = k_means_manhattan(X_weighted, n_clusters=k, random_state=0)
#     # 计算簇内平方和（WCSS）
#     total_wcss = 0
#     for i in range(k):
#         cluster_points = X_weighted[labels == i]
#         centroid = np.median(cluster_points, axis=0)
#         total_wcss += np.sum(np.abs(cluster_points - centroid).sum(axis=1))
#     wcss.append(total_wcss)
#
# # 绘制肘部法图形
# plt.figure(figsize=(8, 4))
# plt.plot(K, wcss, 'bo-')
# plt.xlabel('聚类数量 k')
# plt.ylabel('簇内距离总和（Total Within-Cluster Sum of Distances）')
# plt.title('肘部法确定最佳聚类数量')
# plt.xticks(K)
# plt.show()
#
# # 使用轮廓系数法确定最佳聚类数量
# silhouette_scores = []
# K = range(2, 11)
# for k in K:
#     print(f"正在计算 {k} 个簇的轮廓系数...")
#     labels = k_means_manhattan(X_weighted, n_clusters=k, random_state=0)
#     # 计算轮廓系数
#     silhouette_avg = silhouette_score(X_weighted, labels, metric='manhattan')
#     silhouette_scores.append(silhouette_avg)
#
# # 绘制轮廓系数图形
# plt.figure(figsize=(8, 4))
# plt.plot(K, silhouette_scores, 'bo-')
# plt.xlabel('聚类数量 k')
# plt.ylabel('平均轮廓系数')
# plt.title('轮廓系数法确定最佳聚类数量')
# plt.xticks(K)
# plt.show()
#
# # 根据肘部法和轮廓系数法选择最佳聚类数量，例如这里选择 k=4
# optimal_k = 4
#
# # 使用最佳聚类数量进行最终聚类
# labels = k_means_manhattan(X_weighted, n_clusters=optimal_k, random_state=0)
# metrics_df['k-means-Manhattan'] = labels
#
# # 将聚类结果保存至新文件，避免覆盖原始文件
# # output_file_path = './data/Day6_Neuron_Calcium_Metrics_with_Manhattan_clusters.xlsx'
# metrics_df.to_excel(file_path, index=False, sheet_name='Windows100_step10')
#
# print(f'聚类结果已保存至文件: {file_path}')


# V2 加速
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from scipy.spatial.distance import cdist
import matplotlib.pyplot as plt
from numba import njit
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

# 自定义函数：基于曼哈顿距离的 K-means 算法（使用矢量化和 Numba 加速）
def k_means_manhattan(X, n_clusters, max_iters=100, tol=1e-4, random_state=None):
    if random_state is not None:
        np.random.seed(random_state)

    n_samples, n_features = X.shape
    # 随机初始化质心（通过选择随机索引）
    initial_centroids_indices = np.random.choice(n_samples, n_clusters, replace=False)
    centroids = X[initial_centroids_indices]

    # 初始化标签
    labels = np.full(n_samples, -1)
    for iteration in range(max_iters):
        # 第一步：计算每个样本到每个质心的曼哈顿距离（使用矢量化计算）
        distance_matrix = cdist(X, centroids, metric='cityblock')

        # 第二步：根据最近的质心分配标签
        new_labels = np.argmin(distance_matrix, axis=1)

        # 第三步：检查收敛（标签是否不再变化）
        if np.array_equal(new_labels, labels):
            print(f"算法在第 {iteration+1} 次迭代后收敛。")
            break
        labels = new_labels

        # 第四步：更新质心（计算每个簇中点的中位数作为新的质心）
        for k in range(n_clusters):
            cluster_points = X[labels == k]
            if len(cluster_points) > 0:
                centroids[k] = np.median(cluster_points, axis=0)

    return labels

# 使用肘部法确定最佳聚类数量
wcss = []
K = range(1, 11)
for k in K:
    print(f"正在计算 {k} 个簇的聚类...")
    labels = k_means_manhattan(X_weighted, n_clusters=k, random_state=0)
    # 计算簇内距离总和（Total Within-Cluster Sum of Distances）
    total_wcss = 0
    for i in range(k):
        cluster_points = X_weighted[labels == i]
        centroid = np.median(cluster_points, axis=0)
        total_wcss += np.sum(np.abs(cluster_points - centroid))
    wcss.append(total_wcss)

# 绘制肘部法图形
plt.figure(figsize=(8, 4))
plt.plot(K, wcss, 'bo-')
plt.xlabel('聚类数量 k')
plt.ylabel('簇内距离总和（Total Within-Cluster Sum of Distances）')
plt.title('肘部法确定最佳聚类数量')
plt.xticks(K)
plt.show()

# 使用轮廓系数法确定最佳聚类数量
silhouette_scores = []
K = range(2, 11)
for k in K:
    print(f"正在计算 {k} 个簇的轮廓系数...")
    labels = k_means_manhattan(X_weighted, n_clusters=k, random_state=0)
    # 计算轮廓系数
    silhouette_avg = silhouette_score(X_weighted, labels, metric='manhattan')
    silhouette_scores.append(silhouette_avg)

# 绘制轮廓系数图形
plt.figure(figsize=(8, 4))
plt.plot(K, silhouette_scores, 'bo-')
plt.xlabel('聚类数量 k')
plt.ylabel('平均轮廓系数')
plt.title('轮廓系数法确定最佳聚类数量')
plt.xticks(K)
plt.show()

# 根据肘部法和轮廓系数法选择最佳聚类数量，例如这里选择 k=4
optimal_k = 6

# 使用最佳聚类数量进行最终聚类
labels = k_means_manhattan(X_weighted, n_clusters=optimal_k, random_state=0)
metrics_df['k-means-Manhattan'] = labels

# 将聚类结果保存至新文件，避免覆盖原始文件
# output_file_path = './data/Day6_Neuron_Calcium_Metrics_with_Manhattan_clusters.xlsx'
metrics_df.to_excel(file_path, index=False, sheet_name='Windows100_step10')

print(f'聚类结果已保存至文件: {file_path}')
