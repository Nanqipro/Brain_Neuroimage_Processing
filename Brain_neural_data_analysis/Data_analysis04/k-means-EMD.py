# import pandas as pd
# import numpy as np
# from scipy.stats import wasserstein_distance
#
# # 加载数据
# file_path = './data/Day6_Neuron_Calcium_Metrics.xlsx'  # 请替换为实际文件路径
# metrics_df = pd.read_excel(file_path)
#
# # 定义用于聚类的特征列
# features = ['Start Time', 'Amplitude', 'Peak', 'Decay Time', 'Rise Time', 'Latency', 'Frequency']
# X = metrics_df[features].values
#
#
# # 自定义函数：计算 EMD 距离矩阵
# def compute_emd_distance_matrix(X):
#     n_samples = X.shape[0]
#     distance_matrix = np.zeros((n_samples, n_samples))
#     for i in range(n_samples):
#         for j in range(i + 1, n_samples):
#             distance = wasserstein_distance(X[i], X[j])
#             distance_matrix[i, j] = distance
#             distance_matrix[j, i] = distance
#     return distance_matrix
#
#
# # 自定义函数：基于 EMD 距离的 K-means 算法
# def k_means_emd(X, n_clusters, max_iters=100, tol=1e-4, random_state=None):
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
#         # 第一步：计算每个样本到每个质心的距离（使用 EMD 距离）
#         distance_matrix = np.zeros((n_samples, n_clusters))
#         for i in range(n_samples):
#             for j in range(n_clusters):
#                 distance_matrix[i, j] = wasserstein_distance(X[i], centroids[j])
#
#         # 第二步：根据最近的质心分配标签
#         new_labels = np.argmin(distance_matrix, axis=1)
#
#         # 第三步：检查收敛
#         if np.array_equal(new_labels, labels):
#             break
#         labels = new_labels
#
#         # 第四步：更新质心（计算每个簇中点的均值）
#         for k in range(n_clusters):
#             cluster_points = X[labels == k]
#             if len(cluster_points) > 0:
#                 centroids[k] = np.mean(cluster_points, axis=0)
#
#     return labels
#
#
# # 使用自定义的 EMD-K-means 进行聚类
# n_clusters = 5
# random_state = 0
# metrics_df['k-means-EMD'] = k_means_emd(X, n_clusters, random_state=random_state)
#
# # 将聚类结果保存至原文件
# metrics_df.to_excel(file_path, index=False,sheet_name='Windows100_step10')
#
# print(f'聚类结果已保存至原文件的 "k-means-EMD" 列中: {file_path}')

# # V1 优化版本
# import pandas as pd
# import numpy as np
# from scipy.stats import wasserstein_distance
# from sklearn.preprocessing import MinMaxScaler
# import matplotlib.pyplot as plt
# from sklearn.metrics import silhouette_score
# from sklearn.metrics import pairwise_distances
# import warnings
# warnings.filterwarnings("ignore")
#
# # 加载数据
# file_path = './data/Day6_Neuron_Calcium_Metrics.xlsx'  # 请替换为实际文件路径
# metrics_df = pd.read_excel(file_path,sheet_name='Windows100_step10')
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
# # 特征标准化（使用 MinMaxScaler 以适用于 EMD 距离）
# scaler = MinMaxScaler()
# X_scaled = scaler.fit_transform(X)
#
# # 定义特征的权重（根据需要调整权重值）
# weights = {
#     'Start Time': 0.0,
#     'Amplitude': 1.0,
#     'Peak': 1.0,
#     'Decay Time': 1.0,
#     'Rise Time': 1.0,
#     'Latency': 0.8,
#     'Frequency': 0.8
# }
#
# # 将权重应用于特征
# weight_values = np.array([weights[feature] for feature in features])
# X_weighted = X_scaled * weight_values
#
# # 将数据转换为 NumPy 数组
# # X_weighted = X_weighted.values
#
# # 自定义函数：基于 EMD 距离的 K-means 算法
# def k_means_emd(X, n_clusters, max_iters=100, tol=1e-4, random_state=None):
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
#         # 第一步：计算每个样本到每个质心的距离（使用 EMD 距离）
#         distance_matrix = np.zeros((n_samples, n_clusters))
#         for i in range(n_samples):
#             for j in range(n_clusters):
#                 distance_matrix[i, j] = wasserstein_distance(X[i], centroids[j])
#
#         # 第二步：根据最近的质心分配标签
#         new_labels = np.argmin(distance_matrix, axis=1)
#
#         # 第三步：检查收敛
#         if np.array_equal(new_labels, labels):
#             print(f"算法在第 {iteration+1} 次迭代后收敛。")
#             break
#         labels = new_labels
#
#         # 第四步：更新质心（计算每个簇中点的均值）
#         for k in range(n_clusters):
#             cluster_points = X[labels == k]
#             if len(cluster_points) > 0:
#                 centroids[k] = np.mean(cluster_points, axis=0)
#
#     return labels
#
# # 自定义函数：计算簇内距离总和（用于肘部法）
# def compute_wcsd(X, labels):
#     n_clusters = np.unique(labels).size
#     total_distance = 0.0
#     for k in range(n_clusters):
#         cluster_points = X[labels == k]
#         centroid = np.mean(cluster_points, axis=0)
#         # 计算簇内点到质心的距离之和
#         for point in cluster_points:
#             total_distance += wasserstein_distance(point, centroid)
#     return total_distance
#
# # 使用肘部法确定最佳聚类数量
# wcsd = []
# K = range(1, 6)  # 由于计算量大，这里选择较小的范围
# for k in K:
#     print(f"正在计算 {k} 个簇的聚类...")
#     labels = k_means_emd(X_weighted, n_clusters=k, random_state=0)
#     total_distance = compute_wcsd(X_weighted, labels)
#     wcsd.append(total_distance)
#
# # 绘制肘部法图形
# plt.figure(figsize=(8, 4))
# plt.plot(K, wcsd, 'bo-')
# plt.xlabel('cluster number k')
# plt.ylabel('Within - Cluster Sum of Distances(WCSD)')
# plt.title('elbow method')
# plt.xticks(K)
# plt.show()
#
# # 使用轮廓系数法确定最佳聚类数量
# # 注意：由于 EMD 距离计算复杂，数据量大时计算全距离矩阵可能不可行
# # 这里提供一个简化的版本，只计算部分样本的轮廓系数
#
# from sklearn.metrics import silhouette_samples
#
# silhouette_scores = []
# K = range(2, 6)
# for k in K:
#     print(f"正在计算 {k} 个簇的轮廓系数...")
#     labels = k_means_emd(X_weighted, n_clusters=k, random_state=0)
#     # 计算每个样本的轮廓系数
#     sample_silhouette_values = []
#     for i in range(len(X_weighted)):
#         own_cluster = labels[i]
#         own_cluster_points = X_weighted[labels == own_cluster]
#         other_clusters = np.unique(labels[labels != own_cluster])
#         if len(other_clusters) == 0:
#             continue
#         # 计算 a(i)
#         a_i = np.mean([wasserstein_distance(X_weighted[i], point) for point in own_cluster_points if not np.array_equal(point, X_weighted[i])])
#         # 计算 b(i)
#         b_i = np.min([
#             np.mean([wasserstein_distance(X_weighted[i], point) for point in X_weighted[labels == other_cluster]])
#             for other_cluster in other_clusters
#         ])
#         # 计算轮廓系数
#         s_i = (b_i - a_i) / max(a_i, b_i)
#         sample_silhouette_values.append(s_i)
#     # 计算平均轮廓系数
#     silhouette_avg = np.mean(sample_silhouette_values)
#     silhouette_scores.append(silhouette_avg)
#
# # 绘制轮廓系数图形
# plt.figure(figsize=(8, 4))
# plt.plot(K, silhouette_scores, 'bo-')
# plt.xlabel('elbow method')
# plt.ylabel('Average Silhouette Coefficient')
# plt.title('Silhouette Coefficient')
# plt.xticks(K)
# plt.show()
#
# # 根据肘部法和轮廓系数法选择最佳聚类数量，例如这里选择 k=3
# optimal_k = 3
#
# # 使用最佳聚类数量进行最终聚类
# labels = k_means_emd(X_weighted, n_clusters=optimal_k, random_state=0)
# metrics_df['k-means-EMD'] = labels
#
# # 将聚类结果保存至新文件，避免覆盖原始文件
# # output_file_path = './data/Day6_Neuron_Calcium_Metrics_with_EMD_clusters.xlsx'
# metrics_df.to_excel(file_path, index=False, sheet_name='Windows100_step10')
#
# print(f'聚类结果已保存至文件: {file_path}')

# V2 加速计算
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score
from numba import njit, prange
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
X = metrics_df[features].values

# 特征标准化（使用 MinMaxScaler 以适用于 EMD 距离）
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

# 定义特征的权重（根据需要调整权重值）
weights = {
    'Start Time': 0.0,
    'Amplitude': 1.0,
    'Peak': 1.0,
    'Decay Time': 1.0,
    'Rise Time': 1.0,
    'Latency': 0.8,
    'Frequency': 0.8
}

# 将权重应用于特征
weight_values = np.array([weights[feature] for feature in features])
X_weighted = X_scaled * weight_values

# 自定义 Numba 加速的 EMD 计算函数
@njit
def emd_distance(u_values, v_values):
    # 计算两个向量之间的一维 EMD（假设数据已排序）
    u_sorted = np.sort(u_values)
    v_sorted = np.sort(v_values)
    distance = np.sum(np.abs(np.cumsum(u_sorted - v_sorted)))
    return distance

# 自定义函数：基于 EMD 距离的 K-means 算法（使用 Numba 加速）
def k_means_emd(X, n_clusters, max_iters=100, tol=1e-4, random_state=None):
    if random_state is not None:
        np.random.seed(random_state)

    n_samples, n_features = X.shape
    # 随机初始化质心
    initial_centroids_indices = np.random.choice(n_samples, n_clusters, replace=False)
    centroids = X[initial_centroids_indices].copy()

    # 初始化标签
    labels = np.full(n_samples, -1)
    for iteration in range(max_iters):
        # 第一步：计算距离矩阵（使用 Numba 加速）
        distance_matrix = np.zeros((n_samples, n_clusters))
        for i in prange(n_samples):
            for j in range(n_clusters):
                distance_matrix[i, j] = emd_distance(X[i], centroids[j])

        # 第二步：根据最近的质心分配标签
        new_labels = np.argmin(distance_matrix, axis=1)

        # 第三步：检查收敛
        if np.array_equal(new_labels, labels):
            print(f"算法在第 {iteration+1} 次迭代后收敛。")
            break
        labels = new_labels

        # 第四步：更新质心
        for k in range(n_clusters):
            cluster_points = X[labels == k]
            if len(cluster_points) > 0:
                centroids[k] = np.mean(cluster_points, axis=0)

    return labels

# 自定义函数：计算簇内距离总和（用于肘部法，使用 Numba 加速）
@njit
def compute_wcsd(X, labels, centroids):
    total_distance = 0.0
    for k in prange(len(centroids)):
        cluster_points = X[labels == k]
        centroid = centroids[k]
        for i in range(cluster_points.shape[0]):
            total_distance += emd_distance(cluster_points[i], centroid)
    return total_distance

# 使用肘部法确定最佳聚类数量
wcsd = []
K = range(1, 6)  # 由于计算量大，这里选择较小的范围
for k in K:
    print(f"正在计算 {k} 个簇的聚类...")
    labels = k_means_emd(X_weighted, n_clusters=k, random_state=0)
    centroids = np.array([X_weighted[labels == i].mean(axis=0) for i in range(k)])
    total_distance = compute_wcsd(X_weighted, labels, centroids)
    wcsd.append(total_distance)

# 绘制肘部法图形
plt.figure(figsize=(8, 4))
plt.plot(K, wcsd, 'bo-')
plt.xlabel('聚类数量 k')
plt.ylabel('簇内距离总和（WCSD）')
plt.title('肘部法确定最佳聚类数量')
plt.xticks(K)
plt.show()

# 使用轮廓系数法确定最佳聚类数量
# 注意：由于 EMD 距离计算复杂，这里对轮廓系数计算进行简化
silhouette_scores = []
K = range(2, 6)
for k in K:
    print(f"正在计算 {k} 个簇的轮廓系数...")
    labels = k_means_emd(X_weighted, n_clusters=k, random_state=0)
    # 计算轮廓系数（抽样计算）
    sample_indices = np.random.choice(len(X_weighted), size=min(100, len(X_weighted)), replace=False)
    sample_X = X_weighted[sample_indices]
    sample_labels = labels[sample_indices]
    # 定义自定义距离矩阵
    distance_matrix = np.zeros((len(sample_X), len(sample_X)))
    for i in prange(len(sample_X)):
        for j in range(i + 1, len(sample_X)):
            dist = emd_distance(sample_X[i], sample_X[j])
            distance_matrix[i, j] = dist
            distance_matrix[j, i] = dist
    silhouette_avg = silhouette_score(distance_matrix, sample_labels, metric='precomputed')
    silhouette_scores.append(silhouette_avg)

# 绘制轮廓系数图形
plt.figure(figsize=(8, 4))
plt.plot(K, silhouette_scores, 'bo-')
plt.xlabel('聚类数量 k')
plt.ylabel('平均轮廓系数')
plt.title('轮廓系数法确定最佳聚类数量')
plt.xticks(K)
plt.show()

# 根据肘部法和轮廓系数法选择最佳聚类数量，例如这里选择 k=3
optimal_k = 6

# 使用最佳聚类数量进行最终聚类
labels = k_means_emd(X_weighted, n_clusters=optimal_k, random_state=0)
metrics_df['k-means-EMD'] = labels

# 将聚类结果保存至新文件，避免覆盖原始文件
# output_file_path = './data/Day6_Neuron_Calcium_Metrics_with_EMD_clusters.xlsx'
metrics_df.to_excel(file_path, index=False, sheet_name='Windows100_step10')

print(f'聚类结果已保存至文件: {file_path}')
