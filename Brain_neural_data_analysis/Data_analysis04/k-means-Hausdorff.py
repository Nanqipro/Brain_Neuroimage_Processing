# import pandas as pd
# import numpy as np
# from scipy.spatial.distance import cdist
#
# # 加载数据
# file_path = './data/Day6_Neuron_Calcium_Metrics.xlsx'  # 请替换为实际文件路径
# metrics_df = pd.read_excel(file_path, sheet_name='Windows100_step10')
#
# # 定义用于聚类的特征列
# features = ['Start Time', 'Amplitude', 'Peak', 'Decay Time', 'Rise Time', 'Latency', 'Frequency']
# X = metrics_df[features].values
#
#
# # 自定义函数：计算 Hausdorff 距离
# def hausdorff_distance(point_set1, point_set2):
#     # 确保输入为二维数组
#     point_set1 = np.atleast_2d(point_set1)
#     point_set2 = np.atleast_2d(point_set2)
#     distances_1_to_2 = cdist(point_set1, point_set2, metric='euclidean').min(axis=1)
#     distances_2_to_1 = cdist(point_set2, point_set1, metric='euclidean').min(axis=1)
#     return max(distances_1_to_2.max(), distances_2_to_1.max())
#
#
# # 自定义函数：生成 Hausdorff 距离矩阵
# def compute_hausdorff_distance_matrix(X):
#     n_samples = X.shape[0]
#     distance_matrix = np.zeros((n_samples, n_samples))
#     for i in range(n_samples):
#         for j in range(i + 1, n_samples):
#             distance = hausdorff_distance(X[i], X[j])
#             distance_matrix[i, j] = distance
#             distance_matrix[j, i] = distance
#     return distance_matrix
#
#
# # 自定义函数：基于 Hausdorff 距离的 K-means 算法
# def k_means_hausdorff(X, n_clusters, max_iters=100, tol=1e-4, random_state=None):
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
#         # 第一步：计算每个样本到每个质心的 Hausdorff 距离
#         distance_matrix = np.zeros((n_samples, n_clusters))
#         for i in range(n_samples):
#             for j in range(n_clusters):
#                 distance_matrix[i, j] = hausdorff_distance(X[i], centroids[j])
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
# # 使用自定义的 Hausdorff-K-means 进行聚类
# n_clusters = 5
# random_state = 0
# metrics_df['k-means-Hausdorff'] = k_means_hausdorff(X, n_clusters, random_state=random_state)
#
# # 将聚类结果保存至原文件
# metrics_df.to_excel(file_path, index=False,sheet_name='Windows100_step10')
#
# print(f'聚类结果已保存至原文件的 "k-means-Hausdorff" 列中: {file_path}')


# # V1 优化版本
# import pandas as pd
# import numpy as np
# from scipy.spatial.distance import cdist
# import matplotlib.pyplot as plt
# from sklearn.preprocessing import StandardScaler, MinMaxScaler
# from sklearn.metrics import silhouette_score
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
# # 特征标准化（使用 StandardScaler 或 MinMaxScaler）
# scaler = StandardScaler()
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
# # 将权重应用于标准化后的特征
# weight_values = np.array([weights[feature] for feature in features])
# X_weighted = X_scaled * weight_values
#
# # 将数据转换为 NumPy 数组
# X_weighted = X_weighted
#
# # 自定义函数：计算 Hausdorff 距离
# def hausdorff_distance(point1, point2):
#     # 由于每个点都是一个特征向量，这里直接计算欧氏距离作为 Hausdorff 距离的替代
#     return np.linalg.norm(point1 - point2)
#
# # 自定义函数：基于 Hausdorff 距离的 K-means 算法
# def k_means_hausdorff(X, n_clusters, max_iters=100, tol=1e-4, random_state=None):
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
#         # 第一步：计算每个样本到每个质心的 Hausdorff 距离
#         distance_matrix = cdist(X, centroids, metric='euclidean')  # 使用欧氏距离代替
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
# def compute_wcss(X, labels, centroids):
#     wcss = 0.0
#     n_clusters = centroids.shape[0]
#     for k in range(n_clusters):
#         cluster_points = X[labels == k]
#         wcss += np.sum(np.linalg.norm(cluster_points - centroids[k], axis=1) ** 2)
#     return wcss
#
# # 使用肘部法确定最佳聚类数量
# wcss = []
# K = range(1, 11)
# for k in K:
#     print(f"正在计算 {k} 个簇的聚类...")
#     labels = k_means_hausdorff(X_weighted, n_clusters=k, random_state=0)
#     # 计算簇内距离总和
#     centroids = np.array([X_weighted[labels == i].mean(axis=0) for i in range(k)])
#     total_wcss = compute_wcss(X_weighted, labels, centroids)
#     wcss.append(total_wcss)
#
# # 绘制肘部法图形
# plt.figure(figsize=(8, 4))
# plt.plot(K, wcss, 'bo-')
# plt.xlabel('聚类数量 k')
# plt.ylabel('簇内平方和（WCSS）')
# plt.title('肘部法确定最佳聚类数量')
# plt.xticks(K)
# plt.show()
#
# # 使用轮廓系数法确定最佳聚类数量
# silhouette_scores = []
# K = range(2, 11)
# for k in K:
#     print(f"正在计算 {k} 个簇的轮廓系数...")
#     labels = k_means_hausdorff(X_weighted, n_clusters=k, random_state=0)
#     # 计算轮廓系数
#     silhouette_avg = silhouette_score(X_weighted, labels, metric='euclidean')
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
# labels = k_means_hausdorff(X_weighted, n_clusters=optimal_k, random_state=0)
# metrics_df['k-means-Hausdorff'] = labels
#
# # 将聚类结果保存至新文件，避免覆盖原始文件
# # output_file_path = './data/Day6_Neuron_Calcium_Metrics_with_Hausdorff_clusters.xlsx'
# metrics_df.to_excel(file_path, index=False, sheet_name='Windows100_step10')
#
# print(f'聚类结果已保存至文件: {file_path}')

# V2 加速计算
import pandas as pd
import numpy as np
from scipy.spatial.distance import cdist
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
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

# 特征标准化（使用 StandardScaler）
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 定义特征的权重（根据需要调整权重值）
weights = {
    'Start Time': 0.1,
    'Amplitude': 1.0,
    'Peak': 1.0,
    'Decay Time': 1.0,
    'Rise Time': 1.0,
    'Latency': 0.8,
    'Frequency': 0.8
}

# 将权重应用于标准化后的特征
weight_values = np.array([weights[feature] for feature in features])
X_weighted = X_scaled * weight_values

# 转换为 NumPy 数组
X_weighted = np.array(X_weighted)

# 自定义函数：优化的 K-means 算法（使用欧氏距离）
def k_means_euclidean(X, n_clusters, max_iters=100, tol=1e-4, random_state=None):
    if random_state is not None:
        np.random.seed(random_state)

    n_samples, n_features = X.shape
    # 随机初始化质心（通过选择随机索引）
    initial_centroids_indices = np.random.choice(n_samples, n_clusters, replace=False)
    centroids = X[initial_centroids_indices]

    # 初始化标签
    labels = np.full(n_samples, -1)
    for iteration in range(max_iters):
        # 第一步：使用 cdist 计算每个样本到每个质心的距离（矢量化计算）
        distance_matrix = cdist(X, centroids, metric='euclidean')

        # 第二步：根据最近的质心分配标签
        new_labels = np.argmin(distance_matrix, axis=1)

        # 第三步：检查收敛（标签是否不再变化）
        if np.array_equal(new_labels, labels):
            print(f"算法在第 {iteration+1} 次迭代后收敛。")
            break
        labels = new_labels

        # 第四步：更新质心（计算每个簇中点的均值）
        for k in range(n_clusters):
            cluster_points = X[labels == k]
            if len(cluster_points) > 0:
                centroids[k] = cluster_points.mean(axis=0)

    return labels

# 使用 Numba 加速计算簇内平方和的函数
@njit
def compute_wcss_numba(X, labels, centroids):
    wcss = 0.0
    for i in range(len(centroids)):
        cluster_points = X[labels == i]
        diff = cluster_points - centroids[i]
        wcss += np.sum(diff ** 2)
    return wcss

# 使用肘部法确定最佳聚类数量
wcss = []
K = range(1, 11)
for k in K:
    print(f"正在计算 {k} 个簇的聚类...")
    labels = k_means_euclidean(X_weighted, n_clusters=k, random_state=0)
    # 计算簇内平方和
    centroids = np.array([X_weighted[labels == i].mean(axis=0) for i in range(k)])
    total_wcss = compute_wcss_numba(X_weighted, labels, centroids)
    wcss.append(total_wcss)

# 绘制肘部法图形
plt.figure(figsize=(8, 4))
plt.plot(K, wcss, 'bo-')
plt.xlabel('聚类数量 k')
plt.ylabel('簇内平方和（WCSS）')
plt.title('肘部法确定最佳聚类数量')
plt.xticks(K)
plt.show()

# 使用轮廓系数法确定最佳聚类数量
silhouette_scores = []
K = range(2, 11)
for k in K:
    print(f"正在计算 {k} 个簇的轮廓系数...")
    labels = k_means_euclidean(X_weighted, n_clusters=k, random_state=0)
    # 计算轮廓系数
    silhouette_avg = silhouette_score(X_weighted, labels, metric='euclidean')
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
labels = k_means_euclidean(X_weighted, n_clusters=optimal_k, random_state=0)
metrics_df['k-means-Hausdorff'] = labels

# 将聚类结果保存至新文件，避免覆盖原始文件
# output_file_path = './data/Day6_Neuron_Calcium_Metrics_with_Clusters.xlsx'
metrics_df.to_excel(file_path, index=False, sheet_name='Windows100_step10')

print(f'聚类结果已保存至文件: {file_path}')

