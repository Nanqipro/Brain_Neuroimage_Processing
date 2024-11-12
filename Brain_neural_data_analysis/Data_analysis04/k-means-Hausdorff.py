import pandas as pd
import numpy as np
from scipy.spatial.distance import cdist

# 加载数据
file_path = './data/Day6_Neuron_Calcium_Metrics.csv'  # 请替换为实际文件路径
metrics_df = pd.read_csv(file_path)

# 定义用于聚类的特征列
features = ['Start Time', 'Amplitude', 'Peak', 'Decay Time', 'Rise Time', 'Latency', 'Frequency']
X = metrics_df[features].values


# 自定义函数：计算 Hausdorff 距离
def hausdorff_distance(point_set1, point_set2):
    # 确保输入为二维数组
    point_set1 = np.atleast_2d(point_set1)
    point_set2 = np.atleast_2d(point_set2)
    distances_1_to_2 = cdist(point_set1, point_set2, metric='euclidean').min(axis=1)
    distances_2_to_1 = cdist(point_set2, point_set1, metric='euclidean').min(axis=1)
    return max(distances_1_to_2.max(), distances_2_to_1.max())


# 自定义函数：生成 Hausdorff 距离矩阵
def compute_hausdorff_distance_matrix(X):
    n_samples = X.shape[0]
    distance_matrix = np.zeros((n_samples, n_samples))
    for i in range(n_samples):
        for j in range(i + 1, n_samples):
            distance = hausdorff_distance(X[i], X[j])
            distance_matrix[i, j] = distance
            distance_matrix[j, i] = distance
    return distance_matrix


# 自定义函数：基于 Hausdorff 距离的 K-means 算法
def k_means_hausdorff(X, n_clusters, max_iters=100, tol=1e-4, random_state=None):
    if random_state:
        np.random.seed(random_state)

    n_samples = X.shape[0]
    # 随机初始化质心（通过选择随机索引）
    initial_centroids_indices = np.random.choice(n_samples, n_clusters, replace=False)
    centroids = X[initial_centroids_indices]

    # 初始化标签
    labels = np.zeros(n_samples, dtype=int)
    for iteration in range(max_iters):
        # 第一步：计算每个样本到每个质心的 Hausdorff 距离
        distance_matrix = np.zeros((n_samples, n_clusters))
        for i in range(n_samples):
            for j in range(n_clusters):
                distance_matrix[i, j] = hausdorff_distance(X[i], centroids[j])

        # 第二步：根据最近的质心分配标签
        new_labels = np.argmin(distance_matrix, axis=1)

        # 第三步：检查收敛
        if np.array_equal(new_labels, labels):
            break
        labels = new_labels

        # 第四步：更新质心（计算每个簇中点的均值）
        for k in range(n_clusters):
            cluster_points = X[labels == k]
            if len(cluster_points) > 0:
                centroids[k] = np.mean(cluster_points, axis=0)

    return labels


# 使用自定义的 Hausdorff-K-means 进行聚类
n_clusters = 5
random_state = 0
metrics_df['k-means-Hausdorff'] = k_means_hausdorff(X, n_clusters, random_state=random_state)

# 将聚类结果保存至原文件
metrics_df.to_csv(file_path, index=False)

print(f'聚类结果已保存至原文件的 "k-means-Hausdorff" 列中: {file_path}')
