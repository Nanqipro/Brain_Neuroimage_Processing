import pandas as pd
import numpy as np

# 加载数据
file_path = './data/Day6_Neuron_Calcium_Metrics.csv'  # 请替换为实际文件路径
metrics_df = pd.read_csv(file_path)

# 定义用于聚类的特征列
features = ['Start Time', 'Amplitude', 'Peak', 'Decay Time', 'Rise Time', 'Latency', 'Frequency']
X = metrics_df[features].values


# 自定义函数：计算曼哈顿距离
def manhattan_distance(point1, point2):
    return np.sum(np.abs(point1 - point2))


# 自定义函数：基于曼哈顿距离的 K-means 算法
def k_means_manhattan(X, n_clusters, max_iters=100, tol=1e-4, random_state=None):
    if random_state:
        np.random.seed(random_state)

    n_samples = X.shape[0]
    # 随机初始化质心（通过选择随机索引）
    initial_centroids_indices = np.random.choice(n_samples, n_clusters, replace=False)
    centroids = X[initial_centroids_indices]

    # 初始化标签
    labels = np.zeros(n_samples, dtype=int)
    for iteration in range(max_iters):
        # 第一步：计算每个样本到每个质心的曼哈顿距离
        distance_matrix = np.zeros((n_samples, n_clusters))
        for i in range(n_samples):
            for j in range(n_clusters):
                distance_matrix[i, j] = manhattan_distance(X[i], centroids[j])

        # 第二步：根据最近的质心分配标签
        new_labels = np.argmin(distance_matrix, axis=1)

        # 第三步：检查收敛（标签是否不再变化）
        if np.array_equal(new_labels, labels):
            break
        labels = new_labels

        # 第四步：更新质心（计算每个簇中点的中位数作为新的质心）
        for k in range(n_clusters):
            cluster_points = X[labels == k]
            if len(cluster_points) > 0:
                centroids[k] = np.median(cluster_points, axis=0)

    return labels


# 使用自定义的曼哈顿距离 K-means 进行聚类
n_clusters = 5
random_state = 0
metrics_df['k-means-Manhattan'] = k_means_manhattan(X, n_clusters, random_state=random_state)

# 将聚类结果保存至原文件
metrics_df.to_csv(file_path, index=False)

print(f'聚类结果已保存至原文件的 "k-means-Manhattan" 列中: {file_path}')
