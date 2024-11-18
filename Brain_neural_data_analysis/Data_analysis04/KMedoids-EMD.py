import pandas as pd
import numpy as np
from scipy.stats import wasserstein_distance
from sklearn_extra.cluster import KMedoids  # KMedoids 支持自定义距离矩阵

# 加载数据
file_path = './data/Day6_Neuron_Calcium_Metrics.csv'  # 请替换为实际文件路径
metrics_df = pd.read_csv(file_path)

# 定义用于聚类的特征列
features = ['Start Time', 'Amplitude', 'Peak', 'Decay Time', 'Rise Time', 'Latency', 'Frequency']

# 提取特征数据
X = metrics_df[features].values

# 计算 EMD 距离矩阵
n_samples = X.shape[0]
distance_matrix = np.zeros((n_samples, n_samples))

for i in range(n_samples):
    for j in range(i + 1, n_samples):
        distance = wasserstein_distance(X[i], X[j])  # 计算 EMD 距离
        distance_matrix[i, j] = distance
        distance_matrix[j, i] = distance

# 使用 KMedoids 聚类，基于预计算的 EMD 距离矩阵
kmedoids = KMedoids(n_clusters=5, metric="precomputed", random_state=0)
metrics_df['KMedoids-EMD'] = kmedoids.fit_predict(distance_matrix)

# 将聚类结果保存至原文件
metrics_df.to_csv(file_path, index=False)

print(f'聚类结果已保存至原文件的 "KMedoids-EMD" 列中: {file_path}')
