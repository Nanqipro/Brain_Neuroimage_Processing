import pandas as pd
import numpy as np
from sklearn.cluster import SpectralClustering
from numba import jit

# 加载数据
file_path = './data/Day6_Neuron_Calcium_Metrics.csv'  # 请替换为实际文件路径
metrics_df = pd.read_csv(file_path)

# 定义用于聚类的特征列
features = ['Start Time', 'Amplitude', 'Peak', 'Decay Time', 'Rise Time', 'Latency', 'Frequency']
X = metrics_df[features].values

# 使用 Numba 加速自定义相似度矩阵计算
@jit(nopython=True)
def compute_rbf_similarity_matrix(X, gamma=1.0):
    n_samples = X.shape[0]
    similarity_matrix = np.zeros((n_samples, n_samples))
    for i in range(n_samples):
        for j in range(i, n_samples):
            distance = np.sum((X[i] - X[j]) ** 2)
            similarity = np.exp(-gamma * distance)
            similarity_matrix[i, j] = similarity
            similarity_matrix[j, i] = similarity  # 矩阵是对称的
    return similarity_matrix

# 计算相似度矩阵
gamma_value = 1.0
similarity_matrix = compute_rbf_similarity_matrix(X, gamma=gamma_value)

# 执行 Spectral Clustering 聚类
spectral = SpectralClustering(n_clusters=5, affinity='precomputed', random_state=0)
metrics_df['Spectral'] = spectral.fit_predict(similarity_matrix)

# 将结果保存回原文件
metrics_df.to_csv(file_path, index=False)

print(f'Spectral Clustering 结果已保存至原文件的 "Spectral" 列中: {file_path}')

# # GPU版
# import pandas as pd
# import cupy as cp
# from sklearn.cluster import SpectralClustering
#
# # 加载数据
# file_path = './data/Day6_Neuron_Calcium_Metrics.csv'  # 请替换为实际文件路径
# metrics_df = pd.read_csv(file_path)
#
# # 定义用于聚类的特征列
# features = ['Start Time', 'Amplitude', 'Peak', 'Decay Time', 'Rise Time', 'Latency', 'Frequency']
# X = metrics_df[features].values
#
#
# # 使用 CuPy 加速自定义相似度矩阵计算
# def compute_rbf_similarity_matrix_gpu(X, gamma=1.0):
#     X_gpu = cp.asarray(X)  # 将数据移至 GPU
#     n_samples = X_gpu.shape[0]
#     similarity_matrix = cp.zeros((n_samples, n_samples), dtype=cp.float32)
#
#     for i in range(n_samples):
#         for j in range(i, n_samples):
#             distance = cp.sum((X_gpu[i] - X_gpu[j]) ** 2)
#             similarity = cp.exp(-gamma * distance)
#             similarity_matrix[i, j] = similarity
#             similarity_matrix[j, i] = similarity  # 矩阵是对称的
#
#     return cp.asnumpy(similarity_matrix)  # 将结果移回 CPU
#
#
# # 计算相似度矩阵
# gamma_value = 1.0
# similarity_matrix = compute_rbf_similarity_matrix_gpu(X, gamma=gamma_value)
#
# # 执行 Spectral Clustering 聚类
# spectral = SpectralClustering(n_clusters=5, affinity='precomputed', random_state=0)
# metrics_df['Spectral'] = spectral.fit_predict(similarity_matrix)
#
# # 将结果保存回原文件
# metrics_df.to_csv(file_path, index=False)
#
# print(f'Spectral Clustering 结果已保存至原文件的 "Spectral" 列中: {file_path}')
