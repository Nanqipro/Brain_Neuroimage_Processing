# import pandas as pd
# import numpy as np
# from sklearn.cluster import SpectralClustering
# from numba import jit
#
# # 加载数据
# file_path = './data/Day6_Neuron_Calcium_Metrics.xlsx'  # 请替换为实际文件路径
# metrics_df = pd.read_excel(file_path, sheet_name='Windows100_step10')
#
# # 定义用于聚类的特征列
# features = ['Start Time', 'Amplitude', 'Peak', 'Decay Time', 'Rise Time', 'Latency', 'Frequency']
# X = metrics_df[features].values
#
# # 使用 Numba 加速自定义相似度矩阵计算
# @jit(nopython=True)
# def compute_rbf_similarity_matrix(X, gamma=1.0):
#     n_samples = X.shape[0]
#     similarity_matrix = np.zeros((n_samples, n_samples))
#     for i in range(n_samples):
#         for j in range(i, n_samples):
#             distance = np.sum((X[i] - X[j]) ** 2)
#             similarity = np.exp(-gamma * distance)
#             similarity_matrix[i, j] = similarity
#             similarity_matrix[j, i] = similarity  # 矩阵是对称的
#     return similarity_matrix
#
# # 计算相似度矩阵
# gamma_value = 1.0
# similarity_matrix = compute_rbf_similarity_matrix(X, gamma=gamma_value)
#
# # 执行 Spectral Clustering 聚类
# spectral = SpectralClustering(n_clusters=5, affinity='precomputed', random_state=0)
# metrics_df['Spectral'] = spectral.fit_predict(similarity_matrix)
#
# # 将结果保存回原文件
# metrics_df.to_excel(file_path, index=False, sheet_name='Windows100_step10')
#
# print(f'Spectral Clustering 结果已保存至原文件的 "Spectral" 列中: {file_path}')

# V1 优化版本
import pandas as pd
import numpy as np
from sklearn.cluster import SpectralClustering
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from sklearn.metrics.pairwise import rbf_kernel
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
    'Decay Time': 1.5,
    'Rise Time': 1.5,
    'Latency': 1.0,
    'Frequency': 1.0
}

# 将权重应用于标准化后的特征
weight_values = np.array([weights[feature] for feature in features])
X_weighted = X_scaled * weight_values

# 计算相似度矩阵（使用 sklearn 的 rbf_kernel 函数，优化计算速度）
gamma_value = 1.0  # 根据需要调整 gamma 值
similarity_matrix = rbf_kernel(X_weighted, gamma=gamma_value)

# 使用轮廓系数法确定最佳聚类数量
range_n_clusters = range(2, 11)
silhouette_avg_scores = []

for n_clusters in range_n_clusters:
    print(f"正在计算 n_clusters = {n_clusters} 的 Spectral Clustering...")
    spectral = SpectralClustering(n_clusters=n_clusters, affinity='precomputed', random_state=0)
    labels = spectral.fit_predict(similarity_matrix)
    # 计算轮廓系数
    silhouette_avg = silhouette_score(X_weighted, labels)
    silhouette_avg_scores.append(silhouette_avg)
    print(f"对于 n_clusters = {n_clusters}，平均轮廓系数为: {silhouette_avg:.4f}")

# 绘制轮廓系数曲线
plt.figure(figsize=(8, 4))
plt.plot(range_n_clusters, silhouette_avg_scores, 'bo-')
plt.xlabel('聚类数量')
plt.ylabel('平均轮廓系数')
plt.title('不同聚类数量的平均轮廓系数')
plt.xticks(range_n_clusters)
plt.show()

# 确定最佳聚类数量（选择平均轮廓系数最大的数量）
optimal_n_clusters = range_n_clusters[np.argmax(silhouette_avg_scores)]
print(f"最佳聚类数量为: {optimal_n_clusters}")

# 使用最佳聚类数量进行 Spectral Clustering
spectral = SpectralClustering(n_clusters=optimal_n_clusters, affinity='precomputed', random_state=0)
metrics_df['Spectral'] = spectral.fit_predict(similarity_matrix)

# 将结果保存至新文件，避免覆盖原始文件
# output_file_path = './data/Day6_Neuron_Calcium_Metrics_with_Spectral_clusters.xlsx'
metrics_df.to_excel(file_path, index=False, sheet_name='Windows100_step10')

print(f"Spectral Clustering 结果已保存至文件: {file_path}")


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
