# # 结果保存到新的表中
# import pandas as pd
# from sklearn.cluster import KMeans
#
# # 加载数据
# file_path = './data/Day6_Neuron_Calcium_Metrics.csv'  # 请替换为实际文件路径
# metrics_df = pd.read_csv(file_path)
#
# # 定义用于聚类的特征列
# features = ['Start Time', 'Amplitude', 'Peak', 'Decay Time', 'Rise Time', 'Latency', 'Frequency']
#
# # 准备数据进行聚类
# X = metrics_df[features]
#
# # 设定K-means的聚类数量，例如这里假设为3个簇
# kmeans = KMeans(n_clusters=5, random_state=0)
# metrics_df['k-means-ED'] = kmeans.fit_predict(X)
#
# # 保存聚类结果到文件
# output_file_path = './data/Day6_Neuron_Calcium_Metrics_with_KMeans.csv'  # 输出文件路径
# metrics_df.to_csv(output_file_path, index=False)
#
# # 输出文件保存路径
# print(f'聚类结果已保存至: {output_file_path}')


# v1
# import pandas as pd
# from sklearn.cluster import KMeans
#
# # 加载数据
# file_path = './data/Day6_Neuron_Calcium_Metrics.xlsx'  # 请替换为实际文件路径
# metrics_df = pd.read_excel(file_path)
#
# # 定义用于聚类的特征列
# features = ['Start Time', 'Amplitude', 'Peak', 'Decay Time', 'Rise Time', 'Latency', 'Frequency']
#
# # 准备数据进行聚类
# X = metrics_df[features]
#
# # 设定K-means的聚类数量，例如这里假设为5个簇
# kmeans = KMeans(n_clusters=5, random_state=0)
# metrics_df['k-means-ED'] = kmeans.fit_predict(X)
#
# # 将聚类结果保存至原文件
# metrics_df.to_excel(file_path, index=False)
#
# print(f'聚类结果已保存至原文件的 "k-means-ED" 列中: {file_path}')

# # v2 优化版
# import pandas as pd
# import numpy as np
# from sklearn.cluster import KMeans
# from sklearn.preprocessing import StandardScaler
# from sklearn.metrics import silhouette_score
# import matplotlib.pyplot as plt
#
# # 忽略警告信息
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
# # 准备数据进行聚类
# X = metrics_df[features]
#
# # 特征标准化
# scaler = StandardScaler()
# X_scaled = scaler.fit_transform(X)
#
# # 使用肘部法确定最佳聚类数量
# distortions = []
# K = range(1, 11)
# for k in K:
#     kmeans = KMeans(n_clusters=k, random_state=0)
#     kmeans.fit(X_scaled)
#     distortions.append(kmeans.inertia_)
#
# # 绘制肘部法图形
# plt.figure(figsize=(8, 4))
# plt.plot(K, distortions, 'bo-')
# plt.xlabel('聚类数量 k')
# plt.ylabel('SSE（Sum of Squared Errors）')
# plt.title('肘部法确定最佳聚类数量')
# plt.xticks(K)
# plt.show()
#
# # 使用轮廓系数法确定最佳聚类数量
# silhouette_scores = []
# K = range(2, 11)
# for k in K:
#     kmeans = KMeans(n_clusters=k, random_state=0)
#     labels = kmeans.fit_predict(X_scaled)
#     score = silhouette_score(X_scaled, labels)
#     silhouette_scores.append(score)
#
# # 绘制轮廓系数图形
# plt.figure(figsize=(8, 4))
# plt.plot(K, silhouette_scores, 'bo-')
# plt.xlabel('聚类数量 k')
# plt.ylabel('轮廓系数')
# plt.title('轮廓系数法确定最佳聚类数量')
# plt.xticks(K)
# plt.show()
#
# # 根据肘部法和轮廓系数法选择最佳聚类数量，例如这里选择 k=4
# optimal_k = 4
#
# # 进行K-means聚类
# kmeans = KMeans(n_clusters=optimal_k, random_state=0)
# metrics_df['k-means-ed'] = kmeans.fit_predict(X_scaled)
#
# # 将聚类结果保存至新文件
# # output_file_path = './data/Day6_temp.xlsx'
# metrics_df.to_excel(file_path, index=False, sheet_name='Windows100_step10')
#
# print(f'聚类结果已保存至文件: {file_path}')


# V3 考虑权重
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt

# 忽略警告信息
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

# 准备数据进行聚类
X = metrics_df[features]

# 特征标准化
scaler = StandardScaler()
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

# 将权重转换为数组，与特征列对应
weight_values = np.array([weights[feature] for feature in features])

# 对标准化后的特征应用权重
X_weighted = X_scaled * weight_values

# 使用肘部法确定最佳聚类数量
distortions = []
K = range(1, 11)
for k in K:
    kmeans = KMeans(n_clusters=k, random_state=0)
    kmeans.fit(X_weighted)
    distortions.append(kmeans.inertia_)

# 绘制肘部法图形
plt.figure(figsize=(8, 4))
plt.plot(K, distortions, 'bo-')
plt.xlabel('cluster number k')
plt.ylabel('SSE（Sum of Squared Errors）')
plt.title('elbow method')
plt.xticks(K)
plt.show()

# 使用轮廓系数法确定最佳聚类数量
silhouette_scores = []
K = range(2, 11)
for k in K:
    kmeans = KMeans(n_clusters=k, random_state=0)
    labels = kmeans.fit_predict(X_weighted)
    score = silhouette_score(X_weighted, labels)
    silhouette_scores.append(score)

# 绘制轮廓系数图形
plt.figure(figsize=(8, 4))
plt.plot(K, silhouette_scores, 'bo-')
plt.xlabel('cluster number k')
plt.ylabel('Silhouette Coefficient')
plt.title('Silhouette Coefficient Method')
plt.xticks(K)
plt.show()

# 根据肘部法和轮廓系数法选择最佳聚类数量，例如这里选择 k=4
optimal_k = 6

# 进行K-means聚类
kmeans = KMeans(n_clusters=optimal_k, random_state=0)
metrics_df['k-means-ed'] = kmeans.fit_predict(X_weighted)

# 将聚类结果保存至新文件，避免覆盖原始文件
# output_file_path = './data/Day6_Neuron_Calcium_Metrics_with_clusters.xlsx'
metrics_df.to_excel(file_path, index=False, sheet_name='Windows100_step10')

print(f'聚类结果已保存至文件: {file_path}')
