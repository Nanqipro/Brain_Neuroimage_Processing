# import pandas as pd
# from sklearn.cluster import AgglomerativeClustering
#
# # 加载数据
# file_path = './data/Day6_Neuron_Calcium_Metrics.xlsx'  # 请替换为实际文件路径
# metrics_df = pd.read_excel(file_path, sheet_name='Windows100_step10')
#
# # 定义用于聚类的特征列
# features = ['Start Time', 'Amplitude', 'Peak', 'Decay Time', 'Rise Time', 'Latency', 'Frequency']
# X = metrics_df[features].values
#
# # 执行层次聚类
# agglomerative = AgglomerativeClustering(n_clusters=5, metric='euclidean', linkage='ward')
# metrics_df['Hierarchical'] = agglomerative.fit_predict(X)
#
# # 将结果保存回原文件
# metrics_df.to_excel(file_path, index=False, sheet_name='Windows100_step10')
#
# print(f'层次聚类结果已保存至原文件的 "Hierarchical" 列中: {file_path}')

# V1 优化版本
import pandas as pd
import numpy as np
from sklearn.cluster import AgglomerativeClustering
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from scipy.cluster.hierarchy import dendrogram, linkage
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

# 使用轮廓系数法确定最佳聚类数量
range_n_clusters = range(2, 11)
silhouette_avg_scores = []

for n_clusters in range_n_clusters:
    clusterer = AgglomerativeClustering(n_clusters=n_clusters, linkage='ward')
    cluster_labels = clusterer.fit_predict(X_weighted)
    silhouette_avg = silhouette_score(X_weighted, cluster_labels)
    silhouette_avg_scores.append(silhouette_avg)
    print(f"聚类数量为 {n_clusters} 时，平均轮廓系数为: {silhouette_avg:.4f}")

# 绘制轮廓系数曲线
plt.figure(figsize=(8, 4))
plt.plot(range_n_clusters, silhouette_avg_scores, 'bo-')
plt.xlabel('聚类数量')
plt.ylabel('平均轮廓系数')
plt.title('不同聚类数量的平均轮廓系数')
plt.xticks(range_n_clusters)
plt.show()

# 确定最佳聚类数量（例如，选择平均轮廓系数最大的数量）
optimal_n_clusters = range_n_clusters[np.argmax(silhouette_avg_scores)]
print(f"基于轮廓系数法的最佳聚类数量为: {optimal_n_clusters}")

# 使用最佳聚类数量进行层次聚类
agglomerative = AgglomerativeClustering(n_clusters=optimal_n_clusters, linkage='ward')
metrics_df['Hierarchical'] = agglomerative.fit_predict(X_weighted)

# 可选：绘制聚类树状图（Dendrogram）
linked = linkage(X_weighted, method='ward')

plt.figure(figsize=(10, 7))
dendrogram(linked, labels=metrics_df.index.tolist())
plt.xlabel('样本索引')
plt.ylabel('距离')
plt.title('层次聚类树状图')
plt.show()

# 将结果保存至新文件，避免覆盖原始文件
# output_file_path = './data/Day6_Neuron_Calcium_Metrics_with_Hierarchical_clusters.xlsx'
metrics_df.to_excel(file_path, index=False, sheet_name='Windows100_step10')

print(f'层次聚类结果已保存至文件: {file_path}')

