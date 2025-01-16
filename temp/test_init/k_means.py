# n1~n51的50维聚类，并且使用主成分分析（PCA）将 50 维数据降维到 2 维，以便于在散点图中展示聚类结果。
import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
# from mpl_toolkits.mplot3d import Axes3D
import numpy as np

# 此处修改：设置支持CJK字符的字体，如SimHei（黑体）
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用黑体显示中文
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

# 加载数据集
file_path = './data/trace_homecage.xlsx'
data = pd.read_excel(file_path)

# 提取神经元数据（n1到n51列）
neuron_data = data.iloc[:, 1:]

# 执行k-means聚类，分成5类
kmeans = KMeans(n_clusters=5, random_state=42)
kmeans.fit(neuron_data.T)  # 此处修改：对神经元（列）进行聚类，而不是对行进行聚类

# 将聚类标签添加到神经元列
neuron_clusters = pd.DataFrame({'Neuron': neuron_data.columns, 'Cluster': kmeans.labels_})

# 打印每个神经元所属的聚类
print("神经元的聚类结果:")
print(neuron_clusters)

# 统计每个类中的神经元数量
cluster_counts = neuron_clusters['Cluster'].value_counts()
print("\n每个聚类中的神经元数量:")
print(cluster_counts)

# 将聚类后的结果保存为Excel文件
output_path = './data/clustered_neuron_data.xlsx'
neuron_clusters.to_excel(output_path, index=False)

print(f"聚类结果已保存至 {output_path}")

# 可视化聚类结果
# 此处使用主成分分析（PCA）降维以便于可视化展示

pca = PCA(n_components=2)
neuron_data_pca = pca.fit_transform(neuron_data.T)  # 转置后对神经元进行PCA降维

# 绘制聚类结果的散点图
plt.figure(figsize=(10, 6))
plt.scatter(neuron_data_pca[:, 0], neuron_data_pca[:, 1], c=kmeans.labels_, cmap='viridis', s=100)
plt.title('基于神经元波动特征的K-means聚类结果', fontsize=14)
plt.xlabel('PCA Component 1')
plt.ylabel('PCA Component 2')
plt.colorbar(label='Cluster')
plt.show()

