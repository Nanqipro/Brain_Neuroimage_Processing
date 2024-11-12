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



import pandas as pd
from sklearn.cluster import KMeans

# 加载数据
file_path = './data/Day6_Neuron_Calcium_Metrics.csv'  # 请替换为实际文件路径
metrics_df = pd.read_csv(file_path)

# 定义用于聚类的特征列
features = ['Start Time', 'Amplitude', 'Peak', 'Decay Time', 'Rise Time', 'Latency', 'Frequency']

# 准备数据进行聚类
X = metrics_df[features]

# 设定K-means的聚类数量，例如这里假设为5个簇
kmeans = KMeans(n_clusters=5, random_state=0)
metrics_df['k-means-ED'] = kmeans.fit_predict(X)

# 将聚类结果保存至原文件
metrics_df.to_csv(file_path, index=False)

print(f'聚类结果已保存至原文件的 "k-means-ED" 列中: {file_path}')
