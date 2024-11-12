import pandas as pd
from sklearn.cluster import AgglomerativeClustering

# 加载数据
file_path = './data/Day6_Neuron_Calcium_Metrics.csv'  # 请替换为实际文件路径
metrics_df = pd.read_csv(file_path)

# 定义用于聚类的特征列
features = ['Start Time', 'Amplitude', 'Peak', 'Decay Time', 'Rise Time', 'Latency', 'Frequency']
X = metrics_df[features].values

# 执行层次聚类
agglomerative = AgglomerativeClustering(n_clusters=5, metric='euclidean', linkage='ward')
metrics_df['Hierarchical'] = agglomerative.fit_predict(X)

# 将结果保存回原文件
metrics_df.to_csv(file_path, index=False)

print(f'层次聚类结果已保存至原文件的 "Hierarchical" 列中: {file_path}')
