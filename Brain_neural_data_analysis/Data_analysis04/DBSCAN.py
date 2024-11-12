import pandas as pd
from sklearn.cluster import DBSCAN

# 加载数据
file_path = './data/Day6_Neuron_Calcium_Metrics.csv'  # 请替换为实际文件路径
metrics_df = pd.read_csv(file_path)

# 定义用于聚类的特征列
features = ['Start Time', 'Amplitude', 'Peak', 'Decay Time', 'Rise Time', 'Latency', 'Frequency']
X = metrics_df[features].values

# 设置 DBSCAN 的参数并执行聚类
dbscan = DBSCAN(eps=0.5, min_samples=5)  # 可根据数据特性调整 eps 和 min_samples
metrics_df['DBSCAN'] = dbscan.fit_predict(X)

# 将结果保存回原文件
metrics_df.to_csv(file_path, index=False)

print(f'DBSCAN 聚类结果已保存至原文件的 "DBSCAN" 列中: {file_path}')
