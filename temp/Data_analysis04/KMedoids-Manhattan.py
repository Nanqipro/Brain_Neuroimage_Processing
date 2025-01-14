import pandas as pd
from sklearn_extra.cluster import KMedoids  # 需要安装 scikit-learn-extra

# 加载数据
file_path = './data/Day6_Neuron_Calcium_Metrics.csv'  # 请替换为实际文件路径
metrics_df = pd.read_csv(file_path)

# 定义用于聚类的特征列
features = ['Start Time', 'Amplitude', 'Peak', 'Decay Time', 'Rise Time', 'Latency', 'Frequency']

# 准备数据进行聚类
X = metrics_df[features]

# 使用KMedoids进行聚类，并指定曼哈顿距离
kmedoids = KMedoids(n_clusters=5, metric='manhattan', random_state=0)
metrics_df['KMedoids-Manhattan'] = kmedoids.fit_predict(X)

# 将聚类结果保存至原文件
metrics_df.to_csv(file_path, index=False)

print(f'聚类结果已保存至原文件的 "KMedoids-Manhattan" 列中: {file_path}')
