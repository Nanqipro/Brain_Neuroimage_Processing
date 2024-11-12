import pandas as pd
from umap import UMAP
import matplotlib.pyplot as plt
import seaborn as sns

# 加载数据
file_path = './data/Day6_Neuron_Calcium_Metrics.csv'  # 请替换为实际文件路径
metrics_df = pd.read_csv(file_path)

# 定义用于 UMAP 的特征列
features = ['Start Time', 'Amplitude', 'Peak', 'Decay Time', 'Rise Time', 'Latency', 'Frequency']
X = metrics_df[features].values

# 使用 UMAP 将数据降维到 2D 空间
umap = UMAP(n_components=2, n_neighbors=15, min_dist=0.1, random_state=0)
X_umap = umap.fit_transform(X)

# 将 UMAP 结果加入原数据框
metrics_df['UMAP-1'] = X_umap[:, 0]
metrics_df['UMAP-2'] = X_umap[:, 1]

# 可视化 UMAP 结果
plt.figure(figsize=(10, 8))
sns.scatterplot(
    x='UMAP-1', y='UMAP-2',
    hue='k-means-ED',
    data=metrics_df,
    palette='viridis',
    legend='full'
)
plt.title("UMAP Clustering of Neuron Calcium Metrics")
plt.xlabel("UMAP Dimension 1")
plt.ylabel("UMAP Dimension 2")
plt.show()

# 保存结果至文件
# output_file_path = './data/Day6_Neuron_Calcium_Metrics_with_UMAP.csv'
metrics_df.to_csv(file_path, index=False)

print(f'UMAP 聚类结果已保存至: {file_path}')
