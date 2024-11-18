import pandas as pd
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import seaborn as sns

# 加载数据
file_path = './data/Day6_Neuron_Calcium_Metrics.xlsx'  # 请替换为实际文件路径
metrics_df = pd.read_excel(file_path, sheet_name='Windows100_step10')

# 定义用于 t-SNE 的特征列
features = ['Start Time', 'Amplitude', 'Peak', 'Decay Time', 'Rise Time', 'Latency', 'Frequency']
X = metrics_df[features].values

# 使用 t-SNE 将数据降维到 2D 空间
tsne = TSNE(n_components=2, perplexity=30, learning_rate=200, random_state=0)
X_tsne = tsne.fit_transform(X)

# 将 t-SNE 结果加入原数据框
metrics_df['t-SNE-1'] = X_tsne[:, 0]
metrics_df['t-SNE-2'] = X_tsne[:, 1]

# 可视化 t-SNE 结果
plt.figure(figsize=(10, 8))
sns.scatterplot(
    x='t-SNE-1', y='t-SNE-2',
    hue='GMM',
    data=metrics_df,
    palette='viridis',
    legend='full'
)
plt.title("t-SNE Clustering of Neuron Calcium Metrics")
plt.xlabel("t-SNE Dimension 1")
plt.ylabel("t-SNE Dimension 2")
plt.show()

# 保存结果至文件
# output_file_path = './data/Day6_Neuron_Calcium_Metrics_with_tSNE.csv'
metrics_df.to_excel(file_path, index=False,sheet_name='Windows100_step10')

print(f't-SNE 聚类结果已保存至: {file_path}')
