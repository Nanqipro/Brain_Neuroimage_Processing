import pandas as pd
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns

# 加载数据
file_path = './data/Day6_Neuron_Calcium_Metrics.xlsx'  # 请替换为实际文件路径
metrics_df = pd.read_excel(file_path,sheet_name='Windows100_step10')

# 定义用于 PCA 的特征列
features = ['Start Time', 'Amplitude', 'Peak', 'Decay Time', 'Rise Time', 'Latency', 'Frequency']
X = metrics_df[features].values

# 使用 PCA 将数据降维到 2D 空间
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)

# 将 PCA 结果加入原数据框
metrics_df['PCA-1'] = X_pca[:, 0]
metrics_df['PCA-2'] = X_pca[:, 1]

# 可视化 PCA 结果
plt.figure(figsize=(10, 8))
sns.scatterplot(
    x='PCA-1', y='PCA-2',
    data=metrics_df,
    hue='GMM',  # 使用聚类列区分颜色，可替换为其他聚类列
    palette='viridis',
    legend='full'
)
plt.title("PCA Clustering of Neuron Calcium Metrics")
plt.xlabel("PCA Dimension 1")
plt.ylabel("PCA Dimension 2")
plt.show()

# 保存结果至文件
# output_file_path = './data/Day6_Neuron_Calcium_Metrics_with_PCA.csv'
metrics_df.to_excel(file_path, index=False,sheet_name='Windows100_step10')

print(f'PCA 降维结果已保存至: {file_path}')
