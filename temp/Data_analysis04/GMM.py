# import pandas as pd
# from sklearn.mixture import GaussianMixture
#
# # 加载数据
# file_path = './data/Day6_Neuron_Calcium_Metrics.xlsx'  # 请替换为实际文件路径
# metrics_df = pd.read_excel(file_path, sheet_name='Windows100_step10')
#
# # 定义用于聚类的特征列
# features = ['Start Time', 'Amplitude', 'Peak', 'Decay Time', 'Rise Time', 'Latency', 'Frequency']
# X = metrics_df[features].values
#
# # 执行 Gaussian Mixture Model 聚类
# gmm = GaussianMixture(n_components=5, random_state=0)
# metrics_df['GMM'] = gmm.fit_predict(X)
#
# # 将结果保存回原文件
# metrics_df.to_excel(file_path, index=False, sheet_name='Windows100_step10')
#
# print(f'GMM 聚类结果已保存至原文件的 "GMM" 列中: {file_path}')

# V1 优化版本
import pandas as pd
import numpy as np
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
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
    'Decay Time': 1.0,
    'Rise Time': 1.0,
    'Latency': 1.5,
    'Frequency': 1.5
}

# 将权重应用于标准化后的特征
weight_values = np.array([weights[feature] for feature in features])
X_weighted = X_scaled * weight_values

# 使用 BIC 和 AIC 确定最佳聚类数量
n_components = range(1, 11)
models = [GaussianMixture(n_components=n, random_state=0).fit(X_weighted) for n in n_components]
bics = [model.bic(X_weighted) for model in models]
aics = [model.aic(X_weighted) for model in models]

# 绘制 BIC 和 AIC 曲线
plt.figure(figsize=(8, 4))
plt.plot(n_components, bics, label='BIC', marker='o')
plt.plot(n_components, aics, label='AIC', marker='o')
plt.xlabel('聚类数量（n_components）')
plt.ylabel('信息准则值')
plt.title('BIC 和 AIC 与聚类数量的关系')
plt.legend()
plt.show()

# 使用轮廓系数法确定最佳聚类数量
silhouette_scores = []
for n in n_components:
    if n == 1:
        silhouette_scores.append(np.nan)
        continue
    gmm = GaussianMixture(n_components=n, random_state=0)
    labels = gmm.fit_predict(X_weighted)
    score = silhouette_score(X_weighted, labels)
    silhouette_scores.append(score)

# 绘制轮廓系数曲线
plt.figure(figsize=(8, 4))
plt.plot(n_components, silhouette_scores, 'bo-')
plt.xlabel('聚类数量（n_components）')
plt.ylabel('平均轮廓系数')
plt.title('轮廓系数与聚类数量的关系')
plt.show()

# 根据 BIC 选择最佳聚类数量
optimal_n = n_components[np.argmin(bics)]
print(f'基于 BIC 的最佳聚类数量为: {optimal_n}')

# 使用最佳聚类数量进行 GMM 聚类
gmm = GaussianMixture(n_components=optimal_n, random_state=0)
metrics_df['GMM'] = gmm.fit_predict(X_weighted)

# 将结果保存至新文件，避免覆盖原始文件
# output_file_path = './data/Day6_Neuron_Calcium_Metrics_with_GMM_clusters.xlsx'
metrics_df.to_excel(file_path, index=False, sheet_name='Windows100_step10')

print(f'GMM 聚类结果已保存至文件: {file_path}')
